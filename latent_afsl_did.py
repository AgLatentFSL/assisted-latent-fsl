"""Does the advantage of the selected ViT patch-mean readout over the pooled output depend on same-date references?
Local analysis on cached features and manifests; no model or VLM calls. Usage: python latent_afsl_did.py <run_dir>"""
import json, os, sys
import numpy as np
import pandas as pd
from scipy import stats
import latent_afsl as LA
from latent_afsl import A, B, DINO_DEFAULT, K, rname

out = sys.argv[1]; dd_ = os.path.join(out, "did"); os.makedirs(dd_, exist_ok=True); NB = 5000
M = json.load(open(os.path.join(out, "manifest.json"))); H = tuple(M["selected_vit"])
ids, dqm, DB, panel, banks, base_bank, base_q = LA.all_ids(out)
FT = LA.Feats(out, ids)
panel["capture_date"] = panel["capture_date"].astype(str); banks["capture_date"] = banks["capture_date"].astype(str)
test = panel[panel["role"] != "dev"].reset_index(drop=True); qids, qlab, qdate = test["query_id"].tolist(), np.array(test["label"]), np.array(test["capture_date"]); nq = len(qids)
L = LA.labels_frame(); meta = L.set_index("Filename"); cached = set(FT.ix["vit"])
READ = {"A": A, "H": H, "B": B, "DINO": DINO_DEFAULT}

def topk(S):
    return np.argsort(-S, axis=1, kind="stable")[:, :K]            # descending cosine, ties to the lower bank index

def votes(S, top, blab):
    return np.array([LA.vote([blab[j] for j in t], [float(S[q, j]) for j in t]) for q, t in enumerate(top)])

# ---------------------------------------------------------------- audit banks (unmatched): per query x bank correctness
res = {}
for bt in ("DD", "SD"):
    for s, b in banks[banks.bank_type == bt].groupby("bank_seed"):
        b = b.sort_values("bank_idx"); bl, bd, bid = np.array(b["label"]), np.array(b["capture_date"]), b["ref_id"].tolist()
        for k, r in READ.items():
            S = FT.get(r, qids) @ FT.get(r, bid).T; top = topk(S); v = votes(S, top, bl)
            res[(bt, s, k)] = {"ok": v == qlab, "sd": (bd[top] == qdate[:, None]).sum(1), "sdc": ((bd[top] == qdate[:, None]) & (bl[top] == qlab[:, None])).sum(1), "same": (bl[top] == qlab[:, None]).sum(1)}
seeds = sorted(banks["bank_seed"].unique())
def mat(bt, k, f="ok"): return np.stack([res[(bt, s, k)][f] for s in seeds], axis=1).astype(float)       # queries x banks
HA = {bt: mat(bt, "H") - mat(bt, "A") for bt in ("DD", "SD")}

# ---------------------------------------------------------------- matched construction: each same-date bank with the query's own date made ineligible
# For query q and same-date bank Bk, the references on q's capture date are replaced by references of the same class from other dates
# (seeded, from the cached image set, not a panel query, not already in the bank). Size, class counts and all other references stay fixed.
panel_q = set(panel["query_id"]); pool_by_class = {c: sorted(f for f in cached if meta.at[f, "Species"] == c and f not in panel_q) for c in sorted(set(qlab))}
matched = {k: {"ok_with": np.zeros((nq, len(seeds))), "ok_without": np.zeros((nq, len(seeds)))} for k in READ}
nsd = np.zeros((nq, len(seeds))); nsdc = np.zeros((nq, len(seeds))); repl_fail = 0
feat_cache = {k: {} for k in READ}
def fv(k, f):
    if f not in feat_cache[k]: feat_cache[k][f] = FT.get(READ[k], [f])[0]
    return feat_cache[k][f]
Qx = {k: FT.get(r, qids) for k, r in READ.items()}
for si, s in enumerate(seeds):
    b = banks[(banks.bank_type == "SD") & (banks.bank_seed == s)].sort_values("bank_idx"); bl, bd, bid = np.array(b["label"]), np.array(b["capture_date"]), b["ref_id"].tolist(); inb = set(bid)
    Bx = {k: FT.get(r, bid) for k, r in READ.items()}
    for qi in range(nq):
        on = np.where(bd == qdate[qi])[0]; nsd[qi, si] = len(on); nsdc[qi, si] = int((bl[on] == qlab[qi]).sum())
        for k in READ: matched[k]["ok_with"][qi, si] = res[("SD", s, k)]["ok"][qi]
        if len(on) == 0:
            for k in READ: matched[k]["ok_without"][qi, si] = res[("SD", s, k)]["ok"][qi]
            continue
        rng = np.random.default_rng([LA.SEED, 300, int(s), qi]); new = {}; used = set()
        for j in on:
            cand = [f for f in pool_by_class[bl[j]] if f not in inb and f not in used and meta.at[f, "date"] != qdate[qi]]
            if not cand: repl_fail += 1; continue
            new[j] = cand[int(rng.integers(len(cand)))]; used.add(new[j])
        for k in READ:
            X = Bx[k].copy()
            for j, f in new.items(): X[j] = fv(k, f)
            sims = Qx[k][qi] @ X.T; t = np.argsort(-sims, kind="stable")[:K]
            matched[k]["ok_without"][qi, si] = LA.vote([bl[j] for j in t], [float(sims[j]) for j in t]) == qlab[qi]
HA_m = (matched["H"]["ok_with"] - matched["A"]["ok_with"]) - (matched["H"]["ok_without"] - matched["A"]["ok_without"])          # queries x banks

# ---------------------------------------------------------------- estimates and intervals
rng = np.random.default_rng(LA.SEED); dates = sorted(set(qdate)); didx = {d: np.where(qdate == d)[0] for d in dates}
def unmatched_est(qsel, bsd, bdd): return HA["SD"][np.ix_(qsel, bsd)].mean() - HA["DD"][np.ix_(qsel, bdd)].mean()
def matched_est(qsel, bsd): return HA_m[np.ix_(qsel, bsd)].mean()
allq, allb = np.arange(nq), np.arange(len(seeds))
def boot(kind):
    o = []
    for _ in range(NB):
        dsel = rng.integers(0, len(dates), len(dates)); qsel = np.concatenate([didx[dates[i]] for i in dsel])               # resample capture dates; every query of a date stays together
        b1, b2 = rng.integers(0, len(seeds), len(seeds)), rng.integers(0, len(seeds), len(seeds))                             # resample bank instances
        o.append(unmatched_est(qsel, b1, b2) if kind == "u" else matched_est(qsel, b1))
    return float(np.percentile(o, 2.5)), float(np.percentile(o, 97.5))
def boot_dates_only(kind):
    o = []
    for _ in range(NB):
        dsel = rng.integers(0, len(dates), len(dates)); qsel = np.concatenate([didx[dates[i]] for i in dsel]); o.append(unmatched_est(qsel, allb, allb) if kind == "u" else matched_est(qsel, allb))
    return float(np.percentile(o, 2.5)), float(np.percentile(o, 97.5))
E = {"unmatched": {"estimate": float(unmatched_est(allq, allb, allb)), "ci_date_and_bank": boot("u"), "ci_date_only": boot_dates_only("u"),
                   "H_minus_A_same_date": float(HA["SD"].mean()), "H_minus_A_date_disjoint": float(HA["DD"].mean())},
     "matched": {"estimate": float(matched_est(allq, allb)), "ci_date_and_bank": boot("m"), "ci_date_only": boot_dates_only("m"),
                 "H_minus_A_with_same_date": float((matched["H"]["ok_with"] - matched["A"]["ok_with"]).mean()), "H_minus_A_without_same_date": float((matched["H"]["ok_without"] - matched["A"]["ok_without"]).mean()),
                 "replacements_not_found": int(repl_fail)}}
dq = HA["SD"].mean(1) - HA["DD"].mean(1); dqm_ = HA_m.mean(1)
E["unmatched"]["wilcoxon_p_query"] = float(stats.wilcoxon(dq).pvalue); E["matched"]["wilcoxon_p_query"] = float(stats.wilcoxon(dqm_[dqm_ != 0]).pvalue) if np.any(dqm_ != 0) else 1.0
dm = np.array([dq[didx[d]].mean() for d in dates]); dmm = np.array([dqm_[didx[d]].mean() for d in dates])
E["unmatched"]["dates_positive_negative"] = [int((dm > 0).sum()), int((dm < 0).sum())]; E["matched"]["dates_positive_negative"] = [int((dmm > 0).sum()), int((dmm < 0).sum())]
E["unmatched"]["date_level_wilcoxon_p"] = float(stats.wilcoxon(dm).pvalue); E["matched"]["date_level_wilcoxon_p"] = float(stats.wilcoxon(dmm[dmm != 0]).pvalue) if np.any(dmm != 0) else 1.0

# ---------------------------------------------------------------- per-date and per-class sensitivity
rows = []
for d in dates:
    q = didx[d]; rest = np.setdiff1d(allq, q)
    rows.append({"capture_date": d, "queries": len(q), "classes": ", ".join(f"{c} ({n})" for c, n in pd.Series(qlab[q]).value_counts().items()),
                 "same_date_refs_in_bank": float(nsd[q].mean()), "of_which_query_class": float(nsdc[q].mean()),
                 "H_minus_A_date_disjoint": float(HA["DD"][q].mean()), "H_minus_A_same_date": float(HA["SD"][q].mean()), "did_unmatched": float(dq[q].mean()), "did_matched": float(dqm_[q].mean()),
                 "share_of_total_unmatched": float(dq[q].sum() / dq.sum()) if dq.sum() else float("nan"), "share_of_total_matched": float(dqm_[q].sum() / dqm_.sum()) if dqm_.sum() else float("nan"),
                 "estimate_leaving_date_out_unmatched": float(unmatched_est(rest, allb, allb)), "estimate_leaving_date_out_matched": float(matched_est(rest, allb))})
PD = pd.DataFrame(rows); PD.to_csv(os.path.join(dd_, "per_date.csv"), index=False)
PC = pd.DataFrame([{"class": c, "queries": int((qlab == c).sum()), "query_dates": int(len(set(qdate[qlab == c]))), "same_date_refs_in_bank": float(nsd[qlab == c].mean()),
                    "H_minus_A_date_disjoint": float(HA["DD"][qlab == c].mean()), "H_minus_A_same_date": float(HA["SD"][qlab == c].mean()), "did_unmatched": float(dq[qlab == c].mean()), "did_matched": float(dqm_[qlab == c].mean()),
                    "estimate_leaving_class_out_unmatched": float(unmatched_est(np.where(qlab != c)[0], allb, allb)), "estimate_leaving_class_out_matched": float(matched_est(np.where(qlab != c)[0], allb))} for c in sorted(set(qlab))])
PC.to_csv(os.path.join(dd_, "per_class.csv"), index=False)

# ---------------------------------------------------------------- mechanism at the level of eligibility and neighbor labels
mech = []
for k in READ:
    sd, sdc, same = mat("SD", k, "sd"), mat("SD", k, "sdc"), mat("SD", k, "same"); w, wo = matched[k]["ok_with"], matched[k]["ok_without"]; has = nsd > 0
    mech.append({"readout": rname(READ[k]), "top4_share_from_query_date": float(sd.sum() / (sd.size * K)), "prompts_with_a_same_date_ref": float((sd > 0).mean()),
                 "same_date_refs_with_correct_label": float(sdc.sum() / max(sd.sum(), 1)), "other_date_refs_with_correct_label": float((same - sdc).sum() / (same.size * K - sd.sum())),
                 "capture_rate_of_available_same_date_same_class_refs": float(sdc.sum() / max(nsdc.sum(), 1)), "acc_without_same_date": float(wo.mean()), "acc_with_same_date": float(w.mean()),
                 "vote_fixes_when_eligible": int(((wo == 0) & (w == 1)).sum()), "vote_breaks_when_eligible": int(((wo == 1) & (w == 0)).sum()), "query_bank_pairs": int(w.size), "pairs_with_same_date_refs_available": int(has.sum()),
                 "acc_date_disjoint_audit_banks": float(mat("DD", k).mean()), "acc_same_date_audit_banks": float(mat("SD", k).mean())})
MC = pd.DataFrame(mech); MC.to_csv(os.path.join(dd_, "mechanism.csv"), index=False)
pair = {}
for nm, w_ in (("without same-date references (matched)", "ok_without"), ("with same-date references (matched)", "ok_with")):
    a, h = matched["A"][w_], matched["H"][w_]; pair[nm] = {"H_fixes": int(((a == 0) & (h == 1)).sum()), "H_breaks": int(((a == 1) & (h == 0)).sum())}
for bt, nm in (("DD", "date-disjoint audit banks"), ("SD", "same-date audit banks")):
    a, h = mat(bt, "A"), mat(bt, "H"); pair[nm] = {"H_fixes": int(((a == 0) & (h == 1)).sum()), "H_breaks": int(((a == 1) & (h == 0)).sum())}

# ---------------------------------------------------------------- how the audit banks were generated; split audit
bank_gen = {"DD": "bank seed s: 11 per class sampled with generator seed [42, s, 0] from images whose capture date is not a query date", "SD": "bank seed s: 11 per class sampled with generator seed [42, s, 1] from images on all dates",
            "paired_seeds": "DD seed s and SD seed s are independent draws from different pools, so seed pairing across bank types carries no matching; readouts are paired within each query and bank instance, and bank types are compared through per-query means",
            "shared_images_between_DD_and_SD_bank_of_the_same_seed_mean": float(np.mean([len(set(banks[(banks.bank_type == 'DD') & (banks.bank_seed == s)].ref_id) & set(banks[(banks.bank_type == 'SD') & (banks.bank_seed == s)].ref_id)) for s in seeds])),
            "SD_bank_images_on_any_query_date_mean": float(banks[banks.bank_type == "SD"]["capture_date"].isin(set(panel.capture_date)).groupby(banks[banks.bank_type == "SD"]["bank_seed"]).sum().mean()),
            "SD_bank_images_on_the_querys_own_date_mean": float(nsd.mean())}
dqm["capture_date"] = dqm["capture_date"].astype(str); DB["capture_date"] = DB["capture_date"].astype(str); base_bank["capture_date"] = base_bank["capture_date"].astype(str)
tq_dates, dq_dates, db_dates = set(test["capture_date"]), set(dqm["capture_date"]), set(DB["capture_date"]); bb_dates = set(base_bank["capture_date"]); aud_dd = banks[banks.bank_type == "DD"]
split = {"test_query_dates": len(tq_dates), "development_query_dates": len(dq_dates), "development_bank_dates": len(db_dates), "frozen_test_bank_dates": len(bb_dates), "audit_DD_bank_dates": int(aud_dd["capture_date"].nunique()),
         "dev_query_dates_also_test_query_dates": len(dq_dates & tq_dates), "dev_bank_dates_also_test_query_dates": len(db_dates & tq_dates), "dev_query_dates_also_frozen_test_bank_dates": len(dq_dates & bb_dates),
         "dev_query_dates_also_audit_DD_bank_dates": len(dq_dates & set(aud_dd["capture_date"])), "dev_bank_dates_also_any_test_bank_date": len(db_dates & (bb_dates | set(aud_dd["capture_date"]))),
         "test_query_dates_also_test_bank_dates": len(tq_dates & (bb_dates | set(aud_dd["capture_date"]))), "dev_query_dates_also_dev_bank_dates": len(dq_dates & db_dates),
         "dev_images_that_are_test_queries": len((set(dqm.query_id) | set(DB.ref_id)) & set(panel.query_id)), "dev_query_images_in_frozen_test_bank": len(set(dqm.query_id) & set(base_bank.ref_id)),
         "dev_query_images_in_audit_DD_banks": len(set(dqm.query_id) & set(aud_dd.ref_id)), "dev_bank_images_in_audit_SD_banks": len(set(DB.ref_id) & set(banks[banks.bank_type == "SD"].ref_id)),
         "test_queries_with_a_dev_bank_image_from_the_same_date": int(sum(d in db_dates for d in test["capture_date"])), "n_test_queries": nq,
         "all_dates": int(L["date"].nunique()), "dates_used_by_neither_test_nor_dev_queries": int(L["date"].nunique() - len(tq_dates | dq_dates))}

# ---------------------------------------------------------------- sample size for a fresh 2 x 2 replication, from the paired VLM outcomes
T = pd.read_csv(os.path.join(out, "transitions_per_query.csv")); g = T["group"].astype(str)
def design(d):
    d = np.asarray(d, float); n = len(d); m = pd.Series(d).groupby(g.to_numpy()); k = m.size().to_numpy(); mbar = n / len(k)
    msb = (k * (m.mean().to_numpy() - d.mean()) ** 2).sum() / (len(k) - 1); msw = ((d - m.transform("mean").to_numpy()) ** 2).sum() / (n - len(k))
    n0 = (n - (k ** 2).sum() / n) / (len(k) - 1); icc = max(0.0, (msb - msw) / (msb + (n0 - 1) * msw)); deff = 1 + (mbar - 1) * icc
    return {"mean": float(d.mean()), "sd": float(d.std(ddof=1)), "icc_by_date": float(icc), "mean_queries_per_date": float(mbar), "design_effect": float(deff)}
def need(dsg, delta=None, power=0.8):
    delta = dsg["mean"] if delta is None else delta; z = stats.norm.ppf(0.975) + stats.norm.ppf(power); n = (z * dsg["sd"] / delta) ** 2
    return {"target_effect_pp": 100 * delta, "queries_if_independent": int(np.ceil(n)), "queries_with_date_design_effect": int(np.ceil(n * dsg["design_effect"]))}
c = lambda x: T["correct_" + x].astype(int)
inter = design((c("H_V") - c("A_V")) - (c("H_S") - c("A_S"))); simple = design(c("H_V") - c("A_V"))
SS = {"interaction": {**inter, **need(inter)}, "vote_aware_simple_contrast": {**simple, **need(simple)}, "interaction_if_true_effect_is_half": need(inter, inter["mean"] / 2)}
q_dates_used = tq_dates; fresh = L[~L["Filename"].isin(set(panel.query_id) | set(base_q.query_id) | set(base_bank.ref_id)) & ~L["date"].isin(q_dates_used)]
SS["fresh_date_feasibility"] = {"dates_never_used_for_test_queries": int(fresh["date"].nunique()), "images_on_those_dates_per_class": fresh["Species"].value_counts().sort_index().to_dict(), "dates_per_class_on_those_dates": fresh.groupby("Species")["date"].nunique().to_dict(),
                                "note": "a fresh test needs query dates outside the 18 used here and a bank from still other dates; classes with one remaining date cannot supply both"}
json.dump({"selected_readout": rname(H), "baseline": rname(A), "n_queries": nq, "n_dates": len(dates), "banks_per_type": len(seeds), "estimates": E, "paired_fixes_breaks_H_vs_A": pair, "bank_generation": bank_gen, "split_audit": split, "sample_size": SS},
          open(os.path.join(dd_, "did_results.json"), "w"), indent=1)

fp = lambda p: "<0.001" if p < 0.001 else f"{p:.3f}"; pp = lambda x: f"{100 * x:+.1f} pp"; pc = lambda x: f"{100 * x:.1f}%"
u, m = E["unmatched"], E["matched"]
T_ = ["# Does the patch-mean advantage depend on same-date references?\n",
      f"Readouts: **{rname(H)}** (development-selected) against **{rname(A)}**. {nq} held-out DeepWeeds queries on {len(dates)} capture dates, {len(seeds)} bank instances per bank type, 99 images per bank. Local analysis on cached features; no model or VLM calls.\n",
      "## 1. Difference-in-differences in 4-NN accuracy\n",
      "| Design | (patch mean − pooled) with same-date references | (patch mean − pooled) without | Difference-in-differences | 95% CI, dates and banks resampled | 95% CI, dates resampled | Dates positive / negative | Date-level Wilcoxon p |", "|---|---|---|---|---|---|---|---|",
      f"| Audit banks (independent same-date and date-disjoint draws) | {pp(u['H_minus_A_same_date'])} | {pp(u['H_minus_A_date_disjoint'])} | **{pp(u['estimate'])}** | {pp(u['ci_date_and_bank'][0])} to {pp(u['ci_date_and_bank'][1])} | {pp(u['ci_date_only'][0])} to {pp(u['ci_date_only'][1])} | {u['dates_positive_negative'][0]} / {u['dates_positive_negative'][1]} | {fp(u['date_level_wilcoxon_p'])} |",
      f"| Matched banks (only the query's own date made ineligible) | {pp(m['H_minus_A_with_same_date'])} | {pp(m['H_minus_A_without_same_date'])} | **{pp(m['estimate'])}** | {pp(m['ci_date_and_bank'][0])} to {pp(m['ci_date_and_bank'][1])} | {pp(m['ci_date_only'][0])} to {pp(m['ci_date_only'][1])} | {m['dates_positive_negative'][0]} / {m['dates_positive_negative'][1]} | {fp(m['date_level_wilcoxon_p'])} |",
      "\nBoth readouts are scored on the same query and the same bank instance, and the difference is taken within that pair. The interval resamples capture dates as clusters, keeps every query of a sampled date, and resamples bank instances; each replicate keeps the full query-by-bank matrix of repeated measurements.\n",
      f"**How the banks were generated.** {bank_gen['DD']}. {bank_gen['SD']}. {bank_gen['paired_seeds']}. A same-date and a date-disjoint bank of the same seed share {bank_gen['shared_images_between_DD_and_SD_bank_of_the_same_seed_mean']:.1f} images on average. "
      f"A same-date bank holds {bank_gen['SD_bank_images_on_any_query_date_mean']:.1f} images on any of the 18 query dates and {bank_gen['SD_bank_images_on_the_querys_own_date_mean']:.2f} on the query's own date.\n",
      "**Matched construction.** The audit comparison changes two things: the query's own date becomes eligible, and about a third of the bank moves to other query dates. The matched construction changes only the first. For each query and each same-date bank, "
      f"the references on the query's capture date are replaced by references of the same class from other dates (seeded; {m['replacements_not_found']} replacements not found). Bank size, class counts and all other references stay fixed.\n",
      "## 2. Per-date sensitivity\n",
      "| Capture date | Queries | Classes (queries) | Same-date refs in a bank | Of the query's class | DiD, audit banks | DiD, matched | Estimate without this date, audit | Estimate without this date, matched |", "|---|---|---|---|---|---|---|---|---|"]
T_ += [f"| {w['capture_date']} | {w['queries']} | {w['classes']} | {w['same_date_refs_in_bank']:.2f} | {w['of_which_query_class']:.2f} | {pp(w['did_unmatched'])} | {pp(w['did_matched'])} | {pp(w['estimate_leaving_date_out_unmatched'])} | {pp(w['estimate_leaving_date_out_matched'])} |" for _, w in PD.iterrows()]
T_ += [f"\nLeave-one-date-out range: audit banks {pp(PD['estimate_leaving_date_out_unmatched'].min())} to {pp(PD['estimate_leaving_date_out_unmatched'].max())}; matched {pp(PD['estimate_leaving_date_out_matched'].min())} to {pp(PD['estimate_leaving_date_out_matched'].max())}.\n",
       "### Per class\n", "| Class | Queries | Query dates | Same-date refs in a bank | (patch mean − pooled), date-disjoint | (patch mean − pooled), same-date | DiD, audit | DiD, matched | Estimate without this class, audit | Estimate without this class, matched |", "|---|---|---|---|---|---|---|---|---|---|"]
T_ += [f"| {w['class']} | {w['queries']} | {w['query_dates']} | {w['same_date_refs_in_bank']:.2f} | {pp(w['H_minus_A_date_disjoint'])} | {pp(w['H_minus_A_same_date'])} | {pp(w['did_unmatched'])} | {pp(w['did_matched'])} | {pp(w['estimate_leaving_class_out_unmatched'])} | {pp(w['estimate_leaving_class_out_matched'])} |" for _, w in PC.iterrows()]
T_ += ["\n## 3. How the advantage arises (reference eligibility and neighbor labels)\n",
       "| Readout | Top-4 share from the query's date | Prompts with a same-date reference | Same-date references with the correct label | Other-date references with the correct label | Available same-date same-class references that enter the top 4 | 4-NN accuracy without → with same-date eligibility (matched) | Vote fixes / breaks when same-date references become eligible |", "|---|---|---|---|---|---|---|---|"]
T_ += [f"| {w['readout']} | {pc(w['top4_share_from_query_date'])} | {pc(w['prompts_with_a_same_date_ref'])} | {pc(w['same_date_refs_with_correct_label'])} | {pc(w['other_date_refs_with_correct_label'])} | {pc(w['capture_rate_of_available_same_date_same_class_refs'])} | {pc(w['acc_without_same_date'])} → {pc(w['acc_with_same_date'])} | {w['vote_fixes_when_eligible']} / {w['vote_breaks_when_eligible']} of {w['query_bank_pairs']} |" for _, w in MC.iterrows()]
T_ += [f"\nSame-date references are available in {int(MC['pairs_with_same_date_refs_available'].iloc[0])} of {int(MC['query_bank_pairs'].iloc[0])} query-bank pairs.\n",
       "Paired vote changes of the patch mean relative to the pooled output (query-bank pairs):\n", "| Bank | Patch mean fixes a pooled error | Patch mean breaks a pooled success | Net |", "|---|---|---|---|"]
T_ += [f"| {nm} | {v['H_fixes']} | {v['H_breaks']} | {v['H_fixes'] - v['H_breaks']:+d} |" for nm, v in pair.items()]
T_ += ["\n## 4. Split audit\n", "| Check | Value |", "|---|---|"]
T_ += [f"| {k.replace('_', ' ')} | {v} |" for k, v in split.items()]
T_ += ["\n## 5. Sample size for a fresh 2 × 2 replication (from the 153 paired VLM outcomes)\n", "| Target | Observed mean | SD of the paired difference | Intraclass correlation by date | Design effect | Queries for 80% power, independent | Queries with the date design effect | Requests |", "|---|---|---|---|---|---|---|---|",
       f"| Readout-by-prompt interaction (all four cells) | {pp(SS['interaction']['mean'])} | {SS['interaction']['sd']:.3f} | {SS['interaction']['icc_by_date']:.3f} | {SS['interaction']['design_effect']:.2f} | {SS['interaction']['queries_if_independent']} | {SS['interaction']['queries_with_date_design_effect']} | {4 * SS['interaction']['queries_with_date_design_effect']} |",
       f"| Interaction, if the true effect is half the observed one | {SS['interaction_if_true_effect_is_half']['target_effect_pp']:+.1f} pp | | | | {SS['interaction_if_true_effect_is_half']['queries_if_independent']} | {SS['interaction_if_true_effect_is_half']['queries_with_date_design_effect']} | {4 * SS['interaction_if_true_effect_is_half']['queries_with_date_design_effect']} |",
       f"| Vote-aware simple contrast (patch mean versus pooled, two cells) | {pp(SS['vote_aware_simple_contrast']['mean'])} | {SS['vote_aware_simple_contrast']['sd']:.3f} | {SS['vote_aware_simple_contrast']['icc_by_date']:.3f} | {SS['vote_aware_simple_contrast']['design_effect']:.2f} | {SS['vote_aware_simple_contrast']['queries_if_independent']} | {SS['vote_aware_simple_contrast']['queries_with_date_design_effect']} | {2 * SS['vote_aware_simple_contrast']['queries_with_date_design_effect']} |",
       f"\nFresh dates: {SS['fresh_date_feasibility']['dates_never_used_for_test_queries']} capture dates were never used for test queries. Images on those dates per class: {SS['fresh_date_feasibility']['images_on_those_dates_per_class']}. Dates per class: {SS['fresh_date_feasibility']['dates_per_class_on_those_dates']}.\n"]
notes = os.path.join(dd_, "NOTES.md")
if os.path.exists(notes): T_.append(open(notes).read())
open(os.path.join(dd_, "DID_SUMMARY.md"), "w").write("\n".join(T_) + "\n"); print("\n".join(T_))
