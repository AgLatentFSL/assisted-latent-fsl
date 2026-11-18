"""Groups of hidden layers within an encoder: saved Claude predictions of the K = 2 downstream test (500 queries).

Reads only saved files of runs/layer_vlm_k2 and the atlas index. Sends no request.

    python layer_groups_k2.py ../runs/layer_vlm_k2
"""
import os, sys, json, itertools, time
import numpy as np, pandas as pd
from scipy import stats

RUN = sys.argv[1]; ATLAS = os.path.join(os.path.dirname(os.path.abspath(__file__)), "atlas")
OUT = os.path.join(RUN, "layer_groups"); os.makedirs(OUT, exist_ok=True)
SEED, NB, NPERM, ALPHA, K = 20261115, 10000, 100000, 0.05, 2
num = lambda l: int(l.split("_")[1]) if l.startswith("layer_") else 999
def md(df):
    df = df.astype(object).where(df.notna(), ""); c = [str(x) for x in df.columns]
    return "\n".join(["| " + " | ".join(c) + " |", "|" + "---|" * len(c)] + ["| " + " | ".join(f"{v:.4g}" if isinstance(v, float) else str(v) for v in r) + " |" for r in df.values])
def holm(p):
    p = np.asarray(p, float); o = np.argsort(p); m = len(p); adj = np.empty(m); run = 0.0
    for i, j in enumerate(o): run = max(run, (m - i) * p[j]); adj[j] = min(1.0, run)
    return adj

# ================================================================ 1. reconstruct and audit
PQ0 = pd.read_csv(os.path.join(RUN, "per_query_predictions.csv"), dtype=str, keep_default_na=False)
Cn = pd.read_csv(os.path.join(RUN, "conditions.csv"), dtype=str, keep_default_na=False); Cn = Cn[Cn.encoder != ""]; ENC = list(dict.fromkeys(Cn.encoder))
PQ = PQ0[PQ0.condition.isin(Cn.condition)].copy(); key = ["dataset", "query_folder"]; pre = json.load(open(os.path.join(RUN, "preregistration.json")))
audit = {"run": os.path.abspath(RUN), "model_preregistered": pre["model"], "K_preregistered": pre["K"], "prompt": pre["prompt"], "conditions_with_an_encoder": int(Cn.condition.nunique()),
         "duplicate_condition_query_rows": int(PQ.duplicated(["condition"] + key).sum()), "queries_per_condition": sorted(set(PQ.groupby("condition").size())),
         "all_conditions_share_the_same_queries": len({frozenset(zip(g.dataset, g.query_folder)) for _, g in PQ.groupby("condition")}) == 1, "gold_labels_per_query_max": int(PQ.groupby(key).true_label.nunique().max()),
         "queries_per_dataset": PQ[PQ.condition == Cn.condition.iloc[0]].groupby("dataset").size().to_dict(), "missing_predictions": int((PQ.pred == "").sum()), "unparseable_predictions_scored_wrong": int((PQ.pred == "INVALID").sum())}
mods, turns, miss = set(), set(), 0
for k in PQ.payload.unique():
    f = os.path.join(RUN, "responses", k + ".json")
    if not os.path.exists(f): miss += 1; continue
    j = json.load(open(f)); mods.update(j["meta"].get("model_ids", [j.get("model")])); turns.add(j["meta"].get("num_turns"))
audit.update(distinct_requests=int(PQ.payload.nunique()), stored_responses_missing=miss, models_in_responses=sorted(mods), turns_per_request=sorted(turns))
parts = []; inv = set()
for ch in pd.read_csv(os.path.join(ATLAS, "index.csv"), dtype=str, keep_default_na=False, chunksize=200000):
    inv.update(zip(ch.model, ch.layer)); ch = ch[ch["rank"].astype(int) <= K]; parts.append(ch[[(m, l) in set(zip(Cn.encoder, Cn.layer)) for m, l in zip(ch.model, ch.layer)]])
IDX = pd.concat(parts); IDX["rank"] = IDX["rank"].astype(int); IDX = IDX.sort_values("rank"); demo = IDX.groupby(["model", "layer"] + key).reference_id.agg(tuple)
SI = pd.read_csv(os.path.join(ATLAS, "split_images.csv"), dtype=str, keep_default_na=False); bank = set(zip(SI[SI.role == "reference"].dataset, SI[SI.role == "reference"].image_id)); qs = set(zip(SI[SI.role == "query"].dataset, SI[SI.role == "query"].image_id))
audit.update(references_per_prompt=sorted(set(demo.map(len))), reference_bank_size_per_dataset=SI[SI.role == "reference"].groupby("dataset").size().to_dict(),
             demonstrations_outside_the_reference_bank=int(sum((d, r) not in bank for d, r in zip(IDX.dataset, IDX.reference_id))), demonstrations_that_are_queries=int(sum((d, r) in qs for d, r in zip(IDX.dataset, IDX.reference_id))))
PQ["ok"] = PQ.pred == PQ.true_label                       # original scoring rule: exact match of the parsed prediction with the gold label
W = PQ.pivot(index=key, columns="condition", values="ok").astype(bool); n = len(W); ds = W.index.get_level_values(0).values; DS = list(np.unique(ds))
cond = {(r.encoder, r.layer): r.condition for r in Cn.itertuples()}; X = {c: W[c].values.astype(np.float64) for c in W.columns}
sig = {c: tuple(demo.loc[e, l].reindex(W.index).values) for (e, l), c in cond.items()}
ndiff = lambda a, b: int(sum(x != y for x, y in zip(sig[a], sig[b])))
PUB = pd.read_csv(os.path.join(RUN, "results_by_condition.csv")).set_index("condition"); A = []
for r in Cn.itertuples():
    c = r.condition; b = cond[(r.encoder, "final_output")]; w_, l_ = int((W[c] & ~W[b]).sum()), int((~W[c] & W[b]).sum())
    A.append({"encoder": r.encoder, "layer": r.layer, "condition": c, "n_queries": int(W[c].notna().sum()), "missing": int((PQ[PQ.condition == c].pred == "").sum()), "unparseable": int((PQ[PQ.condition == c].pred == "INVALID").sum()),
              "accuracy": 100 * W[c].mean(), "published_accuracy": PUB.loc[c, "accuracy"], "accuracy_reproduced": abs(100 * W[c].mean() - PUB.loc[c, "accuracy"]) < 1e-9, "wins_vs_pooled": w_, "losses_vs_pooled": l_,
              "published_wins": int(PUB.loc[c, "wins_vs_own_pooled"]), "published_losses": int(PUB.loc[c, "losses_vs_own_pooled"]), "wins_losses_reproduced": (w_, l_) == (int(PUB.loc[c, "wins_vs_own_pooled"]), int(PUB.loc[c, "losses_vs_own_pooled"])),
              "queries_with_demonstrations_different_from_pooled": ndiff(c, b), "model": pre["model"], "K": K})
A = pd.DataFrame(A); A.to_csv(os.path.join(OUT, "run_audit.csv"), index=False)
audit.update(matched_queries_all_comparisons=n, accuracies_reproduced=int(A.accuracy_reproduced.sum()), wins_losses_reproduced=int(A.wins_losses_reproduced.sum()),
             hidden_layers_identical_to_pooled=[f"{r.encoder} {r.layer}" for r in A.itertuples() if r.layer != "final_output" and r.queries_with_demonstrations_different_from_pooled == 0],
             hidden_layers_almost_identical_to_pooled=[f"{r.encoder} {r.layer} ({r.queries_with_demonstrations_different_from_pooled} query differs)" for r in A.itertuples() if r.layer != "final_output" and 0 < r.queries_with_demonstrations_different_from_pooled <= 5],
             other_runs_kept_separate={"layer_vlm_allk": "same protocol and split, K = 1, 2, 4, 8 for six encoders, layers selected against the own pooled output; still running, not used",
                                       "all other runs with saved Claude answers (arbab_eval_seed42, deepweeds_*, bean_leaf_seed42, paddy_*, latent_afsl, summary_only_pilot_seed42)": "four references or other prompts, other splits; not K = 2, not used"})
INV = pd.DataFrame([{"encoder": m, "layer": l, "downstream_predictions": (m, l) in cond, "status": "evaluated with Claude at K = 2" if (m, l) in cond else "retrieval scores only; unavailable for the downstream analysis"} for m, l in sorted(inv, key=lambda t: (ENC.index(t[0]) if t[0] in ENC else 99, num(t[1])))])
INV.to_csv(os.path.join(OUT, "layer_inventory.csv"), index=False); audit["layers_in_the_atlas"] = len(INV); audit["layers_with_downstream_predictions"] = int(INV.downstream_predictions.sum())
json.dump(audit, open(os.path.join(OUT, "audit.json"), "w"), indent=1, default=str)

# ================================================================ 2. groups, written before any new contrast is computed
STAGES = {e: {"early (blocks 1-4)": range(1, 5), "middle (blocks 5-8)": range(5, 9), "late (blocks 9-12)": range(9, 13)} for e in ["ViT", "DINOv2", "CLIP", "SigLIP2", "BioCLIP", "DINOv3", "FLAVA"]}
STAGES["Qwen3VL"] = {"early (blocks 1-8)": range(1, 9), "middle (blocks 9-16)": range(9, 17), "late (blocks 17-24)": range(17, 25)}
STAGES["ResNet50"] = {"stage 1 (blocks 1-3)": range(1, 4), "stage 2 (blocks 4-7)": range(4, 8), "stage 3 (blocks 8-13)": range(8, 14), "stage 4 (blocks 14-16)": range(14, 17)}
tested = {e: sorted([l for (m, l) in cond if m == e and l != "final_output"], key=num) for e in ENC}; G = []
for e in ENC:
    for g, rg in STAGES[e].items():
        mem = [l for l in tested[e] if num(l) in rg]
        if mem: G.append({"encoder": e, "scheme": "architecture", "group": g, "layers": mem})
    G.append({"encoder": e, "scheme": "tested_depth", "group": "shallower two tested layers", "layers": tested[e][:2]}); G.append({"encoder": e, "scheme": "tested_depth", "group": "deeper three tested layers", "layers": tested[e][2:]})
    G.append({"encoder": e, "scheme": "all", "group": "all tested hidden layers", "layers": tested[e]})
manifest = {"written_at": time.strftime("%Y-%m-%d %H:%M:%S"), "written_before_the_group_contrasts_were_computed": True,
            "note": "the accuracy of every single layer and the layer-against-layer tests (layer_pairs/) were known when these groups were written; the groups use depth only, never the Claude accuracy",
            "schemes": {"architecture": "primary. Transformers with 12 blocks: thirds of the depth (blocks 1-4, 5-8, 9-12). Qwen3-VL tower, 24 blocks: thirds (1-8, 9-16, 17-24). ResNet-50: its four residual stages (3, 4, 6, 3 bottleneck blocks: 1-3, 4-7, 8-13, 14-16).",
                        "tested_depth": "secondary. The five tested layers of an encoder in depth order: the two shallowest against the three deepest. It exists because the five tested layers are adjacent and deep, so the architecture groups leave most encoders with one usable group.",
                        "all": "all five tested hidden layers together, against the pooled output only."},
            "rules": {"group_against_group": "both groups need at least two evaluated layers", "group_against_pooled": "every group, also with one layer (size is reported)", "outcome_selected_groups": "none explored"},
            "tested_layers": {e: [num(l) for l in tested[e]] for e in ENC}, "groups": [{**g, "layers": [num(l) for l in g["layers"]], "n_layers": len(g["layers"])} for g in G],
            "statistics": {"estimate": "mean over queries of (fraction of the layers of A that are correct) minus (fraction of B correct); pooled output: its binary correctness", "interval": f"paired bootstrap, {NB} replicates, seed {SEED}, queries resampled within dataset (overall) or within the dataset (per dataset); pointwise 95%, unadjusted",
                           "test": f"paired sign-flip randomization test on the query-level differences, {NPERM} flips, two-sided, p = (1 + count) / (1 + flips); Holm correction", "cross_group_layer_pairs": "two-sided exact McNemar; Holm"},
            "families": {"GG_overall_within_encoder": "unique group-against-group contrasts of one encoder (both schemes)", "GG_overall_global": "unique group-against-group contrasts of all encoders", "GP_overall_within_encoder": "unique group-against-pooled contrasts of one encoder, including all tested hidden layers",
                         "GP_overall_global": "unique group-against-pooled contrasts of all encoders", "per_dataset": "the same contrasts times five datasets, within encoder and global, separately for GG and GP",
                         "layer_pairs_within_encoder": "unique cross-group layer pairs of one encoder", "layer_pairs_global": "unique cross-group layer pairs of all encoders", "layer_against_pooled": "each tested hidden layer against the pooled output, within encoder and global"}}
json.dump(manifest, open(os.path.join(OUT, "group_manifest.json"), "w"), indent=1)

# ================================================================ 3. group averages
rng = np.random.default_rng(SEED); Cw = np.zeros((NB, n)); ixd = {d: np.where(ds == d)[0] for d in DS}
for d in DS: Cw[:, ixd[d]] = rng.multinomial(len(ixd[d]), np.ones(len(ixd[d])) / len(ixd[d]), size=NB)
prng = np.random.default_rng(SEED + 1); SG = prng.integers(0, 2, size=(NPERM, n), dtype=np.int8) * 2 - 1
date = np.array([q.split("__")[-1].split("-")[0] if d == "DeepWeeds" else "" for d, q in W.index]); dw = ixd.get("DeepWeeds", np.array([], int)); dates = np.unique(date[dw]); crng = np.random.default_rng(SEED + 2)
CL = crng.integers(0, len(dates), size=(NB, len(dates))) if len(dates) else None; audit_dates = len(dates)
def frac(e, layers): return np.mean([X[cond[(e, l)]] for l in layers], axis=0)
def est(d, ix):
    v = d[ix]; b = Cw[:, ix] @ v / len(ix) * 100; assert np.isfinite(b).all(); o = abs(v.sum()); pm = np.abs(SG[:, ix].astype(np.float64) @ v)
    return {"n_matched": len(ix), "diff_pp": 100 * v.mean(), "ci95_lo_unadjusted": float(np.percentile(b, 2.5)), "ci95_hi_unadjusted": float(np.percentile(b, 97.5)), "p_signflip": float((1 + (pm >= o - 1e-9).sum()) / (1 + NPERM))}
def cluster_ci(d):                                          # DeepWeeds only: capture dates resampled as units
    v = d[dw]; s = np.array([v[date[dw] == t].sum() for t in dates]); m = np.array([(date[dw] == t).sum() for t in dates]); b = s[CL].sum(1) / m[CL].sum(1) * 100
    return float(np.percentile(b, 2.5)), float(np.percentile(b, 97.5))
rows, seen = [], set()
def add(e, fam, ga, gb, la, lb, scheme, sens=""):
    k_ = (e, fam, frozenset(la), frozenset(lb), sens)
    if k_ in seen: return
    seen.add(k_); a = frac(e, la); b = frac(e, lb); d = a - b
    for scope, ix in [("overall", np.arange(n))] + [(x, ixd[x]) for x in DS]:
        r = {"family": fam, "encoder": e, "scheme": scheme, "group_A": ga, "layers_A": " ".join(str(num(l)) for l in la), "group_B": gb, "layers_B": "pooled" if lb == ["final_output"] else " ".join(str(num(l)) for l in lb), "n_layers_A": len(la), "n_layers_B": len(lb),
             "scope": scope, "acc_A": 100 * a[ix].mean(), "acc_B": 100 * b[ix].mean(), **est(d, ix), "sensitivity": sens}
        if scope == "DeepWeeds" and audit_dates: r["cluster_ci95_lo_by_capture_date"], r["cluster_ci95_hi_by_capture_date"] = cluster_ci(d)
        rows.append(r)
for e in ENC:
    for sch in ["architecture", "tested_depth"]:
        gs = [g for g in G if g["encoder"] == e and g["scheme"] == sch]
        for ga, gb in itertools.combinations(gs, 2):
            if len(ga["layers"]) >= 2 and len(gb["layers"]) >= 2: add(e, "GG", gb["group"], ga["group"], gb["layers"], ga["layers"], sch)   # deeper group minus shallower group
    for g in [g for g in G if g["encoder"] == e]: add(e, "GP", g["group"], "pooled output", g["layers"], ["final_output"], g["scheme"])
    dupl = [l for l in tested[e] if ndiff(cond[(e, l)], cond[(e, "final_output")]) == 0]
    for g in [g for g in G if g["encoder"] == e and set(g["layers"]) & set(dupl) and set(g["layers"]) - set(dupl)]:
        add(e, "GP", g["group"], "pooled output", [l for l in g["layers"] if l not in dupl], ["final_output"], g["scheme"], sens="without the layers whose demonstrations equal the pooled output: " + " ".join(str(num(l)) for l in dupl))
R = pd.DataFrame(rows); main = R.sensitivity == ""
for c in ["holm_within_encoder", "holm_global", "tests_within_encoder", "tests_global"]: R[c] = np.nan
for fam in ["GG", "GP"]:
    for per_ds in [False, True]:
        g = R[main & (R.family == fam) & ((R.scope != "overall") == per_ds)]
        if not len(g): continue
        R.loc[g.index, "holm_global"] = holm(g.p_signflip); R.loc[g.index, "tests_global"] = len(g)
        for e, h in g.groupby("encoder"): R.loc[h.index, "holm_within_encoder"] = holm(h.p_signflip); R.loc[h.index, "tests_within_encoder"] = len(h)
R["correction_family"] = np.where(main, R.family + np.where(R.scope == "overall", "_overall", "_per_dataset"), "none (sensitivity)")
R.to_csv(os.path.join(OUT, "group_contrasts.csv"), index=False)

# ================================================================ 4. every layer of A against every layer of B
def mcn(a, b):
    oa, ob = int((W[a] & ~W[b]).sum()), int((~W[a] & W[b]).sum())
    return {"acc_A": 100 * W[a].mean(), "acc_B": 100 * W[b].mean(), "diff_pp": 100 * (W[a].mean() - W[b].mean()), "only_A_correct": oa, "only_B_correct": ob, "n_matched": n, "mcnemar_exact_p": 1.0 if oa + ob == 0 else float(stats.binomtest(oa, oa + ob, 0.5).pvalue)}
P, pseen = [], {}
GGo = R[main & (R.family == "GG") & (R.scope == "overall")]
for r in GGo.itertuples():
    for la in r.layers_A.split():
        for lb in r.layers_B.split():
            a, b = f"layer_{int(la):02d}", f"layer_{int(lb):02d}"; k_ = (r.encoder, a, b)
            if k_ not in pseen: pseen[k_] = {"family": "cross_group_layer_pair", "encoder": r.encoder, "A": a, "B": b, **mcn(cond[(r.encoder, a)], cond[(r.encoder, b)]), "queries_with_different_demonstrations": ndiff(cond[(r.encoder, a)], cond[(r.encoder, b)]), "group_contrasts": []}
            pseen[k_]["group_contrasts"].append(f"{r.scheme}: {r.group_A} vs {r.group_B}")
for e in ENC:
    for l in tested[e]:
        nd = ndiff(cond[(e, l)], cond[(e, "final_output")]); P.append({"family": "layer_against_pooled", "encoder": e, "A": l, "B": "final_output", **mcn(cond[(e, l)], cond[(e, "final_output")]), "queries_with_different_demonstrations": nd, "group_contrasts": [], "excluded_identical_demonstrations": nd == 0})
P = pd.DataFrame(list(pseen.values()) + P); P["excluded_identical_demonstrations"] = P.excluded_identical_demonstrations.fillna(False).astype(bool); P["group_contrasts"] = P.group_contrasts.map("; ".join)
for c in ["holm_within_encoder", "holm_global", "tests_within_encoder", "tests_global"]: P[c] = np.nan
for fam, g in P[~P.excluded_identical_demonstrations].groupby("family"):
    P.loc[g.index, "holm_global"] = holm(g.mcnemar_exact_p); P.loc[g.index, "tests_global"] = len(g)
    for e, h in g.groupby("encoder"): P.loc[h.index, "holm_within_encoder"] = holm(h.mcnemar_exact_p); P.loc[h.index, "tests_within_encoder"] = len(h)
P.to_csv(os.path.join(OUT, "layer_pair_contrasts.csv"), index=False)
CP = P[P.family == "cross_group_layer_pair"].set_index(["encoder", "A", "B"]); S4 = []
for r in GGo.itertuples():
    pr = [CP.loc[(r.encoder, f"layer_{int(a):02d}", f"layer_{int(b):02d}")] for a in r.layers_A.split() for b in r.layers_B.split()]; d = np.array([x.diff_pp for x in pr]); hw = np.array([x.holm_within_encoder for x in pr]); hg = np.array([x.holm_global for x in pr])
    S4.append({"encoder": r.encoder, "scheme": r.scheme, "group_A": r.group_A, "layers_A": r.layers_A, "group_B": r.group_B, "layers_B": r.layers_B, "layer_pairs": len(pr), "pairs_A_higher": int((d > 0).sum()), "pairs_B_higher": int((d < 0).sum()), "pairs_equal": int((d == 0).sum()),
               "supported_A_within_encoder": int(((d > 0) & (hw < ALPHA)).sum()), "supported_B_within_encoder": int(((d < 0) & (hw < ALPHA)).sum()), "supported_A_global": int(((d > 0) & (hg < ALPHA)).sum()), "supported_B_global": int(((d < 0) & (hg < ALPHA)).sum()),
               "smallest_gap_A_minus_B_pp": float(d.min()), "largest_gap_A_minus_B_pp": float(d.max()), "every_A_beats_every_B": bool(((d > 0) & (hw < ALPHA)).all()), "every_B_beats_every_A": bool(((d < 0) & (hw < ALPHA)).all())})
S4 = pd.DataFrame(S4); S4.to_csv(os.path.join(OUT, "group_pair_layer_summary.csv"), index=False)

# ================================================================ 5. summaries
O = R[main & (R.scope == "overall")]; D = R[main & (R.scope != "overall")]; ES = []
def rev(r):
    d = D[(D.family == r.family) & (D.encoder == r.encoder) & (D.layers_A == r.layers_A) & (D.layers_B == r.layers_B)]; x = d[np.sign(d.diff_pp) == -np.sign(r.diff_pp)]
    return "; ".join(f"{t.scope} {t.diff_pp:+.1f}" for t in x.itertuples()) or "none"
for e in ENC:
    gg = O[(O.family == "GG") & (O.encoder == e)]; gp = O[(O.family == "GP") & (O.encoder == e)]; allp = gp[gp.scheme == "all"].iloc[0]; lp = P[(P.family == "layer_against_pooled") & (P.encoder == e) & ~P.excluded_identical_demonstrations]
    row = {"encoder": e, "tested_layers": " ".join(str(num(l)) for l in tested[e]), "group_pairs_evaluable": len(gg)}
    if len(gg):
        b = gg.sort_values("p_signflip").iloc[0]; s = S4[(S4.encoder == e) & (S4.layers_A == b.layers_A) & (S4.layers_B == b.layers_B)].iloc[0]
        row.update(strongest_group_pair=f"blocks {b.layers_A} minus blocks {b.layers_B}", gg_diff_pp=b.diff_pp, gg_ci=f"{b.ci95_lo_unadjusted:.1f} to {b.ci95_hi_unadjusted:.1f}", gg_p=b.p_signflip, gg_holm_within=b.holm_within_encoder, gg_holm_global=b.holm_global,
                   gg_supported_within=b.holm_within_encoder < ALPHA, gg_supported_global=b.holm_global < ALPHA, gg_dataset_reversals=rev(b), gg_layer_pairs_supported=f"{s.supported_A_within_encoder + s.supported_B_within_encoder} of {s.layer_pairs}", gg_every_member_wins=bool(s.every_A_beats_every_B or s.every_B_beats_every_A))
    row.update(all_hidden_minus_pooled_pp=allp.diff_pp, gp_ci=f"{allp.ci95_lo_unadjusted:.1f} to {allp.ci95_hi_unadjusted:.1f}", gp_p=allp.p_signflip, gp_holm_within=allp.holm_within_encoder, gp_holm_global=allp.holm_global, gp_supported_within=allp.holm_within_encoder < ALPHA, gp_supported_global=allp.holm_global < ALPHA,
               gp_dataset_reversals=rev(allp), layers_above_pooled_supported_within=int(((lp.diff_pp > 0) & (lp.holm_within_encoder < ALPHA)).sum()), layers_below_pooled_supported_within=int(((lp.diff_pp < 0) & (lp.holm_within_encoder < ALPHA)).sum()), layers_tested_against_pooled=len(lp))
    ES.append(row)
ES = pd.DataFrame(ES); ES.to_csv(os.path.join(OUT, "encoder_summary.csv"), index=False)
q = O[(O.encoder == "Qwen3VL") & (O.scheme == "all")].iloc[0]; draft = json.load(open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "paper_tables", "layer_vlm_numbers.json")))["encoder_level"]["Qwen3VL"]
sanity = {"draft": draft, "this_analysis": {"mean_of_five_layers_pp": q.diff_pp, "lo": q.ci95_lo_unadjusted, "hi": q.ci95_hi_unadjusted}, "note": "the estimate is the same quantity on the same 500 queries; the draft interval resamples the 500 queries without stratification, this one resamples within dataset, and the seeds differ"}
json.dump(sanity, open(os.path.join(OUT, "qwen_sanity_check.json"), "w"), indent=1)
show = ["encoder", "scheme", "layers_A", "layers_B", "n_matched", "acc_A", "acc_B", "diff_pp", "ci95_lo_unadjusted", "ci95_hi_unadjusted", "p_signflip", "holm_within_encoder", "holm_global"]
out = ["# Groups of hidden layers within an encoder (K = 2, 500 queries, saved Claude predictions)", "", "Exploratory. No request was sent.", "", "## Audit", ""] + [f"- {k}: {v}" for k, v in audit.items()]
out += ["", "## Layers without downstream predictions", "", md(INV.groupby("encoder", sort=False).apply(lambda g: pd.Series({"layers_in_atlas": len(g), "evaluated_with_Claude": int(g.downstream_predictions.sum()), "retrieval_only": " ".join(str(num(l)) if l.startswith("layer") else l for l in g[~g.downstream_predictions].layer)})).reset_index()), "",
        "## Groups", "", md(pd.DataFrame(manifest["groups"]).assign(layers=lambda d: d.layers.map(lambda x: " ".join(map(str, x))))), "", "## Group against group, overall (A is the deeper group)", "", md(O[O.family == "GG"][show]), "",
        f"Tests: within encoder {O[O.family == 'GG'].groupby('encoder', sort=False).size().to_dict()}, global {int((O.family == 'GG').sum())}.", "", "## Every layer of A against every layer of B", "", md(S4), "", "## Group against pooled output, overall", "", md(O[O.family == "GP"][show + ["n_layers_A"]]), "",
        f"Tests: within encoder {O[O.family == 'GP'].groupby('encoder', sort=False).size().to_dict()}, global {int((O.family == 'GP').sum())}.", "", "## Sensitivity: without layers whose demonstrations equal the pooled output", "", md(R[~main & (R.scope == "overall")][show[:10] + ["sensitivity"]]) if (~main).any() else "none", "",
        "## Per dataset: contrasts with Holm within encoder below 0.05", "", md(D[D.holm_within_encoder < ALPHA][["family", "scope"] + show]) if (D.holm_within_encoder < ALPHA).any() else "none", "", f"Per-dataset tests: GG global {int((D.family == 'GG').sum())}, GP global {int((D.family == 'GP').sum())}.", "",
        "## DeepWeeds: capture dates as resampling units", "", f"{audit_dates} capture dates among the 100 DeepWeeds queries (from the file names). No other dataset has a sampling unit in its metadata.", "",
        md(R[main & (R.scope == "DeepWeeds") & (R.family == "GG")][["encoder", "layers_A", "layers_B", "diff_pp", "ci95_lo_unadjusted", "ci95_hi_unadjusted", "cluster_ci95_lo_by_capture_date", "cluster_ci95_hi_by_capture_date"]]), "", "## Summary per encoder", "", md(ES), "", "## Qwen3-VL sanity check", "", json.dumps(sanity, indent=1), ""]
open(os.path.join(OUT, "SUMMARY.md"), "w").write("\n".join(out)); print("\n".join(out))
