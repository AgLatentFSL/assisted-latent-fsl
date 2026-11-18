"""Held-out retrieval evaluation for latent_afsl.py (stages: test, ageval). No VLM calls."""
import json, os
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.metrics import f1_score
import latent_afsl as LA
from latent_afsl import A, B, B_RAW, D, DINO_DEFAULT, K, rname

NBOOT = 10000


def derived(out):
    """Contrast readouts fixed by development data: best intermediate CLS (C), best intermediate patch mean (E)."""
    M = json.load(open(os.path.join(out, "manifest.json"))); Dv = pd.read_csv(os.path.join(out, "dev_retrieval.csv"))
    def best(model, tok, blocks):
        s = Dv[(Dv.model == model) & (Dv.token == tok) & Dv.block.isin(blocks)].sort_values(["macro_f1", "acc", "block"], ascending=[False, False, True]).iloc[0]
        return (model, tok, int(s["block"]))
    H, Hd = tuple(M["selected_vit"]), tuple(M["selected_dinov2"])
    return {"H": H, "Hd": Hd, "C": best("vit", "cls", range(1, 12)), "E": best("vit", "mean", range(1, 12)), "Cd": best("dinov2", "cls", range(1, 12)), "Ed": best("dinov2", "mean", range(1, 13))}


def ci(d, rng):
    m = d[rng.integers(0, len(d), size=(NBOOT, len(d)))].mean(axis=1); return float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5))


def cluster_ci(d, groups, rng):
    g = pd.Series(d).groupby(np.asarray(groups)); sums, ns = g.sum().to_numpy(), g.size().to_numpy()
    idx = rng.integers(0, len(sums), size=(NBOOT, len(sums))); m = sums[idx].sum(axis=1) / ns[idx].sum(axis=1)
    return float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5))


def wil(d):
    return 1.0 if not np.any(d != 0) else float(stats.wilcoxon(d).pvalue)


def stage_test(out):
    ids, dqm, DB, panel, banks, base_bank, base_q = LA.all_ids(out)
    FT = LA.Feats(out, ids); R = derived(out); H, Hd = R["H"], R["Hd"]
    classes = sorted(panel["label"].unique()); panel["capture_date"] = panel["capture_date"].astype(str)
    test = panel[panel["role"] != "dev"].reset_index(drop=True)          # 153 test queries; the 27 dev-role queries are prompt-development queries
    qids, qlab, qdate = test["query_id"].tolist(), test["label"].tolist(), test["capture_date"].tolist()
    readouts = [A, B_RAW] + LA.VIT_CANDS + LA.DINO_CANDS
    key = {"A": A, "B": B, "Braw": B_RAW, "C": R["C"], "D": D, "E": R["E"], "H": H, "DINO": DINO_DEFAULT, "Hd": Hd}
    banks["capture_date"] = banks["capture_date"].astype(str)
    grid, perq = [], {}
    for bt in ("DD", "SD"):
        for r in readouts:
            Qx = FT.get(r, qids); ms = []; acc = np.zeros(len(qids)); same = np.zeros(len(qids)); sd = np.zeros(len(qids)); sdc = np.zeros(len(qids)); top1 = np.zeros(len(qids)); nb = 0
            for s, b in banks[banks.bank_type == bt].groupby("bank_seed"):
                b = b.sort_values("bank_idx"); bl, bd = b["label"].tolist(), b["capture_date"].tolist()
                res = LA.retrieve(Qx, FT.get(r, b["ref_id"]), bl, qlab); ms.append(LA.summarize(res, qlab, classes)); nb += 1
                acc += [x["vote"] == t for x, t in zip(res, qlab)]; same += [x["same"] for x in res]; top1 += [x["top1"] for x in res]
                sd += [sum(bd[j] == d_ for j in x["top"]) for x, d_ in zip(res, qdate)]
                sdc += [sum(bd[j] == d_ and bl[j] == t for j in x["top"]) for x, d_, t in zip(res, qdate, qlab)]
            grid.append({"bank_type": bt, "readout": rname(r), "model": r[0], "token": r[1], "block": r[2], **{k: float(np.mean([m[k] for m in ms])) for k in ms[0]},
                         "acc_min_bank": float(min(m["acc"] for m in ms)), "acc_max_bank": float(max(m["acc"] for m in ms)), "top4_same_date_frac": float(sd.sum() / (nb * len(qids) * K)),
                         "same_class_refs_same_date_frac": float(sdc.sum() / max(same.sum(), 1))})
            perq[(bt, r)] = {"acc": acc / nb, "same": same / nb / K, "top1": top1 / nb}
    G = pd.DataFrame(grid); G.to_csv(os.path.join(out, "test_retrieval.csv"), index=False)
    rng = np.random.default_rng(LA.SEED); cons = []
    def contrast(name, x, y, question):
        for bt in ("DD", "SD"):
            for m in ("acc", "same", "top1"):
                d = perq[(bt, y)][m] - perq[(bt, x)][m]; lo, hi = ci(d, rng); clo, chi = cluster_ci(d, qdate, rng)
                fresh = d[(test["role"] == "audit_only").to_numpy()]
                cons.append({"contrast": name, "question": question, "x": rname(x), "y": rname(y), "bank_type": bt, "metric": m, "x_mean": float(perq[(bt, x)][m].mean()), "y_mean": float(perq[(bt, y)][m].mean()),
                             "diff": float(d.mean()), "ci_lo": lo, "ci_hi": hi, "date_cluster_ci_lo": clo, "date_cluster_ci_hi": chi, "wilcoxon_p": wil(d), "queries_up": int((d > 0).sum()), "queries_down": int((d < 0).sum()),
                             "dates_up": int((pd.Series(d).groupby(qdate).mean() > 0).sum()), "dates_down": int((pd.Series(d).groupby(qdate).mean() < 0).sum()), "diff_fresh_90": float(fresh.mean())})
    contrast("H - A", A, H, "selected hidden state versus the pooled baseline (primary retrieval contrast)")
    contrast("B - A", A, B, "effect of the pooler: final CLS (pooler input) versus pooler_output")
    contrast("Braw - B", B, B_RAW, "effect of the final LayerNorm on the final CLS")
    contrast("C - B", B, R["C"], "depth with the CLS token: development-best intermediate CLS versus final CLS")
    contrast("E - D", D, R["E"], "depth with patch tokens: development-best intermediate patch mean versus final patch mean")
    contrast("D - B", B, D, "token readout at block 12: patch mean versus CLS")
    hb = H[2]; contrast(f"mean - cls at block {hb}", ("vit", "cls", hb), ("vit", "mean", hb), f"token readout at block {hb}: patch mean versus CLS")
    contrast("H - B", B, H, "selected hidden state versus the stronger final-CLS baseline")
    contrast("Hd - DINO", DINO_DEFAULT, Hd, "DINOv2: development-selected readout versus the conventional final CLS")
    contrast("DINO - A", A, DINO_DEFAULT, "backbone comparison: DINOv2 final CLS versus ViT pooler_output")
    Cn = pd.DataFrame(cons); Cn.to_csv(os.path.join(out, "test_contrasts.csv"), index=False)
    # per-class and per-date drivers of the primary retrieval contrast (date-disjoint banks)
    d = perq[("DD", H)]["acc"] - perq[("DD", A)]["acc"]
    pd.DataFrame({"label": qlab, "d": d, "a": perq[("DD", A)]["acc"], "h": perq[("DD", H)]["acc"]}).groupby("label").agg(n=("d", "size"), pooled_acc=("a", "mean"), selected_acc=("h", "mean"), diff=("d", "mean")).to_csv(os.path.join(out, "test_H_minus_A_by_class.csv"))
    pd.DataFrame({"date": qdate, "d": d, "a": perq[("DD", A)]["acc"], "h": perq[("DD", H)]["acc"]}).groupby("date").agg(n=("d", "size"), pooled_acc=("a", "mean"), selected_acc=("h", "mean"), diff=("d", "mean")).to_csv(os.path.join(out, "test_H_minus_A_by_date.csv"))

    # ---- frozen base bank: per-query neighborhoods for the key readouts (this is the bank the VLM sees)
    bb = base_bank.reset_index(drop=True); bl, bd, bid = bb["label"].tolist(), bb["capture_date"].astype(str).tolist(), bb["ref_id"].tolist()
    allq = panel.reset_index(drop=True); rows = []; nb = {}
    for nm, r in key.items():
        res = LA.retrieve(FT.get(r, allq["query_id"]), FT.get(r, bid), bl, allq["label"].tolist()); nb[nm] = res
        for (_, q), x in zip(allq.iterrows(), res):
            rows.append({"query_id": q["query_id"], "query_date": q["capture_date"], "role": q["role"], "true_label": q["label"], "readout_key": nm, "readout": rname(r),
                         "neighbor_ids": json.dumps([bid[j] for j in x["top"]]), "neighbor_labels": json.dumps(x["labels"]), "neighbor_dates": json.dumps([bd[j] for j in x["top"]]),
                         "cosines": json.dumps([round(c, 6) for c in x["cos"]]), "vote": x["vote"], "margin": x["margin"], "same_class": x["same"], "top1_same": x["top1"], "vote_correct": int(x["vote"] == q["label"])})
    N = pd.DataFrame(rows); N.to_csv(os.path.join(out, "base_bank_neighbors.csv"), index=False)
    tmask = (allq["role"] != "dev").to_numpy(); ov = []
    for x, y in (("A", "H"), ("A", "B"), ("B", "C"), ("D", "E"), ("B", "D"), ("B", "H"), ("DINO", "Hd"), ("A", "DINO")):
        sh = np.array([len(set(a["top"]) & set(b["top"])) for a, b in zip(nb[x], nb[y])])[tmask]; ident = np.array([a["top"] == b["top"] for a, b in zip(nb[x], nb[y])])[tmask]
        vc = np.array([a["vote"] != b["vote"] for a, b in zip(nb[x], nb[y])])[tmask]; lab = np.array([a["labels"] == b["labels"] for a, b in zip(nb[x], nb[y])])[tmask]
        ax = np.array([a["vote"] == t for a, t in zip(nb[x], allq["label"])])[tmask]; ay = np.array([b["vote"] == t for b, t in zip(nb[y], allq["label"])])[tmask]
        ov.append({"pair": f"{y} vs {x}", "x": rname(key[x]), "y": rname(key[y]), "mean_shared_of_4": float(sh.mean()), "identical_ordered_top4": int(ident.sum()), "same_label_sequence": int(lab.sum()),
                   "vote_changed": int(vc.sum()), "vote_fixed": int((~ax & ay).sum()), "vote_broken": int((ax & ~ay).sum()), "x_vote_acc": float(ax.mean()), "y_vote_acc": float(ay.mean()), "n": int(tmask.sum())})
    pd.DataFrame(ov).to_csv(os.path.join(out, "base_bank_overlap.csv"), index=False)

    # ---- verification against existing artifacts
    ver = {}
    old = dict(np.load(os.path.join(LA.AUDIT, "embeddings_vit.npz"))); smp = sorted(old)[:300]
    new = FT.get(A, smp); o = np.stack([old[f] for f in smp]).astype(np.float64); o /= np.linalg.norm(o, axis=1, keepdims=True)
    ver["vit_pooled_fresh_vs_audit_cache_min_cosine_300"] = float((new * o).sum(axis=1).min())
    old = dict(np.load(os.path.join(LA.AUDIT, "embeddings_dinov2_last.npz"))); new = FT.get(DINO_DEFAULT, smp); o = np.stack([old[f] for f in smp]).astype(np.float64); o /= np.linalg.norm(o, axis=1, keepdims=True)
    ver["dinov2_cls12_fresh_vs_audit_cache_min_cosine_300"] = float((new * o).sum(axis=1).min())
    def same_top4(path, r, qlist):
        saved = {json.loads(l)["query_id"]: [x[0] for x in json.loads(l)["ranked"][:K]] for l in open(path)}
        res = LA.retrieve(FT.get(r, qlist), FT.get(r, bid), bl, ["x"] * len(qlist)); return int(sum([bid[j] for j in x["top"]] == saved[q] for q, x in zip(qlist, res))), len(qlist)
    ver["vit_pooled_top4_matches_base_run_20_queries"] = same_top4(os.path.join(LA.BASE, "offline", "rankings_vit.jsonl"), A, base_q["query_id"].tolist())
    ver["dinov2_cls12_top4_matches_base_run_20_queries"] = same_top4(os.path.join(LA.BASE, "offline", "rankings_dinov2_last.jsonl"), DINO_DEFAULT, base_q["query_id"].tolist())
    dvq = pd.read_csv(os.path.join(LA.RUNS, "deepweeds_demo_value_seed42", "offline_retrieval.csv"))["query_id"].tolist()
    ver["dinov2_cls12_top4_matches_demo_value_run_90_queries"] = same_top4(os.path.join(LA.RUNS, "deepweeds_demo_value_seed42", "rankings_dinov2_last.jsonl"), DINO_DEFAULT, dvq)
    json.dump({"verification": ver, "derived_readouts": {k: list(v) for k, v in R.items()}, "n_test_queries": len(test), "test_dates": sorted(set(qdate)),
               "test_class_counts": test["label"].value_counts().sort_index().to_dict(), "queries_per_date": test["capture_date"].value_counts().sort_index().to_dict()}, open(os.path.join(out, "test_meta.json"), "w"), indent=1)
    print(json.dumps(ver, indent=1)); print({k: rname(v) for k, v in R.items()})
    show = G[(G.bank_type == "DD") & G.readout.isin([rname(v) for v in key.values()])][["readout", "acc", "macro_f1", "top1", "same_frac", "acc_min_bank", "acc_max_bank"]]
    print(show.round(3).to_string(index=False))
    print(Cn[(Cn.metric == "acc") & (Cn.bank_type == "DD")][["contrast", "x_mean", "y_mean", "diff", "ci_lo", "ci_hi", "date_cluster_ci_lo", "date_cluster_ci_hi", "wilcoxon_p", "dates_up", "dates_down", "diff_fresh_90"]].round(3).to_string(index=False))
    print(pd.DataFrame(ov)[["pair", "mean_shared_of_4", "identical_ordered_top4", "same_label_sequence", "vote_fixed", "vote_broken", "x_vote_acc", "y_vote_acc"]].round(3).to_string(index=False))


def stage_ageval(out):
    R = derived(out); H, Hd = R["H"], R["Hd"]
    LS = os.path.join(LA.RUNS, "latent_search"); OFF = os.path.join(LA.RUNS, "offline_layer_stats")
    Fv, Fd = dict(np.load(os.path.join(LS, "feats_vit.npz"))), dict(np.load(os.path.join(LS, "feats_dinov2.npz")))
    arb = pd.read_csv(os.path.join(LA.RUNS, "arbab_eval_seed42", "per_query_results.csv"), keep_default_na=False)
    def X(r, ds):
        F = Fv if r[0] == "vit" else Fd; k = f"{ds}|eval"
        M = F[k + "|extra"] if r[1] == "pooled" else F[k + "|" + r[1]][:, r[2] - 1]
        M = M.astype(np.float64); return M / np.linalg.norm(M, axis=1, keepdims=True)
    key = {"A": A, "B": B, "C": R["C"], "D": D, "E": R["E"], "H": H, "DINO": DINO_DEFAULT, "Hd": Hd}; rows, votes, nbr = [], [], []
    for ds in ["Bean Leaf Lesions", "Mango Leaf Disease", "Dangerous Insects", "DeepWeeds", "Soybean Seeds", "Soybean Diseases"]:
        pool = pd.read_csv(os.path.join(OFF, f"pool_{ds.replace(' ', '_')}.csv")); y = pool["label"].astype(str).tolist(); cl = sorted(set(y)); rec = {"dataset": ds, "n": len(y)}
        for nm, r in key.items():
            x = X(r, ds); S = x @ x.T; np.fill_diagonal(S, -np.inf); pred, same, tops = [], [], []
            for q in range(len(y)):
                top = np.lexsort((np.arange(len(y)), -S[q]))[:K]; labs = [y[j] for j in top]; pred.append(LA.vote(labs, [float(S[q, j]) for j in top])); same.append(sum(l == y[q] for l in labs)); tops.append(top.tolist())
            rec[f"{nm}_acc"] = float(np.mean([p == t for p, t in zip(pred, y)])); rec[f"{nm}_f1"] = float(f1_score(y, pred, labels=cl, average="macro", zero_division=0)); rec[f"{nm}_same"] = float(np.mean(same) / K)
            votes.append(pd.DataFrame({"dataset": ds, "query_idx": range(len(y)), "true_label": y, "readout_key": nm, "vote": pred, "same_class": same, "top4_idx": [json.dumps(t) for t in tops]}))
        rows.append(rec)
    T = pd.DataFrame(rows); V = pd.concat(votes, ignore_index=True); T.to_csv(os.path.join(out, "ageval_retrieval.csv"), index=False); V.to_csv(os.path.join(out, "ageval_votes.csv"), index=False)
    W = V.pivot(index=["dataset", "query_idx"], columns="readout_key", values="vote"); tl = V.drop_duplicates(["dataset", "query_idx"]).set_index(["dataset", "query_idx"])["true_label"]
    tl = tl.loc[W.index]
    a = arb.set_index(["dataset", "query_idx"])
    chk = {"n": len(W), "A_vote_equals_cached_vote_vit": int((W["A"].astype(str) == a.loc[W.index, "vote_vit"].astype(str)).sum()), "DINO_vote_equals_cached_vote_dinov2": int((W["DINO"].astype(str) == a.loc[W.index, "vote_dinov2"].astype(str)).sum())}
    pooled = {k: float((W[k] == tl).mean()) for k in key}
    def mc(x, y):
        cx, cy = (W[x] == tl), (W[y] == tl); w, l = int((cy & ~cx).sum()), int((~cy & cx).sum()); return {"wins": w, "losses": l, "p": 1.0 if w + l == 0 else float(stats.binomtest(w, w + l, 0.5).pvalue)}
    json.dump({"check_against_cached_votes": chk, "pooled_vote_accuracy_583": pooled, "H_vs_A": mc("A", "H"), "B_vs_A": mc("A", "B"), "H_vs_B": mc("B", "H"), "Hd_vs_DINO": mc("DINO", "Hd"), "DINO_vs_A": mc("A", "DINO"),
               "readouts": {k: rname(v) for k, v in key.items()}}, open(os.path.join(out, "ageval_meta.json"), "w"), indent=1)
    print(chk, {k: round(100 * v, 1) for k, v in pooled.items()}); print("H vs A", mc("A", "H"), "| B vs A", mc("A", "B"), "| Hd vs DINO", mc("DINO", "Hd"))
    print(T[["dataset", "n"] + [f"{k}_acc" for k in key]].round(3).to_string(index=False))


def run(out, stage):
    assert os.path.exists(os.path.join(out, "manifest.json")), "run the dev stage first; the manifest must exist before test metrics"
    {"test": stage_test, "ageval": stage_ageval}[stage](out)
