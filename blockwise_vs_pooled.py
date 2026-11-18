"""Blockwise retrieval against the pooled-embedding baseline of Arshad et al., at K = 1, 2, 4, 8. Local only; zero VLM calls.

Baseline: google/vit-base-patch16-224-in21k pooler_output ("vit|native" in runs/layer_retrieval_audit), the retriever of
the assisted few-shot pipeline. Candidates: every block readout of every cached encoder.

Stages (run in this order; each later stage refuses to run before the earlier one has written its files):
  features   encode the DeepWeeds development split with every encoder (compressed cache, resumable)
  dev        sweep blocks and K on the development split
  select     apply the pre-recorded selection rule to the development sweep and lock config_locked.json
  holdout    evaluate the locked configuration on the date-disjoint held-out queries (paired cluster bootstrap)
  sweep      exploratory sweep over every split of the layer audit (all datasets)
  report     figures, SUMMARY.md and checks.json
Usage (from repo/): python blockwise_vs_pooled.py <out_dir> <stage> [encoders for the features stage]
"""
import hashlib, itertools, json, os, sys, time
import numpy as np, pandas as pd

OUT, STAGE, EXTRA = sys.argv[1], sys.argv[2], sys.argv[3:]
os.makedirs(OUT, exist_ok=True)
sys.argv = ["x", os.path.join("runs", "layer_retrieval_audit"), "none"]
import layer_audit as LA

RUNS = os.path.join("runs"); KS = (1, 2, 4, 8); DEPTH = 8; RBO_P = 0.9; SEED = 20261115; NBOOT = 10000
BASE = "vit|native"                    # the pooled embedding of Arshad et al.
TOLERANCE_PP = 2.0                     # development results within this many percentage points of the best count as similar (fixed before any held-out result of this analysis)
ENC_ORDER = ["vit", "dinov2", "clip", "resnet", "siglip2", "bioclip", "dinov3", "qwen", "flava"]
dw = lambda i: os.path.join("data", "deepweeds", "images", i)
md5 = lambda p: hashlib.md5(open(p, "rb").read()).hexdigest()


# ------------------------------------------------------------------ evaluation units
def dev_unit():
    Q = pd.read_csv(os.path.join(RUNS, "latent_afsl", "dev_queries.csv")); B = pd.read_csv(os.path.join(RUNS, "latent_afsl", "dev_banks.csv"))
    q = [(i, dw(i), l, None, str(d)) for i, l, d in zip(Q.query_id, Q.label, Q.capture_date)]
    banks = [(f"dev bank seed {s}", [(i, dw(i), l, None, str(d)) for i, l, d in zip(g.ref_id, g.label, g.capture_date)]) for s, g in B.groupby("bank_seed")]
    return {"name": "DeepWeeds | development (date-disjoint, 10 banks)", "role": "development", "dataset": "DeepWeeds", "queries": q, "banks": banks}


def holdout_unit():
    if os.environ.get("BVP_CODE_TEST") == "1":       # code test only: the development unit stands in, so no held-out number is produced
        u = dev_unit(); return {**u, "name": "CODE TEST (development unit as stand-in)", "role": "holdout", "banks": u["banks"][:1]}
    P = pd.read_csv(os.path.join(RUNS, "deepweeds_split_audit", "query_panel.csv")); P = P[P.role.isin(["eval", "audit_only"])]
    R = pd.read_csv(os.path.join(RUNS, "deepweeds_matched_seed42", "reference_manifest.csv"))
    q = [(i, dw(i), l, m, str(d)) for i, l, m, d in zip(P.query_id, P.label, P.md5, P.capture_date)]
    return {"name": "DeepWeeds | held-out capture dates (153 queries, frozen date-disjoint bank)", "role": "holdout", "dataset": "DeepWeeds", "queries": q,
            "banks": [("frozen base bank (99)", [(i, dw(i), l, m, str(d)) for i, l, m, d in zip(R.ref_id, R.label, R.md5, R.capture_date)])]}


def audit_units():
    S, _ = LA.splits(); cov = {s["split_name"]: s for s in json.load(open(os.path.join(LA.out, "coverage_manifest.json")))["splits"]}; U = []
    for s in S:
        if not cov[s["split"]]["same_label_metrics"]: continue                 # numeric targets or one image per class: no diagnosis label to match
        U.append({"name": s["split"], "role": "exploratory sweep", "dataset": s["dataset"], "queries": [(a, b, c, d, "") for a, b, c, d in s["queries"]], "banks": [("bank", [(a, b, c, d, "") for a, b, c, d in s["bank"]])]})
    return U


def dev_split_for_features():
    u = dev_unit(); imgs = sorted({t[1] for t in u["queries"]} | {t[1] for _, b in u["banks"] for t in b})
    return {"split": u["name"], "tag": "devdw" + hashlib.md5(u["name"].encode()).hexdigest()[:5], "images": imgs}


# ------------------------------------------------------------------ cached latents, addressed by image path
_files, _open = {}, {}
def feature_index(enc):
    if enc not in _files:
        idx = {}
        for f in sorted(os.listdir(os.path.join(LA.out, "features"))):
            if f.startswith(enc + "__"):
                try: z = np.load(os.path.join(LA.out, "features", f))
                except Exception: continue                     # a cache file that is still being written
                [idx.setdefault(p, (f, i)) for i, p in enumerate(z["images"])]
        _files[enc] = idx
    return _files[enc]


def vectors(rep, paths):
    enc, ro = rep.split("|")[:2]; idx = feature_index(enc); key = "native" if ro == "native" else (f"gap{int(rep[-2:]):02d}" if enc == "resnet" else ro); rows = []
    for p in paths:
        f, i = idx[p]
        if (f, key) not in _open: _open[(f, key)] = np.load(os.path.join(LA.out, "features", f))[key]
        a = _open[(f, key)]; rows.append(a[i] if (ro == "native" or enc == "resnet") else a[i, int(rep[-2:]) - 1])
    X = np.stack(rows).astype(np.float64); return X / np.linalg.norm(X, axis=1, keepdims=True)


def available_reps(paths):
    R = [r for r in LA.reps() if all(p in feature_index(r[1]) for p in paths)]; miss = sorted({r[1] for r in LA.reps()} - {r[1] for r in R}); return R, miss


# ------------------------------------------------------------------ metrics
def rbo_ext(A, B, p=RBO_P):
    """Extrapolated rank-biased overlap of two equal-length ranked lists without ties (Webber, Moffat and Zobel 2010, Eq. 32 with s = l = k)."""
    k = A.shape[1]; agree = np.zeros((A.shape[0], k))
    for d in range(1, k + 1): agree[:, d - 1] = (A[:, :d, None] == B[:, None, :d]).any(2).sum(1) / d
    w = p ** np.arange(1, k + 1); return agree[:, -1] * p ** k + (1 - p) / p * (agree * w).sum(1)


def ndcg(rel, n_rel, k):
    """Binary relevance (Järvelin and Kekäläinen 2002). Ideal DCG uses min(k, same-diagnosis references in the eligible bank); NaN if the bank holds none."""
    disc = 1 / np.log2(np.arange(2, k + 2)); dcg = (rel[:, :k] * disc).sum(1); ideal = np.array([disc[:min(k, int(r))].sum() for r in n_rel]); return np.where(ideal > 0, dcg / np.where(ideal > 0, ideal, 1), np.nan)


def rank_unit(u, reps):
    """top-DEPTH reference indices per representation and bank; fixed bank order (by reference id) is the tie-break."""
    out = []
    for bname, bank in u["banks"]:
        bank = sorted(bank, key=lambda t: t[0]); qid = np.array([t[0] for t in u["queries"]]); bid = np.array([t[0] for t in bank])
        qm = np.array([t[3] or md5(t[1]) for t in u["queries"]]); bm = np.array([t[3] or md5(t[1]) for t in bank])
        inel = (qid[:, None] == bid[None, :]) | (qm[:, None] == bm[None, :]); ql = np.array([str(t[2]) for t in u["queries"]]); bl = np.array([str(t[2]) for t in bank])
        match = (ql[:, None] == bl[None, :]) & ~inel; elig = (~inel).sum(1); top = {}
        for r in reps:
            S = vectors(r[0], [t[1] for t in u["queries"]]) @ vectors(r[0], [t[1] for t in bank]).T; S[inel] = -np.inf; top[r[0]] = np.argsort(-S, kind="stable")[:, :DEPTH]
        out.append({"bank": bname, "bid": bid, "bl": bl, "ql": ql, "qid": qid, "match": match, "elig": elig, "n_rel": match.sum(1), "top": top,
                    "self_pairs": int((qid[:, None] == bid[None, :]).sum()), "dup_pairs": int(inel.sum() - (qid[:, None] == bid[None, :]).sum()),
                    "qdate": np.array([t[4] for t in u["queries"]]), "bdate": np.array([t[4] for t in bank])})
    return out


def pair_metrics(R, a, b, k):
    """per-query comparison of representation a (candidate) with b (reference list) at depth k"""
    A, B = R["top"][a][:, :k], R["top"][b][:, :k]; shared = (A[:, :, None] == B[:, None, :]).any(2).sum(1); ident = (A == B).all(1)
    ma = np.take_along_axis(R["match"], A, 1); mb = np.take_along_axis(R["match"], B, 1)
    return {"replaced": k - shared, "identical_order": ident, "same_set_reordered": (shared == k) & ~ident, "rbo": rbo_ext(A, B), "match_a": ma.sum(1), "match_b": mb.sum(1), "ndcg_a": ndcg(ma, R["n_rel"], k), "ndcg_b": ndcg(mb, R["n_rel"], k)}


def evaluate(u, tag):
    """writes per-query rows and aggregates for one unit; returns the aggregate frame"""
    paths = [t[1] for t in u["queries"]] + [t[1] for _, b in u["banks"] for t in b]; reps, missing = available_reps(paths); ids = [r[0] for r in reps]; assert BASE in ids, "pooled baseline latents missing"
    fam = {}; [fam.setdefault(r[4], []).append(r[0]) for r in reps if r[4]]; prev = {b: a for f in fam.values() for a, b in zip(f[:-1], f[1:])}
    RK = rank_unit(u, reps); agg, stab = [], []; pq_path = os.path.join(OUT, f"per_query_{tag}.csv.gz")
    if os.path.exists(pq_path): os.remove(pq_path)
    import gzip
    with gzip.open(pq_path, "wt", newline="") as fh:              # written one representation and K at a time to keep memory small
        first_chunk = True
        for r in reps:
            rid = r[0]
            for k in KS:
                parts = []
                for R in RK:
                    ok = R["elig"] >= DEPTH; m = pair_metrics(R, rid, BASE, k); ha, hb = m["match_a"] > 0, m["match_b"] > 0
                    tr = np.select([~hb & ha, hb & ~ha, hb & ha], ["added", "lost", "both"], "neither")
                    d = {"unit": u["name"], "bank": R["bank"], "query_id": R["qid"], "query_label": R["ql"], "query_date": R["qdate"], "representation": rid, "encoder": r[1], "readout": r[2], "block": r[3] if r[4] else "", "K": k,
                         "eligible": ok, "same_dx_in_bank": R["n_rel"], "replaced_vs_pooled": m["replaced"], "identical_order_vs_pooled": m["identical_order"], "same_set_reordered_vs_pooled": m["same_set_reordered"], "rbo_vs_pooled": np.round(m["rbo"], 4),
                         "same_dx_pooled": m["match_b"], "same_dx_block": m["match_a"], "transition": tr, "ndcg_pooled": np.round(m["ndcg_b"], 4), "ndcg_block": np.round(m["ndcg_a"], 4)}
                    if rid in prev:
                        p = pair_metrics(R, rid, prev[rid], k); d.update(replaced_vs_previous_block=p["replaced"], identical_order_vs_previous_block=p["identical_order"], same_set_reordered_vs_previous_block=p["same_set_reordered"], rbo_vs_previous_block=np.round(p["rbo"], 4))
                    else: d.update(replaced_vs_previous_block="", identical_order_vs_previous_block="", same_set_reordered_vs_previous_block="", rbo_vs_previous_block="")
                    parts.append(pd.DataFrame(d))
                chunk = pd.concat(parts, ignore_index=True); chunk = chunk[chunk.eligible]; chunk.to_csv(fh, index=False, header=first_chunk); first_chunk = False; agg.append(aggregate(chunk))
    for R in RK:
        nq = len(R["qid"])
        for f, members in fam.items():           # first block after which the ordered top-K list stays identical through the final block
            for k in KS:
                T = np.stack([R["top"][x][:, :k] for x in members], 1); same_as_last = (T == T[:, -1:, :]).all(2); first = np.array([next(i for i in range(len(members)) if same_as_last[q, i:].all()) for q in range(nq)])
                stab.append(pd.DataFrame({"unit": u["name"], "bank": R["bank"], "query_id": R["qid"], "family": f, "K": k, "first_stable_block": [int(members[i][-2:]) for i in first], "final_block": int(members[-1][-2:])}))
    ST = pd.concat(stab, ignore_index=True); ST.to_csv(os.path.join(OUT, f"stabilization_{tag}.csv.gz"), index=False, compression="gzip")
    A = pd.concat(agg, ignore_index=True); A.to_csv(os.path.join(OUT, f"aggregate_{tag}.csv"), index=False)
    audit = {"unit": u["name"], "role": u["role"], "queries": len(u["queries"]), "banks": len(u["banks"]), "bank_sizes": sorted({len(b) for _, b in u["banks"]}), "paired_query_bank_rows": int(sum(len(R["qid"]) for R in RK)),
             "encoders_evaluated": sorted({r[1] for r in reps}, key=ENC_ORDER.index), "encoders_without_latents": missing, "representations": len(reps),
             "self_match_pairs_excluded": int(sum(R["self_pairs"] for R in RK)), "byte_identical_pairs_excluded": int(sum(R["dup_pairs"] for R in RK)), "queries_with_fewer_than_8_eligible_references": int(sum((R["elig"] < DEPTH).sum() for R in RK)),
             "query_bank_pairs_without_a_same_diagnosis_reference": int(sum((R["n_rel"] == 0).sum() for R in RK)), "missing_labels": int(sum((R["ql"] == "nan").sum() + (R["bl"] == "nan").sum() for R in RK)),
             "query_ids_in_a_bank": int(sum(np.isin(R["qid"], R["bid"]).sum() for R in RK)),
             "query_dates_shared_with_bank_dates": (sorted(set(itertools.chain.from_iterable(set(R["qdate"]) & set(R["bdate"]) for R in RK))) if u["queries"][0][4] else "no capture-date metadata")}
    return A, audit


def aggregate(PQ):
    g = PQ.groupby(["unit", "representation", "encoder", "readout", "block", "K"], sort=False, dropna=False); rows = []
    for key, d in g:
        n = len(d); k = key[5]; t = d.transition.value_counts(); a, l, b, ne = (int(t.get(x, 0)) for x in ("added", "lost", "both", "neither")); miss = a + ne; dm = d.same_dx_block - d.same_dx_pooled; nd = d.ndcg_block - d.ndcg_pooled
        r = dict(zip(["unit", "representation", "encoder", "readout", "block", "K"], key)); r.update(n_paired=n, **{f"pct_replaced_{j}_of_K": (100 * (d.replaced_vs_pooled == j).mean() if j <= k else np.nan) for j in range(DEPTH + 1)},
                 mean_fraction_replaced_vs_pooled=float((d.replaced_vs_pooled / k).mean()), pct_identical_order_vs_pooled=100 * d.identical_order_vs_pooled.mean(), pct_same_set_reordered_vs_pooled=100 * d.same_set_reordered_vs_pooled.mean(), mean_rbo_vs_pooled=d.rbo_vs_pooled.mean(),
                 n_added=a, n_lost=l, n_both=b, n_neither=ne, pct_added=100 * a / n, pct_lost=100 * l / n, pct_both=100 * b / n, pct_neither=100 * ne / n, net_coverage_pp=100 * (a - l) / n,
                 coverage_pooled_pct=100 * (l + b) / n, coverage_block_pct=100 * (a + b) / n, pooled_misses=miss, pooled_misses_recovered_pct=(100 * a / miss if miss else np.nan),
                 mean_same_dx_pooled=d.same_dx_pooled.mean(), mean_same_dx_block=d.same_dx_block.mean(), mean_same_dx_change=dm.mean(), pct_same_dx_up=100 * (dm > 0).mean(), pct_same_dx_equal=100 * (dm == 0).mean(), pct_same_dx_down=100 * (dm < 0).mean(),
                 mean_ndcg_pooled=d.ndcg_pooled.mean(), mean_ndcg_block=d.ndcg_block.mean(), mean_ndcg_change=nd.mean(), n_ndcg_defined=int(nd.notna().sum()))
        if (d.replaced_vs_previous_block != "").any():
            pv = d[d.replaced_vs_previous_block != ""]; rp = pd.to_numeric(pv.replaced_vs_previous_block)
            r.update(mean_fraction_replaced_vs_previous_block=float((rp / k).mean()), pct_identical_order_vs_previous_block=100 * pv.identical_order_vs_previous_block.astype(bool).mean(),
                     pct_same_set_reordered_vs_previous_block=100 * pv.same_set_reordered_vs_previous_block.astype(bool).mean(), pct_any_replaced_vs_previous_block=100 * (rp > 0).mean(), mean_rbo_vs_previous_block=pd.to_numeric(pv.rbo_vs_previous_block).mean())
        rows.append(r)
    return pd.DataFrame(rows)


def cluster_boot(d, col_group, rng):
    """paired net coverage change (pp) with a cluster bootstrap over col_group; both methods are resampled together"""
    g = d.groupby(col_group).agg(a=("transition", lambda x: (x == "added").sum()), l=("transition", lambda x: (x == "lost").sum()), n=("transition", "size")); A, L, N = g.a.values, g.l.values, g.n.values
    idx = rng.integers(0, len(g), size=(NBOOT, len(g))); est = 100 * (A[idx].sum(1) - L[idx].sum(1)) / N[idx].sum(1)
    return float(100 * (A.sum() - L.sum()) / N.sum()), [float(np.percentile(est, 2.5)), float(np.percentile(est, 97.5))], len(g)


# ------------------------------------------------------------------ stages
P = lambda f: os.path.join(OUT, f)
if STAGE == "features":
    s = dev_split_for_features(); keep = EXTRA or list(LA.CKPT); full = dict(LA.CKPT); LA.CKPT = {k: v for k, v in full.items() if k in keep}; LA.extract_all([s]); LA.CKPT = full
    print("development latents:", [f for f in os.listdir(os.path.join(LA.out, "features")) if s["tag"] in f])

elif STAGE == "dev":
    A, audit = evaluate(dev_unit(), "development"); json.dump({"development": audit, "written": time.strftime("%Y-%m-%d %H:%M:%S")}, open(P("audit_development.json"), "w"), indent=1)
    print(json.dumps(audit, indent=1))

elif STAGE == "select":
    assert not os.path.exists(P("config_locked.json")), "configuration already locked"; assert not os.path.exists(P("aggregate_holdout.csv")), "held-out results exist"
    A = pd.read_csv(P("aggregate_development.csv")); C = A[A.block.notna() & (A.representation != BASE)].copy(); C["block"] = C.block.astype(int)
    best = C.net_coverage_pp.max(); near = C[C.net_coverage_pp >= best - TOLERANCE_PP]; kmin = near.K.min(); nk = near[near.K == kmin]
    early = nk.sort_values("block").groupby(["encoder", "readout"], sort=False).head(1)                 # earliest block within each encoder readout
    early = early.assign(order=early.encoder.map(ENC_ORDER.index)).sort_values(["net_coverage_pp", "order", "readout"], ascending=[False, True, True]); sel = early.iloc[0]
    PQ = pd.read_csv(P("per_query_development.csv.gz"), usecols=["query_id", "query_date", "representation", "K", "transition"], dtype={"query_date": str})
    d = PQ[(PQ.representation == sel.representation) & (PQ.K == sel.K)]; est, ci, ng = cluster_boot(d, "query_date", np.random.default_rng(SEED))
    cfg = {"locked_at": time.strftime("%Y-%m-%d %H:%M:%S"), "baseline": {"representation": BASE, "description": LA.NATIVE_DESC[BASE]},
           "candidate": {"representation": sel.representation, "encoder": sel.encoder, "checkpoint": LA.CKPT[sel.encoder], "readout": sel.readout, "block": int(sel.block), "K": int(sel.K)},
           "primary_metric": "net coverage change in percentage points = P(block list holds at least one same-diagnosis reference) - P(pooled list holds at least one), same K, all paired eligible queries",
           "selection_rule": f"development sweep over every block readout and K in {list(KS)}; all results within {TOLERANCE_PP} pp of the best development net coverage change count as similar; among them the smallest K; then the earliest block within each encoder readout; "
                             "remaining candidates from different encoders or readouts are ordered by development net coverage change, then by the encoder order of this file, then by readout name",
           "tolerance_pp": TOLERANCE_PP, "development": {"best_net_coverage_pp_any_candidate": float(best), "candidates_within_tolerance": int(len(near)), "K_values_within_tolerance": sorted(int(x) for x in near.K.unique()),
                                                          "remaining_after_K_and_earliest_block": early[["representation", "K", "net_coverage_pp"]].round(2).to_dict("records"),
                                                          "selected_net_coverage_pp": float(sel.net_coverage_pp), "selected_added": int(sel.n_added), "selected_lost": int(sel.n_lost), "selected_n_paired": int(sel.n_paired),
                                                          "selected_cluster_bootstrap_95ci_pp": ci, "bootstrap_unit": "capture date of the query", "bootstrap_groups": ng},
           "credible_candidate": bool(ci[0] > 0), "credibility_rule": "the 95% date-cluster bootstrap interval of the development net coverage change excludes zero",
           "held_out_comparison": {"unit": holdout_unit()["name"], "same_K_as_candidate": int(sel.K), "inference": f"paired cluster bootstrap over capture dates, {NBOOT} resamples, seed {SEED}"},
           "rbo": {"persistence_p": RBO_P, "depth": "K", "variant": "extrapolated, equal-length lists"}, "seed": SEED, "bootstrap_resamples": NBOOT}
    json.dump(cfg, open(P("config_locked.json"), "w"), indent=1); print(json.dumps(cfg, indent=1))

elif STAGE == "holdout":
    cfg = json.load(open(P("config_locked.json"))); A, audit = evaluate(holdout_unit(), "holdout"); cand, K = cfg["candidate"]["representation"], cfg["candidate"]["K"]
    PQ = pd.read_csv(P("per_query_holdout.csv.gz"), dtype={"query_date": str}); d = PQ[(PQ.representation == cand) & (PQ.K == K)]; rng = np.random.default_rng(SEED)
    est, ci, ng = cluster_boot(d, "query_date", rng); est_q, ci_q, nqg = cluster_boot(d, "query_id", np.random.default_rng(SEED)); t = d.transition.value_counts()
    from scipy.stats import binomtest
    a, l = int(t.get("added", 0)), int(t.get("lost", 0)); byc = d.groupby("query_label").transition.value_counts().unstack(fill_value=0).reindex(columns=["added", "lost", "both", "neither"], fill_value=0)
    res = {"evaluated_at": time.strftime("%Y-%m-%d %H:%M:%S"), "candidate": cand, "K": K, "n_paired": len(d), "added": a, "lost": l, "both": int(t.get("both", 0)), "neither": int(t.get("neither", 0)),
           "coverage_pooled_pct": 100 * (l + int(t.get("both", 0))) / len(d), "coverage_block_pct": 100 * (a + int(t.get("both", 0))) / len(d), "net_coverage_pp": est, "ci95_date_cluster_bootstrap_pp": ci, "independent_date_groups": ng,
           "ci95_query_bootstrap_pp_sensitivity": ci_q, "exact_mcnemar_p": float(binomtest(a, a + l, 0.5).pvalue) if a + l else 1.0, "pooled_misses": a + int(t.get("neither", 0)), "mean_same_dx_change": float((d.same_dx_block - d.same_dx_pooled).mean()),
           "mean_ndcg_change": float((d.ndcg_block - d.ndcg_pooled).mean()), "mean_fraction_replaced_vs_pooled": float((d.replaced_vs_pooled / K).mean()), "pct_identical_order_vs_pooled": float(100 * d.identical_order_vs_pooled.mean()),
           "by_diagnosis": byc.reset_index().to_dict("records"), "audit": audit,
           "status": "exploratory: layer-wise aggregates on these held-out queries (same-label references at K = 4, audit Section 8, and runs/latent_afsl) were read earlier in this project, before this analysis was specified"}
    json.dump(res, open(P("holdout_result.json"), "w"), indent=1); print(json.dumps({k: v for k, v in res.items() if k not in ("audit", "by_diagnosis")}, indent=1)); print(byc)

elif STAGE == "sweep":
    assert os.path.exists(P("config_locked.json")), "lock the configuration first"; AG, AU = [], []
    for i, u in enumerate(audit_units()):
        A, audit = evaluate(u, f"sweep_{i:02d}"); AG.append(A); AU.append(audit); print("swept", u["name"], flush=True)
    pd.concat(AG, ignore_index=True).to_csv(P("aggregate_sweep_all_datasets.csv"), index=False); json.dump(AU, open(P("audit_sweep.json"), "w"), indent=1)
    [os.remove(P(f)) for f in os.listdir(OUT) if f.startswith("aggregate_sweep_") and f != "aggregate_sweep_all_datasets.csv"]

elif STAGE == "report":
    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.family": "Helvetica", "font.size": 7, "axes.spines.top": False, "axes.spines.right": False, "axes.linewidth": 0.6})
    cfg = json.load(open(P("config_locked.json"))); H = json.load(open(P("holdout_result.json"))); D = pd.read_csv(P("aggregate_development.csv")); HO = pd.read_csv(P("aggregate_holdout.csv"))
    SW = pd.read_csv(P("aggregate_sweep_all_datasets.csv")) if os.path.exists(P("aggregate_sweep_all_datasets.csv")) else None
    COL = {"vit": "#0072B2", "dinov2": "#D55E00", "clip": "#009E73", "resnet": "#CC79A7", "siglip2": "#E69F00", "bioclip": "#56B4E9", "dinov3": "#000000", "qwen": "#8C5A2B", "flava": "#7F7F7F"}
    MK = {"vit": "o", "dinov2": "s", "clip": "^", "resnet": "D", "siglip2": "v", "bioclip": "P", "dinov3": "X", "qwen": "h", "flava": "*"}
    NM = {"vit": "ViT", "dinov2": "DINOv2", "clip": "CLIP", "resnet": "ResNet-50", "siglip2": "SigLIP 2", "bioclip": "BioCLIP", "dinov3": "DINOv3", "qwen": "Qwen3-VL tower", "flava": "FLAVA"}
    PRIM = {e: LA.LAYERSPEC[e][0][0] for e in LA.LAYERSPEC}; encs = [e for e in ENC_ORDER if e in set(D.encoder)]
    B = D[D.block.notna() & (D.representation != BASE)].copy(); B["block"] = B.block.astype(int)
    def series(e, k, col):
        d = B[(B.encoder == e) & (B.readout == PRIM[e]) & (B.K == k)].sort_values("block"); return np.linspace(0, 1, len(d)), d[col].values
    cand, K = cfg["candidate"]["representation"], cfg["candidate"]["K"]; ce = cfg["candidate"]["encoder"]
    # ---- figure 1: development, adjacent-block changes and replacement relative to the pooled list
    fig, ax = plt.subplots(1, 3, figsize=(7.2, 2.7), gridspec_kw={"width_ratios": [1.15, 1.15, 1]})
    for e in encs:
        for a, k in zip(ax[:2], (1, 4)):
            x, y = series(e, k, "mean_fraction_replaced_vs_previous_block"); a.plot(x[1:], 100 * y[1:], color=COL[e], marker=MK[e], ms=2.6, lw=1.0, mec="white", mew=0.3, label=NM[e])
    for a, k, t in zip(ax[:2], (1, 4), "ab"):
        a.set_title(f"K = {k}", fontsize=7.5); a.set_xticks([0, 0.5, 1], ["First block", "Middle", "Last block"]); a.set_ylabel("Replaced vs. previous block (%)"); a.text(-0.24, 1.06, t, transform=a.transAxes, fontsize=10, fontweight="bold")
    ramp = plt.get_cmap("Blues")
    for i, k in enumerate(KS):
        d = B[B.K == k]; left = 0
        for j in range(k + 1):
            v = d[f"pct_replaced_{j}_of_K"].mean(); ax[2].barh(i, v, left=left, color=ramp(0.15 + 0.8 * j / max(k, 1)), edgecolor="white", lw=0.4)
            if v > 9: ax[2].text(left + v / 2, i, f"{j}", ha="center", va="center", fontsize=6, color="white" if j / max(k, 1) > 0.5 else "#111")
            left += v
    ax[2].set_yticks(range(4), [f"K = {k}" for k in KS]); ax[2].invert_yaxis(); ax[2].set_xlabel("Queries (%)"); ax[2].text(-0.2, 1.06, "c", transform=ax[2].transAxes, fontsize=10, fontweight="bold"); ax[2].set_xlim(0, 100)
    fig.legend(*ax[0].get_legend_handles_labels(), frameon=False, ncol=5, fontsize=6.5, loc="upper center", bbox_to_anchor=(0.5, 1.0)); fig.tight_layout(rect=(0, 0, 1, 0.87)); fig.savefig(P("fig_development_changes.png"), dpi=600); plt.close(fig)
    # ---- figure 2: coverage added and lost by block and K (development), held-out panel for the locked candidate
    fig, ax = plt.subplots(1, 5, figsize=(7.2, 2.7), gridspec_kw={"width_ratios": [1, 1, 1, 1, 1.75]})
    for a, k, t in zip(ax[:4], KS, "abcd"):
        for e in encs:
            x, y = series(e, k, "pct_added"); a.plot(x, y, color=COL[e], lw=1.0, marker=MK[e], ms=2.2, mec="white", mew=0.3, label=NM[e]); x, y = series(e, k, "pct_lost"); a.plot(x, -y, color=COL[e], lw=0.9, ls=(0, (2, 1.5)))
        if k == K:
            r = D[(D.representation == cand) & (D.K == K)].iloc[0]; xs, _ = series(ce, K, "pct_added"); xb = xs[cfg["candidate"]["block"] - (0 if ce == "resnet" else 1)]; a.scatter([xb, xb], [r.pct_added, -r.pct_lost], s=42, facecolors="none", edgecolors="#111", lw=0.9, zorder=5)
        a.axhline(0, color="#444", lw=0.6); a.set_title(f"K = {k}", fontsize=7.5); a.set_xticks([0, 1], ["First", "Last"]); a.text(-0.25, 1.06, t, transform=a.transAxes, fontsize=10, fontweight="bold")
    ax[0].set_ylabel("Queries (%): added (solid, up), lost (dashed, down)")
    a = ax[4]; a.axis("off"); a.text(-0.05, 1.06, "e", transform=a.transAxes, fontsize=10, fontweight="bold")
    lines = [("Held-out queries, locked candidate", True), (f"{NM[ce]}, {cfg['candidate']['readout']} token, block {cfg['candidate']['block']}, K = {K}", False), ("", False), (f"Paired queries: {H['n_paired']}", False), (f"Added an example: {H['added']}", False), (f"Lost all examples: {H['lost']}", False),
             (f"Both: {H['both']}   Neither: {H['neither']}", False), (f"Coverage: {H['coverage_pooled_pct']:.1f}% to {H['coverage_block_pct']:.1f}%", False), (f"Net change: {H['net_coverage_pp']:+.1f} pp", True),
             (f"95% CI {H['ci95_date_cluster_bootstrap_pp'][0]:+.1f} to {H['ci95_date_cluster_bootstrap_pp'][1]:+.1f} pp", False), (f"({H['independent_date_groups']} capture dates, cluster bootstrap)", False)]
    for i, (s_, bold) in enumerate(lines): a.text(0, 0.95 - i * 0.085, s_, transform=a.transAxes, fontsize=6.2, fontweight="bold" if bold else "normal", va="top")
    fig.legend(*ax[0].get_legend_handles_labels(), frameon=False, ncol=5, fontsize=6.5, loc="upper center", bbox_to_anchor=(0.42, 1.0)); fig.tight_layout(rect=(0, 0, 1, 0.86)); fig.savefig(P("fig_coverage.png"), dpi=600); plt.close(fig)
    # ---- checks
    chk = {}
    for nm_, A in (("development", D), ("holdout", HO)) + ((("all-dataset sweep", SW),) if SW is not None else ()):
        chk[nm_] = {"transition_counts_sum_to_n": bool(((A.n_added + A.n_lost + A.n_both + A.n_neither) == A.n_paired).all()), "rates_reproduce_net_coverage": bool(np.allclose(A.pct_added - A.pct_lost, A.net_coverage_pp)),
                    "net_equals_coverage_difference": bool(np.allclose(A.coverage_block_pct - A.coverage_pooled_pct, A.net_coverage_pp)), "rows": int(len(A))}
    for nm_, f in (("development", "per_query_development.csv.gz"), ("holdout", "per_query_holdout.csv.gz")):
        q = pd.read_csv(P(f), usecols=["identical_order_vs_pooled", "rbo_vs_pooled", "same_dx_pooled", "same_dx_block", "ndcg_pooled", "ndcg_block", "replaced_vs_pooled"]); i = q[q.identical_order_vs_pooled]
        chk[nm_].update(exact_ordered_matches=int(len(i)), exact_matches_have_rbo_1=bool((i.rbo_vs_pooled == 1).all()), exact_matches_have_equal_same_dx=bool((i.same_dx_pooled == i.same_dx_block).all()),
                        exact_matches_have_equal_ndcg=bool(((i.ndcg_pooled == i.ndcg_block) | (i.ndcg_pooled.isna() & i.ndcg_block.isna())).all()), exact_matches_have_zero_replaced=bool((i.replaced_vs_pooled == 0).all()))
    base = D[D.representation == BASE]; chk["pooled_against_itself"] = {"net_coverage_all_zero": bool((base.net_coverage_pp == 0).all()), "identical_order_100pct": bool((base.pct_identical_order_vs_pooled == 100).all())}
    json.dump(chk, open(P("checks.json"), "w"), indent=1)
    # ---- summary
    dv = cfg["development"]; AD = json.load(open(P("audit_development.json")))["development"]; AH = H["audit"]; sel = D[(D.representation == cand) & (D.K == K)].iloc[0]; hsel = HO[(HO.representation == cand) & (HO.K == K)].iloc[0]
    ST = pd.read_csv(P("stabilization_development.csv.gz")); fam = f"{ce}|{cfg['candidate']['readout']}"; st = ST[(ST.family == fam) & (ST.K == K)]
    L = []; w = L.append
    w("# Blockwise retrieval against the pooled baseline: is a VLM follow-up justified?\n")
    w("Local analysis, zero VLM calls. Baseline: the pooled ViT embedding of Arshad et al. (`google/vit-base-patch16-224-in21k`, `pooler_output`), reproduced with the repository pipeline. Candidates: every block readout of nine encoders at K = 1, 2, 4 and 8.\n")
    w("## Summary for review\n")
    ci = H["ci95_date_cluster_bootstrap_pp"]; credible = cfg["credible_candidate"]; supported = ci[0] > 0
    w(f"- **Locked candidate:** {NM[ce]} (`{cfg['candidate']['checkpoint']}`), {cfg['candidate']['readout']} token of block {cfg['candidate']['block']}, with **K = {K}** demonstrations. It was selected on the DeepWeeds development split by a rule fixed in advance (`config_locked.json`).")
    w(f"- **Development ({dv['selected_n_paired']} paired query-bank rows, {AD['queries']} queries, {AD['banks']} banks):** the candidate adds a same-diagnosis example where the pooled list has none in {dv['selected_added']} rows and loses all such examples in {dv['selected_lost']}; net {dv['selected_net_coverage_pp']:+.1f} pp (95% CI {dv['selected_cluster_bootstrap_95ci_pp'][0]:+.1f} to {dv['selected_cluster_bootstrap_95ci_pp'][1]:+.1f}, {dv['bootstrap_groups']} capture dates).")
    w(f"- **Held-out ({H['n_paired']} queries on {H['independent_date_groups']} capture dates, frozen date-disjoint bank of 99 images):** the block added a same-diagnosis example on **{H['added']}** queries and lost all such examples on **{H['lost']}**. "
      f"Coverage rises from {H['coverage_pooled_pct']:.1f}% to {H['coverage_block_pct']:.1f}%: **net {H['net_coverage_pp']:+.1f} pp** (95% CI {ci[0]:+.1f} to {ci[1]:+.1f}, cluster bootstrap over capture dates, {NBOOT} resamples, seed {SEED}; exact McNemar p {'< 0.001' if H['exact_mcnemar_p'] < 0.001 else '= ' + format(H['exact_mcnemar_p'], '.3f')}). "
      f"Resampling queries instead of dates gives {H['ci95_query_bootstrap_pp_sensitivity'][0]:+.1f} to {H['ci95_query_bootstrap_pp_sensitivity'][1]:+.1f}.")
    lastb = max(LA.LAYERSPEC[ce][1]); vb = B[B.encoder == "vit"]; vbest = vb.loc[vb.net_coverage_pp.idxmax()]
    w(f"- **What kind of candidate this is:** " + (f"block {cfg['candidate']['block']} is the last block of {NM[ce]}, and for this encoder it equals the output the model exposes. The result is therefore about the choice of encoder, not about an intermediate layer. " if cfg["candidate"]["block"] == lastb else f"block {cfg['candidate']['block']} of {lastb} is an intermediate layer of {NM[ce]}. ")
      + f"Intermediate layers of the baseline's own ViT gain little under date separation: the best ViT block readout on development is {vbest.representation} at K = {int(vbest.K)} with {vbest.net_coverage_pp:+.1f} pp.")
    w(f"- **What the VLM would receive:** {100 * H['mean_fraction_replaced_vs_pooled']:.0f}% of the pooled references are replaced at K = {K}, and the ordered list is identical for {H['pct_identical_order_vs_pooled']:.0f}% of the held-out queries. Mean same-diagnosis references change by {H['mean_same_dx_change']:+.2f} and nDCG@{K} by {H['mean_ndcg_change']:+.3f}.")
    byc = pd.DataFrame(H["by_diagnosis"]); loss = byc[byc.lost > byc.added]
    w("- **Class-specific losses:** " + ("none; no diagnosis has more queries losing all examples than gaining one." if not len(loss) else "; ".join(f"{r.query_label} (added {r.added}, lost {r.lost})" for r in loss.itertuples()) + ".") + f" Each class has about {int(round(H['n_paired'] / len(byc)))} held-out queries, so per-class counts are small.")
    w(f"- **Status of the held-out result:** {H['status']}.")
    w("- **Split limitations:** one dataset (DeepWeeds) has capture-date metadata, so the confirmation covers one dataset. The development split is the held-out split with the roles reversed (development queries come from the dates of the held-out bank), so the two share capture dates and are not independent draws. "
      f"The bootstrap rests on {H['independent_date_groups']} date groups, which is few for a reliable interval; the raw paired counts are given for that reason.\n")
    w("## Selection on development data\n")
    w(f"Rule: {cfg['selection_rule']}. Best development net coverage change of any candidate: {dv['best_net_coverage_pp_any_candidate']:+.1f} pp; {dv['candidates_within_tolerance']} candidates lie within the tolerance, at K in {dv['K_values_within_tolerance']}. "
      f"After taking the smallest K and the earliest block per encoder readout, these remained (ordered by the tie rule): " + "; ".join(f"{r['representation']} ({r['net_coverage_pp']:+.1f} pp)" for r in dv["remaining_after_K_and_earliest_block"][:8]) + ". The first is the locked candidate. "
      + ("The development interval excludes zero, so the candidate counts as credible." if credible else "The development interval includes zero, so the development data do not show a credible candidate; the held-out evaluation is reported for completeness.") + "\n")
    w("**Best block per encoder on development, by K** (net coverage change in pp; pooled coverage in the last row):\n")
    w("| Encoder (readout) | " + " | ".join(f"K = {k}" for k in KS) + " |"); w("|---|" + "---|" * len(KS))
    for e in encs:
        for ro in LA.LAYERSPEC[e][0]:
            cells = []
            for k in KS:
                d = B[(B.encoder == e) & (B.readout == ro) & (B.K == k)]; r = d.loc[d.net_coverage_pp.idxmax()]; cells.append(f"{r.net_coverage_pp:+.1f} (block {int(r.block)})")
            w(f"| {NM[e]} ({ro}) | " + " | ".join(cells) + " |")
    w("| Pooled coverage (%) | " + " | ".join(f"{D[(D.representation == BASE) & (D.K == k)].coverage_pooled_pct.iloc[0]:.1f}" for k in KS) + " |\n")
    w("## How the retrieved examples change (development)\n")
    w(f"![Development changes](fig_development_changes.png)\n\n**Fig. 1 | Changes of the retrieved references on the development split.** **a**, **b**, Share of the top-K references that a block replaces relative to the previous block, for K = 1 and K = 4 (primary readout of each encoder). **c**, Share of queries by the number of pooled top-K references that a block replaces (numbers inside the bars), averaged over all block readouts.\n")
    w("| K | Pooled references replaced by the candidate block series (mean, %) | Identical ordered list (%) | Same references, other order (%) | Rank-biased overlap |"); w("|---|---|---|---|---|")
    for k in KS:
        r = D[(D.representation == cand) & (D.K == k)].iloc[0]; w(f"| {k} | {100 * r.mean_fraction_replaced_vs_pooled:.0f} | {r.pct_identical_order_vs_pooled:.1f} | {r.pct_same_set_reordered_vs_pooled:.1f} | {r.mean_rbo_vs_pooled:.2f} |")
    w(f"\nThe table is for the locked block against the pooled list. Rank-biased overlap is the extrapolated form for two equal-length lists at depth K with persistence p = {RBO_P} (Webber, Moffat and Zobel 2010); it is a secondary, rank-sensitive measure.\n")
    last = st.final_block.iloc[0]; dist = st.first_stable_block.value_counts(normalize=True).sort_index()
    w(f"**Stabilization of the candidate's block series ({NM[ce]}, {cfg['candidate']['readout']}, K = {K}).** The ordered top-K list is final only at the last block for {100 * dist.get(last, 0):.0f}% of the query-bank rows, and it is already final by the middle block ({last // 2}) for {100 * dist[dist.index <= last // 2].sum():.0f}%. "
      "So the lists keep changing up to the last blocks and there is no plateau after which they stop. `stabilization_development.csv.gz` has every query, readout and K.\n")
    w("## Coverage added and lost\n")
    w("![Coverage](fig_coverage.png)\n\n**Fig. 2 | Same-diagnosis coverage relative to the pooled list.** **a** to **d**, Development split: share of paired rows in which a block supplies a same-diagnosis example where the pooled list has none (solid, upward) and in which it loses all such examples (dashed, downward), by depth, for K = 1, 2, 4 and 8. The circle marks the locked candidate. **e**, Held-out result of the locked candidate.\n")
    w("**Locked candidate, all K, development and held-out** (same K for pooled and block):\n")
    w("| Split | K | Paired n | Added | Lost | Both | Neither | Net (pp) | Pooled misses recovered | Mean same-diagnosis change | nDCG change |"); w("|---|---|---|---|---|---|---|---|---|---|---|")
    for nm_, A in (("Development", D), ("Held-out", HO)):
        for k in KS:
            r = A[(A.representation == cand) & (A.K == k)].iloc[0]
            w(f"| {nm_} | {k}{' (locked)' if k == K else ''} | {int(r.n_paired)} | {int(r.n_added)} ({r.pct_added:.1f}%) | {int(r.n_lost)} ({r.pct_lost:.1f}%) | {int(r.n_both)} | {int(r.n_neither)} | {r.net_coverage_pp:+.1f} | {int(r.n_added)} of {int(r.pooled_misses)} | {r.mean_same_dx_change:+.2f} | {r.mean_ndcg_change:+.3f} |")
    w("\nOnly the locked K on the held-out split is the planned comparison; the other held-out rows are descriptive.\n")
    w(f"**Held-out result by diagnosis (K = {K}):**\n"); w("| Diagnosis | Added | Lost | Both | Neither |"); w("|---|---|---|---|---|")
    for r in byc.itertuples(): w(f"| {r.query_label} | {r.added} | {r.lost} | {r.both} | {r.neither} |")
    if SW is not None:
        w("\n## Exploratory sweep over all datasets\n")
        w(f"The locked candidate against the pooled list on every labeled split of the layer audit (leave-one-out pools unless stated). These splits played no part in the selection, and the pools have no source-group separation, so they are descriptive.\n")
        w("| Split | Paired n | " + " | ".join(f"Net pp, K = {k}" for k in KS) + " |"); w("|---|---|" + "---|" * len(KS))
        for un, g in SW[SW.representation == cand].groupby("unit", sort=False): w(f"| {un.replace(' | loader pool', '')} | {int(g.n_paired.iloc[0])} | " + " | ".join(f"{g[g.K == k].net_coverage_pp.iloc[0]:+.1f}" for k in KS) + " |")
        w("\n`aggregate_sweep_all_datasets.csv` holds every encoder, block, K and split.\n")
    w("## Audit of the inputs\n")
    w(f"- **Pooled baseline:** the real `pooler_output` vectors, cached by the layer audit and checked there against the repository's earlier cache (identical ranked top four on 777 of 777 queries outside one stale Durum Wheat cache). The final block is never used in its place.")
    w(f"- **Same queries, bank and eligibility for pooled and blockwise lists:** yes; both are ranked on the same query set and bank with the same exclusions and the same tie-break (reference id).")
    for nm_, a in (("Development", AD), ("Held-out", AH)):
        w(f"- **{nm_}:** {a['queries']} queries, {a['banks']} bank(s) of {a['bank_sizes']} images, {a['paired_query_bank_rows']} paired rows; encoders {', '.join(NM[e] for e in a['encoders_evaluated'])}" + (f"; without latents: {', '.join(a['encoders_without_latents'])}" if a["encoders_without_latents"] else "") +
          f". Query ids inside a bank: {a['query_ids_in_a_bank']}; byte-identical query-reference pairs excluded: {a['byte_identical_pairs_excluded']}; queries with fewer than 8 eligible references: {a['queries_with_fewer_than_8_eligible_references']}; "
          f"rows without any same-diagnosis reference in the bank: {a['query_bank_pairs_without_a_same_diagnosis_reference']}; missing labels: {a['missing_labels']}; capture dates shared by queries and their bank: {len(a['query_dates_shared_with_bank_dates']) if isinstance(a['query_dates_shared_with_bank_dates'], list) else a['query_dates_shared_with_bank_dates']}.")
    w("- **nDCG:** binary relevance; the ideal list holds min(K, same-diagnosis references in the eligible bank); a query whose bank holds none would be undefined and excluded (count above).")
    w("- **Not measured:** robustness to changes of the input image. No repeated queries or controlled perturbations exist in these splits; agreement between adjacent blocks is evidence of consistency across layers only.\n")
    w("## Request for API credits\n")
    n_calls = 2 * H["n_paired"]
    if credible and supported:
        w(f"We ask for credits for one paired comparison on the {H['n_paired']} held-out DeepWeeds queries with the frozen date-disjoint bank: the pooled ViT retrieval of Arshad et al. against {NM[ce]} block {cfg['candidate']['block']} ({cfg['candidate']['readout']} token), both with K = {K} demonstrations and the standard assisted prompt. "
          f"With one call per query and condition this is {n_calls} calls. " + ("Earlier runs on these queries used four demonstrations, so no compatible output is cached and nothing can be reused. " if K != 4 else "Outputs of the pooled condition that are cached for the same queries, bank, prompt and K can be reused and would halve the request. ") +
          f"The retrieval evidence is that the block supplies a same-diagnosis example on {H['added']} held-out queries where the pooled list supplies none and loses all such examples on {H['lost']} (net {H['net_coverage_pp']:+.1f} pp, 95% CI {ci[0]:+.1f} to {ci[1]:+.1f}). "
          "This is a statement about the examples the VLM receives. Whether those examples improve its diagnoses is the question the pilot would test.\n")
    else:
        w(f"The retrieval evidence does not support a request yet. " + ("The development data show no credible candidate. " if not credible else "") + (f"On the held-out queries the interval for the net coverage change ({ci[0]:+.1f} to {ci[1]:+.1f} pp) includes zero. " if not supported else "")
          + f"If a pilot is still wanted, the smallest informative design is the paired comparison of pooled retrieval and the locked block at K = {K} on the {H['n_paired']} held-out queries ({n_calls} calls at one call per query and condition), stated as a test of whether the changed examples matter and not as a confirmation.\n")
    w("## Reproduce\n")
    w("```\ncd code/ageval-dinov2-pilot/repo\npython blockwise_vs_pooled.py ../runs/blockwise_vs_pooled features   # encode the development split (cached, resumable)\npython blockwise_vs_pooled.py ../runs/blockwise_vs_pooled dev\npython blockwise_vs_pooled.py ../runs/blockwise_vs_pooled select     # locks config_locked.json\npython blockwise_vs_pooled.py ../runs/blockwise_vs_pooled holdout\npython blockwise_vs_pooled.py ../runs/blockwise_vs_pooled sweep\npython blockwise_vs_pooled.py ../runs/blockwise_vs_pooled report\n```\n")
    w(f"Seed {SEED}; {NBOOT} bootstrap resamples. Inputs are listed in `input_manifest.json`. Files: `per_query_*.csv.gz` (one row per query, bank, representation and K), `aggregate_*.csv` (denominators included), `stabilization_*.csv.gz`, `config_locked.json`, `holdout_result.json`, `audit_*.json`, `checks.json`, two figures.\n")
    w("References: Webber, Moffat and Zobel (2010), doi:10.1145/1852102.1852106; Järvelin and Kekäläinen (2002), doi:10.1145/582415.582418; Smucker, Allan and Carterette (2007), doi:10.1145/1321440.1321528.\n")
    open(P("SUMMARY.md"), "w").write("\n".join(L))
    man = {"script": "repo/blockwise_vs_pooled.py", "latents": sorted(os.listdir(os.path.join(LA.out, "features"))), "split_files": ["runs/latent_afsl/dev_queries.csv", "runs/latent_afsl/dev_banks.csv", "runs/deepweeds_split_audit/query_panel.csv", "runs/deepweeds_matched_seed42/reference_manifest.csv", "runs/layer_retrieval_audit/coverage_manifest.json"],
           "checkpoints": LA.CKPT, "K": list(KS), "ranking_depth": DEPTH, "seed": SEED}
    json.dump(man, open(P("input_manifest.json"), "w"), indent=1); print(json.dumps(chk, indent=1)); print("\n".join(L)[:3000])
