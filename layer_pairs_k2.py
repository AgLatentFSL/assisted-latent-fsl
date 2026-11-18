"""Layer-versus-layer comparison of the saved Claude predictions of the K = 2 downstream test (appendix table of all conditions).

Reads only saved files of runs/layer_vlm_k2 and the atlas index. Sends no request.

    python layer_pairs_k2.py ../runs/layer_vlm_k2
"""
import os, sys, json, itertools, glob
import numpy as np, pandas as pd
from scipy import stats

RUN = sys.argv[1]; ATLAS = os.path.join(os.path.dirname(os.path.abspath(__file__)), "atlas")
OUT = os.path.join(RUN, "layer_pairs"); os.makedirs(OUT, exist_ok=True)
SEED, NB, ALPHA, K = 20261115, 10000, 0.05, 2
checks = {}
def md(df):
    df = df.astype(object).where(df.notna(), ''); c = [str(x) for x in df.columns]
    return '\n'.join(['| ' + ' | '.join(c) + ' |', '|' + '---|' * len(c)] + ['| ' + ' | '.join(str(v) for v in r) + ' |' for r in df.values])

pd.DataFrame.to_markdown = lambda self, index=False: md(self)

# ---------------------------------------------------------------- 1. paired data
PQ = pd.read_csv(os.path.join(RUN, "per_query_predictions.csv"), dtype=str, keep_default_na=False)
Cn = pd.read_csv(os.path.join(RUN, "conditions.csv"), dtype=str, keep_default_na=False)
Cn = Cn[Cn.encoder != ""]; ENC = list(dict.fromkeys(Cn.encoder))
PQ = PQ[PQ.condition.isin(Cn.condition)]
key = ["dataset", "query_folder"]
checks["conditions"] = int(Cn.condition.nunique()); checks["rows"] = len(PQ)
checks["duplicate_condition_query_rows"] = int(PQ.duplicated(["condition"] + key).sum())
checks["queries_per_condition"] = sorted(set(PQ.groupby("condition").size()))
qsets = pd.Series({c: frozenset(zip(g.dataset, g.query_folder)) for c, g in PQ.groupby("condition")})
checks["all_conditions_share_the_same_queries"] = bool(len(set(qsets)) == 1)
checks["ground_truth_labels_per_query"] = int(PQ.groupby(key).true_label.nunique().max())
checks["queries_per_dataset"] = PQ[PQ.condition == Cn.condition.iloc[0]].groupby("dataset").size().to_dict()
checks["missing_predictions"] = int((PQ.pred == "").sum()); checks["invalid_predictions"] = int((PQ.pred == "INVALID").sum())
labels = PQ.groupby("dataset").true_label.unique().map(set).to_dict()
checks["predictions_outside_the_label_set"] = int(sum(p not in labels[d] for d, p in zip(PQ.dataset, PQ.pred)))
# model and prompt protocol from the stored responses
mods, turns, missing = set(), set(), 0
for k in PQ.payload.unique():
    f = os.path.join(RUN, "responses", k + ".json")
    if not os.path.exists(f): missing += 1; continue
    j = json.load(open(f)); mods.update(j["meta"].get("model_ids", [j.get("model")])); turns.add(j["meta"].get("num_turns"))
checks["distinct_requests"] = int(PQ.payload.nunique()); checks["stored_responses_missing"] = missing; checks["models_in_responses"] = sorted(mods); checks["turns_per_request"] = sorted(turns)
pre = json.load(open(os.path.join(RUN, "preregistration.json"))); checks["preregistered_model"] = pre["model"]; checks["preregistered_K"] = pre["K"]
# ordered demonstrations from the atlas index (rank 1..K)
need = set(zip(Cn.encoder, Cn.layer)); parts = []
for ch in pd.read_csv(os.path.join(ATLAS, "index.csv"), dtype=str, keep_default_na=False, chunksize=200000):
    ch = ch[ch["rank"].astype(int) <= K]; parts.append(ch[[(m, l) in need for m, l in zip(ch.model, ch.layer)]])
IDX = pd.concat(parts); IDX["rank"] = IDX["rank"].astype(int); IDX = IDX.sort_values("rank")
demo = IDX.groupby(["model", "layer"] + key).reference_id.agg(tuple)
checks["references_per_prompt"] = sorted(set(demo.map(len)))
SI = pd.read_csv(os.path.join(ATLAS, "split_images.csv"), dtype=str, keep_default_na=False)
bank = set(zip(SI[SI.role == "reference"].dataset, SI[SI.role == "reference"].image_id)); qs = set(zip(SI[SI.role == "query"].dataset, SI[SI.role == "query"].image_id))
checks["reference_bank_size_per_dataset"] = SI[SI.role == "reference"].groupby("dataset").size().to_dict()
checks["demonstrations_outside_the_reference_bank"] = int(sum((d, r) not in bank for d, r in zip(IDX.dataset, IDX.reference_id)))
checks["demonstrations_that_are_queries"] = int(sum((d, r) in qs for d, r in zip(IDX.dataset, IDX.reference_id)))

PQ["ok"] = PQ.pred == PQ.true_label                      # scoring rule of the original report: exact match of the parsed prediction
W = PQ.pivot(index=key, columns="condition", values="ok").astype(bool); n = len(W); ds = W.index.get_level_values(0).values
PAY = PQ.pivot(index=key, columns="condition", values="payload")
cond = {(r.encoder, r.layer): r.condition for r in Cn.itertuples()}
# same ordered demonstrations for every query: checked on the atlas ranks and on the request hashes
sig = {c: tuple(demo.loc[e, l].reindex(W.index).values) for (e, l), c in cond.items()}
same_demo = lambda a, b: sig[a] == sig[b]
for a, b in itertools.combinations(Cn.condition, 2): assert same_demo(a, b) == bool((PAY[a] == PAY[b]).all()), (a, b)
checks["matched_n_all_comparisons"] = n

# reproduce the published table
PUB = pd.read_csv(os.path.join(RUN, "results_by_condition.csv")).set_index("condition")
disc = []
for c in Cn.condition:
    if abs(100 * W[c].mean() - PUB.loc[c, "accuracy"]) > 1e-9: disc.append((c, "accuracy"))
    b = PUB.loc[c, "own_pooled"]; w_ = int((W[c] & ~W[b]).sum()); l_ = int((~W[c] & W[b]).sum())
    if (w_, l_) != (int(PUB.loc[c, "wins_vs_own_pooled"]), int(PUB.loc[c, "losses_vs_own_pooled"])): disc.append((c, "wins/losses vs pooled"))
checks["discrepancies_with_published_table"] = disc
json.dump(checks, open(os.path.join(OUT, "checks.json"), "w"), indent=1, default=str)

# ---------------------------------------------------------------- 2. all pairs
rng = np.random.default_rng(SEED); Cw = np.zeros((NB, n), dtype=np.float64)
for d in np.unique(ds):                                   # stratified by dataset: each dataset keeps its number of queries in every replicate
    ix = np.where(ds == d)[0]; Cw[:, ix] = rng.multinomial(len(ix), np.ones(len(ix)) / len(ix), size=NB)
def paired(a, b):
    d = W[a].astype(np.int8).values - W[b].astype(np.int8).values; oa, ob = int((d == 1).sum()), int((d == -1).sum()); est = np.einsum('ij,j->i', Cw, d.astype(np.float64)) / n * 100; assert np.isfinite(est).all()
    return {"acc_A": 100 * W[a].mean(), "acc_B": 100 * W[b].mean(), "diff_A_minus_B_pp": 100 * d.mean(), "only_A_correct": oa, "only_B_correct": ob, "n_matched": n,
            "mcnemar_exact_p": 1.0 if oa + ob == 0 else float(stats.binomtest(oa, oa + ob, 0.5).pvalue), "ci95_lo_unadjusted": float(np.percentile(est, 2.5)), "ci95_hi_unadjusted": float(np.percentile(est, 97.5)),
            "queries_with_different_demonstrations": int(sum(x != y for x, y in zip(sig[a], sig[b])))}
num = lambda l: int(l.split("_")[1]) if l.startswith("layer_") else 999
rows = []
for e in ENC:
    L = sorted([l for (m, l) in cond if m == e and l != "final_output"], key=num)
    grp = {}                                               # layers with the same ordered demonstrations for every query
    for l in L + ["final_output"]: grp.setdefault(sig[cond[(e, l)]], []).append(l)
    rep = {l: g[0] for g in grp.values() for l in g}
    for a, b in itertools.combinations(L, 2): rows.append({"family": "layer_vs_layer", "encoder": e, "A": a, "B": b, **paired(cond[(e, a)], cond[(e, b)]), "rep": tuple(sorted((rep[a], rep[b]), key=num))})
    for a in L: rows.append({"family": "layer_vs_pooled", "encoder": e, "A": a, "B": "final_output", **paired(cond[(e, a)], cond[(e, "final_output")]), "rep": tuple(sorted((rep[a], rep["final_output"]), key=num))})
R = pd.DataFrame(rows)
R["identical_demonstrations"] = R.rep.map(lambda t: t[0] == t[1])
R["duplicate_of_earlier_contrast"] = R.duplicated(["family", "encoder", "rep"]) & ~R.identical_demonstrations
R["in_correction"] = ~R.identical_demonstrations & ~R.duplicate_of_earlier_contrast

# ---------------------------------------------------------------- 3. Holm
def holm(p):
    p = np.asarray(p, float); o = np.argsort(p); m = len(p); adj = np.empty(m); run = 0.0
    for i, j in enumerate(o): run = max(run, (m - i) * p[j]); adj[j] = min(1.0, run)
    return adj
for col in ["holm_within_encoder", "holm_global", "tests_within_encoder", "tests_global"]: R[col] = np.nan
for fam, g in R[R.in_correction].groupby("family"):
    R.loc[g.index, "holm_global"] = holm(g.mcnemar_exact_p); R.loc[g.index, "tests_global"] = len(g)
    for e, h in g.groupby("encoder"): R.loc[h.index, "holm_within_encoder"] = holm(h.mcnemar_exact_p); R.loc[h.index, "tests_within_encoder"] = len(h)
for i in R.index[R.duplicate_of_earlier_contrast]:        # a duplicated contrast carries the adjusted values of the contrast it repeats
    src = R[(R.family == R.family[i]) & (R.encoder == R.encoder[i]) & (R.rep == R.rep[i]) & R.in_correction].iloc[0]
    for col in ["holm_within_encoder", "holm_global", "tests_within_encoder", "tests_global"]: R.loc[i, col] = src[col]
R["sig_within_encoder"] = R.holm_within_encoder < ALPHA; R["sig_global"] = R.holm_global < ALPHA
R.drop(columns="rep").to_csv(os.path.join(OUT, "all_comparisons.csv"), index=False)

# ---------------------------------------------------------------- 4. summary
LL = R[R.family == "layer_vs_layer"]; LP = R[R.family == "layer_vs_pooled"]; S = []
def get(e, a, b):
    r = LL[(LL.encoder == e) & (((LL.A == a) & (LL.B == b)) | ((LL.A == b) & (LL.B == a)))].iloc[0]; s = 1 if r.A == a else -1
    return {"diff": s * r.diff_A_minus_B_pp, "only_a": r.only_A_correct if s == 1 else r.only_B_correct, "only_b": r.only_B_correct if s == 1 else r.only_A_correct, "lo": s * r.ci95_lo_unadjusted if s == 1 else -r.ci95_hi_unadjusted,
            "hi": r.ci95_hi_unadjusted if s == 1 else -r.ci95_lo_unadjusted, "p": r.mcnemar_exact_p, "hw": r.holm_within_encoder, "hg": r.holm_global, "ident": r.identical_demonstrations}
for e in ENC:
    L = sorted([l for (m, l) in cond if m == e and l != "final_output"], key=num); acc = {l: 100 * W[cond[(e, l)]].mean() for l in L}
    order = sorted(L, key=lambda l: (-acc[l], num(l))); lead, second = order[0], order[1]; g = get(e, lead, second)
    beats_w = [l for l in order[1:] if get(e, lead, l)["hw"] < ALPHA and get(e, lead, l)["diff"] > 0]; beats_g = [l for l in order[1:] if get(e, lead, l)["hg"] < ALPHA and get(e, lead, l)["diff"] > 0]
    raw = [l for l in order[1:] if get(e, lead, l)["p"] < ALPHA]
    h = LL[(LL.encoder == e) & LL.in_correction]
    S.append({"encoder": e, "layers_tested": " ".join(str(num(l)) for l in L), "pooled_accuracy": 100 * W[cond[(e, "final_output")]].mean(), "highest_observed_layer": num(lead), "accuracy": acc[lead], "tied_at_top": sum(acc[l] == acc[lead] for l in L) > 1,
              "runner_up": num(second), "runner_up_accuracy": acc[second], "gap_pp": g["diff"], "gap_ci95_lo_unadjusted": g["lo"], "gap_ci95_hi_unadjusted": g["hi"], "gap_only_leader": g["only_a"], "gap_only_runner_up": g["only_b"], "gap_p": g["p"], "gap_holm_within": g["hw"],
              "lowest_layer_accuracy": acc[order[-1]], "range_pp": acc[lead] - acc[order[-1]], "leader_beats_raw_p": " ".join(str(num(l)) for l in raw) or "none", "leader_beats_after_holm_within_encoder": " ".join(str(num(l)) for l in beats_w) or "none",
              "leader_beats_after_global_holm": " ".join(str(num(l)) for l in beats_g) or "none", "distinct_layer_pairs_tested": len(h), "smallest_raw_p_any_pair": h.mcnemar_exact_p.min(), "smallest_holm_within_any_pair": h.holm_within_encoder.min(), "leader_ahead_of_all_tested_layers": len(beats_w) == len(L) - 1})
S = pd.DataFrame(S); S.to_csv(os.path.join(OUT, "encoder_summary.csv"), index=False)

f = lambda x: f"{x:.1f}"; pf = lambda p: f"{p:.3f}" if p >= 0.001 else f"{p:.1e}"
out = ["# Layer against layer within each encoder (K = 2, 500 queries, saved Claude predictions)", "", "Exploratory analysis of the existing evaluation set. No request was sent.", "", "## Checks", ""]
out += [f"- {k}: {v}" for k, v in checks.items()]
ident = R[R.identical_demonstrations][["family", "encoder", "A", "B"]]; dup = R[R.duplicate_of_earlier_contrast][["family", "encoder", "A", "B"]]
out += ["", "## Conditions with the same ordered demonstrations for every query", "", ident.to_markdown(index=False) if len(ident) else "none", "", "Contrasts that repeat another contrast because of this (counted once):", "", dup.to_markdown(index=False) if len(dup) else "none", ""]
nl = LL[LL.in_correction]; npool = LP[LP.in_correction]
out += ["## Tests in each correction", "", f"- layer against layer, within encoder: {nl.groupby('encoder').size().to_dict()}", f"- layer against layer, global: {len(nl)}", f"- layer against pooled output, within encoder: {npool.groupby('encoder').size().to_dict()}", f"- layer against pooled output, global: {len(npool)}",
        "", f"Layer pairs with raw p < 0.05: {int((nl.mcnemar_exact_p < ALPHA).sum())} of {len(nl)}; after Holm within encoder: {int(nl.sig_within_encoder.sum())}; after global Holm: {int(nl.sig_global.sum())}.",
        f"Layer against pooled with raw p < 0.05: {int((npool.mcnemar_exact_p < ALPHA).sum())} of {len(npool)}; after Holm within encoder: {int(npool.sig_within_encoder.sum())}; after global Holm: {int(npool.sig_global.sum())}.", "", "## Summary per encoder", "", S.round(4).to_markdown(index=False), ""]
show = ["encoder", "A", "B", "acc_A", "acc_B", "diff_A_minus_B_pp", "ci95_lo_unadjusted", "ci95_hi_unadjusted", "only_A_correct", "only_B_correct", "mcnemar_exact_p", "holm_within_encoder", "holm_global", "queries_with_different_demonstrations"]
out += ["## Layer pairs with raw p < 0.05", "", LL[LL.mcnemar_exact_p < ALPHA][show].round(4).to_markdown(index=False), "", "## Named contrasts", "", pd.concat([LL[(LL.encoder == "FLAVA") & LL.A.isin(["layer_09", "layer_10"]) & LL.B.isin(["layer_09", "layer_10"])],
        LL[(LL.encoder == "Qwen3VL") & ((LL.A == "layer_16") | (LL.B == "layer_16"))]])[show].round(4).to_markdown(index=False), "", "## Layer against pooled output", "", LP[show].round(4).to_markdown(index=False), ""]
open(os.path.join(OUT, "SUMMARY.md"), "w").write("\n".join(out)); print("\n".join(out))
