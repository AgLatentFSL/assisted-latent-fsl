"""Early, middle and late blocks within each encoder at K = 2: retrieval (saved atlas) and Claude accuracy (saved predictions).

All nine encoders, all five datasets, 500 queries. Reads saved files only. Sends no request.

    python depth_groups_k2.py ../runs/layer_vlm_k2 [../runs/layer_vlm_allk]

The optional second run adds K = 2 conditions of the all-budget run, but only those with all 500 queries answered.
"""
import os, sys, json, re, itertools, time
import numpy as np, pandas as pd

RUN = sys.argv[1]; EXTRA = sys.argv[2] if len(sys.argv) > 2 else None
ATLAS = os.path.join(os.path.dirname(os.path.abspath(__file__)), "atlas")
OUT = os.path.join(RUN, "depth_groups"); os.makedirs(OUT, exist_ok=True)
SEED, NB, NPERM, ALPHA, K = 20261115, 10000, 100000, 0.05, 2
ENC = ["ViT", "DINOv2", "CLIP", "ResNet50", "SigLIP2", "BioCLIP", "DINOv3", "Qwen3VL", "FLAVA"]
num = lambda l: int(l.split("_")[1]) if l.startswith("layer_") else 999
def md(df):
    df = df.astype(object).where(df.notna(), ""); c = [str(x) for x in df.columns]
    return "\n".join(["| " + " | ".join(c) + " |", "|" + "---|" * len(c)] + ["| " + " | ".join(f"{v:.4g}" if isinstance(v, float) else str(v) for v in r) + " |" for r in df.values])
def holm(p):
    p = np.asarray(p, float); o = np.argsort(p); m = len(p); adj = np.empty(m); run = 0.0
    for i, j in enumerate(o): run = max(run, (m - i) * p[j]); adj[j] = min(1.0, run)
    return adj

# ---------------------------------------------------------------- groups (depth only)
GROUPS = {e: {"early": range(1, 5), "middle": range(5, 9), "late": range(9, 13)} for e in ENC}
GROUPS["Qwen3VL"] = {"early": range(1, 9), "middle": range(9, 17), "late": range(17, 25)}
GROUPS["ResNet50"] = {"early": range(1, 8), "middle": range(8, 14), "late": range(14, 17)}
RULE = {e: "12 blocks, equal thirds: atlas layers 1-4, 5-8, 9-12 (zero-indexed blocks 0-3, 4-7, 8-11)" for e in ENC}
RULE["Qwen3VL"] = "24 blocks, equal thirds: atlas layers 1-8, 9-16, 17-24 (zero-indexed blocks 0-7, 8-15, 16-23)"
RULE["ResNet50"] = "16 bottleneck blocks in four stages (3, 4, 6, 3); equal thirds do not exist: early = stages 1 and 2 (layers 1-7), middle = stage 3 (layers 8-13), late = stage 4 (layers 14-16); the stem (layer 0) is not a residual block and is left out"
manifest = {"written_at": time.strftime("%Y-%m-%d %H:%M:%S"), "numbering": "atlas layer_b is the output of the b-th block counted from 1 (forward hooks on the blocks in layer_audit.py, or hidden_states[1:]); the zero-indexed block is b - 1; there is no other offset. final_output is the pooled output and is kept as a separate baseline.",
            "qwen_check": "layer_audit.py registers one hook per block of Qwen3VLVisionModel.blocks and asserts 24 captured outputs: 24 vision blocks, equal thirds apply. Atlas layers 16-20 (the ones with Claude predictions) are zero-indexed blocks 15-19: one middle block (15) and four late blocks (16-19).",
            "rule": RULE, "groups_atlas_layers": {e: {g: list(r) for g, r in GROUPS[e].items()} for e in ENC}, "groups_zero_indexed_blocks": {e: {g: [b - 1 for b in r] for g, r in GROUPS[e].items()} for e in ENC},
            "note": "groups use depth only; the single-layer Claude accuracies were known when this file was written",
            "retrieval_metrics": {"coverage (primary, original protocol)": "the two retrieved references hold at least one reference of the query's class", "same-class share": "share of the two retrieved references with the query's class"},
            "downstream_metric": "Claude correct (exact match of the parsed prediction with the gold label)",
            "statistics": f"per query: mean of the metric over the evaluated blocks of each group, paired difference; 95% bootstrap interval, {NB} replicates, seed {SEED}, queries resampled within dataset; two-sided sign-flip randomization test on the query-level differences, {NPERM} flips; Holm over the three overall contrasts of an encoder within each metric (also reported over all encoders); dataset-level intervals are descriptive",
            "downstream_rule": "a group needs at least one block with saved predictions for all 500 queries; groups with one block are flagged"}
json.dump(manifest, open(os.path.join(OUT, "group_manifest.json"), "w"), indent=1)

# ---------------------------------------------------------------- retrieval: the atlas at K = 2
T = pd.read_csv(os.path.join(ATLAS, "analysis", "tables", "per_query_topk.csv.gz"), dtype=str, keep_default_na=False, usecols=["dataset", "query", "encoder", "layer", "K", "in_cohort", "same_class"])
T = T[(T.K == str(K))].copy(); assert (T.in_cohort == "True").all(); T["encoder"] = T.encoder.str.split("_", n=1).str[1]; T["dataset"] = T.dataset.str.split("_", n=1).str[1].str.replace("_", " "); T["sc"] = T.same_class.astype(int)
QI = T[["dataset", "query"]].drop_duplicates().sort_values(["dataset", "query"]).reset_index(drop=True); n = len(QI); ds = QI.dataset.values; DS = list(np.unique(ds)); ixd = {d: np.where(ds == d)[0] for d in DS}; qkey = pd.MultiIndex.from_frame(QI)
M = {"retrieval coverage": {}, "retrieval same-class share": {}, "Claude accuracy": {}}
for (e, l), g in T.groupby(["encoder", "layer"]):
    v = g.set_index(["dataset", "query"]).sc.reindex(qkey).values; assert not np.isnan(v).any(); M["retrieval coverage"][(e, l)] = (v >= 1).astype(float); M["retrieval same-class share"][(e, l)] = v / K

# ---------------------------------------------------------------- downstream: saved Claude predictions
PQ = pd.read_csv(os.path.join(RUN, "per_query_predictions.csv"), dtype=str, keep_default_na=False); Cn = pd.read_csv(os.path.join(RUN, "conditions.csv"), dtype=str, keep_default_na=False); Cn = Cn[Cn.encoder != ""]
src = {}
for r in Cn.itertuples():
    g = PQ[PQ.condition == r.condition].set_index(["dataset", "query_folder"]); assert len(g) == n and (g.pred != "").all(); ok = (g.pred == g.true_label).reindex(qkey); assert ok.notna().all()
    M["Claude accuracy"][(r.encoder, r.layer)] = ok.values.astype(float); src[(r.encoder, r.layer)] = os.path.basename(os.path.normpath(RUN))
pending = []
def parse(raw):
    try: return str(json.loads(raw)["prediction"])
    except Exception:
        m = re.search(r"\{.*\}", raw, re.S)
        try: return str(json.loads(m.group(0))["prediction"])
        except Exception: return "INVALID"
if EXTRA and os.path.exists(os.path.join(EXTRA, "request_plan.csv")):
    C2 = pd.read_csv(os.path.join(EXTRA, "conditions.csv"), dtype=str, keep_default_na=False); C2 = C2[(C2.encoder != "") & (C2.K == str(K))]; P2 = pd.read_csv(os.path.join(EXTRA, "request_plan.csv"), dtype=str, keep_default_na=False)
    old = PQ.drop_duplicates("payload").set_index("payload").pred
    for r in C2.itertuples():
        if (r.encoder, r.layer) in M["Claude accuracy"]: continue
        g = P2[P2.condition == r.condition]; have = [os.path.exists(os.path.join(EXTRA, "responses", k + ".json")) for k in g.payload]
        if len(g) == n and all(have):
            pr = [parse(json.load(open(os.path.join(EXTRA, "responses", k + ".json")))["raw"]) for k in g.payload]
            chk = [(p, old[k]) for p, k in zip(pr, g.payload) if k in old.index]; assert all(a == b for a, b in chk), "parser differs from the original report"
            ok = pd.Series(np.array(pr) == g.true_label.values, index=pd.MultiIndex.from_arrays([g.dataset, g.query_folder])).reindex(qkey); assert ok.notna().all()
            M["Claude accuracy"][(r.encoder, r.layer)] = ok.values.astype(float); src[(r.encoder, r.layer)] = os.path.basename(os.path.normpath(EXTRA))
        else: pending.append({"encoder": r.encoder, "layer": r.layer, "answered": int(sum(have)), "of": len(g)})

# ---------------------------------------------------------------- coverage inventory
gof = lambda e, l: next((g for g, r in GROUPS[e].items() if num(l) in r), "pooled output" if l == "final_output" else "not in a group (stem)")
INV = []
for e in ENC:
    for l in sorted({l for (m, l) in M["retrieval coverage"] if m == e}, key=num):
        has = (e, l) in M["Claude accuracy"]; r = {"encoder": e, "atlas_layer": l, "zero_indexed_block": num(l) - 1 if l.startswith("layer_") else "", "group": gof(e, l), "retrieval_K2_queries": n, "Claude_predictions": has, "Claude_source_run": src.get((e, l), ""), "Claude_queries": n if has else 0}
        for d in DS: r[f"retrieval_n_{d}"] = len(ixd[d]); r[f"Claude_n_{d}"] = len(ixd[d]) if has else 0
        INV.append(r)
INV = pd.DataFrame(INV); INV.to_csv(os.path.join(OUT, "coverage.csv"), index=False)

# ---------------------------------------------------------------- contrasts
rng = np.random.default_rng(SEED); Cw = np.zeros((NB, n))
for d in DS: Cw[:, ixd[d]] = rng.multinomial(len(ixd[d]), np.ones(len(ixd[d])) / len(ixd[d]), size=NB)
SG = np.random.default_rng(SEED + 1).integers(0, 2, size=(NPERM, n), dtype=np.int8) * 2 - 1
def est(a, b, ix, test):
    d = (a - b)[ix]; bs = Cw[:, ix] @ d / len(ix) * 100; r = {"n_matched": len(ix), "mean_A": 100 * a[ix].mean(), "mean_B": 100 * b[ix].mean(), "diff_pp": 100 * d.mean(), "ci95_lo": float(np.percentile(bs, 2.5)), "ci95_hi": float(np.percentile(bs, 97.5))}
    if test: r["p_signflip"] = float((1 + (np.abs(SG[:, ix].astype(np.float64) @ d) >= abs(d.sum()) - 1e-9).sum()) / (1 + NPERM))
    return r
rows = []
for metric, V in M.items():
    for e in ENC:
        mem = {g: [f"layer_{b:02d}" for b in r if (e, f"layer_{b:02d}") in V] for g, r in GROUPS[e].items()}; avg = {g: np.mean([V[(e, l)] for l in ls], axis=0) for g, ls in mem.items() if ls}
        pairs = [("middle", "early"), ("late", "early"), ("late", "middle")] + [(g, "pooled") for g in ["early", "middle", "late"]]
        for ga, gb in pairs:
            fam = "group_vs_pooled" if gb == "pooled" else "group_vs_group"; base = {"metric": metric, "family": fam, "encoder": e, "A": ga, "B": gb, "blocks_A_atlas": " ".join(str(num(l)) for l in mem[ga]), "blocks_B_atlas": "pooled" if gb == "pooled" else " ".join(str(num(l)) for l in mem[gb]),
                    "blocks_A_zero_indexed": " ".join(str(num(l) - 1) for l in mem[ga]), "blocks_B_zero_indexed": "pooled" if gb == "pooled" else " ".join(str(num(l) - 1) for l in mem[gb]), "n_blocks_A": len(mem[ga]), "n_blocks_B": 1 if gb == "pooled" else len(mem[gb]), "blocks_in_group_A": len(GROUPS[e][ga]), "blocks_in_group_B": 1 if gb == "pooled" else len(GROUPS[e][gb])}
            if not mem[ga] or (gb != "pooled" and not mem[gb]) or (gb == "pooled" and (e, "final_output") not in V):
                rows.append({**base, "scope": "overall", "status": "unavailable: no saved Claude predictions for " + " and ".join(g for g in ([ga] + ([gb] if gb != "pooled" else [])) if not mem[g]) + " blocks"}); continue
            b = V[(e, "final_output")] if gb == "pooled" else avg[gb]; st = "evaluated" if min(base["n_blocks_A"], base["n_blocks_B"] if gb != "pooled" else 9) >= 2 else "evaluated, one block in a group"
            if metric == "Claude accuracy" and (len(mem[ga]) < len(GROUPS[e][ga]) or (gb != "pooled" and len(mem[gb]) < len(GROUPS[e][gb]))): st += ", partial block coverage"
            for scope, ix in [("overall", np.arange(n))] + [(d, ixd[d]) for d in DS]: rows.append({**base, "scope": scope, "status": st, **est(avg[ga], b, ix, scope == "overall")})
R = pd.DataFrame(rows)
for c in ["holm_within_encoder", "holm_all_encoders", "tests_within_encoder", "tests_all_encoders"]: R[c] = np.nan
for (metric, fam), g in R[(R.scope == "overall") & R.p_signflip.notna()].groupby(["metric", "family"]):
    R.loc[g.index, "holm_all_encoders"] = holm(g.p_signflip); R.loc[g.index, "tests_all_encoders"] = len(g)
    for e, h in g.groupby("encoder"): R.loc[h.index, "holm_within_encoder"] = holm(h.p_signflip); R.loc[h.index, "tests_within_encoder"] = len(h)
def verdict(r):
    if str(r.status).startswith("unavailable"): return "untested"
    w = r.A if r.diff_pp > 0 else r.B
    if r.holm_all_encoders < ALPHA: return f"{w} higher (holds across encoders)"
    if r.holm_within_encoder < ALPHA: return f"{w} higher (within encoder)"
    return "inconclusive"
R["verdict"] = [verdict(r) if r.scope == "overall" else "" for r in R.itertuples()]
R.to_csv(os.path.join(OUT, "depth_group_contrasts.csv"), index=False)

# ---------------------------------------------------------------- summary
O = R[R.scope == "overall"]; cell = lambda r: "untested" if str(r.status).startswith("unavailable") else f"{r.diff_pp:+.1f} ({r.ci95_lo:.1f}, {r.ci95_hi:.1f}), Holm {r.holm_within_encoder:.3g}: {r.verdict}"
def wide(metric, fam):
    g = O[(O.metric == metric) & (O.family == fam)]; rows = []
    for e in ENC:
        h = g[g.encoder == e]; r = {"encoder": e}
        for x in h.itertuples(): r[f"{x.A} minus {x.B}" + (f" [blocks {x.blocks_A_atlas} vs {x.blocks_B_atlas}]" if metric == "Claude accuracy" and not str(x.status).startswith("unavailable") and fam == "group_vs_group" else "")] = cell(x)
        rows.append(r)
    return pd.DataFrame(rows)
def rev(metric):
    g = R[(R.metric == metric) & (R.family == "group_vs_group") & (R.scope != "overall") & R.diff_pp.notna()].merge(O[["metric", "encoder", "A", "B", "diff_pp", "verdict"]], on=["metric", "encoder", "A", "B"], suffixes=("", "_overall"))
    g = g[(np.sign(g.diff_pp) == -np.sign(g.diff_pp_overall)) & (g.verdict_overall != "inconclusive")]
    return md(g[["encoder", "A", "B", "scope", "diff_pp", "ci95_lo", "ci95_hi", "diff_pp_overall"]]) if len(g) else "none"
cov = INV[INV.group.isin(["early", "middle", "late"])].groupby(["encoder", "group"], sort=False).agg(blocks=("atlas_layer", "size"), retrieval_blocks=("retrieval_K2_queries", lambda s: int((s > 0).sum())), Claude_blocks=("Claude_predictions", "sum"), Claude_atlas_layers=("atlas_layer", lambda s: " ".join(str(num(l)) for l in s[INV.loc[s.index, "Claude_predictions"]]))).reset_index()
out = ["# Early, middle and late blocks within each encoder (K = 2, 500 queries, five datasets)", "", "No request was sent. Differences in percentage points; 95% bootstrap intervals are pointwise; Holm over the three contrasts of an encoder within the metric.", "", "## Numbering and groups", "", manifest["numbering"], "", manifest["qwen_check"], ""] + [f"- {e}: {RULE[e]}" for e in ENC]
out += ["", "## Coverage (every block has retrieval results for 100 queries per dataset; Claude predictions cover all five datasets, 100 queries each, for the listed layers)", "", md(cov), "", f"K = 2 conditions of the all-budget run not complete yet (not used): {pending if pending else 'none' if EXTRA else 'run not passed'}", "",
        "## Retrieval, coverage (primary): group against group", "", md(wide("retrieval coverage", "group_vs_group")), "", "## Retrieval, same-class share: group against group", "", md(wide("retrieval same-class share", "group_vs_group")), "", "## Retrieval, coverage: group against pooled output", "", md(wide("retrieval coverage", "group_vs_pooled")), "",
        "## Retrieval coverage: group means (percent)", "", md(O[(O.metric == "retrieval coverage") & (O.family == "group_vs_pooled")].pivot(index="encoder", columns="A", values="mean_A").reindex(ENC)[["early", "middle", "late"]].assign(pooled=O[(O.metric == "retrieval coverage") & (O.family == "group_vs_pooled") & (O.A == "late")].set_index("encoder").mean_B).reset_index()), "",
        "## Retrieval coverage: datasets where a supported overall result reverses", "", rev("retrieval coverage"), "", "## Claude accuracy: group against group", "", md(wide("Claude accuracy", "group_vs_group").fillna("")), "", "## Claude accuracy: group against pooled output", "", md(wide("Claude accuracy", "group_vs_pooled")), "",
        "## Claude accuracy by dataset (descriptive)", "", md(R[(R.metric == "Claude accuracy") & (R.family == "group_vs_group") & (R.scope != "overall") & R.diff_pp.notna()][["encoder", "A", "B", "blocks_A_atlas", "blocks_B_atlas", "scope", "n_matched", "mean_A", "mean_B", "diff_pp", "ci95_lo", "ci95_hi"]]), ""]
open(os.path.join(OUT, "SUMMARY.md"), "w").write("\n".join(out)); print("\n".join(out))
