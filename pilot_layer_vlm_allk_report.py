"""Report of the all-budget within-encoder layer test (pilot_layer_vlm_allk.py). No VLM call.
Each hidden layer is compared with the pooled output of its own encoder at the same K, on the queries answered in both conditions.
Usage (from repo/): python pilot_layer_vlm_allk_report.py <run_dir>
Writes results_by_condition.csv, encoder_level.csv, per_query_predictions.csv, fig_pooled_vs_best_layer.{png,pdf}, fig_all_layers.{png,pdf}, SUMMARY.md.
"""
import json, os, sys
import numpy as np, pandas as pd
from scipy import stats
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "paper_tools"))
import biomor_style as BS
os.environ.setdefault("KAGGLE_USERNAME", "x"); os.environ.setdefault("KAGGLE_KEY", "x")
import inference as inf

out = sys.argv[1]; P = lambda f: os.path.join(out, f); KS = (1, 2, 4, 8); SEED, NB = 20261115, 10000
ENC = ["ViT", "CLIP", "ResNet50", "SigLIP2", "Qwen3VL", "FLAVA"]; NM = {"ViT": "ViT", "CLIP": "CLIP", "ResNet50": "ResNet-50", "SigLIP2": "SigLIP 2", "Qwen3VL": "Qwen3-VL tower", "FLAVA": "FLAVA"}
ATLAS = "atlas"; INV = {"1_ViT": "ViT", "3_CLIP": "CLIP", "4_ResNet50": "ResNet50", "5_SigLIP2": "SigLIP2", "8_Qwen3VL": "Qwen3VL", "9_FLAVA": "FLAVA"}
Cn = pd.read_csv(P("conditions.csv"), keep_default_na=False); PL = pd.read_csv(P("request_plan.csv"), dtype=str, keep_default_na=False)
def pred(k):
    f = P(os.path.join("responses", k + ".json"))
    if not os.path.exists(f): return None
    try: return str(inf.extract_json(json.load(open(f))["raw"])["prediction"])
    except Exception: return "INVALID"
PL["pred"] = PL.payload.map({k: pred(k) for k in PL.payload.unique()}); PL["answered"] = PL.pred.notna(); PL["ok"] = (PL.pred == PL.true_label).astype(float); PL.loc[~PL.answered, "ok"] = np.nan
PL[["condition", "dataset", "query_folder", "true_label", "pred", "payload"]].to_csv(P("per_query_predictions.csv"), index=False)
W = PL.pivot(index=["dataset", "query_folder"], columns="condition", values="ok"); n_all = len(W); done = int(PL.drop_duplicates("payload").answered.sum()); total = PL.payload.nunique()
# retrieval coverage of each condition (same-class reference among the top K), from the atlas
TQ = pd.read_csv(os.path.join(ATLAS, "analysis", "tables", "per_query_topk.csv.gz"), usecols=["encoder", "layer", "K", "same_class"]); TQ = TQ[TQ.encoder.isin(INV)]; COV = (TQ.assign(c=TQ.same_class > 0).groupby(["encoder", "layer", "K"]).c.mean() * 100)
cov = lambda c: float(COV.loc[({v: k for k, v in INV.items()}[c.encoder], c.layer, int(c.K))])
rng = np.random.default_rng(SEED)
def paired(a, b):
    m = W[[a]].assign(_b=W[b]).dropna(); d = (m[a] - m["_b"]).values; n = len(d)
    if n == 0: return None
    C = rng.multinomial(n, np.ones(n) / n, size=NB).astype(np.float64); est = C @ d / n * 100; w_, l_ = int((d > 0.5).sum()), int((d < -0.5).sum())
    return {"n": n, "acc_a": 100 * m[a].mean(), "acc_b": 100 * m["_b"].mean(), "delta": 100 * d.mean(), "lo": float(np.percentile(est, 2.5)), "hi": float(np.percentile(est, 97.5)), "wins": w_, "losses": l_, "p": 1.0 if w_ + l_ == 0 else float(stats.binomtest(w_, w_ + l_, 0.5).pvalue)}
rows = []
for c in Cn.itertuples():
    if not c.encoder: continue
    base = f"{c.encoder}|final_output|K{c.K}"; v = paired(c.condition, base); z = paired(c.condition, "zero_shot")
    if v is None: continue
    r = {"condition": c.condition, "encoder": c.encoder, "layer": c.layer, "K": int(c.K), "is_pooled": c.layer == "final_output", "n": v["n"], "accuracy": v["acc_a"], "own_pooled_accuracy": v["acc_b"], "retrieval_coverage": cov(c), "retrieval_change_vs_own_pooled_pp": cov(c) - float(COV.loc[({v_: k for k, v_ in INV.items()}[c.encoder], "final_output", int(c.K))]),
         "delta_vs_own_pooled_pp": v["delta"], "ci_lo": v["lo"], "ci_hi": v["hi"], "wins": v["wins"], "losses": v["losses"], "p": v["p"], "delta_vs_zero_shot_pp": z["delta"] if z else np.nan, "zs_lo": z["lo"] if z else np.nan, "zs_hi": z["hi"] if z else np.nan}
    for d_, g in W.groupby(level=0): r[f"acc_{d_}"] = 100 * g[c.condition].mean()
    rows.append(r)
R = pd.DataFrame(rows); LY = R[~R.is_pooled].copy(); LY["ln"] = [int(l.split("_")[1]) for l in LY.layer]; PO = R[R.is_pooled]
T = LY[(LY.wins + LY.losses) > 0].sort_values("p"); m = len(T); holm = np.minimum(1, np.maximum.accumulate((m - np.arange(m)) * T.p.values)); LY.loc[T.index, "p_holm"] = holm; R.loc[T.index, "p_holm"] = holm; R.to_csv(P("results_by_condition.csv"), index=False)
# encoder level: mean of the five layers against the own pooled output
EL = []
for k in KS:
    for e in ENC:
        cs = list(LY[(LY.encoder == e) & (LY.K == k)].condition); base = f"{e}|final_output|K{k}"; mm = W[cs + [base]].dropna()
        if not len(mm) or not cs: continue
        d = (mm[cs].mean(axis=1) - mm[base]).values; n = len(d); C = rng.multinomial(n, np.ones(n) / n, size=NB).astype(np.float64); est = C @ d / n * 100
        EL.append({"K": k, "encoder": e, "n": n, "pooled_accuracy": 100 * mm[base].mean(), "mean_of_five_layers_accuracy": 100 * mm[cs].values.mean(), "delta_pp": 100 * d.mean(), "ci_lo": float(np.percentile(est, 2.5)), "ci_hi": float(np.percentile(est, 97.5)), "layers_above": int((LY[(LY.encoder == e) & (LY.K == k)].delta_vs_own_pooled_pp > 0).sum())})
EL = pd.DataFrame(EL); EL.to_csv(P("encoder_level.csv"), index=False); zs = 100 * W["zero_shot"].mean()
best = LY.loc[LY.groupby(["K", "encoder"]).delta_vs_own_pooled_pp.idxmax()]

# ---- figures in the style of the paper
BS.apply(); C_POOL, C_HID = BS.P["lavender"], BS.P["green"]; Y0 = 40
def save(fig, name): fig.savefig(P(name + ".pdf"), bbox_inches="tight", pad_inches=0.02); fig.savefig(P(name + ".png"), dpi=600, bbox_inches="tight", pad_inches=0.02); plt.close(fig)
ZH = plt.Line2D([], [], color="#7F7F7F", lw=0.9, ls=":", label=f"Zero-shot, {zs:.1f}%")
fig, axs = plt.subplots(4, 1, figsize=(6.9, 6.0))
for ax, k in zip(axs, KS):
    for i, e in enumerate(ENC):
        p = PO[(PO.encoder == e) & (PO.K == k)]; b = best[(best.encoder == e) & (best.K == k)]
        if not len(p) or not len(b): continue
        p, b = p.iloc[0], b.iloc[0]
        for x, v, c in ((i - 0.2, p.accuracy, C_POOL), (i + 0.2, b.accuracy, C_HID)): BS.bar(ax, x, v - Y0, 0.38, c, bottom=Y0); BS.vlabel(ax, x, v, f"{v:.1f}", fs=5.4, pad=0.3)
    ax.axhline(zs, color="#7F7F7F", lw=0.9, ls=":", zorder=1); lab = [f"{NM[e]}\nblock {int(best[(best.encoder == e) & (best.K == k)].ln.iloc[0])}" if len(best[(best.encoder == e) & (best.K == k)]) else NM[e] for e in ENC]
    ax.set_xticks(range(len(ENC)), lab); ax.set_ylim(Y0, 82); ax.set_xlim(-0.6, len(ENC) - 0.4); ax.set_ylabel(f"K = {k}\nVLM accuracy (%)")
BS.top_legend(axs[0], [BS.patch(C_POOL, "Pooled output of the encoder"), BS.patch(C_HID, "Best of five hidden layers of the same encoder"), ZH], ncol=3); fig.tight_layout(h_pad=0.8); save(fig, "fig_pooled_vs_best_layer")
fig, axs = plt.subplots(4, 1, figsize=(6.9, 6.6))
for ax, k in zip(axs, KS):
    xt, xl = [], []
    for i, e in enumerate(ENC):
        p = PO[(PO.encoder == e) & (PO.K == k)]; g = LY[(LY.encoder == e) & (LY.K == k)].sort_values("ln"); x0 = i * 7
        if len(p): BS.bar(ax, x0, p.accuracy.iloc[0] - Y0, 0.9, C_POOL, bottom=Y0); BS.vlabel(ax, x0, p.accuracy.iloc[0], f"{p.accuracy.iloc[0]:.1f}", fs=4.6, pad=0.3)
        xt.append(x0); xl.append("P")
        for j, r in enumerate(g.itertuples()): BS.bar(ax, x0 + 1 + j, r.accuracy - Y0, 0.9, C_HID, bottom=Y0); BS.vlabel(ax, x0 + 1 + j, r.accuracy, f"{r.accuracy:.1f}", fs=4.6, pad=0.3); xt.append(x0 + 1 + j); xl.append(str(r.ln))
        ax.text(x0 + 2.5, Y0 - 6.5, NM[e], ha="center", va="top", fontsize=6.0, color=BS.INK)
    ax.axhline(zs, color="#7F7F7F", lw=0.9, ls=":", zorder=1); ax.set_xticks(xt, xl, fontsize=4.6); ax.tick_params(axis="x", length=1.5, pad=1); ax.set_ylim(Y0, 84); ax.set_xlim(-1, len(ENC) * 7 - 1); ax.set_ylabel(f"K = {k}\nVLM accuracy (%)")
BS.top_legend(axs[0], [BS.patch(C_POOL, "Pooled output (P)"), BS.patch(C_HID, "Hidden layer (block number)"), ZH], ncol=3); fig.tight_layout(h_pad=1.6); save(fig, "fig_all_layers")

# ---- summary
led = P("request_ledger.jsonl"); L_ = [json.loads(l) for l in open(led)] if os.path.exists(led) else []; cost = sum(e.get("cost") or 0 for e in L_); sent = len(L_); failed = sum(not e.get("ok") for e in L_)
ll = lambda r: f"{NM[r.encoder]} block {int(r.ln)}"; ci = lambda r: f"{r.delta_vs_own_pooled_pp:+.1f} pp (CI {r.ci_lo:+.1f} to {r.ci_hi:+.1f}; {int(r.wins)} wins, {int(r.losses)} losses)"
S = [f"# Hidden layers against the pooled output of the same encoder with the VLM, K = 1, 2, 4, 8\n",
     f"VLM `claude-sonnet-5-5`, standard assisted prompt, {n_all} queries with a disjoint reference bank, six encoders. Requests answered: {done:,} of {total:,}" + ("" if done == total else " (**run not complete; numbers below use the queries answered in both conditions of a comparison**)") + f". Sent in this run: {sent:,} ({failed} failed), recorded API-equivalent cost ${cost:.2f}. Zero-shot: {zs:.1f}%.\n",
     "Every hidden layer is compared with the pooled output of its own encoder at the same K. Outputs of different encoders are not compared.\n",
     "![Pooled output and best hidden layer](fig_pooled_vs_best_layer.png)\n", "**Fig. 1 | VLM accuracy with the pooled output and with the best of five hidden layers of the same encoder,** by budget. The accuracy axis starts at 40%.\n", "## Result by budget\n"]
for k in KS:
    g = LY[LY.K == k]; up, dn = g[g.ci_lo > 0], g[g.ci_hi < 0]; rho, pr = stats.spearmanr(g.retrieval_change_vs_own_pooled_pp, g.delta_vs_own_pooled_pp) if len(g) > 2 else (np.nan, np.nan)
    S.append(f"- **K = {k}:** {len(up)} of {len(g)} hidden layers are above the pooled output of their encoder with an interval that excludes zero, {len(dn)} below. " + ("Above: " + "; ".join(f"{ll(r)} {r.accuracy:.1f}% against {r.own_pooled_accuracy:.1f}%, {ci(r)}" for r in up.sort_values("delta_vs_own_pooled_pp", ascending=False).itertuples()) + ". " if len(up) else "")
             + ("Below: " + "; ".join(f"{ll(r)} {ci(r)}" for r in dn.sort_values("delta_vs_own_pooled_pp").itertuples()) + ". " if len(dn) else "") + f"Spearman correlation of retrieval change and VLM change over the layers: {rho:+.2f} (p = {pr:.2f}).")
sig = LY[LY.p_holm < 0.05]
S += [f"\n**Multiplicity.** {m} layer comparisons. After a Holm correction, {len(sig)} are significant at 0.05" + (": " + "; ".join(f"{ll(r)} at K = {r.K} (adjusted p = {r.p_holm:.3f})" for r in sig.itertuples()) if len(sig) else f" (smallest adjusted p = {LY.p_holm.min():.3f})") + ".\n",
      "## Pooled output and best hidden layer of each encoder\n", "| K | Encoder | Pooled (%) | Best layer | Layer (%) | Change (pp) | 95% CI | Wins / losses | p | Holm p |", "|---|---|---|---|---|---|---|---|---|---|"]
for k in KS:
    for e in ENC:
        b = best[(best.encoder == e) & (best.K == k)]
        if len(b): r = b.iloc[0]; S.append(f"| {k} | {NM[e]} | {r.own_pooled_accuracy:.1f} | block {int(r.ln)} | {r.accuracy:.1f} | {r.delta_vs_own_pooled_pp:+.1f} | {r.ci_lo:+.1f} to {r.ci_hi:+.1f} | {int(r.wins)} / {int(r.losses)} | {r.p:.3f} | {r.p_holm:.3f} |")
S += ["\nThe best layer of an encoder is picked from five on the same queries, so its change is optimistic.\n", "## Mean of the five hidden layers against the pooled output (does not depend on picking one layer)\n", "| K | Encoder | Pooled (%) | Mean of five layers (%) | Change (pp) | 95% CI | Layers above pooled |", "|---|---|---|---|---|---|---|"]
for r in EL.itertuples(): S.append(f"| {r.K} | {NM[r.encoder]} | {r.pooled_accuracy:.1f} | {r.mean_of_five_layers_accuracy:.1f} | {r.delta_pp:+.1f} | {r.ci_lo:+.1f} to {r.ci_hi:+.1f} | {r.layers_above} of 5 |")
S += ["\n![All layers](fig_all_layers.png)\n", "**Fig. 2 | Every tested condition.** P: pooled output; numbers: block of the hidden layer.\n", "## Limits\n", "- The five layers per encoder and budget were selected by their retrieval gain over the pooled output of the same encoder on the evaluated queries.", "- Results at K = 2 from an earlier run were known before this plan was frozen.",
      "- Queries and references are not separated by capture date.", "- One VLM and one request per prompt.\n", "Files: `results_by_condition.csv`, `encoder_level.csv`, `per_query_predictions.csv`, `conditions.csv`, `preregistration.json`, `request_plan.csv`, `responses/`.\n"]
open(P("SUMMARY.md"), "w").write("\n".join(S)); print("\n".join(S))
