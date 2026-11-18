"""Builds the downstream layer test of the paper from runs/layer_vlm_k2 (pilot_layer_vlm.py):
figs/fig_layer_vlm.{pdf,png} (main text), figs/appendix/fig_layer_vlm_all.{pdf,png}, appendix_vlm.tex and tables/layer_vlm_numbers.json.
Usage: python tools/make_layer_vlm.py [RUN_DIR]     (default ../../code/ageval-dinov2-pilot/runs/layer_vlm_k2)
"""
import json, os, sys
import numpy as np, pandas as pd
from scipy import stats
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__))); RUN = sys.argv[1] if len(sys.argv) > 1 else os.path.join(HERE, "runs", "layer_vlm_k2")
plt.rcParams.update({"font.family": "Helvetica", "font.size": 7, "axes.spines.top": False, "axes.spines.right": False, "axes.linewidth": 0.6, "pdf.fonttype": 42})
ENC = ["ViT", "DINOv2", "CLIP", "ResNet50", "SigLIP2", "BioCLIP", "DINOv3", "Qwen3VL", "FLAVA"]; MAIN = ["ViT", "CLIP", "ResNet50", "SigLIP2", "Qwen3VL", "FLAVA"]          # the main text reports six encoders; the appendix keeps all nine
COL = {"ViT": "#0072B2", "DINOv2": "#D55E00", "CLIP": "#009E73", "ResNet50": "#CC79A7", "SigLIP2": "#E69F00", "BioCLIP": "#56B4E9", "DINOv3": "#000000", "Qwen3VL": "#8C5A2B", "FLAVA": "#7F7F7F"}
NM = {"ViT": "ViT", "DINOv2": "DINOv2", "CLIP": "CLIP", "ResNet50": "ResNet-50", "SigLIP2": "SigLIP 2", "BioCLIP": "BioCLIP", "DINOv3": "DINOv3", "Qwen3VL": "Qwen3-VL tower", "FLAVA": "FLAVA"}
INK, MUT, SEED, NB = "#222222", "#666666", 20261115, 10000
R = pd.read_csv(os.path.join(RUN, "results_by_condition.csv"), keep_default_na=False, na_values=[""]); R["encoder"] = R.encoder.fillna("")
P = pd.read_csv(os.path.join(RUN, "per_query_predictions.csv"), dtype=str, keep_default_na=False); P["ok"] = P.pred == P.true_label
W = P.pivot(index=["dataset", "query_folder"], columns="condition", values="ok").astype(int); n = len(W)
zs = R[R.condition == "zero_shot"].iloc[0]; PO = R[R.layer == "final_output"].set_index("encoder").loc[ENC].reset_index()
LY = R[(R.encoder != "") & (R.layer != "final_output")].copy(); LY["ln"] = [int(l.split("_")[1]) for l in LY.layer]; ll = lambda r: f"L{r.ln}"
own = lambda e: "pooled_vit" if e == "ViT" else f"{e}|final_output"
best = LY.loc[LY.groupby("encoder", sort=False).delta_vs_own_pooled_pp.idxmax()].set_index("encoder").loc[ENC].reset_index()
rho, prho = stats.spearmanr(LY.retrieval_net_coverage_vs_own_pooled_pp, LY.delta_vs_own_pooled_pp)
# tests that are not identical to the pooled output (the last block of DINOv2 and DINOv3 returns the same references), Holm correction
T = LY[(LY.wins_vs_own_pooled + LY.losses_vs_own_pooled) > 0].sort_values("p_vs_own_pooled").reset_index(drop=True); m = len(T); T["holm"] = np.minimum(1, np.maximum.accumulate((m - np.arange(m)) * T.p_vs_own_pooled.values))
# encoder level: mean accuracy of the five tested layers against the own pooled output (paired bootstrap over queries)
C = np.random.default_rng(SEED).multinomial(n, np.ones(n) / n, size=NB).astype(np.float32); EL = {}
for e in ENC:
    d = W[list(LY[LY.encoder == e].condition)].mean(axis=1).values - W[own(e)].values; est = (d.astype(np.float32) @ C.T) / n * 100
    EL[e] = {"mean_of_five_layers_pp": round(100 * d.mean(), 2), "lo": round(float(np.percentile(est, 2.5)), 2), "hi": round(float(np.percentile(est, 97.5)), 2), "layers_above": int((LY[LY.encoder == e].delta_vs_own_pooled_pp > 0).sum())}
up, dn = LY[LY.ci_lo > 0], LY[LY.ci_hi < 0]; vit_up = LY[LY.vit_ci_lo > 0]; vit_dn = LY[LY.vit_ci_hi < 0]
L6 = LY[LY.encoder.isin(MAIN)]; r6, p6 = stats.spearmanr(L6.retrieval_net_coverage_vs_own_pooled_pp, L6.delta_vs_own_pooled_pp)
N = {"n": n, "main_text_six_encoders": {"layers": len(L6), "above_own_pooled": int((L6.ci_lo > 0).sum()), "below_own_pooled": int((L6.ci_hi < 0).sum()), "spearman": round(r6, 3), "spearman_p": float(p6), "pooled_range": [float(PO[PO.encoder.isin(MAIN)].accuracy.min()), float(PO[PO.encoder.isin(MAIN)].accuracy.max())]}, "zero_shot": zs.accuracy, "layers": len(LY), "above_own_pooled": len(up), "below_own_pooled": len(dn), "above_pooled_vit": len(vit_up), "below_pooled_vit": len(vit_dn), "spearman": round(rho, 3), "spearman_p": float(prho),
     "non_identical_tests": m, "holm_min": [{"layer": f"{r.encoder} {ll(r)}", "p": r.p_vs_own_pooled, "holm": r.holm} for r in T.head(4).itertuples()], "encoder_level": EL,
     "pooled": {r.encoder: {"acc": r.accuracy, "coverage": r.retrieval_coverage, "vs_zero_shot": r.delta_vs_zero_shot_pp, "zs_ci": [r.zs_ci_lo, r.zs_ci_hi], "wl_zs": [int(r.wins_vs_zero_shot), int(r.losses_vs_zero_shot)], "p_zs": r.p_vs_zero_shot} for r in PO.itertuples()},
     "best": {r.encoder: {"layer": ll(r), "acc": r.accuracy, "delta": r.delta_vs_own_pooled_pp, "ci": [r.ci_lo, r.ci_hi], "wl": [int(r.wins_vs_own_pooled), int(r.losses_vs_own_pooled)], "p": r.p_vs_own_pooled, "cov_change": r.retrieval_net_coverage_vs_own_pooled_pp, "vs_pooled_vit": r.delta_vs_pooled_vit_pp} for r in best.itertuples()}}
os.makedirs(os.path.join(HERE, "tables"), exist_ok=True); json.dump(N, open(os.path.join(HERE, "tables", "layer_vlm_numbers.json"), "w"), indent=1)

# ---- figures in the style of the paper (tools/biomor_style.py): pastel hatched bars, value labels above the bars
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import biomor_style as BS
BS.apply(); C_ZS, C_POOL, C_HID = BS.P["ref"], BS.P["lavender"], BS.P["green"]; Y0 = 40
def save(fig, path):
    fig.savefig(path + ".pdf", bbox_inches="tight", pad_inches=0.02); fig.savefig(path + ".png", dpi=600, bbox_inches="tight", pad_inches=0.02); plt.close(fig)
def zline(ax, x1): ax.axhline(zs.accuracy, color="#7F7F7F", lw=0.9, ls=":", zorder=1)
ZH = plt.Line2D([], [], color="#7F7F7F", lw=0.9, ls=":", label=f"Zero-shot, {zs.accuracy:.1f}%")
# main text: pooled output and best hidden layer of each encoder
fig, ax = plt.subplots(figsize=(6.9, 1.75))
for i, e in enumerate(MAIN):
    p, b = PO[PO.encoder == e].iloc[0], best[best.encoder == e].iloc[0]
    for x, v, c in ((i - 0.2, p.accuracy, C_POOL), (i + 0.2, b.accuracy, C_HID)): BS.bar(ax, x, v - Y0, 0.38, c, bottom=Y0); BS.vlabel(ax, x, v, f"{v:.1f}", fs=5.4, pad=0.3)
zline(ax, len(MAIN)); ax.set_xticks(range(len(MAIN)), [f"{NM[e]}\nblock {best[best.encoder == e].iloc[0].ln}" for e in MAIN]); ax.set_ylim(Y0, 66); ax.set_xlim(-0.6, len(MAIN) - 0.4); ax.set_ylabel("VLM accuracy (%)")
BS.top_legend(ax, [BS.patch(C_POOL, "Pooled output of the encoder"), BS.patch(C_HID, "Best of five hidden layers"), ZH], ncol=3)
save(fig, os.path.join(HERE, "figs", "fig_layer_vlm"))
# appendix: every tested layer
fig, ax = plt.subplots(figsize=(6.9, 2.0)); xt, xl = [], []
for i, e in enumerate(ENC):
    p = PO[PO.encoder == e].iloc[0]; g = LY[LY.encoder == e].sort_values("ln"); x0 = i * 7
    BS.bar(ax, x0, p.accuracy - Y0, 0.9, C_POOL, bottom=Y0); BS.vlabel(ax, x0, p.accuracy, f"{p.accuracy:.1f}", fs=4.6, pad=0.3); xt.append(x0); xl.append("P")
    for k, r in enumerate(g.itertuples()): BS.bar(ax, x0 + 1 + k, r.accuracy - Y0, 0.9, C_HID, bottom=Y0); BS.vlabel(ax, x0 + 1 + k, r.accuracy, f"{r.accuracy:.1f}", fs=4.6, pad=0.3); xt.append(x0 + 1 + k); xl.append(str(r.ln))
    ax.text(x0 + 2.5, Y0 - 3.4, NM[e], ha="center", va="top", fontsize=6.0, color=BS.INK)
zline(ax, 0); ax.set_xticks(xt, xl, fontsize=4.0); ax.tick_params(axis="x", length=1.5, pad=1); ax.set_ylim(Y0, 67); ax.set_xlim(-1, len(ENC) * 7 - 1); ax.set_ylabel("VLM accuracy (%)")
BS.top_legend(ax, [BS.patch(C_POOL, "Pooled output (P)"), BS.patch(C_HID, "Hidden layer (block number)"), ZH], ncol=3)
save(fig, os.path.join(HERE, "figs", "appendix", "fig_layer_vlm_all"))
# appendix: retrieval change against VLM change
fig, a = plt.subplots(figsize=(3.3, 2.3)); a.grid(True, axis="both")
for e in ENC: g = LY[LY.encoder == e]; a.scatter(g.retrieval_net_coverage_vs_own_pooled_pp, g.delta_vs_own_pooled_pp, s=16, color=COL[e], edgecolors="white", linewidths=0.4, label=NM[e], zorder=3)
a.axhline(0, color=BS.SPINE, lw=0.7); a.axvline(0, color=BS.SPINE, lw=0.7); a.set_xlabel("Change in same-class coverage (pp)"); a.set_ylabel("Change in VLM accuracy (pp)"); a.legend(loc="lower center", bbox_to_anchor=(0.5, 1.02), ncol=3, handletextpad=0.1, columnspacing=0.8)
save(fig, os.path.join(HERE, "figs", "appendix", "fig_layer_vlm_scatter"))

# ---- appendix section
PP, PCT, BEST = "\\pp", "\\%", "\\best "
pv = lambda p: "$<$0.001" if p < 0.001 else f"{p:.3f}"; sg = lambda v: f"{v:+.1f}".replace("-", "$-$"); DS = sorted(c[4:] for c in R.columns if c.startswith("acc_")); DSN = {d: d.replace("Leaf Lesions", "Leaf").replace(" Disease", "").replace("Soybean ", "Soy. ") for d in DS}
rows = []
for e in ENC:
    p = PO[PO.encoder == e].iloc[0]; rows.append(f"{NM[e]} & pooled & {p.retrieval_coverage:.1f} & & {p.accuracy:.1f} & & & & " + " & ".join(f"{p['acc_' + d]:.0f}" for d in DS) + " \\\\")
    for r in LY[LY.encoder == e].sort_values("ln").itertuples():
        same = r.wins_vs_own_pooled + r.losses_vs_own_pooled == 0; hi = BEST if r.ci_lo > 0 else ""
        rows.append(f" & block {r.ln} & {r.retrieval_coverage:.1f} & {sg(r.retrieval_net_coverage_vs_own_pooled_pp)} & {hi}{r.accuracy:.1f} & " + ("\\multicolumn{3}{c|}{same references as the pooled output} & " if same else f"{hi}{sg(r.delta_vs_own_pooled_pp)} ({sg(r.ci_lo)} to {sg(r.ci_hi)}) & {int(r.wins_vs_own_pooled)} / {int(r.losses_vs_own_pooled)} & {pv(r.p_vs_own_pooled)} & ") + " & ".join(f"{getattr(r, 'acc_' + d.replace(' ', '_')) if False else R.loc[r.Index, 'acc_' + d]:.0f}" for d in DS) + " \\\\")
    if e != ENC[-1]: rows.append("\\midrule")
el = lambda e: f"{sg(EL[e]['mean_of_five_layers_pp'])}{PP} (95{PCT} CI {sg(EL[e]['lo'])} to {sg(EL[e]['hi'])})"
h = T.iloc[0]
tex = f"""% Generated by tools/make_layer_vlm.py from runs/layer_vlm_k2. Do not edit by hand.
\\clearpage
\\section{{Hidden layers against the pooled output with the VLM}}
\\label{{app:vlm}}

This appendix gives every condition of the first downstream test, which covers nine encoders with two references per query. Its answers for the six encoders of the main text are the two-reference results of Sec.~\\ref{{sec:vlm}}, and Appendix~\\ref{{app:vlmk}} gives the other budgets. The test uses {n} queries with the disjoint reference bank of Appendix~\\ref{{app:atlas}}, two references per query, the standard assisted prompt and Claude Sonnet 5.5.

\\textbf{{Conditions.}} The test has 55 conditions: zero-shot prompting, the pooled (final) output of each of the nine encoders, and five hidden layers of each encoder. The five layers of an encoder were selected by their retrieval gain at $K=2$ on these queries; the reference of this selection was the pooled ViT output, and not the pooled output of the encoder. The conditions with the pooled outputs of the eight other encoders were added after the first results, so that each layer is compared with the pooled output of its own encoder. Identical prompts are sent once: the 27{{,}}500 query-condition pairs correspond to 12{{,}}943 distinct requests. All {n} queries are answered in all conditions.

\\textbf{{Statistics.}} Each layer is compared with the pooled output of its encoder on the same queries: accuracy change with a paired bootstrap interval over queries (10{{,}}000 replicates), wins, losses and the exact McNemar test. The last block of DINOv2 and of DINOv3 returns the references of the pooled output for every query, so {m} of the {len(LY)} comparisons are not identical by construction.

\\textbf{{Result.}} Table~\\ref{{tab:app-vlm}} and Fig.~\\ref{{fig:app-vlm}} give all layers. {len(up)} layers are above the pooled output of their encoder with an interval that excludes zero, {len(dn)} are below it, and {len(LY) - len(up) - len(dn)} cannot be separated from it. The single $p$-values are not corrected for the {m} comparisons. With a Holm correction, the smallest adjusted $p$-value is {h.holm:.3f} ({NM[h.encoder]}, block {h.ln}), so no single layer is significant after the correction. The comparison at the level of the encoder does not depend on the choice of one layer: the mean accuracy of the five tested layers against the pooled output is {el('Qwen3VL')} for the Qwen3-VL tower, {el('FLAVA')} for FLAVA, {el('ResNet50')} for ResNet-50, {el('SigLIP2')} for SigLIP 2, {el('BioCLIP')} for BioCLIP, {el('ViT')} for the ViT, {el('DINOv2')} for DINOv2, {el('CLIP')} for CLIP and {el('DINOv3')} for DINOv3. Over the {len(LY)} layers, the Spearman correlation between the same-class coverage change and the accuracy change, both against the pooled output of the encoder, is {rho:.2f} (Fig.~\\ref{{fig:app-vlmsc}}).

\\begin{{figure*}}[t]
\\centering
\\includegraphics[width=\\linewidth]{{figs/appendix/fig_layer_vlm_all.pdf}}
\\caption{{\\textbf{{VLM accuracy with the pooled output and with each tested hidden layer}} of the nine encoders. Two references per query, {n} queries, Claude Sonnet 5.5. The accuracy axis starts at 40\\%.}}
\\label{{fig:app-vlm}}
\\end{{figure*}}

\\begin{{figure}}[t]
\\centering
\\includegraphics[width=\\linewidth]{{figs/appendix/fig_layer_vlm_scatter.pdf}}
\\caption{{\\textbf{{Retrieval change against VLM change.}} Each point is one of the {len(LY)} hidden layers: its change in same-class coverage and its change in VLM accuracy, both against the pooled output of its encoder.}}
\\label{{fig:app-vlmsc}}
\\end{{figure}}

\\begin{{table*}}[t]
\\centering
\\caption{{\\textbf{{All conditions of the downstream test.}} Same-class coverage (share of queries with at least one same-class reference of two, in percent) and its change against the pooled output of the encoder; VLM accuracy in percent, its change against the pooled output with the 95\\% interval, wins and losses, and the exact McNemar $p$-value; accuracy by dataset. Zero-shot prompting reaches {zs.accuracy:.1f}\\%. Layers above the pooled output with an interval that excludes zero in green.}}
\\label{{tab:app-vlm}}
\\scriptsize
\\setlength{{\\tabcolsep}}{{3.2pt}}\\renewcommand{{\\arraystretch}}{{0.93}}
\\begin{{tabular}}{{@{{}}ll|cc|cccc|{'c' * len(DS)}@{{}}}}
\\toprule
 & & \\multicolumn{{2}}{{c|}}{{Retrieval coverage}} & \\multicolumn{{4}}{{c|}}{{VLM accuracy, {n} queries}} & \\multicolumn{{{len(DS)}}}{{c}}{{VLM accuracy by dataset}} \\\\
Encoder & Latent & \\% & Change & \\% & Change (95\\% CI) & Wins / losses & $p$ & {' & '.join(DSN[d] for d in DS)} \\\\
\\midrule
""" + "\n".join(rows) + """
\\bottomrule
\\end{tabular}
\\end{table*}
"""
open(os.path.join(HERE, "appendix_vlm.tex"), "w").write(tex); print(json.dumps(N, indent=1))
