"""Builds the appendix (figures, tables, appendix.tex) and the main-text gain figure from the layer-wise reference atlas.

Inputs: the atlas analysis tables (code/AgEval-LayerWise-Reference-Atlas/analysis) and the date-disjoint check
(code/ageval-dinov2-pilot/runs/blockwise_vs_pooled). Retrieval only; no model or VLM call.
Usage (from the paper root):  <python> tools/make_appendix.py [ATLAS_DIR] [BLOCKWISE_DIR]
"""
import json, os, shutil, sys
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ATLAS = sys.argv[1] if len(sys.argv) > 1 else os.path.join(HERE, "atlas")
BVP = sys.argv[2] if len(sys.argv) > 2 else os.path.join(HERE, "runs", "blockwise_vs_pooled")
AN = os.path.join(ATLAS, "analysis"); TB = os.path.join(AN, "tables"); FA = os.path.join(HERE, "figs", "appendix"); TX = os.path.join(HERE, "tables"); os.makedirs(FA, exist_ok=True); os.makedirs(TX, exist_ok=True)
plt.rcParams.update({"font.family": "Helvetica", "font.size": 7, "axes.spines.top": False, "axes.spines.right": False, "axes.linewidth": 0.6, "pdf.fonttype": 42})
KS = (1, 2, 4, 8); SEED, NBOOT = 20261115, 10000; POOLED = ("1_ViT", "final_output"); GAIN, LOSS = "#0072B2", "#E69F00"
NAME = {"1_ViT": "ViT", "2_DINOv2": "DINOv2", "3_CLIP": "CLIP", "4_ResNet50": "ResNet-50", "5_SigLIP2": "SigLIP 2", "6_BioCLIP": "BioCLIP", "7_DINOv3": "DINOv3", "8_Qwen3VL": "Qwen3-VL tower", "9_FLAVA": "FLAVA"}
COL = {"1_ViT": "#0072B2", "2_DINOv2": "#D55E00", "3_CLIP": "#009E73", "4_ResNet50": "#CC79A7", "5_SigLIP2": "#E69F00", "6_BioCLIP": "#56B4E9", "7_DINOv3": "#000000", "8_Qwen3VL": "#8C5A2B", "9_FLAVA": "#7F7F7F"}
ENCS = list(NAME); sl = lambda l: "output" if l == "final_output" else ("stem" if "stem" in l else "L" + str(int(l.split("_")[1]))); tx = lambda s: str(s).replace("_", r"\_").replace("%", r"\%").replace("&", r"\&")
lab = lambda e, l: f"{NAME[e]} {sl(l)}"

AGG = pd.read_csv(os.path.join(TB, "aggregate_by_layer.csv")); A = AGG.set_index(["encoder", "layer", "K"]); row = lambda e, l, k: A.loc[(e, l, k)]
LAYERS = {e: list(AGG[(AGG.encoder == e) & (AGG.K == 1)].layer) for e in ENCS}; hidden = lambda e: [l for l in LAYERS[e] if l != "final_output"]
BYD = pd.concat([pd.read_csv(os.path.join(TB, f"by_dataset_top{k}.csv")) for k in KS]); D = BYD.set_index(["dataset", "encoder", "layer", "K"]); DS = list(BYD.dataset.drop_duplicates()); dsn = lambda d: d[3:].replace("_", " ")
STAB = pd.read_csv(os.path.join(TB, "consistency_across_depth.csv")); AUD = json.load(open(os.path.join(AN, "audit.json"))); N = int(AUD["queries"])
PQ = pd.read_csv(os.path.join(TB, "per_query_topk.csv.gz"), usecols=["dataset", "query", "encoder", "layer", "K", "transition_vs_final_output"]); PQ = PQ[PQ.layer != "final_output"]   # every layer against the final (pooled) output of its own encoder
PQ["d"] = PQ.transition_vs_final_output.map({"0 to >=1": 1, ">=1 to 0": -1}).fillna(0).astype(np.int8)

# ------------------------------------------------------------------ net coverage change with a paired bootstrap over queries
C = np.random.default_rng(SEED).multinomial(N, np.ones(N) / N, size=NBOOT).astype(np.float32); RANK = {}
for k in KS:
    g = PQ[PQ.K == k]; M = g.pivot_table(index=["encoder", "layer"], columns=["dataset", "query"], values="d"); est = M.values.astype(np.float32) @ C.T / N * 100
    R = pd.DataFrame({"net": M.values.mean(1) * 100, "lo": np.percentile(est, 2.5, axis=1), "hi": np.percentile(est, 97.5, axis=1)}, index=M.index)
    ds = g.groupby(["encoder", "layer", "dataset"]).d.mean().unstack() * 100; R["pos_ds"] = (ds > 0).sum(1); R["neg_ds"] = (ds < 0).sum(1)
    R["gained"] = (M.values == 1).sum(1); R["lost"] = (M.values == -1).sum(1); R = R.reset_index(); R["depth"] = [(hidden(e).index(l) + 1) / len(hidden(e)) for e, l in zip(R.encoder, R.layer)]
    RANK[k] = R.sort_values(["net", "lo"], ascending=False).reset_index(drop=True)
pd.concat([RANK[k].assign(K=k) for k in KS]).to_csv(os.path.join(TX, "layer_gain_ranking.csv"), index=False)

# ------------------------------------------------------------------ figures
def letter(a, t, x=-0.1, y=1.05): a.text(x, y, t, transform=a.transAxes, fontsize=9, fontweight="bold")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import biomor_style as BS
BESTL = {k: {e: RANK[k][RANK[k].encoder == e].iloc[0] for e in ENCS} for k in KS}          # layer with the largest net coverage change against the own pooled output
pcov = lambda e, k: float(row(e, "final_output", k).pct_at_least_one_same_class)
MAIN = [e for e in ("4_ResNet50", "1_ViT", "3_CLIP", "9_FLAVA", "5_SigLIP2", "8_Qwen3VL") if e in ENCS]          # encoders of the main text; the appendix keeps all nine
def within_bars(ax, k, fs=5.2, names=True, ENCS=ENCS):
    for i, e in enumerate(ENCS):
        r = BESTL[k][e]; p = pcov(e, k); v = float(row(e, r.layer, k).pct_at_least_one_same_class)
        for x, h, c in ((i - 0.2, p, BS.P["lavender"]), (i + 0.2, v, BS.P["green"])): BS.bar(ax, x, h - Y0, 0.38, c, bottom=Y0); BS.vlabel(ax, x, h, f"{h:.1f}", fs=fs, pad=0.4)
    ax.set_xticks(range(len(ENCS)), [f"{NAME[e]}\nblock {int(BESTL[k][e].layer.split('_')[1])}" for e in ENCS] if names else [f"{int(BESTL[k][e].layer.split('_')[1])}" for e in ENCS]); ax.set_xlim(-0.6, len(ENCS) - 0.4)
Y0 = 50; HAND = lambda: [BS.patch(BS.P["lavender"], "Pooled output of the encoder"), BS.patch(BS.P["green"], "Best hidden layer of the same encoder")]
def fig_within():
    """main text: coverage of the pooled output and of the best hidden layer of the same encoder, one reference per query"""
    with plt.rc_context(BS.RC):
        fig, ax = plt.subplots(figsize=(6.9, 1.75)); within_bars(ax, 1, fs=5.6, ENCS=MAIN); ax.set_ylim(Y0, 96); ax.set_ylabel("Queries with a same-class\nreference (%)"); BS.top_legend(ax, HAND(), ncol=2)
        for ext, kw in ((".pdf", {}), (".png", {"dpi": 600})): fig.savefig(os.path.join(HERE, "figs", "fig_within_retrieval" + ext), bbox_inches="tight", pad_inches=0.02, **kw)
        plt.close(fig)
        fig, axs = plt.subplots(4, 1, figsize=(6.9, 6.4))
        for a, k in zip(axs, KS): within_bars(a, k); a.set_ylim(Y0, 112); a.set_ylabel(f"Top {k}\ncoverage (%)")
        BS.top_legend(axs[0], HAND(), ncol=2); fig.tight_layout(h_pad=0.8)
        fig.savefig(os.path.join(FA, "app_within_coverage.pdf"), bbox_inches="tight", pad_inches=0.02); plt.close(fig)

def fig_dataset_maps(col, scale, cmap, vmin, vmax, label, fname, center=False):
    """five datasets (rows) by four budgets (columns); in each panel encoders by depth"""
    fig, axs = plt.subplots(len(DS), 4, figsize=(7.2, 8.6), sharey=True)
    for i, d in enumerate(DS):
        for j, k in enumerate(KS):
            a = axs[i, j]
            for r_, e in enumerate(ENCS):
                hid = hidden(e); v = np.array([[scale * D.loc[(d, e, l, k)][col] for l in hid]]); im = a.pcolormesh(np.linspace(0, 1, len(hid) + 1), [r_, r_ + 0.86], v, cmap=cmap, vmin=vmin, vmax=vmax, edgecolors="white", linewidth=0.2)
                a.pcolormesh([1.04, 1.13], [r_, r_ + 0.86], np.array([[scale * D.loc[(d, e, "final_output", k)][col]]]), cmap=cmap, vmin=vmin, vmax=vmax)
                jb = int(np.argmax(v[0])); a.scatter([(jb + 0.5) / len(hid)], [r_ + 0.43], s=5, c="white", edgecolors="#111", linewidths=0.4, zorder=4)
            a.set_ylim(len(ENCS), -0.1); a.set_yticks([r_ + 0.43 for r_ in range(len(ENCS))], [NAME[e] for e in ENCS], fontsize=5.6); a.set_xticks([0.07, 0.5, 0.9, 1.085], ["First", "Mid", "Last", "Out"] if i == len(DS) - 1 else []); a.tick_params(length=0, labelsize=5.4)
            for s_ in a.spines.values(): s_.set_visible(False)
            if i == 0: a.set_title(f"Top {k}", fontsize=7.2)
            if j == 3: a.text(1.2, len(ENCS) / 2, dsn(d), rotation=270, va="center", ha="left", fontsize=6.6, fontweight="bold")
    cb = fig.colorbar(im, ax=axs, fraction=0.016, pad=0.07, location="bottom"); cb.set_label(label, fontsize=6.6); cb.outline.set_visible(False); cb.ax.tick_params(labelsize=6); cb.solids.set_rasterized(False)
    fig.subplots_adjust(left=0.115, right=0.955, top=0.965, bottom=0.115, wspace=0.07, hspace=0.1); fig.savefig(os.path.join(FA, fname)); plt.close(fig)

fig_within()
fig_dataset_maps("fraction_of_slots_same_class", 100, "cividis", 15, 100, "Reference slots with the class of the query (%)", "app_dataset_same_class.pdf")
lim = 30; fig_dataset_maps("net_coverage_pp_vs_final_output", 1, "PuOr", -lim, lim, "Net coverage change against the pooled output of the same encoder (pp); purple = gain, orange = loss", "app_dataset_net_coverage.pdf")
COPY = {"fig1_same_class_by_encoder_and_depth.pdf": "app_depth_map.pdf", "fig3_adjacent_layer_replacement.pdf": "app_adjacent_replacement.pdf",
        "fig5_vit_layer10_vs_pooled.pdf": "app_vit_layer10.pdf", "fig6_layerwise_same_class.pdf": "app_layerwise_same_class.pdf", "fig7_layerwise_replacement.pdf": "app_layerwise_replacement.pdf", "fig8_settling_and_datasets.pdf": "app_settling_datasets.pdf",
        "supp_transitions_vs_final_output.pdf": "app_transitions_vs_final.pdf"}
for s_, d_ in COPY.items(): shutil.copyfile(os.path.join(AN, "figures", s_), os.path.join(FA, d_))
shutil.copyfile(os.path.join(ATLAS, "assets", "example_query_all_layers.jpg"), os.path.join(FA, "app_example_query.jpg"))

# ------------------------------------------------------------------ numbers
adj = {k: 100 * np.mean([row(e, l, k).fraction_replaced_vs_previous_layer for e in ENCS for l in hidden(e)[1:]]) for k in KS}; idn = {k: np.mean([row(e, l, k).pct_identical_order_vs_previous_layer for e in ENCS for l in hidden(e)[1:]]) for k in KS}
sig_pos = {k: int((RANK[k].lo > 0).sum()) for k in KS}; sig_neg = {k: int((RANK[k].hi < 0).sum()) for k in KS}; ncand = len(RANK[1])
first_third = {k: RANK[k][[isinstance(x, float) and x <= 1 / 3 + 1e-9 for x in RANK[k].depth]] for k in KS}
H = json.load(open(os.path.join(BVP, "holdout_result.json"))); CFG = json.load(open(os.path.join(BVP, "config_locked.json"))); HO = pd.read_csv(os.path.join(BVP, "aggregate_holdout.csv")); DV = pd.read_csv(os.path.join(BVP, "aggregate_development.csv"))
cand = CFG["candidate"]["representation"]; ci = H["ci95_date_cluster_bootstrap_pp"]; dv = CFG["development"]
vdev = DV[(DV.encoder == "vit") & DV.block.notna() & (DV.representation != "vit|native")]; vbest = vdev.loc[vdev.net_coverage_pp.idxmax()]
r1 = RANK[1]; r4 = RANK[4]; all5 = {k: RANK[k][RANK[k].pos_ds == len(DS)] for k in KS}
ENC_ABOVE = {k: {NAME[e]: int((RANK[k][RANK[k].encoder == e].lo > 0).sum()) for e in ENCS} for k in KS}
M6 = {k: RANK[k][RANK[k].encoder.isin(MAIN)] for k in KS}; FT6 = {k: M6[k][M6[k].depth <= 1 / 3 + 1e-9] for k in KS}
NUM = {"N": N, "main_text_six_encoders": {"hidden_layers": int(len(M6[1])), "above_own_pooled": {k: int((M6[k].lo > 0).sum()) for k in KS}, "below_own_pooled": {k: int((M6[k].hi < 0).sum()) for k in KS}, "first_third_n": int(len(FT6[1])), "first_third_max_net": {k: round(float(FT6[k].net.max()), 1) for k in KS}}, "pooled_cov": {k: {NAME[e]: round(pcov(e, k), 1) for e in ENCS} for k in KS}, "best_within": {k: {NAME[e]: [sl(BESTL[k][e].layer), round(float(BESTL[k][e].net), 1), round(float(BESTL[k][e].lo), 1), round(float(BESTL[k][e].hi), 1), int(BESTL[k][e].gained), int(BESTL[k][e].lost)] for e in ENCS} for k in KS}, "layers_above_by_encoder": ENC_ABOVE, "adj": {k: round(adj[k], 1) for k in KS}, "ident": {k: round(float(idn[k]), 1) for k in KS},
       "sig_pos": sig_pos, "sig_neg": sig_neg, "candidates": ncand, "top": {k: [(lab(r.encoder, r.layer), round(r.net, 1), round(float(r.lo), 1), round(float(r.hi), 1), int(r.pos_ds), int(r.gained), int(r.lost)) for r in RANK[k].head(6).itertuples()] for k in KS},
       "bottom": {k: [(lab(r.encoder, r.layer), round(r.net, 1), round(float(r.lo), 1), round(float(r.hi), 1), int(r.neg_ds)) for r in RANK[k].tail(6).itertuples()] for k in KS},
       "positive_on_all_datasets": {k: [(lab(r.encoder, r.layer), round(r.net, 1)) for r in all5[k].itertuples()] for k in KS}, "first_third_max_net": {k: round(float(first_third[k].net.max()), 1) for k in KS}, "first_third_n": int(len(first_third[1])),
       "vit_best_dev": [vbest.representation, int(vbest.K), round(float(vbest.net_coverage_pp), 1)]}
json.dump(NUM, open(os.path.join(TX, "appendix_numbers.json"), "w"), indent=1)

# ------------------------------------------------------------------ LaTeX
PCT, PPU, BEST = "\\%", "\\pp", "\\best "
f1 = lambda v: f"{v:.1f}"; pp = lambda v: f"{v:+.1f}".replace("-", "$-$").replace("+", "$+$"); ciq = lambda r: f"{pp(r.net)} ({pp(r.lo)} to {pp(r.hi)})"
L = []; w = L.append
w("% Generated by tools/make_appendix.py from the layer-wise reference atlas. Do not edit by hand.")
w(r"\clearpage" + "\n" + r"\appendix" + "\n" + r"\setcounter{figure}{0}\renewcommand{\thefigure}{A\arabic{figure}}\setcounter{table}{0}\renewcommand{\thetable}{A\arabic{table}}")
w(r"\section{Layer-wise retrieval analysis with nine encoders}" + "\n" + r"\label{app:atlas}" + "\n")
w(f"This appendix reports the retrieval analysis behind Sec.~\\ref{{sec:latent}} and Sec.~\\ref{{sec:discussion}} in full: every layer of nine encoders, the five datasets, and budgets of 1, 2, 4 and 8 references. It measures retrieval only, and every layer is compared with the pooled output of its own encoder. Appendix~\\ref{{app:vlm}} gives the test with the VLM.\n")
w(r"\subsection{Setup}" + "\n")
w(f"\\textbf{{Disjoint queries and references.}} For each of the five datasets we use 100 test queries and a separate, class-balanced reference bank of 100 images (seed 42), {N} queries in total. The queries are the AgEval pool of Sec.~\\ref{{sec:setup}}, completed to 100 images; the bank is drawn from the rest of the dataset. "
  "No image is in both sets, and no bank image is byte-identical to a query. This differs from the leave-one-out pools of the main experiments, so the numbers of this appendix are not the numbers of Fig.~\\ref{fig:layers}. Queries and references are not separated by capture date; Sec.~\\ref{app:holdout} adds a date-disjoint check for the ViT.\n")
w("\\textbf{Encoders and layers.} Table~\\ref{tab:app-enc} lists the nine encoders. A layer is read at the CLS token where the model has one, at the mean of the patch tokens otherwise, and at the averaged feature map for ResNet-50. The final layer normalization of each transformer is applied before the readout. "
  f"Together the encoders have {int(AUD['layers_without_final_output'])} layers; each encoder also has a \\emph{{final output}}, the pooled vector it exposes for retrieval. The pooled ViT output is the retriever of assisted few-shot learning~\\cite{{arshad2025assisted}}. The baseline of a layer is always the final output of its own encoder; outputs of different encoders are not compared. All vectors are normalized to unit length and ranked by cosine similarity.\n")
w("\\textbf{Measures.} For a budget $K$, \\emph{slots matching} is the share of the top-$K$ reference slots filled by a reference of the query's class, and \\emph{coverage} is the share of queries with at least one such reference. \\emph{Replaced} is the share of the top-$K$ slots that hold a reference the previous layer did not retrieve. "
  "Against the pooled output of the same encoder at the same $K$, a query is \\emph{gained} if the layer supplies a same-class reference where the pooled list has none, and \\emph{lost} if the layer loses every same-class reference of the pooled list. The \\emph{net coverage change} is the gained rate minus the lost rate over all paired queries. "
  f"Intervals are 95\\% paired bootstrap intervals over the {N} queries ({NBOOT:,} resamples).\n".replace(",", "{,}"))
w(r"\begin{table}[h]" + "\n" + r"\centering" + "\n" + r"\caption{\textbf{Encoders of the layer-wise analysis.} Readout of a layer, number of layers, and the final output.}" + "\n" + r"\label{tab:app-enc}" + "\n" + r"\scriptsize\setlength{\tabcolsep}{2.2pt}" + "\n" + r"\begin{tabular}{@{}llcl@{}}" + "\n" + r"\toprule" + "\n" + r"Encoder & Layer readout & Layers & Final output \\" + "\n" + r"\midrule")
for nm_, ro, nl, fo in (("ViT-B/16~\\cite{dosovitskiy2021image}", "CLS", 12, "pooled output"), ("DINOv2 ViT-B/14~\\cite{oquab2024dinov2}", "CLS", 12, "last CLS token"), ("CLIP ViT-B/16~\\cite{radford2021learning}", "CLS", 12, "projected embedding"), ("ResNet-50~\\cite{he2016deep}", "feature map", "stem + 16", "pooled output"),
                        ("SigLIP 2 ViT-B/16~\\cite{tschannen2025siglip2}", "patch mean", 12, "pooled embedding"), ("BioCLIP ViT-B/16~\\cite{stevens2024bioclip}", "CLS", 12, "projected embedding"), ("DINOv3 ViT-B/16~\\cite{simeoni2025dinov3}", "CLS", 12, "last CLS token"),
                        ("Qwen3-VL-2B vision tower~\\cite{qwen3vl2025}", "patch mean", 24, "merged visual tokens"), ("FLAVA image encoder~\\cite{singh2022flava}", "CLS", 12, "projected embedding")):
    w(f"{nm_} & {ro} & {nl} & {fo} \\\\")
w(r"\bottomrule" + "\n" + r"\end{tabular}" + "\n" + r"\end{table}" + "\n")
w(r"\subsection{Same-class retrieval and coverage}" + "\n")
def wtxt(k): return "; ".join(f"{NAME[e]} {sl(BESTL[k][e].layer)} {pp(BESTL[k][e].net)}\\pp{{}} ({pp(BESTL[k][e].lo)} to {pp(BESTL[k][e].hi)})" for e in ENCS if BESTL[k][e].lo > 0)
w(f"Figure~\\ref{{fig:app-depth}} maps the share of matching slots over encoders and depth. Figure~\\ref{{fig:app-cov}} compares, for each encoder and budget, the coverage of the pooled output with the coverage of the hidden layer of the same encoder that has the largest net coverage change. "
  f"The hidden layers above the pooled output of their encoder with an interval that excludes zero are, as the best layer of an encoder: at $K=1$, {wtxt(1)}; at $K=2$, {wtxt(2)}; at $K=4$, {wtxt(4) or 'none'}; at $K=8$, {wtxt(8) or 'none'}. "
  "The last block of DINOv2 and of DINOv3 returns the references of the final output, and no earlier block of these encoders is above it.\n")
for f, labn, cap in (("app_depth_map.pdf", "depth", f"\\textbf{{Same-class slots by encoder and depth.}} Share of the top-$K$ slots filled by a same-class reference, for every layer (one cell per layer) and for the final output (separate cell). The dot marks the best layer of each encoder. {N} queries."),
                     ("app_within_coverage.pdf", "cov", f"\\textbf{{Coverage of the pooled output and of the best hidden layer of the same encoder,}} for budgets of 1, 2, 4 and 8 references. Share of the {N} queries with at least one same-class reference. The block under each encoder is its hidden layer with the largest net coverage change. The axis starts at 50\\%.")):
    w(r"\begin{figure*}[t]" + "\n" + r"\centering" + "\n" + f"\\includegraphics[width=\\linewidth]{{figs/appendix/{f}}}" + "\n" + f"\\caption{{{cap}}}" + "\n" + f"\\label{{fig:app-{labn}}}" + "\n" + r"\end{figure*}" + "\n")
w(r"\subsection{Turnover between layers}" + "\n")
w(f"Between adjacent layers {', '.join(f'{f1(adj[k])}{PCT} ($K={k}$)' for k in KS)} of the reference slots are replaced (Fig.~\\ref{{fig:app-adj}}), and the ordered list is identical for {', '.join(f'{f1(idn[k])}{PCT}' for k in KS)} of the queries. "
  f"The lists do not reach a plateau: at $K=4$ the ordered list takes its last-layer form only at the last layer for {f1(STAB[STAB.K == 4].pct_settled_only_at_last_layer.min())}\\% to {f1(STAB[STAB.K == 4].pct_settled_only_at_last_layer.max())}\\% of the queries, depending on the encoder (Fig.~\\ref{{fig:app-settle}}a). This describes agreement between layers and not robustness to changes of the input image.\n")
w(r"\begin{figure*}[t]" + "\n" + r"\centering" + "\n" + r"\includegraphics[width=\linewidth]{figs/appendix/app_adjacent_replacement.pdf}" + "\n" + r"\caption{\textbf{Replacement between adjacent layers.} Share of the top-$K$ slots that a layer fills with a reference the previous layer did not retrieve.}" + "\n" + r"\label{fig:app-adj}" + "\n" + r"\end{figure*}" + "\n")
w(r"\subsection{Every layer of every encoder}" + "\n")
notes = []
for e in ENCS:
    hid = hidden(e); bl = {k: max(hid, key=lambda l: (row(e, l, k).fraction_of_slots_same_class, -hid.index(l))) for k in KS}; churn = max(hid[1:], key=lambda l: row(e, l, 4).fraction_replaced_vs_previous_layer)
    peak = "the last layer at every budget" if all(bl[k] == hid[-1] for k in KS) else ", ".join(f"{sl(bl[k])} ($K={k}$)" for k in KS); notes.append(f"\\textbf{{{NAME[e]}}}: best at {peak}; largest replacement at {sl(churn)} ({f1(100 * row(e, churn, 4).fraction_replaced_vs_previous_layer)}\\% of the top-4 slots)")
w("Figures~\\ref{fig:app-lw} and~\\ref{fig:app-lwr} give the share of matching slots and the replacement at every layer, with real layer numbers. " + ". ".join(notes) + ".\n")
for f, labn, cap in (("app_layerwise_same_class.pdf", "lw", "\\textbf{Same-class references at every layer.} One panel per encoder, one line per budget. Diamonds: final output. Share of the top-$K$ slots with the class of the query."),
                     ("app_layerwise_replacement.pdf", "lwr", "\\textbf{Replacement at every layer.} Share of the top-$K$ slots that the previous layer did not retrieve. Same layout as Fig.~\\ref{fig:app-lw}.")):
    w(r"\begin{figure*}[t]" + "\n" + r"\centering" + "\n" + f"\\includegraphics[width=0.9\\linewidth]{{figs/appendix/{f}}}" + "\n" + f"\\caption{{{cap}}}" + "\n" + f"\\label{{fig:app-{labn}}}" + "\n" + r"\end{figure*}" + "\n")
w(r"\subsection{Results by dataset}" + "\n")
w("Figures~\\ref{fig:app-ds} and~\\ref{fig:app-dsnet} repeat the depth maps for each dataset, as the share of matching slots and as the net coverage change against the pooled output of the same encoder. Figure~\\ref{fig:app-settle}b gives the final outputs by dataset. "
  "The first layers of every encoder are below the pooled output of that encoder on every dataset.\n")
for f, labn, cap in (("app_dataset_same_class.pdf", "ds", "\\textbf{Same-class slots by dataset.} Rows: datasets. Columns: budgets. In each panel, encoders by depth as in Fig.~\\ref{fig:app-depth}; 100 queries per dataset."),
                     ("app_dataset_net_coverage.pdf", "dsnet", "\\textbf{Net coverage change against the pooled output of the same encoder, by dataset.} Same layout as Fig.~\\ref{fig:app-ds}. Purple: the layer gains more queries than it loses. Orange: it loses more.")):
    w(r"\begin{figure*}[p]" + "\n" + r"\centering" + "\n" + f"\\includegraphics[width=0.93\\linewidth]{{figs/appendix/{f}}}" + "\n" + f"\\caption{{{cap}}}" + "\n" + f"\\label{{fig:app-{labn}}}" + "\n" + r"\end{figure*}" + "\n")
w(r"\begin{figure*}[t]" + "\n" + r"\centering" + "\n" + r"\includegraphics[width=\linewidth]{figs/appendix/app_settling_datasets.pdf}" + "\n" + r"\caption{\textbf{Late settling and dataset effects.} \textbf{a.} Share of queries whose ordered top-$K$ list takes its last-layer form only at the last layer. \textbf{b.} Share of the top-4 slots with the class of the query at the final output of each encoder, by dataset.}" + "\n" + r"\label{fig:app-settle}" + "\n" + r"\end{figure*}" + "\n")
w(r"\subsection{Largest and lowest gains}" + "\n" + r"\label{app:gains}" + "\n")
w(f"Table~\\ref{{tab:app-gain}} ranks the {ncand} hidden layers by their net coverage change against the pooled output of their own encoder. At $K=1$, {sig_pos[1]} of them are above that output with an interval that excludes zero and {sig_neg[1]} are below it; at $K=2$, {sig_pos[2]} and {sig_neg[2]}; at $K=4$, {sig_pos[4]} and {sig_neg[4]}; at $K=8$, {sig_pos[8]} and {sig_neg[8]}. "
  f"The {NUM['first_third_n']} layers in the first third of their encoder gain little over the pooled output of their encoder (largest net change {pp(NUM['first_third_max_net'][1])}\\pp{{}} at $K=1$ and {pp(NUM['first_third_max_net'][4])}\\pp{{}} at $K=4$). "
  + (f"At $K=1$ the layers with a positive net change on all five datasets are {', '.join(f'{n_} ({pp(v)}{PPU})' for n_, v in NUM['positive_on_all_datasets'][1])}. " if NUM["positive_on_all_datasets"][1] else "")
  + ("No layer is positive on all five datasets at $K=4$. " if not NUM["positive_on_all_datasets"][4] else "")
  + "The ranking is computed on the queries it describes, so the leading layers are candidates and not validated choices.\n")
w(r"\begin{table*}[t]" + "\n" + r"\centering" + "\n" + f"\\caption{{\\textbf{{Largest and lowest net coverage changes against the pooled output of the same encoder,}} in percentage points with 95\\% bootstrap intervals, {N} queries. Datasets: number of the five datasets with a positive (largest) or negative (lowest) net change. Largest gain per budget in green.}}" + "\n" + r"\label{tab:app-gain}" + "\n" + r"\scriptsize\setlength{\tabcolsep}{3.4pt}" + "\n" + r"\begin{tabular}{@{}cl|lcc|lcc@{}}" + "\n" + r"\toprule" + "\n" +
  r" & & \multicolumn{3}{c|}{Largest gains} & \multicolumn{3}{c}{Lowest} \\" + "\n" + r"$K$ & Rank & Layer & Net change (95\% CI) & Datasets & Layer & Net change (95\% CI) & Datasets \\" + "\n" + r"\midrule")
for k in KS:
    T, Bm = RANK[k].head(5), RANK[k].tail(5).iloc[::-1]
    for i in range(5):
        t, bt = T.iloc[i], Bm.iloc[i]; w((f"{k}" if i == 0 else "") + f" & {i + 1} & {lab(t.encoder, t.layer)} & {BEST if i == 0 else ''}{ciq(t)} & {int(t.pos_ds)} & {lab(bt.encoder, bt.layer)} & {ciq(bt)} & {int(bt.neg_ds)} \\\\")
    w(r"\midrule" if k != 8 else r"\bottomrule")
w(r"\end{tabular}" + "\n" + r"\end{table*}" + "\n")
v = {k: row("1_ViT", "layer_10", k) for k in KS}
w(r"\subsection{One layer across budgets and one query across layers}" + "\n")
w("Figure~\\ref{fig:app-v10} follows block 10 of the ViT against its pooled output. " + "; ".join(f"at $K={k}$ it gains {int(v[k].n_0_to_ge1_vs_final_output)} queries and loses {int(v[k].n_ge1_to_0_vs_final_output)} ({pp(v[k].net_coverage_pp_vs_final_output)}\\pp)" for k in KS)
  + ". A small net change is the balance of a much larger exchange of queries, and the room for a gain shrinks as the budget grows. Figure~\\ref{fig:app-ex} shows the four references of one query at every layer of every encoder.\n")
w(r"\begin{figure*}[t]" + "\n" + r"\centering" + "\n" + r"\includegraphics[width=\linewidth]{figs/appendix/app_vit_layer10.pdf}" + "\n" + r"\caption{\textbf{ViT block 10 against the pooled ViT output.} \textbf{a.} The four paired outcomes as shares of the queries (numbers are query counts). \textbf{b.} Coverage of the two lists; labels give the net change. \textbf{c.} Share of slots replaced and share of identical ordered lists.}" + "\n" + r"\label{fig:app-v10}" + "\n" + r"\end{figure*}" + "\n")
w(r"\begin{figure*}[p]" + "\n" + r"\centering" + "\n" + r"\includegraphics[height=0.86\textheight]{figs/appendix/app_example_query.jpg}" + "\n" + r"\caption{\textbf{The four references of one query at every layer of every encoder.} Columns are layers. An orange frame marks a reference that the previous layer did not retrieve. A blue disc with a check marks a reference of the query's class. The number below a column counts such references.}" + "\n" + r"\label{fig:app-ex}" + "\n" + r"\end{figure*}" + "\n")
w(r"\subsection{Date-disjoint check}" + "\n" + r"\label{app:holdout}" + "\n")
vk = int(vbest.K); vb_ = int(vbest.block)
w(f"The split above does not separate capture dates. For DeepWeeds, the only dataset with a capture date, we ran a separate check with date-disjoint banks for the ViT of the published pipeline. On a development split with 180 queries and ten banks of 99 images, we compared every hidden layer of the ViT with its pooled output at every budget. "
  f"The largest net coverage change of a ViT layer is {pp(NUM['vit_best_dev'][2])}\\pp{{}} (block {vb_}, $K={vk}$). The gain of a hidden layer of the ViT over its pooled output is therefore smaller under date separation than on the split above, where block 10 gains {pp(BESTL[1]['1_ViT'].net)}\\pp{{}} at $K=1$. The check covers one dataset and is exploratory.\n")
w(r"\subsection{Paired outcomes at every depth}" + "\n")
w("Figure~\\ref{fig:app-trf} gives the gained and lost rates of every layer against the final output of its own encoder.\n")
w(r"\begin{figure*}[t]" + "\n" + r"\centering" + "\n" + r"\includegraphics[width=\linewidth]{figs/appendix/app_transitions_vs_final.pdf}" + "\n" + r"\caption{\textbf{Gained and lost rates against the final output of the same encoder,} by depth and budget. Solid, upward: gained. Dashed, downward: lost.}" + "\n" + r"\label{fig:app-trf}" + "\n" + r"\end{figure*}" + "\n")
w(r"\subsection{Checks}" + "\n")
i_ = AUD["issues"]
w(f"All {AUD['encoder_layer_lists_found']:,} lists (query, encoder, layer) are complete. ".replace(",", "{,}") + f"Lists that are not an ordered prefix of their top-8 list: {i_['not_a_prefix_of_top8']}; references that are query images: {i_['reference_that_is_a_query_image']}; byte-identical query and reference pairs: {AUD['md5_shared_by_a_query_and_a_reference']}. "
  "The four paired outcomes sum to the number of queries for every layer and budget, their rates reproduce the coverage difference, coverage never falls when the budget grows, and identical rankings give identical measures.\n")
open(os.path.join(HERE, "appendix.tex"), "w").write("\n".join(L) + "\n"); print(json.dumps(NUM, indent=1)[:2500])
