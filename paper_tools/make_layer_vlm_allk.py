"""Builds the all-budget downstream layer test of the paper from runs/layer_vlm_allk (pilot_layer_vlm_allk.py and its report):
figs/fig_vlm_allk.{pdf,png} and figs/fig_vlm_k.{pdf,png} (main text), figs/appendix/fig_vlm_allk_all.{pdf,png},
tables/tab_vlm_allk.tex (main text), appendix_vlm_allk.tex and tables/layer_vlm_allk_numbers.json.
Usage: python tools/make_layer_vlm_allk.py [RUN_DIR]     (default ../../code/ageval-dinov2-pilot/runs/layer_vlm_allk)
"""
import json, os, sys
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__))); RUN = sys.argv[1] if len(sys.argv) > 1 else os.path.join(HERE, "runs", "layer_vlm_allk")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import biomor_style as BS
ENC = ["ResNet50", "ViT", "CLIP", "FLAVA", "SigLIP2", "Qwen3VL"]; KS = [1, 2, 4, 8]          # ordered by publication year, old to new
YEAR = {"ResNet50": 2016, "ViT": 2021, "CLIP": 2021, "FLAVA": 2022, "SigLIP2": 2025, "Qwen3VL": 2025}
NM = {"ViT": "ViT", "CLIP": "CLIP", "ResNet50": "ResNet-50", "SigLIP2": "SigLIP 2", "Qwen3VL": "Qwen3-VL tower", "FLAVA": "FLAVA"}; SH = {**NM, "Qwen3VL": "Qwen3-VL"}
R = pd.read_csv(os.path.join(RUN, "results_by_condition.csv"), keep_default_na=False, na_values=[""]); R["encoder"] = R.encoder.fillna(""); R["K"] = R.K.astype(int)
EL = pd.read_csv(os.path.join(RUN, "encoder_level.csv")); _q = pd.read_csv(os.path.join(RUN, "per_query_predictions.csv"), dtype=str, keep_default_na=False); _q = _q[_q.condition == "zero_shot"]; zs = float(100 * (_q.pred == _q.true_label).mean()); n = int(R.n.iloc[0]); NC = len(R) + 1
PO = R[(R.layer == "final_output")].set_index(["K", "encoder"]); LY = R[(R.encoder != "") & (R.layer != "final_output")].copy(); LY["ln"] = [int(l.split("_")[1]) for l in LY.layer]
best = LY.loc[LY.groupby(["K", "encoder"]).delta_vs_own_pooled_pp.idxmax()].set_index(["K", "encoder"])
sg = lambda v, d=1: ("$+$" if v >= 0 else "$-$") + f"{abs(v):.{d}f}"

N = {"n": n, "zero_shot": zs, "conditions": NC, "layer_comparisons": int(len(LY)), "holm_min": float(LY.p_holm.min()), "holm_significant": int((LY.p_holm < 0.05).sum()), "by_K": {}, "cells": {}}
for k in KS:
    p = np.array([PO.loc[(k, e)].accuracy for e in ENC]); b = np.array([best.loc[(k, e)].accuracy for e in ENC]); g = LY[LY.K == k]
    N["by_K"][k] = {"pooled_mean": round(float(p.mean()), 2), "pooled_min": float(p.min()), "pooled_max": float(p.max()), "best_hidden_mean": round(float(b.mean()), 2), "best_hidden_min": float(b.min()), "best_hidden_max": float(b.max()),
                    "gain_mean_pp": round(float((b - p).mean()), 2), "encoders_hidden_higher": int((b > p).sum()), "layers_above_ci": int((g.ci_lo > 0).sum()), "layers_below_ci": int((g.ci_hi < 0).sum()), "layers_higher": int((g.delta_vs_own_pooled_pp > 0).sum()), "layers": len(g)}
    for e in ENC:
        x = best.loc[(k, e)]; m = EL[(EL.K == k) & (EL.encoder == e)].iloc[0]
        N["cells"][f"K{k}|{e}"] = {"pooled": float(PO.loc[(k, e)].accuracy), "best_block": int(x.ln), "best": float(x.accuracy), "delta": round(float(x.delta_vs_own_pooled_pp), 2), "ci": [round(float(x.ci_lo), 2), round(float(x.ci_hi), 2)], "wl": [int(x.wins), int(x.losses)], "p": float(x.p), "holm": float(x.p_holm),
                                   "mean_of_five_delta": round(float(m.delta_pp), 2), "mean_of_five_ci": [round(float(m.ci_lo), 2), round(float(m.ci_hi), 2)], "layers_above": int(m.layers_above)}
N["cells_hidden_higher"] = int(sum(v["delta"] > 0 for v in N["cells"].values())); N["cells_total"] = len(N["cells"])
N["encoders_rising_with_K"] = {"pooled": int(sum(all(PO.loc[(KS[i + 1], e)].accuracy > PO.loc[(KS[i], e)].accuracy for i in range(3)) for e in ENC)), "best_hidden": int(sum(all(best.loc[(KS[i + 1], e)].accuracy > best.loc[(KS[i], e)].accuracy for i in range(3)) for e in ENC))}
json.dump(N, open(os.path.join(HERE, "tables", "layer_vlm_allk_numbers.json"), "w"), indent=1)

# ------------------------------------------------------------------ figures (bar style of the paper)
BS.apply(); C_POOL, C_HID = BS.P["lavender"], BS.P["green"]; L_POOL, L_HID = BS.LINE["sft"], BS.LINE["grpo"]; Y0 = 40
def save(fig, path):
    fig.savefig(path + ".pdf", bbox_inches="tight", pad_inches=0.02); fig.savefig(path + ".png", dpi=600, bbox_inches="tight", pad_inches=0.02); plt.close(fig)
ZH = plt.Line2D([], [], color="#7F7F7F", lw=0.9, ls=":", label=f"Zero-shot, {zs:.1f}%")
fig, axs = plt.subplots(2, 2, figsize=(6.9, 3.25), sharey=True)
for ax, k in zip(axs.ravel(), KS):
    for i, e in enumerate(ENC):
        for x, v, c in ((i - 0.2, PO.loc[(k, e)].accuracy, C_POOL), (i + 0.2, best.loc[(k, e)].accuracy, C_HID)): BS.bar(ax, x, v - Y0, 0.38, c, bottom=Y0); BS.vlabel(ax, x, v, f"{v:.1f}", fs=5.0, pad=0.4)
    ax.axhline(zs, color="#7F7F7F", lw=0.9, ls=":", zorder=1); ax.set_xticks(range(len(ENC)), [f"{SH[e]}\nblock {int(best.loc[(k, e)].ln)}" for e in ENC], fontsize=5.4); ax.set_ylim(Y0, 82); ax.set_xlim(-0.6, len(ENC) - 0.4)
    ax.set_title(f"{k} reference{'s' if k > 1 else ''} per query", fontsize=6.8, pad=2)
for ax in axs[:, 0]: ax.set_ylabel("VLM accuracy (%)")
fig.legend(handles=[BS.patch(C_POOL, "Assisted few-shot learning (pooled output)"), BS.patch(C_HID, "Assisted latent few-shot learning (best of five hidden layers)"), ZH], loc="lower center", bbox_to_anchor=(0.5, 0.985), ncol=3, frameon=False, handlelength=1.6, columnspacing=1.2)
fig.tight_layout(h_pad=0.9, w_pad=0.6); save(fig, os.path.join(HERE, "figs", "fig_vlm_allk"))

# accuracy against the number of references: mean over the six encoders, band = range over the encoders
fig, ax = plt.subplots(figsize=(3.3, 1.85)); xs = np.arange(len(KS))
for key, col, lab, off in (("pooled", L_POOL, "Assisted (pooled output)", (11, -6)), ("best_hidden", L_HID, "Assisted latent (hidden layer)", (-9, 4))):
    m = [N["by_K"][k][f"{key}_mean"] for k in KS]; lo = [N["by_K"][k][f"{key}_min"] for k in KS]; hi = [N["by_K"][k][f"{key}_max"] for k in KS]
    ax.fill_between(xs, lo, hi, color=col, alpha=0.18, lw=0); ax.plot(xs, m, color=col, lw=1.4, marker="o", ms=3.4, label=lab)
    for x, v in zip(xs, m): ax.annotate(f"{v:.1f}", (x, v), textcoords="offset points", xytext=off, ha="center", fontsize=5.4, color=BS.INK)
ax.axhline(zs, color="#7F7F7F", lw=0.9, ls=":"); ax.set_xticks(xs, [str(k) for k in KS]); ax.set_xlabel("References per query ($K$)"); ax.set_ylabel("VLM accuracy (%)"); ax.set_ylim(48, 74); ax.set_xlim(-0.3, len(KS) - 0.6)
h, _ = ax.get_legend_handles_labels(); ax.legend(handles=h + [ZH], loc="upper left", frameon=False, handlelength=1.6, borderaxespad=0.1)
save(fig, os.path.join(HERE, "figs", "fig_vlm_k"))

# appendix: every tested condition
fig, axs = plt.subplots(4, 1, figsize=(6.9, 6.4), sharey=True)
for ax, k in zip(axs, KS):
    xt, xl = [], []
    for i, e in enumerate(ENC):
        g = LY[(LY.K == k) & (LY.encoder == e)].sort_values("ln"); x0 = i * 7; v = PO.loc[(k, e)].accuracy
        BS.bar(ax, x0, v - Y0, 0.9, C_POOL, bottom=Y0); BS.vlabel(ax, x0, v, f"{v:.1f}", fs=4.6, pad=0.4); xt.append(x0); xl.append("P")
        for j, r in enumerate(g.itertuples(), 1): BS.bar(ax, x0 + j, r.accuracy - Y0, 0.9, C_HID, bottom=Y0); BS.vlabel(ax, x0 + j, r.accuracy, f"{r.accuracy:.1f}", fs=4.6, pad=0.4); xt.append(x0 + j); xl.append(str(r.ln))
        ax.text(x0 + 2.5, 83.5, NM[e], ha="center", va="bottom", fontsize=6.0)
    ax.axhline(zs, color="#7F7F7F", lw=0.9, ls=":", zorder=1); ax.set_xticks(xt, xl, fontsize=5.0); ax.set_ylim(Y0, 83); ax.set_xlim(-1, len(ENC) * 7 - 1); ax.set_ylabel(f"$K={k}$\nVLM accuracy (%)")
fig.legend(handles=[BS.patch(C_POOL, "Pooled output (P)"), BS.patch(C_HID, "Hidden layer (block number)"), ZH], loc="lower center", bbox_to_anchor=(0.5, 0.99), ncol=3, frameon=False, handlelength=1.6, columnspacing=1.2)
fig.tight_layout(h_pad=1.0); os.makedirs(os.path.join(HERE, "figs", "appendix"), exist_ok=True); save(fig, os.path.join(HERE, "figs", "appendix", "fig_vlm_allk_all"))

# main text: every tested condition (compact), and accuracy against the budget by dataset
fig, axs = plt.subplots(4, 1, figsize=(6.9, 4.5), sharey=True)
for ax, k in zip(axs, KS):
    xt, xl = [], []
    for i, e in enumerate(ENC):
        g = LY[(LY.K == k) & (LY.encoder == e)].sort_values("ln"); x0 = i * 7; v = PO.loc[(k, e)].accuracy
        BS.bar(ax, x0, v - Y0, 0.9, C_POOL, bottom=Y0); BS.vlabel(ax, x0, v, f"{v:.1f}", fs=4.4, pad=0.5); xt.append(x0); xl.append("P")
        for j, r in enumerate(g.itertuples(), 1): BS.bar(ax, x0 + j, r.accuracy - Y0, 0.9, C_HID, bottom=Y0); BS.vlabel(ax, x0 + j, r.accuracy, f"{r.accuracy:.1f}", fs=4.4, pad=0.5); xt.append(x0 + j); xl.append(str(r.ln))
        if k == KS[0]: ax.text(x0 + 2.5, 88, NM[e], ha="center", va="bottom", fontsize=6.0)
    ax.axhline(zs, color="#7F7F7F", lw=0.9, ls=":", zorder=1); ax.set_xticks(xt, xl, fontsize=4.8); ax.tick_params(axis="x", length=1.5, pad=1); ax.set_ylim(Y0, 86); ax.set_yticks([40, 60, 80]); ax.set_xlim(-1, len(ENC) * 7 - 1); ax.set_ylabel(f"$K={k}$\naccuracy (%)", fontsize=6.0)
fig.legend(handles=[BS.patch(C_POOL, "Pooled output (P)"), BS.patch(C_HID, "Hidden layer (block number)"), ZH], loc="lower center", bbox_to_anchor=(0.5, 0.985), ncol=3, frameon=False, handlelength=1.6, columnspacing=1.2)
fig.tight_layout(h_pad=0.5); save(fig, os.path.join(HERE, "figs", "fig_vlm_all_layers"))

DSN = [c[4:] for c in R.columns if c.startswith("acc_")]; zsd = {d: float(100 * (_q[_q.dataset == d].pred == _q[_q.dataset == d].true_label).mean()) for d in DSN}; N["by_dataset"] = {}
fig, axs = plt.subplots(1, len(DSN), figsize=(6.9, 1.55), sharey=True); xs = np.arange(len(KS))
for ax, d in zip(axs, DSN):
    pm = [float(np.mean([PO.loc[(k, e)][f"acc_{d}"] for e in ENC])) for k in KS]; hm = [float(np.mean([best.loc[(k, e)][f"acc_{d}"] for e in ENC])) for k in KS]
    N["by_dataset"][d] = {"zero_shot": zsd[d], "pooled": [round(v, 1) for v in pm], "best_hidden": [round(v, 1) for v in hm]}
    ax.plot(xs, pm, color=L_POOL, lw=1.3, marker="o", ms=2.8, label="Pooled output"); ax.plot(xs, hm, color=L_HID, lw=1.3, marker="o", ms=2.8, label="Best hidden layer"); ax.axhline(zsd[d], color="#7F7F7F", lw=0.9, ls=":")
    ax.set_title(d, fontsize=6.4, pad=2); ax.set_xticks(xs, [str(k) for k in KS]); ax.set_xlabel("$K$", labelpad=1); ax.set_ylim(15, 95)
axs[0].set_ylabel("VLM accuracy (%)"); h, _ = axs[0].get_legend_handles_labels()
fig.legend(handles=h + [plt.Line2D([], [], color="#7F7F7F", lw=0.9, ls=":", label="Zero-shot")], loc="lower center", bbox_to_anchor=(0.5, 0.97), ncol=3, frameon=False, handlelength=1.6, columnspacing=1.2)
fig.tight_layout(w_pad=0.5); save(fig, os.path.join(HERE, "figs", "fig_vlm_datasets"))
# hero figure: accuracy against the number of references for every encoder (columns), over all queries and by dataset (rows)
ROWS = [("All five datasets", None)] + [(d, d) for d in DSN]; hc = {"higher": 0, "equal": 0, "lower": 0}
fig, axs = plt.subplots(len(ROWS), len(ENC), figsize=(6.9, 5.0), sharex=True, sharey="row")
for i, (title, d) in enumerate(ROWS):
    col = "accuracy" if d is None else f"acc_{d}"; z = zs if d is None else zsd[d]; lo, hi = 1e9, -1e9
    for j, e in enumerate(ENC):
        ax = axs[i, j]; pm = np.array([float(PO.loc[(k, e)][col]) for k in KS]); hm = np.array([float(best.loc[(k, e)][col]) for k in KS])
        if d is not None:
            for a_, b_ in zip(hm, pm): hc["higher" if a_ > b_ else "equal" if a_ == b_ else "lower"] += 1
        ax.fill_between(xs, pm, hm, where=hm >= pm, color=L_HID, alpha=0.28, lw=0, interpolate=True); ax.fill_between(xs, pm, hm, where=hm < pm, color=L_POOL, alpha=0.28, lw=0, interpolate=True)
        ax.plot(xs, pm, color=L_POOL, lw=1.1, marker="o", ms=2.2, label="Assisted few-shot learning (pooled output)"); ax.plot(xs, hm, color=L_HID, lw=1.1, marker="o", ms=2.2, label="Assisted latent few-shot learning (best hidden layer, proposed)"); ax.axhline(z, color="#7F7F7F", lw=0.8, ls=":")
        lo, hi = min(lo, pm.min(), hm.min(), z), max(hi, pm.max(), hm.max(), z); ax.tick_params(axis="both", labelsize=5.0, length=2, pad=1.5); ax.set_xticks(xs, [str(k) for k in KS])
        if i == 0: ax.set_title(SH[e], fontsize=6.6, pad=3)
        if i == len(ROWS) - 1: ax.set_xlabel("References $K$", fontsize=5.8, labelpad=1)
    axs[i, 0].set_ylim(lo - 4, hi + 4); axs[i, 0].set_ylabel(("All five\ndatasets" if d is None else title.replace(" ", "\n", 1)) + "\n(%)", fontsize=5.6, fontweight="bold" if d is None else "normal")
    if d is None:
        for ax in axs[i]: ax.set_facecolor("#F6F6F6")
h, _ = axs[0, 0].get_legend_handles_labels()
fig.legend(handles=h + [plt.Line2D([], [], color="#7F7F7F", lw=0.8, ls=":", label="Zero-shot")], loc="lower center", bbox_to_anchor=(0.5, 0.985), ncol=3, frameon=False, handlelength=1.8, columnspacing=1.4, fontsize=6.4)
fig._ca_lettered = True; fig.tight_layout(h_pad=0.35, w_pad=0.3); save(fig, os.path.join(HERE, "figs", "fig_hero")); N["hero_cells_encoder_dataset_budget"] = hc
# cost: input tokens per request, from the usage record of every saved response (the hidden layer changes the images, not their number)
PLN = pd.read_csv(os.path.join(RUN, "request_plan.csv"), dtype=str, keep_default_na=False); CND = pd.read_csv(os.path.join(RUN, "conditions.csv"), dtype=str, keep_default_na=False)
def _tok(k):
    u = json.load(open(os.path.join(RUN, "responses", k + ".json")))["meta"].get("usage", {}); return u.get("input_tokens", 0) + u.get("cache_creation_input_tokens", 0) + u.get("cache_read_input_tokens", 0)
_t = {k: _tok(k) for k in PLN.payload.unique()}; PLN["tok"] = PLN.payload.map(_t); TK = PLN.groupby("condition").tok.mean(); tok0 = float(TK["zero_shot"])
tokc = lambda k, e, layer: float(TK[f"{e}|{layer}|K{k}"])
N["tokens"] = {"zero_shot": round(tok0, 1), "by_K": {k: {"pooled": round(float(np.mean([tokc(k, e, "final_output") for e in ENC])), 1), "best_hidden": round(float(np.mean([tokc(k, e, best.loc[(k, e)].layer) for e in ENC])), 1)} for k in KS}}
# trade-off figure: accuracy against the prompt cost; hollow marker = pooled output, filled marker = best hidden layer of the same encoder
ECOL = {"ResNet50": "#CC79A7", "ViT": "#0072B2", "CLIP": "#009E73", "FLAVA": "#7F7F7F", "SigLIP2": "#E69F00", "Qwen3VL": "#8C5A2B"}; EMK = {"ResNet50": "D", "ViT": "o", "CLIP": "^", "FLAVA": "s", "SigLIP2": "v", "Qwen3VL": "p"}
fig, ax = plt.subplots(figsize=(3.3, 2.75)); ax.grid(True, axis="both", color=BS.GRID, ls="--", lw=0.5)
for k in KS:
    for j, e in enumerate(ENC):
        x = N["tokens"]["by_K"][k]["pooled"] * (1 + (j - 2.5) * 0.011); yp = float(PO.loc[(k, e)].accuracy); yh = float(best.loc[(k, e)].accuracy)
        ax.plot([x, x], [yp, yh], color=ECOL[e], lw=0.9, alpha=0.9, zorder=2); ax.scatter([x], [yp], s=13, marker=EMK[e], facecolor="white", edgecolor=ECOL[e], lw=0.8, zorder=3); ax.scatter([x], [yh], s=13, marker=EMK[e], facecolor=ECOL[e], edgecolor=ECOL[e], lw=0.8, zorder=4)
xm = [N["tokens"]["by_K"][k]["pooled"] for k in KS]
ax.plot([tok0] + xm, [zs] + [N["by_K"][k]["pooled_mean"] for k in KS], color=L_POOL, lw=1.3, ls="--", zorder=1); ax.plot([tok0] + xm, [zs] + [N["by_K"][k]["best_hidden_mean"] for k in KS], color=L_HID, lw=1.5, zorder=1)
ax.scatter([tok0], [zs], s=42, marker="*", color="#444444", zorder=5); ax.annotate("Zero-shot", (tok0, zs), textcoords="offset points", xytext=(4, -9), fontsize=5.4, color="#444444")
for k, x in zip(KS, xm): ax.annotate(f"$K={k}$", (x, 74.6), ha="center", fontsize=5.6, color=BS.INK)
ax.set_xscale("log"); ax.set_xticks([1300, 1500, 1800, 2100, 2400], ["1.3", "1.5", "1.8", "2.1", "2.4"]); ax.minorticks_off(); ax.set_xlim(1210, 2560); ax.set_ylim(49, 76); ax.set_xlabel("Prompt cost: input tokens per query (thousands)"); ax.set_ylabel("VLM accuracy (%)")
from matplotlib.lines import Line2D
H1 = [Line2D([], [], color=L_POOL, lw=1.3, ls="--", marker="o", ms=3.2, mfc="white", mec="#555555", label="Assisted (pooled output)"), Line2D([], [], color=L_HID, lw=1.5, marker="o", ms=3.2, mfc="#555555", mec="#555555", label="Assisted latent (hidden layer)")]
H2 = [Line2D([], [], color=ECOL[e], lw=0, marker=EMK[e], ms=3.2, label=SH[e]) for e in ENC]
fig.legend(handles=H1, loc="lower center", bbox_to_anchor=(0.55, 0.985), ncol=2, frameon=False, fontsize=5.6, handlelength=2.4, columnspacing=1.0)
ax.legend(handles=H2, loc="lower right", frameon=False, fontsize=5.4, ncol=2, handletextpad=0.2, columnspacing=0.8, borderaxespad=0.2, labelspacing=0.25)
fig.tight_layout(pad=0.3); save(fig, os.path.join(HERE, "figs", "fig_tradeoff"))
# heat map: mean accuracy of the five tested hidden layers minus the pooled output, encoder by budget (does not pick a layer)
from matplotlib.colors import LinearSegmentedColormap
Mh = np.array([[N["cells"][f"K{k}|{e}"]["mean_of_five_delta"] for k in KS] for e in ENC]); Lo = np.array([[N["cells"][f"K{k}|{e}"]["mean_of_five_ci"][0] for k in KS] for e in ENC]); Hi = np.array([[N["cells"][f"K{k}|{e}"]["mean_of_five_ci"][1] for k in KS] for e in ENC])
cmap = LinearSegmentedColormap.from_list("pg", ["#8C74C0", "#FFFFFF", "#2E9B66"]); fig, ax = plt.subplots(figsize=(3.3, 2.05)); ax.grid(False)
imh = ax.pcolormesh(np.arange(len(KS) + 1) - 0.5, np.arange(len(ENC) + 1) - 0.5, Mh, cmap=cmap, vmin=-4, vmax=4, edgecolors="white", linewidth=1.2); ax.invert_yaxis()
for i in range(len(ENC)):
    for j in range(len(KS)):
        star = "*" if (Lo[i, j] > 0 or Hi[i, j] < 0) else ""; ax.text(j, i, f"{Mh[i, j]:+.1f}{star}", ha="center", va="center", fontsize=6.4, fontweight="bold" if star else "normal", color=BS.INK)
ax.set_xticks(range(len(KS)), [f"$K={k}$" for k in KS]); ax.set_yticks(range(len(ENC)), [SH[e] for e in ENC]); ax.tick_params(length=0); ax.minorticks_off()
for sp in ax.spines.values(): sp.set_visible(False)
cb = fig.colorbar(imh, ax=ax, fraction=0.045, pad=0.03); cb.solids.set_rasterized(False); cb.set_label("Change (pp)", fontsize=5.8); cb.ax.tick_params(labelsize=5.4, length=2); cb.outline.set_visible(False)
fig.tight_layout(pad=0.3); save(fig, os.path.join(HERE, "figs", "fig_mean5_heat"))
N["mean_of_five_positive"] = int(sum(v["mean_of_five_delta"] > 0 for v in N["cells"].values()))
json.dump(N, open(os.path.join(HERE, "tables", "layer_vlm_allk_numbers.json"), "w"), indent=1)
T = ["\\begin{tabular}{@{}l" + "c" * 4 + "@{}}", "\\toprule", "Encoder & " + " & ".join(f"$K={k}$" for k in KS) + " \\\\", "\\midrule"]
for e in ENC:
    c = []
    for k in KS:
        v = N["cells"][f"K{k}|{e}"]; c.append(("\\best " if v["mean_of_five_delta"] > 0 else "") + f"{sg(v['mean_of_five_delta'])} ({sg(v['mean_of_five_ci'][0])} to {sg(v['mean_of_five_ci'][1])})")
    T.append(NM[e] + " & " + " & ".join(c) + " \\\\")
T += ["\\bottomrule", "\\end{tabular}"]; open(os.path.join(HERE, "tables", "tab_vlm_mean5.tex"), "w").write("\n".join(T) + "\n")

# ------------------------------------------------------------------ main-text table
L = ["\\begin{tabular}{@{}l" + "cl" * 4 + "@{}}", "\\toprule", " & " + " & ".join(f"\\multicolumn{{2}}{{c}}{{$K={k}$}}" for k in KS) + " \\\\", "".join(f"\\cmidrule(lr){{{2 + 2 * i}-{3 + 2 * i}}}" for i in range(4)),
     "Encoder & " + " & ".join("Pooled & Hidden layer" for _ in KS) + " \\\\", "\\midrule"]
for e in ENC:
    c = []
    for k in KS:
        v = N["cells"][f"K{k}|{e}"]; hi = v["best"] > v["pooled"]; c += [("" if hi else "\\best ") + f"{v['pooled']:.1f}", ("\\best " if hi else "") + f"{v['best']:.1f} (b{v['best_block']}, {sg(v['delta'])})"]
    L.append(("ViT~\\cite{arshad2025assisted}" if e == "ViT" else NM[e]) + " & " + " & ".join(c) + " \\\\")
L += ["\\midrule", "Mean of six encoders & " + " & ".join(f"{N['by_K'][k]['pooled_mean']:.1f} & \\best {N['by_K'][k]['best_hidden_mean']:.1f} ({sg(N['by_K'][k]['gain_mean_pp'])})" for k in KS) + " \\\\", "\\bottomrule", "\\end{tabular}"]
open(os.path.join(HERE, "tables", "tab_vlm_allk.tex"), "w").write("\n".join(L) + "\n")

# ------------------------------------------------------------------ appendix
A = ["% generated by tools/make_layer_vlm_allk.py; do not edit by hand", "\\section{Hidden layers with the VLM at every budget: all conditions}", "\\label{app:vlmk}", "",
     f"This appendix lists every condition of the test of Sec.~\\ref{{sec:vlm}}: for each budget of 1, 2, 4 and 8 references and each of the six encoders, the pooled output and five hidden layers, on the {n} queries with the disjoint reference bank. "
     f"The five layers of an encoder at a budget are the layers with the largest gain in same-class coverage over the pooled output of the same encoder at that budget. The test has {NC} conditions and {len(LY)} comparisons of a hidden layer with the pooled output of its encoder. "
     f"Each hidden layer is compared only with the pooled output of its own encoder at the same budget. With a Holm correction over the {len(LY)} comparisons, the smallest adjusted $p$-value is {LY.p_holm.min():.2f}.", "",
     "Figure~\\ref{fig:vlmall} shows every condition, and Fig.~\\ref{fig:vlm} shows the pooled output and the best hidden layer of each encoder by budget.", "",
     "\\begin{figure*}[t]", "\\centering", "\\includegraphics[width=\\linewidth]{figs/fig_vlm_all_layers.pdf}", "\\caption{\\textbf{Every tested condition.} Pooled output (P) and the five tested hidden layers (block numbers) of each encoder, by $K$. 500 queries.}", "\\label{fig:vlmall}", "\\end{figure*}", "",
     "\\begin{figure*}[t]", "\\centering", "\\includegraphics[width=\\linewidth]{figs/fig_vlm_allk.pdf}", "\\caption{\\textbf{Baseline and proposed method by budget.} VLM accuracy with the pooled output and with the best of five hidden layers, whose block is given under each encoder.}", "\\label{fig:vlm}", "\\end{figure*}", ""]
for k in KS:
    A += ["\\begin{table*}[t]", "\\centering", f"\\caption{{\\textbf{{All conditions with {k} reference{'s' if k > 1 else ''} per query.}} Coverage and accuracy in percent; changes against the pooled output of the encoder with the 95\\% interval; Holm $p$ over the {len(LY)} comparisons. Hidden layers above the pooled output in green.}}",
          f"\\label{{tab:app-vlmk{k}}}", "\\scriptsize", "\\setlength{\\tabcolsep}{3.2pt}", "\\begin{tabular}{@{}llcccccccc@{}}", "\\toprule", "Encoder & Latent & Coverage & Change & Accuracy & Change (95\\% CI) & Wins & Losses & $p$ & Holm $p$ \\\\", "\\midrule"]
    for e in ENC:
        p = PO.loc[(k, e)]; A.append(f"{NM[e]} & pooled output & {p.retrieval_coverage:.1f} & & {p.accuracy:.1f} & & & & & \\\\")
        for r in LY[(LY.K == k) & (LY.encoder == e)].sort_values("ln").itertuples():
            A.append(f" & block {r.ln} & {r.retrieval_coverage:.1f} & {sg(r.retrieval_change_vs_own_pooled_pp)} & " + ("\\best " if r.delta_vs_own_pooled_pp > 0 else "") + f"{r.accuracy:.1f} & {sg(r.delta_vs_own_pooled_pp)} ({sg(r.ci_lo)} to {sg(r.ci_hi)}) & {int(r.wins)} & {int(r.losses)} & {r.p:.3f} & {r.p_holm:.2f} \\\\")
        m = EL[(EL.K == k) & (EL.encoder == e)].iloc[0]; A.append(f" & mean of five & & & {m.mean_of_five_layers_accuracy:.1f} & {sg(m.delta_pp)} ({sg(m.ci_lo)} to {sg(m.ci_hi)}) & & & & \\\\" + ("" if e == ENC[-1] else "\\addlinespace[1pt]"))
    A += ["\\bottomrule", "\\end{tabular}", "\\end{table*}", ""]
open(os.path.join(HERE, "appendix_vlm_allk.tex"), "w").write("\n".join(A) + "\n")
print(json.dumps({k: v for k, v in N.items() if k in ("tokens", "mean_of_five_positive")}, indent=0))
