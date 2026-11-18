# Figures, README and checks. Executed by topk_analysis.py in its namespace (AGG, BYD, PQ, STAB, Q, TOP, MATCH, audit, ...).
# The README is figure-led: every result is a figure with a short reading; the tables sit in collapsible blocks and CSV files.
import gzip, shutil
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

plt.rcParams.update({"font.family": "Helvetica", "font.size": 7, "axes.spines.top": False, "axes.spines.right": False, "axes.linewidth": 0.6, "xtick.major.width": 0.6, "ytick.major.width": 0.6})
COL = {"1_ViT": "#0072B2", "2_DINOv2": "#D55E00", "3_CLIP": "#009E73", "4_ResNet50": "#CC79A7", "5_SigLIP2": "#E69F00", "6_BioCLIP": "#56B4E9", "7_DINOv3": "#000000", "8_Qwen3VL": "#8C5A2B", "9_FLAVA": "#7F7F7F"}
MK = {"1_ViT": "o", "2_DINOv2": "s", "3_CLIP": "^", "4_ResNet50": "D", "5_SigLIP2": "v", "6_BioCLIP": "P", "7_DINOv3": "X", "8_Qwen3VL": "h", "9_FLAVA": "*"}
KCOL = {1: "#9ecae1", 2: "#4292c6", 4: "#08519c", 8: "#08306b"}; KMK = {1: "o", 2: "s", 4: "^", 8: "D"}; GAIN, LOSS = "#0072B2", "#E69F00"
for k in KS:                                   # class tables are large: keep them compressed
    src = os.path.join(TB, f"by_class_top{k}.csv")
    if os.path.exists(src):
        with open(src, "rb") as a, gzip.open(src + ".gz", "wb") as b: shutil.copyfileobj(a, b)
        os.remove(src)
AGG["pct_slots"] = 100 * AGG.fraction_of_slots_same_class; AGG["pct_slots_replaced_prev"] = 100 * AGG.fraction_replaced_vs_previous_layer
for nm in ("final_output", "pooled_vit"):
    AGG[f"pct_0_to_ge1_vs_{nm}"] = 100 * AGG[f"n_0_to_ge1_vs_{nm}"] / AGG.n_queries; AGG[f"pct_ge1_to_0_vs_{nm}"] = 100 * AGG[f"n_ge1_to_0_vs_{nm}"] / AGG.n_queries
A = AGG.set_index(["encoder", "layer", "K"]); N = int(COH.sum()); row = lambda e, l, k: A.loc[(e, l, k)]
f1 = lambda v: f"{v:.1f}"; f2 = lambda v: f"{v:.2f}"; pp = lambda v: f"{v:+.1f}"; sl = lambda l: "final" if l == "final_output" else ("stem" if "stem" in l else "L" + str(int(l.split("_")[1])))
def best(e, k, nm, with_final):
    cand = hidden(e) + (["final_output"] if with_final else []); v = [row(e, l, k)[f"net_coverage_pp_vs_{nm}"] for l in cand]; return cand[int(np.argmax(v))]          # earliest layer on ties
HL = {(e, k, nm): best(e, k, nm, nm == "pooled_vit" and e != POOLED[0]) for e in ENCS for k in KS for nm in ("final_output", "pooled_vit")}
def save(fig, name): fig.savefig(os.path.join(FG, name), dpi=600); fig.savefig(os.path.join(FG, name[:-4] + ".pdf")); plt.close(fig)      # PNG for the README, vector PDF for papers
def letter(a, t, x=-0.1, y=1.06): a.text(x, y, t, transform=a.transAxes, fontsize=10, fontweight="bold")

# ------------------------------------------------------------------ figures
def fig_depth_map():
    """encoder x depth map of the share of matching slots, one panel per budget"""
    fig, ax = plt.subplots(1, 4, figsize=(7.2, 2.9), sharey=True); vals = AGG.pct_slots; vmin, vmax = np.floor(vals.min() / 5) * 5, np.ceil(vals.max() / 5) * 5
    for a, k, t in zip(ax, KS, "abcd"):
        for i, e in enumerate(ENCS):
            hid = hidden(e); v = np.array([[row(e, l, k).pct_slots for l in hid]]); im = a.pcolormesh(np.linspace(0, 1, len(hid) + 1), [i, i + 0.86], v, cmap="cividis", vmin=vmin, vmax=vmax, edgecolors="white", linewidth=0.25)
            a.pcolormesh([1.04, 1.13], [i, i + 0.86], np.array([[row(e, "final_output", k).pct_slots]]), cmap="cividis", vmin=vmin, vmax=vmax); j = int(np.argmax(v[0])); a.scatter([(j + 0.5) / len(hid)], [i + 0.43], s=9, c="white", edgecolors="#111", linewidths=0.5, zorder=4)
        a.set_ylim(len(ENCS), -0.1); a.set_yticks([i + 0.43 for i in range(len(ENCS))], [NAME[e] for e in ENCS]); a.set_xticks([0.07, 0.5, 0.9, 1.085], ["First", "Mid", "Last", "Out"]); a.tick_params(length=0, labelsize=5.8); a.tick_params(axis="y", labelsize=7); a.set_title(f"Top {k}", fontsize=7.5); letter(a, t, -0.02, 1.05)
        for s_ in a.spines.values(): s_.set_visible(False)
    cb = fig.colorbar(im, ax=ax, fraction=0.018, pad=0.015); cb.set_label("Reference slots with the query's class (%)", fontsize=6.5); cb.outline.set_visible(False); cb.solids.set_rasterized(False)
    fig.text(0.45, 0.015, "Layer depth (each cell is one layer; Out = the encoder's final output; dot = best layer of the encoder)", ha="center", fontsize=6.5); fig.subplots_adjust(left=0.115, right=0.9, top=0.88, bottom=0.14, wspace=0.1); save(fig, "fig1_same_class_by_encoder_and_depth.png")

def fig_coverage():
    """coverage of the pooled ViT, of each encoder's final output and of its best layer"""
    fig, ax = plt.subplots(1, 4, figsize=(7.2, 2.7), sharey=True)
    for a, k, t in zip(ax, KS, "abcd"):
        pool = row(*POOLED, k).pct_at_least_one_same_class; a.axvline(pool, color="#444", lw=0.8, ls=(0, (3, 2)))
        for i, e in enumerate(ENCS):
            l = HL[(e, k, "pooled_vit")]; x0, x1 = row(e, "final_output", k).pct_at_least_one_same_class, row(e, l, k).pct_at_least_one_same_class
            a.plot([x0, x1], [i, i], color="#B0B0B0", lw=1.0, zorder=1); a.scatter([x0], [i], s=16, facecolors="white", edgecolors=COL[e], linewidths=1.0, zorder=3); a.scatter([x1], [i], s=20, color=COL[e], marker=MK[e], zorder=4)
            a.text(max(x0, x1) + 0.7, i, sl(l), va="center", fontsize=5.4, color="#333")
        a.set_ylim(len(ENCS) - 0.5, -0.6); a.set_yticks(range(len(ENCS)), [NAME[e] for e in ENCS]); a.set_title(f"Top {k}", fontsize=7.5); a.set_xlabel("Coverage (%)"); a.margins(x=0.18); a.grid(axis="x", color="#ECECEC", lw=0.5); a.set_axisbelow(True); letter(a, t, -0.04)
    fig.legend(handles=[Line2D([], [], color="#444", lw=0.8, ls=(0, (3, 2)), label="Pooled ViT (Arshad et al.)"), Line2D([], [], marker="o", ls="", mfc="white", mec="#444", ms=4, label="Encoder's final output"), Line2D([], [], marker="o", ls="", color="#444", ms=4, label="Best layer of the encoder (label)")],
               frameon=False, ncol=3, fontsize=6.5, loc="upper center", bbox_to_anchor=(0.5, 1.0)); fig.tight_layout(rect=(0, 0, 1, 0.91)); save(fig, "fig2_coverage_pooled_final_best.png")

def fig_lines(col, ylabel, fname, second=None):
    fig, ax = plt.subplots(1, 4, figsize=(7.2, 2.5), sharey=True)
    for a, k, t in zip(ax, KS, "abcd"):
        for e in ENCS:
            hid = hidden(e); x = np.linspace(0, 1, len(hid)); y = np.array([row(e, l, k)[col] for l in hid]); m = ~np.isnan(y); a.plot(x[m], y[m], color=COL[e], marker=MK[e], ms=2.4, lw=0.9, mec="white", mew=0.3, label=NAME[e])
            if second: a.plot(x, -np.array([row(e, l, k)[second] for l in hid]), color=COL[e], lw=0.8, ls=(0, (2, 1.5)))
        if second: a.axhline(0, color="#444", lw=0.6)
        a.set_title(f"Top {k}", fontsize=7.5); a.set_xticks([0, 0.5, 1], ["First", "Middle", "Last"]); a.set_xlabel("Layer depth"); letter(a, t, -0.12, 1.07)
    ax[0].set_ylabel(ylabel); fig.legend(*ax[0].get_legend_handles_labels(), frameon=False, ncol=5, fontsize=6.5, loc="upper center", bbox_to_anchor=(0.5, 1.0)); fig.tight_layout(rect=(0, 0, 1, 0.86)); save(fig, fname)

def fig_gain_loss():
    """queries gained and lost by each encoder's best layer against the pooled ViT"""
    fig, ax = plt.subplots(1, 4, figsize=(7.2, 2.7), sharey=True); lim = max(max(row(e, HL[(e, k, "pooled_vit")], k).n_0_to_ge1_vs_pooled_vit, row(e, HL[(e, k, "pooled_vit")], k).n_ge1_to_0_vs_pooled_vit) for e in ENCS for k in KS) * 1.25
    for a, k, t in zip(ax, KS, "abcd"):
        for i, e in enumerate(ENCS):
            r = row(e, HL[(e, k, "pooled_vit")], k); g, l = int(r.n_0_to_ge1_vs_pooled_vit), int(r.n_ge1_to_0_vs_pooled_vit); a.barh(i, g, color=GAIN, height=0.62); a.barh(i, -l, color=LOSS, height=0.62)
            a.text(g + lim * 0.03, i, str(g), va="center", fontsize=5.6); a.text(-l - lim * 0.03, i, str(l), va="center", ha="right", fontsize=5.6)
        a.axvline(0, color="#333", lw=0.6); a.set_xlim(-lim, lim); a.set_ylim(len(ENCS) - 0.5, -0.6); a.set_yticks(range(len(ENCS)), [NAME[e] for e in ENCS]); a.set_title(f"Top {k}", fontsize=7.5); a.set_xlabel("Queries"); a.set_xticks([-int(lim * 0.6), 0, int(lim * 0.6)], [int(lim * 0.6), 0, int(lim * 0.6)]); letter(a, t, -0.04)
    fig.legend(handles=[Patch(color=GAIN, label="Gained a same-class reference (0 to >=1)"), Patch(color=LOSS, label="Lost every same-class reference (>=1 to 0)")], frameon=False, ncol=2, fontsize=6.5, loc="upper center", bbox_to_anchor=(0.5, 1.0)); fig.tight_layout(rect=(0, 0, 1, 0.91)); save(fig, "fig4_gained_lost_vs_pooled.png")

def fig_vit10():
    fig, ax = plt.subplots(1, 3, figsize=(7.2, 2.5)); a = ax[0]; segs = [("n_ge1_to_ge1_vs_final_output", "#C9CED6", "Both have one (>=1 to >=1)"), ("n_0_to_ge1_vs_final_output", GAIN, "Gained (0 to >=1)"), ("n_ge1_to_0_vs_final_output", LOSS, "Lost (>=1 to 0)"), ("n_0_to_0_vs_final_output", "#5A5F66", "Neither (0 to 0)")]
    for i, k in enumerate(KS):
        r = row("1_ViT", "layer_10", k); left = 0
        for col, c, lab in segs:
            v = 100 * r[col] / N; a.barh(i, v, left=left, color=c, height=0.66, label=lab if i == 0 else None, edgecolor="white", lw=0.4)
            if v > 4.5: a.text(left + v / 2, i, f"{int(r[col])}", ha="center", va="center", fontsize=5.8, color="white" if c in (GAIN, "#5A5F66") else "#111")
            left += v
    a.set_yticks(range(4), [f"Top {k}" for k in KS]); a.invert_yaxis(); a.set_xlim(0, 100); a.set_xlabel(f"Queries (%), n = {N}"); a.legend(frameon=False, fontsize=5.6, loc="upper center", bbox_to_anchor=(0.5, 1.33), ncol=2); letter(a, "a", -0.16, 1.2)
    a = ax[1]; x = range(4); b_ = [row(*POOLED, k).pct_at_least_one_same_class for k in KS]; v = [row("1_ViT", "layer_10", k).pct_at_least_one_same_class for k in KS]
    a.fill_between(x, b_, v, color=COL["1_ViT"], alpha=0.15, lw=0); a.plot(x, b_, color="#666", marker="s", ms=3.2, lw=1.1, label="Pooled output"); a.plot(x, v, color=COL["1_ViT"], marker="o", ms=3.4, lw=1.3, label="Layer 10")
    for i, k in enumerate(KS): a.text(i, max(v[i], b_[i]) + 1.6, pp(v[i] - b_[i]), ha="center", fontsize=5.8, color=COL["1_ViT"])
    a.set_xticks(x, [f"Top {k}" for k in KS]); a.set_ylabel("Coverage (%)"); a.legend(frameon=False, fontsize=6, loc="lower right"); letter(a, "b", -0.2, 1.2)
    a = ax[2]; r_ = [100 * row("1_ViT", "layer_10", k).fraction_replaced_vs_final_output for k in KS]; idn = [row("1_ViT", "layer_10", k).pct_identical_order_vs_final_output for k in KS]
    a.bar([i - 0.2 for i in x], r_, width=0.38, color="#0072B2", label="Slots replaced"); a.bar([i + 0.2 for i in x], idn, width=0.38, color="#B8BDC6", label="Identical ordered list")
    for i in x: a.text(i - 0.2, r_[i] + 1.5, f"{r_[i]:.0f}", ha="center", fontsize=5.6); a.text(i + 0.2, idn[i] + 1.5, f"{idn[i]:.0f}", ha="center", fontsize=5.6)
    a.set_xticks(x, [f"Top {k}" for k in KS]); a.set_ylabel("Percent"); a.legend(frameon=False, fontsize=6, loc="upper right"); letter(a, "c", -0.2, 1.2); fig.tight_layout(); save(fig, "fig5_vit_layer10_vs_pooled.png")

def fig_layerwise(col, ylabel, fname, final_marker):
    """one panel per encoder, one line per budget, real layer numbers on the x axis"""
    fig, axs = plt.subplots(3, 3, figsize=(7.2, 5.6), sharey=True)
    for a, e in zip(axs.ravel(), ENCS):
        hid = hidden(e); xs = [0 if "stem" in l else int(l.split("_")[1]) for l in hid]
        for k in KS:
            y = np.array([row(e, l, k)[col] for l in hid]); m = ~np.isnan(y); a.plot(np.array(xs)[m], y[m], color=KCOL[k], marker=KMK[k], ms=2.4, lw=1.0, mec="white", mew=0.3, label=f"Top {k}")
            if final_marker: a.scatter([xs[-1] + max(1, len(hid) * 0.07)], [row(e, "final_output", k)[col]], marker="D", s=11, color=KCOL[k], edgecolors="#111", linewidths=0.4, zorder=4)
        a.set_title(NAME[e], fontsize=7.5, color=COL[e], fontweight="bold"); a.grid(axis="y", color="#ECECEC", lw=0.5); a.set_axisbelow(True); a.set_xticks([x for x in xs if x % (4 if len(hid) > 12 else 2) == 0 or x == xs[-1]])
    for a in axs[-1]: a.set_xlabel("Layer")
    for a in axs[:, 0]: a.set_ylabel(ylabel)
    h, l = axs[0, 0].get_legend_handles_labels()
    if final_marker: h.append(Line2D([], [], marker="D", ls="", color="#777", mec="#111", ms=4)); l.append("Final output")
    fig.legend(h, l, frameon=False, ncol=5, fontsize=6.8, loc="upper center", bbox_to_anchor=(0.5, 1.0)); fig.tight_layout(rect=(0, 0, 1, 0.95)); save(fig, fname)

def fig_settle_dataset():
    fig, ax = plt.subplots(1, 2, figsize=(7.2, 3.0), gridspec_kw={"width_ratios": [1, 1.5]}); a = ax[0]
    M = np.array([[STAB[(STAB.encoder == e) & (STAB.K == k)].pct_settled_only_at_last_layer.iloc[0] for k in KS] for e in ENCS]); a.pcolormesh(np.arange(M.shape[1] + 1) - 0.5, np.arange(M.shape[0] + 1) - 0.5, M, cmap="cividis", vmin=0, vmax=100, edgecolors="white", linewidth=0.5); a.set_ylim(M.shape[0] - 0.5, -0.5)
    for i in range(M.shape[0]):
        for j in range(M.shape[1]): a.text(j, i, f"{M[i, j]:.0f}", ha="center", va="center", fontsize=6, color="white" if M[i, j] < 55 else "#111")
    a.set_xticks(range(4), [f"Top {k}" for k in KS]); a.set_yticks(range(len(ENCS)), [NAME[e] for e in ENCS]); a.tick_params(length=0); a.set_title("Lists that settle only at the last layer (%)", fontsize=7.2); letter(a, "a", -0.42, 1.04)
    a = ax[1]; ds = list(Q.dataset.unique()); D = BYD[(BYD.K == 4)].set_index(["dataset", "encoder", "layer"]); M = np.array([[100 * D.loc[(d, e, "final_output")].fraction_of_slots_same_class for d in ds] for e in ENCS]); a.pcolormesh(np.arange(M.shape[1] + 1) - 0.5, np.arange(M.shape[0] + 1) - 0.5, M, cmap="cividis", edgecolors="white", linewidth=0.5); a.set_ylim(M.shape[0] - 0.5, -0.5)
    for i in range(M.shape[0]):
        for j in range(M.shape[1]): a.text(j, i, f"{M[i, j]:.0f}", ha="center", va="center", fontsize=6, color="white" if M[i, j] < (M.min() + M.max()) / 2 else "#111", fontweight="bold" if M[i, j] == M[:, j].max() else "normal")
    a.set_xticks(range(len(ds)), [d[3:].replace("_", "\n", 1).replace("_", " ") for d in ds]); a.set_yticks(range(len(ENCS)), [NAME[e] for e in ENCS]); a.tick_params(length=0); a.set_title("Top-4 slots with the query's class at the final output (%), by dataset", fontsize=7.2); letter(a, "b", -0.26, 1.04)
    for x in ax:
        for s_ in x.spines.values(): s_.set_visible(False)
    fig.tight_layout(); save(fig, "fig8_settling_and_datasets.png")

fig_depth_map(); fig_coverage(); fig_lines("pct_slots_replaced_prev", "Top-K slots replaced\nrelative to the previous layer (%)", "fig3_adjacent_layer_replacement.png"); fig_gain_loss(); fig_vit10()
fig_layerwise("pct_slots", "Slots with the query's class (%)", "fig6_layerwise_same_class.png", True); fig_layerwise("pct_slots_replaced_prev", "Slots replaced vs previous layer (%)", "fig7_layerwise_replacement.png", False); fig_settle_dataset()
fig_lines("pct_0_to_ge1_vs_final_output", "Queries (%): gained (solid, up),\nlost (dashed, down)", "supp_transitions_vs_final_output.png", second="pct_ge1_to_0_vs_final_output"); fig_lines("pct_0_to_ge1_vs_pooled_vit", "Queries (%): gained (solid, up),\nlost (dashed, down)", "supp_transitions_vs_pooled_vit.png", second="pct_ge1_to_0_vs_pooled_vit")
for old in ("adjacent_layer_replacement_top1_2_4_8.png", "coverage_transitions_vs_final_output_top1_2_4_8.png", "coverage_transitions_vs_pooled_vit_top1_2_4_8.png"):
    if os.path.exists(os.path.join(FG, old)): os.remove(os.path.join(FG, old))

# ------------------------------------------------------------------ tables (kept in collapsible blocks and CSV files)
TABLES = {}
def md_table(name, head, rows):
    TABLES[name] = pd.DataFrame(rows, columns=head); TABLES[name].to_csv(os.path.join(TB, f"readme_{name}.csv"), index=False)
    return f"<!-- table:{name} -->\n| " + " | ".join(head) + " |\n|" + "---|" * len(head) + "\n" + "\n".join("| " + " | ".join(str(c) for c in r) + " |" for r in rows)
def fold(title, content): return f"<details>\n<summary>{title}</summary>\n\n{content}\n\n</details>\n"
def bestk(k): return max(((e, HL[(e, k, "pooled_vit")]) for e in ENCS), key=lambda t: row(t[0], t[1], k).net_coverage_pp_vs_pooled_vit)

def tables_k(k):
    out = []; rows = []
    for e in ENCS:
        l = HL[(e, k, "final_output")]; r = row(e, l, k)
        rows.append([NAME[e], f1(r.baseline_coverage_pct_final_output), lname(l), f1(r.pct_at_least_one_same_class), f"{int(r.n_0_to_ge1_vs_final_output)} of {int(r.baseline_misses_final_output)}", int(r.n_ge1_to_0_vs_final_output), pp(r.net_coverage_pp_vs_final_output),
                     f"{f2(r.mean_replaced_vs_final_output)} ({f1(100 * r.fraction_replaced_vs_final_output)}%)", f1(r.pct_identical_order_vs_final_output)])
    out.append(f"**Each encoder's best layer against its own `final_output`** ({N} paired queries).\n\n" + md_table(f"top{k}_within_encoder", ["Encoder", "Final-output coverage (%)", "Best layer", "Layer coverage (%)", "Misses recovered (0 to >=1)", "Coverage lost (>=1 to 0)", "Net change (pp)", f"References replaced (mean of {k})", "Identical ordered list (%)"], rows))
    rows = []; pool = row(*POOLED, k)
    for e in ENCS:
        fo = row(e, "final_output", k); l = HL[(e, k, "pooled_vit")]; r = row(e, l, k)
        rows.append([NAME[e], f1(fo.pct_at_least_one_same_class), pp(fo.net_coverage_pp_vs_pooled_vit), lname(l), f1(r.pct_at_least_one_same_class), f"{int(r.n_0_to_ge1_vs_pooled_vit)} of {int(r.baseline_misses_pooled_vit)}", int(r.n_ge1_to_0_vs_pooled_vit), pp(r.net_coverage_pp_vs_pooled_vit),
                     f"{f2(r.mean_replaced_vs_pooled_vit)} ({f1(100 * r.fraction_replaced_vs_pooled_vit)}%)"])
    out.append(f"**Every encoder against the pooled ViT output of Arshad et al.** (pooled coverage {f1(pool.pct_at_least_one_same_class)}%, {int(pool.baseline_misses_pooled_vit)} misses of {N}).\n\n" + md_table(f"top{k}_vs_pooled_vit", ["Encoder", "Final-output coverage (%)", "Final-output net change (pp)", "Best layer", "Layer coverage (%)", "Misses recovered (0 to >=1)", "Coverage lost (>=1 to 0)", "Net change (pp)", f"References replaced (mean of {k})"], rows))
    rows = []; dist_head = " / ".join(str(j) for j in range(k + 1)) if k <= 2 else f"0 / ... / {k}"
    for e in ENCS:
        fo = row(e, "final_output", k); hid = hidden(e)[1:]; s = STAB[(STAB.encoder == e) & (STAB.K == k)].iloc[0]
        rows.append([NAME[e], f2(fo.mean_same_class), f1(fo.pct_at_least_one_same_class), f1(fo.pct_all_K_same_class), " / ".join(f"{fo[f'pct_queries_with_{j}_same_class']:.0f}" for j in range(k + 1)), f1(100 * np.mean([row(e, l, k).fraction_replaced_vs_previous_layer for l in hid])),
                     f1(np.mean([row(e, l, k).pct_identical_order_vs_previous_layer for l in hid])), f1(np.mean([row(e, l, k).pct_same_ids_other_order_vs_previous_layer for l in hid])), f1(s.pct_settled_only_at_last_layer)])
    out.append("**Same-class matches at the final output, turnover between adjacent layers, agreement across depth.**\n\n" + md_table(f"top{k}_distribution_turnover", ["Encoder", f"Mean same class (of {k})", "At least one (%)", f"All {k} (%)", f"Queries with {dist_head} matches (%)", "Slots replaced per layer step (%)", "Identical ordered list, adjacent layers (%)", "Same references, other order (%)", "Settles only at the last layer (%)"], rows))
    head = ["Encoder", "Layer", f"Mean same class (of {k})", "At least one (%)", f"All {k} (%)", "Replaced vs previous layer (mean)", "Identical to previous layer (%)", "Replaced vs final output (mean)", "0 to >=1 / >=1 to 0 vs final output", "Net vs final output (pp)", "0 to >=1 / >=1 to 0 vs pooled ViT", "Net vs pooled ViT (pp)"]; rows = []
    for e in ENCS:
        for l in LAYERS[e]:
            r = row(e, l, k); first = np.isnan(r.mean_replaced_vs_previous_layer)
            rows.append([NAME[e], lname(l), f2(r.mean_same_class), f1(r.pct_at_least_one_same_class), f1(r.pct_all_K_same_class), "" if first else f2(r.mean_replaced_vs_previous_layer), "" if first else f1(r.pct_identical_order_vs_previous_layer), f2(r.mean_replaced_vs_final_output),
                         f"{int(r.n_0_to_ge1_vs_final_output)} / {int(r.n_ge1_to_0_vs_final_output)}", pp(r.net_coverage_pp_vs_final_output), f"{int(r.n_0_to_ge1_vs_pooled_vit)} / {int(r.n_ge1_to_0_vs_pooled_vit)}", pp(r.net_coverage_pp_vs_pooled_vit)])
    out.append(f"**Every encoder and layer** ({len(rows)} rows; for `final output` the previous layer is the last hidden layer).\n\n" + md_table(f"top{k}_all_layers", head, rows))
    return "\n\n".join(out)

def section_k(k):
    pool = row(*POOLED, k); bp = bestk(k); rb = row(bp[0], bp[1], k); adj = [row(e, l, k).fraction_replaced_vs_previous_layer for e in ENCS for l in hidden(e)[1:]]; ident = [row(e, l, k).pct_identical_order_vs_previous_layer for e in ENCS for l in hidden(e)[1:]]
    bs = max(((e, l) for e in ENCS for l in LAYERS[e]), key=lambda t: row(t[0], t[1], k).pct_slots); v = row("1_ViT", "layer_10", k); st = STAB[STAB.K == k].pct_settled_only_at_last_layer
    L = [f"### Top {k}\n",
         f"- **Pooled baseline:** at least one same-class reference for {f1(pool.pct_at_least_one_same_class)}% of the {N} queries ({int(pool.baseline_misses_pooled_vit)} queries without one)" + (f"; {f1(100 * pool.fraction_of_slots_same_class)}% of the {k} slots match." if k > 1 else ". With one slot, the share of matching slots, the coverage and the share with all references matching are the same number."),
         f"- **Turnover:** {f1(100 * np.mean(adj))}% of the top-{k} slots change between adjacent layers (from {f1(100 * min(adj))}% to {f1(100 * max(adj))}% over layers); the ordered list stays identical for {f1(np.mean(ident))}% of the queries." + (" Here \"replaced\" means the first reference changed." if k == 1 else ""),
         f"- **Largest net coverage change against the pooled output:** {NAME[bp[0]]} {lname(bp[1])}, {pp(rb.net_coverage_pp_vs_pooled_vit)} pp ({int(rb.n_0_to_ge1_vs_pooled_vit)} queries gained, {int(rb.n_ge1_to_0_vs_pooled_vit)} lost). Most matching slots: {NAME[bs[0]]} {lname(bs[1])} ({f1(row(bs[0], bs[1], k).pct_slots)}%).",
         f"- **ViT layer 10 against the pooled output:** coverage {f1(v.baseline_coverage_pct_final_output)}% to {f1(v.pct_at_least_one_same_class)}% ({int(v.n_0_to_ge1_vs_final_output)} gained, {int(v.n_ge1_to_0_vs_final_output)} lost, {pp(v.net_coverage_pp_vs_final_output)} pp); {f1(100 * v.fraction_replaced_vs_final_output)}% of the slots replaced.",
         f"- **Agreement across depth:** the ordered list reaches its last-layer form only at the last layer for {f1(st.min())}% to {f1(st.max())}% of the queries, depending on the encoder.\n",
         fold(f"Tables for top {k}", tables_k(k)), f"Breakdowns for top {k}: [by dataset](analysis/tables/by_dataset_top{k}.csv), [by class with sample sizes](analysis/tables/by_class_top{k}.csv.gz).\n"]
    return "\n".join(L)

def layer_tables():
    out = []
    for e in ENCS:
        rows = []
        for l in LAYERS[e]:
            c = [lname(l)]
            for k in KS:
                x = row(e, l, k); c += [f1(x.pct_slots)] + ([f1(x.pct_at_least_one_same_class)] if k > 1 else [])
            for k in KS:
                x = row(e, l, k); c.append("" if np.isnan(x.mean_replaced_vs_previous_layer) else f1(100 * x.fraction_replaced_vs_previous_layer))
            c += [pp(row(e, l, k).net_coverage_pp_vs_pooled_vit) for k in KS]; rows.append(c)
        head = ["Layer", "Top 1: slots matching"] + [h for k in KS[1:] for h in (f"Top {k}: slots matching", f"Top {k}: coverage")] + [f"Top {k}: replaced" for k in KS] + [f"Top {k}: net vs pooled (pp)" for k in KS]
        out.append(fold(f"{NAME[e]}: every layer at every budget (percent)", md_table(f"layerwise_{e}", head, rows)))
    return "\n".join(out)

def layer_notes():
    L = []
    for e in ENCS:
        hid = hidden(e); bl = {k: max(hid, key=lambda l: (row(e, l, k).pct_slots, -hid.index(l))) for k in KS}; churn = max(hid[1:], key=lambda l: row(e, l, 4).fraction_replaced_vs_previous_layer); calm = min(hid[1:], key=lambda l: row(e, l, 4).fraction_replaced_vs_previous_layer)
        peak = "its last layer at every budget" if all(bl[k] == hid[-1] for k in KS) else ", ".join(f"{sl(bl[k])} (top {k})" for k in KS); fo, la = row(e, "final_output", 4), row(e, hid[-1], 4)
        fin = "final output equals the last layer" if e in FINAL_EQUALS_LAST else f"final output differs from the last layer ({f1(la.pct_slots)}% against {f1(fo.pct_slots)}% of top-4 slots matching)"
        L.append(f"- **{NAME[e]}:** peaks at {peak}; largest reshuffle at {sl(churn)} ({f1(100 * row(e, churn, 4).fraction_replaced_vs_previous_layer)}% of top-4 slots), steadiest at {sl(calm)} ({f1(100 * row(e, calm, 4).fraction_replaced_vs_previous_layer)}%); {fin}.")
    return "\n".join(L) + "\n"

def examples():
    k = 4; e, l = "1_ViT", "layer_10"; d = PQ[(PQ.encoder == e) & (PQ.layer == l) & (PQ.K == k)].reset_index(drop=True); L = [f"Two matched queries as illustrations, taken after the aggregate results were computed: the first query in folder order with each transition, for ViT layer 10 against the pooled `final_output` at top {k}.\n"]
    for tr, txt in (("0 to >=1", "gained"), (">=1 to 0", "lost")):
        g = d[d.transition_vs_final_output == tr]
        if not len(g): L.append(f"- **{txt.capitalize()} (`{tr}`):** no such query at top {k}."); continue
        r = g.iloc[0]; qi = int(np.where((Q.dataset == r.dataset) & (Q["query"] == r["query"]))[0][0]); cl = lambda key: ", ".join(c.replace("_", " ") for c in ref_cls[TOP[key][qi, :k]].tolist()); base = f"{r.dataset}/{r['query']}/{e}"
        L.append(f"- **{txt.capitalize()} (`{tr}`):** {r.dataset[3:].replace('_', ' ')}, query class {r.query_class.replace('_', ' ')}. Pooled references in rank order: {cl((e, 'final_output'))}. Layer 10 references: {cl((e, l))}. "
                 f"Folders: [`final_output/top{k}`]({base}/final_output/top{k}), [`{l}/top{k}`]({base}/{l}/top{k}).")
    return "\n".join(L) + "\n"

# ------------------------------------------------------------------ README text
b = {k: row(*POOLED, k) for k in KS}; adjk = {k: 100 * np.mean([row(e, l, k).fraction_replaced_vs_previous_layer for e in ENCS for l in hidden(e)[1:]]) for k in KS}; idk = {k: np.mean([row(e, l, k).pct_identical_order_vs_previous_layer for e in ENCS for l in hidden(e)[1:]]) for k in KS}
tops = {k: bestk(k) for k in KS}; v10 = {k: row("1_ViT", "layer_10", k) for k in KS}; proto = audit.get("protocol", "leave-one-out pool"); DISJ = proto == "disjoint"
SCOPE = (f"{N} queries (100 per dataset) and a separate reference bank of 100 images per dataset; no image is in both" if DISJ else f"{N} queries; each dataset is one pool of about 100 images and every image retrieves from the rest of its pool")

def across_tables():
    rows = []
    for k in KS:
        r, bb = v10[k], b[k]; hist = json.loads(r.paired_change_hist_vs_final_output); up = sum(v for d, v in hist.items() if int(d) > 0); dn = sum(v for d, v in hist.items() if int(d) < 0)
        rows.append([k, f2(bb.mean_same_class), f2(r.mean_same_class), f1(100 * bb.fraction_of_slots_same_class), f1(100 * r.fraction_of_slots_same_class), f1(bb.pct_at_least_one_same_class), f1(r.pct_at_least_one_same_class), int(r.n_0_to_ge1_vs_final_output), int(r.n_ge1_to_0_vs_final_output),
                     int(r.n_ge1_to_ge1_vs_final_output), int(r.n_0_to_0_vs_final_output), pp(r.net_coverage_pp_vs_final_output), f"{f1(r.baseline_misses_recovered_pct_final_output)}% of {int(r.baseline_misses_final_output)}", f1(100 * r.fraction_replaced_vs_final_output), f1(r.pct_identical_order_vs_final_output), f"{up} / {N - up - dn} / {dn}"])
    t1 = md_table("vit_layer10_across_k", ["K", "Pooled: mean same class", "Layer 10: mean same class", "Pooled: slots matching (%)", "Layer 10: slots matching (%)", "Pooled coverage (%)", "Layer 10 coverage (%)", "0 to >=1", ">=1 to 0", ">=1 to >=1", "0 to 0", "Net change (pp)", "Pooled misses recovered", "Slots replaced (%)", "Identical ordered list (%)", "Queries with more / equal / fewer matches"], rows)
    rows = []
    for k in KS:
        c = AGG[(AGG.K == k) & ~((AGG.encoder == POOLED[0]) & (AGG.layer == "final_output"))].sort_values(["net_coverage_pp_vs_pooled_vit", "fraction_of_slots_same_class"], ascending=False).head(3); s = AGG[AGG.K == k].sort_values("fraction_of_slots_same_class", ascending=False).iloc[0]
        rows.append([k, f1(b[k].pct_at_least_one_same_class)] + [f"{NAME[r.encoder]} {lname(r.layer)} ({pp(r.net_coverage_pp_vs_pooled_vit)})" for r in c.itertuples()] + [f"{NAME[s.encoder]} {lname(s.layer)} ({f1(100 * s.fraction_of_slots_same_class)}%)"])
    t2 = md_table("top_layers_by_k", ["K", "Pooled coverage (%)", "First (net pp)", "Second (net pp)", "Third (net pp)", "Largest share of matching slots"], rows)
    return f"**ViT layer 10 against the pooled ViT `final_output`** ({N} paired queries).\n\n{t1}\n\n**Layers with the largest net coverage change against the pooled ViT output** (in-sample ranking; the margins between the leaders are a few queries).\n\n{t2}"

def results():
    L = []; w = L.append
    w("## Results at a glance\n")
    w(f"Scope: {SCOPE}; five datasets, nine encoders, {audit['layers_without_final_output']} layers, top 1, 2, 4 and 8. The same queries and the same reference bank are used for every encoder, layer and budget. "
      "The baseline of the published pipeline is the pooled ViT output of Arshad et al. (`1_ViT/final_output`). Every result is a description of these queries; no layer is selected here.\n")
    w("![Same-class slots by encoder and depth](analysis/figures/fig1_same_class_by_encoder_and_depth.png)\n")
    w("**Fig. 1 | Where in each encoder the retrieved references carry the query's class.** **a** to **d**, Share of the top-K reference slots filled by a same-class reference, for every layer of every encoder (one cell per layer, depth from left to right) and for the encoder's final output (separate cell). "
      f"The dot marks the best layer of each encoder. Color scale shared by the four panels. {N} queries.\n")
    w(f"- **The layer decides the examples.** Between adjacent layers {', '.join(f'{f1(adjk[k])}% (top {k})' for k in KS)} of the reference slots are replaced, and the ordered list is identical for only {', '.join(f'{f1(idk[k])}% (top {k})' for k in KS)} of the queries.")
    w(f"- **The pooled baseline leaves room mainly at small budgets.** It gives at least one same-class reference to {', '.join(f'{f1(b[k].pct_at_least_one_same_class)}% (top {k})' for k in KS)} of the queries, so {int(b[1].baseline_misses_pooled_vit)} queries can be rescued at top 1 and {int(b[8].baseline_misses_pooled_vit)} at top 8.")
    w("- **Best layers against the pooled baseline.** " + "; ".join(f"top {k}: {NAME[tops[k][0]]} {lname(tops[k][1])}, {pp(row(*tops[k], k).net_coverage_pp_vs_pooled_vit)} pp" for k in KS) + ". They are found on the queries they are scored on.")
    w("- **A small net change hides a large exchange.** ViT layer 10 against the pooled output: " + "; ".join(f"top {k}, {int(v10[k].n_0_to_ge1_vs_final_output)} gained and {int(v10[k].n_ge1_to_0_vs_final_output)} lost ({pp(v10[k].net_coverage_pp_vs_final_output)} pp)" for k in KS) + ".")
    w("- **Same class is not accuracy.** No vision-language model is called. Whether the changed examples improve its answers is not measured here.\n")
    w("## Coverage: pooled baseline, final outputs and best layers\n")
    w("![Coverage](analysis/figures/fig2_coverage_pooled_final_best.png)\n")
    w(f"**Fig. 2 | Share of queries with at least one same-class reference.** **a** to **d**, Dashed line: pooled ViT output. Open circle: the encoder's final output. Filled marker: the layer of that encoder with the largest net coverage change against the pooled output (label: layer). {N} queries per point.\n")
    w("![Gained and lost](analysis/figures/fig4_gained_lost_vs_pooled.png)\n")
    w("**Fig. 3 | Queries gained and lost by each encoder's best layer against the pooled ViT output.** **a** to **d**, Blue: queries for which the layer supplies a same-class reference where the pooled list has none. Orange: queries for which it loses every same-class reference of the pooled list. "
      f"The net coverage change is the difference of the two bars over {N} queries. For the ViT, the pooled output is its own final output.\n")
    w("## Turnover: how much the references change with depth\n")
    w("![Adjacent-layer replacement](analysis/figures/fig3_adjacent_layer_replacement.png)\n")
    w(f"**Fig. 4 | Replacement between adjacent layers.** **a** to **d**, Share of the top-K slots that a layer fills with a reference the previous layer did not retrieve. Denominator: K slots times {N} queries. Shared y axis.\n")
    w("## One layer followed across budgets: ViT layer 10\n")
    w("![ViT layer 10](analysis/figures/fig5_vit_layer10_vs_pooled.png)\n")
    r1, r8 = v10[1], v10[8]
    w(f"**Fig. 5 | ViT layer 10 against the pooled ViT output at each budget.** **a**, The four paired transitions as shares of the {N} queries (numbers are query counts). **b**, Coverage of the two lists; the label is the net change in percentage points. **c**, Share of slots replaced and share of identical ordered lists.\n")
    w(f"From the folders: {f1(r1.pct_at_least_one_same_class)}% against {f1(r1.baseline_coverage_pct_final_output)}% at top 1, and {f2(v10[2].mean_same_class)} against {f2(b[2].mean_same_class)}, {f2(v10[4].mean_same_class)} against {f2(b[4].mean_same_class)} and {f2(r8.mean_same_class)} against {f2(b[8].mean_same_class)} same-class references at top 2, 4 and 8"
      + (", which reproduces the published figures (75% against 71%; 1.46 against 1.36; 2.76 against 2.49; 4.93 against 4.39)." if not DISJ else ". These differ from the earlier leave-one-out figures (75% against 71%; 1.46 against 1.36; 2.76 against 2.49; 4.93 against 4.39) because the references now come from a separate bank.")
      + f" As the budget grows the pooled list misses fewer queries ({int(r1.baseline_misses_final_output)} at top 1, {int(r8.baseline_misses_final_output)} at top 8), so there is less coverage to gain; at top 8 layer 10 differs from the pooled output by {pp(100 * (r8.fraction_of_slots_same_class - b[8].fraction_of_slots_same_class))} points of matching slots.\n")
    w(fold("Tables: ViT layer 10 at every budget, and the leading layers by budget", across_tables()))
    w("## Layer-wise analysis, encoder by encoder\n")
    w("![Layer-wise same class](analysis/figures/fig6_layerwise_same_class.png)\n")
    w(f"**Fig. 6 | Same-class references at every layer of every encoder.** One panel per encoder, one line per budget, real layer numbers on the x axis (ResNet-50: 0 is the stem). Diamonds: the encoder's final output. Share of the top-K slots with the query's class; {N} queries.\n")
    w("![Layer-wise replacement](analysis/figures/fig7_layerwise_replacement.png)\n")
    w("**Fig. 7 | How much each layer changes the references of the layer before it.** Same layout as Fig. 6. Share of the top-K slots that the previous layer did not retrieve.\n")
    w(layer_notes())
    w(layer_tables())
    w("## Agreement across depth and differences between datasets\n")
    w("![Settling and datasets](analysis/figures/fig8_settling_and_datasets.png)\n")
    w("**Fig. 8 | Late settling and dataset effects.** **a**, Share of queries whose ordered top-K list reaches its last-hidden-layer form only at the last layer. **b**, Share of the top-4 slots with the query's class at each encoder's final output, by dataset; bold marks the best encoder of each dataset.\n")
    w("The lists do not reach a plateau: at top 4 and top 8 almost every list still changes at the last layer. This describes agreement between layers, not robustness to changes of the input image.\n")
    w("## A query followed through every layer\n")
    if os.path.exists(os.path.join(ROOT, "assets", "example_query_all_layers.jpg")):
        w("![Example query](assets/example_query_all_layers.jpg)\n"); w("**Fig. 9 | The four references of one query at every layer of every encoder.** Columns are layers. An orange frame marks a reference that the previous layer did not retrieve. A blue disc with a check marks a reference of the query's class. The number below each column counts such references.\n")
    w(examples())
    w("## The four budgets\n"); w("Each budget in five statements; the tables are folded.\n")
    for k in KS: w(section_k(k))
    return "\n".join(L)

def interpretation():
    L = []; w = L.append; w("## Interpretation and next experiment\n"); k = 4; bp = tops[k]; r = row(bp[0], bp[1], k); v = v10[1]
    w(f"**What a VLM would receive.** The layer that feeds the retriever decides a large part of the prompt: between neighboring layers a quarter to a third of the example slots differ on average, and against the pooled output of the published pipeline most lists differ in at least one example. "
      f"Part of that change is useful in the sense measured here. With one example per query, ViT layer 10 gives a same-class example to {int(v.n_0_to_ge1_vs_final_output)} queries that the pooled output leaves without one and takes it away from {int(v.n_ge1_to_0_vs_final_output)}. "
      f"With four examples the best layer here ({NAME[bp[0]]} {lname(bp[1])}) gains {int(r.n_0_to_ge1_vs_pooled_vit)} queries and loses {int(r.n_ge1_to_0_vs_pooled_vit)}. With eight examples almost every query already has a same-class example, so the layer mostly changes how many of the eight match.\n")
    w("**What is not measured.** Downstream performance. A same-class example is a proxy for a useful example. In an earlier run of this project a retriever with more same-class references did not raise the accuracy of the VLM.\n")
    dwd = audit["deepweeds_capture_dates"]
    w(f"**Status of the evidence.** Every query was used to describe every layer, and queries and references are not separated by capture date or capture group (for DeepWeeds, {dwd['share_of_pooled_top4_references_from_the_query_date_pct']}% of the pooled top-4 references come from the capture date of their query). "
      "There is no untouched holdout inside the atlas, so the results are exploratory and no layer is selected. A confirmation needs queries from capture dates, plants or fields that are absent from the atlas, with a reference bank that is disjoint from them by the same unit.\n")
    hp = os.path.join(AN, "deepweeds_holdout", "holdout_result.json")
    if os.path.exists(hp):
        H = json.load(open(hp)); C = json.load(open(os.path.join(AN, "deepweeds_holdout", "config_locked.json"))); c = C["candidate"]; ci = H["ci95_date_cluster_bootstrap_pp"]; dv = C["development"]
        if os.path.exists(os.path.join(AN, "deepweeds_holdout", "fig_coverage.png")):
            w("![Date-disjoint check](analysis/deepweeds_holdout/fig_coverage.png)\n"); w("**Fig. 10 | Date-disjoint check on DeepWeeds, outside the atlas.** **a** to **d**, Development split: queries gained (solid, up) and lost (dashed, down) against the pooled list, by depth and budget; the circle marks the locked candidate. **e**, Held-out result of the locked candidate.\n")
        w(f"**Date-disjoint check outside the atlas** ([summary](analysis/deepweeds_holdout/SUMMARY.md), [locked configuration](analysis/deepweeds_holdout/config_locked.json)). DeepWeeds queries that are not in the atlas were split by capture date. The candidate was chosen on a development split by net coverage change, with the rule and a {C['tolerance_pp']}-point tolerance recorded beforehand: "
          f"`{c['representation']}` ({c['checkpoint']}, {c['readout']} token, block {c['block']}) at K = {c['K']}. Development: {dv['selected_net_coverage_pp']:+.1f} pp (95% CI {dv['selected_cluster_bootstrap_95ci_pp'][0]:+.1f} to {dv['selected_cluster_bootstrap_95ci_pp'][1]:+.1f}). "
          f"Held-out, {H['n_paired']} queries with a date-disjoint bank: {H['added']} queries gained, {H['lost']} lost, net {H['net_coverage_pp']:+.1f} pp (95% CI {ci[0]:+.1f} to {ci[1]:+.1f}; paired cluster bootstrap over {H['independent_date_groups']} capture dates, {C['bootstrap_resamples']} resamples, seed {C['seed']}). "
          f"The held-out result is labeled exploratory there, because layer-wise aggregates on those queries had been read earlier, and {H['independent_date_groups']} date groups are few for a reliable interval.\n")
        ok = C["credible_candidate"] and ci[0] > 0; n = H["n_paired"]
        w((f"**Proposed follow-up.** One same-budget comparison on the {n} held-out DeepWeeds queries with the frozen date-disjoint bank: pooled ViT retrieval against `{c['representation']}`, both with K = {c['K']} examples and the standard assisted prompt. Two conditions per query give {2 * n} calls" + ("; earlier runs on these queries used four examples, so no cached output is compatible and nothing can be reused. " if c["K"] != 4 else f"; cached outputs of the pooled condition for the same queries, bank, prompt and K can be reused and would reduce the request to {n} calls. ") +
           "The claim to test is whether the changed examples improve the answers; the retrieval result alone does not show that.\n") if ok else
          (f"**Proposed follow-up.** The paired retrieval evidence does not yet support a pilot: the interval for the held-out net coverage change includes zero or the development result is not credible. If a pilot is run anyway, the smallest informative design is one same-budget comparison of pooled retrieval and `{c['representation']}` at K = {c['K']} on the {n} held-out queries ({2 * n} calls for two conditions per query).\n"))
    else:
        w(f"**Proposed follow-up.** No candidate is put forward from the atlas alone. A date-disjoint check on DeepWeeds decides whether a pilot is justified; its result is added here when it is available. A same-budget comparison of pooled retrieval and one layer on the atlas queries would cost {2 * N} calls for two conditions per query.\n")
    return "\n".join(L)

def audit_md():
    a = audit; i = a["issues"]; dwd = a["deepweeds_capture_dates"]
    elig = (f"- **Disjoint sets:** each dataset has {next(iter(a['split_images'].values()))['queries']} queries and a separate reference bank of {next(iter(a['split_images'].values()))['references']} images. Retrieved references that are query images: {i['reference_that_is_a_query_image']}. Images listed as both query and reference: {a['images_listed_as_both_query_and_reference']}. "
            f"Query-reference pairs with the same MD5: {a['md5_shared_by_a_query_and_a_reference']}." if DISJ else
            f"- **Leave-one-out pool:** every reference is one of the {a['queries']} pool images of its dataset (references outside the pool: {i['reference_that_is_not_a_query_image']}). A query retrieved as its own reference: {i['self_match_by_file_identity']}. Byte-identical images in the pools: {a['byte_identical_images_among_the_query_images']}.")
    return "\n".join(["## Audit of the atlas\n",
        f"Checked against the folders: {a['queries']} queries, {a['datasets']} datasets, {a['encoders']} encoders, {a['layers_without_final_output']} layers plus one `final_output` per encoder. All {a['encoder_layer_lists_found']:,} encoder-layer lists are present ({a['encoder_layer_folders_expected']:,} expected), with {a['reference_files_expected']:,} reference files. No query is lost to the common cohort.\n",
        f"- **Nested lists:** lists that are not an ordered prefix of their top-8 list: {i['not_a_prefix_of_top8']}; incomplete rank sequences: {i['incomplete_ranks']}; lists with a repeated reference: {i['repeated_reference_in_a_list']}.",
        f"- **Tags:** `match`/`diff` tags that disagree with the classes: {i['match_tag_disagrees_with_classes']}; `first`/`kept`/`new` tags that disagree with a recomputed comparison of adjacent layers: {i['new_kept_tag_disagrees_with_recomputation']}; unparseable file names: {i['unparseable_file_names']}.",
        f"- **Identities:** {a['distinct_canonical_reference_ids']} canonical reference identities (dataset, class, source file name) map to {a['distinct_reference_files']} distinct files; identities with two files: {i['canonical_id_with_two_files']}.",
        elig + " All encoders rank the same bank.",
        f"- **`index.csv`:** {a['index_csv_rows']:,} rows; entries that disagree with the folders: {a['index_csv_entries_that_disagree_with_the_folders']}.",
        f"- **Capture groups:** queries and references are not separated by date or capture group. Only DeepWeeds has a capture date in its file names ({dwd['distinct_query_dates']} dates over {dwd['queries']} queries); {dwd['share_of_pooled_top1_references_from_the_query_date_pct']}% of the pooled top-1 and {dwd['share_of_pooled_top4_references_from_the_query_date_pct']}% of the pooled top-4 references come from the capture date of their query.",
        "- **Baselines:** the pooled method of Arshad et al. is `1_ViT/final_output`: same checkpoint (`google/vit-base-patch16-224-in21k`), readout (`pooler_output`), preprocessing and cosine ranking. For DINOv2, DINOv3 and ResNet-50 the `final_output` equals the last layer; for the other six encoders it is a different vector and is compared separately.\n"])

def reproduction():
    return "\n".join(["## Reproduction\n", "```\ncd AgEval-LayerWise-Reference-Atlas\npython analysis/topk_analysis.py --reparse     # walks the folder tree (about one minute), then tables, figures and this README\npython analysis/topk_analysis.py               # reuses analysis/cache/parsed.npz\npython analysis/render_readme_pdf.py           # README.pdf\n```\n",
        "Requirements: Python 3.9 or later with numpy, pandas and matplotlib. The analysis reads only this folder and makes no model, VLM or API call.\n",
        "- Code: [analysis/topk_analysis.py](analysis/topk_analysis.py) (parsing, audit, metrics), [analysis/topk_report.py](analysis/topk_report.py) (figures, README, checks), [analysis/render_readme_pdf.py](analysis/render_readme_pdf.py).",
        "- Inputs and checks: [input manifest](analysis/input_manifest.json), [audit](analysis/audit.json), [checks](analysis/checks.json).",
        "- Tables: [per query](analysis/tables/per_query_topk.csv.gz) (one row per query, encoder, layer and K), [by layer](analysis/tables/aggregate_by_layer.csv), by dataset ([top 1](analysis/tables/by_dataset_top1.csv), [top 2](analysis/tables/by_dataset_top2.csv), [top 4](analysis/tables/by_dataset_top4.csv), [top 8](analysis/tables/by_dataset_top8.csv)), "
        "[agreement across depth](analysis/tables/consistency_across_depth.csv), [ViT layer 10](analysis/tables/vit_layer10_vs_final_output.csv). Every folded README table is also saved as `analysis/tables/readme_*.csv`.",
        "- Figures: `analysis/figures/fig1` to `fig8`; supplementary depth plots of the paired transitions against [each encoder's final output](analysis/figures/supp_transitions_vs_final_output.png) and against [the pooled ViT](analysis/figures/supp_transitions_vs_pooled_vit.png).",
        "- The folders are built from cached latents by `make_retrieval_samples.py` and the base README by `make_retrieval_readme.py`. Running the second script rewrites this README; run `analysis/topk_analysis.py` afterwards.\n",
        f"Conventions. A reference is identified by dataset, reference class and source file name; rank and the `match`, `diff`, `first`, `kept`, `new` tags are not part of the identity. Turnover at K is K minus the number of shared identities. Coverage is the share of queries with at least one same-class reference; the net coverage change is the `0 to >=1` rate minus the `>=1 to 0` rate over all paired queries. "
        f"Rank-biased overlap (in the tables only) is the extrapolated form for two equal-length lists without ties at depth K with persistence p = {RBO_P} (Webber, Moffat and Zobel 2010). \"Settles\" means that the ordered top-K list of a layer and of every later hidden layer equals the list of the last hidden layer.\n"])

# ------------------------------------------------------------------ write the README: results first, description after
rp = os.path.join(ROOT, "README.md"); txt = open(rp).read(); txt = re.sub(r"<!-- TOPK:\w+:BEGIN.*?-->\n.*?<!-- TOPK:\w+:END -->\n*", "", txt, flags=re.S)
parts = re.split(r"(?m)^## (.+)$", txt); head = parts[0].rstrip(); SEC = {parts[i].strip(): parts[i + 1].strip("\n") for i in range(1, len(parts), 2)}
head = re.sub(r"\n> \*\*Summary\.\*\*.*", "", head, flags=re.S).rstrip()                      # the summary paragraph is rewritten from the current numbers
summary = (f"> **Summary.** Assisted few-shot learning retrieves labeled reference images for a query with a frozen image encoder, and every layer of the encoder retrieves a different set. This atlas lays out the references retrieved by every layer of nine encoders for {SCOPE}. "
           f"Between adjacent layers {f1(adjk[4])}% of the top-4 reference slots are replaced. Against the pooled ViT output of the published pipeline, the best layer changes the share of queries with a same-class reference by {pp(row(*tops[1], 1).net_coverage_pp_vs_pooled_vit)} points at top 1 and {pp(row(*tops[8], 8).net_coverage_pp_vs_pooled_vit)} points at top 8. The atlas covers retrieval only; no vision-language model is involved.")
def keep(t): return f"## {t}\n\n{SEC[t].strip()}\n" if t in SEC else ""
out_md = [head + "\n\n" + summary + "\n", results(), interpretation()]
for t in ("Background", "Encoders and layers", "Organization of the data"): out_md.append(keep(t))
out_md += [audit_md(), keep("Limitations"), keep("Methods"), keep("Data and code availability"), reproduction()]
open(rp, "w").write("\n".join(x for x in out_md if x).rstrip() + "\n")

# ------------------------------------------------------------------ checks
chk = {"paired_queries": N, "protocol": proto}
chk["four_transitions_sum_to_the_paired_query_count"] = bool(all(np.allclose(AGG[f"n_0_to_0_vs_{nm}"] + AGG[f"n_0_to_ge1_vs_{nm}"] + AGG[f"n_ge1_to_0_vs_{nm}"] + AGG[f"n_ge1_to_ge1_vs_{nm}"], AGG.n_queries) for nm in ("final_output", "pooled_vit")))
chk["transition_rates_reproduce_the_coverage_difference"] = bool(all(np.allclose(AGG[f"net_coverage_pp_vs_{nm}"], AGG.pct_at_least_one_same_class - AGG[f"baseline_coverage_pct_{nm}"]) for nm in ("final_output", "pooled_vit")))
cov = AGG.pivot_table(index=["encoder", "layer"], columns="K", values="pct_at_least_one_same_class"); chk["coverage_never_falls_with_larger_K"] = bool(((cov[2] >= cov[1]) & (cov[4] >= cov[2]) & (cov[8] >= cov[4])).all())
pq_cov = PQ.pivot_table(index=["dataset", "query", "encoder", "layer"], columns="K", values="same_class"); chk["per_query_at_least_one_match_is_monotone_in_K"] = bool((((pq_cov[1] > 0) <= (pq_cov[2] > 0)) & ((pq_cov[2] > 0) <= (pq_cov[4] > 0)) & ((pq_cov[4] > 0) <= (pq_cov[8] > 0))).all())
for nm in ("final_output", "pooled_vit"):
    i = PQ[PQ[f"identical_order_vs_{nm}"]]; chk[f"identical_rankings_vs_{nm}"] = {"rows": int(len(i)), "same_class_counts_equal": bool((i.same_class == i[f"same_class_{nm}"]).all()), "rbo_is_1": bool((i[f"rbo_vs_{nm}"] == 1).all()), "replaced_is_0": bool((i[f"replaced_vs_{nm}"] == 0).all()),
                                                 "transition_is_on_the_diagonal": bool(i[f"transition_vs_{nm}"].isin(["0 to 0", ">=1 to >=1"]).all())}
readme = open(rp).read(); bad = []
for nm, T in TABLES.items():                      # every README table equals its saved CSV
    seg = readme.split(f"<!-- table:{nm} -->\n", 1)[1].split("\n\n", 1)[0].strip().split("\n"); cells = [[c.strip() for c in ln.strip().strip("|").split("|")] for ln in seg[2:]]
    ref = pd.read_csv(os.path.join(TB, f"readme_{nm}.csv"), dtype=str, keep_default_na=False).values.tolist()
    if cells != [[str(c) for c in r] for r in ref]: bad.append(nm)
chk["readme_tables_checked_against_saved_csv"] = len(TABLES); chk["readme_tables_that_differ_from_their_csv"] = bad
for f in os.listdir(TB):
    if f.startswith("readme_") and f[7:-4] not in TABLES: os.remove(os.path.join(TB, f))
links = re.findall(r"\]\(([^)#]+)\)", readme); self_made = {"analysis/checks.json", "analysis/input_manifest.json"}
chk["relative_links_checked"] = len(links); chk["relative_links_that_do_not_resolve"] = sorted({l for l in links if not l.startswith("http") and l not in self_made and not os.path.exists(os.path.join(ROOT, l))})
V = V10.set_index(["layer", "K"]); chk["vit_layer10_vs_final_output"] = {"top1_coverage_pct": [round(float(V.loc[("layer_10", 1)].pct_at_least_one_same_class), 1), round(float(V.loc[("final_output", 1)].pct_at_least_one_same_class), 1)],
                                                                         **{f"top{k}_mean_same_class": [round(float(V.loc[("layer_10", k)].mean_same_class), 2), round(float(V.loc[("final_output", k)].mean_same_class), 2)] for k in (2, 4, 8)},
                                                                         "published_for_the_leave_one_out_pools": "75 vs 71; 1.46 vs 1.36; 2.76 vs 2.49; 4.93 vs 4.39"}
json.dump(chk, open(os.path.join(AN, "checks.json"), "w"), indent=1)
json.dump({"atlas_root": ".", "read": ["<dataset>/<query>/<encoder>/<layer>/top{1,2,4,8}/ (file names only)", "<dataset>/<query>/QUERY__* (file identity and MD5)", "index.csv (cross-check)", "split.json and split_images.csv (retrieval protocol)", "README.md (description sections are kept)"],
           "datasets": sorted(Q.dataset.unique()), "queries": int(nq), "encoders": ENCS, "layers": LAYERS, "K": list(KS), "rbo_persistence": RBO_P, "random_seed": None, "note": "the analysis is deterministic; no resampling is used inside the atlas",
           "command": "python analysis/topk_analysis.py --reparse", "generated_at": time.strftime("%Y-%m-%d %H:%M:%S")}, open(os.path.join(AN, "input_manifest.json"), "w"), indent=1)
print(json.dumps(chk, indent=1))
