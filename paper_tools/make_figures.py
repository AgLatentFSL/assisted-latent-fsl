"""Builds every data figure and image asset of the paper from the saved run directories. No model or VLM calls.

Usage (from the paper root):  <python> tools/make_figures.py [RUNS_DIR] [DATA_DIR]
Defaults point at the pilot checkout: code/ageval-dinov2-pilot/{runs,repo/data}.
Style: bioMoR Figure 5 conventions (tools/biomor_style.py). All rasters are saved at 600 dpi.
"""
import json, os, sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch, Rectangle
from PIL import Image
sys.path.insert(0, os.path.dirname(__file__))
import biomor_style as BS

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PILOT = HERE
RUNS = sys.argv[1] if len(sys.argv) > 1 else os.path.join(PILOT, "runs")
DATA = sys.argv[2] if len(sys.argv) > 2 else os.path.join(PILOT, "data")
FIG = os.path.join(HERE, "figs")
BS.apply()
P = BS.P
# one meaning per color across the paper
C_ZS, C_VLM = P["ref"], P["lavender"]          # zero-shot, VLM with demonstrations
C_DD, C_SD = P["blue"], P["salmon"]                                # date-disjoint bank, same-date bank


def save(fig, name):
    fig.savefig(os.path.join(FIG, name + ".pdf"), bbox_inches="tight", pad_inches=0.02)
    fig.savefig(os.path.join(FIG, name + ".png"), dpi=600, bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)


# ------------------------------------------------------------------ main result (grouped bars, full width)
C_RAND, C_ARB, C_DINO = P["blue"], P["lavender"], P["green"]
DS_ORDER = ["Bean Leaf Lesions", "Mango Leaf Disease", "DeepWeeds", "Soybean Seeds", "Soybean Diseases"]
AUDIT = os.path.join(RUNS, "layer_retrieval_audit")
VITKEY = "vit"   # pooled ViT output
TEASER_QUERY = "bean_rust_train.195.jpg"   # zero-shot answers Angular Leaf Spot; assisted 4-shot answers Bean Rust (runs/arbab_eval_seed42)


def arbab():
    R = pd.read_csv(os.path.join(RUNS, "arbab_eval_seed42", "per_query_results.csv"), keep_default_na=False)
    R = R[R.dataset.isin(DS_ORDER)].reset_index(drop=True)   # the paper reports the five datasets in DS_ORDER
    for c, col in (("Z", "pred_Z"), ("R", "pred_R"), ("A", "pred_A"), ("D", "pred_D")):
        R["ok_" + c] = R[col].astype(str) == R["true_label"].astype(str)
    return R


def fig_main():
    R = arbab()
    fig, ax = plt.subplots(figsize=(6.9, 1.75))
    conds = [("Z", C_ZS, "Zero-shot"), ("R", C_RAND, "Random 4-shot"), ("A", C_ARB, "Assisted 4-shot, ViT pooled (Arshad et al.)")]
    groups = DS_ORDER + [f"All {len(R)} queries"]
    for i, ds in enumerate(groups):
        g = R if ds.startswith("All") else R[R.dataset == ds]
        for j, (c, col, _) in enumerate(conds):
            v = 100 * g["ok_" + c].mean(); x = i + (j - 1) * 0.26
            BS.bar(ax, x, v, 0.25, col); BS.vlabel(ax, x, v, f"{v:.0f}" if not ds.startswith("All") else f"{v:.1f}", fs=5.0, pad=1.0)
    ax.axvline(len(groups) - 1.5, color="#B0B0B0", lw=0.6, ls=":")
    ax.set_xticks(range(len(groups)), [g.replace(" ", "\n", 1) if not g.startswith("All") else f"All {len(R)}\nqueries" for g in groups]); ax.set_ylim(0, 112); ax.set_yticks(range(0, 101, 20))
    ax.set_ylabel("Accuracy (%)"); ax.set_xlim(-0.55, len(groups) - 0.45)
    BS.top_legend(ax, [BS.patch(col, lab) for _, col, lab in conds], ncol=3, y=1.0)
    save(fig, "fig_main")


# ------------------------------------------------------------------ accuracy against the number of same-label references (dot and line)
def fig_dose():
    R = arbab()
    Pd = R[["same_A", "ok_A", "ok_Z"]].set_axis(["same", "ok", "z"], axis=1)
    g = Pd.groupby("same").agg(n=("ok", "size"), acc=("ok", "mean"), z=("z", "mean")); out = {int(k): (int(r.n), round(100 * r.acc, 1), round(100 * r.z, 1)) for k, r in g.iterrows()}
    fig, ax = plt.subplots(figsize=(3.3, 1.75))
    ax.fill_between(g.index, 100 * g.z, 100 * g.acc, color=C_VLM, alpha=0.22, lw=0)
    ax.plot(g.index, 100 * g.z, color="#8C8C8C", lw=1.2, marker="s", ms=3.2, label="Zero-shot, same queries")
    ax.plot(g.index, 100 * g.acc, color="#7A5BA6", lw=1.4, marker="o", ms=3.6, label="Assisted 4-shot")
    for k, r in g.iterrows():
        ax.text(k, 100 * r.acc + 3.0, f"{100 * r.acc:.0f}", ha="center", fontsize=5.6, color="#5B4282"); ax.text(k, 100 * r.z - 6.5, f"{100 * r.z:.0f}", ha="center", fontsize=5.6, color="#666666")
        ax.text(k, 6, f"n={int(r.n)}", ha="center", fontsize=5.0, color="#777777")
    ax.set_xticks(range(5)); ax.set_xlabel("Retrieved references with the query label (of 4)"); ax.set_ylabel("Accuracy (%)"); ax.set_ylim(0, 92); ax.set_xlim(-0.35, 4.35)
    ax.legend(loc="upper left", fontsize=5.4)
    save(fig, "fig_dose"); return out


# ------------------------------------------------------------------ per-class heat map of same-label references
def fig_leak_heat():
    pc = pd.read_csv(os.path.join(RUNS, "deepweeds_split_audit", "per_class.csv"), header=[0, 1, 2], index_col=0)
    cls = sorted(pc.index)
    cols = [("vit", "DD"), ("vit", "SD")]
    A = np.array([[pc.loc[c, ("same_class", bt, enc)] for enc, bt in cols] for c in cls])
    G = np.stack([A[:, 1] - A[:, 0]], axis=1)
    order = np.argsort(-G.mean(axis=1))
    A, G, cls = A[order], G[order], [cls[i] for i in order]
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(3.3, 2.05), gridspec_kw={"width_ratios": [2, 1], "wspace": 0.08})
    for ax, M, cm, vmin, vmax in ((a1, A, "YlGnBu", 0, 4), (a2, G, "OrRd", 0, 1.6)):   # vector cells (no embedded raster)
        ax.pcolormesh(np.arange(M.shape[1] + 1) - 0.5, np.arange(M.shape[0] + 1) - 0.5, M, cmap=cm, vmin=vmin, vmax=vmax, edgecolors="white", linewidth=0.6)
        ax.set_ylim(M.shape[0] - 0.5, -0.5)
    for ax, M, labs in ((a1, A, ["Date-disjoint\nbank", "Same-date\nbank"]), (a2, G, ["Gain"])):
        ax.grid(False); ax.set_xticks(range(M.shape[1]), labs, fontsize=5.0); ax.xaxis.tick_top(); ax.tick_params(length=0)
        for s in ax.spines.values():
            s.set_visible(False)
        for i in range(M.shape[0]):
            for j in range(M.shape[1]):
                v = M[i, j]
                dark = v > (2.4 if ax is a1 else 0.95)
                ax.text(j, i, (f"{v:.1f}" if ax is a1 else f"{v:+.1f}"), ha="center", va="center", fontsize=5.4, color="white" if dark else BS.INK)
    a1.set_yticks(range(len(cls)), cls, fontsize=5.6); a2.set_yticks([])
    fig._ca_lettered = True
    a1.set_xlabel("Same-label references (of 4)", fontsize=6.0); a2.set_xlabel("Same-date minus\ndisjoint", fontsize=6.0)
    save(fig, "fig_leak_heat"); return {c: [round(float(x), 2) for x in g] for c, g in zip(cls, G)}


# ------------------------------------------------------------------ layer sweep from the retrieval audit (lines)
def fig_layers():
    PM = pd.read_csv(os.path.join(AUDIT, "layer_pair_metrics.csv")); PM = PM[PM.split.isin([d + " | loader pool" for d in DS_ORDER])]
    VN = PM[PM.comparison == "block versus native baseline"]; VA = PM[PM.comparison == "Arshad pooled ViT versus representation"]
    qw = lambda d, c: float((d[c] * d.n_queries).sum() / d.n_queries.sum()); out = {}
    fig, ax = plt.subplots(figsize=(3.3, 1.95))
    for fam, col, ls, lab in (("vit|cls", "#7A5BA6", "-", "ViT, CLS"), ("vit|mean", "#A38FCB", "--", "ViT, patch mean"),
                               ("clip|cls", "#C4682B", "-", "CLIP, CLS"), ("clip|mean", "#E4B04F", "--", "CLIP, patch mean")):
        v = [qw(VN[(VN.family == fam) & (VN.block_a == b)], "same_label_of4_a") for b in range(1, 13)]; out[fam] = [round(x, 2) for x in v]
        ax.plot(range(1, 13), v, color=col, ls=ls, lw=1.2, marker="o", ms=2.0, label=lab)
    pool = qw(VA.drop_duplicates("split"), "same_label_of4_a"); out["vit|native"] = round(pool, 2)
    for e in ("clip", "resnet"): out[e + "|native"] = round(qw(VA[VA.rep_b == e + "|native"], "same_label_of4_b"), 2)
    rn = VN[VN.family == "resnet|gap"]; out["resnet|gap"] = [round(qw(rn[rn.block_a == b], "same_label_of4_a"), 2) for b in range(17)]
    ax.axhline(pool, color="#909090", lw=0.8, ls=":", label=f"ViT pooled output (Arshad et al.), {pool:.2f}")
    ax.set_xticks(range(1, 13)); ax.set_xlabel("Transformer block"); ax.set_ylabel("Same-label references (of 4)")
    ax.set_ylim(1.55, 2.85); fig.legend(*ax.get_legend_handles_labels(), loc="upper center", bbox_to_anchor=(0.54, 1.12), fontsize=4.8, ncol=3, columnspacing=0.9, handlelength=1.8)
    save(fig, "fig_layers"); return out


# ------------------------------------------------------------------ teaser thumbnails (Bean Leaf Lesions query and its four references of the pooled ViT output)
def teaser_assets():
    pool = pd.read_csv(os.path.join(RUNS, "offline_layer_stats", "pool_Bean_Leaf_Lesions.csv"))
    E = np.load(os.path.join(RUNS, "offline_layer_stats", "embeddings", "Bean_Leaf_Lesions.npz"))[VITKEY].astype(np.float64)
    E /= np.linalg.norm(E, axis=1, keepdims=True)
    q = [i for i, p_ in enumerate(pool["path"]) if os.path.basename(p_) == TEASER_QUERY][0]
    s_ = E @ E[q]; s_[q] = -np.inf; top = sorted(range(len(pool)), key=lambda j: (-s_[j], j))[:4]
    assert all(pool["label"][j] == "Bean Rust" for j in top)
    os.makedirs(os.path.join(FIG, "teaser"), exist_ok=True)
    opn = lambda i: Image.open(os.path.join(PILOT, pool["path"][i])).convert("RGB").resize((600, 600), Image.LANCZOS)
    opn(q).save(os.path.join(FIG, "teaser", "query.jpg"), quality=92)
    for k, j in enumerate(top): opn(j).save(os.path.join(FIG, "teaser", f"ref{k + 1}.jpg"), quality=92)


if __name__ == "__main__":
    fig_main(); print("accuracy by same-label references (n, assisted, zero-shot):", fig_dose()); print("same-date gain per class (ViT):", fig_leak_heat())
    print("layer sweep:", fig_layers()); teaser_assets()
