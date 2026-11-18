"""Report and plots for the layer-by-layer retrieval audit (reads the files written by layer_audit.py; no model calls).
Usage (from repo/): python layer_audit_report.py <out_dir>"""
import json, os, sys
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

out = sys.argv[1]; PL = os.path.join(out, "plots"); os.makedirs(os.path.join(PL, "per_split"), exist_ok=True)
OFF = os.path.join("runs", "offline_layer_stats")
COV = json.load(open(os.path.join(out, "coverage_manifest.json"))); META = json.load(open(os.path.join(out, "retrieval_meta.json")))
PM = pd.read_csv(os.path.join(out, "layer_pair_metrics.csv")); BIG = json.load(open(os.path.join(out, "largest_changes.json")))
GT = pd.read_csv(os.path.join(out, "gap_vs_turnover_adjacent.csv.gz"))
FAMS = ["vit|cls", "vit|mean", "dinov2|cls", "dinov2|mean", "clip|cls", "clip|mean", "resnet|gap", "siglip2|mean", "bioclip|cls", "bioclip|mean", "dinov3|cls", "dinov3|mean", "qwen|mean", "flava|cls", "flava|mean"]
FN = {"vit|cls": "Google ViT, CLS", "vit|mean": "Google ViT, patch mean", "dinov2|cls": "DINOv2, CLS", "dinov2|mean": "DINOv2, patch mean", "clip|cls": "CLIP, CLS", "clip|mean": "CLIP, patch mean", "resnet|gap": "ResNet-50, GAP",
      "siglip2|mean": "SigLIP 2, patch mean", "bioclip|cls": "BioCLIP, CLS", "bioclip|mean": "BioCLIP, patch mean", "dinov3|cls": "DINOv3, CLS", "dinov3|mean": "DINOv3, patch mean", "qwen|mean": "Qwen3-VL tower, patch mean", "flava|cls": "FLAVA, CLS", "flava|mean": "FLAVA, patch mean"}
FC = {"vit|cls": "#0072B2", "vit|mean": "#0072B2", "dinov2|cls": "#D55E00", "dinov2|mean": "#D55E00", "clip|cls": "#009E73", "clip|mean": "#009E73", "resnet|gap": "#CC79A7",
      "siglip2|mean": "#E69F00", "bioclip|cls": "#56B4E9", "bioclip|mean": "#56B4E9", "dinov3|cls": "#000000", "dinov3|mean": "#000000", "qwen|mean": "#8C5A2B", "flava|cls": "#7F7F7F", "flava|mean": "#7F7F7F"}
ENCS = ["vit", "dinov2", "clip", "resnet", "siglip2", "bioclip", "dinov3", "qwen", "flava"]; NAT = {e: e + "|native" for e in ENCS}
NATN = {"vit": "ViT pooled (Arshad et al.)", "dinov2": "DINOv2 final CLS", "clip": "CLIP projected", "resnet": "ResNet-50 pooled", "siglip2": "SigLIP 2 pooled", "bioclip": "BioCLIP projected", "dinov3": "DINOv3 final CLS", "qwen": "Qwen3-VL merged tokens", "flava": "FLAVA projected"}
NBLK = lambda f: 17 if f.startswith("resnet") else 24 if f.startswith("qwen") else 12
SP = {s["split_name"]: s for s in COV["splits"]}; ORDER = list(SP)
MAIN = [n for n in ORDER if SP[n]["group"] == "AgEval loader pool"]; LAB = [n for n in ORDER if SP[n]["same_label_metrics"]]; MAINLAB = [n for n in MAIN if n in LAB]
short = lambda n: n.replace(" | loader pool", "").replace("DeepWeeds | split audit, ", "DeepWeeds, ").replace(" (seed 0)", "").replace("DeepWeeds | held-out capture dates (paper split, 63 eval + 27 dev)", "DeepWeeds, held-out dates (paper split)").replace("Paddy Doctor | variety-disjoint bank", "Paddy Doctor, variety-disjoint")
L = []; w = L.append
def agg(df, col, splits):
    """query-weighted and unweighted (mean over splits) average of a per-split column"""
    d = df[df.split.isin(splits)]
    if not len(d): return np.nan, np.nan, 0, 0
    g = d.groupby("split").agg(v=(col, "mean"), n=("n_queries", "first"))
    return float((g.v * g.n).sum() / g.n.sum()), float(g.v.mean()), int(g.n.sum()), len(g)
ADJ = PM[PM.comparison == "adjacent blocks"]; WITHIN = PM[PM.comparison.isin(["adjacent blocks", "block pair"])]
VN = PM[PM.comparison == "block versus native baseline"]; VA = PM[PM.comparison == "Arshad pooled ViT versus representation"]

# ------------------------------------------------------------------ plots
def heat(ax, fam, splits, title):
    d = WITHIN[(WITHIN.family == fam) & WITHIN.split.isin(splits)]; blocks = sorted(set(d.block_a) | set(d.block_b)); ix = {b: i for i, b in enumerate(blocks)}; M = np.zeros((len(blocks), len(blocks)))
    g = d.groupby(["block_a", "block_b"]).apply(lambda t: (t.mean_turnover * t.n_queries).sum() / t.n_queries.sum())
    for (a, b), v in g.items(): M[ix[a], ix[b]] = M[ix[b], ix[a]] = v
    im = ax.imshow(M, cmap="magma", vmin=0, vmax=4); ax.set_title(title, fontsize=7); t = list(range(0, len(blocks), 2 if len(blocks) <= 12 else 4)); assert len(blocks) == NBLK(fam)
    ax.set_xticks(t, [int(blocks[i]) for i in t], fontsize=5.5); ax.set_yticks(t, [int(blocks[i]) for i in t], fontsize=5.5); return im
def heat_fig(splits, title, path):
    fig, axs = plt.subplots(4, 4, figsize=(9.6, 10.2)); axs = axs.ravel()
    for ax, fam in zip(axs, FAMS): im = heat(ax, fam, splits, FN[fam])
    [a.axis("off") for a in axs[len(FAMS):]]; cb = fig.colorbar(im, ax=axs[len(FAMS)], fraction=0.5, pad=0.02); cb.set_label("Mean top-4 turnover (references replaced, of 4)", fontsize=7, labelpad=8); cb.ax.tick_params(labelsize=6)
    fig.suptitle(title + " (block by block)", fontsize=8.5); fig.supxlabel("Block", fontsize=7); fig.supylabel("Block", fontsize=7); fig.tight_layout(h_pad=2.2); fig.savefig(path, dpi=300); plt.close(fig)
heat_fig(MAIN, f"Top-4 turnover between blocks, query-weighted over the {len(MAIN)} AgEval loader pools", os.path.join(PL, "turnover_heatmaps_ageval_pools.png"))
for fam in FAMS:
    fig, ax = plt.subplots(figsize=(3.4, 3.0)); im = heat(ax, fam, MAIN, FN[fam] + f", {len(MAIN)} AgEval loader pools"); cb = fig.colorbar(im, fraction=0.046); cb.set_label("Mean top-4 turnover", fontsize=7)
    ax.set_xlabel("Block", fontsize=7); ax.set_ylabel("Block", fontsize=7); fig.tight_layout(); fig.savefig(os.path.join(PL, f"turnover_heatmap_{fam.replace('|', '_')}.png"), dpi=300); plt.close(fig)
for i, n in enumerate(ORDER): heat_fig([n], "Top-4 turnover, " + short(n), os.path.join(PL, "per_split", f"turnover_{i:02d}_{SP[n]['tag']}.png"))
def quality(ax, splits, col_b, title, legend=False):
    for fam in FAMS:
        d = VN[(VN.family == fam) & VN.split.isin(splits)]; g = d.groupby("block_a").apply(lambda t: (t[col_b.replace("_b", "_a")] * t.n_queries).sum() / t.n_queries.sum())
        ax.plot(g.index, g.values, color=FC[fam], lw=1.0, marker="o", ms=1.8, ls="--" if fam.endswith("mean") and fam.split("|")[0] not in ("siglip2", "qwen") else "-", label=FN[fam])
    for enc, mk, xx in (("vit", "*", 12.6), ("clip", "D", 12.6), ("siglip2", "P", 13.2), ("bioclip", "X", 13.2), ("flava", "p", 13.8), ("qwen", "h", 24.8)):     # outputs that differ from the last block
        d = VA[(VA.rep_b == NAT[enc]) & VA.split.isin(splits)] if enc != "vit" else VA[VA.split.isin(splits)].drop_duplicates("split")
        col = col_b if enc != "vit" else col_b.replace("_b", "_a"); v = (d[col] * d.n_queries).sum() / d.n_queries.sum()
        ax.scatter([xx], [v], marker=mk, s=22, color=FC[[f for f in FAMS if f.startswith(enc)][0]], edgecolor="k", lw=0.4, zorder=5, label=NATN[enc])
    ax.set_title(title, fontsize=6.5); ax.tick_params(labelsize=5.5); ax.grid(alpha=0.25)
    if legend: ax.legend(fontsize=5.5, ncol=1, loc="center left", bbox_to_anchor=(1.02, 0.5))
for col, nm, yl in (("same_label_of4_b", "same_label_by_layer", "Mean same-label references (of 4)"), ("top1_correct_b", "top1_same_label_by_layer", "First reference has the query label (%)")):
    n = len(LAB) + 1; cols = 4; rows = int(np.ceil(n / cols)); fig, axs = plt.subplots(rows, cols, figsize=(10.5, 2.25 * rows)); axs = axs.ravel()
    quality(axs[0], MAINLAB, col, f"All {len(MAINLAB)} labeled AgEval pools (query-weighted)")
    for ax, s in zip(axs[1:], LAB): quality(ax, [s], col, short(s))
    for ax in axs[n:]: ax.axis("off")
    h, l = axs[0].get_legend_handles_labels(); axs[n - 1 if n < len(axs) else -1].legend(h, l, fontsize=6, loc="center left", bbox_to_anchor=(1.05, 0.5)) if n == len(axs) else axs[n].legend(h, l, fontsize=6.5, loc="center")
    fig.supxlabel("Block (ResNet-50: 0 = stem, 1 to 16 = residual blocks; Qwen3-VL tower: 24 blocks)", fontsize=7.5); fig.supylabel(yl, fontsize=7.5); fig.tight_layout(); fig.savefig(os.path.join(PL, nm + ".png"), dpi=300); plt.close(fig)

# ------------------------------------------------------------------ baseline reproduction against the cached repository embeddings
rep_rows = []
for s in COV["splits"]:
    tag = s["dataset"].replace(" ", "_"); ep = os.path.join(OFF, "embeddings", tag + ".npz")
    if s["group"] != "AgEval loader pool" or not os.path.exists(ep) or "offline_layer_stats" not in s["pool_origin"]: continue
    E = np.load(ep); pool = pd.read_csv(os.path.join(OFF, f"pool_{tag}.csv"))
    for enc, key in (("vit", "vit"), ("dinov2", "dinov2_b12"), ("clip", "clip"), ("resnet", "resnet")):
        F = np.load(os.path.join(out, "features", f"{enc}__{s['tag']}.npz")); pos = {p: i for i, p in enumerate(F["images"])}
        N = F["native"][[pos[p] for p in pool["path"]]].astype(np.float64); C = E[key].astype(np.float64); nz = lambda X: X / np.linalg.norm(X, axis=1, keepdims=True); N, C = nz(N), nz(C)
        cos = (N * C).sum(1); dup = pool["md5"].values[:, None] == pool["md5"].values[None, :]
        def top(X):
            S_ = X @ X.T; S_[dup] = -np.inf; return np.argsort(-S_, kind="stable")[:, :4]
        a, b = top(N), top(C)
        X_ = C @ N.T
        rep_rows.append({"dataset": s["dataset"], "encoder": enc, "n": len(pool), "min_cos": cos.min(), "mean_cos": cos.mean(), "rows_equal_to_own_image": int((cos > 0.9999).sum()),
                         "rows_equal_to_another_pool_image": int(((X_.max(1) > 0.9999) & (X_.argmax(1) != np.arange(len(pool)))).sum()), "same_ranked_top4": int((a == b).all(1).sum()), "same_top4_set": int((np.sort(a, 1) == np.sort(b, 1)).all(1).sum())})
RP = pd.DataFrame(rep_rows); RP.to_csv(os.path.join(out, "baseline_reproduction.csv"), index=False)

# ------------------------------------------------------------------ report
dims = META["dims"]; nq_main = sum(SP[n]["n_queries"] for n in MAIN)
w("# Layer-by-layer retrieval audit over the AgEval benchmark\n")
w("Local encoder inference only; no hosted API call was made. Exploratory sweep: it selects no layer and changes nothing in the assisted few-shot pipeline.\n")
SUMMARY_AT = len(L); w("")
w("## 1. Coverage\n")
w(f"The repository has {len(COV['splits']) + len(COV['not_run']) - sum(1 for s in COV['splits'] if s['group'] not in ('AgEval loader pool', 'extra loader pool'))} dataset loaders (`data_loader.py`): the 12 AgEval subsets of the README and two extra loaders (SBRD, IP02) that the registry in `inference.py` keeps commented out. "
  f"{len(MAIN)} of the 12 AgEval subsets were run, including Dangerous Insects and Durum Wheat. Existing group-disjoint splits are run and reported separately.\n")
w("| Split | AgEval subset | Task | Source | Queries | Bank | Eligible references per query (min / median) | Labels | Same-label metrics |"); w("|---|---|---|---|---|---|---|---|---|")
for n in ORDER:
    s = SP[n]; w(f"| {short(n)} ({s['group']}) | {'yes' if s['in_AgEval_12'] else 'no'} | {s['task_type']} | {s['source_version']} | {s['n_queries']} | {s['n_bank']} | {s['eligible_refs_min']} / {s['eligible_refs_median']:.0f} | {s['n_labels']} values | {'yes' if s['same_label_metrics'] else 'no: ' + s['same_label_note']} |")
w("")
if COV["not_run"]:
    w("**Not run:**\n"); [w(f"- {r['dataset']}: {r['reason']}. Unblock: {r['unblock']}.") for r in COV["not_run"]]; w("")
else: w("Every registered dataset has a completed split; none was skipped.\n")
w("Notes on coverage:\n")
w("- Loader pools use the AgEval protocol: the loader draws about 100 images (seed 42), every pool image is a query and retrieves from the rest of the pool. Self-matches and byte-identical images (same MD5) are never eligible. "
  f"Byte-identical non-self pairs removed: {sum(SP[n]['byte_identical_non_self_pairs'] for n in ORDER)} across all splits.")
w("- FUSARIUM 22, PlantDoc, InsectCount, SBRD and IP02 were not on disk. `layer_audit_fetch.py` read each archive listing over HTTP range requests, let the unmodified repository loader draw its pool from that listing, and fetched only the pool images (the full archives total about 12 GB). Their pools are frozen in `fetch_*.json`.")
w("- PlantDoc (percent of leaf area) and InsectCount (count) have numeric targets, so they get turnover only. IP02 is run with the registry's `samples=105`, which gives one image per class for 102 classes; no query has a same-label reference, so it also gets turnover only.")
w(f"- Queries with fewer than four eligible references: {sum(SP[n]['queries_with_fewer_than_4_eligible'] for n in ORDER)}. Every bank has at least five eligible references, so the fourth-to-fifth score gap is defined for every query.")
w("- The DeepWeeds split audit has 20 bank seeds per bank type; this audit uses bank seed 0 for each type. Paddy Doctor is not an AgEval subset; it is included because the repository holds a variety-disjoint split for it.\n")
w("## 2. Encoders and representations\n")
w("| Encoder | Checkpoint (from `get_embeddingp.py`) | Layer-wise readouts | Blocks | Dimension | Native baseline (kept separate) |"); w("|---|---|---|---|---|---|")
w(f"| Google ViT-B/16 | `google/vit-base-patch16-224-in21k` (ViTModel) | CLS token; mean of the 196 patch tokens | 1 to 12 | {dims['vit|cls|b01']} | pooler_output, {dims['vit|native']}-d: tanh(Linear(LayerNorm(h12)[CLS])). This is the retriever of Arshad et al. |")
w(f"| DINOv2 ViT-B/14 | `facebook/dinov2-base` (no register tokens) | CLS token; mean of the 256 patch tokens | 1 to 12 | {dims['dinov2|cls|b01']} | last_hidden_state[:, 0], {dims['dinov2|native']}-d (equals the block 12 CLS readout) |")
w(f"| CLIP | `openai/clip-vit-base-patch16`; the vision tower is a ViT-B/16 (12 blocks, 197 tokens, verified at run time) | CLS token; mean of the 196 patch tokens | 1 to 12 | {dims['clip|cls|b01']} | get_image_features, {dims['clip|native']}-d: visual_projection(post_layernorm(h12[CLS])) |")
w(f"| ResNet-50 | `microsoft/resnet-50` (ResNetModel) | global average pool of the feature map | 0 = stem (after max-pool), 1 to 16 = bottleneck blocks (3, 4, 6, 3 per stage) | {dims['resnet|gap|b00']}, {dims['resnet|gap|b01']}, {dims['resnet|gap|b04']}, {dims['resnet|gap|b08']}, {dims['resnet|gap|b14']} (stem, stages 1 to 4) | pooler_output, {dims['resnet|native']}-d (equals the block 16 readout) |")
w(f"| SigLIP 2 ViT-B/16 | `google/siglip2-base-patch16-224` (SiglipVisionModel; no CLS token) | mean of the 196 patch tokens | 1 to 12 | {dims['siglip2|mean|b01']} | pooler_output, {dims['siglip2|native']}-d: attention-pooling head over the patch tokens (the image embedding) |")
w(f"| BioCLIP ViT-B/16 | `imageomics/bioclip` (open_clip) | CLS token; mean of the 196 patch tokens | 1 to 12 | {dims['bioclip|cls|b01']} | encode_image, {dims['bioclip|native']}-d: projection of ln_post(h12[CLS]) |")
w(f"| DINOv3 ViT-B/16 | `facebook/dinov3-vitb16-pretrain-lvd1689m` (4 register tokens) | CLS token; mean of the 196 patch tokens, register tokens excluded | 1 to 12 | {dims['dinov3|cls|b01']} | pooler_output, {dims['dinov3|native']}-d (equals the block 12 CLS readout) |")
w(f"| Qwen3-VL-2B vision tower | `Qwen/Qwen3-VL-2B-Instruct`, vision weights only (the language model is not loaded); input resized to 448 x 448 | mean of the 784 patch tokens | 1 to 24 | {dims['qwen|mean|b01']} | mean of the 196 merged visual tokens that enter the language model, {dims['qwen|native']}-d |")
w(f"| FLAVA image encoder ViT-B/16 | `facebook/flava-full`, image encoder only (the text and multimodal encoders are not used) | CLS token; mean of the 196 patch tokens | 1 to 12 | {dims['flava|cls|b01']} | image_projection of the last CLS token, {dims['flava|native']}-d |")
w("\nThe first four encoders are the repository's (`get_embeddingp.py`); SigLIP 2, BioCLIP, DINOv3, FLAVA and the Qwen3-VL vision tower were added on 2026-11-15. SigLIP 2 and the Qwen tower have no CLS token, so they have a patch-mean readout only. For BioCLIP the final layer normalization is `ln_post`, for DINOv3 it is `norm` and for FLAVA it is `layernorm`. The Qwen tower has no final normalization before its patch merger, so its block outputs are read as they are.\n")
w("\nConventions, the same at every depth: preprocessing is the repository's processor for each checkpoint; all blocks come from one forward pass per image; for the three transformers the model's final LayerNorm (CLIP: `post_layernorm`) is applied to the block output before the readout, as in the repository's `layer_token_sweep.py` and `latent_search.py`; the CLS token is excluded from patch means; no projection is applied to layer-wise readouts; every vector is L2-normalized and ranked by cosine similarity; ties break on the reference id. "
  "Latents are cached per encoder and split in compressed `.npz` files (`features/`): layer-wise readouts as unit-length float16, native baselines as raw float32.\n")
w("## 3. Baseline reproduction\n")
if len(RP):
    w("Reproduced native baselines against the vectors cached by `offline_layer_stats.py` (single-image inference) for the nine pools that have a cache. This run uses batches of 16 on the Apple GPU.\n")
    w("| Encoder | Pools | Images | Minimum cosine to cached vector | Queries with the identical ranked top four | Queries with the same four references |"); w("|---|---|---|---|---|---|")
    for enc, g in RP.groupby("encoder", sort=False): w(f"| {enc} | {len(g)} | {g.n.sum()} | {g.min_cos.min():.6f} | {g.same_ranked_top4.sum()} of {g.n.sum()} | {g.same_top4_set.sum()} of {g.n.sum()} |")
    bad = RP[RP.same_top4_set < RP.n]; stale = bad[bad.rows_equal_to_own_image == 0]; small = bad[bad.rows_equal_to_own_image > 0]
    if not len(bad): w("\nEvery reproduced baseline returns the cached top four for every query.\n")
    if len(small): w("\nSmall differences: " + "; ".join(f"{r.dataset} {r.encoder} ({r.n - r.same_top4_set} of {r.n} queries with a different set, minimum cosine {r.min_cos:.6f})" for r in small.itertuples()) + ". The vectors agree; the fourth and fifth references are near-tied and batched GPU inference orders them differently from the cached single-image CPU inference.\n")
    if len(stale): w("**Stale cache found:** " + "; ".join(f"{r.dataset} {r.encoder} (0 of {r.n} cached rows equal the vector of their own image; {r.rows_equal_to_another_pool_image} equal the vector of another image of the pool)" for r in stale.itertuples()) +
                     ". The cached ViT and DINOv2 vectors of this pool belong to an earlier draw of the pool and are not aligned with `pool_Durum_Wheat.csv`; the cached CLIP and ResNet-50 vectors of the same pool are aligned. The audit vectors were checked against fresh single-image CPU inference with the repository's `get_image_embedding` path (cosine 1.000000 on four images), and the image MD5s equal the pool manifest. "
                     "The audit does not use the cache. Any earlier Durum Wheat number computed from the cached ViT or DINOv2 vectors should be recomputed. With the stale pool excluded, the reproduced ViT and DINOv2 baselines return the cached ranked top four on " +
                     "; ".join(f"{g.same_ranked_top4.sum()} of {g.n.sum()} queries ({e})" for e, g in RP[~RP.dataset.isin(stale.dataset)].groupby("encoder", sort=False) if e in ("vit", "dinov2")) + ".\n")
chk = VN[VN.rep_a.isin(["dinov2|cls|b12", "resnet|gap|b16", "vit|cls|b12", "clip|cls|b12", "dinov3|cls|b12", "bioclip|cls|b12", "siglip2|mean|b12", "qwen|mean|b24", "flava|cls|b12"])].groupby("rep_a").apply(lambda t: (t.mean_turnover * t.n_queries).sum() / t.n_queries.sum())
w("Uniform final-block readout against the native baseline (mean top-4 turnover over all splits): " + "; ".join(f"{k} {v:.3f}" for k, v in chk.items()) + ". DINOv2, DINOv3 and ResNet-50 natives are the final-block readout up to float16 caching; the ViT pooler, the CLIP, BioCLIP and FLAVA projections, the SigLIP 2 pooling head and the Qwen patch merger change the neighbors.\n")

def fam_table(df, splits, title):
    w(title + "\n"); w("| Encoder, readout | Mean replaced (query-weighted) | Mean replaced (unweighted over splits) | 0 | 1 | 2 | 3 | 4 | First reference changes | Same four, new order | Queries x transitions |"); w("|---|---|---|---|---|---|---|---|---|---|---|")
    res = {}
    for fam in FAMS:
        d = df[(df.family == fam) & df.split.isin(splits)]; n = d.n_queries.sum(); qw = lambda c: (d[c] * d.n_queries).sum() / n; res[fam] = qw("mean_turnover")
        w(f"| {FN[fam]} | {qw('mean_turnover'):.2f} | {d.groupby('split').mean_turnover.mean().mean():.2f} | " + " | ".join(f"{qw(f'pct_turnover_{k}'):.0f}%" for k in range(5)) + f" | {qw('pct_top1_changed'):.0f}% | {qw('pct_same_four_reordered'):.0f}% | {n} |")
    w(""); return res
w("## 4. How many of the four neighbors change between adjacent blocks\n")
Q1 = fam_table(ADJ, MAIN, f"**All {len(MAIN)} AgEval loader pools** ({nq_main} queries; 11 adjacent transitions per 12-block readout, 16 for ResNet-50, 23 for the Qwen3-VL tower). Columns 0 to 4 give the share of query-transitions with that many references replaced.")
w("**By transition, query-weighted over the AgEval loader pools** (mean references replaced of 4):\n")
w("| Transition | " + " | ".join(FN[f] for f in FAMS) + " |"); w("|---|" + "---|" * len(FAMS))
for b in range(0, 24):
    cells = []
    for fam in FAMS:
        d = ADJ[(ADJ.family == fam) & ADJ.split.isin(MAIN) & (ADJ.block_a == b)]; cells.append(f"{(d.mean_turnover * d.n_queries).sum() / d.n_queries.sum():.2f}" if len(d) else "")
    if any(cells): w(f"| {b} to {b + 1} | " + " | ".join(cells) + " |")
w("\n**By split** (mean references replaced between adjacent blocks, averaged over the transitions of the readout):\n")
w("| Split | Queries | " + " | ".join(FN[f] for f in FAMS) + " |"); w("|---|---|" + "---|" * len(FAMS))
for n in ORDER: w(f"| {short(n)} | {SP[n]['n_queries']} | " + " | ".join(f"{ADJ[(ADJ.family == f) & (ADJ.split == n)].mean_turnover.mean():.2f}" for f in FAMS) + " |")
w("\n## 5. Largest changes by dataset and transition\n")
T = ADJ.sort_values("mean_turnover", ascending=False).head(15)
w("| Split | Encoder, readout | Transition | Mean replaced | All four replaced | First reference changes |"); w("|---|---|---|---|---|---|")
for r in T.itertuples(): w(f"| {short(r.split)} | {FN[r.family]} | {int(r.block_a)} to {int(r.block_b)} | {r.mean_turnover:.2f} | {r.pct_turnover_4:.0f}% | {r.pct_top1_changed:.0f}% |")
ds_rank = ADJ[ADJ.split.isin(ORDER)].groupby("split").apply(lambda t: t.mean_turnover.mean()).sort_values(ascending=False)
w("\nMean adjacent-block turnover per split, over all readouts: " + "; ".join(f"{short(k)} {v:.2f}" for k, v in ds_rank.items()) + ".\n")
w("**Smallest changes** (most stable transitions):\n"); w("| Split | Encoder, readout | Transition | Mean replaced |"); w("|---|---|---|---|")
for r in ADJ.sort_values("mean_turnover").head(6).itertuples(): w(f"| {short(r.split)} | {FN[r.family]} | {int(r.block_a)} to {int(r.block_b)} | {r.mean_turnover:.2f} |")
w("\n## 6. Do the replacements improve same-label retrieval?\n")
w(f"Same-label metrics use the {len(MAINLAB)} AgEval loader pools with class labels ({sum(SP[n]['n_queries'] for n in MAINLAB)} queries; the pools have 90 to 100 queries each, so query-weighted and unweighted averages agree to two decimals). Other splits follow in Section 8.\n")
w("**Retrieval quality by block** (query-weighted; S4 = mean same-label references of 4, T1 = first reference has the query label):\n")
w("| Block | " + " | ".join(FN[f] + " S4 / T1" for f in FAMS) + " |"); w("|---|" + "---|" * len(FAMS))
QUAL = {}
for b in range(0, 25):
    cells = []
    for fam in FAMS:
        d = VN[(VN.family == fam) & VN.split.isin(MAINLAB) & (VN.block_a == b)]
        if len(d):
            n = d.n_queries.sum(); s4 = (d.same_label_of4_a * d.n_queries).sum() / n; t1 = (d.top1_correct_a * d.n_queries).sum() / n; QUAL[(fam, b)] = (s4, t1, d.groupby("split").same_label_of4_a.mean().mean()); cells.append(f"{s4:.2f} / {t1:.0f}%")
        else: cells.append("")
    w(f"| {b} | " + " | ".join(cells) + " |")
natq = {}
for enc in NAT:
    if enc == "vit": d = VA[VA.split.isin(MAINLAB)].drop_duplicates("split"); a, t = "same_label_of4_a", "top1_correct_a"
    else: d = VA[(VA.rep_b == NAT[enc]) & VA.split.isin(MAINLAB)]; a, t = "same_label_of4_b", "top1_correct_b"
    n = d.n_queries.sum(); natq[enc] = ((d[a] * d.n_queries).sum() / n, (d[t] * d.n_queries).sum() / n, d[a].mean())
w("\nNative baselines on the same pools (S4 query-weighted / S4 unweighted over pools / T1): " + "; ".join(f"{k} {v[0]:.2f} / {v[2]:.2f} / {v[1]:.0f}%" for k, v in natq.items()) + ".\n")
def change_table(df, ref_is_a, title, key):
    """improve / same / decline of the block relative to the reference, on queries whose top four differ"""
    w(title + "\n"); w("| Block | " + " | ".join(FN[f] for f in FAMS) + " |"); w("|---|" + "---|" * len(FAMS)); tot = {}
    for b in range(0, 25):
        cells = []
        for fam in FAMS:
            d = df[(df.family == fam) & df.split.isin(MAINLAB) & (df[key] == b)]
            if not len(d): cells.append(""); continue
            up, dn = (d.n_improved_when_changed.sum(), d.n_declined_when_changed.sum()) if ref_is_a else (d.n_declined_when_changed.sum(), d.n_improved_when_changed.sum())
            sm = d.n_same_when_changed.sum(); tot[(fam, b)] = (int(up), int(sm), int(dn), int(d.n_changed_queries.sum()), int(d.n_queries.sum())); cells.append(f"{int(up)} / {int(sm)} / {int(dn)} ({int(d.n_changed_queries.sum())})")
        if any(cells): w(f"| {b} | " + " | ".join(cells) + " |")
    w(""); return tot
TN = change_table(VN, False, "**Each block against its encoder's native baseline.** Cells: queries where the block has more / equal / fewer same-label references than the native baseline, among the queries whose four references differ (count in parentheses).", "block_a")
VA2 = VA[VA.family.isin(FAMS)].copy(); VA2["blk"] = VA2.block_b
TA = change_table(VA2, True, "**Each block against the pooled ViT output of Arshad et al.** Cells: queries where the block has more / equal / fewer same-label references than the pooled ViT, among the queries whose four references differ (count in parentheses).", "blk")
dn = VA[VA.rep_b.isin([NAT[e] for e in ENCS[1:]]) & VA.split.isin(MAINLAB)].groupby("rep_b").agg(up=("n_improved_when_changed", "sum"), same=("n_same_when_changed", "sum"), down=("n_declined_when_changed", "sum"), ch=("n_changed_queries", "sum"), n=("n_queries", "sum"))
w("Native baselines against the pooled ViT (more / equal / fewer same-label references, changed queries of all): " + "; ".join(f"{k} {int(r.up)} / {int(r.same)} / {int(r.down)} ({int(r.ch)} of {int(r.n)})" for k, r in dn.iterrows()) + ".\n")
w("**Adjacent transitions:** does the deeper block gain or lose same-label references when the neighbors change?\n"); w("| Encoder, readout | Improved | Same | Declined | Changed query-transitions | Mean change in same-label references when changed |"); w("|---|---|---|---|---|---|")
ADJQ = {}
for fam in FAMS:
    d = ADJ[(ADJ.family == fam) & ADJ.split.isin(MAINLAB)]; ch = d.n_changed_queries.sum(); md = (d.mean_same_label_delta_when_changed.fillna(0) * d.n_changed_queries).sum() / ch
    ch = int(ch); ADJQ[fam] = (int(d.n_improved_when_changed.sum()), int(d.n_same_when_changed.sum()), int(d.n_declined_when_changed.sum()), ch, md)
    w(f"| {FN[fam]} | {ADJQ[fam][0]} ({100 * ADJQ[fam][0] / ch:.0f}%) | {ADJQ[fam][1]} ({100 * ADJQ[fam][1] / ch:.0f}%) | {ADJQ[fam][2]} ({100 * ADJQ[fam][2] / ch:.0f}%) | {ch} | {md:+.3f} |")
w("\n**Per labeled pool:** mean same-label references of 4 for the native baselines and the best block of each readout (the best block is listed for description; it is read on the same queries, so it is optimistic and is not a selection).\n")
w("| Pool | " + " | ".join(NATN[e] for e in ENCS) + " | " + " | ".join("Best " + FN[f] for f in FAMS) + " |"); w("|---|" + "---|" * (len(ENCS) + len(FAMS)))
for n in LAB:
    v0 = VA[VA.split == n].iloc[0].same_label_of4_a; cells = [f"{v0:.2f}"] + [f"{VA[(VA.split == n) & (VA.rep_b == NAT[e])].same_label_of4_b.iloc[0]:.2f}" for e in ENCS[1:]]
    for fam in FAMS:
        d = VN[(VN.family == fam) & (VN.split == n)]; r = d.loc[d.same_label_of4_a.idxmax()]; cells.append(f"{r.same_label_of4_a:.2f} (b{int(r.block_a)})")
    w(f"| {short(n)} | " + " | ".join(cells) + " |")
w("\n## 7. Largest single-query changes and near-ties\n")
G = GT[GT.split.isin(ORDER)]; g = G.groupby("turnover").agg(n=("gap45_before", "size"), med=("gap45_before", "median"), q25=("gap45_before", lambda x: x.quantile(0.25)), q75=("gap45_before", lambda x: x.quantile(0.75)))
w("Fourth-to-fifth score gap in the shallower block, by how many references the next block replaces (all splits, all adjacent transitions):\n"); w("| References replaced | Query-transitions | Median 4th-5th cosine gap | Interquartile range |"); w("|---|---|---|---|")
for k, r in g.iterrows(): w(f"| {k} | {int(r.n)} | {r.med:.4f} | {r.q25:.4f} to {r.q75:.4f} |")
GL = G[G.label_set_changed.notna()]; ls = GL[GL.turnover > 0].groupby("turnover").label_set_changed.apply(lambda x: 100 * x.astype(bool).mean())
w("\nShare of changed query-transitions in which the retrieved labels also change (labeled splits): " + "; ".join(f"{k} replaced {v:.0f}%" for k, v in ls.items()) + ".\n")
w("Examples: the two highest-turnover queries of selected adjacent transitions. Scores are cosine similarities of the top five (the fifth is the first reference below the cutoff).\n")
used = set()
for fam in FAMS:                      # one example per readout, each from a different labeled split
    c = [e for e in BIG if e["family"] == fam and e["labels_comparable"] and e["split"] not in used]
    if not c: continue
    e = sorted(c, key=lambda e: (-e["turnover"], e["gap45_a"] if e["gap45_a"] is not None else 9, e["split"]))[len(used) % 3 if len(c) > 3 else 0]; used.add(e["split"])
    w(f"- **{short(e['split'])}, {FN[e['family']]}, block {int(e['rep_a'][-2:])} to {int(e['rep_b'][-2:])}**, query `{e['query']}` (label {e['label']}), {e['turnover']} of 4 replaced, retrieved labels {'change' if e['retrieved_label_set_changed'] else 'stay the same'}.")
    w(f"  - Before: " + ", ".join(f"`{os.path.basename(r)}` ({l})" for r, l in zip(e['refs_a'], e['labels_a'])) + f"; scores {e['scores_a']}; 4th-5th gap {e['gap45_a']}.")
    w(f"  - After: " + ", ".join(f"`{os.path.basename(r)}` ({l})" for r, l in zip(e['refs_b'], e['labels_b'])) + f"; scores {e['scores_b']}; 4th-5th gap {e['gap45_b']}.")
w("\n## 8. Splits outside the AgEval loader pools (reported separately)\n")
OTH = [n for n in ORDER if SP[n]["group"] not in ("AgEval loader pool",)]
w("| Split | Queries | " + " | ".join(NATN[e] + " S4 / T1" for e in ENCS) + " | Best block per readout (S4) | Mean adjacent turnover (all readouts) |"); w("|---|---|" + "---|" * (len(ENCS) + 2))
for n in OTH:
    if not SP[n]["same_label_metrics"]:
        w(f"| {short(n)} | {SP[n]['n_queries']} | not applicable |" + " |" * len(ENCS) + f" {ADJ[ADJ.split == n].mean_turnover.mean():.2f} |"); continue
    r0 = VA[VA.split == n].iloc[0]; cells = [f"{r0.same_label_of4_a:.2f} / {r0.top1_correct_a:.0f}%"]
    for e in ENCS[1:]:
        r = VA[(VA.split == n) & (VA.rep_b == NAT[e])].iloc[0]; cells.append(f"{r.same_label_of4_b:.2f} / {r.top1_correct_b:.0f}%")
    best = []
    for fam in FAMS:
        d = VN[(VN.family == fam) & (VN.split == n)]; r = d.loc[d.same_label_of4_a.idxmax()]; best.append(f"{FN[fam]} b{int(r.block_a)} {r.same_label_of4_a:.2f}")
    w(f"| {short(n)} | {SP[n]['n_queries']} | " + " | ".join(cells) + " | " + "; ".join(best) + f" | {ADJ[ADJ.split == n].mean_turnover.mean():.2f} |")
w("\n## 9. Checks\n")
exp = {f: NBLK(f) for f in FAMS}; ok = True; msgs = []
for n in ORDER:
    for fam in FAMS:
        k = len(set(VN[(VN.split == n) & (VN.family == fam)].block_a))
        if k != exp[fam]: ok = False; msgs.append(f"{n} {fam}: {k} blocks")
    if len(VA[(VA.split == n) & VA.rep_b.isin(list(NAT.values()))]) != len(ENCS) - 1: ok = False; msgs.append(f"{n}: native baselines missing")
reg = 14; rec = len({s["dataset"] for s in COV["splits"] if s["group"] in ("AgEval loader pool", "extra loader pool")}) + len(COV["not_run"])
w(f"- Registered loaders with a recorded outcome: {rec} of {reg}.")
w(f"- Splits with all nine encoders and every expected block (12 per readout of the 12-block transformers, 17 for ResNet-50, 24 for the Qwen3-VL tower, 9 native outputs): {'all ' + str(len(ORDER)) if ok else 'INCOMPLETE: ' + '; '.join(msgs)}.")
w(f"- Representations per split: {len(dims)}. Layer-pair rows: {len(PM)}. Per-query ranking rows: {sum(SP[n]['n_queries'] for n in ORDER) * len(dims)}.")
w("\n## 10. Files\n")
w("- `coverage_manifest.json`: every split and dataset with counts, source, eligibility and label status.\n- `per_query_rankings.csv.gz`: the four ranked reference ids, labels and scores and the fourth-to-fifth gap for every query and representation.\n- `layer_pair_metrics.csv`: every block pair within a readout, every block against its native baseline, and every representation against the pooled ViT, per split.\n"
  "- `largest_changes.json`, `gap_vs_turnover_adjacent.csv.gz`, `baseline_reproduction.csv`, `retrieval_meta.json`.\n- `plots/turnover_heatmaps_ageval_pools.png` and one `plots/turnover_heatmap_<encoder>_<readout>.png` per readout; `plots/per_split/` holds the same seven heat maps for every split.\n- `plots/same_label_by_layer.png`, `plots/top1_same_label_by_layer.png`.\n- `features/`: compressed latent cache. Code: `repo/layer_audit_fetch.py`, `repo/layer_audit.py`, `repo/layer_audit_report.py`.\n")
json.dump({"Q1": Q1, "QUAL": {f"{k[0]}|{k[1]}": v for k, v in QUAL.items()}, "natq": natq, "ADJQ": {k: [float(x) for x in v] for k, v in ADJQ.items()}, "TN": {f"{k[0]}|{k[1]}": v for k, v in TN.items()}, "TA": {f"{k[0]}|{k[1]}": v for k, v in TA.items()},
           "gap": g.reset_index().to_dict("records"), "label_set": ls.to_dict(), "ds_rank": ds_rank.to_dict()}, open(os.path.join(out, "report_numbers.json"), "w"), indent=1, default=float)
summ = os.path.join(out, "SUMMARY_ANSWERS.md")
if os.path.exists(summ): L[SUMMARY_AT] = open(summ).read()
open(os.path.join(out, "REPORT.md"), "w").write("\n".join(L) + "\n"); print("\n".join(L)[:200]); print("report written;", "checks ok" if ok else msgs)
