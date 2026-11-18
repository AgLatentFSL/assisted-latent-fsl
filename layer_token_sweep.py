"""Layer and token sweep for the original ViT and DINOv2 on the cached AgEval pools. No VLM calls.

For every transformer block k = 1..12 of google/vit-base-patch16-224-in21k and facebook/dinov2-base:
  cls  = CLS token of hidden_states[k], after the model's final LayerNorm
  mean = mean of the patch tokens of hidden_states[k], after the final LayerNorm
plus the ViT pooler_output (the AgEval retriever). Leave-one-out retrieval inside each pool, top 4.
Also saves a PCA map of the last-block patch tokens for a few images (out_dir/patch_pca.png).
Usage: python layer_token_sweep.py <offline_layer_stats_dir>
"""
import os, sys, json
import numpy as np
import pandas as pd
import torch
from PIL import Image
from transformers import AutoImageProcessor, AutoModel, ViTModel
from pilot_matched_labels import vote

run = sys.argv[1]; out = os.path.join(run, "layer_token_sweep"); os.makedirs(out, exist_ok=True)
TASKS = ["Soybean Seeds", "Mango Leaf Disease", "Bean Leaf Lesions", "Soybean Diseases", "Dangerous Insects", "DeepWeeds", "Yellow Rust 19", "IDC", "Durum Wheat"]
MODELS = {"vit": ("google/vit-base-patch16-224-in21k", ViTModel, 1), "dinov2": ("facebook/dinov2-base", AutoModel, 1)}   # last value: number of non-patch tokens
K = 4

def feats(name):
    mid, cls_, n_special = MODELS[name]
    proc = AutoImageProcessor.from_pretrained(mid); model = cls_.from_pretrained(mid).eval()
    cache = os.path.join(out, f"feats_{name}.npz")
    F = dict(np.load(cache)) if os.path.exists(cache) else {}
    for t in TASKS:
        if f"{t}|cls" in F: continue
        pool = pd.read_csv(os.path.join(run, f"pool_{t.replace(' ', '_')}.csv"))
        C, M, Pp = [], [], []
        for p in pool["path"]:
            x = proc(images=Image.open(p).convert("RGB"), return_tensors="pt")
            with torch.no_grad():
                o = model(**x, output_hidden_states=True)
                hs = torch.stack([model.layernorm(h)[0] for h in o.hidden_states[1:]])        # 12 x tokens x 768
            assert hs.shape[0] == 12
            C.append(hs[:, 0].numpy()); M.append(hs[:, n_special:].mean(dim=1).numpy())
            if name == "vit": Pp.append(o.pooler_output[0].numpy())
        F[f"{t}|cls"] = np.stack(C).astype(np.float32); F[f"{t}|mean"] = np.stack(M).astype(np.float32)
        if name == "vit": F[f"{t}|pooler"] = np.stack(Pp).astype(np.float32)
        np.savez(cache, **F); print(f"{name}: {t} done ({len(pool)})", flush=True)
    return F, proc, model

def score(X, y):
    X = X.astype(np.float64); X /= np.linalg.norm(X, axis=1, keepdims=True); S = X @ X.T; np.fill_diagonal(S, -np.inf)
    same, acc = [], []
    for q in range(len(y)):
        top = sorted(range(len(y)), key=lambda j: (-S[q, j], j))[:K]
        same.append(sum(y[j] == y[q] for j in top)); acc.append(vote([y[j] for j in top], [S[q, j] for j in top]) == y[q])
    return float(np.mean(same)), float(np.mean(acc))

rows, keep = [], {}
for name in MODELS:
    F, proc, model = feats(name); keep[name] = (proc, model)
    for t in TASKS:
        y = pd.read_csv(os.path.join(run, f"pool_{t.replace(' ', '_')}.csv"))["label"].astype(str).tolist()
        for tok in ("cls", "mean"):
            for k in range(12):
                s, a = score(F[f"{t}|{tok}"][:, k], y); rows.append({"task": t, "model": name, "token": tok, "block": k + 1, "same_class": s, "knn_acc": a})
        if name == "vit":
            s, a = score(F[f"{t}|pooler"], y); rows.append({"task": t, "model": name, "token": "pooler", "block": 13, "same_class": s, "knn_acc": a})
R = pd.DataFrame(rows); R.to_csv(os.path.join(out, "layer_token_sweep.csv"), index=False)
G = R[R.task != "Durum Wheat"].groupby(["model", "token", "block"])[["same_class", "knn_acc"]].mean().reset_index()
L = ["# Layer and token sweep: original ViT versus DINOv2\n", "Mean over eight AgEval pools (Durum Wheat omitted: every representation scores 4 of 4). Leave-one-out retrieval, top 4. These pools have no source-group split.\n",
     "## 4-NN accuracy (%) by block\n", "| Representation | " + " | ".join(f"b{k}" for k in range(1, 13)) + " | Best block |", "|---|" + "---|" * 13]
for m, tok, nm in (("vit", "cls", "ViT, CLS"), ("vit", "mean", "ViT, mean of patch tokens"), ("dinov2", "cls", "DINOv2, CLS"), ("dinov2", "mean", "DINOv2, mean of patch tokens")):
    g = G[(G.model == m) & (G.token == tok)].set_index("block")["knn_acc"]
    L.append(f"| {nm} | " + " | ".join(("**" if k == g.idxmax() else "") + f"{100 * g[k]:.1f}" + ("**" if k == g.idxmax() else "") for k in range(1, 13)) + f" | {g.idxmax()} |")
p = G[(G.model == "vit") & (G.token == "pooler")].iloc[0]
L += [f"\nThe AgEval retriever (ViT `pooler_output`) reaches {100 * p['knn_acc']:.1f}% 4-NN accuracy and {p['same_class']:.2f} same-class references.\n",
      "## Same-class references in the top 4 by block\n", "| Representation | " + " | ".join(f"b{k}" for k in range(1, 13)) + " |", "|---|" + "---|" * 12]
for m, tok, nm in (("vit", "cls", "ViT, CLS"), ("vit", "mean", "ViT, mean of patch tokens"), ("dinov2", "cls", "DINOv2, CLS"), ("dinov2", "mean", "DINOv2, mean of patch tokens")):
    g = G[(G.model == m) & (G.token == tok)].set_index("block")["same_class"]
    L.append(f"| {nm} | " + " | ".join(f"{g[k]:.2f}" for k in range(1, 13)) + " |")
best = R[R.task != "Durum Wheat"].sort_values("knn_acc", ascending=False).groupby("task").head(1)
L += ["\n## Best representation per task (4-NN accuracy)\n", "| Task | Best | 4-NN accuracy | ViT pooler | ViT CLS b12 | ViT mean b12 | DINOv2 CLS b12 | DINOv2 mean b12 |", "|---|---|---|---|---|---|---|---|"]
def val(t, m, tok, b): return 100 * R[(R.task == t) & (R.model == m) & (R.token == tok) & (R.block == b)]["knn_acc"].iloc[0]
for _, w in best.iterrows():
    t = w["task"]
    L.append(f"| {t} | {w['model']} {w['token']} b{w['block'] if w['block'] < 13 else ''} | {100 * w['knn_acc']:.0f}% | {val(t, 'vit', 'pooler', 13):.0f}% | {val(t, 'vit', 'cls', 12):.0f}% | {val(t, 'vit', 'mean', 12):.0f}% | {val(t, 'dinov2', 'cls', 12):.0f}% | {val(t, 'dinov2', 'mean', 12):.0f}% |")
open(os.path.join(out, "SUMMARY.md"), "w").write("\n".join(L) + "\n"); print("\n".join(L))

# ---- PCA of last-block patch tokens (first three components -> RGB), one PCA per model over the shown images
ex = []
for t in ("DeepWeeds", "Bean Leaf Lesions", "Mango Leaf Disease", "Soybean Seeds"):
    pool = pd.read_csv(os.path.join(run, f"pool_{t.replace(' ', '_')}.csv")); ex.append((t, pool["path"].iloc[0], pool["label"].iloc[0]))
tiles = []
for name, (proc, model) in keep.items():
    toks, grids = [], []
    for _, p, _ in ex:
        x = proc(images=Image.open(p).convert("RGB"), return_tensors="pt")
        with torch.no_grad():
            h = model.layernorm(model(**x, output_hidden_states=True).hidden_states[-1])[0, 1:].numpy()
        toks.append(h); grids.append(int(round(h.shape[0] ** 0.5)))
    A = np.concatenate(toks); A = A - A.mean(0); U, S_, Vt = np.linalg.svd(A, full_matrices=False); P = A @ Vt[:3].T
    P = (P - np.percentile(P, 1, axis=0)) / (np.percentile(P, 99, axis=0) - np.percentile(P, 1, axis=0)); P = np.clip(P, 0, 1)
    i = 0; row = []
    for h, g in zip(toks, grids):
        row.append(Image.fromarray((P[i:i + g * g].reshape(g, g, 3) * 255).astype(np.uint8)).resize((448, 448), Image.NEAREST)); i += g * g
    tiles.append(row)
src = [Image.open(p).convert("RGB").resize((448, 448), Image.LANCZOS) for _, p, _ in ex]
S = Image.new("RGB", (448 * 4 + 30, 448 * 3 + 20), "white")
for r, row in enumerate([src] + tiles):
    for c, im in enumerate(row): S.paste(im, (c * 458, r * 458))
S.save(os.path.join(out, "patch_pca.png")); json.dump([{"task": t, "path": p, "label": l} for t, p, l in ex], open(os.path.join(out, "patch_pca_images.json"), "w"), indent=1)
