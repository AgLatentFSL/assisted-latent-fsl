"""Assisted latent few-shot search on five datasets (latent_search.py without Dangerous Insects; same rule, same cached features and development pools). No VLM calls.

Encoders: original ViT (google/vit-base-patch16-224-in21k), CLIP ViT-B/16 vision tower, DINOv2 ViT-B/14.
Candidates per encoder: block 1..12 x {cls, mean of patch tokens, cls+mean concatenated}, the mean of the last four blocks,
the ViT pooler_output (AgEval retriever) and the CLIP projected embedding; then similarity fusion of the best
representation of each encoder. The configuration is selected on development pools that share no image with the
evaluation pools (the AgEval loader pools) and is frozen in selection.json before evaluation pools are scored.
Usage: python latent_search.py <out_dir>
"""
import itertools, json, os, sys
import numpy as np
import pandas as pd
import torch
from PIL import Image
from scipy import stats
from transformers import AutoImageProcessor, AutoModel, CLIPModel, CLIPProcessor, ViTModel
import data_loader as dl
from pilot_matched_labels import vote

out = sys.argv[1]; os.makedirs(out, exist_ok=True)
OFF = os.path.join("runs", "offline_layer_stats")
DATASETS = {"Bean Leaf Lesions": dl.load_and_prepare_data_bean_leaf, "DeepWeeds": dl.load_and_prepare_data_DeepWeeds, "Soybean Seeds": dl.load_and_prepare_data_soybean_seeds,
            "Mango Leaf Disease": dl.load_and_prepare_data_mango_leaf, "Soybean Diseases": dl.load_and_prepare_data_Soybean_PNAS}
K = 4

# ---------------------------------------------------------------- pools
def pools():
    P = {}
    for ds, loader in DATASETS.items():
        tag = ds.replace(" ", "_")
        ev = pd.read_csv(os.path.join(OFF, f"pool_{tag}.csv")); ev["label"] = ev["label"].astype(str)
        dpath = os.path.join(out, f"dev_pool_{tag}.csv")
        if not os.path.exists(dpath):
            big, classes, _ = loader(400)
            big = pd.DataFrame({"path": big[0], "label": big[1].astype(str)})
            big = big[~big["path"].map(os.path.basename).isin(set(ev["path"].map(os.path.basename)))]
            per = max(1, 100 // big["label"].nunique())
            dev = big.groupby("label", sort=True).head(per).sample(frac=1.0, random_state=43).reset_index(drop=True)
            dev.to_csv(dpath, index=False)
        dev = pd.read_csv(dpath); dev["label"] = dev["label"].astype(str)
        assert not set(dev["path"].map(os.path.basename)) & set(ev["path"].map(os.path.basename))
        P[ds] = {"dev": dev, "eval": ev[["path", "label"]]}
    return P

# ---------------------------------------------------------------- features: per image, 12 blocks x {cls, mean}
def load(enc):
    if enc == "vit":
        return AutoImageProcessor.from_pretrained("google/vit-base-patch16-224-in21k"), ViTModel.from_pretrained("google/vit-base-patch16-224-in21k").eval()
    if enc == "dinov2":
        return AutoImageProcessor.from_pretrained("facebook/dinov2-base"), AutoModel.from_pretrained("facebook/dinov2-base").eval()
    return CLIPProcessor.from_pretrained("openai/clip-vit-base-patch16"), CLIPModel.from_pretrained("openai/clip-vit-base-patch16").eval()

def extract(enc, proc, model, path):
    img = Image.open(path).convert("RGB")
    with torch.no_grad():
        if enc == "clip":
            x = proc(images=img, return_tensors="pt")
            o = model.vision_model(pixel_values=x["pixel_values"], output_hidden_states=True)
            hs = torch.stack([model.vision_model.post_layernorm(h)[0] for h in o.hidden_states[1:]])
            extra = model.visual_projection(o.pooler_output)[0].numpy()
        else:
            x = proc(images=img, return_tensors="pt")
            o = model(**x, output_hidden_states=True)
            hs = torch.stack([model.layernorm(h)[0] for h in o.hidden_states[1:]])
            extra = o.pooler_output[0].numpy() if enc == "vit" else None
    return hs[:, 0].numpy().astype(np.float32), hs[:, 1:].mean(dim=1).numpy().astype(np.float32), extra

def features(P):
    F = {}
    for enc in ("vit", "dinov2", "clip"):
        cache = os.path.join(out, f"feats_{enc}.npz")
        C = dict(np.load(cache)) if os.path.exists(cache) else {}
        proc = model = None
        for ds in P:
            for split in ("dev", "eval"):
                key = f"{ds}|{split}"
                if key + "|cls" in C: continue
                if model is None: proc, model = load(enc)
                r = [extract(enc, proc, model, p) for p in P[ds][split]["path"]]
                C[key + "|cls"] = np.stack([a for a, _, _ in r]); C[key + "|mean"] = np.stack([b for _, b, _ in r])
                if r[0][2] is not None: C[key + "|extra"] = np.stack([c for _, _, c in r]).astype(np.float32)
                np.savez(cache, **C); print(f"features {enc} {key} ({len(r)})", flush=True)
        F[enc] = C
    return F

def norm(X):
    X = X.astype(np.float64); return X / np.linalg.norm(X, axis=1, keepdims=True)

def rep(F, enc, ds, split, spec):
    """spec: ('cls'|'mean'|'cat', block 1..12) | ('last4', 'cls'|'mean') | ('extra',)"""
    C = F[enc]; key = f"{ds}|{split}"
    if spec[0] == "extra": return norm(C[key + "|extra"])
    if spec[0] == "last4": return norm(np.mean([norm(C[key + "|" + spec[1]][:, b]) for b in range(8, 12)], axis=0))
    if spec[0] == "cat": return norm(np.concatenate([norm(C[key + "|cls"][:, spec[1] - 1]), norm(C[key + "|mean"][:, spec[1] - 1])], axis=1))
    return norm(C[key + "|" + spec[0]][:, spec[1] - 1])

def sim(F, cfg, ds, split):
    """cfg: list of (enc, spec); cosine similarity matrices are averaged (fusion)."""
    return np.mean([(lambda X: X @ X.T)(rep(F, e, ds, split, s)) for e, s in cfg], axis=0)

def score(S, y):
    S = S.copy(); np.fill_diagonal(S, -np.inf); same, ok, pred = [], [], []
    for q in range(len(y)):
        top = sorted(range(len(y)), key=lambda j: (-S[q, j], j))[:K]
        same.append(sum(y[j] == y[q] for j in top)); p = vote([y[j] for j in top], [S[q, j] for j in top]); pred.append(p); ok.append(p == y[q])
    return float(np.mean(same)), np.array(ok), pred

def name(cfg):
    def one(e, s):
        E = {"vit": "ViT", "dinov2": "DINOv2", "clip": "CLIP"}[e]
        if s[0] == "extra": return f"{E} " + ("pooler_output" if e == "vit" else "projected embedding")
        if s[0] == "last4": return f"{E} {s[1]}, mean of blocks 9-12"
        return f"{E} {'cls+mean' if s[0] == 'cat' else s[0]} b{s[1]}"
    return " + ".join(one(e, s) for e, s in cfg)

P = pools(); F = features(P)
singles = []
for enc in ("vit", "dinov2", "clip"):
    singles += [[(enc, (tok, b))] for tok in ("cls", "mean", "cat") for b in range(1, 13)] + [[(enc, ("last4", "cls"))], [(enc, ("last4", "mean"))]]
    if enc != "dinov2": singles.append([(enc, ("extra",))])
ARBAB = [("vit", ("extra",))]; DINO12 = [("dinov2", ("cls", 12))]

def table(cfgs, split):
    rows = []
    for cfg in cfgs:
        accs, sames = [], []
        for ds in P:
            s, ok, _ = score(sim(F, cfg, ds, split), P[ds][split]["label"].tolist()); accs.append(ok.mean()); sames.append(s)
        rows.append({"config": name(cfg), "cfg": json.dumps(cfg), "mean_knn_acc": float(np.mean(accs)), "mean_same_class": float(np.mean(sames)), **{ds: float(a) for ds, a in zip(P, accs)}})
    return pd.DataFrame(rows)

sel_path = os.path.join(out, "selection.json")
if not os.path.exists(sel_path):
    D1 = table(singles, "dev")
    best_per_enc = {enc: json.loads(D1[D1["cfg"].str.contains(f'"{enc}"')].sort_values("mean_knn_acc", ascending=False)["cfg"].iloc[0]) for enc in ("vit", "dinov2", "clip")}
    fus = [sum((best_per_enc[e] for e in combo), []) for r in (2, 3) for combo in itertools.combinations(("vit", "dinov2", "clip"), r)]
    fus += [[("dinov2", ("cls", 12)), ("vit", ("extra",))], [("dinov2", ("cls", 12)), ("clip", ("extra",))], [("dinov2", ("cls", 12)), ("vit", ("extra",)), ("clip", ("extra",))]]
    D = pd.concat([D1, table(fus, "dev")], ignore_index=True).sort_values("mean_knn_acc", ascending=False).reset_index(drop=True)
    D.to_csv(os.path.join(out, "dev_search.csv"), index=False)
    json.dump({"selected": json.loads(D["cfg"].iloc[0]), "selected_name": D["config"].iloc[0], "dev_mean_knn_acc": float(D["mean_knn_acc"].iloc[0]), "n_candidates": len(D),
               "best_single_per_encoder": {e: name(c) for e, c in best_per_enc.items()},
               "rule": "highest mean 4-NN accuracy over the five development pools; evaluation pools were not scored before this file was written"}, open(sel_path, "w"), indent=1)
SEL = json.load(open(sel_path)); cfg_sel = [(e, tuple(s)) for e, s in SEL["selected"]]
D = pd.read_csv(os.path.join(out, "dev_search.csv"))

# ---------------------------------------------------------------- held-out evaluation pools
rows, per = [], []
for ds in P:
    y = P[ds]["eval"]["label"].tolist(); res = {}
    for nm, cfg in (("selected", cfg_sel), ("arbab", ARBAB), ("dinov2_cls12", DINO12)):
        s, ok, pred = score(sim(F, cfg, ds, "eval"), y); res[nm] = (s, ok, pred)
    w, l = int((res["selected"][1] & ~res["arbab"][1]).sum()), int((~res["selected"][1] & res["arbab"][1]).sum())
    rows.append({"dataset": ds, "n": len(y), **{f"{nm}_acc": float(res[nm][1].mean()) for nm in res}, **{f"{nm}_same": res[nm][0] for nm in res}, "wins_vs_arbab": w, "losses_vs_arbab": l,
                 "p_vs_arbab": 1.0 if w + l == 0 else float(stats.binomtest(w, w + l, 0.5).pvalue)})
    per.append(pd.DataFrame({"dataset": ds, "path": P[ds]["eval"]["path"], "label": y, "vote_selected": res["selected"][2], "vote_arbab": res["arbab"][2], "vote_dinov2_cls12": res["dinov2_cls12"][2]}))
E = pd.DataFrame(rows); E.to_csv(os.path.join(out, "eval_pools.csv"), index=False); pd.concat(per).to_csv(os.path.join(out, "eval_votes.csv"), index=False)
W, Ls = int(E["wins_vs_arbab"].sum()), int(E["losses_vs_arbab"].sum()); tm = (E["selected_acc"] - E["arbab_acc"]).to_numpy()
L = ["# Assisted latent few-shot: search over encoder, block and token\n",
     f"No VLM calls. {len(D)} candidate representations were ranked by mean 4-NN accuracy on five development pools (about 100 images each) that share no image with the evaluation pools. "
     f"The selection was frozen before the evaluation pools were scored.\n", f"**Selected: {SEL['selected_name']}** (development 4-NN accuracy {100 * SEL['dev_mean_knn_acc']:.1f}%).\n",
     "Best single representation per encoder on development: " + "; ".join(SEL["best_single_per_encoder"].values()) + ".\n",
     "## Top 12 candidates on the development pools\n", "| Rank | Representation | Mean 4-NN accuracy | Mean same-class in top 4 |", "|---|---|---|---|"]
L += [f"| {i + 1} | {w['config']} | {100 * w['mean_knn_acc']:.1f}% | {w['mean_same_class']:.2f} |" for i, w in D.head(12).iterrows()]
ref = D[D["config"].isin([name(ARBAB), name(DINO12), "CLIP projected embedding"])]
L += ["\nReference rows on development: " + "; ".join(f"{w['config']} {100 * w['mean_knn_acc']:.1f}% (rank {i + 1})" for i, w in ref.iterrows()) + ".\n",
      "## Held-out evaluation pools (the AgEval loader pools), 4-NN vote\n", "| Dataset | n | Arshad et al. (ViT pooler_output) | DINOv2 CLS b12 | Selected | Selected vs Arshad et al. (wins / losses, p) |", "|---|---|---|---|---|---|"]
L += [f"| {w['dataset']} | {w['n']} | {100 * w['arbab_acc']:.0f}% ({w['arbab_same']:.2f}) | {100 * w['dinov2_cls12_acc']:.0f}% ({w['dinov2_cls12_same']:.2f}) | **{100 * w['selected_acc']:.0f}%** ({w['selected_same']:.2f}) | {w['wins_vs_arbab']} / {w['losses_vs_arbab']}, p {'<0.001' if w['p_vs_arbab'] < 0.001 else format(w['p_vs_arbab'], '.3f')} |" for _, w in E.iterrows()]
pp = 1.0 if W + Ls == 0 else float(stats.binomtest(W, W + Ls, 0.5).pvalue)
L += [f"| **Mean** | {int(E['n'].sum())} | {100 * E['arbab_acc'].mean():.1f}% | {100 * E['dinov2_cls12_acc'].mean():.1f}% | **{100 * E['selected_acc'].mean():.1f}%** | {W} / {Ls}, p {'<0.001' if pp < 0.001 else format(pp, '.3f')} |",
      f"\nValues in parentheses are the mean same-class references in the top 4. The selected representation is above Arshad et al. on {(tm > 0).sum()} of {len(tm)} datasets and below on {(tm < 0).sum()} "
      f"(task-level Wilcoxon p {stats.wilcoxon(tm).pvalue:.3f}).\n" if np.any(tm != 0) else ""]
open(os.path.join(out, "SUMMARY.md"), "w").write("\n".join(L) + "\n"); print("\n".join(L))
