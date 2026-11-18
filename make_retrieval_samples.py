"""Organized folder of retrieved reference images for nine encoders: dataset / query / model / layer / top1, top2, top4, top8.

Reads the cached latents of runs/layer_retrieval_audit (no model calls). For every query of the five paper datasets,
each of the four encoders and every layer of that encoder, the top 8 references by cosine similarity are placed as hard
links to the dataset images, so the images take no extra disk space. Same eligibility and tie-break as the audit.
Transformers use the CLS token of each block; ResNet-50 uses the pooled feature map of the stem and each residual
block. Each model also gets a final_output folder with the vector the model exposes for retrieval.
Usage (from repo/): python make_retrieval_samples.py <out_dir>
"""
import csv, json, os, re, shutil, sys
import numpy as np
dest = sys.argv[1]; sys.argv = ["x", os.path.join("runs", "layer_retrieval_audit"), "none"]
import layer_audit as LA

DATASETS = ["Bean Leaf Lesions", "Mango Leaf Disease", "DeepWeeds", "Soybean Seeds", "Soybean Diseases"]
ENC = [("vit", "1_ViT", "google/vit-base-patch16-224-in21k", "CLS token after each of the 12 transformer blocks", "pooler_output (a dense layer and tanh on the last CLS token)"),
       ("dinov2", "2_DINOv2", "facebook/dinov2-base", "CLS token after each of the 12 transformer blocks", "last CLS token (identical to layer_12)"),
       ("clip", "3_CLIP", "openai/clip-vit-base-patch16, vision tower", "CLS token after each of the 12 transformer blocks", "projected image embedding (a linear projection of the last CLS token, 512 values)"),
       ("resnet", "4_ResNet50", "microsoft/resnet-50", "averaged feature map after the stem (layer_00_stem) and after each of the 16 residual blocks", "pooled output (identical to layer_16)"),
       ("siglip2", "5_SigLIP2", "google/siglip2-base-patch16-224", "mean of the patch tokens after each of the 12 transformer blocks (the model has no CLS token)", "pooled image embedding (attention-pooling head over the patch tokens)"),
       ("bioclip", "6_BioCLIP", "imageomics/bioclip", "CLS token after each of the 12 transformer blocks", "projected image embedding (a linear projection of the last CLS token, 512 values)"),
       ("dinov3", "7_DINOv3", "facebook/dinov3-vitb16-pretrain-lvd1689m", "CLS token after each of the 12 transformer blocks", "last CLS token (identical to layer_12)"),
       ("qwen", "8_Qwen3VL", "Qwen/Qwen3-VL-2B-Instruct, vision tower only", "mean of the patch tokens after each of the 24 transformer blocks (the tower has no CLS token; input 448 x 448)", "mean of the merged visual tokens that enter the language model"),
       ("flava", "9_FLAVA", "facebook/flava-full, image encoder only", "CLS token after each of the 12 transformer blocks", "projected image embedding (a linear projection of the last CLS token)")]
LAYERS = {}
for e, m, *_ in ENC:
    ro, blocks = LA.LAYERSPEC[e][0][0], LA.LAYERSPEC[e][1]          # CLS token where the model has one, otherwise the patch mean; ResNet-50: pooled map
    LAYERS[m] = [(f"{e}|{ro}|b{b:02d}", "layer_00_stem" if (e == "resnet" and b == 0) else f"layer_{b:02d}") for b in blocks] + [(f"{e}|native", "final_output")]
KS = (1, 2, 4, 8); safe = lambda s: re.sub(r"[^A-Za-z0-9._-]+", "_", str(s)).strip("_")
def link(src, dst):
    try: os.link(src, dst)
    except OSError: shutil.copyfile(src, dst)

SPLIT_DIR = os.environ.get("ATLAS_SPLIT_DIR")           # folder written by make_disjoint_split.py; unset = AgEval loader pools with leave-one-out retrieval
if SPLIT_DIR:
    import pandas as pd
    S = []
    for d in DATASETS:
        tag = d.replace(" ", "_"); q = pd.read_csv(os.path.join(SPLIT_DIR, f"queries_{tag}.csv"), dtype=str); b = pd.read_csv(os.path.join(SPLIT_DIR, f"bank_{tag}.csv"), dtype=str)
        S.append({"dataset": d, "split": d + " | disjoint query and reference sets", "queries": [(LA.rid(p), p, l, m) for p, l, m in zip(q.path, q.label, q.md5)], "bank": sorted([(LA.rid(p), p, l, m) for p, l, m in zip(b.path, b.label, b.md5)], key=lambda t: t[0])})
    PROTOCOL = {"protocol": "disjoint", "description": "100 test queries and a separate reference bank of 100 images per dataset; no image is in both sets and no bank image is byte-identical to a query", "split_files": SPLIT_DIR}
else:
    S, _ = LA.splits(); S = [s for d in DATASETS for s in S if s["split"] == d + " | loader pool"]; assert len(S) == len(DATASETS)
    PROTOCOL = {"protocol": "leave-one-out pool", "description": "AgEval loader pool of about 100 images per dataset; every pool image is a query and retrieves from the rest of the pool"}
_fidx, _fopen = {}, {}
def vectors(rep, paths):
    """unit-length latents of one representation for a list of image paths, from any cache file that holds the image"""
    enc, ro = rep.split("|")[:2]
    if enc not in _fidx:
        _fidx[enc] = {}
        for f in sorted(os.listdir(os.path.join(LA.out, "features"))):
            if f.startswith(enc + "__"): [_fidx[enc].setdefault(p, (f, i)) for i, p in enumerate(np.load(os.path.join(LA.out, "features", f))["images"])]
    key = "native" if ro == "native" else (f"gap{int(rep[-2:]):02d}" if enc == "resnet" else ro); rows = []
    for p in paths:
        f, i = _fidx[enc][p]
        if (f, key) not in _fopen: _fopen[(f, key)] = np.load(os.path.join(LA.out, "features", f))[key]
        a = _fopen[(f, key)]; rows.append(a[i] if (ro == "native" or enc == "resnet") else a[i, int(rep[-2:]) - 1])
    X = np.stack(rows).astype(np.float64); return X / np.linalg.norm(X, axis=1, keepdims=True)
if os.path.exists(dest):                                 # rebuild in place; the analysis/ folder (scripts and results of the top-K analysis) is kept
    for x in os.listdir(dest):
        if x == "analysis": continue
        p_ = os.path.join(dest, x); shutil.rmtree(p_) if os.path.isdir(p_) else os.remove(p_)
os.makedirs(dest, exist_ok=True); rows, chg, nfiles = [], [], 0
for n_s, s in enumerate(S):
    name = s["dataset"]; ddir = os.path.join(dest, f"{n_s + 1:02d}_{safe(name)}"); qpaths = [t[1] for t in s["queries"]]; bpaths = [t[1] for t in s["bank"]]
    bid = np.array([t[0] for t in s["bank"]]); blab = np.array([str(t[2]) for t in s["bank"]])
    inel = (np.array([t[0] for t in s["queries"]])[:, None] == bid[None, :]) | (np.array([t[3] for t in s["queries"]])[:, None] == np.array([t[3] for t in s["bank"]])[None, :])
    ranks = {}
    for m in LAYERS:
        for rep, _ in LAYERS[m]:
            Sim = vectors(rep, qpaths) @ vectors(rep, bpaths).T; Sim[inel] = -np.inf; o = np.argsort(-Sim, kind="stable")[:, :8]; ranks[rep] = (o, np.take_along_axis(Sim, o, 1))
    for q, (qid, qpath, qlab, _) in enumerate(s["queries"]):
        qname = f"q{q + 1:03d}__{safe(qlab)}__{safe(os.path.splitext(os.path.basename(qid))[0])}"; qdir = os.path.join(ddir, qname); os.makedirs(qdir)
        link(qpath, os.path.join(qdir, f"QUERY__{safe(qlab)}{os.path.splitext(qpath)[1].lower() or '.jpg'}")); nfiles += 1
        for m in LAYERS:
            prev = None
            for rep, lname in LAYERS[m]:
                o, sc = ranks[rep]; ref = prev if lname != "final_output" else ranks[LAYERS[m][-2][0]][0]      # final_output is compared with the last layer
                for k in KS:
                    kdir = os.path.join(qdir, m, lname, f"top{k}"); os.makedirs(kdir); before = set(ref[q, :k]) if ref is not None else None
                    for r in range(k):
                        j = o[q, r]; rpath = s["bank"][j][1]
                        tag = "first" if before is None else ("kept" if j in before else "new")
                        link(rpath, os.path.join(kdir, f"{r + 1}__{'match' if blab[j] == str(qlab) else 'diff'}__{tag}__{safe(blab[j])}__{safe(os.path.basename(rpath))}")); nfiles += 1
                c = [name, qname, qlab, m[2:], lname]
                for k in KS: c += [int((blab[o[q, :k]] == str(qlab)).sum()), "" if ref is None else int(len(set(o[q, :k]) - set(ref[q, :k])))]
                chg.append(c)
                for r in range(8): rows.append([name, qname, qlab, m[2:], lname, r + 1, bid[o[q, r]], blab[o[q, r]], f"{sc[q, r]:.5f}", int(blab[o[q, r]] == str(qlab))])
                if lname != "final_output": prev = o
    print(f"{name}: {len(s['queries'])} queries", flush=True)
json.dump({**PROTOCOL, "datasets": {s["dataset"]: {"queries": len(s["queries"]), "reference_bank": len(s["bank"]), "images_in_both": len({t[1] for t in s["queries"]} & {t[1] for t in s["bank"]})} for s in S}}, open(os.path.join(dest, "split.json"), "w"), indent=1)
with open(os.path.join(dest, "split_images.csv"), "w", newline="") as f:
    w = csv.writer(f); w.writerow(["dataset", "role", "image_id", "class", "md5", "query_folder", "source_path"]); seen = set()
    for s in S:
        for role, items in ((("query", s["queries"]), ("reference", s["bank"])) if SPLIT_DIR else (("query", s["queries"]),)):
            for q, t in enumerate(items):
                qf = f"q{q + 1:03d}__{safe(t[2])}__{safe(os.path.splitext(os.path.basename(t[0]))[0])}" if role == "query" else ""; w.writerow([s["dataset"], role, t[0], t[2], t[3], qf, t[1]])
with open(os.path.join(dest, "index.csv"), "w", newline="") as f:
    w = csv.writer(f); w.writerow(["dataset", "query_folder", "query_label", "model", "layer", "rank", "reference_id", "reference_label", "cosine", "same_label"]); w.writerows(rows)
H = ["dataset", "query_folder", "query_label", "model", "layer"] + [x for k in KS for x in (f"match_in_top{k}", f"new_in_top{k}")]
with open(os.path.join(dest, "layer_changes.csv"), "w", newline="") as f:
    w = csv.writer(f); w.writerow(H); w.writerows(chg)

# ---------------------------------------------------------------- README with the layer-change summary
import pandas as pd
C = pd.DataFrame(chg, columns=H); nq = sum(len(s["queries"]) for s in S)
for k in KS: C[f"new_in_top{k}"] = pd.to_numeric(C[f"new_in_top{k}"])
L = [f"# Reference images by model and layer: how the retrieved neighbors change with depth\n",
     "## The idea\n",
     "Assisted few-shot learning shows a vision-language model a few labeled reference images that look like the query image. A frozen image encoder picks them: it turns every image into a vector, and the references with the most similar vectors are retrieved. "
     "An encoder has many layers, and each layer gives a different vector for the same image. So each layer picks its own references for the same query. "
     "This folder puts those references side by side, for every layer of four encoders, so the change from layer to layer can be seen directly: which references a layer keeps, which it replaces, and whether the new ones have the class of the query.\n",
     f"Scope: the five datasets of the paper ({', '.join(DATASETS)}), {nq} queries, four models, every layer, and the top 1, 2, 4 and 8 references. No vision-language model is involved; this is retrieval only.\n",
     "## Models and layers\n", "| Folder | Checkpoint | Layers | `final_output` |", "|---|---|---|---|"]
L += [f"| `{m}` | `{ck}` | {ld} | {fo} |" for _, m, ck, ld, fo in ENC]
L += ["\nEvery vector is scaled to unit length and references are ranked by cosine similarity. For the three transformers the model's final layer normalization is applied to each layer's tokens before the CLS token is read, so all layers are read the same way.\n",
      "## Folder layout\n", "```",
      "<NN_dataset>/", "  q<number>__<query class>__<query file>/      one folder per query", "    QUERY__<class>.jpg                          the query image",
      "    1_ViT/  2_DINOv2/  3_CLIP/  4_ResNet50/     one folder per model", "      layer_01/ ... layer_12/                   one folder per layer (ResNet-50: layer_00_stem, layer_01 ... layer_16)",
      "      final_output/                             the vector the model exposes for retrieval", "        top1/  top2/  top4/  top8/              the k most similar references of that layer, best first",
      "          <rank>__<match|diff>__<first|kept|new>__<reference class>__<file>", "```\n",
      "How to read a reference file name:\n",
      "- `match`: the reference has the class of the query. `diff`: it has another class.",
      "- `new`: the previous layer did not have this reference in the same top-k folder. `kept`: it did. `first`: the first layer of the model, which has no previous layer.",
      "- In `final_output`, `new` and `kept` compare with the last layer of the model.",
      "- `top1` is contained in `top2`, `top2` in `top4`, and `top4` in `top8`.\n",
      "To see one layer change: open a query folder, pick a model, and compare `layer_05/top4` with `layer_06/top4`. The files tagged `new` in `layer_06` are the references that layer 6 brought in; the references of `layer_05` that are missing were dropped.\n",
      "## What changes from layer to layer\n",
      f"Averages over the {nq} queries. *New* is the number of references in the folder that the previous layer did not have. *Match* is the number of references with the class of the query.\n"]
for _, m, *_ in ENC:
    g = C[C.model == m[2:]]; lay = [l for _, l in LAYERS[m]]
    L += [f"### {m[2:]}\n", "| Layer | New in top 4 | New in top 8 | Match in top 1 | Match in top 2 | Match in top 4 | Match in top 8 |", "|---|---|---|---|---|---|---|"]
    for l in lay:
        d = g[g.layer == l]; nw = lambda k: "" if d[f"new_in_top{k}"].isna().all() else f"{d[f'new_in_top{k}'].mean():.2f}"
        L.append(f"| {l.replace('_', ' ')} | {nw(4)} | {nw(8)} | {100 * d.match_in_top1.mean():.0f}% | {d.match_in_top2.mean():.2f} | {d.match_in_top4.mean():.2f} | {d.match_in_top8.mean():.2f} |")
    body = g[g.layer != "final_output"]; per = body.groupby("layer").agg(new4=("new_in_top4", "mean"), m4=("match_in_top4", "mean")).reindex([l for l in lay if l != "final_output"])
    bl, wl = per.m4.idxmax(), per.new4.idxmax()
    L.append(f"\nOn average a layer replaces {body.new_in_top4.mean():.2f} of the four references of the layer before it. The largest change is at {wl.replace('_', ' ')} ({per.new4.max():.2f} of 4 new). "
             f"The layer with the most matching references in the top 4 is {bl.replace('_', ' ')} ({per.m4.max():.2f} of 4); the final output has {g[g.layer == 'final_output'].match_in_top4.mean():.2f}.\n")
L += ["### By dataset\n", "Mean number of new references in the top 4 per layer step (first number) and matching references in the top 4 at the final output (second number).\n",
      "| Dataset | Queries | " + " | ".join(m[2:] for _, m, *_ in ENC) + " |", "|---|---|" + "---|" * len(ENC)]
for s in S:
    g = C[C.dataset == s["dataset"]]
    L.append(f"| {s['dataset']} | {len(s['queries'])} | " + " | ".join(f"{g[(g.model == m[2:]) & (g.layer != 'final_output')].new_in_top4.mean():.2f} / {g[(g.model == m[2:]) & (g.layer == 'final_output')].match_in_top4.mean():.2f}" for _, m, *_ in ENC) + " |")
L += ["\nReading these numbers: the best layers above are read on the same queries that they are scored on, so they describe these pools and are not a tested choice. More matching references is also not the same as higher accuracy of a vision-language model; that has to be tested with the model.\n",
      "## Files\n",
      "- `index.csv`: every query, model, layer and rank (1 to 8) with the reference id, its class, the cosine similarity and whether the class matches.",
      "- `layer_changes.csv`: one row per query, model and layer with the number of matching and new references in the top 1, 2, 4 and 8.",
      "- Image files are hard links to `repo/data`, so the images take no extra disk space. Copying or zipping the folder produces real files.\n",
      "## Rules behind the retrieval\n",
      "- Each dataset is the AgEval loader pool of about 100 class-balanced images (seed 42). Every pool image is a query and retrieves from the rest of the pool.",
      "- The query itself and byte-identical images are never references. Ties break on the reference id.",
      "- Source: cached latents of `runs/layer_retrieval_audit` (local inference). Rebuild from `repo/` with `python make_retrieval_samples.py <out_dir>`.",
      "- The patch-token mean of each transformer layer is in the audit (`runs/layer_retrieval_audit`) and is not laid out here.\n"]
open(os.path.join(dest, "README.md"), "w").write("\n".join(L))
print("files", nfiles, "index rows", len(rows), "layer rows", len(chg))
