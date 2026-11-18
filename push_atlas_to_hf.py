"""Stage and push the atlas, its images, the latent cache, the split files, the reports and the code to a private Hugging Face dataset repo.

The folder tree with about one million hard-linked reference files is not uploaded. The repo holds every image once, index.csv
with the top 8 references of every query, encoder and layer, and analysis/rebuild_folders.py, which lays the tree out again.
Usage (from repo/): python push_atlas_to_hf.py <atlas_dir> <stage_dir> [--push]
"""
import csv, os, re, shutil, sys
ATLAS, STAGE = os.path.abspath(sys.argv[1]), os.path.abspath(sys.argv[2]); REPO = "tirtho149/ageval-layerwise-reference-atlas"
RUNS = os.path.abspath(os.path.join("runs")); safe = lambda s: re.sub(r"[^A-Za-z0-9._-]+", "_", str(s)).strip("_")
def put(src, dst):
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    if os.path.exists(dst): os.remove(dst)
    try: os.link(src, dst)
    except OSError: shutil.copyfile(src, dst)
def tree(src, dst, skip=()):
    for d, _, fs in os.walk(src):
        for f in fs:
            p = os.path.join(d, f); rel = os.path.relpath(p, src)
            if not any(rel.startswith(s) or f == s for s in skip): put(p, os.path.join(dst, rel))
if os.path.exists(STAGE): shutil.rmtree(STAGE)
for f in ("README.pdf", "index.csv", "layer_changes.csv", "split.json", "split_images.csv"): put(os.path.join(ATLAS, f), os.path.join(STAGE, f))
tree(os.path.join(ATLAS, "assets"), os.path.join(STAGE, "assets")); tree(os.path.join(ATLAS, "analysis"), os.path.join(STAGE, "analysis"), skip=("_readme.html", "last_run.log"))
n_img = 0
for r in csv.DictReader(open(os.path.join(ATLAS, "split_images.csv"))):                 # every query and reference image once
    put(os.path.abspath(r["source_path"]), os.path.join(STAGE, "images", safe(r["dataset"]), r["role"], *r["image_id"].split("/"))); n_img += 1
tree(os.path.join(RUNS, "layer_retrieval_audit", "features"), os.path.join(STAGE, "latents"))
tree(os.path.join(RUNS, "disjoint_split_seed42"), os.path.join(STAGE, "splits", "disjoint_split_seed42"))
for f in ("REPORT.md", "coverage_manifest.json", "layer_pair_metrics.csv", "retrieval_meta.json", "baseline_reproduction.csv", "largest_changes.json"):
    p = os.path.join(RUNS, "layer_retrieval_audit", f)
    if os.path.exists(p): put(p, os.path.join(STAGE, "reports", "layer_retrieval_audit", f))
if os.path.isdir(os.path.join(RUNS, "layer_retrieval_audit", "plots")): tree(os.path.join(RUNS, "layer_retrieval_audit", "plots"), os.path.join(STAGE, "reports", "layer_retrieval_audit", "plots"))
if os.path.isdir(os.path.join(RUNS, "blockwise_vs_pooled")): tree(os.path.join(RUNS, "blockwise_vs_pooled"), os.path.join(STAGE, "reports", "blockwise_vs_pooled"))
for f in ("layer_audit.py", "layer_audit_fetch.py", "layer_audit_report.py", "make_disjoint_split.py", "disjoint_features.py", "make_retrieval_samples.py", "make_retrieval_readme.py", "blockwise_vs_pooled.py", "push_atlas_to_hf.py", "data_loader.py", "get_embeddingp.py"):
    put(f, os.path.join(STAGE, "code", f))
open(os.path.join(STAGE, "code", "README.md"), "w").write("""# Code

Scripts that produced this repository, in the order they run (from the AgEval repository checkout, folder `repo/`):

1. `make_disjoint_split.py <split_dir>`: 100 queries and a separate 100-image reference bank per dataset (seed 42). Output in `splits/`.
2. `layer_audit.py <audit_dir> features` and `disjoint_features.py <split_dir>`: latents of every layer of nine encoders, cached per encoder in compressed `.npz` files (`latents/`). Images that are already cached are never encoded again.
3. `make_retrieval_samples.py <atlas_dir>` with `ATLAS_SPLIT_DIR=<split_dir>`: ranks the bank for every query, encoder and layer from the cached latents and writes the folder tree, `index.csv`, `layer_changes.csv`, `split.json`, `split_images.csv`.
4. `make_retrieval_readme.py <atlas_dir>`: base README and the example figure.
5. `analysis/topk_analysis.py --reparse` and `analysis/render_readme_pdf.py` (inside the atlas): top-1, 2, 4, 8 analysis, figures, README, PDF.
6. `layer_audit.py`, `layer_audit_report.py`, `blockwise_vs_pooled.py`: the wider audit over all AgEval datasets and the date-disjoint check (`reports/`).

Requirements: Python 3.9+, numpy, pandas, matplotlib, torch, transformers 4.57, open_clip_torch, pillow, scipy, huggingface_hub. No hosted model or API is called anywhere.
""")
head = """---
license: other
pretty_name: AgEval layer-wise reference atlas
tags: [image-retrieval, few-shot, plant-phenotyping, layer-wise]
---

> **Layout of this repository.** `images/` holds every query and reference image once. `index.csv` lists the top 8 references of every query, encoder and layer. `latents/` is the compressed latent cache of every encoder and layer. `splits/` defines the queries and reference banks. `analysis/` holds the top-K analysis with its tables, figures and cache. `reports/` holds the wider audit and the date-disjoint check. `code/` holds every script.
> The `<dataset>/<query>/<encoder>/<layer>/top{1,2,4,8}/` folders described below are not stored (about one million files). Recreate them with `python analysis/rebuild_folders.py` from the repository root; no model is needed.

"""
open(os.path.join(STAGE, "README.md"), "w").write(head + open(os.path.join(ATLAS, "README.md")).read())
tot = sum(os.path.getsize(os.path.join(d, f)) for d, _, fs in os.walk(STAGE) for f in fs); nfiles = sum(len(fs) for _, _, fs in os.walk(STAGE)); print(f"staged {nfiles} files, {tot / 1e9:.2f} GB, {n_img} images")
if "--push" in sys.argv:
    from huggingface_hub import HfApi
    api = HfApi(); api.create_repo(REPO, repo_type="dataset", private=True, exist_ok=True); api.upload_large_folder(folder_path=STAGE, repo_id=REPO, repo_type="dataset"); print("pushed to", f"https://huggingface.co/datasets/{REPO}")
