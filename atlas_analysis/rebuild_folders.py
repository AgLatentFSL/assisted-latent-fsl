"""Recreate the <dataset>/<query>/<encoder>/<layer>/top{1,2,4,8}/ folder tree from index.csv and the image files. No model is needed.

The published copy holds every image once (images/<dataset>/<role>/<class>/<file>) plus index.csv, which lists the top 8
references of every query, encoder and layer. This script lays the tree out again with hard links (copies if links fail).
Usage (from the atlas root): python analysis/rebuild_folders.py [--images images]
"""
import csv, os, re, shutil, sys
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__))); IMG = os.path.join(ROOT, sys.argv[sys.argv.index("--images") + 1] if "--images" in sys.argv else "images")
safe = lambda s: re.sub(r"[^A-Za-z0-9._-]+", "_", str(s)).strip("_"); KS = (1, 2, 4, 8)
ENC = {"ViT": "1_ViT", "DINOv2": "2_DINOv2", "CLIP": "3_CLIP", "ResNet50": "4_ResNet50", "SigLIP2": "5_SigLIP2", "BioCLIP": "6_BioCLIP", "DINOv3": "7_DINOv3", "Qwen3VL": "8_Qwen3VL", "FLAVA": "9_FLAVA"}
def link(src, dst):
    try: os.link(src, dst)
    except OSError: shutil.copyfile(src, dst)
S = list(csv.DictReader(open(os.path.join(ROOT, "split_images.csv")))); ds_order = list(dict.fromkeys(r["dataset"] for r in S)); ddir = {d: f"{i + 1:02d}_{safe(d)}" for i, d in enumerate(ds_order)}
path = {(r["dataset"], r["role"], r["image_id"]): os.path.join(IMG, safe(r["dataset"]), r["role"], *r["image_id"].split("/")) for r in S}
qimg = {(r["dataset"], r["query_folder"]): r for r in S if r["role"] == "query"}; ref_role = "reference" if any(r["role"] == "reference" for r in S) else "query"
for (d, qf), r in qimg.items():
    qd = os.path.join(ROOT, ddir[d], qf); os.makedirs(qd, exist_ok=True); src = path[(d, "query", r["image_id"])]; link(src, os.path.join(qd, f"QUERY__{safe(r['class'])}{os.path.splitext(src)[1].lower() or '.jpg'}"))
rows = {}; n = 0; ONLY = sys.argv[sys.argv.index("--only") + 1] if "--only" in sys.argv else None          # --only <dataset name>: rebuild one dataset
for r in csv.DictReader(open(os.path.join(ROOT, "index.csv"))):
    if ONLY and r["dataset"] != ONLY: continue
    rows.setdefault((r["dataset"], r["query_folder"], r["model"]), {}).setdefault(r["layer"], []).append(r)
for (d, qf, m), layers in rows.items():
    names = [l for l in layers if l != "final_output"] + ["final_output"]; prev = None; last = None
    for l in names:
        refs = sorted(layers[l], key=lambda r: int(r["rank"])); ids = [r["reference_id"] for r in refs]; base = last if l == "final_output" else prev       # final_output is tagged against the last layer
        for k in KS:
            kd = os.path.join(ROOT, ddir[d], qf, ENC[m], l, f"top{k}"); os.makedirs(kd, exist_ok=True); before = set(base[:k]) if base else None
            for r in refs[:k]:
                tag = "first" if before is None else ("kept" if r["reference_id"] in before else "new"); src = path.get((d, ref_role, r["reference_id"])) or path[(d, "query", r["reference_id"])]
                link(src, os.path.join(kd, f"{r['rank']}__{'match' if r['same_label'] == '1' else 'diff'}__{tag}__{safe(r['reference_label'])}__{safe(os.path.basename(r['reference_id']))}")); n += 1
        if l != "final_output": prev = ids; last = ids
print("reference files written:", n)
