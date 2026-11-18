"""Materialize the loader pools of AgEval datasets that are not on disk, without downloading the full archives.

For each dataset the archive's file list is read over HTTP range requests, the directory tree the repository loader
expects is created with empty placeholder files, the unmodified loader draws its pool (the loaders only list file names;
label files and masks that a loader opens are fetched for real), and then only the pool images are fetched.
Usage (from repo/): python layer_audit_fetch.py <out_dir>     No model or VLM calls.
"""
import io, json, os, sys, zipfile, hashlib, traceback
import fsspec, requests, pandas as pd
import data_loader as dl

out = sys.argv[1]; os.makedirs(out, exist_ok=True)
KAGGLE = "https://www.kaggle.com/api/v1/datasets/download/"
SPECS = {
    "FUSARIUM 22": dict(src=KAGGLE + "tolgahayit/fusarium-wilt-disease-in-chickpea-dataset", root="./data/fusarium-wilt-disease-in-chickpea-dataset", loader=dl.load_and_prepare_data_FUSARIUM22, samples=100,
                        keep=lambda n: n.startswith("FUSARIUM-22/dataset_raw/"), real=lambda n: False),
    "PlantDoc": dict(src=KAGGLE + "sovitrath/leaf-disease-segmentation-with-trainvalid-split", root="./data/leaf-disease-segmentation-with-trainvalid-split", loader=dl.load_and_prepare_data_DiseaseQuantify, samples=100,
                     keep=lambda n: n.startswith("leaf_disease_segmentation/orig_data/train_"), real=lambda n: "/train_masks/" in n),
    "InsectCount": dict(src=["https://zenodo.org/api/records/12747496/files/images.zip/content", "https://zenodo.org/api/records/12747496/files/labels.zip/content"], root="./data/insectcount", loader=dl.load_and_prepare_data_InsectCount, samples=100,
                        keep=lambda n: True, real=lambda n: n.lower().endswith(".txt")),
    "SBRD": dict(src=KAGGLE + "isaacritharson/severity-based-rice-leaf-diseases-dataset", root="./data/Severity_Based_Rice_Leaf_Diseases_Dataset", loader=dl.load_and_prepare_data_SBRD, samples=100,
                 keep=lambda n: n.startswith("Leaf Disease Dataset/train/"), real=lambda n: False),
    "IP02": dict(src=KAGGLE + "rtlmhjbn/ip02-dataset", root="./data/ip02-dataset", loader=dl.load_and_prepare_data_IP02, samples=105,
                 keep=lambda n: n.startswith("classification/train/") or n == "classes.txt", real=lambda n: n == "classes.txt"),
}

def remote_zip(url):
    final = requests.get(url, stream=True, allow_redirects=True, timeout=60).url
    return zipfile.ZipFile(fsspec.open(final, "rb", block_size=1 << 20, cache_type="readahead").open()), final.split("?")[0]

for name in (sys.argv[2:] or list(SPECS)):
    sp = SPECS[name]; tag = name.replace(" ", "_"); done = os.path.join(out, f"fetch_{tag}.json")
    if os.path.exists(done): print(name, "already fetched"); continue
    try:
        root = sp["root"]; assert not os.path.exists(root), f"{root} exists but no fetch record"
        zips = [remote_zip(u) for u in ([sp["src"]] if isinstance(sp["src"], str) else sp["src"])]
        where, n_entries = {}, 0
        for z, _ in zips:
            for info in z.infolist():
                n = info.filename; n_entries += 1
                if info.is_dir() or not sp["keep"](n): continue
                p = os.path.join(root, n); os.makedirs(os.path.dirname(p), exist_ok=True)
                if sp["real"](n): open(p, "wb").write(z.read(info))
                else: open(p, "wb").close()
                where[os.stat(p).st_ino] = (z, info)          # keyed by inode: the loaders rename class folders
        print(name, "archive entries", n_entries, "placeholders", len(where), flush=True)
        data, classes, _ = sp["loader"](sp["samples"])
        paths = [data.at[i, 0] for i in range(len(data))]
        for p in paths:
            z, info = where[os.stat(p).st_ino]; open(p, "wb").write(z.read(info))
        # the loaders may rename class folders after listing; drop every remaining empty placeholder
        removed = 0
        for d, _, fs in os.walk(root):
            for f in fs:
                p = os.path.join(d, f)
                if os.path.getsize(p) == 0: os.remove(p); removed += 1
        json.dump({"dataset": name, "source": sp["src"], "archive_objects": [u for _, u in zips], "archive_entries": n_entries, "files_listed_for_loader": len(where), "pool_images_fetched": len(paths),
                   "placeholders_removed": removed, "loader_samples": sp["samples"], "pool": [{"path": p, "label": str(data.at[i, 1]), "md5": hashlib.md5(open(p, "rb").read()).hexdigest()} for i, p in enumerate(paths)],
                   "note": "only the loader pool is on disk; the loader cannot be rerun against this directory (the pool is frozen in this file)"}, open(done, "w"), indent=1)
        print(name, "pool", len(paths), "classes", classes if len(classes) < 12 else len(classes), flush=True)
    except Exception as ex:
        traceback.print_exc(); json.dump({"dataset": name, "error": f"{type(ex).__name__}: {ex}"[:500]}, open(os.path.join(out, f"fetch_{tag}_FAILED.json"), "w"), indent=1)
