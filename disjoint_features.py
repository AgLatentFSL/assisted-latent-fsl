"""Encode the images of the disjoint split that are not in the latent cache yet (the reference banks and the added queries), with every encoder.
Usage (from repo/): python disjoint_features.py <split_dir>"""
import hashlib, os, sys
import pandas as pd
split = sys.argv[1]; sys.argv = ["x", os.path.join("runs", "layer_retrieval_audit"), "none"]
import layer_audit as LA
have = set()
for f in os.listdir(os.path.join(LA.out, "features")):
    if f.startswith("vit__"):
        import numpy as np; have |= set(np.load(os.path.join(LA.out, "features", f))["images"].tolist())
paths = sorted({p for f in os.listdir(split) if f.endswith(".csv") for p in pd.read_csv(os.path.join(split, f)).path} - have)
print(len(paths), "images to encode", flush=True)
if paths: LA.extract_all([{"split": "Five datasets | disjoint reference banks and added queries", "tag": "disj" + hashlib.md5("".join(paths).encode()).hexdigest()[:6], "images": paths}])
