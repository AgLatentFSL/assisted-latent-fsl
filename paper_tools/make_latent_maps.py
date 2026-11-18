"""Figure: PCA maps of last-block patch tokens (first three components as RGB), one PCA per image and encoder.
Needs torch and transformers. Usage: <python> tools/make_latent_maps.py [RUNS_DIR] [REPO_DIR]"""
import os, sys
import numpy as np, pandas as pd, torch
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image
from transformers import AutoImageProcessor, AutoModel, ViTModel
sys.path.insert(0, os.path.dirname(__file__)); import biomor_style as BS
HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__))); PILOT = HERE
RUNS = sys.argv[1] if len(sys.argv) > 1 else os.path.join(PILOT, "runs"); REPO = sys.argv[2] if len(sys.argv) > 2 else os.path.join(PILOT)
BS.apply()
PICK = [("Bean Leaf Lesions", "Bean Rust"), ("Mango Leaf Disease", "Anthracnose"), ("DeepWeeds", "Lantana"), ("Soybean Seeds", "Spotted"), ("Soybean Diseases", "Frogeye Leaf Spot")]
ims = []
for t, lab in PICK:
    pool = pd.read_csv(os.path.join(RUNS, "offline_layer_stats", f"pool_{t.replace(' ', '_')}.csv")); pool["label"] = pool["label"].astype(str)
    ims.append((t, Image.open(os.path.join(REPO, pool[pool.label == lab]["path"].iloc[0])).convert("RGB")))
rows = [[np.asarray(im.resize((480, 480), Image.LANCZOS)) for _, im in ims]]
for mid, cls in (("google/vit-base-patch16-224-in21k", ViTModel),):
    proc, model = AutoImageProcessor.from_pretrained(mid), cls.from_pretrained(mid).eval(); row = []
    for _, im in ims:
        x = proc(images=im, return_tensors="pt")
        with torch.no_grad():
            h = model.layernorm(model(**x, output_hidden_states=True).hidden_states[-1])[0, 1:].numpy().astype(np.float64)
        g = int(round(h.shape[0] ** 0.5)); A = h - h.mean(0); _, _, Vt = np.linalg.svd(A, full_matrices=False); P = A @ Vt[:3].T
        P = np.clip((P - np.percentile(P, 2, axis=0)) / (np.percentile(P, 98, axis=0) - np.percentile(P, 2, axis=0)), 0, 1)
        row.append(np.asarray(Image.fromarray((P.reshape(g, g, 3) * 255).astype(np.uint8)).resize((480, 480), Image.NEAREST)))
    rows.append(row)
fig, axs = plt.subplots(2, len(ims), figsize=(3.3, 1.45), gridspec_kw={"wspace": 0.04, "hspace": 0.04})
for r, (row, nm) in enumerate(zip(rows, ("Image", "ViT\npatch tokens"))):
    for c, a in enumerate(row):
        ax = axs[r, c]; ax.imshow(a, interpolation="none"); ax.set_xticks([]); ax.set_yticks([]); ax.grid(False)
        for s in ax.spines.values(): s.set_visible(False)
        if r == 0: ax.set_title(ims[c][0].replace(" ", "\n", 1), fontsize=4.6, pad=2)
    axs[r, 0].set_ylabel(nm, fontsize=5.0, rotation=90, labelpad=2)
fig._ca_lettered = True
fig.savefig(os.path.join(HERE, "figs", "fig_latent.pdf"), dpi=600, bbox_inches="tight", pad_inches=0.02); fig.savefig(os.path.join(HERE, "figs", "fig_latent.png"), dpi=600, bbox_inches="tight", pad_inches=0.02)
