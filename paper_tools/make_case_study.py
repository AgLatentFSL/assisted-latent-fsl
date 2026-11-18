"""Builds figs/fig_case_study.{pdf,png} and tables/case_study_numbers.json: real queries of the K = 2 run with the two references
of the pooled output (baseline) and of the best hidden layer (proposed) of the same encoder, and the saved answers of the VLM.
Sources: runs/layer_vlm_k2/per_query_predictions.csv, the atlas index.csv and split_images.csv. No request is sent.

Selection rule (fixed here, not tuned by eye): a "clear success" is a query where both references of the pooled output have another
class and the VLM is wrong, and both references of the hidden layer have the class of the query and the VLM is right; a "clear
failure" is the opposite. The figure shows the first clear success of the Qwen3-VL tower and of FLAVA in dataset order and the
only clear failure of the Qwen3-VL tower. The counts of all clear successes and failures per encoder go to the JSON file.
Usage (from the paper root): <python> tools/make_case_study.py
"""
import json, os, sys
import pandas as pd
from PIL import Image
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__))); PILOT = HERE; ATLAS = os.path.join(HERE, "atlas")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import biomor_style as BS
BEST = {"ResNet50": "layer_08", "ViT": "layer_10", "CLIP": "layer_04", "FLAVA": "layer_10", "SigLIP2": "layer_11", "Qwen3VL": "layer_16"}      # best hidden layer at K = 2 (Table 1)
NM = {"ResNet50": "ResNet-50", "ViT": "ViT", "CLIP": "CLIP", "FLAVA": "FLAVA", "SigLIP2": "SigLIP 2", "Qwen3VL": "Qwen3-VL tower"}
P = pd.read_csv(os.path.join(PILOT, "runs", "layer_vlm_k2", "per_query_predictions.csv"), dtype=str, keep_default_na=False)
I = pd.read_csv(os.path.join(ATLAS, "index.csv"), dtype=str, keep_default_na=False); I = I[I["rank"].astype(int) <= 2]
S = pd.read_csv(os.path.join(ATLAS, "split_images.csv"), dtype=str, keep_default_na=False)
qpath = {(r.dataset, r.query_folder): r.source_path for r in S[S.role == "query"].itertuples()}; rpath = {(r.dataset, r.image_id): r.source_path for r in S[S.role == "reference"].itertuples()}
cases, counts = {}, {}
for e, l in BEST.items():
    pc = "pooled_vit" if e == "ViT" else f"{e}|final_output"; a = P[P.condition == pc].set_index(["dataset", "query_folder"]); b = P[P.condition == f"{e}|{l}"].set_index(["dataset", "query_folder"])
    R = I[I.model == e]; refs = {k: R[R.layer == k].sort_values("rank").groupby(["dataset", "query_folder"]).agg(labs=("reference_label", list), ids=("reference_id", list)) for k in ("final_output", l)}
    suc, fail = [], []
    for ix in a.index:
        t = a.loc[ix].true_label; rp, rh = refs["final_output"].loc[ix], refs[l].loc[ix]; sp = sum(x == t for x in rp.labs); sh = sum(x == t for x in rh.labs)
        c = {"encoder": e, "block": int(l.split("_")[1]), "dataset": ix[0], "query": ix[1], "true": t, "pooled_refs": list(zip(rp.ids, rp.labs)), "hidden_refs": list(zip(rh.ids, rh.labs)), "pooled_answer": a.loc[ix].pred, "hidden_answer": b.loc[ix].pred}
        if a.loc[ix].pred != t and b.loc[ix].pred == t and sp == 0 and sh == 2: suc.append(c)
        if a.loc[ix].pred == t and b.loc[ix].pred != t and sp == 2 and sh == 0: fail.append(c)
    cases[e] = (suc, fail); counts[e] = {"clear_successes": len(suc), "clear_failures": len(fail)}
ROWS = [("Success", cases["Qwen3VL"][0][0]), ("Success", cases["FLAVA"][0][0]), ("Failure", cases["Qwen3VL"][1][0])]
json.dump({"rule": "clear success: both pooled references of another class and VLM wrong, both hidden references of the query class and VLM right; clear failure: the opposite", "counts_K2": counts,
           "total_successes": sum(v["clear_successes"] for v in counts.values()), "total_failures": sum(v["clear_failures"] for v in counts.values()), "shown": [{k: v for k, v in c.items()} | {"kind": kind} for kind, c in ROWS]},
          open(os.path.join(HERE, "tables", "case_study_numbers.json"), "w"), indent=1)

def img(p):
    im = Image.open(os.path.join(PILOT, p)).convert("RGB"); w, h = im.size; s = min(w, h); return im.crop(((w - s) // 2, (h - s) // 2, (w - s) // 2 + s, (h - s) // 2 + s)).resize((420, 420), Image.LANCZOS)
BS.apply(); GOOD, BAD, INK = "#2E9B66", "#D55E00", BS.INK
fig = plt.figure(figsize=(6.9, 3.35)); fig._ca_lettered = True
X = [0.085, 0.31, 0.415, 0.685, 0.79]; W, Hh = 0.095, 0.196                      # query, two baseline references, two proposed references
for r, (kind, c) in enumerate(ROWS):
    y0 = 0.69 - r * 0.305
    def show(x, path, label, col):
        ax = fig.add_axes([x, y0, W, Hh]); ax.imshow(img(path), interpolation="none"); ax.set_xticks([]); ax.set_yticks([]); ax.grid(False)
        for s in ax.spines.values(): s.set_visible(True); s.set_color(col); s.set_linewidth(1.6)
        fig.text(x + W / 2, y0 - 0.012, label.replace(" ", "\n", 1) if len(label) > 14 else label, ha="center", va="top", fontsize=5.4, color=INK, linespacing=0.95)
    show(X[0], qpath[(c["dataset"], c["query"])], f"True: {c['true']}", "#444444")
    for k, (rid, lab) in enumerate(c["pooled_refs"]): show(X[1 + k], rpath[(c["dataset"], rid)], lab, GOOD if lab == c["true"] else BAD)
    for k, (rid, lab) in enumerate(c["hidden_refs"]): show(X[3 + k], rpath[(c["dataset"], rid)], lab, GOOD if lab == c["true"] else BAD)
    for x, ans in ((0.522, c["pooled_answer"]), (0.897, c["hidden_answer"])):
        ok = ans == c["true"]; fig.text(x, y0 + Hh - 0.02, "VLM Answer", ha="left", va="top", fontsize=5.2, color="#666666")
        fig.text(x, y0 + Hh - 0.062, ans.replace(" ", "\n", 1) if len(ans) > 13 else ans, ha="left", va="top", fontsize=6.0, fontweight="bold", color=GOOD if ok else BAD, linespacing=0.95)
        fig.text(x, y0 + 0.012, "Correct" if ok else "Wrong", ha="left", va="bottom", fontsize=5.2, color=GOOD if ok else BAD)
    fig.text(0.012, y0 + Hh / 2 + 0.035, kind, ha="left", va="center", fontsize=6.4, fontweight="bold", color=GOOD if kind == "Success" else BAD)
    fig.text(0.012, y0 + Hh / 2 - 0.03, f"{NM[c['encoder']].replace(' tower', '')}\nBlock {c['block']}\n{c['dataset'].replace(' ', chr(10), 1)}", ha="left", va="center", fontsize=5.2, color=INK, linespacing=1.05)
for x, t, col in ((X[0] + W / 2, "Query", INK), ((X[1] + X[2] + W) / 2 + 0.05, "Baseline: Pooled Output", "#6E7880"), ((X[3] + X[4] + W) / 2 + 0.05, "Proposed: Hidden Layer", "#2F5D3A")):
    fig.text(x, 0.955, t, ha="center", va="center", fontsize=7.0, fontweight="bold", color=col)
fig.add_artist(plt.Line2D([0.665, 0.665], [0.03, 0.93], color="#BBBBBB", lw=0.6, ls="--")); fig.add_artist(plt.Line2D([0.29, 0.29], [0.03, 0.93], color="#BBBBBB", lw=0.6, ls="--"))
for ext in (".pdf", ".png"): fig.savefig(os.path.join(HERE, "figs", "fig_case_study" + ext), dpi=640)          # images are 420 px on 0.66 in, about 640 dpi
plt.close(fig); print(json.dumps(counts))
