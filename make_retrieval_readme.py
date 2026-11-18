"""README (Markdown + PDF) with figures for the layer-wise reference folder. Reads only the folder itself. No model calls.
Usage: python make_retrieval_readme.py <folder>"""
import html, json, os, re, subprocess, sys
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw, ImageFont

D = os.path.abspath(sys.argv[1]); A = os.path.join(D, "assets"); os.makedirs(A, exist_ok=True)
C = pd.read_csv(os.path.join(D, "layer_changes.csv")); nq = C.query_folder.nunique() if False else len(C[(C.model == "ViT") & (C.layer == "layer_01")])
MODELS = [("ViT", "1_ViT", "#0072B2", "google/vit-base-patch16-224-in21k", "12 transformer blocks, CLS token"), ("DINOv2", "2_DINOv2", "#D55E00", "facebook/dinov2-base", "12 transformer blocks, CLS token"),
          ("CLIP", "3_CLIP", "#009E73", "openai/clip-vit-base-patch16 (vision tower)", "12 transformer blocks, CLS token"), ("ResNet50", "4_ResNet50", "#CC79A7", "microsoft/resnet-50", "stem + 16 residual blocks, averaged feature map"),
          ("SigLIP2", "5_SigLIP2", "#E69F00", "google/siglip2-base-patch16-224", "12 transformer blocks, mean of the patch tokens (no CLS token)"), ("BioCLIP", "6_BioCLIP", "#56B4E9", "imageomics/bioclip", "12 transformer blocks, CLS token"),
          ("DINOv3", "7_DINOv3", "#000000", "facebook/dinov3-vitb16-pretrain-lvd1689m", "12 transformer blocks, CLS token"), ("Qwen3VL", "8_Qwen3VL", "#8C5A2B", "Qwen/Qwen3-VL-2B-Instruct (vision tower only)", "24 transformer blocks, mean of the patch tokens (no CLS token), input 448 x 448"),
          ("FLAVA", "9_FLAVA", "#7F7F7F", "facebook/flava-full (image encoder only)", "12 transformer blocks, CLS token")]
MODELS = [x for x in MODELS if x[0] in set(C.model)]
NAME = {"ResNet50": "ResNet-50", "SigLIP2": "SigLIP 2", "Qwen3VL": "Qwen3-VL tower"}; nm = lambda m: NAME.get(m, m)
KIND = {"ViT": "supervised", "DINOv2": "self-supervised", "CLIP": "image-text contrastive", "ResNet50": "supervised CNN", "SigLIP2": "image-text contrastive (sigmoid)", "BioCLIP": "image-text contrastive, biology data", "DINOv3": "self-supervised", "Qwen3VL": "vision tower of a multimodal LLM", "FLAVA": "joint image, text and multimodal objectives"}
SPL = json.load(open(os.path.join(D, "split.json"))) if os.path.exists(os.path.join(D, "split.json")) else {"protocol": "leave-one-out pool"}; DISJ = SPL["protocol"] == "disjoint"
layers = lambda m: [l for l in C[C.model == m].layer.drop_duplicates() if l != "final_output"]
G = {m: C[C.model == m].groupby("layer").agg(new4=("new_in_top4", "mean"), new8=("new_in_top8", "mean"), m1=("match_in_top1", "mean"), m2=("match_in_top2", "mean"), m4=("match_in_top4", "mean"), m8=("match_in_top8", "mean")) for m, *_ in MODELS}

# ---------------------------------------------------------------- figure 1: curves
plt.rcParams.update({"font.family": "Helvetica", "font.size": 7, "axes.spines.top": False, "axes.spines.right": False, "axes.linewidth": 0.6, "xtick.major.width": 0.6, "ytick.major.width": 0.6})
MK = {"ViT": "o", "DINOv2": "s", "CLIP": "^", "ResNet50": "D", "SigLIP2": "v", "BioCLIP": "P", "DINOv3": "X", "Qwen3VL": "h", "FLAVA": "*"}
fig, (a1, a2) = plt.subplots(1, 2, figsize=(7.2, 3.0))
for m, _, col, *_ in MODELS:
    L = layers(m); x = np.linspace(0, 1, len(L)); g = G[m].loc[L]
    a1.plot(x[1:], g.new4.values[1:], color=col, lw=1.2, marker=MK[m], ms=3.2, mec="white", mew=0.4, label=nm(m)); a2.plot(x, g.m4.values, color=col, lw=1.2, marker=MK[m], ms=3.2, mec="white", mew=0.4, label=nm(m))
for ax, t, yl in ((a1, "a", "References replaced (of 4)"), (a2, "b", "References of the query class (of 4)")):
    ax.text(-0.16, 1.03, t, transform=ax.transAxes, fontsize=10, fontweight="bold", va="bottom"); ax.set_ylabel(yl); ax.set_xticks([0, 0.5, 1], ["First layer", "Middle", "Last layer"]); ax.set_xlabel("Relative depth")
fig.legend(*a1.get_legend_handles_labels(), frameon=False, fontsize=7, ncol=5, handlelength=2.4, loc="upper center", bbox_to_anchor=(0.5, 1.0))
fig.tight_layout(rect=(0, 0, 1, 0.88)); fig.savefig(os.path.join(A, "layer_curves.png"), dpi=600); plt.close(fig)

# ---------------------------------------------------------------- figure 2: one query, every layer, every model
W = C[C.layer != "final_output"]; first = W.groupby(["dataset", "query_folder", "model"]).match_in_top4.first(); last = W.groupby(["dataset", "query_folder", "model"]).match_in_top4.last()
gain = (last - first).groupby(level=[0, 1]).sum(); turn = W.groupby(["dataset", "query_folder"]).new_in_top4.sum()
cand = gain[gain >= gain.max() - 2]; ds, qf = turn.loc[cand.index].idxmax(); ddir = [d for d in sorted(os.listdir(D)) if d[3:].replace("_", " ") == ds][0]; qdir = os.path.join(D, ddir, qf)
SC = 1.4                                   # pixel scale: the sheet prints at 600 dpi or more when set to 82% of a two-column text width
z = lambda v: int(round(v * SC)); T, GAP, PAD, HEAD, LAB = z(92), z(8), z(16), z(26), z(150)
def font(sz, bold=False):
    for p in ("/System/Library/Fonts/HelveticaNeue.ttc", "/System/Library/Fonts/Helvetica.ttc"):
        try: return ImageFont.truetype(p, z(sz), index=1 if bold else 0)
        except Exception: pass
    return ImageFont.load_default()
def sq(p, t=T):
    im = Image.open(p).convert("RGB"); w, h = im.size; s = min(w, h); return im.crop(((w - s) // 2, (h - s) // 2, (w - s) // 2 + s, (h - s) // 2 + s)).resize((t, t), Image.LANCZOS)
ncol = max(len(layers(m)) for m, *_ in MODELS); rowh = HEAD + 4 * (T + GAP) + z(22); Wd = PAD + LAB + ncol * (T + GAP) + PAD; Hd = PAD + z(6) + len(MODELS) * rowh + PAD
sheet = Image.new("RGB", (Wd, Hd), "white"); dr = ImageDraw.Draw(sheet)
qfile = [f for f in os.listdir(qdir) if f.startswith("QUERY__")][0]; qlab = qfile[7:].rsplit(".", 1)[0].replace("_", " "); QS = z(112)
for r, (m, mdir, col, *_ ) in enumerate(MODELS):
    y0 = PAD + z(6) + r * rowh; rgb = tuple(int(col[i:i + 2], 16) for i in (1, 3, 5))
    dr.rectangle([PAD, y0 + HEAD, PAD + z(6), y0 + HEAD + 4 * (T + GAP) - GAP], fill=rgb); dr.text((PAD + z(14), y0 + HEAD), nm(m), font=font(20, True), fill="#111")
    if r == 0:
        sheet.paste(sq(os.path.join(qdir, qfile), QS), (PAD + z(14), y0 + HEAD + z(34))); dr.rectangle([PAD + z(14), y0 + HEAD + z(34), PAD + z(14) + QS, y0 + HEAD + z(34) + QS], outline="#111", width=z(3)); dr.text((PAD + z(14), y0 + HEAD + z(152)), "query", font=font(15), fill="#444")
    for c, l in enumerate(layers(m)):
        x = PAD + LAB + int(c * (ncol * (T + GAP) - T) / (len(layers(m)) - 1)); dr.text((x, y0 + z(4)), "stem" if "stem" in l else "L" + str(int(l.split("_")[1])), font=font(15, True), fill="#444")
        files = sorted(os.listdir(os.path.join(qdir, mdir, l, "top4")))
        for k, f in enumerate(files):
            y = y0 + HEAD + k * (T + GAP); sheet.paste(sq(os.path.join(qdir, mdir, l, "top4", f)), (x, y)); _, mt, nw, *_ = f.split("__")
            if nw == "new": dr.rectangle([x, y, x + T - 1, y + T - 1], outline=(230, 159, 0), width=z(6))
            if mt == "match":
                dr.ellipse([x + T - z(26), y + T - z(26), x + T - z(6), y + T - z(6)], fill=(0, 114, 178), outline="white", width=z(2)); dr.line([x + T - z(21), y + T - z(16), x + T - z(17), y + T - z(12), x + T - z(11), y + T - z(20)], fill="white", width=z(3))
        row = C[(C.query_folder == qf) & (C.dataset == ds) & (C.model == m) & (C.layer == l)].iloc[0]
        dr.text((x, y0 + HEAD + 4 * (T + GAP) - z(4)), f"{int(row.match_in_top4)} of 4", font=font(13), fill="#444")
sheet.save(os.path.join(A, "example_query_all_layers.jpg"), quality=88, dpi=(600, 600))

# ---------------------------------------------------------------- content blocks (rendered to Markdown and to HTML)
B = []; add = B.append; tn = [0]
def table(title, head, rows, note=None): tn[0] += 1; add(("table", f"Table {tn[0]}", title, head, rows, note))
steps = {m: G[m].loc[layers(m)] for m, *_ in MODELS}; nlay = sum(len(layers(m)) for m, *_ in MODELS)
names = [nm(m) for m, *_ in MODELS]; lst = lambda xs: (", ".join(xs[:-1]) + " and " + xs[-1]) if len(xs) > 1 else xs[0]
last_best = [nm(m) for m, *_ in MODELS if steps[m].m4.idxmax() == layers(m)[-1]]; early_best = [nm(m) for m, *_ in MODELS if steps[m].m4.idxmax() != layers(m)[-1]]
top = max(MODELS, key=lambda x: G[x[0]].loc["final_output"].m4); topl = max(MODELS, key=lambda x: steps[x[0]].m4.max())
add(("h1", "A layer-by-layer atlas of retrieved reference images for assisted few-shot plant phenotyping"))
add(("meta", f"Resource · {C.dataset.nunique()} datasets · {nq} queries · {len(MODELS)} encoders · {nlay} layers · top 1, 2, 4 and 8 references"))
add(("abstract", "Assisted few-shot learning retrieves labeled reference images for a query with a frozen image encoder. Each layer of the encoder retrieves a different set of references. "
     f"This resource lays out, for {nq} queries from five AgEval datasets, the references retrieved by every layer of {len(MODELS)} encoders: {lst(names)}. "
     f"Between two consecutive layers, {min(g.new4.iloc[1:].mean() for g in steps.values()):.2f} to {max(g.new4.iloc[1:].mean() for g in steps.values()):.2f} of the four references are replaced on average. "
     f"At the final output, {nm(top[0])} retrieves the most references of the query's class ({G[top[0]].loc['final_output'].m4:.2f} of four). "
     + (f"{lst(last_best)} retrieve the most such references at the last layer, whereas {lst(early_best)} do so before the last layer. " if last_best and early_best else "")
     + "The resource covers retrieval only; no vision-language model is involved."))
add(("h2", "Main findings"))
rows = []
for m, mdir, col, ck, ld in MODELS:
    g = steps[m]; fo = G[m].loc["final_output"]
    rows.append([f"**{nm(m)}**", KIND[m], f"{g.new4.iloc[1:].mean():.2f}", f"{g.new4.idxmax().replace('_', ' ')} ({g.new4.max():.2f})", f"{g.m4.idxmax().replace('_', ' ')} ({g.m4.max():.2f})", f"{fo.m4:.2f}"])
table("Summary of layer-wise retrieval for the encoders.", ["Encoder", "Pretraining", "References replaced per layer step (of 4)", "Layer with the largest replacement", "Layer with the most matching references (of 4)", "Matching references at the final output (of 4)"], rows,
      f"Means over {nq} queries. A matching reference has the class of the query.")
add(("ul", ["**Turnover.** Each layer replaces about one to two of the four references of the preceding layer.",
            f"**Depth of best retrieval.** {lst(last_best)} retrieve the most matching references at the last layer. {lst(early_best)} retrieve the most before the last layer." if last_best and early_best else "**Depth of best retrieval.** See Table 1.",
            f"**Strongest retrieval.** At the final output, {nm(top[0])} has the most matching references ({G[top[0]].loc['final_output'].m4:.2f} of four). Over all layers, the maximum is {steps[topl[0]].m4.max():.2f} at {steps[topl[0]].m4.idxmax().replace('_', ' ')} of {nm(topl[0])}.",
            "**Final output.** Where the vector that the model exposes for retrieval differs from the last layer (ViT, CLIP, SigLIP 2, BioCLIP, FLAVA, Qwen3-VL tower), it can retrieve fewer or more matching references than the last layer (Tables 4 onwards)."]))
add(("fig", "assets/layer_curves.png", "Fig. 1", "Replacement and class agreement of the retrieved references by depth.",
     f"**a**, Mean number of the four references that the preceding layer did not retrieve. **b**, Mean number of the four references with the class of the query. Means over {nq} queries. Depth is scaled to a common axis; each encoder has its own color and marker."))
add(("fig", "assets/example_query_all_layers.jpg", "Fig. 2", f"References retrieved by every layer for one query.",
     f"A {ds} query of the class {qlab} (top left) and the four most similar references at each layer of the encoders, with the most similar reference on top. Columns are layers (ResNet-50 starts at the stem; the Qwen3-VL tower has 24 layers). "
     "An orange frame marks a reference that the previous layer did not retrieve. A blue disc with a check marks a reference of the query's class. The number below each column is the count of such references."))
add(("h2", "Background"))
add(("p", "Assisted few-shot learning shows a vision-language model a few labeled reference images that resemble the query image. A frozen image encoder selects them: it maps every image to a vector, and the references with the most similar vectors are retrieved. An encoder has many layers, and each layer yields a different vector for the same image. Each layer therefore selects its own references for the same query. This resource places those references side by side for every layer of the encoders. The encoders differ in how they were pretrained: with class labels, with self-supervision, with image-text pairs, and as the vision tower of a multimodal language model."))
add(("h2", "Encoders and layers"))
FO = {"ViT": "pooler_output: a dense layer and tanh on the last CLS token", "DINOv2": "last CLS token, identical to layer_12", "CLIP": "projected image embedding (512 values)", "ResNet50": "pooled output, identical to layer_16",
      "SigLIP2": "pooled image embedding (attention-pooling head)", "BioCLIP": "projected image embedding (512 values)", "DINOv3": "last CLS token, identical to layer_12", "Qwen3VL": "mean of the merged visual tokens that enter the language model", "FLAVA": "projected image embedding"}
table("Encoders, layers and final outputs.", ["Folder", "Checkpoint", "Layers", "`final_output`"], [[f"`{mdir}`", f"`{ck}`", ld, FO[m]] for m, mdir, col, ck, ld in MODELS],
      "Every vector is scaled to unit length and references are ranked by cosine similarity. The final layer normalization of each transformer is applied to the tokens of each layer before the readout; the Qwen3-VL tower has none before its patch merger, so its layers are read as they are. Only the vision tower of Qwen3-VL and the image encoder of FLAVA are used; the language model of Qwen3-VL and the text and multimodal encoders of FLAVA are not loaded or not used.")
add(("h2", "Organization of the data"))
add(("pre", """<NN_dataset>/
  q<number>__<query class>__<query file>/      one folder per query
    QUERY__<class>.jpg                          the query image
    1_ViT/  2_DINOv2/  3_CLIP/  4_ResNet50/     one folder per encoder
    5_SigLIP2/  6_BioCLIP/  7_DINOv3/  8_Qwen3VL/  9_FLAVA/
      layer_01/ ... layer_12/                   one folder per layer (ResNet-50: layer_00_stem ... layer_16; Qwen3-VL: layer_01 ... layer_24)
      final_output/                             the vector the encoder exposes for retrieval
        top1/  top2/  top4/  top8/              the k most similar references of that layer, best first
          <rank>__<match|diff>__<first|kept|new>__<reference class>__<file>"""))
table("Tags in the reference file names.", ["Tag", "Meaning"], [["`match`", "The reference has the class of the query."], ["`diff`", "The reference has another class."], ["`new`", "The preceding layer did not retrieve this reference in the same top-k folder."],
                                                               ["`kept`", "The preceding layer retrieved it too."], ["`first`", "First layer of the encoder; there is no preceding layer."]],
      "In `final_output`, `new` and `kept` refer to the last layer. `top1` is contained in `top2`, `top2` in `top4`, and `top4` in `top8`.")
add(("p", "To inspect one layer step, open a query folder, select an encoder, and compare `layer_05/top4` with `layer_06/top4`. The files tagged `new` in `layer_06` are the references that layer 6 introduced."))
add(("h2", "Layer-wise results"))
for m, *_ in MODELS:
    g = G[m].loc[layers(m) + ["final_output"]]
    table(f"Layer-wise retrieval of {nm(m)}.", ["Layer", "New in top 4", "New in top 8", "Match in top 1 (%)", "Match in top 2", "Match in top 4", "Match in top 8"],
          [[l.replace("_", " "), "" if np.isnan(r.new4) else f"{r.new4:.2f}", "" if np.isnan(r.new8) else f"{r.new8:.2f}", f"{100 * r.m1:.0f}", f"{r.m2:.2f}", f"{r.m4:.2f}", f"{r.m8:.2f}"] for l, r in g.iterrows()],
          f"Means over {nq} queries. New, references that the preceding layer did not retrieve; match, references with the class of the query.")
add(("h2", "Results by dataset"))
dsr = []
for d in C.dataset.drop_duplicates():
    g = C[C.dataset == d]; dsr.append([d, str(len(g[(g.model == "ViT") & (g.layer == "layer_01")]))] + [f"{g[(g.model == m) & (g.layer != 'final_output')].new_in_top4.mean():.2f} / {g[(g.model == m) & (g.layer == 'final_output')].match_in_top4.mean():.2f}" for m, *_ in MODELS])
table("Replacement and class agreement by dataset.", ["Dataset", "Queries"] + [nm(m) for m, *_ in MODELS], dsr, "First value, new references in the top 4 per layer step; second value, matching references in the top 4 at the final output.")
add(("h2", "Limitations"))
add(("ul", ["The layers with the most matching references are identified on the same queries on which they are scored. They describe these pools and are not a validated choice.", "A larger number of matching references does not imply a higher accuracy of a vision-language model. That relation has to be tested with the model.",
            ("Each dataset is represented by 100 queries and a reference bank of 100 images. " if DISJ else "Each dataset is represented by a pool of about 100 images. ") + "The transformers are read at the CLS token where the model has one, and at the mean of the patch tokens otherwise (SigLIP 2, Qwen3-VL tower), so readouts differ between encoders.", "The Qwen3-VL tower receives a 448 x 448 input and the other encoders 224 x 224."]))
add(("h2", "Methods"))
add(("ul", ["**Data.** Five AgEval datasets (Bean Leaf Lesions, Mango Leaf Disease, DeepWeeds, Soybean Seeds, Soybean Diseases). " + ("Each has 100 test queries and a separate, class-balanced reference bank of 100 images (seed 42). The queries are the AgEval loader pool, topped up to 100; the bank is drawn from the rest of the dataset." if DISJ else "Each is the class-balanced pool of about 100 images drawn by the AgEval loader with seed 42."),
            ("**Retrieval.** Every query retrieves from the reference bank of its dataset. No image is in both sets, and no bank image is byte-identical to a query. Ties are broken by the reference identifier." if DISJ else "**Retrieval.** Every pool image is a query and retrieves from the rest of the pool. The query itself and byte-identical images are never references. Ties are broken by the reference identifier."),
            "**Inference.** Local inference with the checkpoints of Table 2; all layers of an image come from one forward pass."]))
add(("h2", "Data and code availability"))
add(("ul", ["`index.csv`: every query, encoder, layer and rank (1 to 8) with the reference identifier, its class, the cosine similarity and the class agreement.", "`layer_changes.csv`: one row per query, encoder and layer with the number of matching and new references in the top 1, 2, 4 and 8.",
            "`split.json`, `split_images.csv`: the retrieval protocol and every query and reference image with its class.", "`assets/`: the two figures of this document.", "Image files are hard links to the dataset images and take no additional disk space. Copying or compressing the folder produces regular files.",
            "Source: cached latents in `code/ageval-dinov2-pilot/runs/layer_retrieval_audit`; scripts `repo/make_retrieval_samples.py` and `repo/make_retrieval_readme.py`. The patch-token mean of each transformer layer is available in that audit and is not laid out here."]))

md = []
for b in B:
    k = b[0]
    if k == "h1": md.append("# " + b[1])
    elif k == "h2": md.append("## " + b[1])
    elif k == "meta": md.append("*" + b[1] + "*")
    elif k == "abstract": md.append("> **Summary.** " + b[1])
    elif k == "fig": md.append(f"![{b[2]}]({b[1]})\n\n**{b[2]} | {b[3]}** {b[4]}")
    elif k == "p": md.append(b[1])
    elif k == "pre": md.append("```\n" + b[1] + "\n```")
    elif k == "ul": md.append("\n".join("- " + x for x in b[1]))
    elif k == "table": md.append(f"**{b[1]} | {b[2]}**\n\n| " + " | ".join(b[3]) + " |\n|" + "---|" * len(b[3]) + "\n" + "\n".join("| " + " | ".join(r) + " |" for r in b[4]) + (f"\n\n<sub>{b[5]}</sub>" if b[5] else ""))
open(os.path.join(D, "README.md"), "w").write("\n\n".join(md) + "\n")

inl = lambda s: re.sub(r"\*\*([^*]+)\*\*", r"<b>\1</b>", re.sub(r"`([^`]+)`", r"<code>\1</code>", html.escape(s)))
COL = {nm(m): (col, MK[m]) for m, _, col, *_ in MODELS}; SHAPE = {"o": "border-radius:50%", "s": "", "^": "clip-path:polygon(50% 0,100% 100%,0 100%)", "D": "transform:rotate(45deg) scale(.8)", "v": "clip-path:polygon(0 0,100% 0,50% 100%)", "P": "clip-path:polygon(35% 0,65% 0,65% 35%,100% 35%,100% 65%,65% 65%,65% 100%,35% 100%,35% 65%,0 65%,0 35%,35% 35%)", "X": "clip-path:polygon(20% 0,50% 30%,80% 0,100% 20%,70% 50%,100% 80%,80% 100%,50% 70%,20% 100%,0 80%,30% 50%,0 20%)", "h": "clip-path:polygon(25% 5%,75% 5%,100% 50%,75% 95%,25% 95%,0 50%)", "*": "clip-path:polygon(50% 0,61% 35%,98% 35%,68% 57%,79% 91%,50% 70%,21% 91%,32% 57%,2% 35%,39% 35%)"}
def cell(c):
    t = re.sub(r"\*\*|`", "", c)
    return (f"<span class='mk' style='background:{COL[t][0]};{SHAPE[COL[t][1]]}'></span>" if t in COL else "") + inl(c)
H = []
for b in B:
    k = b[0]
    if k == "h1": H.append(f"<div class='kicker'>Resource</div><h1>{inl(b[1])}</h1>")
    elif k == "meta": H.append(f"<p class='meta'>{inl(b[1].replace('Resource · ', ''))}</p>")
    elif k == "abstract": H.append(f"<p class='abstract'>{inl(b[1])}</p>")
    elif k == "h2": H.append(f"<h2>{inl(b[1])}</h2>")
    elif k == "fig": H.append(f"<figure><img src='file://{os.path.join(D, b[1])}'><figcaption><b>{b[2]} | {inl(b[3])}</b> {inl(b[4])}</figcaption></figure>")
    elif k == "p": H.append(f"<p>{inl(b[1])}</p>")
    elif k == "pre": H.append(f"<pre>{html.escape(b[1])}</pre>")
    elif k == "ul": H.append("<ul>" + "".join(f"<li>{inl(x)}</li>" for x in b[1]) + "</ul>")
    elif k == "table": H.append(f"<div class='tb'><div class='tcap'><b>{b[1]} | {inl(b[2])}</b></div><table><thead><tr>" + "".join(f"<th>{inl(c)}</th>" for c in b[3]) + "</tr></thead><tbody>" + "".join("<tr>" + "".join(f"<td>{cell(c)}</td>" for c in r) + "</tr>" for r in b[4]) + "</tbody></table>" + (f"<div class='tnote'>{inl(b[5])}</div>" if b[5] else "") + "</div>")
css = """@page{size:Letter;margin:17mm 17mm 16mm}*{box-sizing:border-box}body{font:9.6pt/1.5 Helvetica,Arial,sans-serif;color:#111;margin:0;-webkit-print-color-adjust:exact;print-color-adjust:exact}
.kicker{font-size:8.5pt;font-weight:700;letter-spacing:.09em;text-transform:uppercase;color:#0072B2;border-top:3pt solid #0072B2;padding-top:5pt;display:inline-block}
h1{font-size:20pt;line-height:1.17;font-weight:700;margin:6pt 0 6pt;letter-spacing:-.005em}.meta{font-size:8.6pt;color:#555;margin:0 0 9pt;padding-bottom:8pt;border-bottom:.6pt solid #999}
.abstract{font-size:10.4pt;line-height:1.5;font-weight:700;margin:0 0 10pt}
h2{font-size:12pt;font-weight:700;margin:15pt 0 5pt;break-after:avoid}p{margin:0 0 7pt;text-align:left}ul{margin:0 0 8pt;padding-left:13pt}li{margin-bottom:3pt}
code{font:8.4pt Menlo,monospace;background:#f1f2f4;padding:0 2pt;border-radius:1.5pt}pre{font:7.9pt/1.5 Menlo,monospace;background:#f5f6f8;border-left:2.5pt solid #0072B2;padding:8pt 10pt;white-space:pre-wrap;break-inside:avoid;margin:4pt 0 9pt}
figure{margin:8pt 0 11pt;break-inside:avoid;text-align:center}figure img{max-width:100%;max-height:205mm}figcaption{font-size:8.5pt;line-height:1.45;text-align:left;margin-top:5pt}
.tb{break-inside:avoid;margin:7pt 0 11pt}.tcap{font-size:8.8pt;margin-bottom:3pt}table{border-collapse:collapse;font-size:8.6pt;width:100%;border-top:1.2pt solid #111;border-bottom:1.2pt solid #111}
th{text-align:left;padding:4pt 7pt 4pt 0;font-weight:700;border-bottom:.6pt solid #111;vertical-align:bottom}td{padding:2.6pt 7pt 2.6pt 0;vertical-align:top;font-variant-numeric:tabular-nums}td:first-child{white-space:nowrap}
.tnote{font-size:7.9pt;color:#444;margin-top:3pt}.mk{display:inline-block;width:7pt;height:7pt;margin-right:5pt}"""
hp = os.path.join(A, "_readme.html"); open(hp, "w").write(f"<!doctype html><html><head><meta charset='utf-8'><title>Layer-by-layer reference atlas</title><style>{css}</style></head><body>{''.join(H)}</body></html>")
subprocess.run(["/Applications/Google Chrome.app/Contents/MacOS/Google Chrome", "--headless=new", "--disable-gpu", "--no-pdf-header-footer", f"--print-to-pdf={os.path.join(D, 'README.pdf')}", "file://" + hp], capture_output=True)
os.remove(hp); print("example query:", ds, qf)
