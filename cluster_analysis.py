"""Class-cluster quality of each representation, from cached vectors (no model or VLM calls).

Reads <run_dir>/embeddings/<task>.npz and <run_dir>/pool_<task>.csv written by offline_layer_stats.py.
Usage: python cluster_analysis.py <run_dir>
"""
import glob, json, os, sys
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score, roc_auc_score, silhouette_samples

run = sys.argv[1]
REPS = {"vit": "A ViT", "dinov2_b12": "B DINOv2 b12", "dinov2_b6": "C DINOv2 b6", "clip": "CLIP", "resnet": "ResNet-50"}
PAIRS = [("dinov2_b12", "dinov2_b6", "C − B"), ("vit", "dinov2_b6", "C − A"), ("vit", "dinov2_b12", "B − A"),
         ("clip", "dinov2_b12", "B − CLIP"), ("resnet", "dinov2_b12", "B − ResNet")]
KS, SEED, NBOOT = [1, 4, 8], 42, 10000
ORDER = ["Durum Wheat", "Soybean Seeds", "Mango Leaf Disease", "Bean Leaf Lesions", "Soybean Diseases", "Dangerous Insects", "DeepWeeds",
         "Yellow Rust 19", "FUSARIUM 22", "IDC"]
rng = np.random.default_rng(SEED)

def knn_correct(S, y, k):  # leave-one-out, majority label, ties broken by summed cosine
    out = np.zeros(len(y), dtype=int)
    for q in range(len(y)):
        order = sorted((j for j in range(len(y)) if j != q), key=lambda j: (-S[q, j], j))[:k]
        cnt, sim = {}, {}
        for j in order:
            cnt[y[j]] = cnt.get(y[j], 0) + 1; sim[y[j]] = sim.get(y[j], 0.0) + S[q, j]
        out[q] = int(max(sorted(cnt), key=lambda c: (cnt[c], sim[c])) == y[q])
    return out

rows, per_sample = [], []
for t in ORDER:
    f = os.path.join(run, "embeddings", t.replace(" ", "_") + ".npz")
    if not os.path.exists(f):
        continue
    emb = np.load(f)
    y = pd.read_csv(os.path.join(run, f"pool_{t.replace(' ', '_')}.csv"))["label"].astype(str).to_numpy()
    n, n_cls = len(y), len(set(y))
    iu = np.triu_indices(n, 1)
    same = (y[:, None] == y[None, :])[iu]
    for r in REPS:
        X = emb[r].astype(np.float64); X /= np.linalg.norm(X, axis=1, keepdims=True)
        S = X @ X.T
        D = np.clip(1 - S, 0, None); np.fill_diagonal(D, 0)
        sil = silhouette_samples(D, y, metric="precomputed")
        # each image: mean cosine to its own class minus mean cosine to the nearest other class, in pooled-SD units is not needed; keep raw margin
        own = np.array([S[i, (y == y[i]) & (np.arange(n) != i)].mean() for i in range(n)])
        other = np.array([max(S[i, y == c].mean() for c in set(y) if c != y[i]) for i in range(n)])
        km = KMeans(n_clusters=n_cls, n_init=20, random_state=SEED).fit_predict(X)
        row = {"task": t, "rep": r, "n": n, "classes": n_cls, "silhouette": sil.mean(), "pair_auroc": roc_auc_score(same, S[iu]),
               "margin": (own - other).mean(), "kmeans_nmi": normalized_mutual_info_score(y, km), "kmeans_ari": adjusted_rand_score(y, km)}
        d = {"task": t, "rep": r, "idx": np.arange(n), "silhouette": sil, "margin": own - other}
        for k in KS:
            c = knn_correct(S, y, k); row[f"knn{k}"] = c.mean(); d[f"knn{k}"] = c
        rows.append(row); per_sample.append(pd.DataFrame(d))
M = pd.DataFrame(rows); PS = pd.concat(per_sample, ignore_index=True)
M.to_csv(os.path.join(run, "cluster_metrics.csv"), index=False); PS.to_csv(os.path.join(run, "cluster_per_image.csv"), index=False)
tasks = [t for t in ORDER if t in set(M["task"])]
METRICS = [("silhouette", "Silhouette (cosine)"), ("pair_auroc", "Pairwise same-class AUROC"), ("knn1", "1-NN accuracy"), ("knn4", "4-NN accuracy"),
           ("knn8", "8-NN accuracy"), ("kmeans_nmi", "k-means NMI"), ("kmeans_ari", "k-means ARI")]

def fp(p):
    return "<0.001" if p < 0.001 else f"{p:.3f}"
def wil(d):
    return 1.0 if not np.any(d != 0) else float(stats.wilcoxon(d).pvalue)
def holm(p):
    p = np.asarray(p); o = np.argsort(p); a = np.empty(len(p)); m = 0.0
    for r, i in enumerate(o):
        m = max(m, (len(p) - r) * p[i]); a[i] = min(1.0, m)
    return a

L = ["# AgEval: which representation forms better class clusters\n",
     f"No model or VLM calls. Vectors come from the cache: the repository loader's pool for each task (about 100 images, seed 42), L2-normalized, cosine similarity. "
     f"{len(tasks)} tasks: {', '.join(tasks)}.\n",
     "- **Silhouette (cosine):** how much closer an image is to its own class than to the nearest other class, from −1 to 1.",
     "- **Pairwise same-class AUROC:** the probability that a same-class pair has higher cosine than a different-class pair; 0.5 is chance.",
     "- **k-NN accuracy:** leave-one-out majority label of the k nearest images.",
     "- **k-means NMI and ARI:** agreement between the labels and k-means clusters (k = number of classes, 20 restarts).\n",
     "## Mean over tasks\n", "| Metric | " + " | ".join(REPS.values()) + " | Friedman p |", "|---|" + "---|" * (len(REPS) + 1)]
summary = {}
for key, name in METRICS:
    W = M.pivot(index="task", columns="rep", values=key).loc[tasks, list(REPS)]
    best = W.mean().idxmax()
    fr = stats.friedmanchisquare(*[W[r].to_numpy() for r in REPS]).pvalue
    summary[key] = {"mean": W.mean().to_dict(), "mean_rank": W.rank(axis=1, ascending=False).mean().to_dict(), "friedman_p": float(fr)}
    L.append(f"| {name} | " + " | ".join(("**" if r == best else "") + f"{W[r].mean():.3f}" + ("**" if r == best else "") for r in REPS) + f" | {fp(fr)} |")
R = pd.concat([M.pivot(index="task", columns="rep", values=k).loc[tasks, list(REPS)].rank(axis=1, ascending=False) for k, _ in METRICS])
L.append("| **Mean rank over all metrics and tasks (1 = best)** | " + " | ".join(f"{R[r].mean():.2f}" for r in REPS) + " | |")
W = M.pivot(index="task", columns="rep", values="silhouette").loc[tasks, list(REPS)]
wins = {r: 0 for r in REPS}
for key, _ in METRICS:
    V = M.pivot(index="task", columns="rep", values=key).loc[tasks, list(REPS)]
    for t in tasks:
        wins[V.loc[t].idxmax()] += 1
L.append("| **Times best (task × metric)** | " + " | ".join(str(wins[r]) for r in REPS) + f" | of {len(tasks) * len(METRICS)} |")

L += ["\n## Paired comparisons\n",
      "Per-image tests pair the same images under two representations. Silhouette uses a Wilcoxon signed-rank test with a bootstrap 95% CI; 4-NN uses an exact McNemar test. "
      "The task-level test is a Wilcoxon signed-rank test on the per-task means, which does not assume that images in one pool are independent.\n",
      "| Comparison | Silhouette diff (95% CI) | Per-image p | Tasks better / worse | Task-level p | Pairwise AUROC diff | 4-NN wins / losses | McNemar p | k-means NMI diff |", "|---|---|---|---|---|---|---|---|---|"]
pair_tables = {}
for x, y_, name in PAIRS:
    P = PS.pivot(index=["task", "idx"], columns="rep")
    d = (P["silhouette"][y_] - P["silhouette"][x])
    groups = [g.to_numpy() for _, g in d.groupby(level=0, sort=False)]
    bm = np.array([np.concatenate([g[rng.integers(0, len(g), len(g))] for g in groups]).mean() for _ in range(2000)])
    tm = np.array([g.mean() for g in groups])
    a, b = P["knn4"][x].to_numpy(), P["knn4"][y_].to_numpy()
    w_, l_ = int(((b == 1) & (a == 0)).sum()), int(((b == 0) & (a == 1)).sum())
    mp = 1.0 if w_ + l_ == 0 else float(stats.binomtest(w_, w_ + l_, 0.5).pvalue)
    Wm = lambda k: M.pivot(index="task", columns="rep", values=k).loc[tasks]
    L.append(f"| {name} | {d.mean():+.3f} ({np.percentile(bm, 2.5):+.3f}, {np.percentile(bm, 97.5):+.3f}) | {fp(wil(d.to_numpy()))} | {(tm > 0).sum()} / {(tm < 0).sum()} | {fp(wil(tm))} | "
             f"{(Wm('pair_auroc')[y_] - Wm('pair_auroc')[x]).mean():+.3f} | {w_} / {l_} | {fp(mp)} | {(Wm('kmeans_nmi')[y_] - Wm('kmeans_nmi')[x]).mean():+.3f} |")
    summary[name] = {"silhouette_diff": float(d.mean()), "ci": [float(np.percentile(bm, 2.5)), float(np.percentile(bm, 97.5))], "per_image_p": wil(d.to_numpy()),
                     "tasks_better": int((tm > 0).sum()), "tasks_worse": int((tm < 0).sum()), "task_level_p": wil(tm), "knn4_wins": w_, "knn4_losses": l_, "mcnemar_p": mp}
    rows_ = []
    for t, g in zip(tasks, groups):
        idx = rng.integers(0, len(g), size=(NBOOT, len(g))); bb = g[idx].mean(axis=1)
        rows_.append({"task": t, "diff": g.mean(), "lo": np.percentile(bb, 2.5), "hi": np.percentile(bb, 97.5), "p": wil(g)})
    pt = pd.DataFrame(rows_); pt["p_holm"] = holm(pt["p"]); pair_tables[name] = pt

for name in ["C − B", "C − A", "B − A"]:
    pt = pair_tables[name]
    L += [f"\n### {name}: per-task silhouette difference\n", "| Task | Diff (95% CI) | Wilcoxon p (Holm) |", "|---|---|---|"]
    L += [f"| {w['task']} | {w['diff']:+.3f} ({w['lo']:+.3f}, {w['hi']:+.3f}) | {fp(w['p'])} ({fp(w['p_holm'])}) |" for _, w in pt.iterrows()]

for key, name in METRICS:
    V = M.pivot(index="task", columns="rep", values=key).loc[tasks, list(REPS)]
    L += [f"\n## {name} per task\n", "| Task | Classes | " + " | ".join(REPS.values()) + " |", "|---|---|" + "---|" * len(REPS)]
    for t in tasks:
        b = V.loc[t].idxmax()
        L.append(f"| {t} | {int(M[M.task == t]['classes'].iloc[0])} | " + " | ".join(("**" if r == b else "") + f"{V.loc[t, r]:.3f}" + ("**" if r == b else "") for r in REPS) + " |")
json.dump(summary, open(os.path.join(run, "cluster_stats.json"), "w"), indent=2)
notes = os.path.join(run, "CLUSTER_NOTES.md")
if os.path.exists(notes):
    L.append("\n" + open(notes).read())
open(os.path.join(run, "CLUSTER_SUMMARY.md"), "w").write("\n".join(L) + "\n")
print("\n".join(L))
