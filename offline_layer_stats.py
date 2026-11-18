"""Offline (no VLM) retrieval statistics over the AgEval classification tasks.

Every image of the loader's pool is a query that retrieves its top-k from the rest
of the pool (the repository's leave-one-out protocol). Conditions:
  A = original ViT pooler_output, B = DINOv2 block 12 CLS, C = DINOv2 block 6 CLS.
Usage: python offline_layer_stats.py <out_dir> [task ...]
"""
import hashlib, json, os, shutil, sys, traceback
from sklearn.metrics import f1_score
import numpy as np
import pandas as pd
from scipy import stats
import data_loader as dl
from get_embeddingp import get_image_embedding

K, SAMPLES, SEED, NBOOT = 4, 100, 42, 10000
TASKS = {
    "Durum Wheat": dl.load_and_prepare_data_DurumWheat, "Soybean Seeds": dl.load_and_prepare_data_soybean_seeds,
    "Mango Leaf Disease": dl.load_and_prepare_data_mango_leaf, "Bean Leaf Lesions": dl.load_and_prepare_data_bean_leaf,
    "Soybean Diseases": dl.load_and_prepare_data_Soybean_PNAS, "Dangerous Insects": dl.load_and_prepare_data_Soybean_Dangerous_Insects,
    "DeepWeeds": dl.load_and_prepare_data_DeepWeeds, "Yellow Rust 19": dl.load_and_prepare_data_YellowRust,
    "FUSARIUM 22": dl.load_and_prepare_data_FUSARIUM22, "IDC": dl.load_and_prepare_data_IDC,
}
KIND = {t: "identification" for t in list(TASKS)[:7]}
KIND.update({"Yellow Rust 19": "severity classification", "FUSARIUM 22": "severity classification", "IDC": "severity classification"})
SWEEP = ["vit"] + [f"dinov2_b{k}" for k in range(1, 13)]
ENC = SWEEP + ["clip", "resnet"]
A, B, C = "vit", "dinov2_b12", "dinov2_b6"
FOUR = ["vit", "clip", "resnet", "dinov2_b12"]  # the repository's three encoders plus DINOv2
NAME = {"vit": "ViT", "clip": "CLIP", "resnet": "ResNet-50", "dinov2_b12": "DINOv2 b12", "dinov2_b6": "DINOv2 b6"}

out = sys.argv[1]
os.makedirs(out, exist_ok=True)
tasks = sys.argv[2:] or list(TASKS)
rows, failed = [], {}
for t in tasks:
    cache = os.path.join(out, f"per_query_{t.replace(' ', '_')}.csv")
    if os.path.exists(cache) and set(ENC) <= set(pd.read_csv(cache, usecols=["encoder"])["encoder"]):
        rows.append(pd.read_csv(cache)); continue
    try:
        data, classes, name = TASKS[t](SAMPLES)
        n = len(data)
        paths, labels = [data.at[i, 0] for i in range(n)], [str(data.at[i, 1]) for i in range(n)]
        md5 = [hashlib.md5(open(p, "rb").read()).hexdigest() for p in paths]
        pd.DataFrame({"idx": range(n), "path": paths, "label": labels, "md5": md5}).to_csv(os.path.join(out, f"pool_{t.replace(' ', '_')}.csv"), index=False)
        lab = np.array(labels)
        # reusable cache: the pool images themselves and every encoder's raw vectors
        img_dir = os.path.join(out, "pool_images", t.replace(" ", "_"))
        os.makedirs(img_dir, exist_ok=True)
        for i, p in enumerate(paths):
            dst = os.path.join(img_dir, f"{i:03d}_{os.path.basename(p)}")
            if not os.path.exists(dst):
                shutil.copyfile(p, dst)
        emb_path = os.path.join(out, "embeddings", f"{t.replace(' ', '_')}.npz")
        os.makedirs(os.path.dirname(emb_path), exist_ok=True)
        emb_cache = dict(np.load(emb_path)) if os.path.exists(emb_path) else {}
        dup = np.array([[md5[i] == md5[j] for j in range(n)] for i in range(n)])  # includes self
        task_rows = []
        for e in ENC:
            if e not in emb_cache:
                embs = [get_image_embedding(p, model_type=e) for p in paths]
                bad = [paths[i] for i, v in enumerate(embs) if isinstance(v, dict)]
                assert not bad, f"embedding failed: {bad[:3]}"
                emb_cache[e] = np.array(embs, dtype=np.float32)
                np.savez_compressed(emb_path, **emb_cache)
            X = emb_cache[e].astype(np.float64)
            assert np.isfinite(X).all()
            X = X / np.linalg.norm(X, axis=1, keepdims=True)
            S = X @ X.T
            S[dup] = -np.inf
            for q in range(n):
                order = sorted(range(n), key=lambda j: (-S[q, j], j))[:K]
                assert q not in order and len(set(order)) == K
                same = int((lab[order] == lab[q]).sum())
                votes = {}
                for j in order:
                    votes[lab[j]] = votes.get(lab[j], 0.0) + S[q, j]
                vote = max(sorted(votes), key=lambda c: (sum(lab[j] == c for j in order), votes[c]))
                task_rows.append({"task": t, "kind": KIND[t], "query_idx": q, "label": lab[q], "encoder": e, "same_class": same,
                                  "top1_same": int(lab[order[0]] == lab[q]), "vote_correct": int(vote == lab[q]), "vote_pred": vote,
                                  "top_idx": json.dumps(order)})
        df = pd.DataFrame(task_rows)
        df.to_csv(cache, index=False)
        rows.append(df)
        print(f"[done] {t}: pool {n}, classes {len(set(labels))}, exact-duplicate pairs {(dup.sum() - n) // 2}", flush=True)
    except Exception as ex:
        failed[t] = f"{type(ex).__name__}: {ex}"
        traceback.print_exc()
        print(f"[failed] {t}: {failed[t]}", flush=True)

allq = pd.concat(rows, ignore_index=True)
rng = np.random.default_rng(SEED)

def boot_ci(d):
    idx = rng.integers(0, len(d), size=(NBOOT, len(d)))
    m = d[idx].mean(axis=1)
    return float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5))

def wilcoxon_p(d):
    return 1.0 if not np.any(d != 0) else float(stats.wilcoxon(d, zero_method="wilcox", alternative="two-sided").pvalue)

def mcnemar_p(x, y):  # exact, two-sided, on discordant pairs
    b, c = int(((y == 1) & (x == 0)).sum()), int(((y == 0) & (x == 1)).sum())
    return (1.0 if b + c == 0 else float(stats.binomtest(b, b + c, 0.5).pvalue)), b, c

def holm(p):
    p = np.asarray(p); order = np.argsort(p); adj = np.empty(len(p)); run = 0.0
    for r, i in enumerate(order):
        run = max(run, (len(p) - r) * p[i]); adj[i] = min(1.0, run)
    return adj

def compare(x_enc, y_enc, frame=None):  # y minus x
    frame = allq if frame is None else frame
    res = []
    for t, g in frame.groupby("task", sort=False):
        P = g.pivot(index="query_idx", columns="encoder")
        d = (P["same_class"][y_enc] - P["same_class"][x_enc]).to_numpy(float)
        lo, hi = boot_ci(d)
        pm, b, c = mcnemar_p(P["vote_correct"][x_enc].to_numpy(), P["vote_correct"][y_enc].to_numpy())
        res.append({"task": t, "kind": KIND[t], "n": len(d), f"mean_{x_enc}": float(P["same_class"][x_enc].mean()),
                    f"mean_{y_enc}": float(P["same_class"][y_enc].mean()), "diff": float(d.mean()), "ci_lo": lo, "ci_hi": hi,
                    "queries_better": int((d > 0).sum()), "queries_worse": int((d < 0).sum()), "queries_tied": int((d == 0).sum()),
                    "wilcoxon_p": wilcoxon_p(d), f"vote_acc_{x_enc}": float(P["vote_correct"][x_enc].mean()),
                    f"vote_acc_{y_enc}": float(P["vote_correct"][y_enc].mean()), "vote_wins": b, "vote_losses": c, "mcnemar_p": pm})
    r = pd.DataFrame(res)
    r["wilcoxon_p_holm"] = holm(r["wilcoxon_p"]); r["mcnemar_p_holm"] = holm(r["mcnemar_p"])
    # pooled over tasks: bootstrap resamples queries within each task; sign test on task means
    P = frame.pivot(index=["task", "query_idx"], columns="encoder")
    d = (P["same_class"][y_enc] - P["same_class"][x_enc])
    per_task = [g.to_numpy(float) for _, g in d.groupby(level=0, sort=False)]
    bm = np.array([np.concatenate([g[rng.integers(0, len(g), len(g))] for g in per_task]).mean() for _ in range(2000)])
    tm = np.array([g.mean() for g in per_task])
    pm, b, c = mcnemar_p(P["vote_correct"][x_enc].to_numpy(), P["vote_correct"][y_enc].to_numpy())
    pooled = {"n_queries": int(len(d)), "n_tasks": len(per_task), "diff": float(d.mean()), "ci_lo": float(np.percentile(bm, 2.5)),
              "ci_hi": float(np.percentile(bm, 97.5)), "wilcoxon_p": wilcoxon_p(d.to_numpy(float)),
              "tasks_better": int((tm > 0).sum()), "tasks_worse": int((tm < 0).sum()),
              "task_sign_test_p": float(stats.binomtest(int((tm > 0).sum()), int((tm != 0).sum()), 0.5).pvalue) if (tm != 0).any() else 1.0,
              "task_level_wilcoxon_p": wilcoxon_p(tm), "vote_wins": b, "vote_losses": c, "mcnemar_p": pm,
              f"vote_acc_{x_enc}": float(P["vote_correct"][x_enc].mean()), f"vote_acc_{y_enc}": float(P["vote_correct"][y_enc].mean())}
    return r, pooled

cb, cb_pool = compare(B, C)
ca, ca_pool = compare(A, C)
ba, ba_pool = compare(A, B)
others = {}
for x in ["clip", "resnet"]:
    others[x] = compare(x, B)
    others[x][0].to_csv(os.path.join(out, f"B_minus_{x}.csv"), index=False)
groups = {}
for kind in ["identification", "severity classification"]:
    sub = allq[allq["kind"] == kind]
    if len(sub):
        groups[kind] = {"C_minus_B": compare(B, C, sub)[1], "C_minus_A": compare(A, C, sub)[1]}
cb.to_csv(os.path.join(out, "C_minus_B.csv"), index=False); ca.to_csv(os.path.join(out, "C_minus_A.csv"), index=False)
ba.to_csv(os.path.join(out, "B_minus_A.csv"), index=False)
sweep = allq.groupby(["task", "encoder"], sort=False)["same_class"].mean().unstack()[SWEEP]
sweep_vote = allq.groupby(["task", "encoder"], sort=False)["vote_correct"].mean().unstack()[SWEEP]
sweep.to_csv(os.path.join(out, "layer_sweep_same_class.csv")); sweep_vote.to_csv(os.path.join(out, "layer_sweep_vote_acc.csv"))
json.dump({"k": K, "loader_samples": SAMPLES, "seed": SEED, "bootstrap": NBOOT, "failed_tasks": failed,
           "C_minus_B_pooled": cb_pool, "C_minus_A_pooled": ca_pool, "B_minus_A_pooled": ba_pool, "by_task_group": groups, "B_minus_clip_pooled": others["clip"][1], "B_minus_resnet_pooled": others["resnet"][1]}, open(os.path.join(out, "stats.json"), "w"), indent=2)

def fp(p):
    return "<0.001" if p < 0.001 else f"{p:.3f}"
def table(r, x, y, xn, yn):
    L = [f"| Task | Group | n | {xn} | {yn} | Diff (95% CI) | Better / worse / tied | Wilcoxon p (Holm) | 4-NN vote acc. {xn} → {yn} | McNemar p (Holm) |", "|---|---|---|---|---|---|---|---|---|---|"]
    for _, w in r.iterrows():
        L.append(f"| {w['task']} | {w['kind']} | {w['n']} | {w[f'mean_{x}']:.2f} | {w[f'mean_{y}']:.2f} | {w['diff']:+.2f} ({w['ci_lo']:+.2f}, {w['ci_hi']:+.2f}) | "
                 f"{w['queries_better']} / {w['queries_worse']} / {w['queries_tied']} | {fp(w['wilcoxon_p'])} ({fp(w['wilcoxon_p_holm'])}) | "
                 f"{100*w[f'vote_acc_{x}']:.0f}% → {100*w[f'vote_acc_{y}']:.0f}% | {fp(w['mcnemar_p'])} ({fp(w['mcnemar_p_holm'])}) |")
    return L
def pooled_line(name, p, x, y):
    return (f"**{name}, pooled over {p['n_tasks']} tasks and {p['n_queries']} queries:** mean difference {p['diff']:+.2f} same-class demonstrations of {K} "
            f"(95% CI {p['ci_lo']:+.2f} to {p['ci_hi']:+.2f}), query-level Wilcoxon p {fp(p['wilcoxon_p'])}. The difference is positive on {p['tasks_better']} tasks and "
            f"negative on {p['tasks_worse']} (sign test p {fp(p['task_sign_test_p'])}, task-level Wilcoxon p {fp(p['task_level_wilcoxon_p'])}). "
            f"4-NN vote accuracy {100*p[f'vote_acc_{x}']:.1f}% → {100*p[f'vote_acc_{y}']:.1f}%, {p['vote_wins']} wins and {p['vote_losses']} losses (McNemar p {fp(p['mcnemar_p'])}).\n")
L = ["# AgEval offline retrieval statistics: DINOv2 block 6 versus block 12 and the original ViT\n",
     f"No VLM calls. Each task uses the repository loader with `samples={SAMPLES}` (seed 42). Every pool image is a query that retrieves its top {K} "
     "from the rest of the pool by cosine similarity. A = original ViT `pooler_output`, B = DINOv2 ViT-B/14 block 12 CLS, C = block 6 CLS.\n",
     f"The main measure is the number of same-class demonstrations in the top {K}. The 4-NN vote accuracy is the accuracy of the majority label "
     "of the retrieved demonstrations, with ties broken by summed cosine.\n",
     "## Four representations: ViT, CLIP, ResNet-50 (repository encoders) and DINOv2 block 12\n", "FOUR_PLACEHOLDER",
     "## C minus B (primary, encoder fixed)\n", pooled_line("C minus B", cb_pool, B, C)] + [pooled_line(f"C minus B, {k} tasks only", g["C_minus_B"], B, C) for k, g in groups.items()] + table(cb, B, C, "B", "C") + [
     "\n## C minus A (encoder also changes)\n", pooled_line("C minus A", ca_pool, A, C)] + [pooled_line(f"C minus A, {k} tasks only", g["C_minus_A"], A, C) for k, g in groups.items()] + table(ca, A, C, "A", "C") + [
     "\n## B minus A (for reference)\n", pooled_line("B minus A", ba_pool, A, B)] + table(ba, A, B, "A", "B") + [
     f"\n## Layer sweep: mean same-class demonstrations in the top {K}\n",
     "| Task | A (ViT) | " + " | ".join(f"b{k}" for k in range(1, 13)) + " | Best block |", "|---|" + "---|" * 14]
for t, w in sweep.iterrows():
    best = max(range(1, 13), key=lambda k: w[f"dinov2_b{k}"])
    L.append(f"| {t} | {w['vit']:.2f} | " + " | ".join(f"{w[f'dinov2_b{k}']:.2f}" for k in range(1, 13)) + f" | {best} |")
L.append("| **Mean over tasks** | " + f"{sweep['vit'].mean():.2f} | " + " | ".join(f"{sweep[f'dinov2_b{k}'].mean():.2f}" for k in range(1, 13)) + " | |")
five = FOUR + [C]
m_same = allq.groupby(["task", "encoder"], sort=False)["same_class"].mean().unstack()[five]
m_acc = allq.groupby(["task", "encoder"], sort=False)["vote_correct"].mean().unstack()[five]
tasks_done = list(m_same.index)
four = [f"Mean same-class demonstrations in the top {K}, with embedding-only accuracy in parentheses. Bold marks the best of the four per task; DINOv2 block 6 is shown for reference.\n",
        "| Task | Group | " + " | ".join(NAME[e] for e in five) + " |", "|---|---|" + "---|" * len(five)]
for t in tasks_done:
    best = max(FOUR, key=lambda e: m_same.loc[t, e])
    four.append(f"| {t} | {KIND[t]} | " + " | ".join(("**" if e == best else "") + f"{m_same.loc[t, e]:.2f} ({100*m_acc.loc[t, e]:.0f}%)" + ("**" if e == best else "") for e in five) + " |")
four.append("| **Mean over tasks** | | " + " | ".join(f"{m_same[e].mean():.2f} ({100*m_acc[e].mean():.1f}%)" for e in five) + " |")
four.append("| **Mean rank among the four (1 = best)** | | " + " | ".join(f"{m_same[FOUR].rank(axis=1, ascending=False)[e].mean():.2f}" if e in FOUR else "" for e in five) + " |")
fr = stats.friedmanchisquare(*[m_same[e].to_numpy() for e in FOUR]) if len(tasks_done) >= 3 else None
four.append("")
if fr is not None:
    four.append(f"Friedman test over the four representations with tasks as blocks: chi-square {fr.statistic:.2f}, p {fp(fr.pvalue)}.\n")
four += [pooled_line("DINOv2 block 12 minus ViT", ba_pool, A, B), pooled_line("DINOv2 block 12 minus CLIP", others["clip"][1], "clip", B),
         pooled_line("DINOv2 block 12 minus ResNet-50", others["resnet"][1], "resnet", B),
         "### DINOv2 block 12 minus CLIP, per task\n"] + table(others["clip"][0], "clip", B, "CLIP", "DINOv2 b12") + [
         "\n### DINOv2 block 12 minus ResNet-50, per task\n"] + table(others["resnet"][0], "resnet", B, "ResNet", "DINOv2 b12") + [""]
i = L.index("FOUR_PLACEHOLDER"); L[i:i + 1] = four
def f1(g):
    return f1_score(g["label"].astype(str), g["vote_pred"].astype(str), labels=sorted(g["label"].astype(str).unique()), average="macro", zero_division=0)
eo = allq[allq["encoder"].isin([A, B, C])].groupby(["task", "encoder"], sort=False).apply(lambda g: pd.Series({"acc": g["vote_correct"].mean(), "f1": f1(g)})).unstack()
eo.to_csv(os.path.join(out, "embedding_only.csv"))
L += [f"\n## Embedding-only prediction (no VLM): majority label of the top {K}\n",
      "Paired p-values are exact McNemar tests on the same queries, Holm-adjusted over tasks in parentheses.\n",
      "| Task | Group | A acc. / macro-F1 | B acc. / macro-F1 | C acc. / macro-F1 | C−B wins / losses, p | C−A wins / losses, p |", "|---|---|---|---|---|---|---|"]
for (t, w), (_, x), (_, y) in zip(eo.iterrows(), cb.iterrows(), ca.iterrows()):
    L.append(f"| {t} | {KIND[t]} | " + " | ".join(f"{100*w[('acc', e)]:.0f}% / {w[('f1', e)]:.3f}" for e in [A, B, C])
             + f" | {x['vote_wins']} / {x['vote_losses']}, {fp(x['mcnemar_p'])} ({fp(x['mcnemar_p_holm'])}) | {y['vote_wins']} / {y['vote_losses']}, {fp(y['mcnemar_p'])} ({fp(y['mcnemar_p_holm'])}) |")
L.append("| **Mean over tasks** | | " + " | ".join(f"{100*eo[('acc', e)].mean():.1f}% / {eo[('f1', e)].mean():.3f}" for e in [A, B, C])
         + f" | {cb_pool['vote_wins']} / {cb_pool['vote_losses']}, {fp(cb_pool['mcnemar_p'])} | {ca_pool['vote_wins']} / {ca_pool['vote_losses']}, {fp(ca_pool['mcnemar_p'])} |")
if failed:
    L += ["\n## Tasks that did not run\n"] + [f"- {t}: {m}" for t, m in failed.items()]
notes = os.path.join(out, "NOTES.md")
if os.path.exists(notes):
    L.append("\n" + open(notes).read())
open(os.path.join(out, "SUMMARY.md"), "w").write("\n".join(L) + "\n")
print("\n".join(L))
