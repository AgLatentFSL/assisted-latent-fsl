"""Report for a retrieval-condition VLM pilot with any set of conditions. Usage: python pilot_report_generic.py <run_dir>"""
import itertools, json, os, sys
import pandas as pd
from scipy import stats
from sklearn.metrics import f1_score

run = sys.argv[1]
cfg = json.load(open(os.path.join(run, "run_config.json")))
classes, k, encs = [str(c) for c in cfg["expected_classes"]], cfg["shots"], cfg["encoders"]
NAME = {"random": "Random demonstrations", "vit": "A. ViT retrieval", "dinov2_last": "B. DINOv2 block 12 retrieval", "dinov2_mid": "C. DINOv2 block 6 retrieval"}
SHORT = {"random": "Random", "vit": "A. ViT", "dinov2_last": "B. DINOv2 block 12", "dinov2_mid": "C. DINOv2 block 6"}
pred = pd.read_csv(os.path.join(run, "vlm", "predictions.csv"), keep_default_na=False)
ledger = [json.loads(l) for l in open(os.path.join(run, "vlm", "request_ledger.jsonl"))]
ok = pred[pred["status"] == "ok"]
valid = sorted(q for q in pred["query_idx"].unique() if all(((ok["query_idx"] == q) & (ok["encoder"] == e)).any() for e in encs))
ok = ok[ok["query_idx"].isin(valid)]
truth = ok.drop_duplicates("query_idx").set_index("query_idx")["true_label"].astype(str)
def fp(p): return "<0.001" if p < 0.001 else f"{p:.3f}"
def vote(labels):  # majority label of the demonstrations, earliest-ranked label wins ties
    labels = json.loads(labels); return max(labels, key=lambda c: (labels.count(c), -labels.index(c)))
res, P, V = {}, {}, {}
for e in encs:
    g = ok[ok["encoder"] == e].set_index("query_idx").loc[valid]
    P[e] = g["prediction"].astype(str); V[e] = g["example_labels"].map(vote)
    c, vc = (P[e] == truth.loc[valid]), (V[e] == truth.loc[valid])
    res[e] = {"n": len(valid), "vlm_correct": int(c.sum()), "vlm_macro_f1": float(f1_score(truth.loc[valid], P[e], labels=classes, average="macro", zero_division=0)),
              "same_class_mean": float(g["same_class_count"].mean()), "vote_correct": int(vc.sum()), "vote_macro_f1": float(f1_score(truth.loc[valid], V[e], labels=classes, average="macro", zero_division=0)),
              "vlm_equals_vote": int((P[e] == V[e]).sum()), "vlm_right_vote_wrong": int((c & ~vc).sum()), "vlm_wrong_vote_right": int((~c & vc).sum()),
              "unparsed": int((~P[e].isin(classes)).sum()),
              "vlm_acc_when_majority_same_class": [int(c[g["same_class_count"] >= 3].sum()), int((g["same_class_count"] >= 3).sum())],
              "vlm_acc_when_few_same_class": [int(c[g["same_class_count"] <= 1].sum()), int((g["same_class_count"] <= 1).sum())]}
pairs = {}
for x, y in itertools.combinations(encs, 2):
    cx, cy = (P[x] == truth.loc[valid]), (P[y] == truth.loc[valid])
    w, l = int((cy & ~cx).sum()), int((~cy & cx).sum())
    pairs[f"{y} vs {x}"] = {"wins": w, "losses": l, "p": 1.0 if w + l == 0 else float(stats.binomtest(w, w + l, 0.5).pvalue)}
json.dump({"model": cfg["model"], "dataset": cfg["dataset"], "requests": len(ledger), "failed": sum(not r["ok"] for r in ledger),
           "conditions": res, "pairs": pairs}, open(os.path.join(run, "metrics.json"), "w"), indent=2)
n = len(valid)
L = [f"# VLM pilot: does better retrieval help the VLM? ({cfg['dataset']})\n",
     f"{n} queries valid in all conditions (seed {cfg['seed']}), {k} shots, pool of {cfg['pool_size']} labeled images, VLM `{cfg['model']}` through `{cfg['vendor']}`. "
     f"Provider requests: {len(ledger)} of {cfg['max_requests']} allowed, {sum(not r['ok'] for r in ledger)} failed.\n",
     "| Method | " + " | ".join(SHORT.get(e, e) for e in encs) + " |", "|---|" + "---|" * len(encs),
     "| Embedding only (majority label of the 4 retrieved images, no VLM) | " + " | ".join(f"{res[e]['vote_correct']}/{n} ({100*res[e]['vote_correct']/n:.0f}%), F1 {res[e]['vote_macro_f1']:.3f}" for e in encs) + " |",
     "| VLM with the 4 retrieved demonstrations | " + " | ".join(f"{res[e]['vlm_correct']}/{n} ({100*res[e]['vlm_correct']/n:.0f}%), F1 {res[e]['vlm_macro_f1']:.3f}" for e in encs) + " |",
     f"| Mean same-class demonstrations of {k} | " + " | ".join(f"{res[e]['same_class_mean']:.2f}" for e in encs) + " |",
     "\n## Detail per condition\n",
     f"| Condition | Same-class in top {k} | VLM accuracy | VLM macro-F1 | Embedding-only accuracy | VLM = majority label | VLM right, vote wrong | VLM wrong, vote right |", "|---|---|---|---|---|---|---|---|"]
for e in encs:
    r = res[e]
    L.append(f"| {NAME.get(e, e)} | {r['same_class_mean']:.2f} | {r['vlm_correct']}/{n} ({100*r['vlm_correct']/n:.0f}%) | {r['vlm_macro_f1']:.3f} | "
             f"{r['vote_correct']}/{n} ({100*r['vote_correct']/n:.0f}%) | {r['vlm_equals_vote']}/{n} | {r['vlm_right_vote_wrong']} | {r['vlm_wrong_vote_right']} |")
L += ["\nEmbedding-only accuracy is the majority label of the same four demonstrations, with no VLM.\n", "## Paired VLM comparisons (exact McNemar)\n",
      "| Comparison | Wins | Losses | p |", "|---|---|---|---|"]
L += [f"| {NAME.get(y_, y_)} versus {NAME.get(x, x)} | {v['wins']} | {v['losses']} | {fp(v['p'])} |" for (x, y_), v in zip(itertools.combinations(encs, 2), pairs.values())]
L += ["\n## VLM accuracy by how many demonstrations share the query's class (all conditions pooled)\n", "| Same-class demonstrations | VLM correct |", "|---|---|"]
allg = ok.copy(); allg["c"] = allg["prediction"].astype(str) == allg["true_label"].astype(str)
for s_, g in allg.groupby("same_class_count"):
    L.append(f"| {s_} of {k} | {int(g['c'].sum())}/{len(g)} ({100*g['c'].mean():.0f}%) |")
notes = os.path.join(run, "NOTES.md")
if os.path.exists(notes): L.append("\n" + open(notes).read())
open(os.path.join(run, "SUMMARY.md"), "w").write("\n".join(L) + "\n"); print("\n".join(L))
