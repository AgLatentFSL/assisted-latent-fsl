"""Metrics and Markdown summary for the retrieval-encoder pilot. Usage: python pilot_report.py <run_dir>"""
import json, os, sys
import pandas as pd
from sklearn.metrics import f1_score

run = sys.argv[1]
cfg = json.load(open(os.path.join(run, "run_config.json")))
classes, k = cfg["expected_classes"], cfg["shots"]
top = pd.read_csv(os.path.join(run, "retrieval", f"top{k}.csv"))
pred = pd.read_csv(os.path.join(run, "vlm", "predictions.csv"), keep_default_na=False)
ledger = [json.loads(l) for l in open(os.path.join(run, "vlm", "request_ledger.jsonl"))]
names = {"vit": "A. ViT pooler_output (original)", "dinov2_last": "B. DINOv2 block 12 CLS", "dinov2_mid": "C. DINOv2 block 6 CLS"}
queries = sorted(top["query_idx"].unique())

# retrieval
sets = {(e, q): list(g.sort_values("rank")["ref_idx"]) for (e, q), g in top.groupby(["encoder", "query_idx"])}
same = top.groupby(["encoder", "query_idx"])["same_class"].sum().unstack(0)
def overlap(a, b):
    rows = []
    for q in queries:
        x, y = sets[(a, q)], sets[(b, q)]
        rows.append({"query_idx": q, "shared": len(set(x) & set(y)), "set_changed": set(x) != set(y), "order_changed": x != y})
    return pd.DataFrame(rows)
bc, ab, ac = overlap("dinov2_last", "dinov2_mid"), overlap("vit", "dinov2_last"), overlap("vit", "dinov2_mid")

# predictions: only queries valid in every condition
ok = pred[pred["status"] == "ok"]
valid = sorted(q for q in queries if all(((ok["query_idx"] == q) & (ok["encoder"] == e)).any() for e in cfg["encoders"]))
P = ok[ok["query_idx"].isin(valid)].pivot(index="query_idx", columns="encoder", values="prediction")
truth = pred.drop_duplicates("query_idx").set_index("query_idx")["true_label"].loc[valid]
metrics = {}
for e in cfg["encoders"]:
    c = int((P[e] == truth).sum())
    metrics[e] = {"correct": c, "n": len(valid), "accuracy": c / len(valid),
                  "macro_f1": float(f1_score(truth, P[e], labels=classes, average="macro", zero_division=0)),
                  "unparsed": int((~P[e].isin(classes)).sum()), "mean_same_class_top_k": float(same[e].mean())}
def wl(d):
    return f"{d['wins']} win{'s' if d['wins'] != 1 else ''}, {d['losses']} loss{'es' if d['losses'] != 1 else ''}"
def paired(x, y):  # y versus x
    cx, cy = P[x] == truth, P[y] == truth
    return {"wins": int((cy & ~cx).sum()), "losses": int((~cy & cx).sum()), "both_correct": int((cy & cx).sum()),
            "both_wrong": int((~cy & ~cx).sum()), "prediction_changed": int((P[x] != P[y]).sum())}
out = {"model": cfg["model"], "vendor": cfg["vendor"], "label_set": classes, "queries_frozen": len(queries), "queries_valid_all_conditions": len(valid),
       "provider_requests": len(ledger), "failed_requests": sum(not r["ok"] for r in ledger), "max_requests": cfg["max_requests"],
       "model_ids_returned": sorted({m for r in ledger for m in (r.get("meta") or {}).get("model_ids", [])}),
       "max_turns_per_request": max((r.get("meta") or {}).get("num_turns") or 0 for r in ledger),
       "conditions": metrics, "C_vs_B": paired("dinov2_last", "dinov2_mid"), "C_vs_A": paired("vit", "dinov2_mid"), "B_vs_A": paired("vit", "dinov2_last"),
       "retrieval_B_vs_C": {"set_changed": int(bc["set_changed"].sum()), "order_or_set_changed": int(bc["order_changed"].sum()),
                            "mean_shared_of_k": float(bc["shared"].mean()), "n": len(queries)},
       "retrieval_A_vs_B_mean_shared": float(ab["shared"].mean()), "retrieval_A_vs_C_mean_shared": float(ac["shared"].mean())}
json.dump(out, open(os.path.join(run, "metrics.json"), "w"), indent=2)

per = pd.DataFrame({"true": truth})
for e in cfg["encoders"]:
    per[f"same_class_{e}"] = same[e].loc[valid]
    per[f"pred_{e}"] = P[e]
per["BC_shared"] = bc.set_index("query_idx")["shared"].loc[valid]
per.to_csv(os.path.join(run, "per_query.csv"))

# example: a query whose selected set changed between B and C, preferring one whose prediction changed
cand = [q for q in valid if set(sets[("dinov2_last", q)]) != set(sets[("dinov2_mid", q)])]
cand.sort(key=lambda q: (P.loc[q, "dinov2_last"] == P.loc[q, "dinov2_mid"], q))
ex = cand[0] if cand else None

L = []
L.append(f"# AgEval pilot: DINOv2 layer choice for few-shot demonstration retrieval\n")
L.append(f"Task {cfg['dataset']}, {len(queries)} queries (seed {cfg['seed']}), {k} shots, pool of {cfg['pool_size']} labeled images, "
         f"VLM `{cfg['model']}` through `{cfg['vendor']}`. Provider requests: {len(ledger)} of {cfg['max_requests']} allowed, "
         f"{out['failed_requests']} failed. Queries valid in all three conditions: {len(valid)} of {len(queries)}.\n")
L.append("## Result\n")
L.append(f"| Condition | Accuracy | Macro-F1 | Mean same-class in top {k} |\n|---|---|---|---|")
for e in cfg["encoders"]:
    m = metrics[e]
    L.append(f"| {names[e]} | {m['correct']}/{m['n']} ({100*m['accuracy']:.0f}%) | {m['macro_f1']:.3f} | {m['mean_same_class_top_k']:.2f} |")
c = out["C_vs_B"]
L.append(f"\nMacro-F1 uses the fixed label set {classes}. Unparsed answers count as errors "
         f"({sum(m['unparsed'] for m in metrics.values())} occurred).\n")
L.append(f"**C versus B (primary, encoder fixed):** {wl(c)}, {c['both_correct']} both correct, "
         f"{c['both_wrong']} both wrong. The prediction differed on {c['prediction_changed']} of {len(valid)} queries.\n")
a = out["C_vs_A"]
L.append(f"**C versus A (encoder also changes):** {wl(a)}, {a['both_correct']} both correct, {a['both_wrong']} both wrong.\n")
b = out["B_vs_A"]
L.append(f"**B versus A:** {wl(b)}.\n")
r = out["retrieval_B_vs_C"]
L.append("## Retrieval\n")
L.append(f"The selected set of {k} changed between B and C on {r['set_changed']} of {r['n']} queries, and the set or its order changed on "
         f"{r['order_or_set_changed']} of {r['n']}. B and C shared {r['mean_shared_of_k']:.2f} of {k} demonstrations on average. "
         f"A shared {out['retrieval_A_vs_B_mean_shared']:.2f} with B and {out['retrieval_A_vs_C_mean_shared']:.2f} with C.\n")
if ex is not None:
    L.append(f"## Example where the layer changed the selection (query idx {ex})\n")
    qrow = top[top["query_idx"] == ex].iloc[0]
    L.append(f"Query `{qrow['query_id']}`, true label {qrow['query_label']}.\n")
    L.append("| Rank | B (block 12) | cosine | C (block 6) | cosine |\n|---|---|---|---|---|")
    gb = top[(top["query_idx"] == ex) & (top["encoder"] == "dinov2_last")].sort_values("rank")
    gc = top[(top["query_idx"] == ex) & (top["encoder"] == "dinov2_mid")].sort_values("rank")
    for (_, x), (_, y) in zip(gb.iterrows(), gc.iterrows()):
        L.append(f"| {x['rank']} | {os.path.basename(x['ref_id'])} ({x['ref_label']}) | {x['cosine']:.3f} | {os.path.basename(y['ref_id'])} ({y['ref_label']}) | {y['cosine']:.3f} |")
    pb, pc = P.loc[ex, "dinov2_last"], P.loc[ex, "dinov2_mid"]
    L.append(f"\nPrediction with B: {pb}. Prediction with C: {pc}. The final prediction {'changed' if pb != pc else 'did not change'}.\n")
L.append("## Per-query table\n")
L.append("| idx | True | A same | A pred | B same | B pred | C same | C pred | B∩C |\n|---|---|---|---|---|---|---|---|---|")
for q, row in per.iterrows():
    L.append(f"| {q} | {row['true']} | {row['same_class_vit']} | {row['pred_vit']} | {row['same_class_dinov2_last']} | {row['pred_dinov2_last']} | "
             f"{row['same_class_dinov2_mid']} | {row['pred_dinov2_mid']} | {row['BC_shared']} |")
notes = os.path.join(run, "NOTES.md")
if os.path.exists(notes):
    L.append("\n" + open(notes).read())
open(os.path.join(run, "SUMMARY.md"), "w").write("\n".join(L))
print("\n".join(L))
