"""Replication on a second dataset and a second VLM, with one added prompt condition.

Dataset: Paddy Doctor (rice, 10 classes) with a group-disjoint split on source-group metadata.
Retriever: frozen DINOv2 block 12 CLS (get_embeddingp "dinov2_last"). Methods on identical queries:
  Z   = VLM zero-shot
  KNN = 4-NN vote of the retrieved references (no VLM)
  S   = VLM with the standard four-image prompt (top-4 references and their labels)
  C   = S plus one sentence that states the label counts of the four references
Models: claude-sonnet-5-5 and claude-opus-5-5 through headless `claude -p`.

Run through `python inference.py --pilot-replication --stage {freeze,offline,vlm,report} --out-dir <dir>`.
"""
import argparse, json, os, time
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.metrics import f1_score
from pilot_matched_labels import K, SEED, fp, md5_file, sha, vote, wjson

ROOT = os.path.join("data", "paddy-doctor")
MODELS = ["claude-sonnet-5-5", "claude-opus-5-5"]
Q_PER_CLASS, BANK_PER_CLASS, REPEAT_PER_CLASS = 5, 11, 2
NEAR_HAMMING = 8
CONDS = ["Z", "S", "C"]


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--pilot-replication", action="store_true")
    p.add_argument("--stage", choices=["freeze", "offline", "vlm", "report"], default="offline")
    p.add_argument("--out-dir", required=True)
    p.add_argument("--max-requests", type=int, default=400)
    return p.parse_args()


def img_path(label, image_id):
    return os.path.join(ROOT, "train_images", label, image_id)


def count_summary(labels):
    """One neutral sentence with the label counts of the retrieved references, most frequent first (ties: first appearance)."""
    order = sorted(dict.fromkeys(labels), key=lambda l: (-labels.count(l), labels.index(l)))
    return f"Label counts among the {len(labels)} labeled reference images above: " + ", ".join(f'"{l}": {labels.count(l)}' for l in order) + "."


def freeze(out):
    qp, bp = os.path.join(out, "query_manifest.csv"), os.path.join(out, "reference_manifest.csv")
    if os.path.exists(qp) and os.path.exists(bp):
        return pd.read_csv(qp, dtype=str), pd.read_csv(bp, dtype=str)
    from split_audit import dhash
    M = pd.read_csv(os.path.join(ROOT, "train.csv"), dtype=str).sort_values("image_id").reset_index(drop=True)
    M["group"] = M["variety"]                                  # source group: rice variety (one planting of one variety)
    classes = sorted(M["label"].unique())
    groups = sorted(M["group"].unique())
    rng = np.random.default_rng(SEED)
    ct = pd.crosstab(M["group"], M["label"])
    for attempt in range(1, 200001):                           # first seeded subset of query groups that leaves every class queries and a full bank
        qg = set(rng.choice(groups, size=rng.integers(2, len(groups) - 1), replace=False).tolist())
        fq, fb = ct.loc[sorted(qg)].sum(), ct.loc[[g for g in groups if g not in qg]].sum()
        if all(fq[c] >= Q_PER_CLASS and fb[c] >= BANK_PER_CLASS for c in classes):
            break
    else:
        raise SystemExit("BLOCKER: no group-disjoint split gives every class queries and a full bank")
    q_rows = []
    for c in classes:
        pool = M[M["group"].isin(qg) & (M["label"] == c)]
        pick = pool.iloc[rng.choice(len(pool), size=Q_PER_CLASS, replace=False)].copy()
        pick["repeat_subset"] = [True] * REPEAT_PER_CLASS + [False] * (Q_PER_CLASS - REPEAT_PER_CLASS)   # prespecified, in sampling order
        q_rows.append(pick)
    Q = pd.concat(q_rows).sort_values("image_id").reset_index(drop=True)
    cand = M[~M["group"].isin(qg)].copy()
    hq = np.stack([dhash(img_path(l, i)) for l, i in zip(Q["label"], Q["image_id"])])
    part, removed = [], 0
    for c in classes:
        pool = cand[cand["label"] == c]
        pick = pool.iloc[np.sort(rng.choice(len(pool), size=min(len(pool), 4 * BANK_PER_CLASS), replace=False))]   # oversample, then drop near-duplicates of queries
        hb = np.stack([dhash(img_path(l, i)) for l, i in zip(pick["label"], pick["image_id"])])
        ok = np.unpackbits(hb[:, None, :] ^ hq[None, :, :], axis=2).sum(axis=2).min(axis=1) > NEAR_HAMMING
        removed += int((~ok).sum())
        part.append(pick[ok].iloc[:BANK_PER_CLASS])
        assert len(part[-1]) == BANK_PER_CLASS
    R = pd.concat(part).sample(frac=1.0, random_state=SEED).reset_index(drop=True)
    qm = pd.DataFrame({"query_id": Q["image_id"], "label": Q["label"], "variety": Q["variety"], "age": Q["age"], "group": Q["group"],
                       "md5": [md5_file(img_path(l, i)) for l, i in zip(Q["label"], Q["image_id"])], "repeat_subset": Q["repeat_subset"]})
    rm = pd.DataFrame({"ref_idx": range(len(R)), "ref_id": R["image_id"], "label": R["label"], "variety": R["variety"], "age": R["age"], "group": R["group"],
                       "md5": [md5_file(img_path(l, i)) for l, i in zip(R["label"], R["image_id"])]})
    assert not set(qm["group"]) & set(rm["group"]) and not set(qm["md5"]) & set(rm["md5"]) and not set(qm["query_id"]) & set(rm["ref_id"])
    qm.to_csv(qp, index=False); rm.to_csv(bp, index=False)
    wjson(os.path.join(out, "split.json"), {
        "dataset": "Paddy Doctor (train.csv: image_id, label, variety, age)", "group": "rice variety", "seed": SEED, "attempts_until_feasible": attempt,
        "query_groups": sorted(qg), "bank_groups": sorted(set(groups) - qg), "images_per_group": M["group"].value_counts().to_dict(),
        "classes_per_group": (ct > 0).sum(axis=1).to_dict(), "groups_per_class": (ct > 0).sum(axis=0).to_dict(),
        "query_class_counts": qm["label"].value_counts().sort_index().to_dict(), "bank_class_counts": rm["label"].value_counts().sort_index().to_dict(),
        "query_groups_per_class": qm.groupby("label")["group"].nunique().to_dict(), "bank_groups_per_class": rm.groupby("label")["group"].nunique().to_dict(),
        "duplicate_checks": {"byte_identical_query_bank_pairs": 0, "shared_groups": 0, "bank_candidates_dropped_as_near_duplicates": removed,
                             "rule": f"dHash Hamming distance <= {NEAR_HAMMING} to any query"},
        "repeat_subset": f"{REPEAT_PER_CLASS} queries per class, the first in sampling order", "classes": classes})
    return qm.astype(str), rm.astype(str)


async def main(inf):
    args = parse_args()
    out = args.out_dir
    os.makedirs(os.path.join(out, "vlm", "responses"), exist_ok=True)
    qm, rm = freeze(out)
    qm["repeat_subset"] = qm["repeat_subset"].astype(str) == "True"
    classes = sorted(rm["label"].unique())
    print(f"frozen: {len(qm)} queries, {len(rm)} bank images, {int(qm['repeat_subset'].sum())} in the repeat subset")
    if args.stage == "freeze":
        return
    assert all(md5_file(img_path(l, i)) == m for l, i, m in zip(qm["label"], qm["query_id"], qm["md5"]))
    assert all(md5_file(img_path(l, i)) == m for l, i, m in zip(rm["label"], rm["ref_id"], rm["md5"]))

    apis = {}
    for m in MODELS:
        a = inf.ClaudeHeadlessAPI(model=m, sandbox_dir=os.path.join(out, "vlm", "sandbox"))
        a.settings = "claude -p defaults (temperature not settable); system prompt: " + a.SYSTEM_PROMPT + "; command: " + " ".join(a.command()[:-1])
        apis[m] = a
    prompt = inf.universal_prompt.format(expected_classes=classes)
    inf.vision_prompt = prompt

    # ---- offline retrieval -------------------------------------------------------
    bank = pd.DataFrame({0: [img_path(l, i) for l, i in zip(rm["label"], rm["ref_id"])], 1: list(rm["label"])})
    epath = os.path.join(out, "embeddings_dinov2_last.npz")
    cache = dict(np.load(epath)) if os.path.exists(epath) else {}
    def emb(p, key):
        if key not in cache:
            e = inf.get_image_embedding(p, model_type="dinov2_last")
            assert not isinstance(e, dict), e
            cache[key] = np.asarray(e, dtype=np.float64)
        return cache[key]
    embeddings = {i: emb(bank.at[i, 0], "bank/" + rm["ref_id"][i]) for i in range(len(bank))}
    api0 = apis[MODELS[0]]
    rows, inputs = [], {}
    with open(os.path.join(out, "rankings_dinov2_last.jsonl"), "w") as fh:
        for f, t in zip(qm["query_id"], qm["label"]):
            r = inf.rank_candidates(-1, emb(img_path(t, f), "query/" + f), embeddings)
            fh.write(json.dumps({"query_id": f, "ranked": [[rm["ref_id"][i], round(s, 6)] for i, s in r]}) + "\n")
            top = [i for i, _ in r[:K]]
            assert len(set(top)) == K
            labs, sims = [rm["label"][i] for i in top], [s for _, s in r[:K]]
            ex, paths, cats, _ = inf.build_examples(api0, bank, top, None)
            assert cats == labs
            q64 = inf.load_image(img_path(t, f))
            summ = count_summary(labs)
            inputs[(f, "Z")] = {"image": q64, "examples": [], "prompt": prompt}
            inputs[(f, "S")] = {"image": q64, "examples": ex, "prompt": prompt}
            inputs[(f, "C")] = {"image": q64, "examples": ex + [{"type": "text", "text": summ}], "prompt": prompt}
            cs, cc = api0.content(inputs[(f, "S")]), api0.content(inputs[(f, "C")])
            assert len(cc) == len(cs) + 1 and cc[:2 * K + 1] == cs[:2 * K + 1] and cc[2 * K + 2:] == cs[2 * K + 1:] and cc[2 * K + 1]["text"] == summ
            counts = sorted(pd.Series(labs).value_counts().tolist(), reverse=True)
            rows.append({"query_id": f, "true_label": t, "repeat_subset": bool(qm.set_index("query_id").at[f, "repeat_subset"]), "refs": json.dumps([rm["ref_id"][i] for i in top]),
                         "ref_labels": json.dumps(labs), "ref_cosines": json.dumps([round(s, 6) for s in sims]), "count_summary": summ, "knn": vote(labs, sims),
                         "margin": counts[0] - (counts[1] if len(counts) > 1 else 0), "true_label_count": labs.count(t)})
    np.savez(epath, **cache)
    J = pd.DataFrame(rows)
    J.to_csv(os.path.join(out, "offline_retrieval.csv"), index=False)

    def key(model, f, cond, rep):
        d = {"vendor": "claude_headless", "model": model, "settings": apis[model].settings, "inputs": inputs[(f, cond)]}
        if rep:
            d["repeat"] = rep
        return sha(json.dumps(d, sort_keys=True).encode())
    plan = [(m, f, c, 0) for m in MODELS for f in J["query_id"] for c in CONDS] + \
           [(m, f, c, 1) for m in MODELS for f in J[J["repeat_subset"]]["query_id"] for c in ("S", "C")]
    prereg = os.path.join(out, "preregistration.json")
    if not os.path.exists(prereg):
        wjson(prereg, {
            "written_before_any_vlm_call": not os.path.exists(os.path.join(out, "vlm", "request_ledger.jsonl")), "written_at": time.strftime("%Y-%m-%d %H:%M:%S"),
            "models": MODELS, "conditions": {"Z": "zero-shot: [prompt, prompt, query image]", "S": "standard: [prompt, 4 x (image, label), prompt, query image]",
                                             "C": "S with one text block after the four pairs: " + count_summary(["<label a>", "<label a>", "<label b>", "<label c>"])},
            "primary": "per model, C versus S on the queries where the 4-NN vote is correct: number of correct votes overturned, with an exact McNemar test on those queries",
            "secondary": ["per model: accuracy and macro-F1 of Z, KNN, S, C on all queries valid in Z, S and C", "per model: wrong votes corrected by S and by C",
                          "per model: exact McNemar for C vs S, S vs Z, S vs KNN, C vs KNN on all valid queries"],
            "repeats": f"S and C are sent a second time for the prespecified repeat subset ({REPEAT_PER_CLASS} queries per class); the repeats estimate how often an identical request changes its answer "
                       "and are not used in the headline accuracy", "planned_requests": len(plan),
            "vote_tie_break": "most frequent label; ties by larger summed cosine, then alphabetical order", "repeat_queries": sorted(J[J["repeat_subset"]]["query_id"])})
    wjson(os.path.join(out, "run_config.json"), {"dataset": "Paddy Doctor", "seed": SEED, "shots": K, "n_queries": len(qm), "bank_size": len(rm), "expected_classes": classes,
                                                 "retriever": "facebook/dinov2-base hidden_states[12] CLS, final LayerNorm, L2", "models": MODELS,
                                                 "settings": {m: apis[m].settings for m in MODELS}, "prompt": prompt, "prompt_sha256": sha(prompt.encode()),
                                                 "max_requests": args.max_requests, "image_format": "inference.load_image: RGB JPEG quality 95, base64"})
    def redact(o):
        if isinstance(o, dict):
            return {k: (f"<base64 jpeg sha256:{sha(v.encode())[:16]}>" if k == "data" and isinstance(v, str) and len(v) > 200 else redact(v)) for k, v in o.items()}
        return [redact(v) for v in o] if isinstance(o, list) else o
    f0 = J["query_id"].iloc[0]
    for c in CONDS:
        wjson(os.path.join(out, f"request_example_{c}.json"), {"query_id": f0, "content": redact(api0.content(inputs[(f0, c)]))})
    print(json.dumps({"knn_correct": f"{int((J['knn'] == J['true_label']).sum())}/{len(J)}", "margin_counts": J["margin"].value_counts().sort_index().to_dict(),
                      "true_label_present": int((J["true_label_count"] > 0).sum()), "mean_true_label_count": float(J["true_label_count"].mean()), "planned_requests": len(plan),
                      "example_summary": J["count_summary"].iloc[0]}, indent=1))
    if args.stage == "offline":
        return

    ledger = os.path.join(out, "vlm", "request_ledger.jsonl")
    def resp(k):
        p = os.path.join(out, "vlm", "responses", k + ".json")
        return json.load(open(p)) if os.path.exists(p) else None
    if args.stage == "vlm":
        used = sum(1 for _ in open(ledger)) if os.path.exists(ledger) else 0
        order = [plan[i] for i in np.random.default_rng([SEED, 5]).permutation(len(plan))]      # models, queries and conditions interleaved
        wjson(os.path.join(out, "vlm", "run_order.json"), [{"n": i + 1, "model": m, "query_id": f, "condition": c, "repeat": r, "payload_sha256": key(m, f, c, r)} for i, (m, f, c, r) in enumerate(order)])
        for attempt in (1, 2):
            for m, f, c, r in order:
                k = key(m, f, c, r)
                if resp(k) is not None:
                    continue
                if used >= args.max_requests:
                    print(f"request cap reached ({used}/{args.max_requests}); {m} {f} {c} rep{r} not sent"); continue
                used += 1
                t0 = time.time()
                e = {"n": used, "t": t0, "model": m, "query_id": f, "condition": c, "repeat": r, "payload_sha256": k, "attempt": attempt}
                try:
                    raw = await apis[m].get_image_information(inputs[(f, c)])
                    wjson(os.path.join(out, "vlm", "responses", k + ".json"), {"raw": raw, "meta": apis[m].last_meta, "model": m, "query_id": f, "condition": c, "repeat": r})
                    e.update(ok=True, elapsed_s=round(time.time() - t0, 2), meta=apis[m].last_meta)
                except Exception as ex:
                    e.update(ok=False, elapsed_s=round(time.time() - t0, 2), error=str(ex)[:500])
                with open(ledger, "a") as fh:
                    fh.write(json.dumps(e) + "\n")
                if not e["ok"] or used % 25 == 0:
                    print(f"[{used}/{args.max_requests}] {m} {f} {c} rep{r}: {'ok' if e['ok'] else 'FAILED ' + e['error'][:100]}", flush=True)
        print(f"provider requests used: {used} of {args.max_requests}")

    # ---- report ---------------------------------------------------------------------
    led = [json.loads(l) for l in open(ledger)] if os.path.exists(ledger) else []
    def parse(k):
        r = resp(k)
        if r is None:
            return None, "missing"
        try:
            p = inf.extract_json(r["raw"])["prediction"]
        except Exception:
            p = None
        return p, ("valid" if p in classes else "invalid")
    recs = []
    for m in MODELS:
        for _, w in J.iterrows():
            d = {"model": m, **{k: w[k] for k in ("query_id", "true_label", "repeat_subset", "ref_labels", "knn", "margin", "true_label_count")}}
            for c in CONDS:
                d[f"pred_{c}"], d[f"status_{c}"] = parse(key(m, w["query_id"], c, 0))
            for c in ("S", "C"):
                d[f"pred_{c}_rep"], d[f"status_{c}_rep"] = parse(key(m, w["query_id"], c, 1)) if w["repeat_subset"] else (None, "not planned")
            recs.append(d)
    R = pd.DataFrame(recs)
    R["correct_KNN"] = R["knn"] == R["true_label"]
    for c in CONDS:
        R[f"correct_{c}"] = R[f"pred_{c}"] == R["true_label"]
    R["valid"] = (R[[f"status_{c}" for c in CONDS]] == "valid").all(axis=1)
    R.to_csv(os.path.join(out, "per_query_results.csv"), index=False)
    COL = {"Z": "pred_Z", "KNN": "knn", "S": "pred_S", "C": "pred_C"}
    NAME = {"Z": "Z. Zero-shot", "KNN": "DINOv2 4-NN vote (no VLM)", "S": "S. Standard four-image prompt", "C": "C. Four-image prompt with label counts"}

    def mcn(a, b):   # a versus b (boolean series)
        w, l = int((a & ~b).sum()), int((~a & b).sum())
        return {"wins": w, "losses": l, "p": 1.0 if w + l == 0 else float(stats.binomtest(w, w + l, 0.5).pvalue)}
    met = {"requests_sent": len(led), "requests_failed": sum(not r["ok"] for r in led), "max_requests": args.max_requests, "models": {}}
    for m in MODELS:
        g = R[(R["model"] == m) & R["valid"]]
        ml = [r for r in led if r["model"] == m and r.get("ok")]
        vc, vw = g[g["correct_KNN"]], g[~g["correct_KNN"]]
        rep = R[(R["model"] == m) & R["repeat_subset"]]
        d = {"model_ids_in_responses": sorted({x for r in ml for x in r["meta"]["model_ids"]}), "turns": sorted({r["meta"]["num_turns"] for r in ml}),
             "requests_ok": len(ml), "n_queries": int((R["model"] == m).sum()), "valid_in_Z_S_C": len(g),
             "invalid_or_missing": {c: int((R[R["model"] == m][f"status_{c}"] != "valid").sum()) for c in CONDS},
             "scores": {c: {"correct": int((g[COL[c]] == g["true_label"]).sum()), "n": len(g),
                            "macro_f1": float(f1_score(g["true_label"], g[COL[c]].astype(str), labels=classes, average="macro", zero_division=0)) if len(g) else float("nan")} for c in ("Z", "KNN", "S", "C")},
             "vote_correct": len(vc), "vote_wrong": len(vw),
             "overturned": {c: int((~vc[f"correct_{c}"]).sum()) for c in ("S", "C")}, "corrected": {c: int(vw[f"correct_{c}"].sum()) for c in ("S", "C")},
             "zero_shot_correct_when_vote_wrong": int(vw["correct_Z"].sum()), "equals_vote": {c: int((g[f"pred_{c}"] == g["knn"]).sum()) for c in ("S", "C")},
             "primary_C_vs_S_on_correct_votes": mcn(vc["correct_C"], vc["correct_S"]), "C_vs_S_on_wrong_votes": mcn(vw["correct_C"], vw["correct_S"]),
             "C_vs_S": mcn(g["correct_C"], g["correct_S"]), "S_vs_Z": mcn(g["correct_S"], g["correct_Z"]), "C_vs_Z": mcn(g["correct_C"], g["correct_Z"]),
             "S_vs_KNN": mcn(g["correct_S"], g["correct_KNN"]), "C_vs_KNN": mcn(g["correct_C"], g["correct_KNN"]), "Z_vs_KNN": mcn(g["correct_Z"], g["correct_KNN"]),
             "prediction_changed_C_vs_S": int((g["pred_C"] != g["pred_S"]).sum()),
             "repeats": {c: {"n": int(((rep[f"status_{c}"] == "valid") & (rep[f"status_{c}_rep"] == "valid")).sum()),
                             "same_answer": int(((rep[f"pred_{c}"] == rep[f"pred_{c}_rep"]) & (rep[f"status_{c}"] == "valid") & (rep[f"status_{c}_rep"] == "valid")).sum()),
                             "correctness_changed": int((((rep[f"pred_{c}"] == rep["true_label"]) != (rep[f"pred_{c}_rep"] == rep["true_label"])) & (rep[f"status_{c}"] == "valid") & (rep[f"status_{c}_rep"] == "valid")).sum())}
                         for c in ("S", "C")},
             "by_margin": [{"margin": int(k), "n": len(s), "KNN": int(s["correct_KNN"].sum()), "Z": int(s["correct_Z"].sum()), "S": int(s["correct_S"].sum()), "C": int(s["correct_C"].sum())} for k, s in g.groupby("margin")]}
        met["models"][m] = d
    wjson(os.path.join(out, "metrics.json"), met)

    sp = json.load(open(os.path.join(out, "split.json")))
    n = len(J)
    L = ["# Paddy Doctor replication: zero-shot, 4-NN, four-image prompt, and four-image prompt with label counts\n",
         f"{n} queries ({Q_PER_CLASS} per class, seed {SEED}) from rice varieties {', '.join(sp['query_groups'])}. Reference bank: {len(rm)} images ({BANK_PER_CLASS} per class) from the other varieties. "
         f"Retriever: DINOv2 block 12 CLS, top {K}. Models through headless `claude -p`: {', '.join('`' + m + '`' for m in MODELS)}. "
         f"Requests sent: {met['requests_sent']} of {met['max_requests']} allowed, {met['requests_failed']} failed.\n",
         f"The 4-NN vote is the same for both models: correct on {int(J['knn'].eq(J['true_label']).sum())} of {n} queries. The true label appears among the four references on {int((J['true_label_count'] > 0).sum())} of {n} queries.\n"]
    for m in MODELS:
        d = met["models"][m]
        if d["valid_in_Z_S_C"] == 0:
            L.append(f"## {m}\n\nNo complete queries yet.\n"); continue
        nv = d["valid_in_Z_S_C"]
        L += [f"## {m}\n", f"Valid in Z, S and C: {nv} of {d['n_queries']} queries (invalid or missing: " + ", ".join(f"{c} {d['invalid_or_missing'][c]}" for c in CONDS) + f"). Response model IDs: {', '.join(d['model_ids_in_responses'])}.\n",
              "| Method | Correct | Accuracy | Macro-F1 (10 classes) | Wrong votes corrected | Correct votes overturned | Same label as the vote |", "|---|---|---|---|---|---|---|"]
        for c in ("Z", "KNN", "S", "C"):
            s = d["scores"][c]
            extra = (f"{d['corrected'][c]}/{d['vote_wrong']} | {d['overturned'][c]}/{d['vote_correct']} | {d['equals_vote'][c]}/{nv}" if c in ("S", "C")
                     else (f"{d['zero_shot_correct_when_vote_wrong']}/{d['vote_wrong']} | {d['vote_correct'] - int(R[(R['model'] == m) & R['valid'] & R['correct_KNN']]['correct_Z'].sum())}/{d['vote_correct']} | " if c == "Z" else " |  | "))
            L.append(f"| {NAME[c]} | {s['correct']}/{s['n']} | {100 * s['correct'] / s['n']:.0f}% | {s['macro_f1']:.3f} | {extra} |")
        L += ["\n| Paired comparison | Wins | Losses | Exact McNemar p |", "|---|---|---|---|"]
        for k_, nm in (("primary_C_vs_S_on_correct_votes", f"C versus S, queries with a correct vote (primary, n = {d['vote_correct']})"), ("C_vs_S_on_wrong_votes", f"C versus S, queries with a wrong vote (n = {d['vote_wrong']})"),
                       ("C_vs_S", "C versus S, all queries"), ("S_vs_Z", "S versus Z"), ("C_vs_Z", "C versus Z"), ("S_vs_KNN", "S versus 4-NN vote"), ("C_vs_KNN", "C versus 4-NN vote"), ("Z_vs_KNN", "Z versus 4-NN vote")):
            x = d[k_]
            L.append(f"| {nm} | {x['wins']} | {x['losses']} | {fp(x['p'])} |")
        L += [f"\nC and S gave different predictions on {d['prediction_changed_C_vs_S']} of {nv} queries.\n",
              "| Vote margin | n | 4-NN correct | Z correct | S correct | C correct |", "|---|---|---|---|---|---|"]
        L += [f"| {b['margin']} | {b['n']} | {b['KNN']} | {b['Z']} | {b['S']} | {b['C']} |" for b in d["by_margin"]]
        rp = d["repeats"]
        L.append(f"\n**Repeat subset.** An identical S request gave the same answer on {rp['S']['same_answer']} of {rp['S']['n']} repeats and changed correctness on {rp['S']['correctness_changed']}. "
                 f"An identical C request gave the same answer on {rp['C']['same_answer']} of {rp['C']['n']} repeats and changed correctness on {rp['C']['correctness_changed']}.\n")
    notes = os.path.join(out, "NOTES.md")
    if os.path.exists(notes):
        L.append(open(notes).read())
    open(os.path.join(out, "SUMMARY.md"), "w").write("\n".join(L) + "\n")
    print("\n".join(L))
