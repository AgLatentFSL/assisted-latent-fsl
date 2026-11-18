"""Matched-label pilot on DeepWeeds, run through `python inference.py --pilot-matched ...`.

Question: do the original AgEval ViT and final-layer DINOv2 select images of different
usefulness to the VLM when the demonstration labels and their order are identical?

  A = original ViT (pooler_output) top-4 references.
  B = for each slot, DINOv2 block 12's highest-ranked unused reference with the label of A's slot.
  Z = zero-shot: same query, instructions and output format, no demonstrations.

Stages: freeze -> offline -> vlm -> report. Each stage runs the earlier ones (idempotent).
The query's ground-truth label is used for stratified sampling and scoring only.
"""
import argparse
import base64
import hashlib
import io
import json
import os
import time

import numpy as np
import pandas as pd
from PIL import Image
from scipy import stats
from sklearn.metrics import f1_score

DATA = os.path.join("data", "deepweeds")
PRIOR = os.path.join("runs", "deepweeds_seed42")
ENC = {"A": "vit", "B": "dinov2_last"}
K, SEED, N_QUERIES, BANK_PER_CLASS, Q_DATE_FRACTION = 4, 42, 20, 11, 0.3


def sha(b):
    return hashlib.sha256(b).hexdigest()


def md5_file(p):
    return hashlib.md5(open(p, "rb").read()).hexdigest()


def path_of(fn):
    return os.path.join(DATA, "images", fn)


def wjson(path, obj):
    with open(path, "w") as f:
        json.dump(obj, f, indent=2, sort_keys=True)


def frozen_csv(path, df):
    """Write once; later runs must reproduce the same frame."""
    if os.path.exists(path):
        old = pd.read_csv(path, dtype=str)
        assert old.equals(df.astype(str).reset_index(drop=True)), f"{path} differs from the frozen manifest"
    else:
        df.to_csv(path, index=False)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--pilot-matched", action="store_true")
    p.add_argument("--stage", choices=["freeze", "offline", "vlm", "report"], default="offline")
    p.add_argument("--model", default="claude-sonnet-5-5")
    p.add_argument("--max-requests", type=int, default=60)
    p.add_argument("--out-dir", required=True)
    return p.parse_args()


# ----------------------------------------------------------------- 1. freeze
def freeze(out):
    L = pd.read_csv(os.path.join(DATA, "labels", "labels.csv")).sort_values("Filename").reset_index(drop=True)
    L["date"] = L["Filename"].str[:8]
    classes = sorted(L["Species"].unique())

    # prior pilot: recover and verify the 54 query IDs and the 99-image pool
    pq = pd.read_csv(os.path.join(PRIOR, "query_manifest.csv"))
    pr = pd.read_csv(os.path.join(PRIOR, "reference_manifest.csv"))
    pp = pd.read_csv(os.path.join(PRIOR, "vlm", "predictions.csv"), keep_default_na=False)
    pr["Filename"] = pr["image_id"].str.split("/").str[-1]
    pq["Filename"] = pq["image_id"].str.split("/").str[-1]
    lab = L.set_index("Filename")["Species"]
    prior_check = {
        "prior_queries": len(pq), "prior_pool": len(pr),
        "query_ids_match_predictions": sorted(pq["image_id"]) == sorted(pp["query_id"].unique()),
        "query_files_exist": bool(all(os.path.exists(path_of(f)) for f in pq["Filename"])),
        "pool_md5_match": bool(all(md5_file(path_of(f)) == m for f, m in zip(pr["Filename"], pr["md5"]))),
        "query_md5_match": bool(all(md5_file(path_of(f)) == pr.set_index("Filename")["md5"][f] for f in pq["Filename"])),
        "labels_match_labels_csv": bool(all(lab[f] == l for f, l in zip(pr["Filename"], pr["label"]))),
    }
    assert all(v for k, v in prior_check.items() if isinstance(v, bool)), prior_check
    prior_pool = set(pr["Filename"])

    L["md5"] = [md5_file(path_of(f)) for f in L["Filename"]]
    fresh = L[~L["Filename"].isin(prior_pool)]  # never shown to the VLM in the prior pilot, as query or demonstration

    # capture date is the source group: a date holds one site visit (about 1.8 species per date).
    # Search, with one seeded generator, for the first query-date subset under which every class
    # has fresh query images on query dates and a full bank on the remaining dates.
    dates = sorted(L["date"].unique())
    rng = np.random.default_rng(SEED)
    n_q = round(Q_DATE_FRACTION * len(dates))
    for attempt in range(1, 100001):
        q_dates = set(rng.choice(dates, size=n_q, replace=False).tolist())
        fq = fresh[fresh["date"].isin(q_dates)]["Species"].value_counts()
        rb = L[~L["date"].isin(q_dates)]["Species"].value_counts()
        if all(fq.get(c, 0) >= 3 and rb.get(c, 0) >= BANK_PER_CLASS for c in classes):
            break
    else:
        raise SystemExit("BLOCKER: no date-disjoint split gives every class fresh queries and a full bank")

    per_class = {c: N_QUERIES // len(classes) for c in classes}
    for c in rng.choice(classes, size=N_QUERIES - sum(per_class.values()), replace=False):
        per_class[c] += 1
    q_rows = []
    for c in classes:
        pool = fresh[(fresh["date"].isin(q_dates)) & (fresh["Species"] == c)].sort_values("Filename")
        q_rows.append(pool.iloc[np.sort(rng.choice(len(pool), size=per_class[c], replace=False))])
    Q = pd.concat(q_rows).reset_index(drop=True)
    q_md5 = set(Q["md5"])

    b_rows = []
    for c in classes:
        pool = L[(~L["date"].isin(q_dates)) & (L["Species"] == c) & (~L["md5"].isin(q_md5))].sort_values("Filename")
        b_rows.append(pool.iloc[np.sort(rng.choice(len(pool), size=BANK_PER_CLASS, replace=False))])
    R = pd.concat(b_rows).sample(frac=1.0, random_state=SEED).reset_index(drop=True)  # class order must not follow index order

    qm = pd.DataFrame({"query_id": Q["Filename"], "label": Q["Species"], "capture_date": Q["date"], "md5": Q["md5"]})
    rm = pd.DataFrame({"ref_idx": range(len(R)), "ref_id": R["Filename"], "label": R["Species"], "capture_date": R["date"], "md5": R["md5"],
                       "in_prior_pool": R["Filename"].isin(prior_pool), "prior_query": R["Filename"].isin(set(pq["Filename"]))})
    assert not set(qm["query_id"]) & set(rm["ref_id"]) and not set(qm["md5"]) & set(rm["md5"])
    assert not set(qm["capture_date"]) & set(rm["capture_date"]) and not set(qm["query_id"]) & prior_pool
    frozen_csv(os.path.join(out, "query_manifest.csv"), qm)
    frozen_csv(os.path.join(out, "reference_manifest.csv"), rm)
    wjson(os.path.join(out, "split.json"), {
        "rule": "capture date (YYYYMMDD in the file name) is the source group; query dates and bank dates are disjoint",
        "official_splits": "labels/{train,val,test}_subset*.csv are random folds; all 60 dates appear in train and test, so they do not separate source groups",
        "seed": SEED, "attempts_until_feasible": attempt, "query_dates": sorted(q_dates), "n_dates": len(dates),
        "fresh_query_rule": "not in the prior 99-image pool (which contains the prior 54 queries); capture date in query_dates",
        "bank_rule": f"{BANK_PER_CLASS} per class from images on non-query dates, not byte-identical to any query; prior-pilot images allowed",
        "query_class_counts": qm["label"].value_counts().sort_index().to_dict(),
        "bank_class_counts": rm["label"].value_counts().sort_index().to_dict(),
        "bank_images_from_prior_pool": int(rm["in_prior_pool"].sum()), "bank_images_that_were_prior_queries": int(rm["prior_query"].sum()),
        "duplicate_checks": {"byte_identical_query_bank_pairs": 0, "byte_identical_groups_in_dataset": int((L["md5"].value_counts() > 1).sum()),
                             "shared_capture_dates": 0, "queries_in_prior_pool": 0},
        "prior_pilot_check": prior_check, "classes": classes})
    return qm, rm, classes


# ---------------------------------------------------------------- 2. offline
def vote(labels, sims):
    """4-NN prediction: most frequent label; ties by larger summed cosine, then alphabetical order."""
    cnt, tot = {}, {}
    for l, s in zip(labels, sims):
        cnt[l] = cnt.get(l, 0) + 1
        tot[l] = tot.get(l, 0.0) + s
    return sorted(cnt, key=lambda c: (-cnt[c], -tot[c], c))[0]


def offline(inf, api, out, qm, rm, classes):
    odir = os.path.join(out, "offline")
    os.makedirs(odir, exist_ok=True)
    bank = pd.DataFrame({0: [path_of(f) for f in rm["ref_id"]], 1: list(rm["label"])})
    rlab = list(rm["label"])
    expected = ['Chinee apple', 'Lantana', 'Negative', 'Snake weed', 'Siam weed', 'Prickly acacia', 'Parthenium', 'Rubber vine', 'Parkinsonia']  # order in data_loader.py
    assert sorted(expected) == classes
    prompt = inf.universal_prompt.format(expected_classes=expected)
    inf.vision_prompt = prompt

    rank, emb_meta = {}, {}
    for cond, enc in ENC.items():
        embeddings = inf.precompute_embeddings(bank, enc)  # the repository path: idx -> vector
        assert len(embeddings) == len(bank)
        M = np.array([embeddings[i] for i in range(len(bank))], dtype=np.float64)
        np.save(os.path.join(odir, f"bank_embeddings_{enc}.npy"), M.astype(np.float32))
        qv = []
        for f in qm["query_id"]:
            e = inf.get_image_embedding(path_of(f), model_type=enc)
            assert not isinstance(e, dict), e
            qv.append(e)
            r = inf.rank_candidates(-1, e, embeddings)  # query is not in the bank, so nothing is excluded
            assert len(r) == len(bank) and all(np.isfinite(s) and -1 - 1e-6 <= s <= 1 + 1e-6 for _, s in r)
            rank[(enc, f)] = r
        np.save(os.path.join(odir, f"query_embeddings_{enc}.npy"), np.array(qv, dtype=np.float32))
        emb_meta[enc] = {"dim": int(M.shape[1]), "finite": bool(np.isfinite(M).all()), "bank_norm_min": float(np.linalg.norm(M, axis=1).min()),
                         "bank_norm_max": float(np.linalg.norm(M, axis=1).max()),
                         "max_query_bank_cosine": float(max(rank[(enc, f)][0][1] for f in qm["query_id"]))}
        with open(os.path.join(odir, f"rankings_{enc}.jsonl"), "w") as fh:
            for f in qm["query_id"]:
                fh.write(json.dumps({"query_id": f, "ranked": [[rm["ref_id"][i], round(s, 6)] for i, s in rank[(enc, f)]]}) + "\n")

    rows, slots, requests = [], [], {}
    for f, ql in zip(qm["query_id"], qm["label"]):
        rv, rd = rank[("vit", f)], rank[("dinov2_last", f)]
        pos_d = {i: p for p, (i, _) in enumerate(rd, 1)}
        sim_d, sim_v = dict(rd), dict(rv)
        a_idx = [i for i, _ in rv[:K]]
        seq = [rlab[i] for i in a_idx]                      # label sequence supplied by the original ViT
        b_idx = []
        for s in range(K):                                   # the query label is not consulted here
            b_idx.append(next(i for i, _ in rd if rlab[i] == seq[s] and i not in b_idx))
        assert [rlab[i] for i in b_idx] == seq and len(set(a_idx)) == K and len(set(b_idx)) == K
        ud = [i for i, _ in rd[:K]]
        knn = {"vit": vote(seq, [sim_v[i] for i in a_idx]), "dinov2_last": vote([rlab[i] for i in ud], [sim_d[i] for i in ud])}
        q64 = inf.load_image(path_of(f))
        inputs = {}
        for cond, idx in (("A", a_idx), ("B", b_idx)):
            ex, paths, cats, _ = inf.build_examples(api, bank, idx, ql)
            assert cats == seq and paths == [bank.at[i, 0] for i in idx]
            inputs[cond] = {"image": q64, "examples": ex, "prompt": prompt}
        inputs["Z"] = {"image": q64, "examples": [], "prompt": prompt}
        ca, cb = api.content(inputs["A"]), api.content(inputs["B"])
        assert len(ca) == len(cb) == 2 * K + 3 and [b["type"] for b in ca] == [b["type"] for b in cb]
        assert all(x == y for x, y in zip(ca, cb) if x["type"] == "text") and ca[-1] == cb[-1]
        for cond in "ABZ":
            key = sha(json.dumps({"vendor": "claude_headless", "model": api.model, "settings": api.settings, "inputs": inputs[cond]}, sort_keys=True).encode())
            requests[(f, cond)] = (key, inputs[cond])
        diff = sum(a != b for a, b in zip(a_idx, b_idx))
        rows.append({"query_id": f, "true_label": ql, "demo_labels": json.dumps(seq), "A_refs": json.dumps([rm["ref_id"][i] for i in a_idx]),
                     "B_refs": json.dumps([rm["ref_id"][i] for i in b_idx]), "differing_slots": diff,
                     "identical_payload": requests[(f, "A")][0] == requests[(f, "B")][0], "same_class_in_demo_labels": sum(l == ql for l in seq),
                     "A_mean_vit_cosine": float(np.mean([sim_v[i] for i in a_idx])), "B_mean_dinov2_cosine": float(np.mean([sim_d[i] for i in b_idx])),
                     "B_dinov2_ranks": json.dumps([pos_d[i] for i in b_idx]), "A_refs_dinov2_ranks": json.dumps([pos_d[i] for i in a_idx]),
                     "unconstrained_vit_top4": json.dumps([rm["ref_id"][i] for i in a_idx]), "unconstrained_vit_labels": json.dumps(seq),
                     "unconstrained_dinov2_top4": json.dumps([rm["ref_id"][i] for i in ud]), "unconstrained_dinov2_labels": json.dumps([rlab[i] for i in ud]),
                     "knn4_vit": knn["vit"], "knn4_dinov2": knn["dinov2_last"],
                     "payload_A": requests[(f, "A")][0], "payload_B": requests[(f, "B")][0], "payload_Z": requests[(f, "Z")][0]})
        assert (diff == 0) == rows[-1]["identical_payload"]
        for s in range(K):
            slots.append({"query_id": f, "slot": s + 1, "label": seq[s], "A_ref": rm["ref_id"][a_idx[s]], "A_vit_cosine": round(sim_v[a_idx[s]], 6),
                          "B_ref": rm["ref_id"][b_idx[s]], "B_dinov2_cosine": round(sim_d[b_idx[s]], 6), "B_dinov2_rank": pos_d[b_idx[s]],
                          "same_image": a_idx[s] == b_idx[s]})
    matched = pd.DataFrame(rows)
    matched.to_csv(os.path.join(odir, "matched_pairs.csv"), index=False)
    pd.DataFrame(slots).to_csv(os.path.join(odir, "matched_slots.csv"), index=False)

    # image delivery: every image block is the repository's JPEG re-encoding of the intended file, decodes to that
    # picture, and carries no file name, path or metadata
    delivery = {"blocks_checked": 0, "max_mean_abs_pixel_diff": 0.0, "metadata_free": True, "text_mentions_file_names": False}
    ids = set(qm["query_id"]) | set(rm["ref_id"])
    for (f, cond), (key, inp) in requests.items():
        content = api.content(inp)
        want = ([] if cond == "Z" else json.loads(matched.set_index("query_id").at[f, f"{cond}_refs"])) + [f]
        imgs = [b for b in content if b["type"] == "image"]
        assert len(imgs) == len(want)
        for b, fn in zip(imgs, want):
            raw = base64.b64decode(b["source"]["data"])
            im = Image.open(io.BytesIO(raw))
            src = Image.open(path_of(fn)).convert("RGB")
            assert im.size == src.size and b["source"]["data"] == inf.load_image(path_of(fn))
            d = float(np.abs(np.asarray(im.convert("RGB"), dtype=float) - np.asarray(src, dtype=float)).mean())
            delivery["max_mean_abs_pixel_diff"] = max(delivery["max_mean_abs_pixel_diff"], d)
            delivery["metadata_free"] &= (b"Exif" not in raw) and (fn.encode() not in raw) and not im.info.get("comment")
            delivery["blocks_checked"] += 1
        text = " ".join(b["text"] for b in content if b["type"] == "text")
        delivery["text_mentions_file_names"] |= any(i in text or i[:-4] in text for i in ids)
    assert delivery["max_mean_abs_pixel_diff"] < 4 and delivery["metadata_free"] and not delivery["text_mentions_file_names"]

    def redact(o):
        if isinstance(o, dict):
            return {k: (f"<base64 jpeg sha256:{sha(v.encode())[:16]}>" if k == "data" and isinstance(v, str) and len(v) > 200 else redact(v)) for k, v in o.items()}
        return [redact(v) for v in o] if isinstance(o, list) else o
    changed = matched[~matched["identical_payload"]]
    sample_q = (changed if len(changed) else matched)["query_id"].iloc[0]
    for cond in "ABZ":
        wjson(os.path.join(odir, f"request_example_{cond}.json"), {"query_id": sample_q, "command": api.command(), "payload_sha256": requests[(sample_q, cond)][0],
                                                                    "content": redact(api.content(requests[(sample_q, cond)][1]))})
    summary = {"queries": len(matched), "identical_AB_payloads": int(matched["identical_payload"].sum()),
               "changed_AB_payloads": int((~matched["identical_payload"]).sum()), "mean_differing_slots": float(matched["differing_slots"].mean()),
               "differing_slots_histogram": matched["differing_slots"].value_counts().sort_index().to_dict(),
               "mean_same_class_in_vit_labels": float(matched["same_class_in_demo_labels"].mean()),
               "unconstrained_dinov2_mean_same_class": float(np.mean([sum(l == t for l in json.loads(x)) for x, t in zip(matched["unconstrained_dinov2_labels"], matched["true_label"])])),
               "embeddings": emb_meta, "image_delivery": delivery, "prompt_sha256": sha(prompt.encode()),
               "vote_tie_break": "most frequent label among the top 4; ties by larger summed cosine, then alphabetical label order",
               "unique_planned_requests": len({k for k, _ in requests.values()})}
    wjson(os.path.join(odir, "offline_summary.json"), summary)
    print(json.dumps({k: v for k, v in summary.items() if k not in ("embeddings",)}, indent=1))
    return matched, requests, expected, summary


# -------------------------------------------------------------------- 3. vlm
async def vlm(inf, api, out, qm, requests, cap):
    vdir = os.path.join(out, "vlm")
    os.makedirs(os.path.join(vdir, "responses"), exist_ok=True)
    ledger = os.path.join(vdir, "request_ledger.jsonl")
    used = sum(1 for _ in open(ledger)) if os.path.exists(ledger) else 0
    rng = np.random.default_rng([SEED, 1])  # seeded order: queries shuffled, condition order shuffled within each query
    plan = [(f, "ABZ"[c]) for f in rng.permutation(list(qm["query_id"])) for c in rng.permutation(3)]
    wjson(os.path.join(vdir, "run_order.json"), [{"n": i + 1, "query_id": f, "condition": c, "payload_sha256": requests[(f, c)][0]} for i, (f, c) in enumerate(plan)])
    for attempt in (1, 2):                                    # second pass retries failures while the cap allows
        for f, cond in plan:
            key, inputs = requests[(f, cond)]
            rpath = os.path.join(vdir, "responses", key + ".json")
            if os.path.exists(rpath):
                continue
            if used >= cap:
                print(f"request cap reached ({used}/{cap}); {f} {cond} not sent")
                continue
            used += 1
            t0 = time.time()
            entry = {"n": used, "t": t0, "query_id": f, "condition": cond, "payload_sha256": key, "attempt": attempt}
            try:
                raw = await api.get_image_information(inputs)
                wjson(rpath, {"raw": raw, "meta": api.last_meta, "model": api.model, "first_query_id": f, "first_condition": cond})
                entry.update(ok=True, elapsed_s=round(time.time() - t0, 2), meta=api.last_meta)
            except Exception as e:
                entry.update(ok=False, elapsed_s=round(time.time() - t0, 2), error=str(e)[:500])
            with open(ledger, "a") as fh:
                fh.write(json.dumps(entry) + "\n")
            print(f"[{used}/{cap}] {f} {cond}: {'ok' if entry['ok'] else 'FAILED ' + entry['error'][:80]}", flush=True)
    print(f"Provider requests used: {used} of {cap}")


# ----------------------------------------------------------------- 4. report
def fp(p):
    return "<0.001" if p < 0.001 else f"{p:.3f}"


def report(inf, out, qm, rm, matched, requests, expected, summary, args):
    vdir = os.path.join(out, "vlm")
    ledger = [json.loads(l) for l in open(os.path.join(vdir, "request_ledger.jsonl"))] if os.path.exists(os.path.join(vdir, "request_ledger.jsonl")) else []
    rows = []
    for f, t in zip(qm["query_id"], qm["label"]):
        for cond in "ABZ":
            key = requests[(f, cond)][0]
            rp = os.path.join(vdir, "responses", key + ".json")
            raw = json.load(open(rp))["raw"] if os.path.exists(rp) else None
            pred, status = None, "missing"
            if raw is not None:
                try:
                    pred = inf.extract_json(raw)["prediction"]
                except Exception:
                    pred = None
                status = "valid" if pred in expected else "invalid"
            rows.append({"query_id": f, "true_label": t, "condition": cond, "payload_sha256": key, "status": status, "prediction": pred, "raw": raw})
    P = pd.DataFrame(rows)
    P.to_csv(os.path.join(vdir, "predictions.csv"), index=False)
    n = len(qm)
    truth = qm.set_index("query_id")["label"]
    W = P.pivot(index="query_id", columns="condition", values="prediction").loc[list(qm["query_id"])]
    S = P.pivot(index="query_id", columns="condition", values="status").loc[list(qm["query_id"])]
    shared = [f for f in qm["query_id"] if all(S.at[f, c] == "valid" for c in "ABZ")]
    M = matched.set_index("query_id")
    preds = {"A": W["A"], "B": W["B"], "Z": W["Z"], "knn_vit": M["knn4_vit"], "knn_dinov2": M["knn4_dinov2"]}
    NAME = {"A": "A. VLM, ViT-selected images", "B": "B. VLM, DINOv2-selected images, ViT label sequence", "Z": "Z. VLM, zero-shot",
            "knn_vit": "4-NN vote, original ViT (no VLM)", "knn_dinov2": "4-NN vote, DINOv2 block 12, unconstrained (no VLM)"}
    cov = {c: int((S[c] == "valid").sum()) for c in "ABZ"}
    cov.update(knn_vit=n, knn_dinov2=n)
    table = {}
    for c, p in preds.items():
        y, x = truth.loc[shared], p.loc[shared]
        table[c] = {"correct": int((x == y).sum()), "n": len(shared), "macro_f1": float(f1_score(y, x.astype(str), labels=sorted(expected), average="macro", zero_division=0)),
                    "coverage": cov[c]}

    def paired(x, y, ids):  # y versus x
        cx, cy = (preds[x].loc[ids] == truth.loc[ids]), (preds[y].loc[ids] == truth.loc[ids])
        w, l = int((cy & ~cx).sum()), int((~cy & cx).sum())
        return {"n": len(ids), f"{y}_correct": int(cy.sum()), f"{x}_correct": int(cx.sum()), f"{y}_only": w, f"{x}_only": l, "both_correct": int((cy & cx).sum()),
                "both_wrong": int((~cy & ~cx).sum()), "pp_difference": (100.0 * (int(cy.sum()) - int(cx.sum())) / len(ids)) if ids else float("nan"),
                "mcnemar_p": 1.0 if w + l == 0 else float(stats.binomtest(w, w + l, 0.5).pvalue)}
    both = [f for f in qm["query_id"] if S.at[f, "A"] == "valid" and S.at[f, "B"] == "valid"]
    changed = [f for f in both if not M.at[f, "identical_payload"]]
    prim, sub = paired("A", "B", both), paired("A", "B", changed)
    tok = {}
    for r in ledger:
        if r.get("ok"):
            u = r["meta"]["usage"]
            tok.setdefault(r["condition"], []).append(u["input_tokens"] + u["cache_creation_input_tokens"] + u["cache_read_input_tokens"])
    model_ids = sorted({m for r in ledger if r.get("ok") for m in r["meta"]["model_ids"]})
    metrics = {"model_requested": args.model, "model_ids_in_responses": model_ids, "turns_per_request": sorted({r["meta"]["num_turns"] for r in ledger if r.get("ok")}),
               "requests_sent": len(ledger), "requests_failed": sum(not r["ok"] for r in ledger), "max_requests": args.max_requests,
               "coverage": cov, "shared_valid_queries": len(shared), "conditions": table, "primary_B_vs_A": prim, "changed_payload_subset_B_vs_A": sub,
               "A_vs_Z": paired("Z", "A", shared), "B_vs_Z": paired("Z", "B", shared),
               "mean_input_tokens": {c: float(np.mean(v)) for c, v in tok.items()}, "offline": summary}
    wjson(os.path.join(out, "metrics.json"), metrics)

    ex = [f for f in changed if W.at[f, "A"] != W.at[f, "B"]]
    ex = sorted(ex, key=lambda f: (W.at[f, "A"] == truth[f]) == (W.at[f, "B"] == truth[f]))[:3]  # correctness flips first
    L = ["# DeepWeeds matched-label pilot: ViT-selected versus DINOv2-selected demonstration images\n",
         f"{n} fresh queries (seed {SEED}), {K} shots, reference bank of {len(rm)} images ({BANK_PER_CLASS} per class). Queries and bank come from disjoint capture dates. "
         f"VLM `{', '.join(model_ids) or args.model}` through headless `claude -p`. Requests sent: {len(ledger)} of {args.max_requests} allowed, "
         f"{metrics['requests_failed']} failed.\n",
         "A and B use the same label in every demonstration slot. The labels are those of the original ViT's top 4 references. "
         "B replaces each image with DINOv2 block 12's highest-ranked unused reference that has the slot's label.\n",
         "## Primary comparison: B versus A\n",
         f"Queries with valid predictions from both: {prim['n']} of {n}. A/B payloads identical on {summary['identical_AB_payloads']} of {n} queries, "
         f"and {summary['mean_differing_slots']:.2f} of {K} image slots differ on average.\n",
         "| Set | n | A correct | B correct | Difference (pp) | B-only wins | A-only wins | Both correct | Both wrong | Exact McNemar p |", "|---|---|---|---|---|---|---|---|---|---|"]
    for name, d in (("All valid pairs (primary)", prim), ("Changed payloads only (descriptive)", sub)):
        L.append(f"| {name} | {d['n']} | {d['A_correct']} | {d['B_correct']} | {d['pp_difference']:+.0f} | {d['B_only']} | {d['A_only']} | {d['both_correct']} | {d['both_wrong']} | {fp(d['mcnemar_p'])} |")
    L.append(f"\nA and B gave different predictions on {int((W['A'].loc[both] != W['B'].loc[both]).sum())} of {len(both)} queries.")
    L += [f"\n## All conditions on the {len(shared)} queries valid in A, B and Z\n", "| Condition | Correct | Accuracy | Macro-F1 (9 classes) | Coverage |", "|---|---|---|---|---|"]
    for c in ["Z", "A", "B", "knn_vit", "knn_dinov2"]:
        t = table[c]
        L.append(f"| {NAME[c]} | {t['correct']}/{t['n']} | {100 * t['correct'] / max(t['n'], 1):.0f}% | {t['macro_f1']:.3f} | {t['coverage']}/{n} |")
    L.append(f"\nThe 4-NN vote takes the most frequent label among the top {K}; ties go to the larger summed cosine, then alphabetical order. "
             f"The original ViT's top {K} hold {summary['mean_same_class_in_vit_labels']:.2f} same-class references on average, and unconstrained DINOv2 holds "
             f"{summary['unconstrained_dinov2_mean_same_class']:.2f}.\n")
    if ex:
        L.append("## Queries where A and B share labels and differ in prediction\n")
        for f in ex:
            a, b = json.loads(M.at[f, "A_refs"]), json.loads(M.at[f, "B_refs"])
            L += [f"**Query `{f}`**, true class {truth[f]}. Demonstration labels: {', '.join(json.loads(M.at[f, 'demo_labels']))}.\n",
                  "| Slot | Label | A reference | B reference |", "|---|---|---|---|"]
            L += [f"| {s + 1} | {json.loads(M.at[f, 'demo_labels'])[s]} | `{a[s]}` | `{b[s]}`{' (same image)' if a[s] == b[s] else ''} |" for s in range(K)]
            L.append(f"\nPrediction with A: {W.at[f, 'A']}. Prediction with B: {W.at[f, 'B']}.\n")
    notes = os.path.join(out, "NOTES.md")
    if os.path.exists(notes):
        L.append(open(notes).read())
    open(os.path.join(out, "SUMMARY.md"), "w").write("\n".join(L) + "\n")
    print("\n".join(L))


async def main(inf):
    args = parse_args()
    out = args.out_dir
    os.makedirs(out, exist_ok=True)
    qm, rm, classes = freeze(out)
    print(f"frozen: {len(qm)} queries, {len(rm)} bank images")
    if args.stage == "freeze":
        return
    api = inf.ClaudeHeadlessAPI(model=args.model, sandbox_dir=os.path.join(out, "vlm", "sandbox"))
    api.settings = "claude -p defaults (temperature not settable); system prompt: " + api.SYSTEM_PROMPT + "; command: " + " ".join(api.command()[:-1])
    matched, requests, expected, summary = offline(inf, api, out, qm, rm, classes)
    wjson(os.path.join(out, "run_config.json"), {
        "dataset": "DeepWeeds", "seed": SEED, "shots": K, "n_queries": len(qm), "bank_size": len(rm), "expected_classes": expected,
        "A": "google/vit-base-patch16-224-in21k pooler_output, cosine top-4 (inference.rank_candidates)",
        "B": "facebook/dinov2-base hidden_states[12] CLS, final LayerNorm, L2; per slot the top-ranked unused reference with A's label",
        "Z": "no demonstrations; content = [prompt, prompt, query image], the repository's 0-shot layout",
        "vendor": "claude_headless", "model": args.model, "settings": api.settings, "max_requests": args.max_requests,
        "prompt": inf.vision_prompt, "prompt_sha256": summary["prompt_sha256"], "image_format": "inference.load_image: RGB JPEG quality 95, base64"})
    if summary["changed_AB_payloads"] == 0:
        print("A and B payloads are identical on every query; the live comparison is skipped.")
        return
    if args.stage == "vlm":
        await vlm(inf, api, out, qm, requests, args.max_requests)
    if args.stage in ("vlm", "report"):
        report(inf, out, qm, rm, matched, requests, expected, summary, args)
