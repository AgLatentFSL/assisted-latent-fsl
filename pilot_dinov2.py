"""Paired retrieval-encoder pilot, run through `python inference.py --pilot ...`.

Conditions share the queries, the reference pool, the prompt and the VLM; only
the encoder that ranks the few-shot demonstrations changes.

Stages
  retrieval : embeddings, validation checks, rankings, manifests, prompt dump (no VLM calls)
  vlm       : retrieval stage, then VLM calls under a hard request cap with an on-disk cache
"""
import argparse
import hashlib
import json
import os
import time

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split


def _sha(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def _write_json(path, obj):
    with open(path, "w") as f:
        json.dump(obj, f, indent=2, sort_keys=True)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--pilot", action="store_true")
    p.add_argument("--stage", choices=["retrieval", "vlm"], default="retrieval")
    p.add_argument("--dataset", default="bean_leaf")
    p.add_argument("--samples", type=int, default=100, help="passed to the dataset loader (reference pool size)")
    p.add_argument("--encoders", nargs="+", default=["vit", "dinov2_last", "dinov2_mid"])
    p.add_argument("--shots", type=int, default=4)
    p.add_argument("--num-queries", type=int, default=20)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--limit-queries", type=int, default=None, help="only process the first N frozen queries")
    p.add_argument("--vendor", default="claude_headless", choices=["claude_headless", "anthropic", "openai", "google", "openrouter"])
    p.add_argument("--model", default="claude-sonnet-5-5")
    p.add_argument("--max-requests", type=int, default=60, help="hard cap on provider requests across restarts")
    p.add_argument("--out-dir", required=True)
    return p.parse_args()


def make_api(inf, vendor, model, out_dir):
    if vendor == "claude_headless":
        return inf.ClaudeHeadlessAPI(model=model, sandbox_dir=os.path.join(out_dir, "vlm", "sandbox"))
    if vendor == "anthropic":
        api = inf.ClaudeAPI(api_key=os.getenv("ANTHROPIC_API_KEY"), model=model)
        api.client = api.client.with_options(max_retries=0)  # every request must be counted
        return api
    if vendor == "openai":
        return inf.GPTAPI(api_key=os.getenv("OPENAI_API_KEY"), model=model)
    if vendor == "google":
        return inf.GeminiAPI(api_key=os.getenv("GOOGLE_API_KEY"), model=model)
    return inf.OpenRouterAPI(api_key=os.getenv("OPENROUTER_API_KEY"), model=model)


async def pilot_main(inf):
    args = parse_args()
    out = args.out_dir
    os.makedirs(os.path.join(out, "retrieval"), exist_ok=True)
    loaders = {"bean_leaf": inf.load_and_prepare_data_bean_leaf, "deepweeds": inf.load_and_prepare_data_DeepWeeds,
               "soybean_diseases": inf.load_and_prepare_data_Soybean_PNAS, "soybean_seeds": inf.load_and_prepare_data_soybean_seeds}
    all_data, expected_classes, dataset_name = loaders[args.dataset](args.samples)
    inf.vision_prompt = inf.universal_prompt.format(expected_classes=expected_classes)
    n = len(all_data)

    # ---- manifests (frozen before any model call) -------------------------
    data_root = os.path.join("data", "bean-leaf-lesions-classification") if args.dataset == "bean_leaf" else "data"
    image_id = [os.path.relpath(all_data.at[i, 0], data_root) for i in range(n)]
    label = [all_data.at[i, 1] for i in range(n)]
    md5 = [hashlib.md5(open(all_data.at[i, 0], "rb").read()).hexdigest() for i in range(n)]
    if args.dataset == "bean_leaf":
        assert all(os.path.basename(os.path.dirname(all_data.at[i, 0])) == label[i] for i in range(n)), "label/folder mismatch"
    assert set(label) <= set(expected_classes)
    ref = pd.DataFrame({"idx": range(n), "image_id": image_id, "label": label, "md5": md5})
    ref_path = os.path.join(out, "reference_manifest.csv")
    if os.path.exists(ref_path):
        old = pd.read_csv(ref_path)
        assert list(old["image_id"]) == image_id and list(old["md5"]) == md5, "reference pool differs from the frozen manifest"
    ref.to_csv(ref_path, index=False)

    _, query_idx = train_test_split(list(range(n)), test_size=args.num_queries, random_state=args.seed, stratify=label)
    query_idx = sorted(query_idx)
    # a query's candidates: every pool image except itself and byte-identical copies of it
    excluded = {q: sorted(j for j in range(n) if j == q or md5[j] == md5[q]) for q in query_idx}
    qman = pd.DataFrame({"idx": query_idx, "image_id": [image_id[q] for q in query_idx], "label": [label[q] for q in query_idx],
                         "excluded_idx": [json.dumps(excluded[q]) for q in query_idx],
                         "n_candidates": [n - len(excluded[q]) for q in query_idx]})
    qman.to_csv(os.path.join(out, "query_manifest.csv"), index=False)
    _write_json(os.path.join(out, "query_class_counts.json"), {
        "query": qman["label"].value_counts().sort_index().to_dict(),
        "reference_pool": ref["label"].value_counts().sort_index().to_dict(),
        "byte_identical_groups_in_pool": int((ref["md5"].value_counts() > 1).sum())})

    api = make_api(inf, args.vendor, args.model, out)
    config = {"dataset": dataset_name, "loader_samples": args.samples, "pool_size": n, "expected_classes": expected_classes,
              "split": "repository protocol: loader samples the pool from train/ (seed 42, class-balanced); each query retrieves from the rest of the pool",
              "num_queries": args.num_queries, "seed": args.seed, "shots": args.shots, "encoders": args.encoders,
              "encoder_models": {"vit": "google/vit-base-patch16-224-in21k pooler_output (tanh(dense(final-LayerNorm CLS)))",
                                 "dinov2_last": "facebook/dinov2-base hidden_states[12] CLS, final LayerNorm, L2",
                                 "dinov2_mid": "facebook/dinov2-base hidden_states[6] CLS, final LayerNorm, L2"},
              "vendor": args.vendor, "model": args.model, "max_requests": args.max_requests,
              "prompt_template": inf.universal_prompt, "vision_prompt": inf.vision_prompt,
              "decoding": ("claude -p defaults (temperature not settable); system prompt: " + api.SYSTEM_PROMPT + "; command: " + " ".join(api.command()[:-1]))
                          if args.vendor == "claude_headless" else "repository defaults in inference.py (temperature 1.0, max_tokens 4096)",
              "image_format": "inference.load_image: RGB JPEG quality 95, base64"}
    _write_json(os.path.join(out, "run_config.json"), config)

    queries = query_idx[: args.limit_queries] if args.limit_queries else query_idx
    tag = f"_first{len(queries)}" if args.limit_queries else ""

    # ---- offline retrieval -------------------------------------------------
    rankings, top_rows, checks = {}, [], []
    for enc in args.encoders:
        if enc == "random":  # control: demonstrations drawn uniformly from the same candidates (seeded per query)
            for q in queries:
                cand = [i for i in range(n) if i not in excluded[q]]
                perm = np.random.default_rng([args.seed, q]).permutation(cand)
                rankings[(enc, q)] = [(int(i), 0.0) for i in perm]
                for rank, (i, s) in enumerate(rankings[(enc, q)][: args.shots], 1):
                    top_rows.append({"encoder": enc, "query_idx": q, "query_id": image_id[q], "query_label": label[q], "rank": rank,
                                     "ref_idx": i, "ref_id": image_id[i], "ref_label": label[i], "cosine": float("nan"),
                                     "same_class": int(label[i] == label[q])})
            continue
        embeddings = inf.precompute_embeddings(all_data, enc)
        assert len(embeddings) == n, f"{enc}: {n - len(embeddings)} embeddings failed"
        mat = np.array([embeddings[i] for i in range(n)], dtype=np.float64)
        np.save(os.path.join(out, "retrieval", f"embeddings_{enc}.npy"), mat.astype(np.float32))
        norms = np.linalg.norm(mat, axis=1)
        for q in queries:
            q_emb = inf.get_image_embedding(all_data.at[q, 0], model_type=enc)
            cand = {i: e for i, e in embeddings.items() if i not in excluded[q]}
            ranked = inf.rank_candidates(q, q_emb, cand)
            top = ranked[: args.shots]
            top_idx = [i for i, _ in top]
            scores = np.array([s for _, s in ranked])
            checks.append({
                "encoder": enc, "query_idx": q, "dim": len(q_emb), "finite": bool(np.isfinite(mat).all() and np.isfinite(scores).all()),
                "query_norm": float(np.linalg.norm(q_emb)), "pool_norm_min": float(norms.min()), "pool_norm_max": float(norms.max()),
                "score_min": float(scores.min()), "score_max": float(scores.max()),
                "scores_in_unit_range": bool((scores <= 1 + 1e-6).all() and (scores >= -1 - 1e-6).all()),
                "sorted_desc": bool((np.diff(scores) <= 1e-12).all()),
                "n_candidates": len(ranked), "unique_top": len(set(top_idx)) == args.shots,
                "no_self": q not in top_idx and not (set(top_idx) & set(excluded[q])),
                "query_emb_matches_pool": bool(np.allclose(q_emb, mat[q], atol=1e-6))})
            rankings[(enc, q)] = ranked
            for rank, (i, s) in enumerate(top, 1):
                top_rows.append({"encoder": enc, "query_idx": q, "query_id": image_id[q], "query_label": label[q], "rank": rank,
                                 "ref_idx": i, "ref_id": image_id[i], "ref_label": label[i], "cosine": round(s, 6),
                                 "same_class": int(label[i] == label[q])})
        with open(os.path.join(out, "retrieval", f"rankings_{enc}{tag}.jsonl"), "w") as f:
            for q in queries:
                f.write(json.dumps({"query_idx": q, "query_id": image_id[q],
                                    "ranked": [[image_id[i], round(s, 6)] for i, s in rankings[(enc, q)]]}) + "\n")
    checks = pd.DataFrame(checks)
    checks.to_csv(os.path.join(out, "retrieval", f"validation_checks{tag}.csv"), index=False)
    top_df = pd.DataFrame(top_rows)
    top_df.to_csv(os.path.join(out, "retrieval", f"top{args.shots}{tag}.csv"), index=False)
    bad = checks[~(checks["finite"] & checks["scores_in_unit_range"] & checks["sorted_desc"] & checks["unique_top"]
                   & checks["no_self"] & checks["query_emb_matches_pool"])]
    print(checks.groupby("encoder").agg(dim=("dim", "first"), qnorm_min=("query_norm", "min"), qnorm_max=("query_norm", "max"),
                                        score_min=("score_min", "min"), score_max=("score_max", "max"),
                                        n_cand=("n_candidates", "min")).to_string())
    assert bad.empty, f"retrieval validation failed:\n{bad}"
    print(f"Retrieval validation passed for {len(queries)} queries x {len(args.encoders)} encoders")

    # ---- prompt assembly (inputs are what the repository's process_image sends) ----
    def assemble(enc, q):
        top_idx = [i for i, _ in rankings[(enc, q)][: args.shots]]
        examples, paths, cats, same = inf.build_examples(api, all_data, top_idx, label[q])
        assert cats == [label[i] for i in top_idx] and paths == [all_data.at[i, 0] for i in top_idx], "example/label misalignment"
        return {"image": inf.load_image(all_data.at[q, 0]), "examples": examples, "prompt": inf.vision_prompt}, top_idx, same

    def redact(obj):  # replace base64 payloads by a hash so a prompt can be read and compared
        if isinstance(obj, dict):
            return {k: (f"<base64 jpeg sha256:{_sha(v.encode())[:16]}>" if k in ("data", "image") and isinstance(v, str) and len(v) > 200
                        else (f"<data url sha256:{_sha(v.encode())[:16]}>" if k == "url" and isinstance(v, str) and len(v) > 200 else redact(v)))
                    for k, v in obj.items()}
        if isinstance(obj, list):
            return [redact(v) for v in obj]
        return obj

    q0 = queries[0]
    for enc in args.encoders:
        inputs, top_idx, _ = assemble(enc, q0)
        _write_json(os.path.join(out, "retrieval", f"prompt_example_{enc}.json"), {
            "query_idx": q0, "query_id": image_id[q0], "query_label": label[q0],
            "query_image_sha256_16": _sha(inputs["image"].encode())[:16],
            "selected": [{"rank": r, "ref_id": image_id[i], "ref_label": label[i],
                          "image_sha256_16": _sha(inf.load_image(all_data.at[i, 0]).encode())[:16]} for r, i in enumerate(top_idx, 1)],
            "inputs": redact(inputs)})

    if args.stage == "retrieval":
        return

    # ---- capped, cached VLM calls -------------------------------------------
    vdir = os.path.join(out, "vlm")
    os.makedirs(os.path.join(vdir, "responses"), exist_ok=True)
    ledger_path = os.path.join(vdir, "request_ledger.jsonl")
    used = sum(1 for _ in open(ledger_path)) if os.path.exists(ledger_path) else 0
    settings = config["decoding"]
    rows = []
    for q in queries:               # query-major order keeps conditions paired if the cap is reached
        for enc in args.encoders:
            inputs, top_idx, same = assemble(enc, q)
            key = _sha(json.dumps({"vendor": args.vendor, "model": args.model, "settings": settings, "inputs": inputs}, sort_keys=True).encode())
            rpath = os.path.join(vdir, "responses", key + ".json")
            row = {"query_idx": q, "query_id": image_id[q], "true_label": label[q], "encoder": enc, "payload_sha256": key,
                   "example_ids": json.dumps([image_id[i] for i in top_idx]), "example_labels": json.dumps([label[i] for i in top_idx]),
                   "same_class_count": same}
            if os.path.exists(rpath):
                resp = json.load(open(rpath))
                row.update(status="ok", source="cache", raw=resp["raw"])
            elif used >= args.max_requests:
                row.update(status="missing", source="request cap reached", raw=None)
            else:
                used += 1
                t0 = time.time()
                entry = {"n": used, "t": t0, "query_idx": q, "encoder": enc, "payload_sha256": key}
                try:
                    raw = await api.get_image_information(inputs)
                    meta = getattr(api, "last_meta", {})
                    _write_json(rpath, {"raw": raw, "meta": meta, "model": args.model, "vendor": args.vendor,
                                        "first_query_idx": q, "first_encoder": enc})
                    entry.update(ok=True, elapsed_s=round(time.time() - t0, 2), meta=meta)
                    row.update(status="ok", source="request", raw=raw)
                except Exception as e:
                    entry.update(ok=False, elapsed_s=round(time.time() - t0, 2), error=str(e)[:500])
                    row.update(status="missing", source=f"request failed: {str(e)[:200]}", raw=None)
                with open(ledger_path, "a") as f:
                    f.write(json.dumps(entry) + "\n")
            pred = "NA"
            if row["raw"] is not None:
                try:
                    pred = inf.extract_json(row["raw"])["prediction"]
                except Exception:
                    pred = "NA"
            row["prediction"] = pred
            rows.append(row)
            print(f"[{used}/{args.max_requests}] q{q} {enc}: {row['status']} ({row['source'][:40]}) true={label[q]} pred={pred}")
            pd.DataFrame(rows).to_csv(os.path.join(vdir, f"predictions{tag}.csv"), index=False)
    print(f"Provider requests used so far: {used} of {args.max_requests}")
