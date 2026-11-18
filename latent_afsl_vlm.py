"""Paired downstream test for Latent-Assisted Few-Shot Learning. Run through
  python inference.py --latent-afsl-vlm --stage {plan,run,report} --out-dir <run_dir> [--sets deepweeds ageval] [--conds A_S H_S A_V H_V] [--max-requests N]

DeepWeeds (primary): 153 held-out queries, frozen date-disjoint bank of 99 images. Conditions:
  A_S = ViT pooler_output retrieval + standard assisted prompt      H_S = selected ViT hidden state + standard prompt
  A_V = ViT pooler_output retrieval + vote-aware prompt             H_V = selected ViT hidden state + vote-aware prompt
  A_L, H_L = label-only control: the query image and the four neighbor labels in order, without the neighbor images
AgEval (historical protocol, six pools, 583 queries): H_S, paired with the cached A_S answers of runs/arbab_eval_seed42.
The VLM, prompt template, class list, image bytes, shot count and ordering rule are fixed; only the retrieved references change.
"""
import argparse, asyncio, json, os, threading, time
from concurrent.futures import ThreadPoolExecutor
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.metrics import f1_score
from pilot_matched_labels import K, SEED, fp, path_of, sha, wjson
from pilot_vote_aware import vote_text

MODEL = "claude-sonnet-5-5"
RUNS = os.path.join("runs"); BASE = os.path.join(RUNS, "deepweeds_matched_seed42"); ARB = os.path.join(RUNS, "arbab_eval_seed42"); OFF = os.path.join(RUNS, "offline_layer_stats")
AG = {"Bean Leaf Lesions": "load_and_prepare_data_bean_leaf", "DeepWeeds": "load_and_prepare_data_DeepWeeds", "Soybean Seeds": "load_and_prepare_data_soybean_seeds",
      "Mango Leaf Disease": "load_and_prepare_data_mango_leaf", "Soybean Diseases": "load_and_prepare_data_Soybean_PNAS", "Dangerous Insects": "load_and_prepare_data_Soybean_Dangerous_Insects"}


def label_text(labels):
    return "The 4 most similar labeled reference images have these labels, in order of similarity: " + ", ".join(f'"{l}"' for l in labels) + "."


async def main(inf):
    ap = argparse.ArgumentParser(); ap.add_argument("--latent-afsl-vlm", action="store_true"); ap.add_argument("--stage", choices=["plan", "run", "report"], default="plan")
    ap.add_argument("--out-dir", required=True); ap.add_argument("--sets", nargs="+", default=["deepweeds"]); ap.add_argument("--conds", nargs="+", default=["A_S", "H_S"])
    ap.add_argument("--max-requests", type=int, default=700); ap.add_argument("--workers", type=int, default=4)
    args = ap.parse_args(); out = args.out_dir; vdir = os.path.join(out, "vlm"); os.makedirs(os.path.join(vdir, "responses"), exist_ok=True)
    M = json.load(open(os.path.join(out, "manifest.json")))
    api0 = inf.ClaudeHeadlessAPI(model=MODEL, sandbox_dir=os.path.join(vdir, "sandbox"))
    settings = "claude -p defaults (temperature not settable); system prompt: " + api0.SYSTEM_PROMPT + "; command: " + " ".join(api0.command()[:-1])
    def key(inp): return sha(json.dumps({"vendor": "claude_headless", "model": MODEL, "settings": settings, "inputs": inp}, sort_keys=True).encode())
    def img_block(b64, lab): return [{"type": "image", "source": {"type": "base64", "media_type": "image/jpeg", "data": b64}}, {"type": "text", "text": f'{{"prediction": "{lab}"}}'}]
    reqs, rows = {}, []
    # ---------------- DeepWeeds, frozen date-disjoint bank
    cfg = json.load(open(os.path.join(BASE, "run_config.json"))); classes_dw = cfg["expected_classes"]; prompt_dw = inf.universal_prompt.format(expected_classes=classes_dw)
    assert sha(prompt_dw.encode()) == cfg["prompt_sha256"] and settings == cfg["settings"]
    N = pd.read_csv(os.path.join(out, "base_bank_neighbors.csv")); N = N[N["role"] != "dev"]
    b64 = {}
    def im(f):
        if f not in b64: b64[f] = inf.load_image(path_of(f))
        return b64[f]
    if "deepweeds" in args.sets:
        for rk in ("A", "H", "DINO"):
            for _, w in N[N["readout_key"] == rk].iterrows():
                ids, labs = json.loads(w["neighbor_ids"]), json.loads(w["neighbor_labels"]); ex = [b for f, l in zip(ids, labs) for b in img_block(im(f), l)]
                variants = {"S": ex, "V": ex + [{"type": "text", "text": vote_text(labs)}], "L": [{"type": "text", "text": label_text(labs)}]}
                for pv, examples in variants.items():
                    inp = {"image": im(w["query_id"]), "examples": examples, "prompt": prompt_dw}; k = key(inp); c = f"{rk}_{pv}"
                    reqs[("deepweeds", w["query_id"], c)] = (k, inp)
                    rows.append({"set": "deepweeds", "dataset": "DeepWeeds", "query_id": w["query_id"], "group": str(w["query_date"]), "role": w["role"], "true_label": w["true_label"], "cond": c, "payload_sha256": k,
                                 "neighbor_ids": w["neighbor_ids"], "neighbor_labels": w["neighbor_labels"], "neighbor_groups": w["neighbor_dates"], "cosines": w["cosines"], "vote": w["vote"], "margin": w["margin"], "same_class": w["same_class"]})
    # ---------------- AgEval pools (leave-one-out)
    if "ageval" in args.sets:
        V = pd.read_csv(os.path.join(out, "ageval_votes.csv")); arb = pd.read_csv(os.path.join(ARB, "per_query_results.csv"), keep_default_na=False).set_index(["dataset", "query_idx"])
        for ds, loader in AG.items():
            data, classes, _ = getattr(inf, loader)(100); classes = [str(c) for c in classes]; n = len(data); lab = [str(data.at[i, 1]) for i in range(n)]
            pool = pd.read_csv(os.path.join(OFF, f"pool_{ds.replace(' ', '_')}.csv")); assert [os.path.basename(p) for p in pool["path"]] == [os.path.basename(data.at[i, 0]) for i in range(n)]
            prompt = inf.universal_prompt.format(expected_classes=classes); imgs = [inf.load_image(data.at[i, 0]) for i in range(n)]
            for rk in ("A", "H"):
                v = V[(V.dataset == ds) & (V.readout_key == rk)].set_index("query_idx")
                for q in range(n):
                    top = json.loads(v.at[q, "top4_idx"]); ex = [b for i in top for b in img_block(imgs[i], lab[i])]
                    inp = {"image": imgs[q], "examples": ex, "prompt": prompt}; k = key(inp)
                    if rk == "A": assert k == arb.at[(ds, q), "payload_A"], f"{ds} q{q}: rebuilt baseline payload differs from the cached run"
                    reqs[("ageval", f"{ds}|{q}", f"{rk}_S")] = (k, inp)
                    rows.append({"set": "ageval", "dataset": ds, "query_id": f"{ds}|{q}", "group": "", "role": "ageval", "true_label": lab[q], "cond": f"{rk}_S", "payload_sha256": k,
                                 "neighbor_ids": json.dumps(top), "neighbor_labels": json.dumps([lab[i] for i in top]), "neighbor_groups": "[]", "cosines": "[]", "vote": v.at[q, "vote"], "margin": -1, "same_class": int(v.at[q, "same_class"])})
    P = pd.DataFrame(rows); P = P[P["cond"].isin(args.conds)].reset_index(drop=True)
    def rpath(k):
        for d in (os.path.join(vdir, "responses"), os.path.join(ARB, "vlm", "responses"), os.path.join(RUNS, "deepweeds_demo_value_seed42", "vlm", "responses")):
            if os.path.exists(os.path.join(d, k + ".json")): return os.path.join(d, k + ".json")
        return None
    P["cached"] = [rpath(k) is not None for k in P["payload_sha256"]]
    plan = {"model": MODEL, "settings": settings, "sets": args.sets, "conditions": args.conds, "max_requests": args.max_requests,
            "per_condition": {f"{s}|{c}": {"queries": int(len(g)), "unique_payloads": int(g["payload_sha256"].nunique()), "cached": int(g.drop_duplicates("payload_sha256")["cached"].sum()),
                                          "uncached": int((~g.drop_duplicates("payload_sha256")["cached"]).sum())} for (s, c), g in P.groupby(["set", "cond"])},
            "identical_payloads_A_S_vs_H_S": {s: int((g[g.cond == "A_S"].set_index("query_id")["payload_sha256"] == g[g.cond == "H_S"].set_index("query_id")["payload_sha256"]).sum()) for s, g in P.groupby("set") if {"A_S", "H_S"} <= set(g.cond)},
            "unique_uncached_total": int((~P.drop_duplicates("payload_sha256")["cached"]).sum()), "selected_readout": M["selected_vit_name"]}
    wjson(os.path.join(vdir, f"request_plan_{'_'.join(args.sets)}_{'_'.join(args.conds)}.json"), plan); print(json.dumps(plan, indent=1))
    ledger = os.path.join(vdir, "request_ledger.jsonl")
    if args.stage == "run":
        used = [sum(1 for _ in open(ledger)) if os.path.exists(ledger) else 0]; lock = threading.Lock(); kk = {}
        for _, w in P.iterrows(): kk.setdefault(w["payload_sha256"], (w["set"], w["query_id"], w["cond"]))
        todo = [k for k in kk if rpath(k) is None]; todo = [todo[i] for i in np.random.default_rng([SEED, 21]).permutation(len(todo))]     # conditions interleaved
        def work(k):
            with lock:
                if used[0] >= args.max_requests: return
                used[0] += 1; nreq = used[0]
            s, q, c = kk[k]; api = inf.ClaudeHeadlessAPI(model=MODEL, sandbox_dir=os.path.join(vdir, "sandbox")); t0 = time.time(); e = {"n": nreq, "t": t0, "set": s, "query_id": q, "cond": c, "payload_sha256": k}
            try:
                raw = asyncio.run(api.get_image_information(reqs[(s, q, c)][1]))
                wjson(os.path.join(vdir, "responses", k + ".json"), {"raw": raw, "meta": api.last_meta, "model": MODEL, "set": s, "query_id": q, "first_cond": c})
                e.update(ok=True, elapsed_s=round(time.time() - t0, 2), model_ids=api.last_meta.get("model_ids"), num_turns=api.last_meta.get("num_turns"))
            except Exception as ex:
                e.update(ok=False, elapsed_s=round(time.time() - t0, 2), error=str(ex)[:300])
            with lock:
                open(ledger, "a").write(json.dumps(e) + "\n")
                if not e["ok"] or nreq % 100 == 0: print(f"[{nreq}/{args.max_requests}] {s} {q} {c}: {'ok' if e['ok'] else 'FAILED ' + e['error'][:120]}", flush=True)
        for attempt in (1, 2):
            with ThreadPoolExecutor(args.workers) as ex_: list(ex_.map(work, [k for k in todo if rpath(k) is None]))
        print(f"requests used so far: {used[0]} of {args.max_requests}", flush=True)
    if args.stage == "plan":
        return
    # ---------------- predictions table (every condition that has any response)
    def pred(k):
        p = rpath(k)
        if p is None: return None, "missing", ""
        try: v = str(inf.extract_json(json.load(open(p))["raw"])["prediction"])
        except Exception: v = None
        return v, ("valid" if v is not None else "invalid"), ("this run" if p.startswith(vdir) else "cached earlier run")
    allrows = pd.DataFrame(rows); pr = [pred(k) for k in allrows["payload_sha256"]]
    allrows["raw_pred"], allrows["status"], allrows["response_source"] = [a for a, _, _ in pr], [b for _, b, _ in pr], [c for _, _, c in pr]
    allrows["correct"] = allrows["raw_pred"].astype(str) == allrows["true_label"].astype(str); allrows["vote_correct"] = allrows["vote"].astype(str) == allrows["true_label"].astype(str)
    allrows[allrows["status"] != "missing"].to_csv(os.path.join(out, "vlm_per_query.csv"), index=False)
    print("per-condition coverage:", allrows.groupby(["set", "cond"])["status"].value_counts().unstack(fill_value=0).to_dict("index"))
