"""Downstream layer test at every budget, within encoders: for K = 1, 2, 4, 8 and each of six encoders, the pooled (final) output and the five hidden
layers with the largest net same-class coverage gain over the pooled output of the SAME encoder at that K (atlas retrieval analysis) are sent to the VLM
with the standard assisted prompt. Each layer is compared with the pooled output of its own encoder at the same K. Same model, client, prompt and image
encoding as the main run. Identical prompts are sent once; answers of earlier runs (main run, layer_vlm_k2) are reused when the request is identical.
Usage (from repo/): python pilot_layer_vlm_allk.py <out_dir> {freeze|vlm|report} [--workers 4] [--max-requests N]
"""
import argparse, asyncio, json, os, shutil, sys, threading, time
from concurrent.futures import ThreadPoolExecutor
import numpy as np, pandas as pd

ap = argparse.ArgumentParser(); ap.add_argument("out"); ap.add_argument("stage", choices=["freeze", "vlm", "report"]); ap.add_argument("--workers", type=int, default=4); ap.add_argument("--max-requests", type=int, default=60000)
args = ap.parse_args(); out = args.out; os.makedirs(os.path.join(out, "responses"), exist_ok=True)
ATLAS = "atlas"; MAIN = os.path.join("runs", "arbab_eval_seed42"); PRIOR = [os.path.join(MAIN, "vlm", "responses"), os.path.join("runs", "layer_vlm_k2", "responses")]; MODEL = "claude-sonnet-5-5"; TOPN = 5; KS = (1, 2, 4, 8); K = max(KS)
ENC = {"ViT": "1_ViT", "CLIP": "3_CLIP", "ResNet50": "4_ResNet50", "SigLIP2": "5_SigLIP2", "Qwen3VL": "8_Qwen3VL", "FLAVA": "9_FLAVA"}; INV = {v: k for k, v in ENC.items()}
P = lambda f: os.path.join(out, f)

if args.stage == "freeze":
    assert not os.path.exists(P("conditions.csv")), "plan already frozen"
    PQ = pd.read_csv(os.path.join(ATLAS, "analysis", "tables", "per_query_topk.csv.gz"), usecols=["encoder", "layer", "K", "transition_vs_final_output"]); PQ = PQ[PQ.encoder.isin(ENC.values()) & (PQ.layer != "final_output")]
    PQ["d"] = PQ.transition_vs_final_output.map({"0 to >=1": 1, ">=1 to 0": -1}).fillna(0); R = PQ.groupby(["K", "encoder", "layer"], sort=False).d.agg(net=lambda x: 100 * x.mean(), gained=lambda x: int((x == 1).sum()), lost=lambda x: int((x == -1).sum())).reset_index()
    rows = [{"condition": "zero_shot", "encoder": "", "layer": "", "K": 0, "rank_in_encoder": "", "retrieval_net_coverage_pp": "", "retrieval_gained": "", "retrieval_lost": ""}]
    for k in KS:
        for e in ENC.values():
            rows.append({"condition": f"{INV[e]}|final_output|K{k}", "encoder": INV[e], "layer": "final_output", "K": k, "rank_in_encoder": "pooled", "retrieval_net_coverage_pp": 0.0, "retrieval_gained": 0, "retrieval_lost": 0})
            g = R[(R.K == k) & (R.encoder == e)].reset_index(drop=True); g["order"] = range(len(g)); g = g.sort_values(["net", "order"], ascending=[False, True]).head(TOPN)          # earlier layer on ties
            rows += [{"condition": f"{INV[e]}|{r.layer}|K{k}", "encoder": INV[e], "layer": r.layer, "K": k, "rank_in_encoder": i + 1, "retrieval_net_coverage_pp": round(r.net, 2), "retrieval_gained": r.gained, "retrieval_lost": r.lost} for i, r in enumerate(g.itertuples())]
    Cn = pd.DataFrame(rows); Cn.to_csv(P("conditions.csv"), index=False)
    json.dump({"frozen_at": time.strftime("%Y-%m-%d %H:%M:%S"), "question": "within one encoder, does a hidden layer change the accuracy of the VLM against the pooled output of the same encoder, at budgets of 1, 2, 4 and 8 references?", "K": list(KS), "model": MODEL, "encoders": list(ENC),
               "queries": "the 500 queries of the atlas (100 per dataset) with the disjoint reference bank of 100 images per dataset (runs/disjoint_split_seed42)",
               "conditions": f"zero-shot; for each K and each encoder the pooled (final) output and the {TOPN} hidden layers with the largest net same-class coverage gain over the pooled output of the same encoder at that K (earlier layer on ties)",
               "primary_analysis": "accuracy of each hidden layer against the pooled output of its own encoder at the same K on the same queries: paired bootstrap interval, wins, losses, exact McNemar test; Holm correction over all layer comparisons of the run is reported next to the single p-values",
               "secondary_analysis": ["mean of the five layers of an encoder against its pooled output, per K", "each condition against zero-shot", "accuracy by dataset", "relation between retrieval coverage change and VLM accuracy change"],
               "known_limits": ["the layers are selected by retrieval gain on the queries that are then evaluated", "results of an earlier run at K = 2 (layer_vlm_k2) were seen before this plan was frozen, and its answers are reused for identical requests", "queries and references are not separated by capture date", "one VLM, one request per prompt"],
               "caching": "responses are stored by the SHA-256 of the full request; identical requests are sent once; identical requests of the main run and of layer_vlm_k2 reuse their stored answers"}, open(P("preregistration.json"), "w"), indent=1)
    print(Cn.to_string()); sys.exit()

import inference as inf
from pilot_matched_labels import sha, wjson
Cn = pd.read_csv(P("conditions.csv"), keep_default_na=False); prompts = json.load(open(os.path.join(MAIN, "prompts.json")))
api0 = inf.ClaudeHeadlessAPI(model=MODEL, sandbox_dir=P("sandbox")); settings = "claude -p defaults (temperature not settable); system prompt: " + api0.SYSTEM_PROMPT + "; command: " + " ".join(api0.command()[:-1])
assert settings == json.load(open(os.path.join(MAIN, "run_config.json")))["settings"], "client settings differ from the main run"
SI = pd.read_csv(os.path.join(ATLAS, "split_images.csv"), dtype=str, keep_default_na=False); path = {(r.dataset, r.role, r.image_id): r.source_path for r in SI.itertuples()}
QS = SI[SI.role == "query"].reset_index(drop=True)
IDX = pd.read_csv(os.path.join(ATLAS, "index.csv"), dtype=str, keep_default_na=False); IDX = IDX[IDX["rank"].astype(int) <= K]
keep = {(r.encoder, r.layer) for r in Cn.itertuples() if r.encoder}; IDX = IDX[[(m, l) in keep for m, l in zip(IDX.model, IDX.layer)]]
refs = {k: g.sort_values("rank", key=lambda s: s.astype(int))[["reference_id", "reference_label"]].values.tolist() for k, g in IDX.groupby(["dataset", "query_folder", "model", "layer"])}
_img = {}
def image(p):
    if p not in _img: _img[p] = inf.load_image(p); assert _img[p] is not None, p
    return _img[p]
def request(q, c):
    """the full request of query row q under condition row c (inputs dict of the main run's client)"""
    ex = []
    if c.K:
        for rid, rlab in refs[(q.dataset, q.query_folder, c.encoder, c.layer)][:int(c.K)]:
            ex += [{"type": "image", "source": {"type": "base64", "media_type": "image/jpeg", "data": image(path[(q.dataset, "reference", rid)])}}, {"type": "text", "text": f'{{"prediction": "{rlab}"}}'}]
    return {"image": image(q.source_path), "examples": ex, "prompt": prompts[q.dataset]}
def key(inp): return sha(json.dumps({"vendor": "claude_headless", "model": MODEL, "settings": settings, "inputs": inp}, sort_keys=True).encode())
rp = lambda k: P(os.path.join("responses", k + ".json"))

plan_path = P("request_plan.csv"); PL0 = pd.read_csv(plan_path, dtype=str, keep_default_na=False) if os.path.exists(plan_path) else pd.DataFrame(columns=["condition"])
rows = []
for c in Cn.itertuples():
    if c.condition in set(PL0.condition): continue
    for q in QS.itertuples(): rows.append({"condition": c.condition, "dataset": q.dataset, "query_folder": q.query_folder, "true_label": q[4], "payload": key(request(q, c))})
    print("planned", c.condition, flush=True)
if rows: pd.concat([PL0, pd.DataFrame(rows)]).to_csv(plan_path, index=False)
PL = pd.read_csv(plan_path, dtype=str, keep_default_na=False)
reused = 0
for k in PL.payload.unique():                      # answers of earlier runs for identical requests
    for d_ in PRIOR:
        src = os.path.join(d_, k + ".json")
        if not os.path.exists(rp(k)) and os.path.exists(src): shutil.copyfile(src, rp(k)); reused += 1
todo = [k for k in PL.payload.unique() if not os.path.exists(rp(k))]
print(f"conditions {len(Cn)}, query-condition pairs {len(PL)}, distinct requests {PL.payload.nunique()}, reused from earlier runs {reused}, already answered {PL.payload.nunique() - len(todo)}, to send {len(todo)}", flush=True)

if args.stage == "vlm":
    first = PL.drop_duplicates("payload").set_index("payload"); cmap = {c.condition: c for c in Cn.itertuples()}; qmap = {(q.dataset, q.query_folder): q for q in QS.itertuples()}
    rng = np.random.default_rng(42); todo = [todo[i] for i in rng.permutation(len(todo))][:args.max_requests]; lock = threading.Lock(); state = {"n": 0, "fail_streak": 0, "stop": False}; ledger = P("request_ledger.jsonl")
    def work(k):
        if state["stop"]: return
        r = first.loc[k]; inp = request(qmap[(r.dataset, r.query_folder)], cmap[r.condition]); assert key(inp) == k
        api = inf.ClaudeHeadlessAPI(model=MODEL, sandbox_dir=P("sandbox")); t0 = time.time(); e = {"t": t0, "payload": k, "condition": r.condition}
        try:
            raw = asyncio.run(api.get_image_information(inp)); wjson(rp(k), {"raw": raw, "meta": api.last_meta, "model": MODEL, "condition": r.condition, "dataset": r.dataset, "query_folder": r.query_folder}); e.update(ok=True, elapsed_s=round(time.time() - t0, 2), cost=api.last_meta.get("total_cost_usd"))
        except Exception as ex: e.update(ok=False, elapsed_s=round(time.time() - t0, 2), error=str(ex)[:300])
        with lock:
            state["n"] += 1; state["fail_streak"] = 0 if e["ok"] else state["fail_streak"] + 1; open(ledger, "a").write(json.dumps(e) + "\n")
            if not e["ok"] and state["fail_streak"] in (1, 10): print(f"[{state['n']}] FAILED: {e['error'][:200]}", flush=True)
            if state["fail_streak"] >= 25 and not state["stop"]: state["stop"] = True; print("25 consecutive failures: stopping; rerun the vlm stage to resume", flush=True)
            if state["n"] % 500 == 0: print(f"[{time.strftime('%H:%M')}] {state['n']} of {len(todo)} sent", flush=True)
    with ThreadPoolExecutor(args.workers) as ex: list(ex.map(work, todo))
    left = sum(not os.path.exists(rp(k)) for k in PL.payload.unique()); print(f"[{time.strftime('%H:%M')}] done; requests still unanswered: {left}", flush=True); sys.exit()


print('report stage: see pilot_layer_vlm_allk_report.py')
