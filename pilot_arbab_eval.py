"""Assisted few-shot learning, evaluated with the AgEval protocol of Arshad et al.

Per dataset: the repository loader samples a class-balanced pool (samples=100, seed 42); every pool image is a query
and retrieves from the rest of the pool (leave-one-out), as in inference.process_image. Four shots. Conditions:
  Z = zero-shot
  R = random demonstrations (seeded per query)
  A = assisted few-shot of Arshad et al.: original ViT retrieval, standard prompt
  O = ours, training-free: DINOv2 block 12 retrieval, vote-aware prompt
Votes of the ViT and DINOv2 references are reported as no-VLM baselines. This protocol has no source-group split.

Run through `python inference.py --pilot-arbab-eval --stage {offline,vlm,report} --out-dir <dir> [--datasets ...]`.
"""
import argparse, asyncio, json, os, threading, time
from concurrent.futures import ThreadPoolExecutor
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.metrics import f1_score
from pilot_matched_labels import K, SEED, fp, sha, vote, wjson
from pilot_vote_aware import vote_text

MODEL = "claude-sonnet-5-5"
OFF = os.path.join("runs", "offline_layer_stats")
DATASETS = {"Bean Leaf Lesions": "load_and_prepare_data_bean_leaf", "DeepWeeds": "load_and_prepare_data_DeepWeeds", "Soybean Seeds": "load_and_prepare_data_soybean_seeds",
            "Mango Leaf Disease": "load_and_prepare_data_mango_leaf", "Soybean Diseases": "load_and_prepare_data_Soybean_PNAS", "Dangerous Insects": "load_and_prepare_data_Soybean_Dangerous_Insects"}
CONDS = ["Z", "R", "A", "D", "O"]


async def main(inf):
    ap = argparse.ArgumentParser(); ap.add_argument("--pilot-arbab-eval", action="store_true")
    ap.add_argument("--stage", choices=["offline", "vlm", "report"], default="offline"); ap.add_argument("--out-dir", required=True)
    ap.add_argument("--datasets", nargs="+", default=list(DATASETS)); ap.add_argument("--max-requests", type=int, default=3000); ap.add_argument("--workers", type=int, default=4)
    args = ap.parse_args(); out = args.out_dir
    os.makedirs(os.path.join(out, "vlm", "responses"), exist_ok=True)
    api0 = inf.ClaudeHeadlessAPI(model=MODEL, sandbox_dir=os.path.join(out, "vlm", "sandbox"))
    settings = "claude -p defaults (temperature not settable); system prompt: " + api0.SYSTEM_PROMPT + "; command: " + " ".join(api0.command()[:-1])
    rows, reqs, prompts = [], {}, {}
    for ds in args.datasets:
        tag = ds.replace(" ", "_")
        data, classes, name = getattr(inf, DATASETS[ds])(100)
        classes = [str(c) for c in classes]; n = len(data)
        pool = pd.read_csv(os.path.join(OFF, f"pool_{tag}.csv"))
        assert [os.path.basename(p) for p in pool["path"]] == [os.path.basename(data.at[i, 0]) for i in range(n)], f"{ds}: pool differs from the cached pool"
        lab = [str(data.at[i, 1]) for i in range(n)]; assert lab == [str(x) for x in pool["label"]]
        E = np.load(os.path.join(OFF, "embeddings", tag + ".npz"))
        prompt = inf.universal_prompt.format(expected_classes=classes); prompts[ds] = prompt
        data[1] = lab
        img = [inf.load_image(data.at[i, 0]) for i in range(n)]
        assert all(x is not None for x in img)
        def block(i):
            return [{"type": "image", "source": {"type": "base64", "media_type": "image/jpeg", "data": img[i]}}, {"type": "text", "text": f'{{"prediction": "{lab[i]}"}}'}]
        rank = {}
        for enc in ("vit", "dinov2_b12"):
            emb = {i: E[enc][i].astype(np.float64) for i in range(n)}
            for q in range(n):
                rank[(enc, q)] = inf.rank_candidates(q, emb[q], emb)[:K]        # the repository's leave-one-out ranking
        for q in range(n):
            a_idx, o_idx = [i for i, _ in rank[("vit", q)]], [i for i, _ in rank[("dinov2_b12", q)]]
            r_idx = [int(i) for i in np.random.default_rng([SEED, q]).permutation([i for i in range(n) if i != q])[:K]]
            assert q not in a_idx + o_idx + r_idx and len(set(a_idx)) == len(set(o_idx)) == len(set(r_idx)) == K
            ex = lambda idx: [b for i in idx for b in block(i)]
            inp = {"Z": {"image": img[q], "examples": [], "prompt": prompt}, "R": {"image": img[q], "examples": ex(r_idx), "prompt": prompt},
                   "A": {"image": img[q], "examples": ex(a_idx), "prompt": prompt},
                   "D": {"image": img[q], "examples": ex(o_idx), "prompt": prompt},
                   "O": {"image": img[q], "examples": ex(o_idx) + [{"type": "text", "text": vote_text([lab[i] for i in o_idx])}], "prompt": prompt}}
            d = {"dataset": ds, "query_idx": q, "query_id": os.path.basename(data.at[q, 0]), "true_label": lab[q],
                 "A_labels": json.dumps([lab[i] for i in a_idx]), "O_labels": json.dumps([lab[i] for i in o_idx]), "R_labels": json.dumps([lab[i] for i in r_idx]),
                 "vote_vit": vote([lab[i] for i in a_idx], [s for _, s in rank[("vit", q)]]), "vote_dinov2": vote([lab[i] for i in o_idx], [s for _, s in rank[("dinov2_b12", q)]]),
                 "same_A": sum(lab[i] == lab[q] for i in a_idx), "same_O": sum(lab[i] == lab[q] for i in o_idx), "same_R": sum(lab[i] == lab[q] for i in r_idx)}
            for c in CONDS:
                k = sha(json.dumps({"vendor": "claude_headless", "model": MODEL, "settings": settings, "inputs": inp[c]}, sort_keys=True).encode())
                d[f"payload_{c}"] = k; reqs[k] = (ds, q, c, inp[c])
            rows.append(d)
        print(f"[offline] {ds}: pool {n}, classes {len(classes)}, ViT vote {sum(r['vote_vit'] == r['true_label'] for r in rows if r['dataset'] == ds)}/{n}, "
              f"DINOv2 vote {sum(r['vote_dinov2'] == r['true_label'] for r in rows if r['dataset'] == ds)}/{n}", flush=True)
    J = pd.DataFrame(rows)
    done = os.path.join(out, "offline_retrieval.csv")
    if os.path.exists(done):
        old = pd.read_csv(done); J = pd.concat([old[~old["dataset"].isin(args.datasets)], J], ignore_index=True)
    J.to_csv(done, index=False)
    pj = os.path.join(out, "prompts.json"); allp = json.load(open(pj)) if os.path.exists(pj) else {}; allp.update(prompts); wjson(pj, allp)
    wjson(os.path.join(out, "run_config.json"), {"model": MODEL, "settings": settings, "shots": K, "loader_samples": 100, "protocol": "AgEval: leave-one-out retrieval inside the loader pool, no source-group split",
                                                 "conditions": {"Z": "zero-shot", "R": "random demonstrations, standard prompt", "A": "original ViT pooler_output retrieval, standard prompt (Arshad et al.)",
                                                                "D": "DINOv2 block 12 CLS retrieval, standard prompt", "O": "DINOv2 block 12 CLS retrieval, vote-aware prompt (training-free)"}, "vote_aware_example": vote_text(["a", "a", "a", "b"]), "max_requests": args.max_requests})
    ledger = os.path.join(out, "vlm", "request_ledger.jsonl")
    def rpath(k): return os.path.join(out, "vlm", "responses", k + ".json")
    if args.stage == "vlm":
        used = [sum(1 for _ in open(ledger)) if os.path.exists(ledger) else 0]; lock = threading.Lock()
        todo = [k for k in reqs if not os.path.exists(rpath(k))]
        todo = [todo[i] for i in np.random.default_rng([SEED, 11]).permutation(len(todo))]
        todo.sort(key=lambda k: args.datasets.index(reqs[k][0]))      # dataset by dataset, conditions and queries interleaved inside
        def work(k):
            with lock:
                if used[0] >= args.max_requests: return
                used[0] += 1; nreq = used[0]
            ds, q, c, inp = reqs[k]; api = inf.ClaudeHeadlessAPI(model=MODEL, sandbox_dir=os.path.join(out, "vlm", "sandbox")); t0 = time.time()
            e = {"n": nreq, "t": t0, "dataset": ds, "query_idx": q, "condition": c, "payload_sha256": k}
            try:
                raw = asyncio.run(api.get_image_information(inp))
                wjson(rpath(k), {"raw": raw, "meta": api.last_meta, "model": MODEL, "dataset": ds, "query_idx": q, "condition": c})
                e.update(ok=True, elapsed_s=round(time.time() - t0, 2), model_ids=api.last_meta.get("model_ids"), num_turns=api.last_meta.get("num_turns"))
            except Exception as ex:
                e.update(ok=False, elapsed_s=round(time.time() - t0, 2), error=str(ex)[:300])
            with lock:
                open(ledger, "a").write(json.dumps(e) + "\n")
                if not e["ok"] or nreq % 100 == 0: print(f"[{nreq}/{args.max_requests}] {ds} q{q} {c}: {'ok' if e['ok'] else 'FAILED ' + e['error'][:120]}", flush=True)
        for attempt in (1, 2):
            pend = [k for k in todo if not os.path.exists(rpath(k))]
            with ThreadPoolExecutor(args.workers) as pool_:
                list(pool_.map(work, pend))
        print(f"requests used: {used[0]} of {args.max_requests}", flush=True)
    if args.stage == "offline":
        return
    # ---- report (all datasets present in offline_retrieval.csv)
    led = [json.loads(l) for l in open(ledger)] if os.path.exists(ledger) else []
    def pred(k):
        if not os.path.exists(rpath(k)): return None
        try: return str(inf.extract_json(json.load(open(rpath(k)))["raw"])["prediction"])
        except Exception: return "INVALID"
    for c in CONDS:
        J[f"pred_{c}"] = [pred(k) for k in J[f"payload_{c}"]]
    J["true_label"] = J["true_label"].astype(str); J["vote_vit"] = J["vote_vit"].astype(str); J["vote_dinov2"] = J["vote_dinov2"].astype(str)
    J["complete"] = J[[f"pred_{c}" for c in CONDS]].notna().all(axis=1)
    J.to_csv(os.path.join(out, "per_query_results.csv"), index=False)
    COL = {"Z": "pred_Z", "R": "pred_R", "A": "pred_A", "D": "pred_D", "O": "pred_O", "VV": "vote_vit", "VD": "vote_dinov2"}
    NAME = {"Z": "Zero-shot", "R": "Random 4-shot", "VV": "ViT 4-NN vote (no VLM)", "VD": "DINOv2 4-NN vote (no VLM)", "A": "Assisted 4-shot, ViT (Arshad et al.)", "D": "Assisted 4-shot, DINOv2, standard prompt", "O": "**Ours: DINOv2, vote-aware prompt**"}
    ORDER = ["Z", "R", "VV", "VD", "A", "D", "O"]
    def mcn(a, b):
        w, l = int((a & ~b).sum()), int((~a & b).sum()); return w, l, (1.0 if w + l == 0 else float(stats.binomtest(w, w + l, 0.5).pvalue))
    met, T = {"requests_sent": len(led), "requests_failed": sum(not r["ok"] for r in led), "datasets": {}}, []
    T += ["# Assisted few-shot learning with the AgEval protocol\n",
          f"Loader pools of about 100 images (seed 42), leave-one-out retrieval inside the pool, 4 shots, VLM `{MODEL}` through headless `claude -p`. Nothing is trained. "
          f"Requests sent: {len(led)}, failed: {met['requests_failed']}. This protocol has no source-group split.\n",
          "## Accuracy (macro-F1) per dataset, on queries complete in all four prompts\n", "| Dataset | n | " + " | ".join(NAME[c] for c in ORDER) + " |", "|---|---|" + "---|" * len(ORDER)]
    tot = {c: [0, 0] for c in ORDER}; comp = []
    for ds, g0 in J.groupby("dataset", sort=False):
        g = g0[g0["complete"]]
        if not len(g): continue
        labs = sorted(g0["true_label"].unique()); cor = {c: (g[COL[c]] == g["true_label"]) for c in ORDER}
        sc = {c: {"correct": int(cor[c].sum()), "macro_f1": float(f1_score(g["true_label"], g[COL[c]].astype(str), labels=labs, average="macro", zero_division=0))} for c in ORDER}
        best = max(ORDER, key=lambda c: sc[c]["correct"])
        T.append(f"| {ds} | {len(g)}/{len(g0)} | " + " | ".join(("**" if c == best else "") + f"{100 * sc[c]['correct'] / len(g):.0f}% ({sc[c]['macro_f1']:.2f})" + ("**" if c == best else "") for c in ORDER) + " |")
        for c in ORDER: tot[c][0] += sc[c]["correct"]; tot[c][1] += len(g)
        vcA, vcO = g[cor["VV"]], g[cor["VD"]]
        d = {"n": len(g), "pool": len(g0), "scores": sc, "same_class": {"A": float(g["same_A"].mean()), "O": float(g["same_O"].mean()), "R": float(g["same_R"].mean())},
             "O_vs_A": mcn(cor["O"], cor["A"]), "O_vs_D": mcn(cor["O"], cor["D"]), "D_vs_A": mcn(cor["D"], cor["A"]), "D_vs_VD": mcn(cor["D"], cor["VD"]),
             "D_overturned": [int((~cor["D"][cor["VD"]]).sum()), int(cor["VD"].sum())], "D_corrected": [int(cor["D"][~cor["VD"]].sum()), int((~cor["VD"]).sum())], "O_vs_R": mcn(cor["O"], cor["R"]), "O_vs_Z": mcn(cor["O"], cor["Z"]), "O_vs_VD": mcn(cor["O"], cor["VD"]), "A_vs_VV": mcn(cor["A"], cor["VV"]), "A_vs_R": mcn(cor["A"], cor["R"]),
             "A_overturned": [int((~cor["A"][cor["VV"]]).sum()), int(cor["VV"].sum())], "O_overturned": [int((~cor["O"][cor["VD"]]).sum()), int(cor["VD"].sum())],
             "A_corrected": [int(cor["A"][~cor["VV"]].sum()), int((~cor["VV"]).sum())], "O_corrected": [int(cor["O"][~cor["VD"]].sum()), int((~cor["VD"]).sum())]}
        met["datasets"][ds] = d; comp.append((ds, d))
    if tot["Z"][1]:
        T.append("| **All queries** | " + str(tot["Z"][1]) + " | " + " | ".join(f"{100 * tot[c][0] / tot[c][1]:.1f}%" for c in ORDER) + " |")
        met["pooled_accuracy"] = {c: tot[c][0] / tot[c][1] for c in ORDER}
    T += ["\n## Ours versus the baselines (wins / losses, exact McNemar p)\n", "| Dataset | vs Assisted ViT (Arshad et al.) | vs DINOv2 standard prompt | vs Random | vs Zero-shot | vs DINOv2 vote |", "|---|---|---|---|---|---|"]
    for ds, d in comp:
        T.append(f"| {ds} | " + " | ".join(f"{d[k][0]} / {d[k][1]}, p {fp(d[k][2])}" for k in ("O_vs_A", "O_vs_D", "O_vs_R", "O_vs_Z", "O_vs_VD")) + " |")
    T += ["\n## Votes overturned and corrected\n", "| Dataset | Same-class refs (ViT / DINOv2) | ViT standard overturns correct ViT votes | DINOv2 standard overturns correct DINOv2 votes | Ours overturns correct DINOv2 votes | ViT standard corrects wrong votes | DINOv2 standard corrects wrong votes | Ours corrects wrong votes |", "|---|---|---|---|---|---|---|---|"]
    for ds, d in comp:
        T.append(f"| {ds} | {d['same_class']['A']:.2f} / {d['same_class']['O']:.2f} | {d['A_overturned'][0]}/{d['A_overturned'][1]} | {d['D_overturned'][0]}/{d['D_overturned'][1]} | {d['O_overturned'][0]}/{d['O_overturned'][1]} | {d['A_corrected'][0]}/{d['A_corrected'][1]} | {d['D_corrected'][0]}/{d['D_corrected'][1]} | {d['O_corrected'][0]}/{d['O_corrected'][1]} |")
    wjson(os.path.join(out, "metrics.json"), met)
    notes = os.path.join(out, "NOTES.md")
    if os.path.exists(notes): T.append("\n" + open(notes).read())
    open(os.path.join(out, "SUMMARY.md"), "w").write("\n".join(T) + "\n"); print("\n".join(T))
