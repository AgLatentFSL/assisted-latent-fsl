"""Held-out test of the vote-aware prompt on Paddy Doctor (variety-disjoint split). Nothing is trained.

Reuses the frozen queries, bank and DINOv2 references of runs/paddy_replication_seed42 (read only) and its cached
zero-shot (Z), standard (S) and label-count (C) answers. Adds V = the vote-aware prompt of pilot_vote_aware.py,
which was written and tried on DeepWeeds development queries only.

Run through `python inference.py --pilot-paddy-vote-aware --stage {offline,vlm,report} --out-dir <dir>`.
"""
import argparse, json, os, time
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.metrics import f1_score
from pilot_matched_labels import K, SEED, fp, md5_file, sha, wjson
from pilot_replication import MODELS, img_path
from pilot_vote_aware import vote_text

BASE = os.path.join("runs", "paddy_replication_seed42")


async def main(inf):
    ap = argparse.ArgumentParser(); ap.add_argument("--pilot-paddy-vote-aware", action="store_true")
    ap.add_argument("--stage", choices=["offline", "vlm", "report"], default="offline"); ap.add_argument("--out-dir", required=True); ap.add_argument("--max-requests", type=int, default=110)
    args = ap.parse_args(); out = args.out_dir
    os.makedirs(os.path.join(out, "vlm", "responses"), exist_ok=True)
    cfg = json.load(open(os.path.join(BASE, "run_config.json"))); classes = cfg["expected_classes"]
    qm = pd.read_csv(os.path.join(BASE, "query_manifest.csv"), dtype=str); rm = pd.read_csv(os.path.join(BASE, "reference_manifest.csv"), dtype=str)
    J = pd.read_csv(os.path.join(BASE, "offline_retrieval.csv")); B = pd.read_csv(os.path.join(BASE, "per_query_results.csv"), keep_default_na=False)
    assert all(md5_file(img_path(l, i)) == m for l, i, m in zip(qm["label"], qm["query_id"], qm["md5"])) and all(md5_file(img_path(l, i)) == m for l, i, m in zip(rm["label"], rm["ref_id"], rm["md5"]))
    assert not set(qm["group"]) & set(rm["group"])
    prompt = inf.universal_prompt.format(expected_classes=classes); assert sha(prompt.encode()) == cfg["prompt_sha256"]; inf.vision_prompt = prompt
    apis = {}
    for m in MODELS:
        a = inf.ClaudeHeadlessAPI(model=m, sandbox_dir=os.path.join(out, "vlm", "sandbox"))
        a.settings = "claude -p defaults (temperature not settable); system prompt: " + a.SYSTEM_PROMPT + "; command: " + " ".join(a.command()[:-1])
        assert a.settings == cfg["settings"][m]; apis[m] = a
    bank = pd.DataFrame({0: [img_path(l, i) for l, i in zip(rm["label"], rm["ref_id"])], 1: list(rm["label"])}); ridx = {f: i for i, f in enumerate(rm["ref_id"])}
    inputs = {}
    for _, w in J.iterrows():
        refs, labs = json.loads(w["refs"]), json.loads(w["ref_labels"])
        ex, _, cats, _ = inf.build_examples(apis[MODELS[0]], bank, [ridx[r] for r in refs], None); assert cats == labs
        inputs[w["query_id"]] = {"image": inf.load_image(img_path(w["true_label"], w["query_id"])), "examples": ex + [{"type": "text", "text": vote_text(labs)}], "prompt": prompt}
    def key(m, f):
        return sha(json.dumps({"vendor": "claude_headless", "model": m, "settings": apis[m].settings, "inputs": inputs[f]}, sort_keys=True).encode())
    prereg = os.path.join(out, "preregistration.json"); ledger = os.path.join(out, "vlm", "request_ledger.jsonl")
    if not os.path.exists(prereg):
        wjson(prereg, {"written_before_any_vlm_call": not os.path.exists(ledger), "written_at": time.strftime("%Y-%m-%d %H:%M:%S"), "models": MODELS,
                       "V": "vote-aware prompt: the standard four-image prompt plus one text block after the four pairs", "vote_aware_text_examples": [vote_text(["a", "a", "a", "b"]), vote_text(["a", "a", "b", "b"])],
                       "provenance": "the vote-aware text was written and tried on 27 DeepWeeds development queries only; no Paddy Doctor query was used to design it",
                       "primary": "per model, V versus the 4-NN vote on the 50 queries, exact McNemar test",
                       "secondary": ["per model: V versus the standard prompt S, versus the label-count prompt C and versus zero-shot Z (cached answers of the base run)",
                                     "per model: correct votes overturned and wrong votes corrected by V"], "planned_requests": len(MODELS) * len(J)})
    print("example:", vote_text(json.loads(J["ref_labels"].iloc[1])))
    if args.stage == "offline":
        return
    def resp(k):
        p = os.path.join(out, "vlm", "responses", k + ".json"); return json.load(open(p)) if os.path.exists(p) else None
    if args.stage == "vlm":
        used = sum(1 for _ in open(ledger)) if os.path.exists(ledger) else 0
        plan = [(m, f) for m in MODELS for f in J["query_id"]]; plan = [plan[i] for i in np.random.default_rng([SEED, 9]).permutation(len(plan))]
        for attempt in (1, 2):
            for m, f in plan:
                k = key(m, f)
                if resp(k) is not None: continue
                if used >= args.max_requests: print("request cap reached"); continue
                used += 1; t0 = time.time(); e = {"n": used, "t": t0, "model": m, "query_id": f, "condition": "V", "payload_sha256": k, "attempt": attempt}
                try:
                    raw = await apis[m].get_image_information(inputs[f])
                    wjson(os.path.join(out, "vlm", "responses", k + ".json"), {"raw": raw, "meta": apis[m].last_meta, "model": m, "query_id": f, "condition": "V"})
                    e.update(ok=True, elapsed_s=round(time.time() - t0, 2), meta=apis[m].last_meta)
                except Exception as ex:
                    e.update(ok=False, elapsed_s=round(time.time() - t0, 2), error=str(ex)[:400])
                open(ledger, "a").write(json.dumps(e) + "\n")
                if not e["ok"] or used % 25 == 0: print(f"[{used}/{args.max_requests}] {m} {f}: {'ok' if e['ok'] else 'FAILED ' + e['error'][:80]}", flush=True)
        print(f"requests used: {used} of {args.max_requests}")
    led = [json.loads(l) for l in open(ledger)] if os.path.exists(ledger) else []
    rows = []
    for _, b in B.iterrows():
        r = resp(key(b["model"], b["query_id"])); p = None
        if r is not None:
            try: p = inf.extract_json(r["raw"])["prediction"]
            except Exception: p = "INVALID"
        d = {k: b[k] for k in ("model", "query_id", "true_label", "ref_labels", "knn", "margin", "pred_Z", "pred_S", "pred_C")}
        d["pred_V"] = p; d["status_V"] = "missing" if p is None else ("valid" if p in classes else "invalid")
        rows.append(d)
    R = pd.DataFrame(rows)
    for c, col in (("KNN", "knn"), ("Z", "pred_Z"), ("S", "pred_S"), ("C", "pred_C"), ("V", "pred_V")):
        R[f"correct_{c}"] = R[col] == R["true_label"]
    R.to_csv(os.path.join(out, "per_query_results.csv"), index=False)
    def mcn(a, b):
        w, l = int((a & ~b).sum()), int((~a & b).sum()); return {"wins": w, "losses": l, "p": 1.0 if w + l == 0 else float(stats.binomtest(w, w + l, 0.5).pvalue)}
    met = {"requests_sent": len(led), "requests_failed": sum(not r["ok"] for r in led), "models": {}}
    NAME = {"Z": "Zero-shot", "KNN": "DINOv2 4-NN vote (no VLM)", "S": "Standard four-image prompt", "C": "Four-image prompt with label counts", "V": "**Vote-aware prompt (ours, no training)**"}
    T = ["# Paddy Doctor: held-out test of the vote-aware prompt\n",
         f"Same 50 queries, 110-image variety-disjoint bank and DINOv2 references as `paddy_replication_seed42`. Nothing is trained: the retriever and the VLMs are frozen. "
         f"New requests: {len(led)}, failed: {met['requests_failed']}. Zero-shot, standard and label-count answers come from the base run.\n"]
    for m in MODELS:
        g = R[(R.model == m) & (R.status_V == "valid")]; vc, vw = g[g.correct_KNN], g[~g.correct_KNN]
        d = {"valid": len(g), "n": int((R.model == m).sum()), "model_ids": sorted({x for r in led if r.get("ok") and r["model"] == m for x in r["meta"]["model_ids"]}),
             "scores": {c: {"correct": int(g[f"correct_{c}"].sum()), "macro_f1": float(f1_score(g.true_label, g[{"KNN": "knn"}.get(c, "pred_" + c)].astype(str), labels=classes, average="macro", zero_division=0))} for c in ("Z", "KNN", "S", "C", "V")},
             "overturned": {c: int((~vc[f"correct_{c}"]).sum()) for c in ("S", "C", "V")}, "corrected": {c: int(vw[f"correct_{c}"].sum()) for c in ("S", "C", "V")}, "vote_correct": len(vc), "vote_wrong": len(vw),
             "equals_vote": {c: int((g["pred_" + c] == g.knn).sum()) for c in ("S", "C", "V")},
             "V_vs_KNN": mcn(g.correct_V, g.correct_KNN), "V_vs_S": mcn(g.correct_V, g.correct_S), "V_vs_C": mcn(g.correct_V, g.correct_C), "V_vs_Z": mcn(g.correct_V, g.correct_Z)}
        met["models"][m] = d
        if not len(g): continue
        T += [f"## {m}\n", f"Valid vote-aware answers: {d['valid']} of {d['n']}.\n", "| Method | Correct | Accuracy | Macro-F1 | Wrong votes corrected | Correct votes overturned | Same label as the vote |", "|---|---|---|---|---|---|---|"]
        for c in ("Z", "KNN", "S", "C", "V"):
            s = d["scores"][c]; ex = (f"{d['corrected'][c]}/{d['vote_wrong']} | {d['overturned'][c]}/{d['vote_correct']} | {d['equals_vote'][c]}/{len(g)}" if c in ("S", "C", "V") else " | | ")
            T.append(f"| {NAME[c]} | {s['correct']}/{len(g)} | {100 * s['correct'] / len(g):.0f}% | {s['macro_f1']:.3f} | {ex} |")
        T += ["\n| Paired comparison | Wins | Losses | Exact McNemar p |", "|---|---|---|---|"]
        T += [f"| {nm} | {d[k]['wins']} | {d[k]['losses']} | {fp(d[k]['p'])} |" for k, nm in (("V_vs_KNN", "Vote-aware versus 4-NN vote (primary)"), ("V_vs_S", "Vote-aware versus standard prompt"), ("V_vs_C", "Vote-aware versus label counts"), ("V_vs_Z", "Vote-aware versus zero-shot"))]
        T.append("")
    wjson(os.path.join(out, "metrics.json"), met)
    open(os.path.join(out, "SUMMARY.md"), "w").write("\n".join(T) + "\n"); print("\n".join(T))
