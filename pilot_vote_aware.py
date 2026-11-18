"""Can a VLM with retrieved examples beat the 4-NN vote on DeepWeeds? Development-gated test.

Reuses the frozen queries, bank and retrieved references of runs/deepweeds_demo_value_seed42 (read only).
Candidates, all with the same four DINOv2 references:
  opus_S   = claude-opus-5-5, standard four-image prompt
  sonnet_V = claude-sonnet-5-5, vote-aware prompt
  opus_V   = claude-opus-5-5, vote-aware prompt
Rule (written before any call): a candidate qualifies if it is correct on more development queries than the
4-NN vote. The qualifying candidate with the most correct answers is locked (ties: sonnet_V, opus_S, opus_V)
and is the only one sent the evaluation queries. If none qualifies, evaluation does not run.

Run through `python inference.py --pilot-vote-aware --stage {offline,dev,eval,report} --out-dir <dir>`.
"""
import argparse, json, os, time
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.metrics import f1_score
from pilot_matched_labels import K, SEED, fp, md5_file, path_of, sha, wjson

BASE = os.path.join("runs", "deepweeds_matched_seed42")
DV = os.path.join("runs", "deepweeds_demo_value_seed42")
CANDS = {"sonnet_V": ("claude-sonnet-5-5", "V"), "opus_S": ("claude-opus-5-5", "S"), "opus_V": ("claude-opus-5-5", "V")}
ORDER = ["sonnet_V", "opus_S", "opus_V"]


def vote_text(labels):
    order = sorted(dict.fromkeys(labels), key=lambda l: (-labels.count(l), labels.index(l)))
    counts = ", ".join(f'"{l}": {labels.count(l)}' for l in order)
    s = (f"The {len(labels)} labeled reference images above were retrieved because they look most similar to the image to classify. "
         f"Label counts among them: {counts}. ")
    if len(order) == 1 or labels.count(order[0]) > labels.count(order[1]):
        s += (f'The most frequent reference label is "{order[0]}". Use it as the default answer. '
              "Choose a different class only if the image clearly shows a different class.")
    else:
        s += "The reference labels do not agree on one class. Decide from the image and the reference images."
    return s


async def main(inf):
    ap = argparse.ArgumentParser()
    ap.add_argument("--pilot-vote-aware", action="store_true"); ap.add_argument("--stage", choices=["offline", "dev", "eval", "report"], default="offline")
    ap.add_argument("--out-dir", required=True); ap.add_argument("--max-requests", type=int, default=210)
    args = ap.parse_args(); out = args.out_dir
    os.makedirs(os.path.join(out, "vlm", "responses"), exist_ok=True)
    cfg = json.load(open(os.path.join(BASE, "run_config.json"))); expected = cfg["expected_classes"]
    rm = pd.read_csv(os.path.join(BASE, "reference_manifest.csv"), dtype={"capture_date": str})
    J = pd.read_csv(os.path.join(DV, "per_query_results.csv"))
    panel = pd.read_csv(os.path.join("runs", "deepweeds_split_audit", "query_panel.csv")).set_index("query_id")
    assert all(md5_file(path_of(f)) == panel.at[f, "md5"] for f in J["query_id"]) and all(md5_file(path_of(f)) == m for f, m in zip(rm["ref_id"], rm["md5"]))
    prompt = inf.universal_prompt.format(expected_classes=expected); assert sha(prompt.encode()) == cfg["prompt_sha256"]; inf.vision_prompt = prompt
    bank = pd.DataFrame({0: [path_of(f) for f in rm["ref_id"]], 1: list(rm["label"])}); ridx = {f: i for i, f in enumerate(rm["ref_id"])}
    apis = {}
    for m in {m for m, _ in CANDS.values()}:
        a = inf.ClaudeHeadlessAPI(model=m, sandbox_dir=os.path.join(out, "vlm", "sandbox"))
        a.settings = "claude -p defaults (temperature not settable); system prompt: " + a.SYSTEM_PROMPT + "; command: " + " ".join(a.command()[:-1]); apis[m] = a
    inputs = {}
    for _, w in J.iterrows():
        refs, labs = json.loads(w["U_refs"]), json.loads(w["U_labels"])
        ex, _, cats, _ = inf.build_examples(apis["claude-opus-5-5"], bank, [ridx[r] for r in refs], None); assert cats == labs
        q64 = inf.load_image(path_of(w["query_id"]))
        inputs[(w["query_id"], "S")] = {"image": q64, "examples": ex, "prompt": prompt}
        inputs[(w["query_id"], "V")] = {"image": q64, "examples": ex + [{"type": "text", "text": vote_text(labs)}], "prompt": prompt}
        # the standard Sonnet payload must equal the one already answered in the base run
        assert sha(json.dumps({"vendor": "claude_headless", "model": "claude-sonnet-5-5", "settings": apis["claude-sonnet-5-5"].settings, "inputs": inputs[(w["query_id"], "S")]}, sort_keys=True).encode()) == w["payload_U"]
    def key(c, f):
        m, p = CANDS[c]
        return sha(json.dumps({"vendor": "claude_headless", "model": m, "settings": apis[m].settings, "inputs": inputs[(f, p)]}, sort_keys=True).encode())
    prereg = os.path.join(out, "preregistration.json"); ledger = os.path.join(out, "vlm", "request_ledger.jsonl"); lock = os.path.join(out, "locked_candidate.json")
    if not os.path.exists(prereg):
        wjson(prereg, {"written_before_any_vlm_call": not os.path.exists(ledger), "written_at": time.strftime("%Y-%m-%d %H:%M:%S"),
                       "candidates": {c: {"model": m, "prompt": "standard four-image prompt" if p == "S" else "vote-aware prompt"} for c, (m, p) in CANDS.items()},
                       "vote_aware_text_examples": [vote_text(["a", "a", "a", "b"]), vote_text(["a", "a", "b", "b"])],
                       "rule": "a candidate qualifies if it is correct on more development queries than the 4-NN vote; the qualifying candidate with the most correct answers is locked "
                               "(ties in the order sonnet_V, opus_S, opus_V) and is the only one sent the evaluation queries; if none qualifies, evaluation does not run",
                       "evaluation_primary": "locked candidate versus the 4-NN vote on the 63 evaluation queries, exact McNemar test",
                       "note": "three candidates are compared on 27 development queries; the evaluation result of the locked candidate is the only confirmatory number",
                       "dev_vote_correct": int(J[J.role == "dev"]["correct_KNN"].sum()), "n_dev": int((J.role == "dev").sum())})
    print("example vote-aware text:", vote_text(json.loads(J["U_labels"].iloc[0])))
    if args.stage == "offline":
        return
    def resp(k):
        p = os.path.join(out, "vlm", "responses", k + ".json"); return json.load(open(p)) if os.path.exists(p) else None
    def pred(k):
        r = resp(k)
        if r is None: return None
        try: return inf.extract_json(r["raw"])["prediction"]
        except Exception: return "INVALID"
    async def run(role, cands):
        used = sum(1 for _ in open(ledger)) if os.path.exists(ledger) else 0
        plan = [(c, f) for f in J[J.role == role]["query_id"] for c in cands]
        plan = [plan[i] for i in np.random.default_rng([SEED, 7 if role == "dev" else 8]).permutation(len(plan))]
        for attempt in (1, 2):
            for c, f in plan:
                k = key(c, f)
                if resp(k) is not None: continue
                if used >= args.max_requests: print("request cap reached"); continue
                used += 1; t0 = time.time(); m = CANDS[c][0]
                e = {"n": used, "t": t0, "stage": role, "candidate": c, "model": m, "query_id": f, "payload_sha256": k, "attempt": attempt}
                try:
                    raw = await apis[m].get_image_information(inputs[(f, CANDS[c][1])])
                    wjson(os.path.join(out, "vlm", "responses", k + ".json"), {"raw": raw, "meta": apis[m].last_meta, "model": m, "query_id": f, "candidate": c})
                    e.update(ok=True, elapsed_s=round(time.time() - t0, 2), meta=apis[m].last_meta)
                except Exception as ex:
                    e.update(ok=False, elapsed_s=round(time.time() - t0, 2), error=str(ex)[:400])
                open(ledger, "a").write(json.dumps(e) + "\n")
                if not e["ok"] or used % 20 == 0: print(f"[{used}/{args.max_requests}] {role} {c} {f}: {'ok' if e['ok'] else 'FAILED ' + e['error'][:80]}", flush=True)
        print(f"{role}: requests used so far {used} of {args.max_requests}")
    D = J[J.role == "dev"]
    if args.stage == "dev":
        assert not os.path.exists(lock) or True
        await run("dev", ORDER)
        if not os.path.exists(lock):
            corr = {c: int(sum(pred(key(c, f)) == t for f, t in zip(D["query_id"], D["true_label"]))) for c in ORDER}
            assert all(pred(key(c, f)) is not None for c in ORDER for f in D["query_id"]), "dev incomplete; rerun --stage dev"
            vote = int(D["correct_KNN"].sum()); qual = [c for c in ORDER if corr[c] > vote]
            wjson(lock, {"dev_correct": corr, "dev_vote_correct": vote, "dev_sonnet_standard_correct": int(D["correct_U"].sum()), "dev_sonnet_zero_shot_correct": int(D["correct_Z"].sum()), "n_dev": len(D),
                         "qualifying": qual, "locked": (max(qual, key=lambda c: (corr[c], -ORDER.index(c))) if qual else None), "locked_at": time.strftime("%Y-%m-%d %H:%M:%S")})
        print("locked:", json.load(open(lock)))
    if not os.path.exists(lock):
        return
    L_ = json.load(open(lock)); cand = L_["locked"]
    if args.stage == "eval":
        if cand is None:
            print("no candidate beat the vote on development; evaluation does not run")
        else:
            await run("eval", [cand])
    # ---- report
    led = [json.loads(l) for l in open(ledger)] if os.path.exists(ledger) else []
    rows = []
    for _, w in J.iterrows():
        d = {k: w[k] for k in ("query_id", "role", "true_label", "U_labels", "knn", "margin", "pred_Z", "pred_U", "correct_Z", "correct_U", "correct_KNN")}
        for c in ORDER:
            d[f"pred_{c}"] = pred(key(c, w["query_id"])); d[f"correct_{c}"] = d[f"pred_{c}"] == w["true_label"]
        rows.append(d)
    R = pd.DataFrame(rows); R.to_csv(os.path.join(out, "per_query_results.csv"), index=False)
    def block(g, c):
        ok = g[f"correct_{c}"]; vc, vw = g[g.correct_KNN], g[~g.correct_KNN]
        w_, l_ = int((ok & ~g.correct_KNN).sum()), int((~ok & g.correct_KNN).sum())
        return {"correct": int(ok.sum()), "n": len(g), "macro_f1": float(f1_score(g.true_label, g[f"pred_{c}"].astype(str), labels=sorted(expected), average="macro", zero_division=0)),
                "overturned": int((~vc[f"correct_{c}"]).sum()), "vote_correct": len(vc), "corrected": int(vw[f"correct_{c}"].sum()), "vote_wrong": len(vw),
                "equals_vote": int((g[f"pred_{c}"] == g.knn).sum()), "vs_vote": {"wins": w_, "losses": l_, "p": 1.0 if w_ + l_ == 0 else float(stats.binomtest(w_, w_ + l_, 0.5).pvalue)}}
    met = {"lock": L_, "requests_sent": len(led), "requests_failed": sum(not r["ok"] for r in led), "model_ids": sorted({x for r in led if r.get("ok") for x in r["meta"]["model_ids"]}),
           "dev": {c: block(R[R.role == "dev"], c) for c in ORDER}}
    E = R[R.role == "eval"]
    if cand and E[f"pred_{cand}"].notna().all():
        met["eval"] = {cand: block(E, cand), "n": len(E), "vote_correct": int(E.correct_KNN.sum()), "sonnet_standard_correct": int(E.correct_U.sum()), "sonnet_zero_shot_correct": int(E.correct_Z.sum())}
        a, b = E[f"correct_{cand}"], E["correct_U"]; w_, l_ = int((a & ~b).sum()), int((~a & b).sum())
        met["eval"]["vs_sonnet_standard"] = {"wins": w_, "losses": l_, "p": 1.0 if w_ + l_ == 0 else float(stats.binomtest(w_, w_ + l_, 0.5).pvalue)}
    wjson(os.path.join(out, "metrics.json"), met)
    NAME = {"sonnet_V": "Sonnet 5.5, vote-aware prompt", "opus_S": "Opus 5.5, standard prompt", "opus_V": "Opus 5.5, vote-aware prompt"}
    T = ["# DeepWeeds: can a VLM with retrieved examples beat the 4-NN vote?\n",
         f"Same queries, bank and DINOv2 references as `deepweeds_demo_value_seed42`. Requests sent: {len(led)}, failed: {met['requests_failed']}. Response model IDs: {', '.join(met['model_ids'])}.\n",
         f"## Development queries (27)\n\nThe 4-NN vote is correct on {L_['dev_vote_correct']}/27. Sonnet 5.5 with the standard prompt is correct on {L_['dev_sonnet_standard_correct']}/27 and zero-shot on {L_['dev_sonnet_zero_shot_correct']}/27 (from the earlier run).\n",
         "| Candidate | Correct | Macro-F1 | Correct votes overturned | Wrong votes corrected | Same label as the vote | Wins / losses versus the vote |", "|---|---|---|---|---|---|---|"]
    for c in ORDER:
        b = met["dev"][c]
        T.append(f"| {NAME[c]} | {b['correct']}/{b['n']} | {b['macro_f1']:.3f} | {b['overturned']}/{b['vote_correct']} | {b['corrected']}/{b['vote_wrong']} | {b['equals_vote']}/{b['n']} | {b['vs_vote']['wins']} / {b['vs_vote']['losses']} (p {fp(b['vs_vote']['p'])}) |")
    T.append(f"\nQualifying candidates (more correct than the vote): {', '.join(NAME[c] for c in L_['qualifying']) or 'none'}. Locked for evaluation: {NAME[cand] if cand else 'none'}.\n")
    if "eval" in met:
        b = met["eval"][cand]
        T += ["## Evaluation queries (63), locked candidate only\n", "| Method | Correct | Macro-F1 |", "|---|---|---|",
              f"| Sonnet 5.5, zero-shot (earlier run) | {met['eval']['sonnet_zero_shot_correct']}/63 | |", f"| Sonnet 5.5, standard prompt (earlier run) | {met['eval']['sonnet_standard_correct']}/63 | |",
              f"| 4-NN vote | {met['eval']['vote_correct']}/63 | |", f"| **{NAME[cand]}** | **{b['correct']}/63** | {b['macro_f1']:.3f} |",
              f"\nLocked candidate versus the vote: {b['vs_vote']['wins']} wins, {b['vs_vote']['losses']} losses, exact McNemar p {fp(b['vs_vote']['p'])}. "
              f"It overturns {b['overturned']} of {b['vote_correct']} correct votes, corrects {b['corrected']} of {b['vote_wrong']} wrong votes, and gives the vote's label on {b['equals_vote']} of 63 queries. "
              f"Versus Sonnet 5.5 with the standard prompt: {met['eval']['vs_sonnet_standard']['wins']} wins, {met['eval']['vs_sonnet_standard']['losses']} losses (p {fp(met['eval']['vs_sonnet_standard']['p'])}).\n"]
    elif cand is None:
        T.append("Evaluation did not run, because no candidate beat the vote on development.\n")
    open(os.path.join(out, "SUMMARY.md"), "w").write("\n".join(T) + "\n"); print("\n".join(T))
