"""Do retrieved demonstrations add value beyond nearest-neighbour classification under capture-date shift?

DeepWeeds, fresh date-disjoint queries (dev and eval roles of the split-audit panel), the frozen
99-image bank of the base matched-label run. Conditions on every query:
  Z   = Claude zero-shot
  KNN = DINOv2 block 12 4-NN vote (no VLM)
  U   = Claude with DINOv2's top-4 references and their labels
Retrieval confidence (fixed before any call): vote margin = top class count minus second class count
among the top 4. A cutoff for "use U when margin >= tau, else Z" is fitted on dev and locked before eval.

Run through `python inference.py --pilot-demo-value --stage {offline,dev,eval,report} --out-dir <dir>`.
"""
import argparse, json, os, time
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.metrics import f1_score
from pilot_matched_labels import K, SEED, fp, md5_file, path_of, sha, vote, wjson

BASE = os.path.join("runs", "deepweeds_matched_seed42")
AUDIT = os.path.join("runs", "deepweeds_split_audit")
TAUS = [0, 1, 2, 3, 4, 5]          # 0 = always U, 5 = always Z
LOW_MARGIN_MAX = 1                 # fixed grouping for the confidence analysis: low = margin <= 1, high = margin >= 2


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--pilot-demo-value", action="store_true")
    p.add_argument("--stage", choices=["offline", "dev", "eval", "report"], default="offline")
    p.add_argument("--out-dir", required=True)
    p.add_argument("--max-requests", type=int, default=190)
    return p.parse_args()


def margin(labels):
    c = sorted(pd.Series(labels).value_counts().tolist(), reverse=True)
    return c[0] - (c[1] if len(c) > 1 else 0)


def gated(tau, m, u, z):
    return u if m >= tau else z


async def main(inf):
    args = parse_args()
    out = args.out_dir
    os.makedirs(os.path.join(out, "vlm", "responses"), exist_ok=True)
    cfg = json.load(open(os.path.join(BASE, "run_config.json")))
    expected, model = cfg["expected_classes"], cfg["model"]
    panel = pd.read_csv(os.path.join(AUDIT, "query_panel.csv"), dtype={"capture_date": str})
    qm = panel[panel["role"].isin(["dev", "eval"])].reset_index(drop=True)
    rm = pd.read_csv(os.path.join(BASE, "reference_manifest.csv"), dtype={"capture_date": str})
    used_before = set(pd.read_csv(os.path.join(BASE, "query_manifest.csv"))["query_id"]) | \
        set(pd.read_csv(os.path.join("runs", "deepweeds_seed42", "reference_manifest.csv"))["image_id"].str.split("/").str[-1])

    # ---- verification ---------------------------------------------------------
    from split_audit import dhash, NEAR_HAMMING
    hq = np.stack([dhash(path_of(f)) for f in qm["query_id"]]); hb = np.stack([dhash(path_of(f)) for f in rm["ref_id"]])
    ham = np.unpackbits(hq[:, None, :] ^ hb[None, :, :], axis=2).sum(axis=2)
    checks = {"query_md5_match": all(md5_file(path_of(f)) == m for f, m in zip(qm["query_id"], qm["md5"])),
              "bank_md5_match": all(md5_file(path_of(f)) == m for f, m in zip(rm["ref_id"], rm["md5"])),
              "shared_capture_dates": len(set(qm["capture_date"]) & set(rm["capture_date"])),
              "shared_images": len(set(qm["query_id"]) & set(rm["ref_id"])) + len(set(qm["md5"]) & set(rm["md5"])),
              "queries_seen_by_vlm_before": len(set(qm["query_id"]) & used_before),
              "min_dhash_distance_query_to_bank": int(ham.min()), "near_duplicate_pairs": int((ham <= NEAR_HAMMING).sum()),
              "n_dev": int((qm["role"] == "dev").sum()), "n_eval": int((qm["role"] == "eval").sum()),
              "class_counts": qm.groupby("role")["label"].value_counts().unstack(0).to_dict()}
    assert checks["query_md5_match"] and checks["bank_md5_match"] and checks["shared_capture_dates"] == 0 and checks["shared_images"] == 0
    assert checks["queries_seen_by_vlm_before"] == 0 and checks["near_duplicate_pairs"] == 0, checks

    api = inf.ClaudeHeadlessAPI(model=model, sandbox_dir=os.path.join(out, "vlm", "sandbox"))
    api.settings = "claude -p defaults (temperature not settable); system prompt: " + api.SYSTEM_PROMPT + "; command: " + " ".join(api.command()[:-1])
    assert api.settings == cfg["settings"]
    prompt = inf.universal_prompt.format(expected_classes=expected)
    assert sha(prompt.encode()) == cfg["prompt_sha256"]
    inf.vision_prompt = prompt

    # ---- offline retrieval (DINOv2 block 12, frozen bank) -----------------------
    bank = pd.DataFrame({0: [path_of(f) for f in rm["ref_id"]], 1: list(rm["label"])})
    emb_path = os.path.join(out, "query_embeddings_dinov2_last.npz")
    embeddings = inf.precompute_embeddings(bank, "dinov2_last")
    saved = np.load(os.path.join(BASE, "offline", "bank_embeddings_dinov2_last.npy"))
    assert np.allclose(np.array([embeddings[i] for i in range(len(bank))]), saved, atol=1e-5), "bank embeddings differ from the base run"
    qcache = dict(np.load(emb_path)) if os.path.exists(emb_path) else {}
    rows, reqs = [], {}
    with open(os.path.join(out, "rankings_dinov2_last.jsonl"), "w") as fh:
        for f, t, role in zip(qm["query_id"], qm["label"], qm["role"]):
            if f not in qcache:
                e = inf.get_image_embedding(path_of(f), model_type="dinov2_last")
                assert not isinstance(e, dict), e
                qcache[f] = np.asarray(e, dtype=np.float64)
            r = inf.rank_candidates(-1, qcache[f], embeddings)
            fh.write(json.dumps({"query_id": f, "ranked": [[rm["ref_id"][i], round(s, 6)] for i, s in r]}) + "\n")
            top = [i for i, _ in r[:K]]
            assert len(set(top)) == K
            labs, sims = [rm["label"][i] for i in top], [s for _, s in r[:K]]
            q64 = inf.load_image(path_of(f))
            ex, paths, cats, _ = inf.build_examples(api, bank, top, None)
            assert cats == labs
            for cond, inputs in (("U", {"image": q64, "examples": ex, "prompt": prompt}), ("Z", {"image": q64, "examples": [], "prompt": prompt})):
                reqs[(f, cond)] = (sha(json.dumps({"vendor": "claude_headless", "model": model, "settings": api.settings, "inputs": inputs}, sort_keys=True).encode()), inputs)
            rows.append({"query_id": f, "role": role, "true_label": t, "U_refs": json.dumps([rm["ref_id"][i] for i in top]), "U_labels": json.dumps(labs),
                         "U_cosines": json.dumps([round(s, 6) for s in sims]), "knn": vote(labs, sims), "margin": margin(labs),
                         "true_label_count": labs.count(t), "top1_cosine": sims[0], "payload_U": reqs[(f, "U")][0], "payload_Z": reqs[(f, "Z")][0]})
    np.savez(emb_path, **qcache)
    J = pd.DataFrame(rows)
    J.to_csv(os.path.join(out, "offline_retrieval.csv"), index=False)
    prereg = os.path.join(out, "preregistration.json")
    if not os.path.exists(prereg):
        wjson(prereg, {
            "written_before_any_vlm_call": not os.path.exists(os.path.join(out, "vlm", "request_ledger.jsonl")), "written_at": time.strftime("%Y-%m-%d %H:%M:%S"),
            "confidence_measure": "vote margin = count of the most frequent label minus count of the second most frequent label among DINOv2 block 12's top 4 (0 to 4)",
            "gated_policy": "predict with U (Claude + top-4 demonstrations) when margin >= tau, otherwise with Z (Claude zero-shot)",
            "tau_candidates": TAUS, "tau_meaning": "0 = always U, 5 = always Z",
            "tau_selection": "the tau with the highest accuracy on the dev queries; ties go to the smallest tau; locked in locked_cutoff.json before any eval request",
            "confidence_analysis": f"on eval, d = correct(U) - correct(Z) per query; low confidence = margin <= {LOW_MARGIN_MAX}, high = margin >= {LOW_MARGIN_MAX + 1}; "
                                   "difference in mean d between the groups with a two-sided permutation test (10000 permutations, seed 42), and Spearman correlation of margin with d",
            "headline_set": "eval queries only; dev results are reported separately", "vote_tie_break": "most frequent label; ties by larger summed cosine, then alphabetical order",
            "dev_queries": sorted(J[J["role"] == "dev"]["query_id"]), "eval_queries": sorted(J[J["role"] == "eval"]["query_id"])})
    wjson(os.path.join(out, "run_config.json"), {**cfg, "bank": BASE, "queries": AUDIT + "/query_panel.csv (roles dev and eval)", "checks": checks,
                                                 "max_requests": args.max_requests, "conditions": ["Z", "KNN", "U"]})
    print(json.dumps({"checks": {k: v for k, v in checks.items() if k != "class_counts"}, "margin_counts": J.groupby("role")["margin"].value_counts().unstack(0).fillna(0).astype(int).to_dict(),
                      "knn_correct": {r: f"{int((g['knn'] == g['true_label']).sum())}/{len(g)}" for r, g in J.groupby("role")}}, indent=1))
    if args.stage == "offline":
        return

    # ---- capped, cached calls ---------------------------------------------------
    ledger = os.path.join(out, "vlm", "request_ledger.jsonl")
    lock = os.path.join(out, "locked_cutoff.json")
    def resp(key):
        p = os.path.join(out, "vlm", "responses", key + ".json")
        return json.load(open(p)) if os.path.exists(p) else None
    def parse(key):
        r = resp(key)
        if r is None:
            return None, "missing"
        try:
            p = inf.extract_json(r["raw"])["prediction"]
        except Exception:
            p = None
        return p, ("valid" if p in expected else "invalid")

    async def run(role):
        used = sum(1 for _ in open(ledger)) if os.path.exists(ledger) else 0
        ids = list(J[J["role"] == role]["query_id"])
        rng = np.random.default_rng([SEED, 3 if role == "dev" else 4])
        plan = [(f, "UZ"[c]) for f in rng.permutation(ids) for c in rng.permutation(2)]
        wjson(os.path.join(out, "vlm", f"run_order_{role}.json"), [{"n": i + 1, "query_id": f, "condition": c, "payload_sha256": reqs[(f, c)][0]} for i, (f, c) in enumerate(plan)])
        for attempt in (1, 2):
            for f, cond in plan:
                key, inputs = reqs[(f, cond)]
                if resp(key) is not None:
                    continue
                if used >= args.max_requests:
                    print(f"request cap reached ({used}/{args.max_requests}); {f} {cond} not sent"); continue
                used += 1
                t0 = time.time()
                e = {"n": used, "t": t0, "stage": role, "query_id": f, "condition": cond, "payload_sha256": key, "attempt": attempt}
                try:
                    raw = await api.get_image_information(inputs)
                    wjson(os.path.join(out, "vlm", "responses", key + ".json"), {"raw": raw, "meta": api.last_meta, "model": model, "query_id": f, "condition": cond})
                    e.update(ok=True, elapsed_s=round(time.time() - t0, 2), meta=api.last_meta)
                except Exception as ex:
                    e.update(ok=False, elapsed_s=round(time.time() - t0, 2), error=str(ex)[:500])
                with open(ledger, "a") as fh:
                    fh.write(json.dumps(e) + "\n")
                if not e["ok"] or used % 20 == 0:
                    print(f"[{used}/{args.max_requests}] {role} {f} {cond}: {'ok' if e['ok'] else 'FAILED ' + e['error'][:80]}", flush=True)
        print(f"{role}: provider requests used so far {used} of {args.max_requests}")

    def table(role):
        g = J[J["role"] == role].copy()
        for c in "UZ":
            pr = [parse(k) for k in g[f"payload_{c}"]]
            g[f"pred_{c}"], g[f"status_{c}"] = [p for p, _ in pr], [s for _, s in pr]
        return g

    if args.stage == "dev":
        assert not any(resp(k) is not None for k in J[J["role"] == "eval"]["payload_U"]), "eval responses already exist"
        await run("dev")
        g = table("dev")
        assert (g["status_U"] == "valid").all() and (g["status_Z"] == "valid").all(), "dev is incomplete; rerun --stage dev to resume before locking"
        if not os.path.exists(lock):
            acc = {t: int(sum(gated(t, m, u, z) == y for m, u, z, y in zip(g["margin"], g["pred_U"], g["pred_Z"], g["true_label"]))) for t in TAUS}
            tau = min(t for t in TAUS if acc[t] == max(acc.values()))
            wjson(lock, {"tau": tau, "dev_correct_by_tau": acc, "n_dev": len(g), "locked_at": time.strftime("%Y-%m-%d %H:%M:%S"),
                         "eval_requests_sent_before_lock": 0, "rule": "use U when margin >= tau, else Z"})
        print("locked:", json.load(open(lock)))
    if args.stage == "eval":
        assert os.path.exists(lock), "run --stage dev first; the cutoff must be locked before evaluation"
        await run("eval")

    # ---- report -------------------------------------------------------------------
    if not os.path.exists(lock):
        return
    tau = json.load(open(lock))["tau"]
    led = [json.loads(l) for l in open(ledger)] if os.path.exists(ledger) else []
    R = pd.concat([table("dev"), table("eval")], ignore_index=True)
    R["pred_G"] = [gated(tau, m, u, z) for m, u, z in zip(R["margin"], R["pred_U"], R["pred_Z"])]
    for c, col in (("U", "pred_U"), ("Z", "pred_Z"), ("KNN", "knn"), ("G", "pred_G")):
        R[f"correct_{c}"] = R[col] == R["true_label"]
    R["valid"] = (R["status_U"] == "valid") & (R["status_Z"] == "valid")
    R["U_equals_vote"] = R["pred_U"] == R["knn"]
    R.to_csv(os.path.join(out, "per_query_results.csv"), index=False)
    COL = {"Z": "pred_Z", "KNN": "knn", "U": "pred_U", "G": "pred_G"}
    NAME = {"Z": "Z. Claude zero-shot", "KNN": "DINOv2 block 12 4-NN vote (no VLM)", "U": "U. Claude with DINOv2 top-4 demonstrations",
            "G": f"Gated: U when margin ≥ {tau}, else Z (cutoff locked on dev)"}

    def score(g, c):
        return {"correct": int(g[f"correct_{c}"].sum()), "n": len(g),
                "macro_f1": float(f1_score(g["true_label"], g[COL[c]].astype(str), labels=sorted(expected), average="macro", zero_division=0)) if len(g) else float("nan")}
    def paired(g, y, x):   # y versus x
        cy, cx = g[f"correct_{y}"], g[f"correct_{x}"]
        w, l = int((cy & ~cx).sum()), int((~cy & cx).sum())
        return {"n": len(g), "wins": w, "losses": l, "both_correct": int((cy & cx).sum()), "both_wrong": int((~cy & ~cx).sum()),
                "pp_difference": 100.0 * (int(cy.sum()) - int(cx.sum())) / max(len(g), 1), "mcnemar_p": 1.0 if w + l == 0 else float(stats.binomtest(w, w + l, 0.5).pvalue)}
    def claude_vs_vote(g):
        vw, vc = g[~g["correct_KNN"]], g[g["correct_KNN"]]
        return {"vote_wrong": len(vw), "claude_corrects_wrong_vote": int(vw["correct_U"].sum()), "vote_correct": len(vc), "claude_overturns_correct_vote": int((~vc["correct_U"]).sum()),
                "U_equals_vote": int(g["U_equals_vote"].sum()), "zero_shot_correct_when_vote_wrong": int(vw["correct_Z"].sum()),
                "of_corrected_votes_zero_shot_also_correct": int((vw["correct_U"] & vw["correct_Z"]).sum())}
    def by_margin(g):
        o = []
        for m, s in g.groupby("margin"):
            o.append({"margin": int(m), "n": len(s), "KNN": int(s["correct_KNN"].sum()), "Z": int(s["correct_Z"].sum()), "U": int(s["correct_U"].sum()),
                      "U_only": int((s["correct_U"] & ~s["correct_Z"]).sum()), "Z_only": int((~s["correct_U"] & s["correct_Z"]).sum())})
        return o
    def confidence(g):
        d = g["correct_U"].astype(int).to_numpy() - g["correct_Z"].astype(int).to_numpy()
        low = (g["margin"] <= LOW_MARGIN_MAX).to_numpy()
        if low.sum() == 0 or (~low).sum() == 0:
            return {"n_low": int(low.sum()), "n_high": int((~low).sum())}
        obs = d[low].mean() - d[~low].mean()
        rng = np.random.default_rng(SEED)
        perm = np.array([(lambda p: d[p].mean() - d[~p].mean())(rng.permutation(low)) for _ in range(10000)])
        rho = stats.spearmanr(g["margin"], d)
        return {"n_low": int(low.sum()), "n_high": int((~low).sum()), "mean_d_low": float(d[low].mean()), "mean_d_high": float(d[~low].mean()),
                "difference_low_minus_high": float(obs), "permutation_p": float((np.abs(perm) >= abs(obs) - 1e-12).mean()),
                "spearman_rho": float(rho.statistic) if np.isfinite(rho.statistic) else None, "spearman_p": float(rho.pvalue) if np.isfinite(rho.pvalue) else None,
                "low": {c: int(g[low][f"correct_{c}"].sum()) for c in ("Z", "U", "KNN")}, "high": {c: int(g[~low][f"correct_{c}"].sum()) for c in ("Z", "U", "KNN")}}
    met = {"model_requested": model, "model_ids_in_responses": sorted({m for r in led if r.get("ok") for m in r["meta"]["model_ids"]}),
           "turns_per_request": sorted({r["meta"]["num_turns"] for r in led if r.get("ok")}), "requests_sent": len(led), "requests_failed": sum(not r["ok"] for r in led),
           "requests_by_stage": pd.Series([r["stage"] for r in led]).value_counts().to_dict() if led else {}, "max_requests": args.max_requests, "locked_cutoff": json.load(open(lock))}
    for role in ("eval", "dev"):
        g = R[(R["role"] == role) & R["valid"]]
        met[role] = {"n_queries": int((R["role"] == role).sum()), "valid_in_U_and_Z": len(g), "invalid_or_missing": {c: int((R[R["role"] == role][f"status_{c}"] != "valid").sum()) for c in "UZ"},
                     "scores": {c: score(g, c) for c in ("Z", "KNN", "U", "G")}, "U_vs_Z": paired(g, "U", "Z"), "U_vs_KNN": paired(g, "U", "KNN"), "Z_vs_KNN": paired(g, "Z", "KNN"),
                     "G_vs_Z": paired(g, "G", "Z"), "G_vs_U": paired(g, "G", "U"), "G_vs_KNN": paired(g, "G", "KNN"), "claude_vs_vote": claude_vs_vote(g),
                     "by_margin": by_margin(g), "confidence": confidence(g),
                     "by_true_label_present": {str(k): {c: [int(s[f"correct_{c}"].sum()), len(s)] for c in ("Z", "KNN", "U")} for k, s in g.groupby(g["true_label_count"] > 0)}}
    wjson(os.path.join(out, "metrics.json"), met)

    E, D = met["eval"], met["dev"]
    done_eval = E["valid_in_U_and_Z"] > 0
    L = ["# DeepWeeds: do retrieved demonstrations add value beyond nearest-neighbour classification under capture-date shift?\n",
         f"Fresh date-disjoint queries from the split-audit panel: {D['n_queries']} dev and {E['n_queries']} eval (3 and 7 per class). Bank: the frozen 99-image date-disjoint bank of the base run. "
         f"Retriever: DINOv2 block 12 CLS. VLM `{', '.join(met['model_ids_in_responses']) or model}` through headless `claude -p`. "
         f"Requests sent: {met['requests_sent']} of {met['max_requests']} allowed, {met['requests_failed']} failed. "
         f"Valid in both U and Z: dev {D['valid_in_U_and_Z']}/{D['n_queries']}, eval {E['valid_in_U_and_Z']}/{E['n_queries']}.\n",
         f"The confidence measure and the cutoff rule were written to `preregistration.json` before any call. The cutoff was fitted on dev and locked before any eval request: "
         f"**tau = {tau}** (dev correct by tau: {met['locked_cutoff']['dev_correct_by_tau']}).\n"]
    for role, M, title in (("eval", E, "Evaluation queries (headline)"), ("dev", D, "Development queries")):
        if M["valid_in_U_and_Z"] == 0:
            continue
        n = M["valid_in_U_and_Z"]
        L += [f"## {title}\n", "| Method | Correct | Accuracy | Macro-F1 (9 classes) |", "|---|---|---|---|"]
        for c in ("Z", "KNN", "U", "G"):
            s = M["scores"][c]
            L.append(f"| {NAME[c]} | {s['correct']}/{s['n']} | {100 * s['correct'] / s['n']:.0f}% | {s['macro_f1']:.3f} |")
        L += ["\n| Paired comparison | Difference (pp) | Wins | Losses | Both correct | Both wrong | Exact McNemar p |", "|---|---|---|---|---|---|---|"]
        for k, nm in (("U_vs_Z", "U versus Z"), ("U_vs_KNN", "U versus 4-NN vote"), ("Z_vs_KNN", "Z versus 4-NN vote"), ("G_vs_Z", "Gated versus Z"), ("G_vs_U", "Gated versus U"), ("G_vs_KNN", "Gated versus 4-NN vote")):
            d = M[k]
            L.append(f"| {nm} | {d['pp_difference']:+.0f} | {d['wins']} | {d['losses']} | {d['both_correct']} | {d['both_wrong']} | {fp(d['mcnemar_p'])} |")
        v = M["claude_vs_vote"]
        L += [f"\n**Claude with demonstrations versus the vote.** The vote is wrong on {v['vote_wrong']} of {n} queries, and U corrects {v['claude_corrects_wrong_vote']} of them "
              f"(zero-shot is also correct on {v['of_corrected_votes_zero_shot_also_correct']} of those {v['claude_corrects_wrong_vote']}). "
              f"The vote is correct on {v['vote_correct']} queries, and U overturns {v['claude_overturns_correct_vote']} of them. U gives the vote's label on {v['U_equals_vote']} of {n} queries.\n",
              "| Vote margin | n | 4-NN correct | Z correct | U correct | U-only wins over Z | Z-only wins over U |", "|---|---|---|---|---|---|---|"]
        L += [f"| {b['margin']} | {b['n']} | {b['KNN']} | {b['Z']} | {b['U']} | {b['U_only']} | {b['Z_only']} |" for b in M["by_margin"]]
        cf = M["confidence"]
        if "permutation_p" in cf:
            L.append(f"\n**Does low retrieval confidence predict that zero-shot does better?** With d = correct(U) − correct(Z), the mean of d is {cf['mean_d_low']:+.2f} on the {cf['n_low']} "
                     f"low-confidence queries (margin ≤ {LOW_MARGIN_MAX}) and {cf['mean_d_high']:+.2f} on the {cf['n_high']} high-confidence queries. The difference is "
                     f"{cf['difference_low_minus_high']:+.2f} (permutation p {fp(cf['permutation_p'])}). Spearman correlation of margin with d: "
                     + (f"{cf['spearman_rho']:+.2f} (p {fp(cf['spearman_p'])})." if cf["spearman_rho"] is not None else "undefined.")
                     + f" Low confidence: Z {cf['low']['Z']}/{cf['n_low']}, U {cf['low']['U']}/{cf['n_low']}, 4-NN {cf['low']['KNN']}/{cf['n_low']}. "
                       f"High confidence: Z {cf['high']['Z']}/{cf['n_high']}, U {cf['high']['U']}/{cf['n_high']}, 4-NN {cf['high']['KNN']}/{cf['n_high']}.\n")
        tl = M["by_true_label_present"]
        L += ["| True label among the 4 demonstrations | Z correct | 4-NN correct | U correct |", "|---|---|---|---|"]
        for k, nm in (("True", "Present"), ("False", "Absent")):
            if k in tl:
                L.append(f"| {nm} | " + " | ".join(f"{tl[k][c][0]}/{tl[k][c][1]}" for c in ("Z", "KNN", "U")) + " |")
        L.append("")
    if not done_eval:
        L.append("Evaluation has not run yet.\n")
    notes = os.path.join(out, "NOTES.md")
    if os.path.exists(notes):
        L.append(open(notes).read())
    open(os.path.join(out, "SUMMARY.md"), "w").write("\n".join(L) + "\n")
    print("\n".join(L))
