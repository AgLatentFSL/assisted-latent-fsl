"""Extension of the DeepWeeds date-disjoint matched-label pilot: adds condition U.

U = Claude with DINOv2 block 12's four highest-ranked references, in descending cosine
order, each with its own label (the saved unconstrained ranking behind the DINOv2 4-NN vote).
A, B and Z come from the base run's cache. The base run directory is read, never written.

Run through `python inference.py --pilot-matched-ext --stage {offline,vlm,report} --base-dir <base> --out-dir <new>`.
"""
import argparse, json, os, shutil, time
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.metrics import f1_score
from pilot_matched_labels import K, fp, md5_file, path_of, sha, vote, wjson


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--pilot-matched-ext", action="store_true")
    p.add_argument("--stage", choices=["offline", "vlm", "report"], default="offline")
    p.add_argument("--base-dir", required=True)
    p.add_argument("--out-dir", required=True)
    p.add_argument("--max-requests", type=int, default=20)
    return p.parse_args()


async def main(inf):
    args = parse_args()
    base, out = args.base_dir, args.out_dir
    assert os.path.abspath(base) != os.path.abspath(out)
    os.makedirs(os.path.join(out, "vlm", "responses"), exist_ok=True)
    cfg = json.load(open(os.path.join(base, "run_config.json")))
    qm = pd.read_csv(os.path.join(base, "query_manifest.csv"), dtype=str)
    rm = pd.read_csv(os.path.join(base, "reference_manifest.csv"), dtype={"ref_id": str, "label": str, "capture_date": str, "md5": str})
    BM = pd.read_csv(os.path.join(base, "offline", "matched_pairs.csv")).set_index("query_id")
    BP = pd.read_csv(os.path.join(base, "vlm", "predictions.csv"), keep_default_na=False)
    expected, model = cfg["expected_classes"], cfg["model"]

    # ---- verification before any call --------------------------------------
    checks = {
        "query_md5_match": all(md5_file(path_of(f)) == m for f, m in zip(qm["query_id"], qm["md5"])),
        "bank_md5_match": all(md5_file(path_of(f)) == m for f, m in zip(rm["ref_id"], rm["md5"])),
        "file_name_dates_match_manifest": all(f[:8] == d for f, d in zip(qm["query_id"], qm["capture_date"])) and all(f[:8] == d for f, d in zip(rm["ref_id"], rm["capture_date"])),
        "shared_capture_dates": len(set(qm["capture_date"]) & set(rm["capture_date"])),
        "shared_images": len(set(qm["query_id"]) & set(rm["ref_id"])) + len(set(qm["md5"]) & set(rm["md5"])),
        "n_queries": len(qm), "n_bank": len(rm), "bank_class_counts": rm["label"].value_counts().sort_index().to_dict()}
    assert checks["query_md5_match"] and checks["bank_md5_match"] and checks["file_name_dates_match_manifest"]
    assert checks["shared_capture_dates"] == 0 and checks["shared_images"] == 0 and len(qm) == 20 and len(rm) == 99

    api = inf.ClaudeHeadlessAPI(model=model, sandbox_dir=os.path.join(out, "vlm", "sandbox"))
    api.settings = "claude -p defaults (temperature not settable); system prompt: " + api.SYSTEM_PROMPT + "; command: " + " ".join(api.command()[:-1])
    assert api.settings == cfg["settings"], "runner settings differ from the base run"
    prompt = inf.universal_prompt.format(expected_classes=expected)
    assert prompt == cfg["prompt"] and sha(prompt.encode()) == cfg["prompt_sha256"]
    inf.vision_prompt = prompt
    bank = pd.DataFrame({0: [path_of(f) for f in rm["ref_id"]], 1: list(rm["label"])})
    ridx = {f: i for i, f in enumerate(rm["ref_id"])}
    rlab = dict(zip(rm["ref_id"], rm["label"]))

    def payload(f, refs):
        ex, paths, cats, _ = inf.build_examples(api, bank, [ridx[r] for r in refs], None)
        assert cats == [rlab[r] for r in refs] and paths == [path_of(r) for r in refs]
        inputs = {"image": inf.load_image(path_of(f)), "examples": ex, "prompt": prompt}
        return sha(json.dumps({"vendor": "claude_headless", "model": model, "settings": api.settings, "inputs": inputs}, sort_keys=True).encode()), inputs

    ranked = {}
    for line in open(os.path.join(base, "offline", "rankings_dinov2_last.jsonl")):
        d = json.loads(line)
        ranked[d["query_id"]] = d["ranked"]
        assert all(d["ranked"][i][1] >= d["ranked"][i + 1][1] for i in range(len(d["ranked"]) - 1)) and len(d["ranked"]) == len(rm)
    rows, reqs = [], {}
    for f, t in zip(qm["query_id"], qm["label"]):
        u = [r for r, _ in ranked[f][:K]]                       # saved unconstrained DINOv2 ranking, descending cosine
        assert u == json.loads(BM.at[f, "unconstrained_dinov2_top4"]) and len(set(u)) == K and f not in u
        ul = [rlab[r] for r in u]
        assert ul == json.loads(BM.at[f, "unconstrained_dinov2_labels"])
        assert vote(ul, [s for _, s in ranked[f][:K]]) == BM.at[f, "knn4_dinov2"]      # tie-break preserved
        a, b, al = json.loads(BM.at[f, "A_refs"]), json.loads(BM.at[f, "B_refs"]), json.loads(BM.at[f, "demo_labels"])
        ka, _ = payload(f, a)
        kb, _ = payload(f, b)
        assert ka == BM.at[f, "payload_A"] and kb == BM.at[f, "payload_B"], "rebuilt A/B payloads differ from the base run"
        ku, inp = payload(f, u)
        reqs[f] = (ku, inp)
        rows.append({"query_id": f, "true_label": t, "AB_labels": json.dumps(al), "U_labels": json.dumps(ul), "A_refs": json.dumps(a), "B_refs": json.dumps(b),
                     "U_refs": json.dumps(u), "U_cosines": json.dumps([s for _, s in ranked[f][:K]]),
                     "slots_changed_U_vs_A": sum(x != y for x, y in zip(u, a)), "slots_changed_U_vs_B": sum(x != y for x, y in zip(u, b)),
                     "images_shared_U_A": len(set(u) & set(a)), "images_shared_U_B": len(set(u) & set(b)),
                     "ordered_labels_changed": ul != al, "label_multiset_changed": sorted(ul) != sorted(al),
                     "true_label_count_AB": al.count(t), "true_label_count_U": ul.count(t), "true_label_in_AB": t in al, "true_label_in_U": t in ul,
                     "payload_A": ka, "payload_B": kb, "payload_Z": BM.at[f, "payload_Z"], "payload_U": ku,
                     "U_payload_equals": "A" if ku == ka else ("B" if ku == kb else ""), "knn4_dinov2": BM.at[f, "knn4_dinov2"], "knn4_vit": BM.at[f, "knn4_vit"]})
    J = pd.DataFrame(rows)
    J.to_csv(os.path.join(out, "u_condition_offline.csv"), index=False)
    off = {"checks": checks, "base_dir": base, "model": model,
           "true_label_present": {"AB": int(J["true_label_in_AB"].sum()), "U": int(J["true_label_in_U"].sum()),
                                  "U_adds": int((J["true_label_in_U"] & ~J["true_label_in_AB"]).sum()), "U_removes": int((~J["true_label_in_U"] & J["true_label_in_AB"]).sum()),
                                  "both": int((J["true_label_in_U"] & J["true_label_in_AB"]).sum()), "neither": int((~J["true_label_in_U"] & ~J["true_label_in_AB"]).sum())},
           "mean_true_label_count": {"AB": float(J["true_label_count_AB"].mean()), "U": float(J["true_label_count_U"].mean())},
           "ordered_labels_changed_U_vs_AB": int(J["ordered_labels_changed"].sum()), "label_multiset_changed_U_vs_AB": int(J["label_multiset_changed"].sum()),
           "mean_slots_changed_U_vs_A": float(J["slots_changed_U_vs_A"].mean()), "mean_slots_changed_U_vs_B": float(J["slots_changed_U_vs_B"].mean()),
           "mean_images_shared_U_A": float(J["images_shared_U_A"].mean()), "mean_images_shared_U_B": float(J["images_shared_U_B"].mean()),
           "U_payload_identical_to_A": int((J["U_payload_equals"] == "A").sum()), "U_payload_identical_to_B": int((J["U_payload_equals"] == "B").sum())}
    wjson(os.path.join(out, "offline_summary.json"), off)
    wjson(os.path.join(out, "run_config.json"), {**cfg, "extension_of": base, "U": "facebook/dinov2-base block 12 CLS: top 4 of the saved ranking, descending cosine, own labels",
                                                 "max_new_requests": args.max_requests})
    print(json.dumps({k: v for k, v in off.items() if k != "checks"}, indent=1))
    if args.stage == "offline":
        return

    # ---- capped, cached calls for U only -------------------------------------
    def find(key):
        for d in (os.path.join(out, "vlm", "responses"), os.path.join(base, "vlm", "responses")):
            p = os.path.join(d, key + ".json")
            if os.path.exists(p):
                r = json.load(open(p))
                assert r["model"] == model
                return r, p
        return None, None
    ledger = os.path.join(out, "vlm", "request_ledger.jsonl")
    used = sum(1 for _ in open(ledger)) if os.path.exists(ledger) else 0
    if args.stage == "vlm":
        order = list(np.random.default_rng([42, 2]).permutation(list(qm["query_id"])))
        wjson(os.path.join(out, "vlm", "run_order.json"), [{"n": i + 1, "query_id": f, "payload_sha256": reqs[f][0]} for i, f in enumerate(order)])
        for attempt in (1, 2):                                # second pass retries failures while the cap allows
            for f in order:
                key, inputs = reqs[f]
                if find(key)[0] is not None:
                    continue
                if used >= args.max_requests:
                    print(f"request cap reached ({used}/{args.max_requests}); {f} U not sent"); continue
                used += 1
                t0 = time.time()
                e = {"n": used, "t": t0, "query_id": f, "condition": "U", "payload_sha256": key, "attempt": attempt}
                try:
                    raw = await api.get_image_information(inputs)
                    wjson(os.path.join(out, "vlm", "responses", key + ".json"), {"raw": raw, "meta": api.last_meta, "model": model, "first_query_id": f, "first_condition": "U"})
                    e.update(ok=True, elapsed_s=round(time.time() - t0, 2), meta=api.last_meta)
                except Exception as ex:
                    e.update(ok=False, elapsed_s=round(time.time() - t0, 2), error=str(ex)[:500])
                with open(ledger, "a") as fh:
                    fh.write(json.dumps(e) + "\n")
                print(f"[{used}/{args.max_requests}] {f} U: {'ok' if e['ok'] else 'FAILED ' + e['error'][:80]}", flush=True)

    # ---- report ---------------------------------------------------------------
    led = [json.loads(l) for l in open(ledger)] if os.path.exists(ledger) else []
    pred, status, src = {}, {}, {}
    for f in qm["query_id"]:
        for c in "ABZ":
            row = BP[(BP["query_id"] == f) & (BP["condition"] == c)].iloc[0]
            assert row["payload_sha256"] == J.set_index("query_id").at[f, f"payload_{c}"]
            r, _ = find(row["payload_sha256"])
            p = None
            if r is not None:
                try: p = inf.extract_json(r["raw"])["prediction"]
                except Exception: p = None
            pred[(f, c)], status[(f, c)] = p, ("missing" if r is None else ("valid" if p in expected else "invalid"))
            assert status[(f, c)] != "valid" or p == row["prediction"]
        r, pth = find(reqs[f][0])
        p = None
        if r is not None:
            try: p = inf.extract_json(r["raw"])["prediction"]
            except Exception: p = None
        pred[(f, "U")], status[(f, "U")] = p, ("missing" if r is None else ("valid" if p in expected else "invalid"))
        src[f] = "" if r is None else ("base cache" if pth.startswith(base) else "new request")
    for c in "ABZU":
        J[f"pred_{c}"] = [pred[(f, c)] for f in J["query_id"]]
        J[f"status_{c}"] = [status[(f, c)] for f in J["query_id"]]
    J["U_response_source"] = [src[f] for f in J["query_id"]]
    for c in ["A", "B", "Z", "U"]:
        J[f"correct_{c}"] = J[f"pred_{c}"] == J["true_label"]
    J["correct_knn4_dinov2"] = J["knn4_dinov2"] == J["true_label"]
    J.to_csv(os.path.join(out, "per_query_results.csv"), index=False)
    T = J.set_index("query_id")
    n = len(T)
    col = {"A": "pred_A", "B": "pred_B", "Z": "pred_Z", "U": "pred_U", "knn": "knn4_dinov2"}
    def valid(c):
        return [f for f in T.index if c == "knn" or T.at[f, f"status_{c}"] == "valid"]
    def score(c, ids):
        y, x = T.loc[ids, "true_label"], T.loc[ids, col[c]].astype(str)
        return {"correct": int((x == y).sum()), "n": len(ids), "macro_f1": float(f1_score(y, x, labels=sorted(expected), average="macro", zero_division=0)) if ids else float("nan")}
    def paired(x):  # U versus x on their shared valid queries
        ids = [f for f in valid("U") if f in set(valid(x))]
        cu, cx = (T.loc[ids, "pred_U"] == T.loc[ids, "true_label"]), (T.loc[ids, col[x]] == T.loc[ids, "true_label"])
        w, l = int((cu & ~cx).sum()), int((~cu & cx).sum())
        return {"n": len(ids), "U": score("U", ids), "other": score(x, ids), "U_only": w, "other_only": l, "both_correct": int((cu & cx).sum()), "both_wrong": int((~cu & ~cx).sum()),
                "pp_difference": 100.0 * (int(cu.sum()) - int(cx.sum())) / len(ids) if ids else float("nan"),
                "prediction_changed": int((T.loc[ids, "pred_U"] != T.loc[ids, col[x]]).sum()),
                "mcnemar_p": 1.0 if w + l == 0 else float(stats.binomtest(w, w + l, 0.5).pvalue)}
    cmp = {x: paired(x) for x in ["A", "B", "Z", "knn"]}
    shared = [f for f in T.index if all(T.at[f, f"status_{c}"] == "valid" for c in "ABZU")]
    cov = {c: len(valid(c)) for c in ["A", "B", "Z", "U"]}
    tok = [r["meta"]["usage"]["input_tokens"] + r["meta"]["usage"]["cache_creation_input_tokens"] + r["meta"]["usage"]["cache_read_input_tokens"] for r in led if r.get("ok")]
    met = {"model_requested": model, "model_ids_in_new_responses": sorted({m for r in led if r.get("ok") for m in r["meta"]["model_ids"]}),
           "turns_per_new_request": sorted({r["meta"]["num_turns"] for r in led if r.get("ok")}), "new_requests_sent": len(led),
           "new_requests_failed": sum(not r["ok"] for r in led), "max_new_requests": args.max_requests,
           "U_responses_from_base_cache": int((T["U_response_source"] == "base cache").sum()), "coverage": cov, "shared_valid_queries": len(shared),
           "all_conditions_on_shared": {c: score(c, shared) for c in ["Z", "A", "B", "U", "knn"]},
           "U_vs_A_primary": cmp["A"], "U_vs_B": cmp["B"], "U_vs_Z": cmp["Z"], "U_vs_dinov2_knn4": cmp["knn"],
           "mean_input_tokens_U": float(np.mean(tok)) if tok else None, "offline": off}
    wjson(os.path.join(out, "metrics.json"), met)

    NAME = {"Z": "Z. VLM, zero-shot", "A": "A. VLM, ViT top 4", "B": "B. VLM, DINOv2 images under ViT's label sequence",
            "U": "U. VLM, DINOv2 top 4 with own labels", "knn": "DINOv2 block 12 4-NN vote (no VLM)"}
    tp = off["true_label_present"]
    L = ["# DeepWeeds date-disjoint pilot, extension: unconstrained DINOv2 demonstrations (U)\n",
         f"Same {n} queries, 99-image date-disjoint bank, prompt, image format and parser as the base run `{os.path.basename(os.path.normpath(base))}`. "
         f"VLM `{model}` through headless `claude -p`. A, B and Z come from the base run's cache. New requests for U: {len(led)} of {args.max_requests} allowed, "
         f"{met['new_requests_failed']} failed. Coverage: " + ", ".join(f"{c} {cov[c]}/{n}" for c in "ABZU") + ".\n",
         "U uses DINOv2 block 12's four highest-ranked references in descending cosine order, each with its own label. "
         "U can differ from A and B in labels, images and order, so the comparisons below are exploratory and descriptive.\n",
         f"## All conditions on the {len(shared)} queries valid in A, B, Z and U\n", "| Condition | Correct | Accuracy | Macro-F1 (9 classes) |", "|---|---|---|---|"]
    for c in ["Z", "A", "B", "U", "knn"]:
        s = met["all_conditions_on_shared"][c]
        L.append(f"| {NAME[c]} | {s['correct']}/{s['n']} | {100 * s['correct'] / max(s['n'], 1):.0f}% | {s['macro_f1']:.3f} |")
    L += ["\n## Paired comparisons of U\n", "| Comparison | n | U correct | Other correct | Difference (pp) | U-only wins | Other-only wins | Both correct | Both wrong | Predictions changed | Exact McNemar p |",
          "|---|---|---|---|---|---|---|---|---|---|---|"]
    for x, nm in (("A", "U versus A (primary)"), ("B", "U versus B"), ("Z", "U versus Z"), ("knn", "U versus DINOv2 4-NN vote")):
        d = cmp[x]
        L.append(f"| {nm} | {d['n']} | {d['U']['correct']} | {d['other']['correct']} | {d['pp_difference']:+.0f} | {d['U_only']} | {d['other_only']} | {d['both_correct']} | {d['both_wrong']} | {d['prediction_changed']} | {fp(d['mcnemar_p'])} |")
    L += ["\n## Does the query's true label appear among the four demonstrations?\n", "| | A and B (ViT label sequence) | U (DINOv2 labels) |", "|---|---|---|",
          f"| Prompts that contain the true label | {tp['AB']}/{n} | {tp['U']}/{n} |",
          f"| Mean demonstrations with the true label, of {K} | {off['mean_true_label_count']['AB']:.2f} | {off['mean_true_label_count']['U']:.2f} |",
          f"\nU adds the true label on {tp['U_adds']} queries and removes it on {tp['U_removes']}. Both prompts contain it on {tp['both']} queries and neither on {tp['neither']}.\n",
          f"The ordered label sequence differs between U and A/B on {off['ordered_labels_changed_U_vs_AB']} of {n} queries, and the set of labels differs on {off['label_multiset_changed_U_vs_AB']}. "
          f"U differs from A in {off['mean_slots_changed_U_vs_A']:.2f} of {K} image slots on average and from B in {off['mean_slots_changed_U_vs_B']:.2f}. "
          f"U shares {off['mean_images_shared_U_A']:.2f} images with A and {off['mean_images_shared_U_B']:.2f} with B on average, in any slot. "
          f"U's payload is identical to A's on {off['U_payload_identical_to_A']} queries and to B's on {off['U_payload_identical_to_B']}.\n",
          "## VLM accuracy by presence of the true label among the demonstrations\n", "| Condition | True label present | True label absent |", "|---|---|---|"]
    for c, flag in (("A", "true_label_in_AB"), ("B", "true_label_in_AB"), ("U", "true_label_in_U")):
        ids = valid(c)
        g = T.loc[ids]
        L.append(f"| {c} | {int(g[g[flag]][f'correct_{c}'].sum())}/{int(g[flag].sum())} | {int(g[~g[flag]][f'correct_{c}'].sum())}/{int((~g[flag]).sum())} |")
    L += ["\n## U versus B, query by query\n",
          "| Query | True class | A/B labels | U labels | Labels changed | True label in A/B → U | B prediction | U prediction | Flip |", "|---|---|---|---|---|---|---|---|---|"]
    for f, w in T.iterrows():
        flip = "" if w["correct_B"] == w["correct_U"] else ("U gains" if w["correct_U"] else "U loses")
        L.append(f"| `{f}` | {w['true_label']} | {', '.join(json.loads(w['AB_labels']))} | {', '.join(json.loads(w['U_labels']))} | {'yes' if w['ordered_labels_changed'] else 'no'} | "
                 f"{w['true_label_count_AB']} → {w['true_label_count_U']} | {w['pred_B']}{' ✓' if w['correct_B'] else ''} | {w['pred_U']}{' ✓' if w['correct_U'] else ''} | {flip} |")
    notes = os.path.join(out, "NOTES.md")
    if os.path.exists(notes):
        L.append("\n" + open(notes).read())
    open(os.path.join(out, "SUMMARY.md"), "w").write("\n".join(L) + "\n")
    print("\n".join(L))
