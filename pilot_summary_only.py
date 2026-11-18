"""Paired pilot: do the four reference images add anything beyond the vote summary?

40 queries from the five-dataset AgEval run (runs/arbab_eval_seed42): 20 with a correct DINOv2 4-NN vote and 20 with a
wrong one, drawn with a fixed seed from the vote and the true label only (no VLM prediction is read before the draw).
Per query, three predictions: the cached DINOv2 vote, the cached full vote-aware prediction (condition O: four reference
images + vote summary), and one new call S: the query image with the vote summary and no reference images.
Same model, client, system prompt, task prompt and image encoding as the main run.
Usage: python pilot_summary_only.py <out_dir> {freeze,vlm,report}
"""
import asyncio, json, os, sys, time
from concurrent.futures import ThreadPoolExecutor
import numpy as np, pandas as pd
import inference as inf
from pilot_arbab_eval import MODEL
from pilot_matched_labels import sha, wjson

out, stage = sys.argv[1], sys.argv[2]; os.makedirs(os.path.join(out, "responses"), exist_ok=True)
SRC = os.path.join("runs", "arbab_eval_seed42"); OFF = os.path.join("runs", "offline_layer_stats")
DATASETS = ["Bean Leaf Lesions", "Mango Leaf Disease", "DeepWeeds", "Soybean Seeds", "Soybean Diseases"]; SEED, PER = 42, 20


def summary_text(labels):
    """vote_text of pilot_vote_aware.py with the references described as not shown; counts and default sentence unchanged."""
    order = sorted(dict.fromkeys(labels), key=lambda l: (-labels.count(l), labels.index(l)))
    counts = ", ".join(f'"{l}": {labels.count(l)}' for l in order)
    s = (f"{len(labels)} labeled reference images were retrieved because they look most similar to the image to classify. They are not shown. "
         f"Label counts among them: {counts}. ")
    if len(order) == 1 or labels.count(order[0]) > labels.count(order[1]):
        s += (f'The most frequent reference label is "{order[0]}". Use it as the default answer. '
              "Choose a different class only if the image clearly shows a different class.")
    else:
        s += "The reference labels do not agree on one class. Decide from the image."
    return s


sel_path = os.path.join(out, "selection.csv")
if stage == "freeze":
    assert not os.path.exists(sel_path), "selection already frozen"
    J = pd.read_csv(os.path.join(SRC, "per_query_results.csv"), keep_default_na=False, usecols=["dataset", "query_idx", "query_id", "true_label", "O_labels", "vote_dinov2"])
    J = J[J.dataset.isin(DATASETS)].reset_index(drop=True); J["vote_correct"] = J.vote_dinov2.astype(str) == J.true_label.astype(str)
    rng = np.random.default_rng(SEED)
    S = pd.concat([J[J.vote_correct].iloc[np.sort(rng.choice(int(J.vote_correct.sum()), PER, replace=False))],
                   J[~J.vote_correct].iloc[np.sort(rng.choice(int((~J.vote_correct).sum()), PER, replace=False))]]).reset_index(drop=True)
    S.to_csv(sel_path, index=False)
    wjson(os.path.join(out, "preregistration.json"), {
        "frozen_at": time.strftime("%Y-%m-%d %H:%M:%S"), "source_run": "runs/arbab_eval_seed42 (five datasets, 493 queries; Dangerous Insects excluded)", "model": MODEL, "seed": SEED,
        "selection": "numpy default_rng(42): 20 of the queries with a correct DINOv2 4-NN vote, then 20 of the queries with a wrong vote; only dataset, true label, reference labels and vote were read",
        "conditions": {"V": "cached DINOv2 4-NN vote", "O": "cached full vote-aware prediction (four reference images + vote summary)", "S": "new: task prompt, vote summary, task prompt, query image; no reference images"},
        "summary_text_example": summary_text(["a", "a", "a", "b"]), "summary_text_tie_example": summary_text(["a", "a", "b", "b"]),
        "deviation_from_O_text": "first sentence drops 'above' and adds 'They are not shown.'; the tie ending drops 'and the reference images'",
        "report": "per vote stratum: paired counts O-only-correct and S-only-correct, correct votes preserved, wrong votes rescued; no benchmark-level accuracy claim",
        "decision_rule": "expand if the reference images rescue more wrong votes and the gain survives any additional correct votes they overturn; if the two prompts behave similarly, prioritize the text-summary explanation",
        "pool_counts": {"vote_correct": int(J.vote_correct.sum()), "vote_wrong": int((~J.vote_correct).sum())}})
    print(S.groupby(["vote_correct", "dataset"]).size()); sys.exit()

S = pd.read_csv(sel_path, keep_default_na=False)
prompts = json.load(open(os.path.join(SRC, "prompts.json")))
api0 = inf.ClaudeHeadlessAPI(model=MODEL, sandbox_dir=os.path.join(out, "sandbox"))
settings = "claude -p defaults (temperature not settable); system prompt: " + api0.SYSTEM_PROMPT + "; command: " + " ".join(api0.command()[:-1])
assert settings == json.load(open(os.path.join(SRC, "run_config.json")))["settings"], "client settings differ from the main run"
rpath = lambda i: os.path.join(out, "responses", f"{i:02d}.json")

if stage == "vlm":
    def work(i):
        if os.path.exists(rpath(i)): return
        r = S.iloc[i]; pool = pd.read_csv(os.path.join(OFF, f"pool_{r.dataset.replace(' ', '_')}.csv"))
        path = pool["path"][int(r.query_idx)]; assert os.path.basename(path) == r.query_id
        inp = {"image": inf.load_image(path), "examples": [{"type": "text", "text": summary_text(json.loads(r.O_labels))}], "prompt": prompts[r.dataset]}
        api = inf.ClaudeHeadlessAPI(model=MODEL, sandbox_dir=os.path.join(out, "sandbox"))
        for attempt in (1, 2):
            try:
                raw = asyncio.run(api.get_image_information(inp))
                wjson(rpath(i), {"raw": raw, "meta": api.last_meta, "dataset": r.dataset, "query_idx": int(r.query_idx), "summary_text": inp["examples"][0]["text"],
                                 "payload_sha256": sha(json.dumps({"settings": settings, "inputs": inp}, sort_keys=True).encode())}); return
            except Exception as ex:
                print(f"q{i} attempt {attempt} failed: {str(ex)[:200]}", flush=True)
    with ThreadPoolExecutor(4) as ex: list(ex.map(work, range(len(S))))
    print("responses:", sum(os.path.exists(rpath(i)) for i in range(len(S))), "of", len(S)); sys.exit()

# ---- report
R = pd.read_csv(os.path.join(SRC, "per_query_results.csv"), keep_default_na=False)[["dataset", "query_idx", "pred_O", "pred_D"]]
S = S.merge(R, on=["dataset", "query_idx"], how="left", validate="one_to_one")
def pred(i):
    try: return str(inf.extract_json(json.load(open(rpath(i)))["raw"])["prediction"])
    except Exception: return "INVALID"
S["pred_S"] = [pred(i) for i in range(len(S))]
for c in "OSD": S["ok_" + c] = S["pred_" + c].astype(str) == S.true_label.astype(str); S["follows_" + c] = S["pred_" + c].astype(str) == S.vote_dinov2.astype(str)
S.to_csv(os.path.join(out, "per_query_results.csv"), index=False)
ids = sorted({m for i in range(len(S)) for m in json.load(open(rpath(i)))["meta"].get("model_ids", [])})
L = ["# Summary-only pilot: reference images versus the vote summary\n",
     f"40 queries from the five-dataset AgEval run, 20 with a correct DINOv2 4-NN vote and 20 with a wrong vote (seed {SEED}, drawn before any VLM prediction was read). "
     f"Full prompt (O): cached predictions. Summary only (S): {len(S)} new calls, model ids {ids}, invalid answers {int((S.pred_S == 'INVALID').sum())}. "
     "The sample balances the two strata by design, so it supports no benchmark-level accuracy.\n",
     "| Vote stratum | n | Both correct | Full prompt only | Summary only | Neither | Full correct | Summary correct | Full returns the vote label | Summary returns the vote label |", "|---|---|---|---|---|---|---|---|---|---|"]
met = {}
for vc, nm in ((True, "Vote correct (preserved)"), (False, "Vote wrong (rescued)")):
    g = S[S.vote_correct == vc]; a, b = g.ok_O, g.ok_S
    m = {"n": len(g), "both": int((a & b).sum()), "full_only": int((a & ~b).sum()), "summary_only": int((~a & b).sum()), "neither": int((~a & ~b).sum()), "full_correct": int(a.sum()), "summary_correct": int(b.sum()),
         "full_follows_vote": int(g.follows_O.sum()), "summary_follows_vote": int(g.follows_S.sum()), "standard_prompt_correct": int(g.ok_D.sum()),
         "by_dataset": {ds: {"n": len(h), "full": int(h.ok_O.sum()), "summary": int(h.ok_S.sum())} for ds, h in g.groupby("dataset")}}
    met["vote_correct" if vc else "vote_wrong"] = m
    L.append(f"| {nm} | {m['n']} | {m['both']} | {m['full_only']} | {m['summary_only']} | {m['neither']} | {m['full_correct']} | {m['summary_correct']} | {m['full_follows_vote']} | {m['summary_follows_vote']} |")
L += ["\n## By dataset (full correct / summary correct / n)\n", "| Vote stratum | " + " | ".join(DATASETS) + " |", "|---|" + "---|" * len(DATASETS)]
for k, nm in (("vote_correct", "Vote correct"), ("vote_wrong", "Vote wrong")):
    L.append(f"| {nm} | " + " | ".join((lambda d: f"{d['full']} / {d['summary']} / {d['n']}" if d else "0")(met[k]["by_dataset"].get(ds)) for ds in DATASETS) + " |")
wjson(os.path.join(out, "metrics.json"), met); open(os.path.join(out, "SUMMARY.md"), "w").write("\n".join(L) + "\n"); print("\n".join(L))
