"""Downstream test: does a layer's retrieval gain change the answers of the VLM?

For a shot budget K, every one of the nine encoders contributes its five layers with the largest net same-class coverage
gain over the pooled ViT baseline (retrieval analysis of the atlas, same K). Each layer's top-K references are sent to the
VLM with the standard assisted prompt, and its accuracy is compared, query by query, with the pooled ViT baseline at the
same K and with zero-shot prompting (K = 0). Same model, client, prompt and image encoding as the main run
(pilot_arbab_eval.py). Identical prompts are sent once; zero-shot answers of the main run are reused when the request is identical.
Usage (from repo/): python pilot_layer_vlm.py <out_dir> {freeze|vlm|report} [--k 2] [--workers 4] [--max-requests N]
"""
import argparse, asyncio, json, os, shutil, sys, threading, time
from concurrent.futures import ThreadPoolExecutor
import numpy as np, pandas as pd

ap = argparse.ArgumentParser(); ap.add_argument("out"); ap.add_argument("stage", choices=["freeze", "amend", "vlm", "report"]); ap.add_argument("--k", type=int, default=2); ap.add_argument("--workers", type=int, default=4); ap.add_argument("--max-requests", type=int, default=20000)
args = ap.parse_args(); out, K = args.out, args.k; os.makedirs(os.path.join(out, "responses"), exist_ok=True)
ATLAS = "atlas"; MAIN = os.path.join("runs", "arbab_eval_seed42"); MODEL = "claude-sonnet-5-5"; TOPN = 5
ENC = {"ViT": "1_ViT", "DINOv2": "2_DINOv2", "CLIP": "3_CLIP", "ResNet50": "4_ResNet50", "SigLIP2": "5_SigLIP2", "BioCLIP": "6_BioCLIP", "DINOv3": "7_DINOv3", "Qwen3VL": "8_Qwen3VL", "FLAVA": "9_FLAVA"}; INV = {v: k for k, v in ENC.items()}
P = lambda f: os.path.join(out, f)

if args.stage == "freeze":
    assert not os.path.exists(P("conditions.csv")), "plan already frozen"
    PQ = pd.read_csv(os.path.join(ATLAS, "analysis", "tables", "per_query_topk.csv.gz"), usecols=["encoder", "layer", "K", "transition_vs_pooled_vit"]); PQ = PQ[PQ.K == K]
    PQ["d"] = PQ.transition_vs_pooled_vit.map({"0 to >=1": 1, ">=1 to 0": -1}).fillna(0); R = PQ.groupby(["encoder", "layer"], sort=False).d.agg(net=lambda x: 100 * x.mean(), gained=lambda x: int((x == 1).sum()), lost=lambda x: int((x == -1).sum())).reset_index()
    rows = [{"condition": "zero_shot", "encoder": "", "layer": "", "K": 0, "rank_in_encoder": "", "retrieval_net_coverage_pp": "", "retrieval_gained": "", "retrieval_lost": ""},
            {"condition": "pooled_vit", "encoder": "ViT", "layer": "final_output", "K": K, "rank_in_encoder": "", "retrieval_net_coverage_pp": 0.0, "retrieval_gained": 0, "retrieval_lost": 0}]
    for e in ENC.values():
        g = R[(R.encoder == e) & (R.layer != "final_output")].reset_index(drop=True); g["order"] = range(len(g)); g = g.sort_values(["net", "order"], ascending=[False, True]).head(TOPN)          # earlier layer on ties
        rows += [{"condition": f"{INV[e]}|{r.layer}", "encoder": INV[e], "layer": r.layer, "K": K, "rank_in_encoder": i + 1, "retrieval_net_coverage_pp": round(r.net, 2), "retrieval_gained": r.gained, "retrieval_lost": r.lost} for i, r in enumerate(g.itertuples())]
    Cn = pd.DataFrame(rows); Cn.to_csv(P("conditions.csv"), index=False)
    json.dump({"frozen_at": time.strftime("%Y-%m-%d %H:%M:%S"), "question": "does the retrieval gain of a layer over the pooled ViT baseline change the accuracy of the VLM at the same shot budget?", "K": K, "model": MODEL,
               "queries": "the 500 queries of the atlas (100 per dataset) with the disjoint reference bank of 100 images per dataset (runs/disjoint_split_seed42)",
               "conditions": f"zero-shot (K = 0); pooled ViT at K = {K}; for each of nine encoders the {TOPN} hidden layers with the largest net same-class coverage gain over the pooled ViT at K = {K} (earlier layer on ties)",
               "prompt": "standard assisted prompt of the main run: task instruction, the K reference images in rank order each followed by its label as {\"prediction\": label}, the instruction again, the query image",
               "primary_analysis": "accuracy of each layer condition against the pooled ViT at the same K on the same queries: wins, losses, exact McNemar test; no correction for the 45 comparisons is applied to single p-values, and the number of comparisons is reported",
               "secondary_analysis": ["each condition against zero-shot", "accuracy and paired counts by dataset", "relation between the retrieval net coverage change of a layer and its VLM accuracy change (Spearman correlation over the 45 layer conditions)", "macro-F1 per dataset"],
               "known_limits": ["the layers are selected by retrieval gain on the same queries that are then evaluated, so the test asks whether a retrieval gain on these queries carries over to the VLM on these queries", "queries and references are not separated by capture date", "one VLM, one request per prompt"],
               "caching": "responses are stored by the SHA-256 of the full request; identical requests are sent once; zero-shot requests that are identical to the main run reuse its stored answers"}, open(P("preregistration.json"), "w"), indent=1)
    print(Cn.to_string()); sys.exit()

if args.stage == "amend":          # 2026-11-15, after the first results: the pooled (final) output of every encoder, so that each layer is compared with the pooled output of its own model
    Cn = pd.read_csv(P("conditions.csv"), keep_default_na=False, dtype=str); new = [{"condition": f"{e}|final_output", "encoder": e, "layer": "final_output", "K": str(K), "rank_in_encoder": "pooled"} for e in ENC if e != "ViT" and f"{e}|final_output" not in set(Cn.condition)]
    pd.concat([Cn, pd.DataFrame(new)]).fillna("").to_csv(P("conditions.csv"), index=False); pr = json.load(open(P("preregistration.json")))
    pr.setdefault("amendments", []).append({"at": time.strftime("%Y-%m-%d %H:%M:%S"), "made_after_seeing_results": True, "change": f"added {len(new)} conditions: the pooled (final) output of each non-ViT encoder at K = {K}; the primary comparison of a layer is now with the pooled output of its own encoder; the comparison with the pooled ViT is kept as a secondary column", "note": "the five layers per encoder were selected by retrieval gain over the pooled ViT, not over the pooled output of their own encoder"})
    json.dump(pr, open(P("preregistration.json"), "w"), indent=1); print(f"added {len(new)} conditions"); sys.exit()

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
        for rid, rlab in refs[(q.dataset, q.query_folder, c.encoder, c.layer)]:
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
for k in PL.payload.unique():                      # answers of the main run for identical requests (zero-shot)
    src = os.path.join(MAIN, "vlm", "responses", k + ".json")
    if not os.path.exists(rp(k)) and os.path.exists(src): shutil.copyfile(src, rp(k)); reused += 1
todo = [k for k in PL.payload.unique() if not os.path.exists(rp(k))]
print(f"conditions {len(Cn)}, query-condition pairs {len(PL)}, distinct requests {PL.payload.nunique()}, reused from the main run {reused}, already answered {PL.payload.nunique() - len(todo)}, to send {len(todo)}", flush=True)

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

# ------------------------------------------------------------------ report
from scipy import stats
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
plt.rcParams.update({"font.family": "Helvetica", "font.size": 7, "axes.spines.top": False, "axes.spines.right": False, "axes.linewidth": 0.6, "pdf.fonttype": 42})
COL = {"ViT": "#0072B2", "DINOv2": "#D55E00", "CLIP": "#009E73", "ResNet50": "#CC79A7", "SigLIP2": "#E69F00", "BioCLIP": "#56B4E9", "DINOv3": "#000000", "Qwen3VL": "#8C5A2B", "FLAVA": "#7F7F7F"}
NM = {"ViT": "ViT", "DINOv2": "DINOv2", "CLIP": "CLIP", "ResNet50": "ResNet-50", "SigLIP2": "SigLIP 2", "BioCLIP": "BioCLIP", "DINOv3": "DINOv3", "Qwen3VL": "Qwen3-VL tower", "FLAVA": "FLAVA"}; SEED, NB = 20261115, 10000
def pred(k):
    if not os.path.exists(rp(k)): return None
    try: return str(inf.extract_json(json.load(open(rp(k)))["raw"])["prediction"])
    except Exception: return "INVALID"
PL["pred"] = PL.payload.map({k: pred(k) for k in PL.payload.unique()}); PL["answered"] = PL.pred.notna(); PL["ok"] = PL.pred == PL.true_label
W = PL.pivot(index=["dataset", "query_folder"], columns="condition", values="ok"); AN = PL.pivot(index=["dataset", "query_folder"], columns="condition", values="answered"); full = AN.all(axis=1); W = W[full].astype(bool); n = len(W)
PL[["condition", "dataset", "query_folder", "true_label", "pred", "payload"]].to_csv(P("per_query_predictions.csv"), index=False)
C = np.random.default_rng(SEED).multinomial(n, np.ones(n) / n, size=NB).astype(np.float32)
def paired(a, b):
    d = W[a].astype(int).values - W[b].astype(int).values; w_, l_ = int((d == 1).sum()), int((d == -1).sum()); est = (d.astype(np.float32) @ C.T) / n * 100
    return {"acc": 100 * W[a].mean(), "delta": 100 * d.mean(), "lo": float(np.percentile(est, 2.5)), "hi": float(np.percentile(est, 97.5)), "wins": w_, "losses": l_, "p": 1.0 if w_ + l_ == 0 else float(stats.binomtest(w_, w_ + l_, 0.5).pvalue)}
# retrieval: does the top-K hold a same-class reference?  (recomputed here so that a layer can be compared with the pooled output of its own encoder)
tl = {(r.dataset, r.query_folder): r.true_label for r in PL.drop_duplicates(["dataset", "query_folder"]).itertuples()}
cov = lambda c: pd.Series({ix: any(l == tl[ix] for _, l in refs[(ix[0], ix[1], c.encoder, c.layer)]) for ix in W.index}).astype(int)
COV = {c.condition: cov(c) for c in Cn.itertuples() if c.encoder}
own = lambda c: "pooled_vit" if c.encoder == "ViT" else f"{c.encoder}|final_output"
rows = []
for c in Cn.itertuples():
    r = {"condition": c.condition, "encoder": c.encoder, "layer": c.layer, "K": c.K, "n": n, "accuracy": 100 * W[c.condition].mean()}
    if c.encoder:
        b = own(c); vo = paired(c.condition, b); r.update(own_pooled=b, own_pooled_accuracy=100 * W[b].mean(), retrieval_coverage=100 * COV[c.condition].mean(), retrieval_net_coverage_vs_own_pooled_pp=100 * (COV[c.condition] - COV[b]).mean(),
                                                            delta_vs_own_pooled_pp=vo["delta"], ci_lo=vo["lo"], ci_hi=vo["hi"], wins_vs_own_pooled=vo["wins"], losses_vs_own_pooled=vo["losses"], p_vs_own_pooled=vo["p"])
        for d, g in W.groupby(level=0): r[f"delta_vs_own_pooled_{d}"] = 100 * (g[c.condition].astype(int) - g[b].astype(int)).mean()
    vb = paired(c.condition, "pooled_vit"); vz = paired(c.condition, "zero_shot")
    r.update(delta_vs_pooled_vit_pp=vb["delta"], vit_ci_lo=vb["lo"], vit_ci_hi=vb["hi"], wins_vs_pooled_vit=vb["wins"], losses_vs_pooled_vit=vb["losses"], p_vs_pooled_vit=vb["p"], delta_vs_zero_shot_pp=vz["delta"], zs_ci_lo=vz["lo"], zs_ci_hi=vz["hi"], wins_vs_zero_shot=vz["wins"], losses_vs_zero_shot=vz["losses"], p_vs_zero_shot=vz["p"])
    for d, g in W.groupby(level=0): r[f"acc_{d}"] = 100 * g[c.condition].mean()
    rows.append(r)
R = pd.DataFrame(rows); R.to_csv(P("results_by_condition.csv"), index=False); zs = R[R.condition == "zero_shot"].iloc[0]
PO = R[(R.layer == "final_output")].set_index("encoder").loc[list(ENC)].reset_index(); LY = R[(R.encoder != "") & (R.layer != "final_output")].copy(); LY["ln"] = [0 if "stem" in l else int(l.split("_")[1]) for l in LY.layer]
ll = lambda r: "stem" if "stem" in r.layer else f"L{int(r.layer.split('_')[1])}"; lab = lambda r: f"{NM[r.encoder]} {ll(r)}"
rho, prho = stats.spearmanr(LY.retrieval_net_coverage_vs_own_pooled_pp, LY.delta_vs_own_pooled_pp); INK, MUT = "#222222", "#666666"
# figure 1: one panel per encoder, each layer against the pooled output of the same encoder
ylo, yhi = min(LY.ci_lo.min(), -1) - 1, max(LY.ci_hi.max(), 1) + 1; fig, axs = plt.subplots(3, 3, figsize=(7.2, 5.2), sharey=True)
for a, e in zip(axs.ravel(), ENC):
    g = LY[LY.encoder == e].sort_values("ln").reset_index(drop=True); p = PO[PO.encoder == e].iloc[0]; a.axhline(0, color="#444", lw=0.7)
    for i, r in g.iterrows():
        a.plot([i, i], [r.ci_lo, r.ci_hi], color=COL[e], lw=1.4, solid_capstyle="round"); a.scatter([i], [r.delta_vs_own_pooled_pp], s=26, color=COL[e], zorder=3, edgecolors="white", linewidths=0.6)
        a.text(i, r.ci_hi + 0.5, f"{r.delta_vs_own_pooled_pp:+.1f}", ha="center", va="bottom", fontsize=5.6, color=INK)
    a.set_xticks(range(len(g)), [ll(r) for r in g.itertuples()], fontsize=6.2); a.set_xlim(-0.6, len(g) - 0.4); a.set_ylim(ylo, yhi + 1.5); a.grid(axis="y", color="#ECECEC", lw=0.5); a.set_axisbelow(True); a.tick_params(length=2)
    a.set_title(f"{NM[e]}", fontsize=7.5, loc="left", fontweight="bold", color=INK, pad=11); a.text(0, 1.03, f"own pooled output: {p.accuracy:.1f}%", transform=a.transAxes, fontsize=6, color=MUT)
for a in axs[:, 0]: a.set_ylabel("Accuracy change against\nown pooled output (pp)")
for a in axs[-1]: a.set_xlabel("Hidden layer")
fig.suptitle(f"VLM accuracy of a hidden layer against the pooled output of the same encoder, K = {K}, {n} paired queries, 95% bootstrap intervals", fontsize=7, x=0.01, ha="left", color=INK)
fig.tight_layout(rect=(0, 0, 1, 0.97), h_pad=1.4); fig.savefig(P("fig_accuracy_change_by_layer.png"), dpi=600); fig.savefig(P("fig_accuracy_change_by_layer.pdf")); plt.close(fig)
# figure 2: a, retrieval gain against VLM gain, both against the own pooled output;  b, pooled output and best layer of each encoder as accuracies
best = LY.loc[LY.groupby("encoder", sort=False).delta_vs_own_pooled_pp.idxmax()].set_index("encoder").loc[list(ENC)].reset_index()
fig, ax = plt.subplots(1, 2, figsize=(7.2, 3.1), gridspec_kw={"width_ratios": [1, 1.2]}); a = ax[0]
for e in ENC: g = LY[LY.encoder == e]; a.scatter(g.retrieval_net_coverage_vs_own_pooled_pp, g.delta_vs_own_pooled_pp, s=20, color=COL[e], edgecolors="white", linewidths=0.5, label=NM[e], zorder=3)
sl_, ic_ = np.polyfit(LY.retrieval_net_coverage_vs_own_pooled_pp, LY.delta_vs_own_pooled_pp, 1); xs = np.array([LY.retrieval_net_coverage_vs_own_pooled_pp.min(), LY.retrieval_net_coverage_vs_own_pooled_pp.max()]); a.plot(xs, sl_ * xs + ic_, color="#444", lw=0.8, ls=(0, (3, 2))); a.axhline(0, color="#999", lw=0.5); a.axvline(0, color="#999", lw=0.5)
a.set_xlabel("Retrieval: same-class coverage change\nagainst own pooled output (pp)"); a.set_ylabel("VLM: accuracy change\nagainst own pooled output (pp)"); a.set_title(f"Spearman correlation {rho:+.2f} (p = {prho:.3f}), {len(LY)} layers", fontsize=7, loc="left"); a.legend(frameon=False, fontsize=5.4, ncol=2, loc="best"); a.text(-0.22, 1.05, "a", transform=a.transAxes, fontsize=9, fontweight="bold")
a = ax[1]; a.axvline(zs.accuracy, color="#999", lw=0.8, ls=(0, (3, 2))); a.text(zs.accuracy + 0.25, -0.75, f"zero-shot {zs.accuracy:.1f}%", fontsize=6, color=MUT, ha="left")
for i, e in enumerate(ENC):
    p, b = PO[PO.encoder == e].iloc[0], best[best.encoder == e].iloc[0]; a.plot([p.accuracy, b.accuracy], [i, i], color=COL[e], lw=1.6, solid_capstyle="round", zorder=2)
    a.scatter([p.accuracy], [i], s=30, facecolors="white", edgecolors=COL[e], linewidths=1.2, zorder=3); a.scatter([b.accuracy], [i], s=30, color=COL[e], edgecolors="white", linewidths=0.6, zorder=4)
    a.text(max(p.accuracy, b.accuracy) + 0.5, i, f"{ll(b)}  {b.delta_vs_own_pooled_pp:+.1f} pp", va="center", fontsize=6, color=INK)
a.set_yticks(range(len(ENC)), [NM[e] for e in ENC], fontsize=6.5); a.set_ylim(len(ENC) + 0.6, -1.1); a.set_xlim(min(zs.accuracy, PO.accuracy.min(), best.accuracy.min()) - 1.5, max(PO.accuracy.max(), best.accuracy.max()) + 5); a.set_xlabel("VLM accuracy (%)"); a.grid(axis="x", color="#ECECEC", lw=0.5); a.set_axisbelow(True); a.tick_params(axis="y", length=0)
a.legend(handles=[plt.Line2D([], [], color=INK, marker="o", mfc="white", ms=4.5, lw=0, label="pooled output"), plt.Line2D([], [], color=INK, marker="o", ms=4.5, lw=0, label="best of five hidden layers")], frameon=False, fontsize=6, loc="lower right", ncol=2, columnspacing=1.0, handletextpad=0.2); a.set_title("Pooled output and best hidden layer of each encoder", fontsize=7, loc="left"); a.text(-0.3, 1.05, "b", transform=a.transAxes, fontsize=9, fontweight="bold")
fig.tight_layout(); fig.savefig(P("fig_retrieval_vs_vlm.png"), dpi=600); fig.savefig(P("fig_retrieval_vs_vlm.pdf")); plt.close(fig)
# summary
S = LY.sort_values("delta_vs_own_pooled_pp", ascending=False); sig_up, sig_dn = LY[LY.ci_lo > 0], LY[LY.ci_hi < 0]; led = P("request_ledger.jsonl"); cost = sum(json.loads(l).get("cost") or 0 for l in open(led)) if os.path.exists(led) else 0; sent = sum(1 for _ in open(led)) if os.path.exists(led) else 0
item = lambda r: f"{lab(r)} {r.accuracy:.1f}% against {r.own_pooled_accuracy:.1f}% ({r.delta_vs_own_pooled_pp:+.1f} pp, CI {r.ci_lo:+.1f} to {r.ci_hi:+.1f}, {int(r.wins_vs_own_pooled)} wins, {int(r.losses_vs_own_pooled)} losses)"
L = [f"# Downstream test at K = {K}: does a hidden layer change the answers of the VLM, compared with the pooled output of the same encoder?\n",
     f"VLM `{MODEL}` through the headless client, standard assisted prompt. {n} of {len(AN)} queries are answered in all {len(Cn)} conditions" + ("" if n == len(AN) else f" ({len(AN) - n} queries wait for answers)") + f". Requests sent: {sent} (recorded API-equivalent cost ${cost:.2f}); zero-shot answers reused from the main run: 493. Every layer is compared with the pooled (final) output of its own encoder on the same queries; the comparison with the pooled ViT is kept in `results_by_condition.csv`.\n",
     "![Accuracy change by layer](fig_accuracy_change_by_layer.png)\n",
     f"**Fig. 1 | VLM accuracy of each hidden layer against the pooled output of the same encoder at K = {K}.** One panel per encoder. Points: accuracy change in percentage points over {n} paired queries. Bars: 95% paired bootstrap intervals ({NB:,} resamples).\n",
     "## Result\n",
     f"- **Layers against their own pooled output:** {len(sig_up)} of {len(LY)} layers are above it with an interval that excludes zero, {len(sig_dn)} are below it, and {len(LY) - len(sig_up) - len(sig_dn)} cannot be told apart from it.",
     ("- **Above:** " + "; ".join(item(r) for r in sig_up.sort_values("delta_vs_own_pooled_pp", ascending=False).itertuples()) + ".") if len(sig_up) else "- **Above:** none.",
     ("- **Below:** " + "; ".join(item(r) for r in sig_dn.sort_values("delta_vs_own_pooled_pp").itertuples()) + ".") if len(sig_dn) else "- **Below:** none.",
     "- **Largest gains:** " + "; ".join(item(r) for r in S.head(5).itertuples()) + ".",
     f"- **Does the retrieval gain predict the VLM gain?** Over the {len(LY)} layers the Spearman correlation between the same-class coverage change and the VLM accuracy change, both against the own pooled output, is {rho:+.2f} (p = {prho:.3f}).",
     f"- **Multiplicity:** the comparisons are {len(LY)} tests without a correction; a single interval that excludes zero is weak evidence.\n",
     "![Retrieval gain against VLM gain](fig_retrieval_vs_vlm.png)\n",
     "**Fig. 2 | Retrieval gain and VLM gain.** **a**, Each point is one hidden layer: its same-class coverage change (retrieval) and its accuracy change (VLM), both against the pooled output of the same encoder. **b**, VLM accuracy with the pooled output of each encoder (open circle) and with its best hidden layer of the five tested (filled circle). Dashed line: zero-shot prompting.\n",
     "## Pooled output of each encoder\n", "| Encoder | Retrieval coverage (%) | VLM accuracy (%) | Change vs zero-shot (pp) | 95% CI | Wins / losses | p | Change vs pooled ViT (pp) |", "|---|---|---|---|---|---|---|---|",
     f"| Zero-shot (K = 0) | | {zs.accuracy:.1f} | | | | | {zs.delta_vs_pooled_vit_pp:+.1f} |"]
for r in PO.itertuples(): L.append(f"| {NM[r.encoder]} | {r.retrieval_coverage:.1f} | {r.accuracy:.1f} | {r.delta_vs_zero_shot_pp:+.1f} | {r.zs_ci_lo:+.1f} to {r.zs_ci_hi:+.1f} | {int(r.wins_vs_zero_shot)} / {int(r.losses_vs_zero_shot)} | {r.p_vs_zero_shot:.3f} | {r.delta_vs_pooled_vit_pp:+.1f} |")
L += ["\n## Best hidden layer of each encoder against its own pooled output\n", "| Encoder | Layer | Retrieval coverage change (pp) | Pooled accuracy (%) | Layer accuracy (%) | Change (pp) | 95% CI | Wins / losses | p |", "|---|---|---|---|---|---|---|---|---|"]
for r in best.itertuples(): L.append(f"| {NM[r.encoder]} | {ll(r)} | {r.retrieval_net_coverage_vs_own_pooled_pp:+.1f} | {r.own_pooled_accuracy:.1f} | {r.accuracy:.1f} | {r.delta_vs_own_pooled_pp:+.1f} | {r.ci_lo:+.1f} to {r.ci_hi:+.1f} | {int(r.wins_vs_own_pooled)} / {int(r.losses_vs_own_pooled)} | {r.p_vs_own_pooled:.3f} |")
L += ["\nThe best layer of an encoder is picked from five on the same queries, so its change is optimistic.\n", "## Limits\n",
      "- The five layers per encoder were selected by their retrieval gain over the pooled ViT on these queries, before the own-pooled comparison was added; the pooled conditions of the other eight encoders were added after the first results were seen (recorded in `preregistration.json`).", "- Queries and references are not separated by capture date.", "- One VLM and one request per prompt.\n",
      "Files: `results_by_condition.csv` (every condition, with dataset columns), `per_query_predictions.csv`, `conditions.csv`, `preregistration.json`, `request_plan.csv`, `responses/`.\n"]
open(P("SUMMARY.md"), "w").write("\n".join(L)); print("\n".join(L))
