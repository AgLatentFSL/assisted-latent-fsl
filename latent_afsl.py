"""Latent-Assisted Few-Shot Learning: which layer and token of a frozen encoder should retrieve the demonstrations?

Stages (each idempotent; outputs in the run folder given as argv[1]):
  dev      : freeze a DeepWeeds date-disjoint development split, extract features, select one ViT hidden-state readout
             (and one DINOv2 alternative) by development 4-NN macro-F1, and write manifest.json BEFORE any test metric.
  test     : retrieval on the held-out panel (180 queries) with the audit's date-disjoint and same-date banks.
  ageval   : retrieval on the six AgEval evaluation pools (historical protocol, no source-group split).
The VLM stages live in latent_afsl_vlm.py. No stage here calls a VLM.
Usage: python latent_afsl.py <run_dir> <dev|test|ageval>
"""
import json, os, sys, time
import numpy as np
import pandas as pd
import torch
from PIL import Image
from scipy import stats
from sklearn.metrics import f1_score
from pilot_matched_labels import DATA, md5_file, path_of, vote

RUNS = os.path.join("runs")
AUDIT, BASE, PRIOR = os.path.join(RUNS, "deepweeds_split_audit"), os.path.join(RUNS, "deepweeds_matched_seed42"), os.path.join(RUNS, "deepweeds_seed42")
K, SEED, DEV_Q_PER_CLASS, BANK_PER_CLASS, N_DEV_BANKS, NEAR_SECONDS = 4, 42, 20, 11, 10, 120
VIT, DINO = "google/vit-base-patch16-224-in21k", "facebook/dinov2-base"
# candidate order is the final tie-breaker of the selection rule
VIT_CANDS = [("vit", "cls", b) for b in range(1, 13)] + [("vit", "mean", b) for b in range(1, 13)]
DINO_CANDS = [("dinov2", "cls", b) for b in range(1, 13)] + [("dinov2", "mean", b) for b in range(1, 13)]
A = ("vit", "pooled", 0); B = ("vit", "cls", 12); B_RAW = ("vit", "clsraw", 12); D = ("vit", "mean", 12); DINO_DEFAULT = ("dinov2", "cls", 12)


def rname(r):
    m = {"vit": "ViT", "dinov2": "DINOv2"}[r[0]]
    return {"pooled": f"{m} pooler_output (Arshad et al.)", "clsraw": f"{m} CLS, block {r[2]}, before final LayerNorm",
            "cls": f"{m} CLS, block {r[2]}", "mean": f"{m} patch mean, block {r[2]}"}[r[1]]


def labels_frame():
    L = pd.read_csv(os.path.join(DATA, "labels", "labels.csv")).sort_values("Filename").reset_index(drop=True)
    p = L["Filename"].str.extract(r"(\d{8})-(\d{6})-(\d+)\.jpg")
    L["date"], L["cam"] = p[0], p[2]; L["t"] = pd.to_datetime(p[0] + p[1], format="%Y%m%d%H%M%S")
    return L


def freeze_dev(out):
    qp, bp = os.path.join(out, "dev_queries.csv"), os.path.join(out, "dev_banks.csv")
    if os.path.exists(qp) and os.path.exists(bp):
        return pd.read_csv(qp, dtype={"capture_date": str}), pd.read_csv(bp, dtype={"capture_date": str})
    L = labels_frame()
    q_dates = set(json.load(open(os.path.join(BASE, "split.json")))["query_dates"])           # test-query dates
    panel = pd.read_csv(os.path.join(AUDIT, "query_panel.csv"))
    base_bank = set(pd.read_csv(os.path.join(BASE, "reference_manifest.csv"))["ref_id"])
    base_q = set(pd.read_csv(os.path.join(BASE, "query_manifest.csv"))["query_id"])
    prior = set(pd.read_csv(os.path.join(PRIOR, "reference_manifest.csv"))["image_id"].str.split("/").str[-1])
    test_q = set(panel["query_id"]) | base_q
    near = np.zeros(len(L), dtype=bool)
    for f in test_q:
        r = L[L["Filename"] == f].iloc[0]
        near |= ((L["date"] == r["date"]) & (L["cam"] == r["cam"]) & ((L["t"] - r["t"]).abs().dt.total_seconds() <= NEAR_SECONDS)).to_numpy()
    L["blocked"] = near | L["Filename"].isin(test_q | prior)
    classes = sorted(L["Species"].unique()); rng = np.random.default_rng([SEED, 101])
    # development runs in the reverse direction of the test: queries on the bank-side dates, banks on the query-side dates
    dq = []
    for c in classes:
        pool = L[~L["date"].isin(q_dates) & (L["Species"] == c) & ~L["Filename"].isin(base_bank | prior)].sort_values("Filename")
        dq.append(pool.iloc[np.sort(rng.choice(len(pool), size=DEV_Q_PER_CLASS, replace=False))])
    DQ = pd.concat(dq).reset_index(drop=True)
    banks = []
    for s in range(N_DEV_BANKS):
        r = np.random.default_rng([SEED, 102, s]); part = []
        for c in classes:
            pool = L[L["date"].isin(q_dates) & (L["Species"] == c) & ~L["blocked"]].sort_values("Filename")
            part.append(pool.iloc[np.sort(r.choice(len(pool), size=BANK_PER_CLASS, replace=False))])
        b = pd.concat(part).sample(frac=1.0, random_state=s).reset_index(drop=True)
        banks.append(pd.DataFrame({"bank_seed": s, "bank_idx": range(len(b)), "ref_id": b["Filename"].to_numpy(), "label": b["Species"].to_numpy(), "capture_date": b["date"].to_numpy()}))
    DB = pd.concat(banks, ignore_index=True)
    dqm = pd.DataFrame({"query_id": DQ["Filename"], "label": DQ["Species"], "capture_date": DQ["date"]})
    assert not set(dqm["capture_date"]) & set(DB["capture_date"]) and not set(DB["ref_id"]) & test_q and not set(dqm["query_id"]) & test_q
    dqm.to_csv(qp, index=False); DB.to_csv(bp, index=False)
    return dqm, DB


def extract(out, model_name, ids):
    path = os.path.join(out, "features", f"{model_name}.npz"); os.makedirs(os.path.dirname(path), exist_ok=True)
    have = dict(np.load(path, allow_pickle=False)) if os.path.exists(path) else {}
    done = set(have["ids"].tolist()) if "ids" in have else set()
    todo = [f for f in ids if f not in done]
    if not todo:
        return have
    from transformers import AutoImageProcessor, AutoModel, ViTModel
    mid = VIT if model_name == "vit" else DINO
    proc = AutoImageProcessor.from_pretrained(mid); model = (ViTModel if model_name == "vit" else AutoModel).from_pretrained(mid).eval()
    acc = {k: [have[k]] if k in have else [] for k in ("cls", "mean", "pooled", "clsraw")}
    idl = list(have["ids"]) if "ids" in have else []
    BS_ = 32 if model_name == "vit" else 4      # DINOv2 runs much slower in large CPU batches
    for i in range(0, len(todo), BS_):
        chunk = todo[i:i + BS_]
        x = proc(images=[Image.open(path_of(f)).convert("RGB") for f in chunk], return_tensors="pt")
        with torch.no_grad():
            o = model(**x, output_hidden_states=True)
            hs = torch.stack([model.layernorm(h) for h in o.hidden_states[1:]], dim=1)      # batch x 12 x tokens x dim, final LayerNorm applied
            assert hs.shape[1] == 12 and torch.allclose(hs[:, 11], o.last_hidden_state, atol=1e-5)
        acc["cls"].append(hs[:, :, 0].numpy().astype(np.float16)); acc["mean"].append(hs[:, :, 1:].mean(dim=2).numpy().astype(np.float16))   # token 0 is CLS; all other tokens are patches
        acc["clsraw"].append(o.hidden_states[12][:, 0].numpy().astype(np.float32))
        if model_name == "vit":
            acc["pooled"].append(o.pooler_output.numpy().astype(np.float32))
        idl += chunk
        if (i // BS_) % (800 // BS_) == (800 // BS_) - 1 or i + BS_ >= len(todo):
            np.savez(path, ids=np.array(idl), **{k: np.concatenate(v) for k, v in acc.items() if v})
            print(f"{model_name}: {len(idl)} images", flush=True)
    return dict(np.load(path, allow_pickle=False))


class Feats:
    def __init__(self, out, ids):
        self.F = {m: extract(out, m, ids) for m in ("vit", "dinov2")}
        self.ix = {m: {f: i for i, f in enumerate(self.F[m]["ids"].tolist())} for m in self.F}
    def get(self, r, ids):
        F, ix = self.F[r[0]], self.ix[r[0]]; rows = [ix[f] for f in ids]
        X = (F["pooled"][rows] if r[1] == "pooled" else F["clsraw"][rows] if r[1] == "clsraw" else F[r[1]][rows, r[2] - 1]).astype(np.float64)
        return X / np.linalg.norm(X, axis=1, keepdims=True)


def retrieve(Qx, Bx, blab, qlab):
    """Top-4 by descending cosine, ties to the lower bank index; vote ties by summed cosine then alphabetical order."""
    S = Qx @ Bx.T; out = []
    for q in range(len(qlab)):
        top = np.lexsort((np.arange(S.shape[1]), -S[q]))[:K]; labs = [blab[j] for j in top]; sims = [float(S[q, j]) for j in top]
        cnt = sorted(pd.Series(labs).value_counts().tolist(), reverse=True)
        out.append({"top": top.tolist(), "labels": labs, "cos": sims, "vote": vote(labs, sims), "margin": cnt[0] - (cnt[1] if len(cnt) > 1 else 0),
                    "same": sum(l == qlab[q] for l in labs), "top1": int(labs[0] == qlab[q])})
    return out


def summarize(res, qlab, classes):
    pred = [r["vote"] for r in res]
    return {"acc": float(np.mean([p == t for p, t in zip(pred, qlab)])), "macro_f1": float(f1_score(qlab, pred, labels=classes, average="macro", zero_division=0)),
            "top1": float(np.mean([r["top1"] for r in res])), "same_frac": float(np.mean([r["same"] for r in res]) / K)}


def all_ids(out):
    dqm, DB = freeze_dev(out)
    panel = pd.read_csv(os.path.join(AUDIT, "query_panel.csv")); banks = pd.read_csv(os.path.join(AUDIT, "banks.csv"))
    base_bank = pd.read_csv(os.path.join(BASE, "reference_manifest.csv")); base_q = pd.read_csv(os.path.join(BASE, "query_manifest.csv"))
    ids = sorted(set(dqm["query_id"]) | set(DB["ref_id"]) | set(panel["query_id"]) | set(banks["ref_id"]) | set(base_bank["ref_id"]) | set(base_q["query_id"]))
    return ids, dqm, DB, panel, banks, base_bank, base_q


def stage_dev(out):
    ids, dqm, DB, panel, banks, base_bank, base_q = all_ids(out)
    mpath = os.path.join(out, "manifest.json")
    FT = Feats(out, ids)
    classes = sorted(dqm["label"].unique()); qlab = dqm["label"].tolist(); rows = []
    for r in [A, B_RAW] + VIT_CANDS + DINO_CANDS:
        Qx = FT.get(r, dqm["query_id"]); ms = []
        for s, b in DB.groupby("bank_seed"):
            b = b.sort_values("bank_idx"); ms.append(summarize(retrieve(Qx, FT.get(r, b["ref_id"]), b["label"].tolist(), qlab), qlab, classes))
        rows.append({"readout": rname(r), "model": r[0], "token": r[1], "block": r[2], **{k: float(np.mean([m[k] for m in ms])) for k in ms[0]}, "macro_f1_sd_over_banks": float(np.std([m["macro_f1"] for m in ms]))})
    Dv = pd.DataFrame(rows); Dv.to_csv(os.path.join(out, "dev_retrieval.csv"), index=False)
    def pick(cands):
        sub = Dv[[(m, t, b) in cands for m, t, b in zip(Dv.model, Dv.token, Dv.block)]].reset_index(drop=True)
        sub["order"] = range(len(sub))
        w = sub.sort_values(["macro_f1", "acc", "order"], ascending=[False, False, True]).iloc[0]
        return (w["model"], w["token"], int(w["block"])), w
    if not os.path.exists(mpath):
        hv, wv = pick(VIT_CANDS); hd, wd = pick(DINO_CANDS)
        dA, dD = Dv[(Dv.model == "vit") & (Dv.token == "pooled")].iloc[0], Dv[(Dv.model == "dinov2") & (Dv.token == "cls") & (Dv.block == 12)].iloc[0]
        json.dump({
            "written_at": time.strftime("%Y-%m-%d %H:%M:%S"), "written_before_test_metrics": not os.path.exists(os.path.join(out, "test_retrieval.csv")),
            "primary_question": "same ViT checkpoint, standard assisted prompt: pooler_output (Arshad et al.) versus the development-selected hidden-state readout",
            "primary_metric": "VLM accuracy on the held-out DeepWeeds test queries with the frozen date-disjoint bank; paired exact McNemar test and a paired bootstrap interval for the accuracy difference",
            "checkpoints": {"vit": VIT, "dinov2": DINO},
            "readouts": {"pooled": "ViTModel.pooler_output = tanh(Linear(LayerNorm(h12)[CLS]))", "cls": "CLS token of hidden_states[b] after the model's final LayerNorm (b = 12 is last_hidden_state[:, 0], the pooler input)",
                         "mean": "mean over all patch tokens (every token except CLS) of hidden_states[b] after the final LayerNorm", "clsraw": "CLS token of hidden_states[12] before the final LayerNorm",
                         "block_indexing": "hidden_states has 13 entries; index 0 is the embedding output and index b is the output of block b", "cosine": "all vectors are L2-normalized; cosine = dot product"},
            "candidates_vit": [list(c) for c in VIT_CANDS], "candidates_dinov2": [list(c) for c in DINO_CANDS],
            "selection_rule": "highest mean 4-NN macro-F1 over the development banks; ties by mean 4-NN accuracy, then by the candidate order listed here",
            "development_split": {"dataset": "DeepWeeds", "direction": "development queries on the bank-side dates of the test split, development banks on the query-side dates; no development image is a test query",
                                  "n_queries": len(dqm), "queries_per_class": DEV_Q_PER_CLASS, "banks": N_DEV_BANKS, "bank_size": int(DB.groupby("bank_seed").size().iloc[0]), "bank_per_class": BANK_PER_CLASS,
                                  "query_dates": sorted(dqm["capture_date"].unique()), "bank_dates": sorted(DB["capture_date"].unique()),
                                  "excluded_from_development": f"the 180 panel queries, the 20 queries of the base run, the prior 99-image pool, and images on a test query's date and camera within {NEAR_SECONDS} s of it; development queries also exclude the frozen base bank"},
            "test_split": {"queries": "audit panel roles eval (63) and audit_only (90): 153 queries on 18 query dates; the 27 dev-role queries are prompt-development queries and are excluded",
                           "bank_for_vlm": "frozen date-disjoint bank of the base run, 99 images, 11 per class", "banks_for_retrieval": "the audit's 20 date-disjoint and 20 same-date banks (99 images each)",
                           "status_of_the_63": "they were answered before by the VLM with DINOv2 block-12 retrieval (zero-shot and standard prompt); no ViT hidden-state readout was scored on them before this manifest",
                           "fresh_subset": "the 90 audit_only queries were never sent to a VLM; reported separately as a sensitivity analysis"},
            "tie_rules": {"ranking": "descending cosine, ties to the lower bank index", "vote": "most frequent label; ties by larger summed cosine, then alphabetical order"},
            "selected_vit": list(hv), "selected_vit_name": rname(hv), "selected_vit_dev": {k: float(wv[k]) for k in ("macro_f1", "acc", "top1", "same_frac")},
            "pooled_dev": {k: float(dA[k]) for k in ("macro_f1", "acc", "top1", "same_frac")},
            "selected_dinov2": list(hd), "selected_dinov2_name": rname(hd), "selected_dinov2_dev": {k: float(wd[k]) for k in ("macro_f1", "acc", "top1", "same_frac")},
            "dinov2_default_dev": {k: float(dD[k]) for k in ("macro_f1", "acc", "top1", "same_frac")},
            "prior_exposure": "a layer sweep and a 123-candidate search were run earlier on the AgEval pools (which include a DeepWeeds pool with same-date references); they motivated this design and are not used to select the readout"}, open(mpath, "w"), indent=1)
    M = json.load(open(mpath))
    print("selected ViT readout:", M["selected_vit_name"], M["selected_vit_dev"], "| pooled:", M["pooled_dev"])
    print("selected DINOv2 readout:", M["selected_dinov2_name"], M["selected_dinov2_dev"], "| default:", M["dinov2_default_dev"])
    print(Dv[Dv.model == "vit"].sort_values("macro_f1", ascending=False).head(8)[["readout", "macro_f1", "acc", "top1", "same_frac"]].round(3).to_string(index=False))
    print(Dv[Dv.model == "dinov2"].sort_values("macro_f1", ascending=False).head(5)[["readout", "macro_f1", "acc", "top1", "same_frac"]].round(3).to_string(index=False))


if __name__ == "__main__":
    out, stage = sys.argv[1], sys.argv[2]; os.makedirs(out, exist_ok=True)
    if stage == "dev":
        stage_dev(out)
    else:
        import latent_afsl_eval
        latent_afsl_eval.run(out, stage)
