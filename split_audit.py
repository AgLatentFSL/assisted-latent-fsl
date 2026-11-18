"""DeepWeeds split audit, no VLM calls.

Freezes a query panel on the query dates of the base date-disjoint split, then compares
reference banks that exclude the query dates (DD) with banks sampled from all dates (SD).
Both bank types have the same size and class counts, exclude every panel query and its
near-duplicates, and are resampled across seeds. Encoders: original ViT pooler_output and
DINOv2 block 12 CLS, through get_embeddingp.get_image_embedding.

Usage: python split_audit.py <out_dir> [--stage freeze|embed|analyze]
"""
import json, os, sys
import numpy as np
import pandas as pd
from PIL import Image
from scipy import stats
from pilot_matched_labels import DATA, md5_file, path_of, vote, wjson, fp

BASE = os.path.join("runs", "deepweeds_matched_seed42")
PRIOR = os.path.join("runs", "deepweeds_seed42")
SEED, PER_CLASS_Q, DEV_PER_CLASS, EVAL_PER_CLASS = 42, 20, 3, 7
BANK_PER_CLASS, N_BANK_SEEDS, K = 11, 20, 4
NEAR_SECONDS, NEAR_HAMMING = 120, 8
ENCODERS = {"vit": "ViT", "dinov2_last": "DINOv2 b12"}
NBOOT = 10000


def dhash(p):
    a = np.asarray(Image.open(p).convert("L").resize((9, 8), Image.BILINEAR), dtype=np.int16)
    return np.packbits((a[:, 1:] > a[:, :-1]).flatten())


def freeze(out):
    qpath, bpath = os.path.join(out, "query_panel.csv"), os.path.join(out, "banks.csv")
    if os.path.exists(qpath) and os.path.exists(bpath):
        return pd.read_csv(qpath, dtype={"capture_date": str}), pd.read_csv(bpath, dtype={"capture_date": str})
    L = pd.read_csv(os.path.join(DATA, "labels", "labels.csv")).sort_values("Filename").reset_index(drop=True)
    p = L["Filename"].str.extract(r"(\d{8})-(\d{6})-(\d+)\.jpg")
    L["date"], L["cam"] = p[0], p[2]
    L["t"] = pd.to_datetime(p[0] + p[1], format="%Y%m%d%H%M%S")
    split = json.load(open(os.path.join(BASE, "split.json")))
    q_dates = set(split["query_dates"])
    used = set(pd.read_csv(os.path.join(PRIOR, "reference_manifest.csv"))["image_id"].str.split("/").str[-1]) \
        | set(pd.read_csv(os.path.join(BASE, "query_manifest.csv"))["query_id"])           # shown to the VLM before
    classes = sorted(L["Species"].unique())
    rng = np.random.default_rng(SEED)
    rows = []
    for c in classes:
        pool = L[L["date"].isin(q_dates) & (L["Species"] == c) & ~L["Filename"].isin(used)].sort_values("Filename")
        pick = pool.iloc[rng.choice(len(pool), size=PER_CLASS_Q, replace=False)].copy()     # random order = role order
        pick["role"] = ["dev"] * DEV_PER_CLASS + ["eval"] * EVAL_PER_CLASS + ["audit_only"] * (PER_CLASS_Q - DEV_PER_CLASS - EVAL_PER_CLASS)
        rows.append(pick)
    Q = pd.concat(rows).sort_values("Filename").reset_index(drop=True)
    Q["md5"] = [md5_file(path_of(f)) for f in Q["Filename"]]

    # near-duplicates of any panel query, removed from every bank of both types
    H = np.stack([dhash(path_of(f)) for f in L["Filename"]])
    hq = H[L["Filename"].isin(set(Q["Filename"])).to_numpy()]
    ham = np.unpackbits(H[:, None, :] ^ hq[None, :, :], axis=2).sum(axis=2).min(axis=1)
    near_hash = ham <= NEAR_HAMMING
    near_time = np.zeros(len(L), dtype=bool)
    for _, q in Q.iterrows():
        near_time |= ((L["date"] == q["date"]) & (L["cam"] == q["cam"]) & ((L["t"] - q["t"]).abs().dt.total_seconds() <= NEAR_SECONDS)).to_numpy()
    L["is_query"] = L["Filename"].isin(set(Q["Filename"]))
    L["near_dup"] = (near_hash | near_time) & ~L["is_query"]
    elig = L[~L["is_query"] & ~L["near_dup"]]
    banks = []
    for seed in range(N_BANK_SEEDS):
        for btype, pool_all in (("DD", elig[~elig["date"].isin(q_dates)]), ("SD", elig)):
            r = np.random.default_rng([SEED, seed, 0 if btype == "DD" else 1])
            part = []
            for c in classes:
                pool = pool_all[pool_all["Species"] == c].sort_values("Filename")
                part.append(pool.iloc[np.sort(r.choice(len(pool), size=BANK_PER_CLASS, replace=False))])
            b = pd.concat(part).sample(frac=1.0, random_state=seed).reset_index(drop=True)   # index order unrelated to class
            banks.append(pd.DataFrame({"bank_seed": seed, "bank_type": btype, "bank_idx": range(len(b)), "ref_id": b["Filename"].to_numpy(),
                                       "label": b["Species"].to_numpy(), "capture_date": b["date"].to_numpy()}))
    B = pd.concat(banks, ignore_index=True)
    qm = pd.DataFrame({"query_id": Q["Filename"], "label": Q["Species"], "capture_date": Q["date"], "md5": Q["md5"], "role": Q["role"]})
    assert not set(qm["query_id"]) & set(B["ref_id"])
    assert not set(B[B["bank_type"] == "DD"]["capture_date"]) & set(qm["capture_date"])
    qm.to_csv(qpath, index=False); B.to_csv(bpath, index=False)
    sd = B[B["bank_type"] == "SD"]
    wjson(os.path.join(out, "freeze.json"), {
        "seed": SEED, "query_dates": sorted(q_dates), "panel_size": len(qm), "per_class": PER_CLASS_Q,
        "panel_rule": "20 per class from images on the base split's query dates, excluding the prior 99-image pool and the 20 queries of the base matched-label run",
        "roles": qm["role"].value_counts().to_dict(), "role_rule": "within each class, in sampling order: 3 dev, 7 eval, 10 audit-only",
        "bank_size": BANK_PER_CLASS * len(classes), "bank_per_class": BANK_PER_CLASS, "bank_seeds": N_BANK_SEEDS,
        "DD": "date-disjoint: sampled from images whose capture date is not a query date",
        "SD": "same-date allowed: sampled from images on all dates",
        "both_exclude": f"every panel query; images on a query's date and camera within {NEAR_SECONDS} s of it; images with 64-bit dHash Hamming distance <= {NEAR_HAMMING} to a query",
        "near_duplicates_removed": {"by_time": int((near_time & ~L['is_query']).sum()), "by_dhash": int((near_hash & ~L['is_query']).sum()), "total": int(L["near_dup"].sum())},
        "byte_identical_query_bank_pairs": 0, "panel_dates_per_class": qm.groupby("label")["capture_date"].nunique().to_dict(),
        "DD_bank_dates_per_class_available": elig[~elig["date"].isin(q_dates)].groupby("Species")["date"].nunique().to_dict(),
        "SD_share_of_bank_images_on_query_dates": float(sd["capture_date"].isin(q_dates).mean()),
        "unique_bank_images": int(B["ref_id"].nunique())})
    return qm, B


def embed(out, qm, B):
    from get_embeddingp import get_image_embedding
    need = sorted(set(qm["query_id"]) | set(B["ref_id"]))
    E = {}
    for enc in ENCODERS:
        path = os.path.join(out, f"embeddings_{enc}.npz")
        have = dict(np.load(path)) if os.path.exists(path) else {}
        todo = [f for f in need if f not in have]
        for i, f in enumerate(todo, 1):
            v = get_image_embedding(path_of(f), model_type=enc)
            assert not isinstance(v, dict), v
            have[f] = np.asarray(v, dtype=np.float32)
            if i % 500 == 0 or i == len(todo):
                np.savez(path, **have)
                print(f"{enc}: {i}/{len(todo)} embedded", flush=True)
        E[enc] = have
    return E


def analyze(out, qm, B, E):
    qlab = dict(zip(qm["query_id"], qm["label"])); qdate = dict(zip(qm["query_id"], qm["capture_date"]))
    rows = []
    for (seed, btype), b in B.groupby(["bank_seed", "bank_type"], sort=True):
        b = b.sort_values("bank_idx")
        lab, dat = b["label"].to_numpy(), b["capture_date"].to_numpy()
        for enc in ENCODERS:
            X = np.stack([E[enc][f] for f in b["ref_id"]]).astype(np.float64); X /= np.linalg.norm(X, axis=1, keepdims=True)
            Qv = np.stack([E[enc][f] for f in qm["query_id"]]).astype(np.float64); Qv /= np.linalg.norm(Qv, axis=1, keepdims=True)
            S = Qv @ X.T
            assert np.isfinite(S).all()
            for qi, f in enumerate(qm["query_id"]):
                top = sorted(range(len(b)), key=lambda j: (-S[qi, j], j))[:K]
                tl = [lab[j] for j in top]
                sd_mask = np.array([dat[j] == qdate[f] for j in top])
                same_cls = np.array([l == qlab[f] for l in tl])
                rows.append({"bank_seed": seed, "bank_type": btype, "encoder": enc, "query_id": f, "label": qlab[f],
                             "same_class": int(same_cls.sum()), "knn_correct": int(vote(tl, [S[qi, j] for j in top]) == qlab[f]),
                             "top4_same_date": int(sd_mask.sum()), "top4_same_date_same_class": int((sd_mask & same_cls).sum()),
                             "bank_same_date": int((dat == qdate[f]).sum()), "bank_same_date_same_class": int(((dat == qdate[f]) & (lab == qlab[f])).sum()),
                             "top1_cosine": float(S[qi, top[0]])})
    R = pd.DataFrame(rows)
    R.to_csv(os.path.join(out, "per_query_seed.csv"), index=False)
    rng = np.random.default_rng(SEED)

    def ci(d):
        idx = rng.integers(0, len(d), size=(NBOOT, len(d)))
        m = d[idx].mean(axis=1)
        return float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5))
    def wil(d):
        return 1.0 if not np.any(d != 0) else float(stats.wilcoxon(d).pvalue)
    # per-query means over bank seeds
    Pq = R.groupby(["bank_type", "encoder", "query_id"])[["same_class", "knn_correct"]].mean()
    Ps = R.groupby(["bank_type", "encoder", "bank_seed"])[["same_class", "knn_correct"]].mean()
    order = list(qm["query_id"])
    res = {"cells": {}, "SD_minus_DD": {}, "encoder_gap": {}}
    for bt in ("DD", "SD"):
        for enc in ENCODERS:
            res["cells"][f"{bt}|{enc}"] = {"same_class_mean": float(Pq.loc[(bt, enc), "same_class"].mean()), "knn_acc": float(Pq.loc[(bt, enc), "knn_correct"].mean()),
                                            "same_class_sd_over_seeds": float(Ps.loc[(bt, enc), "same_class"].std()), "knn_acc_sd_over_seeds": float(Ps.loc[(bt, enc), "knn_correct"].std()),
                                            "knn_acc_min_seed": float(Ps.loc[(bt, enc), "knn_correct"].min()), "knn_acc_max_seed": float(Ps.loc[(bt, enc), "knn_correct"].max())}
    D = {}
    for enc in ENCODERS:
        for m in ("same_class", "knn_correct"):
            d = (Pq.loc[("SD", enc), m].loc[order] - Pq.loc[("DD", enc), m].loc[order]).to_numpy()
            D[(enc, m)] = d
            seeds_pos = int(((Ps.loc[("SD", enc), m] - Ps.loc[("DD", enc), m]) > 0).sum())
            lo, hi = ci(d)
            res["SD_minus_DD"][f"{enc}|{m}"] = {"diff": float(d.mean()), "ci": [lo, hi], "wilcoxon_p": wil(d), "queries_higher": int((d > 0).sum()),
                                               "queries_lower": int((d < 0).sum()), "seeds_with_SD_higher": seeds_pos, "n_seeds": N_BANK_SEEDS}
    for bt in ("DD", "SD"):
        for m in ("same_class", "knn_correct"):
            d = (Pq.loc[(bt, "dinov2_last"), m].loc[order] - Pq.loc[(bt, "vit"), m].loc[order]).to_numpy()
            lo, hi = ci(d)
            res["encoder_gap"][f"{bt}|{m}"] = {"diff": float(d.mean()), "ci": [lo, hi], "wilcoxon_p": wil(d)}
    for m in ("same_class", "knn_correct"):
        d = D[("dinov2_last", m)] - D[("vit", m)]
        lo, hi = ci(d)
        res["encoder_gap"][f"interaction|{m}"] = {"diff": float(d.mean()), "ci": [lo, hi], "wilcoxon_p": wil(d)}
    sd = R[R["bank_type"] == "SD"]
    comp = {}
    for enc in ENCODERS:
        g = sd[sd["encoder"] == enc]
        n_sd, n_od = g["top4_same_date"].sum(), (K - g["top4_same_date"]).sum()
        has = g[g["top4_same_date"] > 0]; no = g[g["top4_same_date"] == 0]
        comp[enc] = {"share_of_top4_from_query_date": float(g["top4_same_date"].mean() / K),
                     "same_class_rate_of_same_date_refs": float(g["top4_same_date_same_class"].sum() / max(n_sd, 1)),
                     "same_class_rate_of_other_date_refs": float((g["same_class"] - g["top4_same_date_same_class"]).sum() / max(n_od, 1)),
                     "knn_acc_when_top4_has_same_date": float(has["knn_correct"].mean()), "knn_acc_when_top4_has_none": float(no["knn_correct"].mean()),
                     "share_of_prompts_with_same_date_ref": float((g["top4_same_date"] > 0).mean())}
    g = sd[sd["encoder"] == "vit"]
    res["SD_composition"] = {"bank_images_on_query_date_mean": float(g["bank_same_date"].mean()), "bank_share_on_query_date": float(g["bank_same_date"].mean() / (BANK_PER_CLASS * 9)),
                             "same_class_rate_of_bank_images_on_query_date": float(g["bank_same_date_same_class"].sum() / max(g["bank_same_date"].sum(), 1)), "by_encoder": comp}
    pc = R.groupby(["label", "bank_type", "encoder"])[["same_class", "knn_correct"]].mean().unstack([1, 2])
    pc.to_csv(os.path.join(out, "per_class.csv"))
    wjson(os.path.join(out, "audit_stats.json"), res)

    fz = json.load(open(os.path.join(out, "freeze.json")))
    c = res["cells"]
    L = ["# DeepWeeds split audit: date-disjoint versus same-date reference banks\n",
         f"No VLM calls. Query panel: {len(qm)} images ({PER_CLASS_Q} per class) on the {len(fz['query_dates'])} query dates of the base split, none shown to the VLM before. "
         f"Each bank holds {fz['bank_size']} images ({BANK_PER_CLASS} per class). {N_BANK_SEEDS} bank seeds per type. Every query retrieves its top {K} by cosine similarity.\n",
         "- **DD (date-disjoint):** bank images come from dates that hold no query.",
         "- **SD (same-date allowed):** bank images come from all dates.",
         f"- **Both:** exclude every panel query, images on a query's date and camera within {NEAR_SECONDS} s of the query, and images with a dHash distance of {NEAR_HAMMING} or less to a query "
         f"({fz['near_duplicates_removed']['total']} images removed: {fz['near_duplicates_removed']['by_time']} by time, {fz['near_duplicates_removed']['by_dhash']} by hash).\n",
         "## Result\n", f"| Bank | Encoder | Same-class in top {K} | 4-NN accuracy | 4-NN accuracy range over seeds |", "|---|---|---|---|---|"]
    for bt, nm in (("DD", "Date-disjoint"), ("SD", "Same-date allowed")):
        for enc, en in ENCODERS.items():
            x = c[f"{bt}|{enc}"]
            L.append(f"| {nm} | {en} | {x['same_class_mean']:.2f} | {100 * x['knn_acc']:.1f}% | {100 * x['knn_acc_min_seed']:.1f}% to {100 * x['knn_acc_max_seed']:.1f}% |")
    L += ["\n## Effect of allowing same-date references (SD minus DD, paired by query)\n",
          "| Encoder | Measure | Difference (95% CI) | Wilcoxon p | Queries higher / lower | Seeds with SD higher |", "|---|---|---|---|---|---|"]
    for enc, en in ENCODERS.items():
        for m, mn, sc in (("same_class", f"Same-class in top {K}", 1), ("knn_correct", "4-NN accuracy (pp)", 100)):
            x = res["SD_minus_DD"][f"{enc}|{m}"]
            L.append(f"| {en} | {mn} | {sc * x['diff']:+.2f} ({sc * x['ci'][0]:+.2f}, {sc * x['ci'][1]:+.2f}) | {fp(x['wilcoxon_p'])} | {x['queries_higher']} / {x['queries_lower']} | {x['seeds_with_SD_higher']}/{x['n_seeds']} |")
    L += ["\nThe confidence intervals resample the 180 queries, with each query averaged over the 20 bank seeds.\n",
          "## DINOv2 minus ViT under each bank type\n", "| Bank | Measure | Difference (95% CI) | Wilcoxon p |", "|---|---|---|---|"]
    for bt, nm in (("DD", "Date-disjoint"), ("SD", "Same-date allowed"), ("interaction", "Change in the gap (SD minus DD)")):
        for m, mn, sc in (("same_class", f"Same-class in top {K}", 1), ("knn_correct", "4-NN accuracy (pp)", 100)):
            x = res["encoder_gap"][f"{bt}|{m}"]
            L.append(f"| {nm} | {mn} | {sc * x['diff']:+.2f} ({sc * x['ci'][0]:+.2f}, {sc * x['ci'][1]:+.2f}) | {fp(x['wilcoxon_p'])} |")
    s = res["SD_composition"]
    L += ["\n## What the same-date bank contains and what gets retrieved\n",
          f"In an SD bank, {s['bank_images_on_query_date_mean']:.1f} of {fz['bank_size']} images ({100 * s['bank_share_on_query_date']:.1f}%) share the query's capture date on average, "
          f"and {100 * s['same_class_rate_of_bank_images_on_query_date']:.0f}% of those images have the query's class.\n",
          "| Encoder | Share of top 4 from the query's date | Prompts with a same-date reference | Same-class rate, same-date references | Same-class rate, other-date references | 4-NN accuracy with a same-date reference | 4-NN accuracy without |",
          "|---|---|---|---|---|---|---|"]
    for enc, en in ENCODERS.items():
        x = s["by_encoder"][enc]
        L.append(f"| {en} | {100 * x['share_of_top4_from_query_date']:.1f}% | {100 * x['share_of_prompts_with_same_date_ref']:.1f}% | {100 * x['same_class_rate_of_same_date_refs']:.0f}% | "
                 f"{100 * x['same_class_rate_of_other_date_refs']:.0f}% | {100 * x['knn_acc_when_top4_has_same_date']:.1f}% | {100 * x['knn_acc_when_top4_has_none']:.1f}% |")
    L += [f"\n## 4-NN accuracy per class\n", "| Class | Query dates | ViT DD | ViT SD | DINOv2 DD | DINOv2 SD |", "|---|---|---|---|---|---|"]
    for cl in sorted(qm["label"].unique()):
        L.append(f"| {cl} | {fz['panel_dates_per_class'][cl]} | " + " | ".join(f"{100 * pc.loc[cl, ('knn_correct', bt, enc)]:.0f}%" for enc in ENCODERS for bt in ("DD", "SD")) + " |")
    notes = os.path.join(out, "NOTES.md")
    if os.path.exists(notes):
        L.append("\n" + open(notes).read())
    open(os.path.join(out, "SUMMARY.md"), "w").write("\n".join(L) + "\n")
    print("\n".join(L))


if __name__ == "__main__":
    out = sys.argv[1]
    stage = sys.argv[sys.argv.index("--stage") + 1] if "--stage" in sys.argv else "analyze"
    os.makedirs(out, exist_ok=True)
    qm, B = freeze(out)
    print(f"frozen: {len(qm)} queries, {B['ref_id'].nunique()} unique bank images in {B.groupby(['bank_seed', 'bank_type']).ngroups} banks", flush=True)
    if stage != "freeze":
        E = embed(out, qm, B)
        if stage == "analyze":
            analyze(out, qm, B, E)
