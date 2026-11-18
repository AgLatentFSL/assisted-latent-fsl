"""Top-1, top-2, top-4 and top-8 retrieval analysis of the atlas. Reads only this folder. No model, VLM or API call.

Usage (from the atlas root):  python analysis/topk_analysis.py [--reparse]
Writes analysis/tables/, analysis/figures/, analysis/input_manifest.json, analysis/audit.json, analysis/checks.json and
rewrites the generated sections of README.md (between the TOPK markers). The parsed atlas is cached in
analysis/cache/parsed.npz; --reparse walks the folder tree again.
"""
import csv, hashlib, json, os, re, sys, time
import numpy as np, pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__))); AN = os.path.join(ROOT, "analysis"); TB = os.path.join(AN, "tables"); FG = os.path.join(AN, "figures"); CA = os.path.join(AN, "cache")
for d in (TB, FG, CA): os.makedirs(d, exist_ok=True)
KS = (1, 2, 4, 8); RBO_P = 0.9; POOLED = ("1_ViT", "final_output")           # the pooled ViT embedding of Arshad et al.
NAME = {"1_ViT": "ViT", "2_DINOv2": "DINOv2", "3_CLIP": "CLIP", "4_ResNet50": "ResNet-50", "5_SigLIP2": "SigLIP 2", "6_BioCLIP": "BioCLIP", "7_DINOv3": "DINOv3", "8_Qwen3VL": "Qwen3-VL tower", "9_FLAVA": "FLAVA"}
FINAL_EQUALS_LAST = {"2_DINOv2", "4_ResNet50", "7_DINOv3"}                      # README encoder table: final_output identical to the last layer
lname = lambda l: "final output" if l == "final_output" else l.replace("_", " ")


# ------------------------------------------------------------------ parse and audit the folder tree
def parse():
    t0 = time.time(); datasets = sorted(d for d in os.listdir(ROOT) if re.match(r"\d\d_", d) and os.path.isdir(os.path.join(ROOT, d)))
    Q, refkey, refino, refpath, lists, issues = [], {}, {}, {}, {}, {k: 0 for k in ("not_a_prefix_of_top8", "incomplete_ranks", "repeated_reference_in_a_list", "match_tag_disagrees_with_classes", "new_kept_tag_disagrees_with_recomputation",
                                                                         "unparseable_file_names", "self_match_by_file_identity", "reference_that_is_a_query_image", "reference_that_is_not_a_query_image", "canonical_id_with_two_files", "missing_layer_folders", "missing_topk_folders")}
    encs, layers = None, {}
    for ds in datasets:
        qdirs = sorted(os.listdir(os.path.join(ROOT, ds))); pool_inodes = set(); qrecs = []
        for q in qdirs:
            qp = os.path.join(ROOT, ds, q); qf = [f for f in os.listdir(qp) if f.startswith("QUERY__")][0]; ino = os.stat(os.path.join(qp, qf)).st_ino; pool_inodes.add(ino)
            qrecs.append((q, q.split("__")[1], ino, os.path.join(ds, q, qf)))
        for q, qcls, qino, qfile in qrecs:
            qp = os.path.join(ROOT, ds, q); qi = len(Q); Q.append({"dataset": ds, "query": q, "query_class": qcls, "query_file": qfile})
            e_here = sorted(d for d in os.listdir(qp) if os.path.isdir(os.path.join(qp, d)))
            if encs is None: encs = e_here
            for e in encs:
                if e not in e_here: issues["missing_layer_folders"] += 1; continue
                ls = sorted(os.listdir(os.path.join(qp, e))); ls = [l for l in ls if l != "final_output"] + ["final_output"]
                if e not in layers: layers[e] = ls
                prev = {}
                for l in layers[e]:
                    lp = os.path.join(qp, e, l); got = {}
                    if not os.path.isdir(lp): issues["missing_layer_folders"] += 1; continue
                    for k in KS:
                        kp = os.path.join(lp, f"top{k}")
                        if not os.path.isdir(kp): issues["missing_topk_folders"] += 1; continue
                        ent = []
                        for f in os.listdir(kp):
                            p = f.split("__", 4)
                            if len(p) != 5 or not p[0].isdigit() or p[1] not in ("match", "diff") or p[2] not in ("first", "kept", "new"): issues["unparseable_file_names"] += 1; continue
                            ent.append((int(p[0]), p[1], p[2], p[3], p[4], f))
                        ent.sort(); ids = []
                        if [x[0] for x in ent] != list(range(1, k + 1)): issues["incomplete_ranks"] += 1
                        for rank, mt, tag, cls, src, f in ent:
                            key = (ds, cls, src)                                   # canonical reference id: dataset, reference class, source file name (no rank, no tags)
                            if key not in refkey:
                                refkey[key] = len(refkey); refino[key] = os.stat(os.path.join(kp, f)).st_ino
                                refpath[key] = os.path.join(ds, q, e, l, f"top{k}", f); issues["reference_that_is_a_query_image" if refino[key] in pool_inodes else "reference_that_is_not_a_query_image"] += 1
                            elif k == 8 and rank == 1 and os.stat(os.path.join(kp, f)).st_ino != refino[key]: issues["canonical_id_with_two_files"] += 1
                            ids.append(refkey[key])
                            if (mt == "match") != (cls == qcls): issues["match_tag_disagrees_with_classes"] += 1
                            if refino[key] == qino: issues["self_match_by_file_identity"] += 1
                            want = "first" if l == layers[e][0] else ("kept" if refkey[key] in prev.get(k, ()) else "new")
                            if tag != want: issues["new_kept_tag_disagrees_with_recomputation"] += 1
                        if len(set(ids)) != len(ids): issues["repeated_reference_in_a_list"] += 1
                        got[k] = ids
                    if 8 in got:
                        for k in (1, 2, 4):
                            if got.get(k) != got[8][:k]: issues["not_a_prefix_of_top8"] += 1
                        lists.setdefault((e, l), {})[qi] = got[8]
                    if l != "final_output": prev = {k: set(v) for k, v in got.items()}          # final_output is tagged against the last layer
        print(f"parsed {ds}: {len(qrecs)} queries, {time.time() - t0:.0f}s", flush=True)
    nq = len(Q); keys = sorted(refkey, key=refkey.get); T = {}
    for (e, l), d in lists.items():
        a = np.full((nq, 8), -1, dtype=np.int32)
        for qi, ids in d.items(): a[qi, :len(ids)] = ids
        T[f"{e}|{l}"] = a
    np.savez_compressed(os.path.join(CA, "parsed.npz"), **T)
    json.dump({"queries": Q, "refs": [list(k) for k in keys], "ref_paths": [refpath[k] for k in keys], "encoders": encs, "layers": layers, "issues": issues, "ref_inodes_distinct": len(set(refino.values())), "parsed_at": time.strftime("%Y-%m-%d %H:%M:%S"), "seconds": round(time.time() - t0)}, open(os.path.join(CA, "parsed.json"), "w"))


if "--reparse" in sys.argv or not os.path.exists(os.path.join(CA, "parsed.npz")): parse()
META = json.load(open(os.path.join(CA, "parsed.json"))); Z = np.load(os.path.join(CA, "parsed.npz")); Q = pd.DataFrame(META["queries"]); REFS = META["refs"]; ENCS = META["encoders"]; LAYERS = META["layers"]; nq = len(Q)
ref_cls = np.array([r[1] for r in REFS]); TOP = {tuple(k.split("|")): Z[k] for k in Z.files}; qcls = Q.query_class.values
MATCH = {k: (ref_cls[np.where(v >= 0, v, 0)] == qcls[:, None]) & (v >= 0) for k, v in TOP.items()}
complete = np.all([(v >= 0).all(1) for v in TOP.values()], axis=0); COH = complete                     # common query cohort: complete top-8 lists for every encoder and layer
hidden = lambda e: [l for l in LAYERS[e] if l != "final_output"]

# ------------------------------------------------------------------ further audits
audit = {"datasets": int(Q.dataset.nunique()), "queries": int(nq), "queries_per_dataset": Q.dataset.value_counts().sort_index().to_dict(), "encoders": len(ENCS), "layers_without_final_output": int(sum(len(hidden(e)) for e in ENCS)),
         "layers_per_encoder": {NAME[e]: len(hidden(e)) for e in ENCS}, "encoder_layer_folders_expected": int(nq * sum(len(LAYERS[e]) for e in ENCS)), "encoder_layer_lists_found": int(sum((v >= 0).all(1).sum() for v in TOP.values())),
         "reference_files_expected": int(nq * sum(len(LAYERS[e]) for e in ENCS) * sum(KS)), "common_cohort_queries": int(COH.sum()), "queries_lost_to_the_common_cohort": int(nq - COH.sum()), "distinct_canonical_reference_ids": len(REFS),
         "distinct_reference_files": META["ref_inodes_distinct"], "issues": META["issues"]}
IDX = pd.read_csv(os.path.join(ROOT, "index.csv"), dtype=str); dmap = {d[3:].replace("_", " "): d for d in Q.dataset.unique()}; emap = {NAME[e].replace("-", "").replace(" ", "").replace("tower", ""): e for e in ENCS}
IDX["e"] = IDX.model.map(lambda m: emap[m]); IDX["d"] = IDX.dataset.map(dmap); qpos = {(r.dataset, r.query): i for i, r in Q.iterrows()}; rpos = {tuple(r): i for i, r in enumerate(REFS)}
san = lambda s: re.sub(r"[^A-Za-z0-9._-]+", "_", str(s)).strip("_")
IDX["ref"] = [rpos.get((d, san(l), san(os.path.basename(r))), -1) for d, l, r in zip(IDX.d, IDX.reference_label, IDX.reference_id)]; IDX["qi"] = [qpos[(d, q)] for d, q in zip(IDX.d, IDX.query_folder)]
mism = 0
for (e, l), g in IDX.groupby(["e", "layer"]):
    a = np.full((nq, 8), -2, dtype=np.int64); a[g.qi.values, g["rank"].astype(int).values - 1] = g.ref.values; mism += int((a != TOP[(e, l)]).sum())
audit["index_csv_rows"] = int(len(IDX)); audit["index_csv_entries_that_disagree_with_the_folders"] = mism
audit["classes_per_dataset_from_index_csv"] = IDX.groupby("d").reference_label.nunique().to_dict(); audit["classes_per_dataset_after_file_name_sanitizing"] = {d: len({r[1] for r in REFS if r[0] == d}) for d in Q.dataset.unique()}
h = {}
for qf in Q.query_file: h.setdefault(hashlib.md5(open(os.path.join(ROOT, qf), "rb").read()).hexdigest(), []).append(qf)
audit["byte_identical_images_among_the_query_images"] = int(sum(len(v) - 1 for v in h.values() if len(v) > 1))
SPLIT = json.load(open(os.path.join(ROOT, "split.json"))) if os.path.exists(os.path.join(ROOT, "split.json")) else {"protocol": "leave-one-out pool"}; DISJOINT = SPLIT["protocol"] == "disjoint"; audit["protocol"] = SPLIT["protocol"]
qfiles = set(Q.query_file); audit["retrieved_references_byte_identical_to_a_query_image"] = int(sum(1 for p in META["ref_paths"] if (m := hashlib.md5(open(os.path.join(ROOT, p), "rb").read()).hexdigest()) in h and not any(os.path.samefile(os.path.join(ROOT, p), os.path.join(ROOT, x)) for x in h[m])))
if os.path.exists(os.path.join(ROOT, "split_images.csv")):
    SI = pd.read_csv(os.path.join(ROOT, "split_images.csv"), dtype=str); g = SI.groupby(["dataset", "role"]).size().unstack(fill_value=0); audit["split_images"] = {d: {"queries": int(r.get("query", 0)), "references": int(r.get("reference", 0))} for d, r in g.iterrows()}
    audit["images_listed_as_both_query_and_reference"] = int(SI.groupby(["dataset", "image_id"]).role.nunique().gt(1).sum()); audit["md5_shared_by_a_query_and_a_reference"] = int(len(set(SI[SI.role == "query"].md5) & set(SI[SI.role == "reference"].md5)))
dwq = Q.dataset.str.contains("DeepWeeds").values
if dwq.any():                                   # capture date = first eight characters of a DeepWeeds file name
    qd = np.array([q.split("__")[2][:8] for q in Q["query"]]); rd = np.array([r[2][:8] for r in REFS]); pv = TOP[POOLED]
    audit["deepweeds_capture_dates"] = {"queries": int(dwq.sum()), "distinct_query_dates": int(len(set(qd[dwq]))), "date_separated": False,
                                        **{f"share_of_pooled_top{k}_references_from_the_query_date_pct": round(float(100 * (rd[pv[dwq][:, :k]] == qd[dwq][:, None]).mean()), 1) for k in KS}}
json.dump(audit, open(os.path.join(AN, "audit.json"), "w"), indent=1)


# ------------------------------------------------------------------ metrics
def rbo_ext(A, B, p=RBO_P):
    k = A.shape[1]; agree = np.stack([(A[:, :d, None] == B[:, None, :d]).any(2).sum(1) / d for d in range(1, k + 1)], 1); w = p ** np.arange(1, k + 1); return agree[:, -1] * p ** k + (1 - p) / p * (agree * w).sum(1)
def cmp(A, B):
    k = A.shape[1]; shared = (A[:, :, None] == B[:, None, :]).any(2).sum(1); ident = (A == B).all(1); return k - shared, ident, (shared == k) & ~ident
TR = np.array(["0 to 0", "0 to >=1", ">=1 to 0", ">=1 to >=1"])
def trans(mb, ma): return TR[np.where(mb == 0, np.where(ma == 0, 0, 1), np.where(ma == 0, 2, 3))]      # baseline count mb, layer count ma

rows = []
for e in ENCS:
    hid = hidden(e)
    for li, l in enumerate(LAYERS[e]):
        top, M = TOP[(e, l)], MATCH[(e, l)]; prev = (hid[li - 1] if (l != "final_output" and li > 0) else (hid[-1] if l == "final_output" else None))
        for k in KS:
            A = top[:, :k]; m = M[:, :k].sum(1); d = {"dataset": Q.dataset.values, "query": Q["query"].values, "query_class": qcls, "encoder": e, "layer": l, "K": k, "in_cohort": COH, "same_class": m}
            for nm, ref in (("previous_layer", (e, prev) if prev else None), ("final_output", (e, "final_output")), ("pooled_vit", POOLED)):
                if ref is None: d.update({f"replaced_vs_{nm}": -1, f"identical_order_vs_{nm}": False, f"same_ids_other_order_vs_{nm}": False}); continue
                B = TOP[ref][:, :k]; rep_, ident, reord = cmp(A, B); d.update({f"replaced_vs_{nm}": rep_, f"identical_order_vs_{nm}": ident, f"same_ids_other_order_vs_{nm}": reord})
                if nm != "previous_layer":
                    mb = MATCH[ref][:, :k].sum(1); d.update({f"same_class_{nm}": mb, f"transition_vs_{nm}": trans(mb, m), f"rbo_vs_{nm}": np.round(rbo_ext(A, B), 4)})
            rows.append(pd.DataFrame(d))
PQ = pd.concat(rows, ignore_index=True); PQ.to_csv(os.path.join(TB, "per_query_topk.csv.gz"), index=False, compression="gzip"); PQ = PQ[PQ.in_cohort]


def agg(d, k):
    n = len(d); r = {"n_queries": n, "mean_same_class": d.same_class.mean(), "fraction_of_slots_same_class": d.same_class.mean() / k, "pct_at_least_one_same_class": 100 * (d.same_class > 0).mean(), "pct_all_K_same_class": 100 * (d.same_class == k).mean()}
    r.update({f"pct_queries_with_{j}_same_class": (100 * (d.same_class == j).mean() if j <= k else np.nan) for j in range(9)})
    for nm in ("previous_layer", "final_output", "pooled_vit"):
        v = d[d[f"replaced_vs_{nm}"] >= 0]
        if not len(v): continue
        x = v[f"replaced_vs_{nm}"]; r.update({f"mean_replaced_vs_{nm}": x.mean(), f"fraction_replaced_vs_{nm}": x.mean() / k, f"pct_identical_order_vs_{nm}": 100 * v[f"identical_order_vs_{nm}"].mean(), f"pct_same_ids_other_order_vs_{nm}": 100 * v[f"same_ids_other_order_vs_{nm}"].mean(),
                                              f"pct_any_replaced_vs_{nm}": 100 * (x > 0).mean()})
        r.update({f"pct_{j}_replaced_vs_{nm}": (100 * (x == j).mean() if j <= k else np.nan) for j in range(9)})
        if nm != "previous_layer":
            t = v[f"transition_vs_{nm}"].value_counts(); c = {s: int(t.get(s, 0)) for s in TR}; miss = c["0 to 0"] + c["0 to >=1"]; dm = v.same_class - v[f"same_class_{nm}"]
            r.update({f"n_0_to_0_vs_{nm}": c["0 to 0"], f"n_0_to_ge1_vs_{nm}": c["0 to >=1"], f"n_ge1_to_0_vs_{nm}": c[">=1 to 0"], f"n_ge1_to_ge1_vs_{nm}": c[">=1 to >=1"],
                      f"baseline_coverage_pct_{nm}": 100 * (c[">=1 to 0"] + c[">=1 to >=1"]) / n, f"net_coverage_pp_vs_{nm}": 100 * (c["0 to >=1"] - c[">=1 to 0"]) / n, f"baseline_misses_{nm}": miss,
                      f"baseline_misses_recovered_pct_{nm}": (100 * c["0 to >=1"] / miss if miss else np.nan), f"mean_same_class_change_vs_{nm}": dm.mean(), f"pct_more_same_class_vs_{nm}": 100 * (dm > 0).mean(), f"pct_fewer_same_class_vs_{nm}": 100 * (dm < 0).mean(),
                      f"paired_change_hist_vs_{nm}": json.dumps({int(a): int(b) for a, b in dm.value_counts().sort_index().items()}), f"mean_rbo_vs_{nm}": v[f"rbo_vs_{nm}"].mean()})
    return r

def table(by):
    out = []
    for key, d in PQ.groupby(by, sort=False): key = key if isinstance(key, tuple) else (key,); out.append({**dict(zip(by, key)), **agg(d, dict(zip(by, key))["K"])})
    return pd.DataFrame(out)
AGG = table(["encoder", "layer", "K"]); AGG.insert(0, "encoder_name", AGG.encoder.map(NAME)); AGG.to_csv(os.path.join(TB, "aggregate_by_layer.csv"), index=False)
BYD = table(["dataset", "encoder", "layer", "K"]); BYC = table(["dataset", "query_class", "encoder", "layer", "K"])
for k in KS:
    BYD[BYD.K == k].to_csv(os.path.join(TB, f"by_dataset_top{k}.csv"), index=False); BYC[BYC.K == k].to_csv(os.path.join(TB, f"by_class_top{k}.csv"), index=False)

# consistency across depth: first hidden layer after which the ordered top-K list stays fixed through the final hidden layer
st = []
for e in ENCS:
    hid = hidden(e)
    for k in KS:
        Tk = np.stack([TOP[(e, l)][:, :k] for l in hid], 1)[COH]; same = (Tk == Tk[:, -1:, :]).all(2); first = np.array([next(i for i in range(len(hid)) if same[q, i:].all()) for q in range(len(Tk))])
        fo = cmp(TOP[(e, hid[-1])][COH][:, :k], TOP[(e, "final_output")][COH][:, :k])
        st.append({"encoder": e, "encoder_name": NAME[e], "K": k, "n_queries": int(len(Tk)), "hidden_layers": len(hid), "pct_settled_only_at_last_layer": 100 * (first == len(hid) - 1).mean(), "pct_settled_by_middle_layer": 100 * (first <= (len(hid) - 1) // 2).mean(),
                   "median_first_settled_layer": hid[int(np.median(first))], "pct_final_output_list_identical_to_last_layer": 100 * fo[1].mean(), "mean_replaced_final_output_vs_last_layer": fo[0].mean(),
                   **{f"pct_settled_at_{l}": 100 * (first == i).mean() for i, l in enumerate(hid)}})
STAB = pd.DataFrame(st); STAB.to_csv(os.path.join(TB, "consistency_across_depth.csv"), index=False)
V10 = AGG[(AGG.encoder == "1_ViT") & AGG.layer.isin(["layer_10", "final_output"])]; V10.to_csv(os.path.join(TB, "vit_layer10_vs_final_output.csv"), index=False)
if os.path.exists(os.path.join(AN, "topk_report.py")): exec(open(os.path.join(AN, "topk_report.py")).read())      # figures, README sections and checks
