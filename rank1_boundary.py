"""Retrieval-only test of a rank-one intervention on one frozen ViT block: signed boundary scatter against regularized Fisher.

Distance family, for a unit direction v and a gain alpha in [0, A], on L2-normalized embeddings (no renormalization afterwards):
    d(v, alpha; q, x) = ||q - x||^2 + alpha * (v^T (q - x))^2
Primary metric: correct-class Hit@4 (at least one reference of the query's class among the four nearest references).
No VLM call. CPU fitting. Everything is fitted per task (each task has its own reference bank and label set).

Stages (run from repo/):
    python rank1_boundary.py <out_dir> prep       fresh train and validation queries for the five paper datasets, ViT features for the new images
    python rank1_boundary.py <out_dir> selftest   exact gain sweep against direct reranking with exact fractions on a small fixture
    python rank1_boundary.py <out_dir> fit        block selection on train, pair mining, directions, margin identity, exact validation calibration, lock
    python rank1_boundary.py <out_dir> test       one evaluation of the locked filters on the 500 atlas queries (the test set of the paper), paired bootstrap, certificate
    python rank1_boundary.py <out_dir> report     REPORT.md from the saved results
The fit stage never loads a test embedding. The test stage refuses to run without lock.json and refuses to run twice.
"""
import hashlib, json, os, sys, time
from fractions import Fraction
import numpy as np, pandas as pd

out, stage = sys.argv[1], sys.argv[2]; ENC = sys.argv[3] if len(sys.argv) > 3 else "vit"
os.makedirs(os.path.join(out, ENC), exist_ok=True); P = lambda *f: os.path.join(out, *f); E = lambda *f: os.path.join(out, ENC, *f)      # split, features and selftest are shared; fit and test outputs are per encoder
# readout and blocks per encoder (repository cache format): CLS token where the model has one, patch mean otherwise, global average pool for ResNet-50
SPEC = {"vit": ("cls", range(1, 13), "google/vit-base-patch16-224-in21k"), "clip": ("cls", range(1, 13), "openai/clip-vit-base-patch16"), "resnet": ("gap", range(17), "microsoft/resnet-50"),
        "siglip2": ("mean", range(1, 13), "google/siglip2-base-patch16-224"), "qwen": ("mean", range(1, 25), "Qwen/Qwen3-VL-2B-Instruct vision tower"), "flava": ("cls", range(1, 13), "facebook/flava-full image encoder"),
        "dinov2": ("cls", range(1, 13), "facebook/dinov2-base"), "bioclip": ("cls", range(1, 13), "imageomics/bioclip"), "dinov3": ("cls", range(1, 13), "facebook/dinov3-vitb16-pretrain-lvd1689m")}
READOUT, BLOCKS, CKPT_NAME = SPEC[ENC]
RUNS = os.path.join("runs"); SPLIT = os.path.join(RUNS, "disjoint_split_seed42"); AUDIT = os.path.join(RUNS, "layer_retrieval_audit")
TASKS = ["Bean Leaf Lesions", "Mango Leaf Disease", "DeepWeeds", "Soybean Seeds", "Soybean Diseases"]; tag = lambda t: t.replace(" ", "_")
# ---- everything below is fixed before any embedding of a test query is read
CFG = {"seed": 20261116, "encoder": f"{ENC} ({CKPT_NAME})", "readout": f"{READOUT} readout per block (repository cache format); pooled baseline = the native output of the encoder", "K": 4,
       "n_train": 300, "n_val": 200,
       "test": "the 500 queries of the atlas and of the paper (runs/disjoint_split_seed42/queries_*.csv, 100 per dataset), unchanged",
       "test_limit": "layer-wise retrieval results on these 500 queries were read earlier in this project and were used to choose the layers of the paper, so the test set is not fresh with respect to the choice of a block; here the block is chosen on the train queries only, and no direction, gain or rho sees a test query", "A": 100.0, "fisher_rho_rel": [1e-3, 1e-2, 1e-1, 1.0, 10.0], "random_draws": 100, "bootstrap": 10000,
       "margin_alphas": [0.5, 1.0, 10.0, 100.0], "root_tie_tol": 1e-12}
K, A = CFG["K"], CFG["A"]
RULES = {
    "block": f"every block of the encoder ({READOUT} readout, blocks {BLOCKS[0]} to {BLOCKS[-1]}); the block with the highest Hit@4 of raw cosine retrieval on the TRAIN queries, pooled over the five tasks; earliest block on ties",
    "pairs": "per task, for each eligible train query (bank has at least one correct-class and at least four wrong-class references): p = nearest correct-class reference, n = fourth-nearest wrong-class reference under the raw hidden-block ranking",
    "gain": "per task and direction, exact sweep of validation Hit@4 over alpha in [0, A]: both endpoints, every breakpoint at which the Hit@4 status of a validation query changes (reference-id tie rule applied at the breakpoint), and the midpoint of every open interval between consecutive such points; the highest validation Hit@4 wins; smallest evaluated gain on ties",
    "fisher_rho": "rho = c * trace(S_pos) / d with c in fisher_rho_rel; per task the (rho, gain) with the highest validation Hit@4; then the smallest gain; then the largest rho",
    "random": "random_draws draws, each one independent Gaussian unit direction per task, generator seeded with seed; each task direction gets its own validation-selected gain; scores are reported as a distribution over draws",
    "svd": "SVD of the centered train-query and bank embeddings of the task; head = first right singular vector; tail = right singular vector of the smallest singular value above sigma_max * max(n, d) * eps",
    "tie_break": "references are ordered by reference id (path); equal distances are broken by that order in every method",
    "aggregate": "queries pooled over the five tasks (repository convention); per-task scores are listed next to it",
    "bootstrap": "paired, stratified by task; queries are resampled within a task; in DeepWeeds capture dates are resampled as clusters"}

# ------------------------------------------------------------------ features
def feature_index(dirs):
    idx = {}
    for d in dirs:
        for f in sorted(os.listdir(d)):
            if f.startswith(ENC + "__") and f.endswith(".npz"):
                for i, p in enumerate(np.load(os.path.join(d, f))["images"].tolist()): idx.setdefault(p, (os.path.join(d, f), i))
    return idx
_Z = {}
def emb(paths, block, idx):
    """Unit-length float64 embeddings: block = a block index of the encoder, or "pooled" = its native output."""
    rows = []; key = "native" if block == "pooled" else f"gap{block:02d}" if ENC == "resnet" else READOUT
    for p in paths:
        f, i = idx[p]
        if (f, key) not in _Z: _Z[(f, key)] = np.load(f)[key]
        rows.append(_Z[(f, key)][i] if block == "pooled" or ENC == "resnet" else _Z[(f, key)][i, block - 1])
    X = np.asarray(rows, dtype=np.float64); return X / np.linalg.norm(X, axis=1, keepdims=True)

# ------------------------------------------------------------------ distance family and ranking
def base_and_proj(Q, X, v=None):
    """d0[q, x] = ||q - x||^2 from explicit differences; s[q, x] = (v^T (q - x))^2."""
    d0 = np.empty((len(Q), len(X)))
    for a in range(0, len(Q), 64): d0[a:a + 64] = ((Q[a:a + 64, None, :] - X[None, :, :]) ** 2).sum(-1)
    return d0, (None if v is None else ((Q @ v)[:, None] - (X @ v)[None, :]) ** 2)
def top(d): return np.argsort(d, axis=1, kind="stable")[:, :K]          # bank is sorted by reference id, so a stable sort breaks ties by id
def hits(d, yq, yb):
    t = top(d); same = yb[t] == yq[:, None]; return same.any(1), same.sum(1), t

def profile(d0, s, pos, neg):
    """Exact Hit@4 status of one query for every alpha in [0, A]. d0, s: per reference. pos, neg: bank indices (ascending = id order).
    Returns (roots R, hit at each root, hit on each open interval between consecutive points of [0, R..., A], hit at 0, hit at A, number of roots)."""
    if len(pos) == 0: return np.zeros(0), np.zeros(0, bool), np.zeros(1, bool), False, False, 0
    if len(neg) < K: return np.zeros(0), np.zeros(0, bool), np.ones(1, bool), True, True, 0
    a = d0[neg][None, :] - d0[pos][:, None]; b = s[neg][None, :] - s[pos][:, None]; tie = neg[None, :] < pos[:, None]      # n is ahead of p when a + alpha b < 0, or = 0 and n has the smaller id
    with np.errstate(divide="ignore", invalid="ignore"): r = np.where(b != 0, -a / b, np.inf)
    R = np.unique(r[(r > 0) & (r < A)])
    def status(al, at_root):
        res = np.empty(len(al), bool)
        for c in range(0, len(al), 256):
            x = al[c:c + 256]; D = a[None] + x[:, None, None] * b[None]; ahead = (D < 0) | ((D == 0) & tie[None])
            if at_root:
                near = np.abs(r[None] - x[:, None, None]) <= CFG["root_tie_tol"] * np.maximum(1.0, x)[:, None, None]; ahead = np.where(near, tie[None], ahead)
            res[c:c + 256] = ahead.sum(-1).min(1) < K
        return res
    edges = np.concatenate([[0.0], R, [A]]); mids = (edges[:-1] + edges[1:]) / 2
    end = status(np.array([0.0, A]), False)
    return R, status(R, True), status(mids, False), bool(end[0]), bool(end[1]), len(R)
def prof_eval(pr, al):
    """Value of a profile at the gains al (0, A, exact roots or any other point)."""
    R, hr, hi, h0, hA, _ = pr; j = np.searchsorted(R, al); v = hi[np.minimum(j, len(hi) - 1)].copy()
    if len(R): isr = (j < len(R)) & (R[np.minimum(j, len(R) - 1)] == al); v[isr] = hr[j[isr]]
    v[al == 0.0] = h0; v[al == A] = hA; return v
def change_points(pr):
    R, hr, hi, h0, hA, _ = pr
    return R[(hr != hi[:-1]) | (hr != hi[1:])] if len(R) else R
def sweep(d0, s, yq, yb):
    """Exact Hit@4 count over alpha in [0, A] for a query set. Returns candidates, counts, profiles, total roots."""
    prof = [profile(d0[i], s[i], np.flatnonzero(yb == yq[i]), np.flatnonzero(yb != yq[i])) for i in range(len(yq))]
    cp = np.unique(np.concatenate([[0.0, A]] + [change_points(p) for p in prof])); cand = np.unique(np.concatenate([cp, (cp[:-1] + cp[1:]) / 2]))
    H = np.zeros(len(cand), int)
    for p in prof: H += prof_eval(p, cand)
    return cand, H, prof, sum(p[5] for p in prof)
def pick(cand, H): j = int(np.flatnonzero(H == H.max())[0]); return float(cand[j]), int(H[j])     # cand is ascending: smallest evaluated gain among the best

# ------------------------------------------------------------------ data access
def load_split(role):
    D = {}
    for t in TASKS:
        f = P("split", f"{role}_{tag(t)}.csv") if role in ("train", "val") else os.path.join(SPLIT, f"{'bank' if role == 'bank' else 'queries'}_{tag(t)}.csv")
        d = pd.read_csv(f, dtype=str, keep_default_na=False); d["group"] = [group_of(t, p) for p in d.path]; D[t] = d.sort_values("path").reset_index(drop=True)
    return D
group_of = lambda t, path: os.path.basename(path)[:8] if t == "DeepWeeds" else ""        # DeepWeeds file names start with the capture date

# ================================================================== prep
if stage == "prep":
    try: import kaggle                                   # the datasets are on disk; the loaders need the module only to download
    except Exception:
        import types; sys.modules["kaggle"] = types.ModuleType("kaggle")
    import data_loader as dl
    assert not os.path.exists(P("split", "manifest.json")), "split already drawn; use the features stage for further encoders"
    os.makedirs(P("split"), exist_ok=True); md5 = lambda p: hashlib.md5(open(p, "rb").read()).hexdigest()
    LOAD = {"Bean Leaf Lesions": dl.load_and_prepare_data_bean_leaf, "Mango Leaf Disease": dl.load_and_prepare_data_mango_leaf, "DeepWeeds": dl.load_and_prepare_data_DeepWeeds,
            "Soybean Seeds": dl.load_and_prepare_data_soybean_seeds, "Soybean Diseases": dl.load_and_prepare_data_Soybean_PNAS}
    NEED = {"train": CFG["n_train"], "val": CFG["n_val"]}; man = {}; new = []
    for ti, t in enumerate(TASKS):
        rng = np.random.default_rng([CFG["seed"], ti])
        full, _, _ = LOAD[t](10 ** 7); full = pd.DataFrame({"path": full[0], "label": full[1].astype(str)}).drop_duplicates("path").sort_values("path").reset_index(drop=True)
        bank = pd.read_csv(os.path.join(SPLIT, f"bank_{tag(t)}.csv"), dtype=str); atlas = pd.read_csv(os.path.join(SPLIT, f"queries_{tag(t)}.csv"), dtype=str)
        used_p = set(bank.path) | set(atlas.path); seen = set(bank.md5) | set(atlas.md5); labels = sorted(bank.label.unique()); assert set(full.label) == set(labels)
        rest = full[~full.path.isin(used_p)]; got = {r: [] for r in NEED}; dup = 0; tot = sum(NEED.values())
        for l in labels:                                  # class-balanced; a class with too few images gives what it has, split in the same proportions
            cand = rest[rest.label == l].path.values; cand = cand[rng.permutation(len(cand))]; uniq = []
            want = {r: NEED[r] // len(labels) + (1 if labels.index(l) < NEED[r] % len(labels) else 0) for r in NEED}
            for p in cand:
                if len(uniq) == sum(want.values()): break
                m = md5(p)
                if m in seen: dup += 1; continue
                seen.add(m); uniq.append((p, l, m))
            if len(uniq) < sum(want.values()): f_ = len(uniq) / sum(want.values()); want = {r: int(want[r] * f_) for r in want}
            a = 0
            for r in NEED: got[r] += uniq[a:a + want[r]]; a += want[r]
        man[t] = {"dataset_images": int(len(full)), "classes": len(labels), "bank": int(len(bank)), "test_queries_atlas": int(len(atlas)), "byte_identical_candidates_skipped": dup}
        for r in NEED:
            d = pd.DataFrame(got[r], columns=["path", "label", "md5"]); d["group"] = [group_of(t, p) for p in d.path]; d.to_csv(P("split", f"{r}_{tag(t)}.csv"), index=False)
            man[t][r] = int(len(d)); man[t][f"{r}_per_class"] = d.label.value_counts().sort_index().to_dict(); new += d.path.tolist()
        allp = [p for r in NEED for p, _, _ in got[r]]; allm = [m for r in NEED for _, _, m in got[r]]
        assert len(set(allp)) == len(allp) and len(set(allm)) == len(allm) and not set(allp) & used_p and not set(allm) & (set(bank.md5) | set(atlas.md5))
        print(t, man[t], flush=True)
    json.dump({"config": CFG, "rule": "bank = the frozen 100-image reference bank of runs/disjoint_split_seed42; test = the 100 atlas queries per dataset of the same folder, unchanged; train and validation queries are fresh class-balanced draws from the rest of each dataset, excluding the bank, the test queries and every byte-identical image", "tasks": man}, open(P("split", "manifest.json"), "w"), indent=1)
    sys.argv = ["x", out, "none"]
    import layer_audit as LA
    LA.CKPT = {ENC: LA.CKPT[ENC]}; new = sorted(new)
    LA.extract_all([{"split": "rank-one experiment | fresh train and validation queries", "tag": "r1" + hashlib.md5("".join(new).encode()).hexdigest()[:8], "images": new}])
    sys.exit()

if stage == "features":                                  # same train and validation images, another encoder (python rank1_boundary.py <out_dir> features <encoder>)
    new = sorted(p for r in ("train", "val") for t in TASKS for p in pd.read_csv(P("split", f"{r}_{tag(t)}.csv")).path); sys.argv = ["x", out, "none"]
    import layer_audit as LA
    LA.CKPT = {ENC: LA.CKPT[ENC]}
    LA.extract_all([{"split": "rank-one experiment | fresh train and validation queries", "tag": "r1" + hashlib.md5("".join(new).encode()).hexdigest()[:8], "images": new}]); sys.exit()

if stage == "dedupe":                                    # re-saved copies of one photo (same class folder, file name differing only by repeated " (Custom)") have different bytes; remove them from train and validation
    import re
    canon = lambda p: (os.path.basename(os.path.dirname(p)), re.sub(r"(\s*\(Custom\))+", "", os.path.basename(p)).strip().lower()); removed = []
    for t in TASKS:
        keep = {canon(p) for r in ("bank", "queries") for p in pd.read_csv(os.path.join(SPLIT, f"{r}_{tag(t)}.csv")).path}       # the bank and the test queries are fixed
        for r in ("val", "train"):                        # validation keeps its images against train; train gives way
            d = pd.read_csv(P("split", f"{r}_{tag(t)}.csv"), dtype=str, keep_default_na=False); c = [canon(p) for p in d.path]; ok = []
            for x in c: ok.append(x not in keep); keep.add(x)
            removed += [{"task": t, "role": r, "path": p} for p, o in zip(d.path, ok) if not o]; d[ok].to_csv(P("split", f"{r}_{tag(t)}.csv"), index=False)
    pd.DataFrame(removed, columns=["task", "role", "path"]).to_csv(P("split", "removed_resaved_copies.csv"), index=False); print(pd.DataFrame(removed).to_string() if removed else "nothing removed"); sys.exit()

# ================================================================== selftest
if stage == "selftest":
    rng = np.random.default_rng(7); bad = 0; n_pts = 0; t0 = time.time()
    for trial in range(60):
        d, nb = 3, 14; X = rng.integers(-3, 4, (nb, d)).astype(float); X[5] = X[2]; X[9] = X[1]         # duplicated references create exact ties
        q = rng.integers(-3, 4, d).astype(float); v = rng.integers(-2, 3, d).astype(float); v[0] += (not v.any())
        yb = rng.integers(0, 3, nb); yb[:3] = [0, 1, 2]; yq = 0; pos, neg = np.flatnonzero(yb == yq), np.flatnonzero(yb != yq)
        d0, s = base_and_proj(q[None], X, v); pr = profile(d0[0], s[0], pos, neg); R = pr[0]; edges = np.concatenate([[0.0], R, [A]])
        F = lambda z: Fraction(int(z)); d0f = [sum((F(q[k]) - F(x[k])) ** 2 for k in range(d)) for x in X]; sf = [sum(F(v[k]) * (F(q[k]) - F(x[k])) for k in range(d)) ** 2 for x in X]
        roots_f = sorted({-(d0f[n] - d0f[p]) / (sf[n] - sf[p]) for p in pos for n in neg if sf[n] != sf[p] and 0 < -(d0f[n] - d0f[p]) / (sf[n] - sf[p]) < Fraction(A)})
        assert len(roots_f) == len(R) and np.allclose([float(z) for z in roots_f], R, rtol=1e-12, atol=0), "root sets differ"
        ef = [Fraction(0)] + roots_f + [Fraction(A)]
        def direct(al):                                   # full reranking with exact arithmetic and the id tie rule
            order = sorted(range(nb), key=lambda j: (d0f[j] + al * sf[j], j)); return any(yb[j] == yq for j in order[:K])
        pts = [(Fraction(0), pr[3]), (Fraction(A), pr[4])] + list(zip(roots_f, pr[1])) + [((ef[i] + ef[i + 1]) / 2, pr[2][i]) for i in range(len(ef) - 1)]
        for al, got in pts: n_pts += 1; bad += direct(al) != bool(got)
        al = np.concatenate([[0.0, A], R, (edges[:-1] + edges[1:]) / 2]); assert (prof_eval(pr, al) == np.concatenate([[pr[3], pr[4]], pr[1], pr[2]])).all()
    res = {"fixtures": 60, "points_checked": n_pts, "disagreements_with_exact_direct_reranking": int(bad), "seconds": round(time.time() - t0, 2)}
    json.dump(res, open(P("selftest.json"), "w"), indent=1); print(res); assert bad == 0; sys.exit()

IDX = feature_index([os.path.join(AUDIT, "features"), P("features")])
BANK = load_split("bank")
def task_data(role, t, block):
    d = load_split(role)[t]; b = BANK[t]
    return d, emb(d.path.tolist(), block, IDX), d.label.values, emb(b.path.tolist(), block, IDX), b.label.values
def transform_check(Q, X, v, al):
    """max |d - Euclidean distance after x -> x + (sqrt(1 + alpha) - 1) v v^T x| (no renormalization)."""
    T = lambda Z: Z + (np.sqrt(1 + al) - 1) * np.outer(Z @ v, v); d0, s = base_and_proj(Q, X, v); return float(np.abs(d0 + al * s - base_and_proj(T(Q), T(X))[0]).max())

# ================================================================== fit
if stage == "fit":
    assert not os.path.exists(E("lock.json")), "already locked"
    assert json.load(open(P("selftest.json")))["disagreements_with_exact_direct_reranking"] == 0, "run selftest first"
    log = {"config": CFG, "rules": RULES}; rngR = np.random.default_rng(CFG["seed"])
    # ---- block selection on train queries
    bs = []
    for blk in BLOCKS:
        h = n = 0
        for t in TASKS:
            _, Q, yq, X, yb = task_data("train", t, blk); hh, _, _ = hits(base_and_proj(Q, X)[0], yq, yb); h += hh.sum(); n += len(yq)
        bs.append({"block": blk, "train_hits": int(h), "n": int(n), "train_hit4_pct": 100 * h / n})
    BS = pd.DataFrame(bs); BS.to_csv(E("block_selection_train.csv"), index=False); BLOCK = int(BS.sort_values(["train_hits", "block"], ascending=[False, True]).block.iloc[0]); print(BS.to_string(), "\nselected block", BLOCK, flush=True)
    setup, margin_rows, val_rows, pairs_all, dirs, curves, cross = {}, [], [], [], {}, {}, []
    for t in TASKS:
        tr, Q, yq, X, yb = task_data("train", t, BLOCK); va, Qv, yv, _, _ = task_data("val", t, BLOCK); b = BANK[t]; d = X.shape[1]
        _, Qp, _, Xp, _ = task_data("train", t, "pooled"); _, Qvp, _, _, _ = task_data("val", t, "pooled")
        # ---- checks: alpha = 0 against cosine retrieval, coverage, duplicates, groups
        d0, _ = base_and_proj(Q, X); cos_top = np.argsort(-(Q @ X.T), axis=1, kind="stable")[:, :K]; raw_top = top(d0)
        d0v, _ = base_and_proj(Qv, X); cos_top_v = np.argsort(-(Qv @ X.T), axis=1, kind="stable")[:, :K]
        st = {"bank": len(b), "classes": int(len(set(yb))), "bank_per_class_min": int(pd.Series(yb).value_counts().min()), "bank_per_class_max": int(pd.Series(yb).value_counts().max())}
        for nm, dd, yy, Dq, ct in (("train", tr, yq, d0, cos_top), ("val", va, yv, d0v, cos_top_v)):
            npos = (yb[None, :] == yy[:, None]).sum(1); nneg = len(yb) - npos; srt = np.sort(Dq, axis=1)
            st[nm] = {"queries": len(dd), "per_class": pd.Series(yy).value_counts().sort_index().to_dict(), "no_correct_class_reference": int((npos == 0).sum()), "fewer_than_4_wrong_class_references": int((nneg < K).sum()),
                      "paths_shared_with_bank": len(set(dd.path) & set(b.path)), "md5_shared_with_bank": len(set(dd.md5) & set(b.md5)),
                      "top4_sets_equal_to_cosine_top4": int((np.sort(top(Dq), 1) == np.sort(ct, 1)).all(1).sum()), "ordered_top4_equal_to_cosine_top4": int((top(Dq) == ct).all(1).sum()),
                      "queries_with_exact_distance_tie_at_rank_4_5": int((srt[:, K - 1] == srt[:, K]).sum()), "min_gap_rank_4_5": float((srt[:, K] - srt[:, K - 1]).min()),
                      "nearest_reference_cosine_ge_0.999_at_block": int(((Q if nm == "train" else Qv) @ X.T).max(1).__ge__(0.999).sum()), "nearest_reference_cosine_ge_0.999_pooled": int(((Qp if nm == "train" else Qvp) @ Xp.T).max(1).__ge__(0.999).sum())}
            if t == "DeepWeeds":
                bd = np.array([group_of(t, p) for p in b.path]); qd = dd.group.values
                st[nm].update({"capture_dates": int(len(set(qd))), "bank_capture_dates": int(len(set(bd))), "queries_on_a_bank_capture_date": int(np.isin(qd, bd).sum()),
                               "queries_with_a_same_date_correct_class_reference": int(sum(((bd == g) & (yb == y)).any() for g, y in zip(qd, yy)))})
        st["paths_shared_train_val"] = len(set(tr.path) & set(va.path)); st["md5_shared_train_val"] = len(set(tr.md5) & set(va.md5))
        if t == "DeepWeeds": st["capture_dates_shared_train_val"] = len(set(tr.group) & set(va.group))
        # ---- pair mining under the raw hidden-block ranking
        order = np.argsort(d0, axis=1, kind="stable"); pi, ni, el = [], [], []
        for i in range(len(yq)):
            o = order[i]; ps = o[yb[o] == yq[i]]; ns = o[yb[o] != yq[i]]
            if len(ps) >= 1 and len(ns) >= K: el.append(i); pi.append(ps[0]); ni.append(ns[K - 1])
        el, pi, ni = np.array(el), np.array(pi), np.array(ni); st["train"]["eligible_for_pairs"] = int(len(el))
        pairs_all.append(pd.DataFrame({"task": t, "query": tr.path.values[el], "positive": b.path.values[pi], "negative_4th_wrong": b.path.values[ni]}))
        t0 = time.time(); DP, DN = Q[el] - X[pi], Q[el] - X[ni]; Spos, Sneg = DP.T @ DP / len(el), DN.T @ DN / len(el); Bm = Sneg - Spos; t_scatter = time.time() - t0
        # ---- directions
        fit_s = {}
        t0 = time.time(); lam, U = np.linalg.eigh(Bm); u = U[:, -1] / np.linalg.norm(U[:, -1]); fit_s["boundary"] = t_scatter + time.time() - t0
        from scipy.linalg import eigh as geigh
        F = {}
        for c in CFG["fisher_rho_rel"]:
            t0 = time.time(); rho = c * np.trace(Spos) / d; w, V = geigh(Sneg, Spos + rho * np.eye(d), subset_by_index=[d - 1, d - 1]); f = V[:, 0] / np.linalg.norm(V[:, 0]); F[c] = (f, rho, float(w[0])); fit_s[f"fisher_c{c:g}"] = t_scatter + time.time() - t0
        t0 = time.time(); Z = np.vstack([Q, X]); Z = Z - Z.mean(0); sv = np.linalg.svd(Z, full_matrices=False); thr = sv[1][0] * max(Z.shape) * np.finfo(float).eps; rank = int((sv[1] > thr).sum()); fit_s["svd"] = time.time() - t0
        D = {"boundary": u, "svd_head": sv[2][0], "svd_tail": sv[2][rank - 1]}
        for c in F: D[f"fisher_c{c:g}"] = F[c][0]
        rnd = rngR.standard_normal((CFG["random_draws"], d)); rnd /= np.linalg.norm(rnd, axis=1, keepdims=True)
        for j in range(CFG["random_draws"]): D[f"random_{j:03d}"] = rnd[j]
        st["svd"] = {"n_vectors": int(Z.shape[0]), "numerical_rank": rank, "threshold": float(thr), "sigma_max": float(sv[1][0]), "sigma_tail": float(sv[1][rank - 1])}
        st["boundary"] = {"lambda_1": float(lam[-1]), "lambda_2": float(lam[-2]), "eigengap": float(lam[-1] - lam[-2]), "lambda_min": float(lam[0]), "positive_eigenvalues": int((lam > 1e-12).sum())}
        st["transform_equivalence_max_abs_error"] = max(transform_check(Qv[:40], X, u, al) for al in (1.0, A))
        # ---- validation calibration (exact); crossing count and runtime are measured on the proposed direction first
        d0v, _ = base_and_proj(Qv, X); best = {}
        for name in D:
            t0 = time.time(); _, s = base_and_proj(Qv, X, D[name]); cand, H, _, nroots = sweep(d0v, s, yv, yb); al, h = pick(cand, H); el_s = time.time() - t0
            best[name] = (al, h); cross.append({"task": t, "direction": name, "pair_roots_in_(0,A)": int(nroots), "status_change_points": int(len(cand) // 2 + 1 - 2), "seconds": round(el_s, 3)})
            if not name.startswith("random"): curves[f"{tag(t)}|{name}|alpha"] = cand; curves[f"{tag(t)}|{name}|hits"] = H
            if name == "boundary": print(f"{t}: exact sweep of the proposed direction: {nroots} pair roots in (0, A), {el_s:.2f} s; estimate for {len(D)} directions {el_s * len(D) / 60:.1f} min", flush=True)
            val_rows.append({"task": t, "direction": name, "n_val": len(yv), "val_hits_alpha0": int(H[0]), "val_hits_best": h, "alpha_selected": al, "val_hits_at_A": int(H[-1]), "fit_seconds": fit_s.get(name, fit_s["svd"] if name.startswith("svd") else 0.0)})
        cF = sorted(CFG["fisher_rho_rel"], key=lambda c: (-best[f"fisher_c{c:g}"][1], best[f"fisher_c{c:g}"][0], -c))[0]; f, rho, wF = F[cF]
        st["fisher"] = {"rho_rel_selected": cF, "rho": float(rho), "generalized_eigenvalue": wF, "alpha": best[f"fisher_c{cF:g}"][0], "squared_alignment_with_u": float((u @ f) ** 2)}
        # ---- exact margin identity on the frozen train pairs (float64)
        proj2 = (U.T @ f) ** 2
        for al in CFG["margin_alphas"]:
            mg = lambda v: ((DN ** 2).sum(1) + al * (DN @ v) ** 2) - ((DP ** 2).sum(1) + al * (DP @ v) ** 2)
            obs = float((mg(u) - mg(f)).mean()); spec = float(al * ((lam[-1] - lam) * proj2).sum()); tol = 1e-10 * max(1.0, al)
            margin_rows.append({"task": t, "alpha": al, "pairs": int(len(el)), "observed_delta": obs, "spectral_delta": spec, "abs_discrepancy": abs(obs - spec), "tolerance": tol, "pass": bool(abs(obs - spec) <= tol), "lambda_1": float(lam[-1]), "eigengap": float(lam[-1] - lam[-2]),
                                "fisher_sq_alignment_leading": float(proj2[-1]), "mean_margin_raw": float(((DN ** 2).sum(1) - (DP ** 2).sum(1)).mean()), "mean_margin_boundary": float(mg(u).mean()), "mean_margin_fisher": float(mg(f).mean())})
        setup[t] = st
        dirs[f"{tag(t)}|boundary"] = u; dirs[f"{tag(t)}|fisher"] = f; dirs[f"{tag(t)}|svd_head"] = D["svd_head"]; dirs[f"{tag(t)}|svd_tail"] = D["svd_tail"]; dirs[f"{tag(t)}|random"] = rnd
        dirs[f"{tag(t)}|alpha"] = np.array([best["boundary"][0], best[f"fisher_c{cF:g}"][0], best["svd_head"][0], best["svd_tail"][0]] + [best[f"random_{j:03d}"][0] for j in range(CFG["random_draws"])])
        print(t, "locked: boundary alpha", best["boundary"], "| fisher c", cF, best[f"fisher_c{cF:g}"], flush=True)
    V = pd.DataFrame(val_rows); V.to_csv(E("validation_all_directions.csv"), index=False); pd.concat(pairs_all).to_csv(E("train_pairs.csv"), index=False); MR = pd.DataFrame(margin_rows); MR.to_csv(E("margin_identity.csv"), index=False)
    pd.DataFrame(cross).to_csv(E("calibration_cost.csv"), index=False); np.savez_compressed(E("validation_curves.npz"), **curves); np.savez_compressed(E("locked_directions.npz"), **dirs)
    assert MR["pass"].all(), "margin identity failed: diagnose before reading retrieval results"
    rv = V[V.direction.str.startswith("random")].assign(draw=lambda x: x.direction.str[-3:].astype(int)).groupby("draw").val_hits_best.sum(); best_draw = int(rv.sort_values(ascending=False, kind="stable").index[0])
    lock = {"locked_at": time.strftime("%Y-%m-%d %H:%M:%S"), "config": CFG, "rules": RULES, "block": BLOCK, "dim": int(d), "reference_manifest": {t: {"file": os.path.join(SPLIT, f"bank_{tag(t)}.csv"), "sha256": hashlib.sha256(open(os.path.join(SPLIT, f"bank_{tag(t)}.csv"), "rb").read()).hexdigest(), "n": len(BANK[t])} for t in TASKS},
            "directions_file": "locked_directions.npz", "directions_sha256": hashlib.sha256(open(E("locked_directions.npz"), "rb").read()).hexdigest(),
            "gains": {t: {"boundary": float(dirs[f"{tag(t)}|alpha"][0]), "fisher": float(dirs[f"{tag(t)}|alpha"][1]), "svd_head": float(dirs[f"{tag(t)}|alpha"][2]), "svd_tail": float(dirs[f"{tag(t)}|alpha"][3])} for t in TASKS},
            "fisher": {t: setup[t]["fisher"] for t in TASKS}, "random_best_validation_draw": best_draw, "calibration": "exact (event sweep over all breakpoints in [0, A], validated by selftest.json)", "setup": setup}
    json.dump(lock, open(E("lock.json"), "w"), indent=1); print("LOCKED block", BLOCK); sys.exit()

# ================================================================== test
if stage == "test":
    assert os.path.exists(E("lock.json")), "no lock"; assert not os.path.exists(E("test_results.json")), "the test queries were already scored"
    L = json.load(open(E("lock.json"))); assert hashlib.sha256(open(E("locked_directions.npz"), "rb").read()).hexdigest() == L["directions_sha256"]
    Dz = np.load(E("locked_directions.npz")); BLOCK = L["block"]; nR = CFG["random_draws"]; METH = ["pooled", "raw", "svd_head", "svd_tail", "fisher", "boundary"]; rows, PQ, cert, tsetup = [], [], {}, {}
    rnd_hits = np.zeros(nR, int); rnd_n = 0; rnd_task = {}
    for t in TASKS:
        assert L["reference_manifest"][t]["n"] == len(BANK[t])
        te, Q, yq, X, yb = task_data("test", t, BLOCK); _, Qp, _, Xp, _ = task_data("test", t, "pooled"); b = BANK[t]; al = Dz[f"{tag(t)}|alpha"]; d0, _ = base_and_proj(Q, X); n = len(yq)
        npos = (yb[None, :] == yq[:, None]).sum(1); srt = np.sort(d0, axis=1); tr = pd.read_csv(P("split", f"train_{tag(t)}.csv"), dtype=str); va = pd.read_csv(P("split", f"val_{tag(t)}.csv"), dtype=str)
        tsetup[t] = {"queries": n, "per_class": pd.Series(yq).value_counts().sort_index().to_dict(), "no_correct_class_reference": int((npos == 0).sum()), "fewer_than_4_wrong_class_references": int((len(yb) - npos < K).sum()),
                     "paths_shared_with_bank_train_val": len(set(te.path) & (set(b.path) | set(tr.path) | set(va.path))), "md5_shared_with_bank_train_val": len(set(te.md5) & (set(b.md5) | set(tr.md5) | set(va.md5))),
                     "ordered_top4_equal_to_cosine_top4": int((top(d0) == np.argsort(-(Q @ X.T), axis=1, kind="stable")[:, :K]).all(1).sum()), "queries_with_exact_distance_tie_at_rank_4_5": int((srt[:, K - 1] == srt[:, K]).sum()),
                     "nearest_reference_cosine_ge_0.999_at_block": int(((Q @ X.T).max(1) >= 0.999).sum()), "nearest_reference_cosine_ge_0.999_pooled": int(((Qp @ Xp.T).max(1) >= 0.999).sum())}
        if t == "DeepWeeds":
            bd = np.array([group_of(t, p) for p in b.path]); qd = te.group.values
            tsetup[t].update({"capture_dates": int(len(set(qd))), "queries_on_a_bank_capture_date": int(np.isin(qd, bd).sum()), "queries_with_a_same_date_correct_class_reference": int(sum(((bd == g) & (yb == y)).any() for g, y in zip(qd, yq))),
                              "capture_dates_shared_with_train": len(set(qd) & set(tr.group)), "capture_dates_shared_with_val": len(set(qd) & set(va.group))})
        res = {}
        for m in METH:
            t0 = time.time()
            if m == "pooled": dd = base_and_proj(Qp, Xp)[0]; a_ = 0.0
            elif m == "raw": dd = d0; a_ = 0.0
            else: a_ = float(al[{"boundary": 0, "fisher": 1, "svd_head": 2, "svd_tail": 3}[m]]); dd = d0 + a_ * base_and_proj(Q, X, Dz[f"{tag(t)}|{m}"])[1]
            h, c, tp = hits(dd, yq, yb); res[m] = (h, c); rows.append({"task": t, "method": m, "alpha": a_, "n": n, "hits": int(h.sum()), "hit4_pct": 100 * h.mean(), "mean_correct_in_top4": float(c.mean()), "retrieval_seconds": round(time.time() - t0, 4)})
            PQ.append(pd.DataFrame({"task": t, "query": te.path.values, "label": yq, "group": te.group.values, "method": m, "hit": h.astype(int), "correct_in_top4": c, "top4": ["|".join(b.path.values[r]) for r in tp]}))
        rh = np.zeros(nR, int)
        for j in range(nR): rh[j] = hits(d0 + al[4 + j] * base_and_proj(Q, X, Dz[f"{tag(t)}|random"][j])[1], yq, yb)[0].sum()
        rnd_hits += rh; rnd_n += n; rnd_task[t] = rh.tolist()
        # ---- certificate against every Fisher gain in [0, A] (locked Fisher direction), and the hindsight Fisher gain
        f = Dz[f"{tag(t)}|fisher"]; _, s = base_and_proj(Q, X, f); dA = d0 + A * s; hp = res["boundary"][0]; cand, H, prof, _ = sweep(d0, s, yq, yb); inR = np.zeros(n, bool); can = np.zeros(n, bool)
        for i in range(n):
            pos, neg = np.flatnonzero(yb == yq[i]), np.flatnonzero(yb != yq[i]); can[i] = bool(prof[i][3] or prof[i][4] or prof[i][1].any() or prof[i][2].any())
            inR[i] = bool(hp[i]) and len(pos) > 0 and all(((d0[i, neg] < d0[i, p]) & (dA[i, neg] < dA[i, p])).sum() >= K for p in pos)
        assert not (inR & can).any()                      # a certified query can never be a Fisher hit
        cert[t] = {"n": n, "R": int(inR.sum()), "L": int((~hp & can).sum()), "proposed_hits": int(hp.sum()), "proposed_misses": int((~hp).sum()), "fisher_hits_at_validation_gain": int(res["fisher"][0].sum()),
                   "fisher_oracle_gain": pick(cand, H)[0], "fisher_oracle_hits": int(H.max()), "fisher_hits_min_over_gain": int(H.min())}
    R = pd.DataFrame(rows); PQ = pd.concat(PQ); R.to_csv(E("test_by_task.csv"), index=False); PQ.to_csv(E("test_per_query.csv.gz"), index=False, compression="gzip")
    W = PQ.pivot_table(index=["task", "query", "group"], columns="method", values="hit").reset_index(); N = len(W); po, fo = int(((W.boundary == 1) & (W.fisher == 0)).sum()), int(((W.boundary == 0) & (W.fisher == 1)).sum())
    assert W.boundary.sum() - W.fisher.sum() == po - fo
    rng = np.random.default_rng(CFG["seed"] + 1); est = {"boundary_minus_fisher": [], "boundary_minus_raw": [], "fisher_minus_raw": []}; units = {}
    for t in TASKS:
        g = W[W.task == t]; key = g.group.values if t == "DeepWeeds" else np.arange(len(g)).astype(str); u_ = sorted(set(key)); units[t] = len(u_)
        units[t + "|arr"] = (np.array([[g.boundary.values[key == k].sum(), g.fisher.values[key == k].sum(), g.raw.values[key == k].sum(), (key == k).sum()] for k in u_]))
    for _ in range(CFG["bootstrap"]):
        tot = np.zeros(4)
        for t in TASKS: a_ = units[t + "|arr"]; tot += a_[rng.integers(len(a_), size=len(a_))].sum(0)
        est["boundary_minus_fisher"].append(100 * (tot[0] - tot[1]) / tot[3]); est["boundary_minus_raw"].append(100 * (tot[0] - tot[2]) / tot[3]); est["fisher_minus_raw"].append(100 * (tot[1] - tot[2]) / tot[3])
    ci = {k: [float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))] for k, v in est.items()}
    from scipy.stats import binomtest
    Rn, Ln = sum(c["R"] for c in cert.values()), sum(c["L"] for c in cert.values()); orc = sum(c["fisher_oracle_hits"] for c in cert.values())
    outj = {"scored_at": time.strftime("%Y-%m-%d %H:%M:%S"), "block": BLOCK, "N": N, "setup_test": tsetup,
            "aggregate": {m: {"hits": int(W[m].sum()), "n": N, "hit4_pct": 100 * W[m].mean(), "mean_correct_in_top4": float(PQ[PQ.method == m].correct_in_top4.mean())} for m in METH},
            "paired_boundary_vs_fisher": {"proposed_only_hits": po, "fisher_only_hits": fo, "delta_pp": 100 * (po - fo) / N, "identity_holds": bool(abs((W.boundary.mean() - W.fisher.mean()) - (po - fo) / N) < 1e-15),
                                          "ci95_pp": ci["boundary_minus_fisher"], "exact_mcnemar_p": float(binomtest(po, po + fo, 0.5).pvalue) if po + fo else 1.0},
            "paired_vs_raw": {m: {"method_only": int(((W[m] == 1) & (W.raw == 0)).sum()), "raw_only": int(((W[m] == 0) & (W.raw == 1)).sum())} for m in METH if m != "raw"},
            "ci95_pp": ci, "bootstrap_units": {t: units[t] for t in TASKS}, "bootstrap_unit_note": "DeepWeeds: capture dates; other tasks: queries (no group field)",
            "random": {"draws": nR, "n": rnd_n, "hit4_pct": (100 * rnd_hits / rnd_n).tolist(), "by_task_hits": rnd_task, "best_validation_draw": L["random_best_validation_draw"]},
            "certificate": {"by_task": cert, "R": Rn, "L": Ln, "L_is_exact": True, "lower_bound_pp": 100 * (Rn - Ln) / N, "strictly_positive": bool(Rn > Ln),
                            "oracle_fisher_hits": orc, "oracle_fisher_hit4_pct": 100 * orc / N, "oracle_gap_pp_proposed_minus_fisher": 100 * (int(W.boundary.sum()) - orc) / N}}
    json.dump(outj, open(E("test_results.json"), "w"), indent=1); print(R.to_string()); print(json.dumps({k: outj[k] for k in ("aggregate", "paired_boundary_vs_fisher", "certificate")}, indent=1)); sys.exit()

# ================================================================== report
if stage == "report":
    L = json.load(open(E("lock.json"))); T = json.load(open(E("test_results.json"))); MR = pd.read_csv(E("margin_identity.csv")); V = pd.read_csv(E("validation_all_directions.csv")); R = pd.read_csv(E("test_by_task.csv")); CC = pd.read_csv(E("calibration_cost.csv"))
    N = T["N"]; ag = T["aggregate"]; pf = T["paired_boundary_vs_fisher"]; ce = T["certificate"]; d = L.get("dim", 768); nt = len(TASKS); S = []
    S += [f"# Rank-one intervention on one frozen block of {ENC}: signed boundary scatter against regularized Fisher\n", "Retrieval only, no VLM call. Commands (from `repo/`): `python rank1_boundary.py ../runs/rank1_boundary <stage> <encoder>` with stages prep, features, selftest, fit, test, report.\n",
          f"- **Encoder and block:** {CFG['encoder']}, {READOUT} readout of block {L['block']}, selected on the train queries ({RULES['block']}). K = {K}, gain interval [0, {A:g}].",
          f"- **Split:** reference bank = frozen 100-image bank per dataset; test = {CFG['test']}; train = {CFG['n_train']} and validation = {CFG['n_val']} fresh queries per dataset.",
          f"- **Limitation of the test set:** {CFG['test_limit']}.", f"- **Calibration:** {L['calibration']}.", f"- **Locked at** {L['locked_at']}; **scored at** {T['scored_at']}.\n", "## Setup and leakage checks\n",
          "| Task | Classes | Bank | Train (eligible pairs) | Validation | Test | No correct-class reference (train / val / test) | Fewer than 4 wrong-class references | Images shared with the bank (path or md5) |", "|---|---|---|---|---|---|---|---|---|"]
    for t in TASKS:
        s_, e_ = L["setup"][t], T["setup_test"][t]
        S.append(f"| {t} | {s_['classes']} | {s_['bank']} | {s_['train']['queries']} ({s_['train']['eligible_for_pairs']}) | {s_['val']['queries']} | {e_['queries']} | {s_['train']['no_correct_class_reference']} / {s_['val']['no_correct_class_reference']} / {e_['no_correct_class_reference']} | "
                 f"{s_['train']['fewer_than_4_wrong_class_references']} / {s_['val']['fewer_than_4_wrong_class_references']} / {e_['fewer_than_4_wrong_class_references']} | {s_['train']['paths_shared_with_bank'] + s_['train']['md5_shared_with_bank'] + s_['val']['paths_shared_with_bank'] + s_['val']['md5_shared_with_bank'] + e_['paths_shared_with_bank_train_val'] + e_['md5_shared_with_bank_train_val']} |")
    dw, dwt = L["setup"]["DeepWeeds"], T["setup_test"]["DeepWeeds"]
    S += ["", f"Group field: DeepWeeds capture date (file name). The splits are not separated by capture date: {dw['train']['queries_on_a_bank_capture_date']} train, {dw['val']['queries_on_a_bank_capture_date']} validation and {dwt['queries_on_a_bank_capture_date']} test queries were taken on a date that is also in the bank; "
          f"the test queries share {dwt['capture_dates_shared_with_train']} dates with train and {dwt['capture_dates_shared_with_val']} with validation. The other four datasets have no plant or date field.",
          "With alpha = 0 the ordered top four equal the cosine top four on " + ", ".join(f"{T['setup_test'][t]['ordered_top4_equal_to_cosine_top4']} of {T['setup_test'][t]['queries']} ({t})" for t in TASKS) + " test queries.\n",
          "## Equal-gain margin identity on the frozen train pairs\n", "| Task | alpha | Observed delta | Spectral delta | Abs. discrepancy | Tolerance | Pass | lambda_1 | Eigengap | Fisher squared alignment with u |", "|---|---|---|---|---|---|---|---|---|---|"]
    S += [f"| {r.task} | {r.alpha:g} | {r.observed_delta:.6e} | {r.spectral_delta:.6e} | {r.abs_discrepancy:.1e} | {r.tolerance:.0e} | {'yes' if r['pass'] else 'NO'} | {r.lambda_1:.4e} | {r.eigengap:.4e} | {r.fisher_sq_alignment_leading:.4f} |" for _, r in MR.iterrows()]
    S += ["", "## Validation and test retrieval\n", "| Method | Fitted values per task | Validation Hit@4 | Test Hit@4 | vs raw block (pp) | vs Fisher (pp) | Mean correct in top 4 |", "|---|---|---|---|---|---|---|"]
    name = {"pooled": "Pooled output, cosine", "raw": "Raw hidden block (alpha = 0)", "svd_head": "SVD head", "svd_tail": "SVD tail", "fisher": "Regularized Fisher", "boundary": "Signed boundary scatter (proposed)"}
    dof = {"pooled": "0", "raw": "0", "svd_head": f"{d - 1} + gain", "svd_tail": f"{d - 1} + gain", "fisher": f"{d - 1} + gain + rho", "boundary": f"{d - 1} + gain"}; nv = int(V[V.direction == "boundary"].n_val.sum())
    def vh(m):
        if m == "pooled": return "not scored"
        if m == "raw": return int(V[V.direction == "boundary"].val_hits_alpha0.sum())
        if m == "fisher": return int(sum(V[(V.task == t) & (V.direction == f"fisher_c{L['fisher'][t]['rho_rel_selected']:g}")].val_hits_best.iloc[0] for t in TASKS))
        return int(V[V.direction == m].val_hits_best.sum())
    for m in name:
        v = vh(m); S.append(f"| {name[m]} | {dof[m]} | {v if isinstance(v, str) else f'{v}/{nv} ({100 * v / nv:.1f}%)'} | {ag[m]['hits']}/{N} ({ag[m]['hit4_pct']:.1f}%) | {ag[m]['hit4_pct'] - ag['raw']['hit4_pct']:+.1f} | {ag[m]['hit4_pct'] - ag['fisher']['hit4_pct']:+.1f} | {ag[m]['mean_correct_in_top4']:.2f} |")
    rp = np.array(T["random"]["hit4_pct"]); S += [f"| Random direction, {T['random']['draws']} draws | gain | | median {np.median(rp):.1f}% (min {rp.min():.1f}%, max {rp.max():.1f}%) | {np.median(rp) - ag['raw']['hit4_pct']:+.1f} | {np.median(rp) - ag['fisher']['hit4_pct']:+.1f} | |",
          f"| Random direction with the best validation score (draw {T['random']['best_validation_draw']}) | gain | | {rp[T['random']['best_validation_draw']]:.1f}% | | | |", "",
          "**Test Hit@4 by task** (hits of " + ", ".join(f"{T['setup_test'][t]['queries']}" for t in TASKS) + "; selected gain in parentheses):\n", "| Method | " + " | ".join(TASKS) + " |", "|---|" + "---|" * nt]
    for m in name: S.append(f"| {name[m]} | " + " | ".join(f"{int(R[(R.task == t) & (R.method == m)].hits.iloc[0])} ({R[(R.task == t) & (R.method == m)].alpha.iloc[0]:.3g})" for t in TASKS) + " |")
    S += ["", "Fisher regularization selected per task: " + "; ".join(f"{t}: c = {L['fisher'][t]['rho_rel_selected']:g}" for t in TASKS) + f" (rho = c trace(S_pos) / d, c in {CFG['fisher_rho_rel']}).",
          f"Fit time per task: boundary {V[V.direction == 'boundary'].fit_seconds.mean():.2f} s, Fisher {V[V.direction.str.startswith('fisher')].fit_seconds.mean():.2f} s per rho, SVD {V[V.direction == 'svd_head'].fit_seconds.mean():.2f} s. "
          f"Retrieval time per task: {R.retrieval_seconds.mean():.3f} s. Exact calibration: {int(CC['pair_roots_in_(0,A)'].sum())} pair roots over {len(CC)} sweeps, {CC.seconds.sum() / 60:.1f} min in total.\n",
          "## Proposed against Fisher on the same test queries\n", f"- Proposed-only hits: **{pf['proposed_only_hits']}**; Fisher-only hits: **{pf['fisher_only_hits']}**; N = {N}.",
          f"- Difference: **{pf['delta_pp']:+.2f} pp** = ({pf['proposed_only_hits']} - {pf['fisher_only_hits']}) / {N}; the identity holds: {pf['identity_holds']}.",
          f"- Paired 95% bootstrap interval: {pf['ci95_pp'][0]:+.2f} to {pf['ci95_pp'][1]:+.2f} pp ({CFG['bootstrap']} resamples; {T['bootstrap_unit_note']}; DeepWeeds has {T['bootstrap_units']['DeepWeeds']} capture dates). Exact McNemar p = {pf['exact_mcnemar_p']:.3f}.",
          f"- Against the raw block: proposed {T['ci95_pp']['boundary_minus_raw'][0]:+.2f} to {T['ci95_pp']['boundary_minus_raw'][1]:+.2f} pp; Fisher {T['ci95_pp']['fisher_minus_raw'][0]:+.2f} to {T['ci95_pp']['fisher_minus_raw'][1]:+.2f} pp.\n",
          "## Sufficient certificate against every Fisher gain in [0, A]\n", f"- |R| = {ce['R']}, |L| = {ce['L']} (exact per-query sweep), certified lower bound ({ce['R']} - {ce['L']}) / {N} = **{ce['lower_bound_pp']:+.2f} pp**. "
          + ("The bound is strictly positive." if ce["strictly_positive"] else "The bound is not positive, so the certificate is inconclusive about superiority."),
          f"- Oracle comparison (Fisher gain chosen in hindsight on the test queries, per task): Fisher {ce['oracle_fisher_hits']}/{N} ({ce['oracle_fisher_hit4_pct']:.1f}%); proposed minus oracle Fisher {ce['oracle_gap_pp_proposed_minus_fisher']:+.2f} pp. The validation-selected Fisher result above is the primary comparison.\n",
          "## Files\n", "`lock.json` (block, gains, rho, rules, reference manifest, setup checks), `locked_directions.npz`, `train_pairs.csv`, `block_selection_train.csv`, `validation_all_directions.csv`, `validation_curves.npz`, `calibration_cost.csv`, `margin_identity.csv`, `test_by_task.csv`, `test_per_query.csv.gz`, `test_results.json`, `selftest.json`, `split/`."]
    open(E("REPORT.md"), "w").write("\n".join(S) + "\n"); print("\n".join(S))
