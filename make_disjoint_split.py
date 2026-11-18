"""Disjoint query and reference sets for the five paper datasets: 100 queries and a separate 100-image reference bank each.

Queries: the existing AgEval loader pool (so earlier results on those images stay usable), topped up to exactly 100 with
images from the least represented classes. Reference bank: 100 class-balanced images drawn from the rest of the dataset.
No image is in both sets, and no bank image is byte-identical to a query. Seed 42. No model calls.
Usage (from repo/): python make_disjoint_split.py <out_dir>
"""
import hashlib, json, os, sys
import numpy as np, pandas as pd
import data_loader as dl

out = sys.argv[1]; os.makedirs(out, exist_ok=True); SEED, NQ, NB = 42, 100, 100
OFF = os.path.join("runs", "offline_layer_stats")
DS = {"Bean Leaf Lesions": dl.load_and_prepare_data_bean_leaf, "Mango Leaf Disease": dl.load_and_prepare_data_mango_leaf, "DeepWeeds": dl.load_and_prepare_data_DeepWeeds,
      "Soybean Seeds": dl.load_and_prepare_data_soybean_seeds, "Soybean Diseases": dl.load_and_prepare_data_Soybean_PNAS}
md5 = lambda p: hashlib.md5(open(p, "rb").read()).hexdigest(); man = {}
for name, loader in DS.items():
    tag = name.replace(" ", "_"); rng = np.random.default_rng([SEED, sorted(DS).index(name)])
    full, classes, _ = loader(10 ** 7); full = pd.DataFrame({"path": full[0], "label": full[1].astype(str)}).drop_duplicates("path").sort_values("path").reset_index(drop=True)
    pool = pd.read_csv(os.path.join(OFF, f"pool_{tag}.csv"), dtype={"label": str})[["path", "label", "md5"]]; assert set(pool.path) <= set(full.path), f"{name}: pool images missing from the full listing"
    labels = sorted(full.label.unique()); rest = full[~full.path.isin(set(pool.path))]
    q = pool.copy(); q["origin"] = "AgEval loader pool"; q_md5 = set(q.md5)
    while len(q) < NQ:                                   # top up from the class with the fewest queries (alphabetical on ties)
        c = min(labels, key=lambda l: ((q.label == l).sum(), l)); cand = rest[(rest.label == c) & ~rest.path.isin(set(q.path))]; p = cand.path.iloc[rng.integers(len(cand))]; m = md5(p)
        if m in q_md5: rest = rest[rest.path != p]; continue
        q = pd.concat([q, pd.DataFrame([{"path": p, "label": c, "md5": m, "origin": "added to reach 100"}])], ignore_index=True); q_md5.add(m)
    rest = rest[~rest.path.isin(set(q.path))]; per = {l: NB // len(labels) + (1 if i < NB % len(labels) else 0) for i, l in enumerate(labels)}; bank = []; dropped = 0
    for l in labels:
        cand = rest[rest.label == l].path.values[rng.permutation((rest.label == l).sum())]; got = []
        for p in cand:
            if len(got) == per[l]: break
            m = md5(p)
            if m in q_md5 or m in {g[2] for g in bank + got}: dropped += 1; continue
            got.append((p, l, m))
        assert len(got) == per[l], f"{name}: class {l} has only {len(got)} images outside the query set"; bank += got
    b = pd.DataFrame(bank, columns=["path", "label", "md5"])
    assert not set(q.path) & set(b.path) and not set(q.md5) & set(b.md5) and len(q) == NQ and len(b) == NB
    q.to_csv(os.path.join(out, f"queries_{tag}.csv"), index=False); b.to_csv(os.path.join(out, f"bank_{tag}.csv"), index=False)
    man[name] = {"dataset_images": int(len(full)), "classes": len(labels), "queries": int(len(q)), "queries_from_the_loader_pool": int((q.origin == "AgEval loader pool").sum()), "queries_added": int((q.origin != "AgEval loader pool").sum()),
                 "queries_per_class": q.label.value_counts().sort_index().to_dict(), "bank": int(len(b)), "bank_per_class": b.label.value_counts().sort_index().to_dict(), "images_in_both_sets": 0, "byte_identical_query_bank_pairs": 0,
                 "bank_candidates_skipped_as_byte_identical": dropped}
    print(name, man[name], flush=True)
json.dump({"seed": SEED, "rule": __doc__.strip().split("\n\n")[1], "datasets": man}, open(os.path.join(out, "manifest.json"), "w"), indent=1)
