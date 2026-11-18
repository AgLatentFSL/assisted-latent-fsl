"""Local retrieval tool for the Qwen latent database. No Claude or other API call.

CLI:
  python qwen_retrieve.py --db ../runs/qwen_latent_db --query data/deepweeds/images/<file>.jpg --mode embedding --bank base_dd99 --k 4
  python qwen_retrieve.py --db ... --query ... --mode hidden_state --bank base_dd99
Callable:
  from qwen_retrieve import Retriever; r = Retriever(db); r.retrieve(query_path, mode="hidden_state", bank_id="base_dd99", k=4)

The response lists ordered references with their image paths, labels and scores. It carries no predicted class and no neighbor vote.
The query's label is never read. Eligibility is applied before ranking: the bank, the query image itself (by ID and by file hash),
the query's capture date, and same-camera images within 120 s of the query.
"""
import argparse, hashlib, json, os, re
import numpy as np
import pandas as pd

MODES = {"embedding": "pooled_embedding.npy", "hidden_state": "pooled_hidden_state.npy"}
NEAR_SECONDS = 120


def _stamp(name):
    m = re.match(r"(\d{8})-(\d{6})-(\d+)\.jpg$", os.path.basename(name))
    return (m.group(1), pd.to_datetime(m.group(1) + m.group(2), format="%Y%m%d%H%M%S"), m.group(3)) if m else (None, None, None)


class Retriever:
    def __init__(self, db):
        self.db = db; self.manifest = json.load(open(os.path.join(db, "manifest.json")))
        self.R = pd.read_csv(os.path.join(db, "references.csv"), dtype={"capture_date": str})
        mem = pd.read_csv(os.path.join(db, "bank_membership.csv")); self.banks = {b: set(g["image_id"]) for b, g in mem.groupby("bank_id")}
        self.V = {m: np.load(os.path.join(db, f)) for m, f in MODES.items()}
        assert all(v.shape[0] == len(self.R) for v in self.V.values()), "pooled vectors and references.csv are not aligned"
        st = [_stamp(i) for i in self.R["image_id"]]; self.R["_t"] = [s[1] for s in st]; self.R["_cam"] = [s[2] for s in st]
        self._enc = None

    def _encoder(self):
        if self._enc is None:
            import qwen_latent_db as Q
            self._enc = Q.Encoder(); m = self.manifest
            for k in ("model_id", "revision", "block", "instruction", "prompt_layout", "pooling", "dtype", "transformers"):
                assert self._enc.meta[k] == m[k], f"encoder setting {k} differs from the database manifest"
            self._pool = Q.pool
        return self._enc

    def encode_query(self, path):
        """Both pooled query vectors from one forward pass, with the stored model, instruction, block and pooling."""
        emb, hid = self._encoder().encode(path)
        assert emb.shape[0] == self.manifest["image_tokens"], "query token count differs from the index"
        return {"embedding": self._pool(emb), "hidden_state": self._pool(hid)}

    def retrieve(self, query, mode="embedding", bank_id="base_dd99", k=4, query_date=None, allow_same_date=False, query_vectors=None):
        assert mode in MODES, f"mode must be one of {list(MODES)}"; assert bank_id in self.banks, f"unknown bank {bank_id}"
        qid = os.path.basename(query); date, t, cam = _stamp(qid); date = query_date or date
        qmd5 = hashlib.md5(open(query, "rb").read()).hexdigest() if os.path.exists(query) else None
        R = self.R; elig = R["indexed"].to_numpy() & R["image_id"].isin(self.banks[bank_id]).to_numpy(); n_bank = int(elig.sum())
        is_self = (R["image_id"] == qid).to_numpy() | ((R["md5"] == qmd5).to_numpy() if qmd5 else False)
        same_date = (R["capture_date"] == date).to_numpy() if date else np.zeros(len(R), bool)
        near = ((R["capture_date"] == date) & (R["_cam"] == cam) & ((R["_t"] - t).abs().dt.total_seconds() <= NEAR_SECONDS)).to_numpy() if t is not None else np.zeros(len(R), bool)
        excl = {"query_image": int((elig & is_self).sum()), "near_duplicate_in_time": int((elig & near & ~is_self).sum()),
                "same_capture_date": 0 if allow_same_date else int((elig & same_date & ~is_self & ~near).sum())}
        elig &= ~is_self & ~near
        if not allow_same_date: elig &= ~same_date
        qv = (query_vectors or self.encode_query(query))[mode].astype(np.float32)
        idx = np.where(elig)[0]; s = self.V[mode][idx] @ qv
        order = np.lexsort((R["image_id"].to_numpy()[idx], -s))[:k]                      # descending cosine; ties broken by image ID
        res = [{"rank": r + 1, "image_id": R["image_id"].iloc[idx[j]], "image_path": os.path.abspath(R["path"].iloc[idx[j]]), "label": R["label"].iloc[idx[j]],
                "capture_date": R["capture_date"].iloc[idx[j]], "score": round(float(s[j]), 6)} for r, j in enumerate(order)]
        warn = [] if date else ["the query file name carries no capture date and none was given; the capture-date rule was not applied"]
        if len(res) < k: warn.append(f"only {len(res)} eligible references are available")
        return {"tool": "retrieve_references", "index_version": self.manifest["index_version"], "mode": mode, "representation": self.manifest["modes"][mode], "model": f"{self.manifest['model_id']}@{self.manifest['revision'][:12]}",
                "instruction": self.manifest["instruction"] if mode == "hidden_state" else None, "bank_id": bank_id, "k": k, "scoring": "cosine similarity of L2-normalized mean-pooled image tokens",
                "query": {"image_id": qid, "image_path": os.path.abspath(query), "capture_date": date}, "eligibility": {"bank_references": n_bank, "excluded": excl, "eligible_after_exclusions": int(elig.sum()), "same_date_allowed": bool(allow_same_date)},
                "results": res, "warnings": warn}


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--db", required=True); ap.add_argument("--query", required=True); ap.add_argument("--mode", choices=list(MODES), default="embedding")
    ap.add_argument("--bank", default="base_dd99"); ap.add_argument("--k", type=int, default=4); ap.add_argument("--query-date", default=None); ap.add_argument("--allow-same-date", action="store_true")
    a = ap.parse_args(); print(json.dumps(Retriever(a.db).retrieve(a.query, a.mode, a.bank, a.k, a.query_date, a.allow_same_date), indent=1))
