"""DeepWeeds retrieval database for Assisted Latent Few-Shot Learning, from one frozen Qwen vision-language checkpoint.

Two representations per reference image, both from forward passes with frozen weights (no generation, no fine-tuning):
  embedding    : the projected visual tokens at the image-token positions of the decoder input (inputs_embeds)
  hidden_state : the output of decoder block BLOCK at the same positions, with a fixed instruction placed before the image
No Claude or other API call is made anywhere in this file.

Usage:
  python qwen_latent_db.py inventory <db_dir>            # manifests only, no model
  python qwen_latent_db.py smoke <db_dir>                # one image: verify both extraction paths, write smoke_test.json
  python qwen_latent_db.py build <db_dir> [--limit N]    # encode eligible references (resumable)
  python qwen_latent_db.py queries <db_dir> <csv> <tag>  # encode a query list (query_id column) into queries_<tag>.npz
"""
import hashlib, json, os, sys, time
import numpy as np
import pandas as pd
from PIL import Image

MODEL_ID, REVISION = "Qwen/Qwen3-VL-2B-Instruct", "89644892e4d85e24eaac8bacfd4f463576704203"
INSTRUCTION = "Focus on visible plant features useful for distinguishing weed species."
BLOCK = 14                # 1-indexed decoder block whose output is stored; the model has 28 blocks, so this is the middle block
SHARD = 256
DATA = os.path.join("data", "deepweeds"); RUNS = os.path.join("runs")
SRC = {"labels": os.path.join(DATA, "labels", "labels.csv"), "base_bank": os.path.join(RUNS, "deepweeds_matched_seed42", "reference_manifest.csv"),
       "base_split": os.path.join(RUNS, "deepweeds_matched_seed42", "split.json"), "base_queries": os.path.join(RUNS, "deepweeds_matched_seed42", "query_manifest.csv"),
       "panel": os.path.join(RUNS, "deepweeds_split_audit", "query_panel.csv"), "audit_banks": os.path.join(RUNS, "deepweeds_split_audit", "banks.csv"),
       "dev_queries": os.path.join(RUNS, "latent_afsl", "dev_queries.csv"), "dev_banks": os.path.join(RUNS, "latent_afsl", "dev_banks.csv"),
       "prior_pool": os.path.join(RUNS, "deepweeds_seed42", "reference_manifest.csv")}


def ipath(f): return os.path.join(DATA, "images", f)
def md5(p): return hashlib.md5(open(p, "rb").read()).hexdigest()
def wjson(p, o):
    with open(p, "w") as f: json.dump(o, f, indent=1, sort_keys=True)


def inventory(db):
    """Reference manifest, bank membership and query registry from the existing split files. Query labels are not written."""
    os.makedirs(db, exist_ok=True)
    L = pd.read_csv(SRC["labels"]); L["date"] = L["Filename"].str[:8]
    lab, dat = L.set_index("Filename")["Species"], L.set_index("Filename")["date"]
    base = pd.read_csv(SRC["base_bank"]); aud = pd.read_csv(SRC["audit_banks"]); dev = pd.read_csv(SRC["dev_banks"])
    mem = [pd.DataFrame({"bank_id": "base_dd99", "image_id": base["ref_id"], "label_in_bank_file": base["label"]})]
    mem += [pd.DataFrame({"bank_id": f"audit_{bt.lower()}_s{int(s):02d}", "image_id": g["ref_id"], "label_in_bank_file": g["label"]}) for (bt, s), g in aud.groupby(["bank_type", "bank_seed"])]
    mem += [pd.DataFrame({"bank_id": f"dev_s{int(s):02d}", "image_id": g["ref_id"], "label_in_bank_file": g["label"]}) for s, g in dev.groupby("bank_seed")]
    mem = pd.concat(mem, ignore_index=True)
    panel = pd.read_csv(SRC["panel"]); bq = pd.read_csv(SRC["base_queries"]); dq = pd.read_csv(SRC["dev_queries"])
    Q = pd.concat([pd.DataFrame({"image_id": panel["query_id"], "split": "panel_" + panel["role"]}), pd.DataFrame({"image_id": bq["query_id"], "split": "base_run_query"}),
                   pd.DataFrame({"image_id": dq["query_id"], "split": "readout_dev_query"})], ignore_index=True)
    Q["capture_date"] = Q["image_id"].str[:8]
    eval_q = set(Q[Q["split"].isin(["panel_eval", "panel_audit_only", "panel_dev", "base_run_query"])]["image_id"])       # evaluation and prompt-development queries: never references
    ids = sorted(set(mem["image_id"]))
    issues = {"ids_not_in_labels_csv": [i for i in ids if i not in lab.index], "missing_image_files": [i for i in ids if not os.path.exists(ipath(i))],
              "label_conflicts_between_bank_files_and_labels_csv": int((mem["label_in_bank_file"] != mem["image_id"].map(lab)).sum()),
              "duplicate_filenames_in_labels_csv": int(L["Filename"].duplicated().sum()),
              "bank_images_that_are_evaluation_or_prompt_dev_queries": sorted(set(ids) & eval_q),
              "bank_images_that_are_readout_dev_queries": len(set(ids) & set(dq["query_id"]))}
    ids = [i for i in ids if i in lab.index and os.path.exists(ipath(i)) and i not in eval_q]
    R = pd.DataFrame({"row": range(len(ids)), "image_id": ids, "path": [ipath(i) for i in ids], "label": [lab[i] for i in ids], "capture_date": [dat[i] for i in ids]})
    R["md5"] = [md5(p) for p in R["path"]]
    R["also_readout_dev_query"] = R["image_id"].isin(set(dq["query_id"]))
    base_md5 = dict(zip(base["ref_id"], base["md5"])); issues["base_bank_md5_mismatches"] = int(sum(R.set_index("image_id")["md5"].get(i) != m for i, m in base_md5.items()))
    issues["duplicate_md5_among_references"] = int(R["md5"].duplicated().sum())
    mem = mem[mem["image_id"].isin(set(ids))][["bank_id", "image_id"]]
    R.to_csv(os.path.join(db, "references.csv"), index=False); mem.to_csv(os.path.join(db, "bank_membership.csv"), index=False); Q.to_csv(os.path.join(db, "query_registry.csv"), index=False)
    inv = {"dataset_images": int(len(L)), "dataset_classes": sorted(L["Species"].unique()), "dataset_capture_dates": int(L["date"].nunique()), "eligible_references": len(R),
           "banks": int(mem["bank_id"].nunique()), "bank_sizes": mem.groupby("bank_id").size().describe()[["min", "max"]].to_dict(), "references_per_class": R["label"].value_counts().sort_index().to_dict(),
           "reference_capture_dates": int(R["capture_date"].nunique()), "query_registry": Q["split"].value_counts().to_dict(), "issues": issues, "source_files": SRC,
           "defined_by": {"image_ids_and_class_names": SRC["labels"], "capture_date": "first eight characters of the file name (YYYYMMDD)", "frozen_reference_bank": SRC["base_bank"],
                          "test_split": SRC["base_split"] + " and " + SRC["panel"], "audit_banks": SRC["audit_banks"], "development_split": SRC["dev_queries"] + " and " + SRC["dev_banks"]},
           "eligibility_rule": "an image is indexed if it belongs to at least one saved bank, is not an evaluation or prompt-development query, has a label in labels.csv and exists on disk; "
                               "images that are readout-development queries stay indexed for the banks that contain them and are excluded at retrieval time when they are the query"}
    wjson(os.path.join(db, "inventory.json"), inv); return R, mem, inv


class Encoder:
    def __init__(self):
        import torch, transformers
        from transformers import AutoProcessor, Qwen3VLForConditionalGeneration
        self.torch = torch; self.device = "mps" if torch.backends.mps.is_available() else "cpu"; self.dtype = torch.float16 if self.device == "mps" else torch.float32
        self.processor = AutoProcessor.from_pretrained(MODEL_ID, revision=REVISION)
        self.model = Qwen3VLForConditionalGeneration.from_pretrained(MODEL_ID, revision=REVISION, torch_dtype=self.dtype).to(self.device).eval()
        for p in self.model.parameters(): p.requires_grad_(False)
        self.lm = self.model.model.language_model; self.n_blocks = len(self.lm.layers); self.image_token_id = self.model.config.image_token_id
        ip = self.processor.image_processor
        self.meta = {"model_id": MODEL_ID, "revision": REVISION, "transformers": transformers.__version__, "torch": torch.__version__, "device": self.device, "dtype": str(self.dtype),
                     "decoder_blocks": self.n_blocks, "hidden_size": int(self.model.config.text_config.hidden_size), "block": BLOCK,
                     "block_indexing": "1-indexed; block b is language_model.layers[b-1]; the stored tensor is the output of that layer's forward call, captured by a forward hook",
                     "deepstack_visual_indexes": list(self.model.config.vision_config.deepstack_visual_indexes), "instruction": INSTRUCTION,
                     "prompt_layout": "chat template, one user turn: the instruction text, then the image; no generation prompt",
                     "image_processor": {k: getattr(ip, k, None) for k in ("patch_size", "merge_size", "min_pixels", "max_pixels", "size", "do_resize", "image_mean", "image_std")},
                     "pooling": "tokens are cast to float32, averaged over the image-token axis, then L2-normalized; cosine similarity is the dot product of two such vectors",
                     "token_storage": "float16, shape [images, image_tokens, hidden_size], row order of references.csv"}
        self._cap = {}
        class Stop(Exception): pass
        self.Stop = Stop
        def pre(mod, args, kwargs): self._cap["inputs_embeds"] = kwargs["inputs_embeds"].detach()
        def post(mod, args, out_):
            self._cap["block"] = (out_[0] if isinstance(out_, tuple) else out_).detach(); raise Stop()          # nothing after this block is needed
        self.lm.register_forward_pre_hook(pre, with_kwargs=True); self.lm.layers[BLOCK - 1].register_forward_hook(post)

    def encode(self, path, instruction=INSTRUCTION, return_debug=False):
        torch = self.torch; img = Image.open(path).convert("RGB")
        msgs = [{"role": "user", "content": [{"type": "text", "text": instruction}, {"type": "image", "image": img}]}]
        text = self.processor.apply_chat_template(msgs, tokenize=False, add_generation_prompt=False)
        inputs = self.processor(text=[text], images=[img], return_tensors="pt").to(self.device)
        self._cap.clear()
        with torch.no_grad():
            try: self.model.model(**inputs)
            except self.Stop: pass
        mask = (inputs["input_ids"][0] == self.image_token_id)
        emb = self._cap["inputs_embeds"][0][mask].float().cpu().numpy(); hid = self._cap["block"][0][mask].float().cpu().numpy()
        if return_debug:
            return emb, hid, inputs, mask
        return emb, hid


def pool(tok):
    v = tok.astype(np.float32).mean(axis=0); return v / np.linalg.norm(v)


def smoke(db):
    R, mem, inv = inventory(db); t0 = time.time(); E = Encoder(); load_s = time.time() - t0; torch = E.torch
    row = R[R["image_id"].isin(set(mem[mem.bank_id == "base_dd99"]["image_id"]))].iloc[0]
    t0 = time.time(); emb, hid, inputs, mask = E.encode(row["path"], return_debug=True); enc_s = time.time() - t0
    ids = inputs["input_ids"][0].cpu(); pos = torch.nonzero(mask.cpu()).flatten().tolist(); tok = E.processor.tokenizer
    with torch.no_grad():
        feats, deep = E.model.model.get_image_features(inputs["pixel_values"].to(E.dtype), inputs["image_grid_thw"])
        direct = torch.cat(feats, dim=0).float().cpu().numpy()
        # a full pass without the early stop, to check that stopping after the block does not change the captured tensor
    emb2, hid2 = E.encode(row["path"])
    plain = [{"role": "user", "content": [{"type": "image", "image": Image.open(row["path"]).convert("RGB")}]}]
    other_emb, other_hid = E.encode(row["path"], instruction="Describe the soil.")
    S = {"image_id": row["image_id"], "label": row["label"], "source_image_size": list(Image.open(row["path"]).size), "model_load_seconds": round(load_s, 1), "first_encode_seconds": round(enc_s, 2),
         "sequence_length": int(len(ids)), "image_token_count": int(mask.sum()), "image_grid_thw": inputs["image_grid_thw"].cpu().tolist(), "pixel_values_shape": list(inputs["pixel_values"].shape),
         "image_token_positions_first_last": [pos[0], pos[-1]], "positions_contiguous": bool(pos == list(range(pos[0], pos[-1] + 1))),
         "token_before_image_span": tok.decode(ids[pos[0] - 1:pos[0]]), "token_after_image_span": tok.decode(ids[pos[-1] + 1:pos[-1] + 2]),
         "text_before_image": tok.decode(ids[:pos[0]]), "instruction_precedes_image": bool(INSTRUCTION in tok.decode(ids[:pos[0]])),
         "embedding_shape": list(emb.shape), "hidden_state_shape": list(hid.shape), "token_counts_agree": bool(emb.shape == hid.shape == (int(mask.sum()), E.meta["hidden_size"])),
         "embedding_equals_get_image_features_max_abs_diff": float(np.abs(emb - direct).max()), "embedding_finite": bool(np.isfinite(emb).all()), "hidden_state_finite": bool(np.isfinite(hid).all()),
         "repeat_encode_max_abs_diff": {"embedding": float(np.abs(emb - emb2).max()), "hidden_state": float(np.abs(hid - hid2).max())},
         "repeat_encode_pooled_cosine": {"embedding": float(pool(emb) @ pool(emb2)), "hidden_state": float(pool(hid) @ pool(hid2))},
         "instruction_changes_embedding_max_abs_diff": float(np.abs(emb - other_emb).max()), "instruction_changes_hidden_state_pooled_cosine": float(pool(hid) @ pool(other_hid)),
         "embedding_vs_hidden_state_pooled_cosine": float(pool(emb) @ pool(hid)), "embedding_token_norm_mean": float(np.linalg.norm(emb, axis=1).mean()), "hidden_state_token_norm_mean": float(np.linalg.norm(hid, axis=1).mean()),
         "deepstack_feature_levels": len(deep), "deepstack_note": f"the vision tower returns {len(deep)} extra feature maps from vision layers {E.meta['deepstack_visual_indexes']}; the text model adds them to the hidden states at the image positions "
                                                                  f"after decoder blocks 1 to {len(deep)}. Block {BLOCK} is past that point, so its output already contains them. The stored embedding is the plain decoder input and does not contain them.",
         "bytes_per_image_tokens_float16": int(emb.astype(np.float16).nbytes + hid.astype(np.float16).nbytes), "model_meta": E.meta}
    wjson(os.path.join(db, "smoke_test.json"), S); print(json.dumps({k: v for k, v in S.items() if k != "model_meta"}, indent=1)); return S


def manifest(db, E, n_tokens):
    m = {**E.meta, "image_tokens": int(n_tokens), "index_version": hashlib.sha256(json.dumps({k: E.meta[k] for k in ("model_id", "revision", "block", "instruction", "prompt_layout", "pooling", "dtype")}, sort_keys=True).encode()).hexdigest()[:12],
         "modes": {"embedding": "projected visual tokens at the image-token positions of the decoder input", "hidden_state": f"output of decoder block {BLOCK} at the same positions"}}
    p = os.path.join(db, "manifest.json")
    if os.path.exists(p):
        old = json.load(open(p)); same = all(old.get(k) == m[k] for k in ("model_id", "revision", "block", "instruction", "prompt_layout", "pooling", "dtype", "image_tokens", "transformers"))
        assert same, "existing manifest was built with a different configuration; use a new database directory"
    else:
        wjson(p, m)
    return m


def build(db, limit=None):
    R, mem, inv = inventory(db); os.makedirs(os.path.join(db, "tokens"), exist_ok=True); E = Encoder(); log = os.path.join(db, "build_log.jsonl"); t_start = time.time()
    # the frozen bank first, then the development banks, then the audit banks
    pri = {b: i for i, b in enumerate(["base_dd99"] + sorted(b for b in mem.bank_id.unique() if b.startswith("dev_")) + sorted(b for b in mem.bank_id.unique() if b.startswith("audit_dd")) + sorted(b for b in mem.bank_id.unique() if b.startswith("audit_sd")))}
    first = mem.assign(p=mem["bank_id"].map(pri)).groupby("image_id")["p"].min()
    order = R.assign(p=R["image_id"].map(first)).sort_values(["p", "row"])
    done = {}
    for f in sorted(os.listdir(os.path.join(db, "tokens"))):
        z = np.load(os.path.join(db, "tokens", f), allow_pickle=False)
        for i in z["ids"].tolist(): done[i] = f
    todo = [r for _, r in order.iterrows() if r["image_id"] not in done]
    if limit: todo = todo[:limit]
    m = None; buf = {"ids": [], "emb": [], "hid": []}; n_shard = len(os.listdir(os.path.join(db, "tokens"))); failed = []
    def flush():
        nonlocal n_shard
        if not buf["ids"]: return
        np.savez(os.path.join(db, "tokens", f"shard_{n_shard:04d}.npz"), ids=np.array(buf["ids"]), embedding=np.stack(buf["emb"]).astype(np.float16), hidden_state=np.stack(buf["hid"]).astype(np.float16))
        n_shard += 1; buf["ids"].clear(); buf["emb"].clear(); buf["hid"].clear()
    for n, r in enumerate(todo, 1):
        try:
            assert md5(r["path"]) == r["md5"]
            emb, hid = E.encode(r["path"])
            assert emb.shape == hid.shape and np.isfinite(emb).all() and np.isfinite(hid).all()
            if m is None: m = manifest(db, E, emb.shape[0])
            assert emb.shape[0] == m["image_tokens"], f"token count {emb.shape[0]} differs from the index ({m['image_tokens']})"
            buf["ids"].append(r["image_id"]); buf["emb"].append(emb); buf["hid"].append(hid)
        except Exception as ex:
            failed.append({"image_id": r["image_id"], "error": f"{type(ex).__name__}: {str(ex)[:200]}"})
        if len(buf["ids"]) >= SHARD: flush()
        if n % 200 == 0:
            el = time.time() - t_start; print(f"{n}/{len(todo)} encoded, {el / n:.2f} s per image, about {(len(todo) - n) * el / n / 60:.0f} min left", flush=True)
    flush()
    with open(log, "a") as f: f.write(json.dumps({"t": time.strftime("%Y-%m-%d %H:%M:%S"), "encoded": len(todo) - len(failed), "failed": failed, "seconds": round(time.time() - t_start, 1), "already_done": len(done)}) + "\n")
    finalize(db)


def finalize(db):
    """Pooled search vectors in references.csv row order; rows without tokens are marked not indexed."""
    R = pd.read_csv(os.path.join(db, "references.csv"), dtype={"capture_date": str}); pe, ph, have = {}, {}, {}
    for f in sorted(os.listdir(os.path.join(db, "tokens"))):
        z = np.load(os.path.join(db, "tokens", f), allow_pickle=False)
        for j, i in enumerate(z["ids"].tolist()): pe[i] = pool(z["embedding"][j]); ph[i] = pool(z["hidden_state"][j]); have[i] = (f, j)
    R["indexed"] = R["image_id"].isin(set(have)); R["token_shard"] = [have.get(i, ("", -1))[0] for i in R["image_id"]]; R["token_row"] = [have.get(i, ("", -1))[1] for i in R["image_id"]]
    D = len(next(iter(pe.values()))) if pe else 0
    np.save(os.path.join(db, "pooled_embedding.npy"), np.stack([pe.get(i, np.zeros(D, np.float32)) for i in R["image_id"]]).astype(np.float32))
    np.save(os.path.join(db, "pooled_hidden_state.npy"), np.stack([ph.get(i, np.zeros(D, np.float32)) for i in R["image_id"]]).astype(np.float32))
    R.to_csv(os.path.join(db, "references.csv"), index=False); print(f"indexed {int(R['indexed'].sum())} of {len(R)} eligible references")


def queries(db, csv, tag):
    Q = pd.read_csv(csv); E = Encoder(); p = os.path.join(db, f"queries_{tag}.npz"); have = dict(np.load(p, allow_pickle=False)) if os.path.exists(p) else {}
    ids = list(have["ids"]) if "ids" in have else []; pe = list(have["embedding"]) if "embedding" in have else []; ph = list(have["hidden_state"]) if "hidden_state" in have else []
    for n, f in enumerate([q for q in Q["query_id"] if q not in set(ids)], 1):
        e, h = E.encode(ipath(f)); ids.append(f); pe.append(pool(e)); ph.append(pool(h))
        if n % 50 == 0: np.savez(p, ids=np.array(ids), embedding=np.stack(pe), hidden_state=np.stack(ph)); print(f"{n} queries", flush=True)
    np.savez(p, ids=np.array(ids), embedding=np.stack(pe), hidden_state=np.stack(ph)); print(f"{len(ids)} query vectors in {p}")


if __name__ == "__main__":
    cmd, db = sys.argv[1], sys.argv[2]
    if cmd == "inventory": print(json.dumps(inventory(db)[2], indent=1))
    elif cmd == "smoke": smoke(db)
    elif cmd == "build": build(db, int(sys.argv[sys.argv.index("--limit") + 1]) if "--limit" in sys.argv else None)
    elif cmd == "finalize": finalize(db)
    elif cmd == "queries": queries(db, sys.argv[3], sys.argv[4])
