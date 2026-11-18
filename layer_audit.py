"""Layer-by-layer retrieval audit over every AgEval dataset. Local encoder inference only; no hosted API call.

For every split (loader pools with leave-one-out retrieval, plus the existing group-disjoint splits) and every
representation (each block of Google ViT-B/16, DINOv2 ViT-B/14, CLIP ViT-B/16 and ResNet-50, plus the repository's
native baselines), every query retrieves its top four references by cosine similarity. The audit measures how many of
the four references change between representations and whether the replacements carry the query's label.
Usage (from repo/): python layer_audit.py <out_dir> {features|retrieve|all}
Exploratory: nothing here selects a layer or changes the assisted few-shot pipeline.
"""
import glob, gzip, hashlib, itertools, json, os, sys, time
import numpy as np, pandas as pd

out, stage = sys.argv[1], (sys.argv[2] if len(sys.argv) > 2 else "all")
os.makedirs(os.path.join(out, "features"), exist_ok=True)
RUNS = os.path.join("runs"); OFF = os.path.join(RUNS, "offline_layer_stats"); K = 4
CKPT = {"vit": "google/vit-base-patch16-224-in21k", "dinov2": "facebook/dinov2-base", "clip": "openai/clip-vit-base-patch16", "resnet": "microsoft/resnet-50",
        "siglip2": "google/siglip2-base-patch16-224", "bioclip": "imageomics/bioclip", "dinov3": "facebook/dinov3-vitb16-pretrain-lvd1689m", "qwen": "Qwen/Qwen3-VL-2B-Instruct", "flava": "facebook/flava-full"}
QWEN_REV, QWEN_SIDE = "89644892e4d85e24eaac8bacfd4f463576704203", 448
ENC_NAME = {"vit": "Google ViT-B/16", "dinov2": "DINOv2 ViT-B/14", "clip": "CLIP ViT-B/16", "resnet": "ResNet-50", "siglip2": "SigLIP 2 ViT-B/16", "bioclip": "BioCLIP ViT-B/16", "dinov3": "DINOv3 ViT-B/16", "qwen": "Qwen3-VL-2B vision tower", "flava": "FLAVA image encoder ViT-B/16"}
# layer-wise readouts per encoder: (readout names, block indices). SigLIP 2 and the Qwen tower have no CLS token.
LAYERSPEC = {"vit": (("cls", "mean"), range(1, 13)), "dinov2": (("cls", "mean"), range(1, 13)), "clip": (("cls", "mean"), range(1, 13)), "resnet": (("gap",), range(17)),
             "siglip2": (("mean",), range(1, 13)), "bioclip": (("cls", "mean"), range(1, 13)), "dinov3": (("cls", "mean"), range(1, 13)), "qwen": (("mean",), range(1, 25)), "flava": (("cls", "mean"), range(1, 13))}
md5 = lambda p: hashlib.md5(open(p, "rb").read()).hexdigest()
rid = lambda p: "/".join(os.path.normpath(p).split(os.sep)[-2:])       # pool reference id: parent folder + file name (file names repeat across class folders)

# ------------------------------------------------------------------ dataset registry and splits
REGISTRY = [  # every loader in data_loader.py; the first twelve are the AgEval subsets of the README
    ("Durum Wheat", "identification", "kaggle:muratkokludataset/durum-wheat-dataset", True), ("Soybean Seeds", "identification", "kaggle:warcoder/soyabean-seeds", True),
    ("Mango Leaf Disease", "identification", "kaggle:aryashah2k/mango-leaf-disease-dataset", True), ("Bean Leaf Lesions", "identification", "kaggle:marquis03/bean-leaf-lesions-classification", True),
    ("Soybean Diseases", "identification", "zenodo:12747481", True), ("Dangerous Insects", "identification", "kaggle:tarundalal/dangerous-insects-dataset", True),
    ("DeepWeeds", "identification", "kaggle:imsparsh/deepweeds", True), ("Yellow Rust 19", "severity classification", "kaggle:tolgahayit/yellowrust19-yellow-rust-disease-in-wheat", True),
    ("FUSARIUM 22", "severity classification", "kaggle:tolgahayit/fusarium-wilt-disease-in-chickpea-dataset", True), ("IDC", "severity classification", "zenodo:12740714", True),
    ("PlantDoc", "quantification (percent leaf area)", "kaggle:sovitrath/leaf-disease-segmentation-with-trainvalid-split", True), ("InsectCount", "quantification (count)", "zenodo:12747496", True),
    ("SBRD", "severity classification", "kaggle:isaacritharson/severity-based-rice-leaf-diseases-dataset", False), ("IP02", "identification", "kaggle:rtlmhjbn/ip02-dataset", False)]
NUMERIC = {"PlantDoc", "InsectCount"}


def splits():
    """Each split: queries and bank as lists of (id, path, label, md5); eligibility = bank minus self and byte-identical images."""
    S, cov = [], []
    for name, kind, src, ageval in REGISTRY:
        tag = name.replace(" ", "_"); pool_csv = os.path.join(OFF, f"pool_{tag}.csv"); fj = os.path.join(out, f"fetch_{tag}.json")
        rec = {"dataset": name, "task_type": kind, "source_version": src, "in_AgEval_12": ageval, "split": "AgEval loader pool, leave-one-out"}
        if os.path.exists(pool_csv):
            P = pd.read_csv(pool_csv, dtype={"label": str}); items = [(rid(p), p, l, m) for p, l, m in zip(P["path"], P["label"], P["md5"])]
            rec["pool_origin"] = "runs/offline_layer_stats (loader samples=100, seed 42; full dataset on disk)"
        elif os.path.exists(fj):
            J = json.load(open(fj)); items = [(rid(r["path"]), r["path"], r["label"], r["md5"]) for r in J["pool"]]
            rec["pool_origin"] = f"loader samples={J['loader_samples']}, seed 42, drawn in this run from the archive listing; only the pool images are on disk"
        else:
            ff = os.path.join(out, f"fetch_{tag}_FAILED.json")
            rec.update(status="not run", reason=(json.load(open(ff))["error"] if os.path.exists(ff) else "dataset not on disk and not fetched"), unblock="download the archive named in source_version into repo/data and rerun layer_audit_fetch.py")
            cov.append(rec); continue
        assert len({i for i, *_ in items}) == len(items), f"{name}: duplicate ids"
        S.append({"split": f"{name} | loader pool", "dataset": name, "kind": kind, "group": "AgEval loader pool" if ageval else "extra loader pool", "queries": items, "bank": items, "rec": rec})
    dw = lambda i: os.path.join("data", "deepweeds", "images", i)
    A = os.path.join(RUNS, "deepweeds_split_audit"); Q = pd.read_csv(os.path.join(A, "query_panel.csv")); B = pd.read_csv(os.path.join(A, "banks.csv")); B = B[B.bank_seed == 0]
    q = [(i, dw(i), l, m) for i, l, m in zip(Q.query_id, Q.label, Q.md5)]
    for bt, nm in (("DD", "date-disjoint bank"), ("SD", "same-date bank allowed")):
        b = [(i, dw(i), l, None) for i, l in zip(B[B.bank_type == bt].ref_id, B[B.bank_type == bt].label)]
        S.append({"split": f"DeepWeeds | split audit, {nm} (seed 0)", "dataset": "DeepWeeds", "kind": "identification", "group": "group-disjoint split" if bt == "DD" else "same-date control", "queries": q, "bank": b,
                  "rec": {"dataset": "DeepWeeds", "task_type": "identification", "source_version": "kaggle:imsparsh/deepweeds", "in_AgEval_12": True, "split": f"runs/deepweeds_split_audit: 180-query panel, {nm}, bank seed 0 of 20", "pool_origin": "frozen split files"}})
    M = os.path.join(RUNS, "deepweeds_matched_seed42"); D = pd.read_csv(os.path.join(RUNS, "deepweeds_demo_value_seed42", "per_query_results.csv")); QM = pd.read_csv(os.path.join(M, "query_manifest.csv")).set_index("query_id"); RM = pd.read_csv(os.path.join(M, "reference_manifest.csv"))
    S.append({"split": "DeepWeeds | held-out capture dates (paper split, 63 eval + 27 dev)", "dataset": "DeepWeeds", "kind": "identification", "group": "group-disjoint split",
              "queries": [(i, dw(i), l, QM.md5.get(i)) for i, l in zip(D.query_id, D.true_label)], "bank": [(i, dw(i), l, m) for i, l, m in zip(RM.ref_id, RM.label, RM.md5)],
              "rec": {"dataset": "DeepWeeds", "task_type": "identification", "source_version": "kaggle:imsparsh/deepweeds", "in_AgEval_12": True, "split": "runs/deepweeds_demo_value_seed42 queries with the date-disjoint bank of runs/deepweeds_matched_seed42", "pool_origin": "frozen split files"}})
    Pd = os.path.join(RUNS, "paddy_replication_seed42"); PQ = pd.read_csv(os.path.join(Pd, "query_manifest.csv")); PR = pd.read_csv(os.path.join(Pd, "reference_manifest.csv")); pp = lambda i, l: os.path.join("data", "paddy-doctor", "train_images", l, i)
    S.append({"split": "Paddy Doctor | variety-disjoint bank", "dataset": "Paddy Doctor", "kind": "identification", "group": "group-disjoint split",
              "queries": [(i, pp(i, l), l, m) for i, l, m in zip(PQ.query_id, PQ.label, PQ.md5)], "bank": [(i, pp(i, l), l, m) for i, l, m in zip(PR.ref_id, PR.label, PR.md5)],
              "rec": {"dataset": "Paddy Doctor", "task_type": "identification", "source_version": "local data/paddy-doctor (not an AgEval subset; used by the paper's held-out test)", "in_AgEval_12": False, "split": "runs/paddy_replication_seed42: 50 queries from two rice varieties, 110-image bank from eight other varieties", "pool_origin": "frozen split files"}})
    for s in S:   # bank order is fixed by reference id, which is also the tie-break
        s["bank"] = sorted(s["bank"], key=lambda t: t[0]); s["tag"] = hashlib.md5(s["split"].encode()).hexdigest()[:10]
        s["images"] = sorted({t[1] for t in s["queries"] + s["bank"]})
    return S, cov


# ------------------------------------------------------------------ features (one forward pass per image and encoder; compressed cache)
def extract_all(S):
    import torch
    from PIL import Image
    from transformers import AutoFeatureExtractor, AutoImageProcessor, AutoModel, CLIPModel, CLIPProcessor, ResNetModel, ViTModel
    dev = "mps" if torch.backends.mps.is_available() else "cpu"; BS = 16
    l2 = lambda t: torch.nn.functional.normalize(t.float(), dim=-1)
    f16 = lambda t: l2(t).cpu().numpy().astype(np.float16)          # layer-wise readouts: unit length, float16
    f32 = lambda t: t.float().cpu().numpy().astype(np.float32)      # native baselines: raw output, float32
    for enc in CKPT:
        todo = [s for s in S if not os.path.exists(os.path.join(out, "features", f"{enc}__{s['tag']}.npz"))]
        if not todo: continue
        if enc in ("siglip2", "bioclip", "dinov3", "qwen", "flava"): extract_new(enc, todo, dev, BS, f16, f32); continue
        if enc == "vit": proc, model = AutoImageProcessor.from_pretrained(CKPT[enc]), ViTModel.from_pretrained(CKPT[enc])
        elif enc == "dinov2": proc, model = AutoImageProcessor.from_pretrained(CKPT[enc]), AutoModel.from_pretrained(CKPT[enc])
        elif enc == "clip": proc, model = CLIPProcessor.from_pretrained(CKPT[enc]), CLIPModel.from_pretrained(CKPT[enc])
        else: proc, model = AutoFeatureExtractor.from_pretrained(CKPT[enc]), ResNetModel.from_pretrained(CKPT[enc])
        model = model.eval().to(dev); feats = []
        if enc == "resnet":
            hooks = [model.embedder.register_forward_hook(lambda m, i, o: feats.append(o.mean(dim=(2, 3))))]
            hooks += [layer.register_forward_hook(lambda m, i, o: feats.append(o.mean(dim=(2, 3)))) for st in model.encoder.stages for layer in st.layers]
            assert len(hooks) == 17
        for s in todo:
            t0 = time.time(); acc = {}
            def add(k, v): acc.setdefault(k, []).append(v)
            for a in range(0, len(s["images"]), BS):
                ims = [Image.open(p).convert("RGB") for p in s["images"][a:a + BS]]
                x = proc(images=ims, return_tensors="pt")["pixel_values"].to(dev)
                with torch.no_grad():
                    if enc in ("vit", "dinov2"):
                        o = model(pixel_values=x, output_hidden_states=True); hs = torch.stack([model.layernorm(h) for h in o.hidden_states[1:]], dim=1)   # [B, 12, T, D]
                        assert hs.shape[1] == 12 and hs.shape[2] == 1 + (x.shape[-1] // model.config.patch_size) ** 2          # one CLS token, no registers
                        add("cls", f16(hs[:, :, 0])); add("mean", f16(hs[:, :, 1:].mean(dim=2)))
                        add("native", f32(o.pooler_output if enc == "vit" else o.last_hidden_state[:, 0]))
                    elif enc == "clip":
                        o = model.vision_model(pixel_values=x, output_hidden_states=True); hs = torch.stack([model.vision_model.post_layernorm(h) for h in o.hidden_states[1:]], dim=1)
                        assert hs.shape[1] == 12 and hs.shape[2] == 197
                        add("cls", f16(hs[:, :, 0])); add("mean", f16(hs[:, :, 1:].mean(dim=2))); add("native", f32(model.visual_projection(o.pooler_output)))
                    else:
                        feats.clear(); o = model(pixel_values=x); assert len(feats) == 17
                        for b, f in enumerate(feats): add(f"gap{b:02d}", f16(f))
                        add("native", f32(o.pooler_output.flatten(1)))
            arr = {k: np.concatenate(v) for k, v in acc.items()}; assert all(np.isfinite(v.astype(np.float32)).all() for v in arr.values())
            np.savez_compressed(os.path.join(out, "features", f"{enc}__{s['tag']}.npz"), images=np.array(s["images"]), **arr)
            print(f"features {enc:6s} {s['split'][:60]:60s} {len(s['images']):4d} images {time.time() - t0:5.1f}s", flush=True)
        del model


def extract_new(enc, todo, dev, BS, f16, f32):
    """SigLIP 2, BioCLIP, DINOv3, FLAVA and the Qwen3-VL vision tower. Block outputs are captured with forward hooks."""
    import torch
    from PIL import Image
    feats = []; hook = lambda m, i, o: feats.append(o[0] if isinstance(o, tuple) else o)
    if enc == "siglip2":
        from transformers import AutoModel, AutoProcessor
        proc = AutoProcessor.from_pretrained(CKPT[enc]); model = AutoModel.from_pretrained(CKPT[enc]).vision_model.eval().to(dev); [l.register_forward_hook(hook) for l in model.encoder.layers]
        prep = lambda ims: proc(images=ims, return_tensors="pt")["pixel_values"]
        def run(x):
            o = model(pixel_values=x); hs = torch.stack([model.post_layernorm(h) for h in feats], dim=1); assert hs.shape[1] == 12 and hs.shape[2] == 196      # no CLS token
            return {"mean": f16(hs.mean(dim=2)), "native": f32(o.pooler_output)}
    elif enc == "dinov3":
        from transformers import AutoImageProcessor, AutoModel
        proc = AutoImageProcessor.from_pretrained(CKPT[enc]); model = AutoModel.from_pretrained(CKPT[enc]).eval().to(dev); [l.register_forward_hook(hook) for l in model.layer]; nreg = model.config.num_register_tokens
        prep = lambda ims: proc(images=ims, return_tensors="pt")["pixel_values"]
        def run(x):
            o = model(pixel_values=x); hs = torch.stack([model.norm(h) for h in feats], dim=1); assert hs.shape[1] == 12 and hs.shape[2] == 1 + nreg + 196
            return {"cls": f16(hs[:, :, 0]), "mean": f16(hs[:, :, 1 + nreg:].mean(dim=2)), "native": f32(o.pooler_output)}                                 # register tokens excluded from the patch mean
    elif enc == "flava":   # image encoder of FLAVA; the text and multimodal encoders are not used
        from transformers import FlavaImageProcessor, FlavaModel
        proc = FlavaImageProcessor.from_pretrained(CKPT[enc]); full = FlavaModel.from_pretrained(CKPT[enc]).eval().to(dev); model = full.image_model; [l.register_forward_hook(hook) for l in model.encoder.layer]
        prep = lambda ims: proc(images=ims, return_tensors="pt")["pixel_values"]
        def run(x):
            o = model(pixel_values=x); hs = torch.stack([model.layernorm(h) for h in feats], dim=1); assert hs.shape[1] == 12 and hs.shape[2] == 197
            return {"cls": f16(hs[:, :, 0]), "mean": f16(hs[:, :, 1:].mean(dim=2)), "native": f32(full.image_projection(o.last_hidden_state[:, 0]))}
    elif enc == "bioclip":
        import open_clip
        full, _, pre = open_clip.create_model_and_transforms("hf-hub:" + CKPT[enc]); full = full.eval().to(dev); v = full.visual; [b.register_forward_hook(hook) for b in v.transformer.resblocks]
        prep = lambda ims: torch.stack([pre(im) for im in ims])
        def run(x):
            e = full.encode_image(x); hs = torch.stack([v.ln_post(h) for h in feats], dim=1); assert hs.shape[1] == 12 and hs.shape[2] == 197
            return {"cls": f16(hs[:, :, 0]), "mean": f16(hs[:, :, 1:].mean(dim=2)), "native": f32(e)}
    else:   # Qwen3-VL vision tower only (the language model is not loaded); one image per pass at a fixed 448 x 448 input
        import transformers.models.qwen3_vl.modeling_qwen3_vl as Q
        from transformers import AutoConfig, AutoProcessor
        from huggingface_hub import hf_hub_download
        from safetensors import safe_open
        cfg = AutoConfig.from_pretrained(CKPT[enc], revision=QWEN_REV).vision_config; model = Q.Qwen3VLVisionModel(cfg)
        f = safe_open(hf_hub_download(CKPT[enc], "model.safetensors", revision=QWEN_REV), "pt"); pref = "model.visual."
        sd = {k[len(pref):]: f.get_tensor(k) for k in f.keys() if k.startswith(pref)}; miss = model.load_state_dict(sd, strict=True); model = model.float().eval().to(dev); [b.register_forward_hook(hook) for b in model.blocks]
        ip = AutoProcessor.from_pretrained(CKPT[enc], revision=QWEN_REV).image_processor; BS = 1
        prep = lambda ims: ip(images=[im.resize((QWEN_SIDE, QWEN_SIDE), Image.BICUBIC) for im in ims], return_tensors="pt")
        def run(x):
            merged, _ = model(x["pixel_values"].to(dev).float(), grid_thw=x["image_grid_thw"].to(dev)); hs = torch.stack(feats, dim=0); assert hs.shape[0] == 24 and hs.shape[1] == (QWEN_SIDE // 16) ** 2
            return {"mean": f16(hs.mean(dim=1)[None]), "native": f32(merged.mean(dim=0)[None])}
    for s in todo:
        t0 = time.time(); acc = {}
        for a in range(0, len(s["images"]), BS):
            x = prep([Image.open(p).convert("RGB") for p in s["images"][a:a + BS]]); x = x.to(dev) if enc != "qwen" else x; feats.clear()
            with torch.no_grad(): r = run(x)
            for k, v_ in r.items(): acc.setdefault(k, []).append(v_)
        arr = {k: np.concatenate(v_) for k, v_ in acc.items()}; assert all(np.isfinite(v_.astype(np.float32)).all() for v_ in arr.values())
        np.savez_compressed(os.path.join(out, "features", f"{enc}__{s['tag']}.npz"), images=np.array(s["images"]), **arr)
        print(f"features {enc:7s} {s['split'][:60]:60s} {len(s['images']):4d} images {time.time() - t0:5.1f}s", flush=True)


def reps():
    """Ordered list of representations: (rep_id, encoder, readout, block, family). family groups a depth series."""
    R = []
    for enc, (ros, blocks) in LAYERSPEC.items():
        for ro in ros: R += [(f"{enc}|{ro}|b{b:02d}", enc, ro, b, f"{enc}|{ro}") for b in blocks]
    R += [(f"{enc}|native", enc, "native", 99, None) for enc in LAYERSPEC]
    return R
NATIVE_DESC = {"vit|native": "ViTModel.pooler_output = tanh(Linear(LayerNorm(h12)[CLS])), 768-d (Arshad et al. retriever)", "dinov2|native": "last_hidden_state[:, 0] = LayerNorm(h12)[CLS], 768-d",
               "clip|native": "CLIPModel.get_image_features = visual_projection(post_layernorm(h12[CLS])), 512-d", "resnet|native": "ResNetModel.pooler_output = global average pool of the last stage, 2048-d",
               "siglip2|native": "SiglipVisionModel.pooler_output = attention-pooling head over the patch tokens, 768-d (the image embedding of get_image_features)",
               "bioclip|native": "open_clip encode_image = ln_post(h12[CLS]) projected to 512-d", "dinov3|native": "pooler_output = norm(h12)[CLS], 768-d",
               "qwen|native": "mean of the merged visual tokens that enter the language model (2 x 2 patch merger, 2048-d)",
               "flava|native": "image_projection(layernorm(h12)[CLS]), 768-d (the image embedding of the contrastive head)"}


def load_rep(F, rid):
    enc, ro = rid.split("|")[:2]
    if ro == "native": X = F[enc]["native"]
    elif enc == "resnet": X = F[enc][f"gap{int(rid[-2:]):02d}"]
    else: X = F[enc][ro][:, int(rid[-2:]) - 1]
    X = X.astype(np.float64); return X / np.linalg.norm(X, axis=1, keepdims=True)


def retrieve(S):
    R = reps(); rank_rows, pair_rows, big, dims = [], [], [], {}
    for s in S:
        F = {enc: np.load(os.path.join(out, "features", f"{enc}__{s['tag']}.npz")) for enc in CKPT}
        pos = {p: i for i, p in enumerate(s["images"])}; assert all(list(F[e]["images"]) == s["images"] for e in F)
        qi = np.array([pos[t[1]] for t in s["queries"]]); bi = np.array([pos[t[1]] for t in s["bank"]])
        qid = [t[0] for t in s["queries"]]; bid = np.array([t[0] for t in s["bank"]]); ql = np.array([str(t[2]) for t in s["queries"]]); bl = np.array([str(t[2]) for t in s["bank"]])
        m5 = {}
        def h(t):
            if t[3] is None or (isinstance(t[3], float) and np.isnan(t[3])):
                if t[1] not in m5: m5[t[1]] = md5(t[1])
                return m5[t[1]]
            return t[3]
        qm = np.array([h(t) for t in s["queries"]]); bm = np.array([h(t) for t in s["bank"]])
        inel = (np.array(qid)[:, None] == bid[None, :]) | (qm[:, None] == bm[None, :])       # self-match and byte-identical images
        elig = (~inel).sum(1); full = elig >= K; has5 = elig >= K + 1
        numeric = s["dataset"] in NUMERIC; possible = ((ql[:, None] == bl[None, :]) & ~inel).sum(1)
        labels_ok = (not numeric) and possible.max() > 0
        s["stats"] = {"n_queries": len(qid), "n_bank": len(bid), "eligible_refs_min": int(elig.min()), "eligible_refs_median": float(np.median(elig)), "queries_with_fewer_than_4_eligible": int((~full).sum()),
                      "excluded_self_or_duplicate_pairs": int(inel.sum()), "byte_identical_non_self_pairs": int(inel.sum() - (np.array(qid)[:, None] == bid[None, :]).sum()),
                      "n_labels": int(len(set(ql) | set(bl))), "queries_with_a_same_label_reference": int((possible > 0).sum()), "same_label_metrics": bool(labels_ok),
                      "same_label_note": ("numeric target, no class label shared by query and reference" if numeric else ("no query has an eligible reference with its label" if not labels_ok else "class labels"))}
        T = {}
        for rid, enc, ro, b, fam in R:
            X = load_rep(F, rid); dims[rid] = X.shape[1]; Sim = X[qi] @ X[bi].T; Sim[inel] = -np.inf
            order = np.argsort(-Sim, kind="stable")[:, :K + 1]; sc = np.take_along_axis(Sim, order, 1)       # stable sort on a bank ordered by reference id = id tie-break
            top = order[:, :K]; same = (bl[top] == ql[:, None]); gap = np.where(has5, sc[:, K - 1] - sc[:, K], np.nan)
            T[rid] = {"top": top, "same4": same.sum(1), "top1": same[:, 0], "gap": gap, "sc": sc}
            for q in range(len(qid)):
                rank_rows.append((s["split"], qid[q], ql[q], rid, enc, ro, b, "|".join(bid[top[q]]), "|".join(bl[top[q]]), "|".join(f"{v:.5f}" for v in sc[q, :K]),
                                  "" if np.isnan(gap[q]) else f"{gap[q]:.5f}", int(same[q].sum()) if labels_ok else "", int(same[q, 0]) if labels_ok else "", int(full[q])))
        def compare(a, b, ctype, fam):
            A, B = T[a], T[b]; shared = (A["top"][:, :, None] == B["top"][:, None, :]).any(2).sum(1); turn = (K - shared)[full]; n = int(full.sum())
            reord = ((shared == K) & (A["top"] != B["top"]).any(1))[full]; d = (B["same4"].astype(int) - A["same4"].astype(int))[full]; ch = turn > 0
            row = {"split": s["split"], "dataset": s["dataset"], "group": s["group"], "comparison": ctype, "family": fam or "", "rep_a": a, "rep_b": b, "block_a": int(a[-2:]) if a[-1].isdigit() else "", "block_b": int(b[-2:]) if b[-1].isdigit() else "",
                   "n_queries": n, "mean_turnover": float(turn.mean()), **{f"pct_turnover_{k}": float(100 * (turn == k).mean()) for k in range(K + 1)},
                   "pct_top1_changed": float(100 * (A["top"][:, 0] != B["top"][:, 0])[full].mean()), "pct_same_four_reordered": float(100 * reord.mean())}
            if labels_ok:
                row.update(top1_correct_a=float(100 * A["top1"][full].mean()), top1_correct_b=float(100 * B["top1"][full].mean()), same_label_of4_a=float(A["same4"][full].mean()), same_label_of4_b=float(B["same4"][full].mean()),
                           n_improved=int((d > 0).sum()), n_same=int((d == 0).sum()), n_declined=int((d < 0).sum()), n_changed_queries=int(ch.sum()),
                           n_improved_when_changed=int((d[ch] > 0).sum()), n_same_when_changed=int((d[ch] == 0).sum()), n_declined_when_changed=int((d[ch] < 0).sum()),
                           mean_same_label_delta_when_changed=float(d[ch].mean()) if ch.any() else np.nan)
            pair_rows.append(row); return shared
        fams = {}
        for rid, enc, ro, b, fam in R:
            if fam: fams.setdefault(fam, []).append(rid)
        for fam, ids in fams.items():
            for a, b in itertools.combinations(ids, 2):
                shared = compare(a, b, "adjacent blocks" if int(b[-2:]) - int(a[-2:]) == 1 else "block pair", fam)
                if int(b[-2:]) - int(a[-2:]) == 1:            # candidates for the largest-change inspection
                    for q in np.argsort(shared, kind="stable")[:2]:
                        if full[q] and K - shared[q] >= 3:
                            A, B = T[a], T[b]
                            big.append({"split": s["split"], "family": fam, "rep_a": a, "rep_b": b, "query": qid[q], "label": ql[q], "turnover": int(K - shared[q]),
                                        "refs_a": list(bid[A["top"][q]]), "labels_a": list(bl[A["top"][q]]), "scores_a": [round(float(v), 4) for v in A["sc"][q]],
                                        "refs_b": list(bid[B["top"][q]]), "labels_b": list(bl[B["top"][q]]), "scores_b": [round(float(v), 4) for v in B["sc"][q]],
                                        "gap45_a": None if np.isnan(A["gap"][q]) else round(float(A["gap"][q]), 4), "gap45_b": None if np.isnan(B["gap"][q]) else round(float(B["gap"][q]), 4),
                                        "labels_comparable": bool(labels_ok), "retrieved_label_set_changed": bool(sorted(bl[A["top"][q]]) != sorted(bl[B["top"][q]]))})
        for rid, enc, ro, b, fam in R:
            if fam: compare(rid, f"{enc}|native", "block versus native baseline", fam)
            if rid != "vit|native": compare("vit|native", rid, "Arshad pooled ViT versus representation", fam or "native")
        # turnover against the 4th-5th score gap and the retrieved classes, adjacent blocks only
        gt = []
        for fam, ids in fams.items():
            for a, b in zip(ids[:-1], ids[1:]):
                A, B = T[a], T[b]; shared = (A["top"][:, :, None] == B["top"][:, None, :]).any(2).sum(1)
                for q in np.where(full & has5)[0]:
                    gt.append((fam, int(K - shared[q]), A["gap"][q], bool(sorted(bl[A["top"][q]]) != sorted(bl[B["top"][q]])) if labels_ok else None))
        s["gap_turnover"] = pd.DataFrame(gt, columns=["family", "turnover", "gap45_before", "label_set_changed"])
        print("retrieved", s["split"], flush=True)
    RK = pd.DataFrame(rank_rows, columns=["split", "query_id", "query_label", "representation", "encoder", "readout", "block", "top4_ref_ids", "top4_ref_labels", "top4_scores", "gap_4th_5th", "same_label_of4", "top1_same_label", "full_top4"])
    RK.to_csv(os.path.join(out, "per_query_rankings.csv.gz"), index=False, compression="gzip")
    PM = pd.DataFrame(pair_rows); PM.to_csv(os.path.join(out, "layer_pair_metrics.csv"), index=False)
    json.dump(big, open(os.path.join(out, "largest_changes.json"), "w"), indent=1)
    pd.concat([g["gap_turnover"].assign(split=g["split"]) for g in S]).to_csv(os.path.join(out, "gap_vs_turnover_adjacent.csv.gz"), index=False, compression="gzip")
    json.dump({"dims": dims, "splits": {s["split"]: s["stats"] for s in S}}, open(os.path.join(out, "retrieval_meta.json"), "w"), indent=1)


if __name__ == "__main__":
    S, cov = splits()
    print(len(S), "splits;", len(cov), "datasets not run;", len({p for s in S for p in s["images"]}), "distinct images", flush=True)
    if stage in ("features", "all"): extract_all(S)
    if stage in ("retrieve", "all"): retrieve(S)
    json.dump({"splits": [{**s["rec"], "split_name": s["split"], "group": s["group"], "tag": s["tag"], "n_images": len(s["images"]), **s.get("stats", {})} for s in S], "not_run": cov}, open(os.path.join(out, "coverage_manifest.json"), "w"), indent=1)
