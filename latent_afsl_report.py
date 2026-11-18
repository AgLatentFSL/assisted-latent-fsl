"""Builds runs/latent_afsl/SUMMARY.md, the transition tables and the paired-neighborhood figure. No model calls.
Usage: python latent_afsl_report.py <run_dir>"""
import json, os, sys
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.metrics import f1_score
from PIL import Image, ImageDraw
from pilot_matched_labels import path_of

out = sys.argv[1]; NB = 10000; rng = np.random.default_rng(42)
M = json.load(open(os.path.join(out, "manifest.json"))); TM = json.load(open(os.path.join(out, "test_meta.json"))); AM = json.load(open(os.path.join(out, "ageval_meta.json")))
G = pd.read_csv(os.path.join(out, "test_retrieval.csv")); Dv = pd.read_csv(os.path.join(out, "dev_retrieval.csv")); Cn = pd.read_csv(os.path.join(out, "test_contrasts.csv"))
OV = pd.read_csv(os.path.join(out, "base_bank_overlap.csv")); NBR = pd.read_csv(os.path.join(out, "base_bank_neighbors.csv")); AG = pd.read_csv(os.path.join(out, "ageval_retrieval.csv"))
V = pd.read_csv(os.path.join(out, "vlm_per_query.csv"), keep_default_na=False) if os.path.exists(os.path.join(out, "vlm_per_query.csv")) else pd.DataFrame()
led = [json.loads(l) for l in open(os.path.join(out, "vlm", "request_ledger.jsonl"))] if os.path.exists(os.path.join(out, "vlm", "request_ledger.jsonl")) else []
classes = sorted(NBR["true_label"].unique())
def fp(p): return "<0.001" if p < 0.001 else f"{p:.3f}"
def pc(x): return f"{100 * x:.1f}%"
H = M["selected_vit_name"]; L = []
L += ["# Latent-Assisted Few-Shot Learning: does the layer and token readout of a frozen encoder change the demonstrations?\n",
      "**Connection to prior work.** Assisted few-shot learning established image-similarity retrieval for selecting visual demonstrations. We test whether changing the layer and token readout within a frozen vision encoder changes the utility of those demonstrations.\n",
      "## 1. Verified baseline and provenance\n",
      "| Item | Arshad et al. (paper PDF and `adaptive_few_shots` at `cb2046c`) | This experiment |", "|---|---|---|",
      "| ViT checkpoint | `google/vit-base-patch16-224-in21k`, `ViTModel` (paper Table 2) | same |",
      "| ViT feature | not stated in the paper; `outputs.pooler_output` in `get_embeddingp.py` | same for the baseline A |",
      "| Pooler | `tanh(Linear(LayerNorm(h12)[CLS]))` | B = `LayerNorm(h12)[CLS]` (pooler input); raw `h12[CLS]` also scored |",
      "| Preprocessing | `AutoImageProcessor` of the checkpoint | same; one forward pass gives every readout |",
      "| Similarity, ranking | cosine, top-k, the query excluded; stable sort | same; ties to the lower bank index |",
      "| Demonstration order | descending similarity | same |",
      "| Shots | 0, 1, 2, 4, 8; headline at 8 shots | 4 shots (difference from the published setup) |",
      "| VLM | GPT-4o | `claude-sonnet-5-5` through headless `claude -p` (difference) |",
      "| Tasks, metric | seven AgEval tasks, F1 | six tasks (Durum Wheat omitted), accuracy and macro-F1 (difference) |",
      "| Published headline | average F1 68.68% (random) to 80.45% (ViT) at 8 shots | not reproduced at 8 shots |",
      "\n| Draft number | Raw-file value | Source | Status |", "|---|---|---|---|",
      "| AgEval, 583 queries: ViT pooled 4-NN 73.1%, standard VLM 65.7% | 73.1%, 65.7% | `arbab_eval_seed42/per_query_results.csv` | verified |",
      "| DINOv2 final CLS: 4-NN 76.2%, standard VLM 64.5%, vote-aware 73.1% | 76.2%, 64.5%, 73.1% | same file | verified |",
      f"| Recomputed votes from independently extracted features | {AM['check_against_cached_votes']['A_vote_equals_cached_vote_vit']}/583 ViT and {AM['check_against_cached_votes']['DINO_vote_equals_cached_vote_dinov2']}/583 DINOv2 votes identical | `latent_afsl/ageval_meta.json` | verified |",
      "| Figure 4: ViT block-9 patch mean 65.8%, pooled 63.6% | 65.8%, 63.6% | `offline_layer_stats/layer_token_sweep/layer_token_sweep.csv` | verified; **eight pools = the six main tasks plus Yellow Rust 19 and IDC**. On the six main pools the values are 74.9% and 73.2% |",
      "| DeepWeeds 63 evaluation and 27 development queries | roles match | `deepweeds_demo_value_seed42`, `deepweeds_vote_aware_seed42` | verified; before this run the 63 had zero-shot and DINOv2 standard-prompt answers only, and the 27 had the vote-aware candidates |",
      "| 123-representation sweep | 123 candidates; `selection.json` written before `eval_pools.csv` | `latent_search` | verified; the earlier layer sweep was scored on the evaluation pools, so its per-block ranking is exploratory |",
      f"| Fresh features versus cached vectors | minimum cosine {TM['verification']['vit_pooled_fresh_vs_audit_cache_min_cosine_300']:.6f} (ViT pooled) and {TM['verification']['dinov2_cls12_fresh_vs_audit_cache_min_cosine_300']:.6f} (DINOv2) on 300 images | `latent_afsl/test_meta.json` | verified |",
      f"| Reproduced neighbor IDs | ViT pooled top 4 identical on {TM['verification']['vit_pooled_top4_matches_base_run_20_queries'][0]}/20 base-run queries; DINOv2 top 4 identical on {TM['verification']['dinov2_cls12_top4_matches_demo_value_run_90_queries'][0]}/90 earlier queries | same | verified |",
      "| Arshad et al. published numbers | read from the PDF; not rerun | `Downloads/137_Assisted_Few_Shot_Learning.pdf` | not reproduced (different VLM and shot count) |\n"]
L += ["## 2. Frozen manifest and development selection\n",
      f"`manifest.json` was written at {M['written_at']}, before any test metric. Rule: {M['selection_rule']}. Candidates: 24 ViT readouts (CLS and patch mean at blocks 1 to 12) and 24 DINOv2 readouts.\n",
      f"- **Development split:** {M['development_split']['n_queries']} DeepWeeds queries on {len(M['development_split']['query_dates'])} dates and {M['development_split']['banks']} banks of {M['development_split']['bank_size']} images on {len(M['development_split']['bank_dates'])} other dates. No development image is a test query.",
      f"- **Selected ViT readout: {H}** (development macro-F1 {M['selected_vit_dev']['macro_f1']:.3f}, accuracy {pc(M['selected_vit_dev']['acc'])}) against the pooled output (macro-F1 {M['pooled_dev']['macro_f1']:.3f}, accuracy {pc(M['pooled_dev']['acc'])}).",
      f"- **DINOv2:** the development winner is {M['selected_dinov2_name']} (macro-F1 {M['selected_dinov2_dev']['macro_f1']:.3f}), which is the conventional readout. No alternative beat it.",
      f"- **Test split:** {TM['n_test_queries']} queries on {len(TM['test_dates'])} capture dates (63 answered before with DINOv2 retrieval, 90 never sent to a VLM); class counts {TM['test_class_counts']}.\n"]
key = [("A", "ViT", "pooler_output", "12 + pooler"), ("Braw", "ViT", "CLS, before final LayerNorm", "12"), ("B", "ViT", "CLS", "12"), ("C", "ViT", "CLS", str(TM["derived_readouts"]["C"][2])),
       ("H", "ViT", "patch mean (selected)", str(TM["derived_readouts"]["H"][2])), ("E", "ViT", "patch mean", str(TM["derived_readouts"]["E"][2])), ("DINO", "DINOv2", "CLS (conventional, selected)", "12")]
rn = NBR.drop_duplicates("readout_key").set_index("readout_key")["readout"]
def row(g, name): return g[g.readout == name].iloc[0]
L += ["## 3. Retrieval table (held-out DeepWeeds queries; 4-NN vote over 20 banks of 99 images)\n",
      "| Checkpoint | Token readout | Block | Dev macro-F1 | Date-disjoint acc. | macro-F1 | Top-1 same class | Same-class frac. of top 4 | Same-date-allowed acc. | Gain from same-date banks | Top-4 share from the query's date |", "|---|---|---|---|---|---|---|---|---|---|---|"]
for k, ck, tok, blk in key:
    d, s, dv = row(G[G.bank_type == "DD"], rn[k]), row(G[G.bank_type == "SD"], rn[k]), row(Dv, rn[k])
    L.append(f"| {ck} | {tok} | {blk} | {dv['macro_f1']:.3f} | {pc(d['acc'])} | {d['macro_f1']:.3f} | {pc(d['top1'])} | {pc(d['same_frac'])} | {pc(s['acc'])} | {100 * (s['acc'] - d['acc']):+.1f} pp | {pc(s['top4_same_date_frac'])} |")
L += ["\nThe full grid (50 readouts, both bank types) is in `test_retrieval.csv`; rows other than the ones above were not selected on development data and are descriptive.\n",
      "### Matched contrasts on date-disjoint banks (paired by query, 153 queries)\n",
      "| Contrast | Question | 4-NN accuracy diff (95% CI) | CI clustered by capture date | Wilcoxon p | Dates up / down | Same-class frac. diff |", "|---|---|---|---|---|---|---|"]
for c in Cn[(Cn.bank_type == "DD") & (Cn.metric == "acc")]["contrast"]:
    a = Cn[(Cn.bank_type == "DD") & (Cn.metric == "acc") & (Cn.contrast == c)].iloc[0]; s = Cn[(Cn.bank_type == "DD") & (Cn.metric == "same") & (Cn.contrast == c)].iloc[0]
    L.append(f"| {c} | {a['question']} | {100 * a['diff']:+.1f} pp ({100 * a['ci_lo']:+.1f}, {100 * a['ci_hi']:+.1f}) | ({100 * a['date_cluster_ci_lo']:+.1f}, {100 * a['date_cluster_ci_hi']:+.1f}) | {fp(a['wilcoxon_p'])} | {a['dates_up']} / {a['dates_down']} | {100 * s['diff']:+.1f} pp |")
L += ["\n### Historical AgEval pools (no source-group split, 583 queries)\n", "| Readout | 4-NN accuracy |", "|---|---|"]
L += [f"| {AM['readouts'][k]} | {pc(v)} |" for k, v in AM["pooled_vote_accuracy_583"].items() if k in ("A", "B", "C", "H", "E", "DINO")]
L.append(f"\nSelected hidden state versus pooled output: {AM['H_vs_A']['wins']} wins, {AM['H_vs_A']['losses']} losses (p {fp(AM['H_vs_A']['p'])}). Final CLS versus pooled output: {AM['B_vs_A']['wins']} wins, {AM['B_vs_A']['losses']} losses (p {fp(AM['B_vs_A']['p'])}). DINOv2 final CLS versus ViT pooled: {AM['DINO_vs_A']['wins']} wins, {AM['DINO_vs_A']['losses']} losses (p {fp(AM['DINO_vs_A']['p'])}).\n")
L += ["### How much each edit changes the neighborhood (frozen base bank, 153 queries)\n", "| Pair | Shared references of 4 | Identical ordered top 4 | Same label sequence | Votes fixed | Votes broken |", "|---|---|---|---|---|---|"]
L += [f"| {w['y']} vs {w['x']} | {w['mean_shared_of_4']:.2f} | {w['identical_ordered_top4']}/{w['n']} | {w['same_label_sequence']}/{w['n']} | {w['vote_fixed']} | {w['vote_broken']} |" for _, w in OV.iterrows() if w["pair"] in ("H vs A", "B vs A", "C vs B", "E vs D", "D vs B", "A vs DINO", "DINO vs A")]
bc = pd.read_csv(os.path.join(out, "test_H_minus_A_by_class.csv")); bd = pd.read_csv(os.path.join(out, "test_H_minus_A_by_date.csv"), dtype={"date": str})
L += ["\n### Selected hidden state minus pooled output by class (date-disjoint banks)\n", "| Class | n | Pooled | Selected | Diff |", "|---|---|---|---|---|"]
L += [f"| {w['label']} | {w['n']} | {pc(w['pooled_acc'])} | {pc(w['selected_acc'])} | {100 * w['diff']:+.1f} pp |" for _, w in bc.iterrows()]
if len(V):
    DW = V[V["set"] == "deepweeds"]; W = DW.pivot(index="query_id", columns="cond", values="raw_pred"); meta = DW.drop_duplicates("query_id").set_index("query_id")
    conds = [c for c in ("A_S", "H_S", "A_V", "H_V", "DINO_S", "DINO_V", "A_L", "H_L") if c in W.columns and W[c].notna().all()]; t = meta.loc[W.index, "true_label"]; grp = meta.loc[W.index, "group"].astype(str); role = meta.loc[W.index, "role"]
    ok = {c: (W[c].astype(str) == t) for c in conds}; vote = {k: DW[DW.cond == k + "_S"].set_index("query_id").loc[W.index, "vote_correct"].astype(bool) for k in ("A", "H", "DINO") if (DW.cond == k + "_S").any()}
    NAME = {"A_S": "ViT pooled + standard prompt (Arshad et al.)", "H_S": f"{H} + standard prompt", "A_V": "ViT pooled + vote-aware prompt", "H_V": f"{H} + vote-aware prompt", "A_L": "ViT pooled, labels only", "H_L": f"{H}, labels only", "DINO_S": "DINOv2 final CLS + standard prompt", "DINO_V": "DINOv2 final CLS + vote-aware prompt"}
    def paired(x, y, mask=None):
        a, b = ok[x], ok[y]
        if mask is not None: a, b = a[mask], b[mask]
        w, l = int((b & ~a).sum()), int((~b & a).sum()); d = (b.astype(int) - a.astype(int)).to_numpy(); g = grp[mask] if mask is not None else grp
        m = d[rng.integers(0, len(d), size=(NB, len(d)))].mean(axis=1); s = pd.Series(d).groupby(g.to_numpy()); su, ns = s.sum().to_numpy(), s.size().to_numpy(); ix = rng.integers(0, len(su), size=(NB, len(su))); cm = su[ix].sum(axis=1) / ns[ix].sum(axis=1)
        gm = s.mean()
        return {"n": len(d), "x": int(a.sum()), "y": int(b.sum()), "wins": w, "losses": l, "diff": float(d.mean()), "ci": (float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5))),
                "cci": (float(np.percentile(cm, 2.5)), float(np.percentile(cm, 97.5))), "p": 1.0 if w + l == 0 else float(stats.binomtest(w, w + l, 0.5).pvalue), "groups_up": int((gm > 0).sum()), "groups_down": int((gm < 0).sum())}
    L += ["\n## 4. Paired VLM result (DeepWeeds, 153 held-out queries, frozen date-disjoint bank, four shots)\n",
          f"VLM `claude-sonnet-5-5`; requests sent in this run: {len(led)}, failed: {sum(not r['ok'] for r in led)}; response model IDs: {sorted({m for r in led for m in (r.get('model_ids') or [])})}. Every prompt has the same template, class list and query bytes; the four retrieved references are the variable.\n",
          "| Condition | Vote of the references | VLM accuracy | VLM macro-F1 |", "|---|---|---|---|"]
    for c in conds:
        vk = vote[c.rsplit("_", 1)[0]]; L.append(f"| {NAME[c]} | {int(vk.sum())}/{len(vk)} ({pc(vk.mean())}) | {int(ok[c].sum())}/{len(t)} ({pc(ok[c].mean())}) | {f1_score(t, W[c].astype(str), labels=classes, average='macro', zero_division=0):.3f} |")
    if {"A_S", "H_S"} <= set(conds):
        p = paired("A_S", "H_S"); pf = paired("A_S", "H_S", (role == "audit_only").to_numpy()); pe = paired("A_S", "H_S", (role == "eval").to_numpy())
        L += [f"\n**Primary comparison (selected hidden state versus pooled output, standard prompt):** {p['y']}/{p['n']} against {p['x']}/{p['n']}, difference {100 * p['diff']:+.1f} pp "
              f"(95% CI {100 * p['ci'][0]:+.1f} to {100 * p['ci'][1]:+.1f}; clustered by capture date {100 * p['cci'][0]:+.1f} to {100 * p['cci'][1]:+.1f}); {p['wins']} wins, {p['losses']} losses, exact McNemar p {fp(p['p'])}. "
              f"The difference is positive on {p['groups_up']} capture dates and negative on {p['groups_down']}. On the 90 queries never sent to a VLM before: {pf['y']}/90 against {pf['x']}/90 ({pf['wins']} wins, {pf['losses']} losses, p {fp(pf['p'])}). "
              f"On the 63 earlier evaluation queries: {pe['y']}/63 against {pe['x']}/63 ({pe['wins']} wins, {pe['losses']} losses).\n"]
        nA, nH = NBR[NBR.readout_key == "A"].set_index("query_id").loc[W.index], NBR[NBR.readout_key == "H"].set_index("query_id").loc[W.index]
        ident = (nA["neighbor_ids"] == nH["neighbor_ids"]).to_numpy(); va, vh = vote["A"].to_numpy(), vote["H"].to_numpy(); samelab = (nA["neighbor_labels"] == nH["neighbor_labels"]).to_numpy()
        cat = np.where(ident, "identical examples", np.where(~va & vh, "latent fixes the vote", np.where(va & ~vh, "latent breaks the vote", "vote outcome unchanged, examples changed")))
        tr = pd.DataFrame({"query_id": W.index, "group": grp.to_numpy(), "role": role.to_numpy(), "true_label": t.to_numpy(), "category": cat, "same_label_sequence": samelab, "vote_A": nA["vote"].to_numpy(), "vote_H": nH["vote"].to_numpy(),
                           **{f"pred_{c}": W[c].to_numpy() for c in conds}, **{f"correct_{c}": ok[c].to_numpy() for c in conds}})
        tr.to_csv(os.path.join(out, "transitions_per_query.csv"), index=False)
        L += ["### Where the standard-prompt outcome changes (categories depend on retrieval; the all-query result above is primary)\n", "| Category | Queries | VLM correct with pooled | VLM correct with selected | Selected wins | Selected losses |", "|---|---|---|---|---|---|"]
        for cname in ["identical examples", "latent fixes the vote", "latent breaks the vote", "vote outcome unchanged, examples changed"]:
            g = tr[tr.category == cname]; L.append(f"| {cname} | {len(g)} | {int(g.correct_A_S.sum())} | {int(g.correct_H_S.sum())} | {int((g.correct_H_S & ~g.correct_A_S).sum())} | {int((~g.correct_H_S & g.correct_A_S).sum())} |")
        g = tr[tr.same_label_sequence & (tr.category != "identical examples")]
        L.append(f"\nSame ordered label sequence with different images: {len(g)} queries; the VLM is correct on {int(g.correct_A_S.sum())} with the pooled images and {int(g.correct_H_S.sum())} with the selected images ({int((g.correct_H_S & ~g.correct_A_S).sum())} wins, {int((~g.correct_H_S & g.correct_A_S).sum())} losses).\n")
    if {"A_S", "H_S", "A_V", "H_V"} <= set(conds):
        dd = (ok["H_V"].astype(int) - ok["A_V"].astype(int)) - (ok["H_S"].astype(int) - ok["A_S"].astype(int)); m = dd.to_numpy()[rng.integers(0, len(dd), size=(NB, len(dd)))].mean(axis=1)
        pv, pa, ph = paired("A_V", "H_V"), paired("A_S", "A_V"), paired("H_S", "H_V")
        L += ["## 5. Readout by prompt (2 x 2)\n", "| | Standard prompt | Vote-aware prompt | Prompt effect (wins / losses, p) |", "|---|---|---|---|",
              f"| ViT pooled output | {int(ok['A_S'].sum())}/153 | {int(ok['A_V'].sum())}/153 | {pa['wins']} / {pa['losses']}, p {fp(pa['p'])} |", f"| {H} | {int(ok['H_S'].sum())}/153 | {int(ok['H_V'].sum())}/153 | {ph['wins']} / {ph['losses']}, p {fp(ph['p'])} |",
              f"| Readout effect (wins / losses, p) | {paired('A_S', 'H_S')['wins']} / {paired('A_S', 'H_S')['losses']}, p {fp(paired('A_S', 'H_S')['p'])} | {pv['wins']} / {pv['losses']}, p {fp(pv['p'])} | interaction {100 * dd.mean():+.1f} pp (95% CI {100 * np.percentile(m, 2.5):+.1f} to {100 * np.percentile(m, 97.5):+.1f}) |",
              "\n### Vote-aware prompting: harmful overrides and beneficial corrections (same references)\n", "| References | Prompt | Correct votes overturned | Wrong votes corrected | Same label as the vote | Standard wrong, vote-aware right | Standard right, vote-aware wrong |", "|---|---|---|---|---|---|---|"]
        for rk, nm in (("A", "ViT pooled"), ("H", H), ("DINO", "DINOv2 final CLS")):
            if f"{rk}_V" not in conds: continue
            vk = vote[rk]; vl = DW[DW.cond == rk + "_S"].set_index("query_id").loc[W.index, "vote"].astype(str)
            for pvn, pl in (("S", "standard"), ("V", "vote-aware")):
                c = f"{rk}_{pvn}"; L.append(f"| {nm} | {pl} | {int((vk & ~ok[c]).sum())}/{int(vk.sum())} | {int((~vk & ok[c]).sum())}/{int((~vk).sum())} | {int((W[c].astype(str) == vl).sum())}/153 | " + (f"{int((~ok[rk + '_S'] & ok[rk + '_V']).sum())} | {int((ok[rk + '_S'] & ~ok[rk + '_V']).sum())} |" if pvn == "V" else " | |"))
        L.append("\nThe vote-aware text names the majority label as the default; with a tie between the two most frequent labels it states the counts without a default. The hard vote breaks ties by summed cosine and then alphabetical order.\n")
    if {"A_S", "A_V", "DINO_S", "DINO_V"} <= set(conds):
        b1, b2, b3, b4 = paired("A_S", "DINO_S"), paired("A_V", "DINO_V"), paired("DINO_S", "DINO_V"), paired("H_V", "DINO_V")
        dd = (ok["DINO_V"].astype(int) - ok["A_V"].astype(int)) - (ok["DINO_S"].astype(int) - ok["A_S"].astype(int)); m = dd.to_numpy()[rng.integers(0, len(dd), size=(NB, len(dd)))].mean(axis=1)
        L += ["\n### Backbone by prompt: ViT pooled output versus DINOv2 final CLS\n", "| | Vote | Standard prompt | Vote-aware prompt |", "|---|---|---|---|",
              f"| ViT pooled output | {int(vote['A'].sum())}/153 | {int(ok['A_S'].sum())}/153 | {int(ok['A_V'].sum())}/153 |", f"| DINOv2 final CLS | {int(vote['DINO'].sum())}/153 | {int(ok['DINO_S'].sum())}/153 | {int(ok['DINO_V'].sum())}/153 |",
              f"| DINOv2 minus ViT (wins / losses, p) | | {b1['wins']} / {b1['losses']}, p {fp(b1['p'])} ({100 * b1['diff']:+.1f} pp, CI {100 * b1['ci'][0]:+.1f} to {100 * b1['ci'][1]:+.1f}) | {b2['wins']} / {b2['losses']}, p {fp(b2['p'])} ({100 * b2['diff']:+.1f} pp, CI {100 * b2['ci'][0]:+.1f} to {100 * b2['ci'][1]:+.1f}) |",
              f"\nBackbone-by-prompt interaction: {100 * dd.mean():+.1f} pp (95% CI {100 * np.percentile(m, 2.5):+.1f} to {100 * np.percentile(m, 97.5):+.1f}). Vote-aware versus standard with DINOv2 references: {b3['wins']} wins, {b3['losses']} losses (p {fp(b3['p'])}). "
              f"DINOv2 vote-aware versus the DINOv2 vote itself: {int((ok['DINO_V'] & ~vote['DINO']).sum())} wins, {int((~ok['DINO_V'] & vote['DINO']).sum())} losses. DINOv2 vote-aware versus ViT patch mean vote-aware: {b4['wins']} wins, {b4['losses']} losses (p {fp(b4['p'])}).\n"]
    # qualitative figure: first query (sorted ID) where the latent helps and first where it hurts, standard prompt
    if {"A_S", "H_S"} <= set(conds):
        picks = [("latent helps", [q for q in sorted(W.index) if ok["H_S"][q] and not ok["A_S"][q]]), ("latent hurts", [q for q in sorted(W.index) if ok["A_S"][q] and not ok["H_S"][q]])]
        picks = [(n, qs[0]) for n, qs in picks if qs]
        if picks:
            S = Image.new("RGB", (5 * 262 + 6, len(picks) * 2 * 300 + 4), "white"); d = ImageDraw.Draw(S)
            for i, (nm, q) in enumerate(picks):
                for j, rk in enumerate(("A", "H")):
                    y0 = (2 * i + j) * 300; w = NBR[(NBR.readout_key == rk) & (NBR.query_id == q)].iloc[0]; ids, labs = json.loads(w["neighbor_ids"]), json.loads(w["neighbor_labels"])
                    S.paste(Image.open(path_of(q)).convert("RGB"), (0, y0 + 40)); d.text((4, y0 + 4), f"[{nm}] query {q}", fill="black"); d.text((4, y0 + 20), f"true: {w['true_label']}", fill="black")
                    for s_, (f, l) in enumerate(zip(ids, labs)):
                        S.paste(Image.open(path_of(f)).convert("RGB"), ((s_ + 1) * 262, y0 + 40)); d.text(((s_ + 1) * 262 + 4, y0 + 4), f"{'pooled' if rk == 'A' else 'selected'} #{s_ + 1}", fill="black"); d.text(((s_ + 1) * 262 + 4, y0 + 20), l, fill=(0, 110, 60) if l == w["true_label"] else (150, 40, 20))
                    d.text((4, y0 + 300 - 0, ), "", fill="black")
                    d.text((266, y0 + 298 - 0), "", fill="black")
                    d.text((4, y0 + 284 + 12), f"vote: {w['vote']}   VLM (standard prompt): {W.at[q, rk + '_S']}", fill="black")
            S.save(os.path.join(out, "paired_neighborhoods.png"))
            L += ["## 6. Representative paired neighborhoods\n", "`paired_neighborhoods.png`: for each case, the first row is the pooled retrieval and the second row the selected hidden state. Cases are the first query in ID order where the selected readout turns a wrong standard-prompt answer into a correct one, and the first where it does the reverse.\n"]
            for nm, q in picks:
                a, h = NBR[(NBR.readout_key == "A") & (NBR.query_id == q)].iloc[0], NBR[(NBR.readout_key == "H") & (NBR.query_id == q)].iloc[0]
                L.append(f"- **{nm}:** `{q}` (true {a['true_label']}). Pooled references {json.loads(a['neighbor_labels'])}, vote {a['vote']}, VLM {W.at[q, 'A_S']}. Selected references {json.loads(h['neighbor_labels'])}, vote {h['vote']}, VLM {W.at[q, 'H_S']}.")
            L.append("")
json.dump({"n_lines": len(L)}, open(os.path.join(out, "report_meta.json"), "w"))
notes = os.path.join(out, "NOTES.md")
if os.path.exists(notes): L.append(open(notes).read())
open(os.path.join(out, "SUMMARY.md"), "w").write("\n".join(L) + "\n"); print("\n".join(L))
