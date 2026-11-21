# Assisted Latent Few-Shot Learning for Vision-Language Models in Agricultural Stress Phenotype Identification

Tirtho Roy, Ushashi Bhattacharjee, Adarsh Krishnamurthy, Soumik Sarkar (Iowa State University)

[Project page](https://aglatentfsl.github.io/) · [Paper (PDF)](https://aglatentfsl.github.io/static/paper/paper.pdf) · [All runs (Hugging Face)](https://huggingface.co/datasets/tirtho149/assisted-latent-fsl-runs)

Assisted few-shot learning retrieves the few-shot examples of a vision-language model (VLM) with the pooled output of a frozen image encoder. Assisted latent few-shot learning retrieves them with a hidden layer of the same encoder and keeps the reference bank, the prompt and the VLM unchanged. On five AgEval identification tasks, six encoders and the 1-, 2-, 4- and 8-shot settings (Claude Sonnet 5.5), the hidden layer gives higher accuracy in 21 of 24 combinations of encoder and shot count, with the same number of prompt tokens.

## Reproduce the paper's numbers (no VLM call)

```bash
git clone https://github.com/AgLatentFSL/assisted-latent-fsl && cd assisted-latent-fsl
python3.9 -m venv .venv && source .venv/bin/activate
pip install -r requirements-release.txt
./reproduce.sh
```

`reproduce.sh` downloads the released runs (about 2.6 GB) into `hf_release/`, unpacks the raw VLM answers, rebuilds every per-query prediction and statistic from them, regenerates Table 1, the main figures and the retrieval appendix, and compares the outputs with the copies of the paper in `paper_tables/`. Expected output:

```
OK   per_query_predictions.csv matches the released run
OK   results_by_condition.csv matches the released run
OK   encoder_level.csv matches the released run
OK   Table 1 matches the paper
OK   all reported numbers match
OK   layer ranking (retrieval screening) matches the paper
OK   retrieval appendix numbers match
```

Figures are written to `figs/` and tables to `tables/`.

## Rerun the experiments from scratch

All commands run from the repository root, with `runs/` and `atlas/` in the root (`reproduce.sh` links them to `hf_release/`; to start fresh, point them at empty folders instead).

1. **Data.** `data_loader.py` downloads each AgEval dataset from its public source into `data/` (Kaggle datasets need a Kaggle API token in `~/.kaggle/kaggle.json`). Images are not redistributed. Every split file lists the MD5 hash of each image, so a fresh download can be checked against the released split.
2. **Split.** `python make_disjoint_split.py runs/disjoint_split_seed42` writes 100 queries and a separate class-balanced 100-image reference bank per dataset (seed 42).
3. **Latents and retrieval.** `python disjoint_features.py runs/disjoint_split_seed42` caches the readout of every block of the nine encoders. `python make_retrieval_samples.py atlas` builds the layer-wise retrieval index, and `cd atlas && python analysis/topk_analysis.py --reparse` writes the same-class coverage tables used to screen the layers.
4. **VLM run.** `python pilot_layer_vlm_allk.py runs/layer_vlm_allk freeze` freezes the five screened layers per encoder and shot count and the request plan before any test result. `... vlm` sends the requests and `... report` writes the statistics. Requests go through the headless Claude Code CLI (`claude -p`, class `ClaudeHeadlessAPI` in `inference.py`), so the CLI must be installed and signed in. Identical requests are sent once and cached in `responses/`.
5. **Paper tables and figures.** `python pilot_layer_vlm_allk_report.py runs/layer_vlm_allk`, then `python paper_tools/make_layer_vlm_allk.py` and `python paper_tools/make_appendix.py atlas runs/blockwise_vs_pooled`.

VLM answers are sampled with default settings, so a fresh step 4 gives close but not identical accuracies; steps 1 to 3 and 5 are deterministic.

## Where each result lives

| Paper element | Run folder (in `runs/`) | Script |
|---|---|---|
| Table 1, Fig. 2 (accuracy against shot count), Fig. 4 (prompt tokens), Fig. 5 (stability) and the appendix with every condition | `layer_vlm_allk` | `pilot_layer_vlm_allk.py`, `pilot_layer_vlm_allk_report.py`, `paper_tools/make_layer_vlm_allk.py` |
| Fig. 3 (same-class retrieval), retrieval appendix with nine encoders | `atlas/analysis`, `layer_retrieval_audit` | `disjoint_features.py`, `make_retrieval_samples.py`, `atlas_analysis/`, `paper_tools/make_appendix.py` |
| Fig. 6 (case study) | `layer_vlm_k2`, `layer_vlm_allk` | `paper_tools/make_case_study.py` |
| First 2-shot test with nine encoders, tests between layers | `layer_vlm_k2` | `pilot_layer_vlm.py`, `layer_pairs_k2.py`, `paper_tools/make_layer_vlm.py`, `paper_tools/make_layer_tests.py` |
| Date-disjoint check on DeepWeeds | `blockwise_vs_pooled` | `blockwise_vs_pooled.py` |
| AgEval protocol, source-group audit, earlier pilots (supporting appendix) | `arbab_eval_seed42`, `deepweeds_*`, `paddy_*`, `latent_afsl`, `latent_search*` | `pilot_arbab_eval.py`, `split_audit.py`, `pilot_*.py`, `latent_*.py` |

Each run folder keeps its manifest or pre-registration (written before the test metrics), the request plan, per-query predictions, a `SUMMARY.md` and the logs. Some runs (`rank1_boundary`, `qwen_latent_db`) are exploratory and are not reported in the paper.

## Acknowledgment

This code builds on the AgEval benchmark and the assisted few-shot learning pipeline of Muhammad Arbab Arshad et al. ([arbab-ml/AgEval](https://github.com/arbab-ml/AgEval)); their original README is kept as `README_AgEval.md`.

## License

The code is released under the MIT License (`LICENSE`). The runs, figures and tables are released under CC BY 4.0 (`LICENSE-DATA.md`). Dataset images are not redistributed and keep the licenses of their original sources.

## Citation

```bibtex
@article{roy2026assistedlatent,
  title  = {Assisted Latent Few-Shot Learning for Vision-Language Models
            in Agricultural Stress Phenotype Identification},
  author = {Roy, Tirtho and Bhattacharjee, Ushashi and
            Krishnamurthy, Adarsh and Sarkar, Soumik},
  year   = {2026},
  note   = {Preprint}
}
```
