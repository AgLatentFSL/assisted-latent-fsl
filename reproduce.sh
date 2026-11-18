#!/usr/bin/env bash
# Regenerates the main results of the paper from the released runs, without any VLM call.
# 1. downloads runs/ and atlas/ from Hugging Face (about 2.6 GB) if they are missing,
# 2. rebuilds per-query predictions and statistics from the raw VLM responses,
# 3. rebuilds Table 1, the main-text figures and the numbers file, and compares them with the paper copies.
set -euo pipefail
cd "$(dirname "$0")"
export KAGGLE_USERNAME="${KAGGLE_USERNAME:-anonymous}" KAGGLE_KEY="${KAGGLE_KEY:-anonymous}"   # data_loader.py imports kaggle
PY="${PYTHON:-python}"
HF=tirtho149/assisted-latent-fsl-runs
if [ ! -d runs ] || [ ! -d atlas ]; then
  "$PY" -c "from huggingface_hub import snapshot_download as s; s('$HF', repo_type='dataset', local_dir='hf_release')"
  [ -e runs ]  || ln -s hf_release runs              # run folders sit at the root of the dataset
  [ -e atlas ] || ln -s hf_release/atlas atlas
fi
# raw VLM answers are packed as one responses.tar.gz per run folder
find -L runs -name responses.tar.gz | while read -r t; do
  d="$(dirname "$t")"; [ -d "$d/responses" ] || tar -xzf "$t" -C "$d"
done
OUT=reproduced/layer_vlm_allk
mkdir -p "$OUT" tables figs/appendix
cp runs/layer_vlm_allk/{conditions.csv,request_plan.csv,preregistration.json} "$OUT"/
[ -e "$OUT/responses" ] || ln -s "$PWD/runs/layer_vlm_allk/responses" "$OUT/responses"
"$PY" pilot_layer_vlm_allk_report.py "$OUT"
"$PY" paper_tools/make_layer_vlm_allk.py "$OUT"
for f in per_query_predictions.csv results_by_condition.csv encoder_level.csv; do
  cmp -s "$OUT/$f" "runs/layer_vlm_allk/$f" && echo "OK   $f matches the released run" || echo "DIFF $f"
done
diff -q tables/tab_vlm_allk.tex paper_tables/tab_vlm_allk.tex && echo "OK   Table 1 matches the paper"
"$PY" -c "import json;a=json.load(open('tables/layer_vlm_allk_numbers.json'));b=json.load(open('paper_tables/layer_vlm_allk_numbers.json'));print('OK   all reported numbers match' if a==b else 'DIFF numbers')"
"$PY" paper_tools/make_appendix.py atlas runs/blockwise_vs_pooled > reproduced/appendix.log
diff -q tables/layer_gain_ranking.csv paper_tables/layer_gain_ranking.csv && echo "OK   layer ranking (retrieval screening) matches the paper"
"$PY" -c "import json;a=json.load(open('tables/appendix_numbers.json'));b=json.load(open('paper_tables/appendix_numbers.json'));print('OK   retrieval appendix numbers match' if a==b else 'DIFF appendix numbers')"
echo "Figures written to figs/ (fig_hero, fig_tradeoff, fig_mean5_heat, ...)."
