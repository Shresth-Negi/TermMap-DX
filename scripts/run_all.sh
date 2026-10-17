#!/bin/bash
# One command: build whatever is missing (indexes, models) and map an evaluation file.
#
#   bash scripts/run_all.sh <input.csv> [output.csv]
#
# Configuration (environment variables):
#   TERMMAP_TAG    model set to use (default _d3llm = 3 dense encoders + LLM stage; needs a GPU)
#                  _d3 = same without the LLM stage (CPU-friendly), _sap = SapBERT only
#   TERMMAP_LLM    local LLM used by the LLM stage (default Qwen/Qwen2.5-7B-Instruct)
#
# Every step is deterministic (fixed seeds, deterministic LightGBM, no sampling).
set -euo pipefail
cd "$(dirname "$0")/.."
source scripts/env.sh
INPUT="${1:?usage: run_all.sh <input.csv> [output.csv]}"
OUTPUT="${2:-outputs/predictions.csv}"
TAG="${TERMMAP_TAG:-_d3llm}"
LLM="${TERMMAP_LLM:-Qwen/Qwen2.5-7B-Instruct}"
A="$TERMMAP_ARTIFACTS_DIR"

case "$TAG" in
  _sap) DENSE="sapbert" ;;
  *)    DENSE="sapbert gte qwen3" ;;
esac
FT="${TAG%llm}"   # stage-1 features are shared between _d3 and _d3llm

$PY scripts/build_index.py --models $DENSE
if [ ! -f "$A/ranker_SNOMEDCT_US_s2$FT.txt" ]; then
  $PY scripts/train.py --dense $DENSE --tag "$FT"
fi
LLM_ARGS=()
if [[ "$TAG" == *llm ]]; then
  if [ ! -f "$A/ranker_SNOMEDCT_US_s2$TAG.txt" ]; then
    [ -f "$A/llm_scores_train$FT.parquet" ] || $PY scripts/llm_score_train.py --tag "$FT" --model "$LLM"
    $PY scripts/train.py --dense $DENSE --tag "$TAG" --feats-tag "$FT" --llm "$A/llm_scores_train$FT.parquet"
  fi
  LLM_ARGS=(--llm-model "$LLM")
fi
$PY scripts/predict.py --input "$INPUT" --output "$OUTPUT" --tag "$TAG" "${LLM_ARGS[@]}"
