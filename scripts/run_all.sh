#!/bin/bash
# One command: build whatever is missing (indexes, models) and map an evaluation file.
#
#   bash scripts/run_all.sh <input.csv> [output.csv]
#
# TERMMAP_TAG selects the model set (default _d3llm14). Each is a strict superset of the previous:
#   _sap4     SapBERT dense + lexical + training memory              CPU only, ~0.2 s/term
#   _d3       + gte-base + Qwen3-Embedding signals                   CPU ok (indexes built once on GPU)
#   _d3ce     + fine-tuned cross-encoder over the shortlist          CPU ok, GPU faster
#   _d3llm    + Qwen2.5-7B-Instruct listwise scores                  needs a GPU (>= 24 GB)
#   _d3llm14  + Qwen2.5-14B-Instruct instead of 7B                   needs a GPU (>= 40 GB)
# Every step is deterministic (fixed seeds, deterministic LightGBM, no sampling).
set -euo pipefail
cd "$(dirname "$0")/.."
source scripts/env.sh
INPUT="${1:?usage: run_all.sh <input.csv> [output.csv]}"
OUTPUT="${2:-outputs/predictions.csv}"
TAG="${TERMMAP_TAG:-_d3llm14}"
A="$TERMMAP_ARTIFACTS_DIR"
CE=0; LLMID=""; LLMSUF=""
case "$TAG" in
  _sap4)    DENSE="sapbert";           BASE=_sap4 ;;
  _d3)      DENSE="sapbert gte qwen3"; BASE=_d3 ;;
  _d3ce)    DENSE="sapbert gte qwen3"; BASE=_d3; CE=1 ;;
  _d3llm)   DENSE="sapbert gte qwen3"; BASE=_d3; CE=1; LLMID=Qwen/Qwen2.5-7B-Instruct ;;
  _d3llm14) DENSE="sapbert gte qwen3"; BASE=_d3; CE=1; LLMID=Qwen/Qwen2.5-14B-Instruct; LLMSUF=_q14 ;;
  *) echo "unknown TERMMAP_TAG $TAG"; exit 1 ;;
esac

$PY scripts/build_index.py --models $DENSE
[ -f "$A/ranker_SNOMEDCT_US_s2$BASE.txt" ] || $PY scripts/train.py --dense $DENSE --tag "$BASE"
ARGS=()
if [ "$CE" = 1 ]; then
  [ -f "$A/ce_scores_train$BASE.parquet" ] || $PY scripts/train_cross_encoder.py --tag "$BASE"
  ARGS+=(--extra "$A/ce_scores_train$BASE.parquet")
fi
if [ -n "$LLMID" ]; then
  [ -f "$A/llm_scores_train$BASE$LLMSUF.parquet" ] || \
    $PY scripts/llm_score_train.py --tag "$BASE" --model "$LLMID" --out-suffix "$LLMSUF"
  ARGS+=(--llm "$A/llm_scores_train$BASE$LLMSUF.parquet" --llm-model-id "$LLMID")
fi
if [ "$TAG" != "$BASE" ] && [ ! -f "$A/ranker_SNOMEDCT_US_s2$TAG.txt" ]; then
  $PY scripts/train.py --dense $DENSE --tag "$TAG" --feats-tag "$BASE" "${ARGS[@]}"
fi
$PY scripts/predict.py --input "$INPUT" --output "$OUTPUT" --tag "$TAG"
