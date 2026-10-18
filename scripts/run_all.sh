#!/bin/bash
# One command: build whatever is missing (indexes, models) and map an evaluation file.
#
#   bash scripts/run_all.sh <input.csv> [output.csv]
#
# TERMMAP_TAG selects the model set (default _d3llm14e4). Each adds to the previous one:
#   _sap4       SapBERT dense + lexical + training memory               CPU only, ~50 ms/term
#   _d3ce4      + gte-base + Qwen3-Embedding + 4-epoch cross-encoder    no GPU needed at inference
#   _d3llm7e4   + Qwen2.5-7B-Instruct listwise scores                   GPU with >= 18 GB free
#   _d3llm14e4  + Qwen2.5-14B-Instruct instead of 7B                    GPU with >= 31 GB free
# If the LLM of the chosen set cannot run (no GPU, not enough free memory), the engine falls
# back automatically: _d3llm14e4 -> _d3llm7e4 -> _d3ce4, and logs why.
# Every step is deterministic (fixed seeds, deterministic LightGBM, no sampling).
set -euo pipefail
cd "$(dirname "$0")/.."
source scripts/env.sh
INPUT="${1:?usage: run_all.sh <input.csv> [output.csv]}"
OUTPUT="${2:-outputs/predictions.csv}"
TAG="${TERMMAP_TAG:-_d3llm14e4}"
A="$TERMMAP_ARTIFACTS_DIR"
D3="sapbert gte qwen3"

# build_set TAG: make sure the artifacts of one model set exist (no-op when already trained)
build_set() {
  local tag="$1" dense="$D3" base=_d3 ce="" llmid="" llmsuf="" fb=""
  case "$tag" in
    _sap4)      dense="sapbert"; base=_sap4 ;;
    _d3ce4)     ce=_d3e4 ;;
    _d3llm7e4)  ce=_d3e4; llmid=Qwen/Qwen2.5-7B-Instruct; fb=_d3ce4 ;;
    _d3llm14e4) ce=_d3e4; llmid=Qwen/Qwen2.5-14B-Instruct; llmsuf=_q14; fb=_d3llm7e4 ;;
    *) echo "unknown model set $tag"; exit 1 ;;
  esac
  [ -n "$fb" ] && build_set "$fb"
  # already trained or downloaded from Hugging Face: nothing to build
  [ -f "$A/ranker_SNOMEDCT_US_s2$tag.txt" ] && [ -f "$A/config$tag.json" ] && return 0
  $PY scripts/build_index.py --models $dense
  [ -f "$A/ranker_SNOMEDCT_US_s2$base.txt" ] || $PY scripts/train.py --dense $dense --tag "$base"
  [ "$tag" = "$base" ] && return
  local args=()
  if [ -n "$ce" ]; then
    [ -f "$A/ce_scores_train$ce.parquet" ] || $PY scripts/train_cross_encoder.py --tag "$base" --out-tag "$ce" --epochs 4
    args+=(--extra "$A/ce_scores_train$ce.parquet")
  fi
  if [ -n "$llmid" ]; then
    [ -f "$A/llm_scores_train$base$llmsuf.parquet" ] || \
      $PY scripts/llm_score_train.py --tag "$base" --model "$llmid" --out-suffix "$llmsuf"
    args+=(--llm "$A/llm_scores_train$base$llmsuf.parquet" --llm-model-id "$llmid" --fallback-tag "$fb")
  fi
  [ -f "$A/ranker_SNOMEDCT_US_s2$tag.txt" ] || \
    $PY scripts/train.py --dense $dense --tag "$tag" --feats-tag "$base" "${args[@]}"
}

# trained artifacts: pulled from Hugging Face (files already present are skipped); if that is not
# possible (offline, repo unavailable) the missing pieces are trained locally instead (needs a GPU)
$PY scripts/download_artifacts.py --sets "$TAG" || echo "[run_all] Hugging Face download unavailable; building locally"
build_set "$TAG"
# deliverable schema (input columns + icd10cm_code, snomed_code) and a detailed copy with
# descriptions, calibrated confidences, margins and top-5 alternatives
$PY scripts/predict.py --input "$INPUT" --output "$OUTPUT" --tag "$TAG" --brief \
    --details-output "${OUTPUT%.csv}_detailed.csv"
