#!/bin/bash
#SBATCH --job-name=termmap_gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=32
#SBATCH --mem=160G
#SBATCH --time=12:00:00
#SBATCH --output=slurm/logs/%x_%j.log
# Full GPU sequence behind the reported numbers. Every step is skipped when its artifact exists.
#   _d3         SapBERT + gte-base + Qwen3-Embedding signals (stage-1 base for everything below)
#   _d3ce       + cross-encoder, 2 epochs                                   (ablation row)
#   _d3ce4      + cross-encoder, 4 epochs                                   (no-LLM set)
#   _d3llm7e4   + Qwen2.5-7B-Instruct listwise scores      (falls back to _d3ce4)
#   _d3llm14e4  + Qwen2.5-14B-Instruct listwise scores     (default; falls back to _d3llm7e4)
# Submit from the repo root:  sbatch slurm/gpu_pipeline.sbatch   (about 2.5 h on one RTX 6000 Ada)
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$(pwd)}"
source scripts/env.sh
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
A="$TERMMAP_ARTIFACTS_DIR"
D="sapbert gte qwen3"
echo "host $(hostname)  job $SLURM_JOB_ID  start $(date)"
nvidia-smi --query-gpu=name,memory.total,memory.used --format=csv

train_set() {  # train_set TAG [train.py args...]; skipped when the set exists
  local tag="$1"; shift
  [ -f "$A/ranker_SNOMEDCT_US_s2$tag.txt" ] || $PY scripts/train.py --dense $D --tag "$tag" "$@"
  $PY scripts/error_analysis.py --tag "$tag" > /dev/null
}

$PY scripts/build_index.py --models $D
train_set _d3
echo "== cross-encoders $(date)"
[ -f "$A/ce_scores_train_d3.parquet" ]   || $PY scripts/train_cross_encoder.py --tag _d3 --epochs 2
[ -f "$A/ce_scores_train_d3e4.parquet" ] || $PY scripts/train_cross_encoder.py --tag _d3 --out-tag _d3e4 --epochs 4
train_set _d3ce  --feats-tag _d3 --extra "$A/ce_scores_train_d3.parquet"
train_set _d3ce4 --feats-tag _d3 --extra "$A/ce_scores_train_d3e4.parquet"
echo "== LLM 7B $(date)"
[ -f "$A/llm_scores_train_d3.parquet" ] || \
  $PY scripts/llm_score_train.py --tag _d3 --model Qwen/Qwen2.5-7B-Instruct --shots 5 --batch 16
train_set _d3llm7e4 --feats-tag _d3 --extra "$A/ce_scores_train_d3e4.parquet" \
  --llm "$A/llm_scores_train_d3.parquet" --llm-model-id Qwen/Qwen2.5-7B-Instruct --fallback-tag _d3ce4
echo "== LLM 14B $(date)"
[ -f "$A/llm_scores_train_d3_q14.parquet" ] || \
  $PY scripts/llm_score_train.py --tag _d3 --model Qwen/Qwen2.5-14B-Instruct --shots 5 --batch 8 --out-suffix _q14
train_set _d3llm14e4 --feats-tag _d3 --extra "$A/ce_scores_train_d3e4.parquet" \
  --llm "$A/llm_scores_train_d3_q14.parquet" --llm-model-id Qwen/Qwen2.5-14B-Instruct --fallback-tag _d3llm7e4
$PY scripts/export_oof.py --tag _d3llm14e4
echo "done $(date)"
