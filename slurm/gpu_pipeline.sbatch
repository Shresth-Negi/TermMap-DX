#!/bin/bash
#SBATCH --job-name=termmap_gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=32
#SBATCH --mem=160G
#SBATCH --time=12:00:00
#SBATCH --output=slurm/logs/%x_%j.log
# GPU stage of the pipeline. Each step adds one component and leaves a usable model set (tag):
#   _d3       SapBERT + gte-base + Qwen3-Embedding dense signals
#   _d3ce     + fine-tuned cross-encoder over the shortlist        (CPU-friendly at inference)
#   _d3llm    + Qwen2.5-7B-Instruct listwise scores                 (needs a GPU at inference)
#   _d3llm14  + Qwen2.5-14B-Instruct listwise scores instead of 7B  (needs a GPU at inference)
# Submit from the repo root:  sbatch slurm/gpu_pipeline.sbatch
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$(pwd)}"
source scripts/env.sh
A="$TERMMAP_ARTIFACTS_DIR"
echo "host $(hostname)  job $SLURM_JOB_ID  start $(date)"
nvidia-smi --query-gpu=name,memory.total,memory.used --format=csv
DENSE="sapbert gte qwen3"
$PY scripts/build_index.py --models $DENSE
[ -f "$A/ranker_SNOMEDCT_US_s2_d3.txt" ] || $PY scripts/train.py --dense $DENSE --tag _d3
$PY scripts/error_analysis.py --tag _d3
echo "== cross-encoder $(date)"
[ -f "$A/ce_scores_train_d3.parquet" ] || $PY scripts/train_cross_encoder.py --tag _d3 --epochs 2
$PY scripts/train.py --dense $DENSE --tag _d3ce --feats-tag _d3 --extra "$A/ce_scores_train_d3.parquet"
$PY scripts/error_analysis.py --tag _d3ce
echo "== LLM 7B $(date)"
[ -f "$A/llm_scores_train_d3.parquet" ] || $PY scripts/llm_score_train.py --tag _d3 --model Qwen/Qwen2.5-7B-Instruct --shots 5 --batch 16
$PY scripts/train.py --dense $DENSE --tag _d3llm --feats-tag _d3 --llm "$A/llm_scores_train_d3.parquet" \
    --llm-model-id Qwen/Qwen2.5-7B-Instruct --extra "$A/ce_scores_train_d3.parquet"
$PY scripts/error_analysis.py --tag _d3llm
echo "== LLM 14B $(date)"
[ -f "$A/llm_scores_train_d3_q14.parquet" ] || $PY scripts/llm_score_train.py --tag _d3 --model Qwen/Qwen2.5-14B-Instruct --shots 5 --batch 8 --out-suffix _q14
$PY scripts/train.py --dense $DENSE --tag _d3llm14 --feats-tag _d3 --llm "$A/llm_scores_train_d3_q14.parquet" \
    --llm-model-id Qwen/Qwen2.5-14B-Instruct --extra "$A/ce_scores_train_d3.parquet"
$PY scripts/error_analysis.py --tag _d3llm14
echo "done $(date)"
