#!/bin/bash
#SBATCH --job-name=termmap_gpu2
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=32
#SBATCH --mem=160G
#SBATCH --time=02:00:00
#SBATCH --output=slurm/logs/%x_%j.log
# GPU validation of the inference path with the LLM loaded inline: sample predictions + hard-case
# demo for the default model set. The engine falls back automatically (and says so) if the GPU it
# gets does not have enough free memory for the LLM.
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$(pwd)}"
source scripts/env.sh
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
echo "host $(hostname)  job $SLURM_JOB_ID  start $(date)"
nvidia-smi --query-gpu=name,memory.total,memory.used --format=csv
TAG=_d3llm14e4
mkdir -p outputs
/usr/bin/time -v $PY scripts/predict.py --input data/sample/sample_input.csv \
    --output outputs/sample_predictions$TAG.csv --tag $TAG 2>&1 | grep -v -i "warn\|Loading weights"
$PY scripts/demo.py --tag $TAG > outputs/demo$TAG.txt 2>&1 || true
grep -v -i "warn\|Loading" outputs/demo$TAG.txt | tail -70
echo "done $(date)"
