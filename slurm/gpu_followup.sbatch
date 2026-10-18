#!/bin/bash
#SBATCH --job-name=termmap_gpu2
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=32
#SBATCH --mem=160G
#SBATCH --time=04:00:00
#SBATCH --output=slurm/logs/%x_%j.log
# Follow-up GPU work: (1) 4-epoch cross-encoder if missing, (2) end-to-end GPU inference check of
# the best LLM model set (inline LLM + cross-encoder) on a sample, (3) hard-case demo output.
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$(pwd)}"
source scripts/env.sh
A="$TERMMAP_ARTIFACTS_DIR"
echo "host $(hostname)  job $SLURM_JOB_ID  start $(date)"
[ -f "$A/ce_scores_train_d3e4.parquet" ] || $PY scripts/train_cross_encoder.py --tag _d3 --out-tag _d3e4 --epochs 4
# best LLM model set by cross-validated headline (written by train.py)
FINAL=$($PY - <<'EOF'
import json, os
A = os.environ["TERMMAP_ARTIFACTS_DIR"]
best = None
for t in ["_d3llm", "_d3llm14", "_final"]:
    p = f"{A}/cv_report{t}.json"
    if os.path.exists(p) and os.path.exists(f"{A}/ranker_SNOMEDCT_US_s2{t}.txt"):
        h = json.load(open(p))["headline_min_top1"]
        if best is None or h > best[0]:
            best = (h, t)
print(best[1])
EOF
)
echo "best model set: $FINAL"
mkdir -p outputs
$PY scripts/predict.py --input data/sample/sample_input.csv --output outputs/sample_predictions$FINAL.csv --tag "$FINAL"
$PY scripts/demo.py --tag "$FINAL" > outputs/demo$FINAL.txt 2>&1 || true
grep -v -i "warn\|Loading" outputs/demo$FINAL.txt | tail -60
echo "done $(date)"
