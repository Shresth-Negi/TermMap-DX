# Source this before running anything: sets venv, caches and work dir.
# Large artifacts (venv, model weights, embedding indexes) live on fast local disk
# under TERMMAP_WORK; override it to relocate.
_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
# optional local overrides (untracked): TERMMAP_WORK=..., HF_TOKEN=...
if [ -f "$_ROOT/.env" ]; then set -a; source "$_ROOT/.env"; set +a; fi
export TERMMAP_WORK="${TERMMAP_WORK:-$_ROOT/.work}"
export HF_HOME="$TERMMAP_WORK/hf"
export HF_HUB_DISABLE_TELEMETRY=1
export TOKENIZERS_PARALLELISM=false
export PIP_CACHE_DIR="$TERMMAP_WORK/pipcache"
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export PYTHONPATH="$REPO_ROOT/src:$PYTHONPATH"
export TERMMAP_ARTIFACTS_DIR="${TERMMAP_ARTIFACTS_DIR:-$TERMMAP_WORK/artifacts}"
PY="$TERMMAP_WORK/venv/bin/python"
export PY
# Never touch GPUs outside a SLURM allocation (the login node also has GPUs attached).
if [ -z "${SLURM_JOB_ID:-}" ]; then
  export CUDA_VISIBLE_DEVICES=""
fi
