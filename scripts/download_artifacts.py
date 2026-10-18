"""Download the trained artifacts from the Hugging Face Hub into $TERMMAP_ARTIFACTS_DIR.

  python scripts/download_artifacts.py                    # default model set and its fallbacks (~4.6 GB)
  python scripts/download_artifacts.py --sets _sap4       # CPU-light set only (~1.2 GB)
  TERMMAP_HF_REPO=<namespace>/<repo> python scripts/download_artifacts.py   # another copy

Files already present with the right sha256 are skipped, so it is safe to re-run.
"""
import argparse

from termmap.hub import download, repo_id
from termmap.paths import ARTIFACTS

ap = argparse.ArgumentParser()
ap.add_argument("--sets", nargs="+", default=["_d3llm14e4"])
ap.add_argument("--repo", default=None)
args = ap.parse_args()
repo = args.repo or repo_id()
print(f"downloading {args.sets} (with fallbacks) from huggingface.co/{repo} -> {ARTIFACTS}", flush=True)
try:
    got = download(args.sets, repo)
except Exception as e:  # offline, private repo without a token, wrong name ...
    raise SystemExit(f"download failed: {type(e).__name__}: {str(e).splitlines()[0][:200]}")
print(f"done: {len(got)} files downloaded, everything else already up to date")
