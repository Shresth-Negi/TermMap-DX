"""Stage the trained artifacts and upload them to the Hugging Face Hub (maintainers only).

  HF_TOKEN=... python scripts/upload_artifacts.py --repo <namespace>/TermMap-DX [--private]

The token is read from the environment only; it is never written to disk or to the repo.
"""
import argparse
import os
from pathlib import Path

from termmap.hub import SETS, stage
from termmap.paths import ARTIFACTS, ROOT

ap = argparse.ArgumentParser()
ap.add_argument("--repo", required=True)
ap.add_argument("--private", action="store_true")
ap.add_argument("--stage-only", action="store_true", help="build the upload folder, do not upload")
args = ap.parse_args()

card = (ROOT / "docs" / "HF_MODEL_CARD.md").read_text()
out = Path(os.environ.get("TERMMAP_WORK", ARTIFACTS.parent)) / "hf_upload"
files = stage(out, SETS, card)
size = sum((out / f).stat().st_size for f in files) / 2**30
print(f"staged {len(files)} files ({size:.2f} GB) in {out}")
if not args.stage_only:
    from huggingface_hub import HfApi
    api = HfApi(token=os.environ["HF_TOKEN"])
    api.create_repo(args.repo, repo_type="model", private=args.private, exist_ok=True)
    api.upload_folder(repo_id=args.repo, repo_type="model", folder_path=str(out),
                      commit_message="TermMap-DX trained artifacts", ignore_patterns=[".cache/*"])
    print(f"uploaded to https://huggingface.co/{args.repo}")
