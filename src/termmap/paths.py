import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
RAW = Path(os.environ.get("TERMMAP_RAW_DIR", ROOT / "data" / "raw"))
ARTIFACTS = Path(os.environ.get("TERMMAP_ARTIFACTS_DIR", ROOT / "artifacts"))
TRAIN_CSV = RAW / "hackathon_train.csv"
