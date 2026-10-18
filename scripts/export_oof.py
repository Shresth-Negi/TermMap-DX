"""Export cross-validated (out-of-fold) predictions for the training file in the long format.

No held-out evaluation input was released with the data, so this file is the honest
"predictions" artefact: each row was predicted by models that never saw its label.

Usage: python scripts/export_oof.py --tag _sap4 [--output outputs/cv_predictions_sap4.csv]
"""
import argparse
import json

import numpy as np
import pandas as pd

from termmap.corpus import SYSTEMS, description_map
from termmap.io import load_train
from termmap.paths import ARTIFACTS, ROOT
from termmap.pipeline import attach_keys, make_keys

ap = argparse.ArgumentParser()
ap.add_argument("--tag", required=True)
ap.add_argument("--output", default=None)
args = ap.parse_args()
tr = load_train()
tr["key"] = attach_keys(tr, make_keys(tr)).values
tr["train_row"] = range(len(tr))  # 0-based data row of hackathon_train.csv (header excluded)
parts = []
for s in SYSTEMS:
    o = pd.read_parquet(ARTIFACTS / f"oof_{s}{args.tag}.parquet")
    cal = json.loads((ARTIFACTS / f"calib_{s}{args.tag}.json").read_text())
    desc = description_map(s)
    row = tr[tr.coding_system == s].set_index("key")["train_row"]
    parts.append(pd.DataFrame({
        "train_row": row.reindex(o["key"]).values,
        "entity": o["entity"], "coding_system": s, "subconcept": o["subconcept"],
        "gold_code": o["code"], "predicted_code": o["pred"], "predicted_description": o["pred"].map(desc),
        "confidence": np.round(np.interp(o["margin"].fillna(0), cal["x"], cal["y"]), 4),
        "correct": o["ok"].astype(int), "fold": o["fold"]}))
out = pd.concat(parts, ignore_index=True).sort_values("train_row").reset_index(drop=True)
path = args.output or ROOT / "outputs" / f"cv_predictions{args.tag}.csv"
out.to_csv(path, index=False)
print(f"wrote {len(out)} rows -> {path}")
print(out.groupby("coding_system")["correct"].mean().round(4).to_string())
