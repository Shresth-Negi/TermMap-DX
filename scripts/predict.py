"""Map an evaluation file end to end.

Input: CSV with an entity (or clinical_entity) column and optionally chunk_text,
chunk_id, coding_system, subconcept. Output: the input columns plus
icd10cm_code and snomed_code for every row (and, when the input names a
coding_system per row, a ``code`` column holding that system's prediction),
together with confidence / margin / top-5 alternatives for each system.

Usage: python scripts/predict.py --input data/raw/eval.csv --output outputs/predictions.csv
"""
import argparse
from pathlib import Path

import pandas as pd

from termmap.engine import Engine
from termmap.io import load_input

ap = argparse.ArgumentParser()
ap.add_argument("--input", required=True)
ap.add_argument("--output", default="outputs/predictions.csv")
ap.add_argument("--dense", nargs="*", default=["sapbert"])
ap.add_argument("--tag", default="")
ap.add_argument("--llm-scores", default="", help="precomputed LLM shortlist scores (see scripts/llm_score.py)")
ap.add_argument("--brief", action="store_true", help="write only the deliverable schema columns")
ap.add_argument("--llm-model", default="", help="score shortlists inline with this local LLM (needs a GPU)")
args = ap.parse_args()

df = load_input(args.input)
eng = Engine(dense=tuple(args.dense), tag=args.tag, llm_model=args.llm_model)
llm = pd.read_parquet(args.llm_scores) if args.llm_scores else None
res = eng.map(df, llm_scores=llm)
out = pd.concat([df.reset_index(drop=True), res.reset_index(drop=True)], axis=1)
if "coding_system" in out:
    out["code"] = [i if s == "ICD10CM" else n if s == "SNOMEDCT_US" else ""
                   for s, i, n in zip(out["coding_system"], out["icd10cm_code"], out["snomed_code"])]
orig_cols = list(pd.read_csv(args.input, nrows=0).columns)
if "clinical_entity" in orig_cols:
    out = out.rename(columns={"entity": "clinical_entity"})
if args.brief:
    keep = [c for c in orig_cols if c in out and c not in ("code", "display")]
    keep += [c for c in ("code", "icd10cm_code", "snomed_code") if c in out and c not in keep]
    out = out[keep]
Path(args.output).parent.mkdir(parents=True, exist_ok=True)
out.to_csv(args.output, index=False)
print(f"wrote {len(out)} rows -> {args.output}  (model set {eng.tag}"
      + (f"; fallback because {eng.fallback_reason})" if eng.fallback_reason else ")"))
