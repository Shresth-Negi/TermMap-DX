"""Demo: map hard cases live and show confidence + alternatives.

  python scripts/demo.py                   # scripted hard cases, best model set this machine can run
  python scripts/demo.py --interactive     # type: term || context
  python scripts/demo.py --tag _d3ce4      # force a model set (_d3llm14e4, _d3llm7e4, _d3ce4, _sap4)
"""
import argparse
import time

import pandas as pd

from termmap.engine import Engine

HARD_CASES = [
    # (entity, context, subconcept, what it shows)
    ("MS", "echo: severe MS with mitral valve area 0.9 cm2, rheumatic mitral valve", "DiagnosisCode", "abbreviation collision (cardiac context)"),
    ("MS", "neurology f/u for relapsing remitting MS, on ocrelizumab, new demyelinating lesions on MRI", "DiagnosisCode", "abbreviation collision (neuro context)"),
    ("RA", "rheumatology: RA on methotrexate, morning stiffness and synovitis of MCP joints", "DiagnosisCode", "abbreviation collision"),
    ("HTN", "pmh: HTN, HLD, DM2", "DiagnosisCode", "abbreviation"),
    ("pulmonary htn", "echo: RVSP 55, moderate pulmonary htn", "DiagnosisCode", "nearest string is the wrong concept"),
    ("chronickidney disease stag 3", "assessment: chronickidney disease stag 3, cr 1.6", "DiagnosisCode", "OCR noise"),
    ("diabetes", "pmh: diabetes, on metformin", "DiagnosisCode", "specificity: unspecified is correct"),
    ("type 2 dm with diabetic nephropathy", "a/p: type 2 dm with diabetic nephropathy, uacr 450", "DiagnosisCode", "specificity: combination code"),
    ("left knee oa", "xr: severe left knee oa, tricompartmental", "DiagnosisCode", "laterality"),
    ("chest pain", "ros: denies chest pain, palpitations", "DiagnosisCode", "negated mention"),
    ("h/o dvt", "pmh: h/o dvt 2015, completed anticoagulation", "DiagnosisCode", "historical mention"),
    ("sob", "pt c/o sob on exertion x 2 weeks", "DiagnosisCode", "abbreviation"),
]

ap = argparse.ArgumentParser()
ap.add_argument("--tag", default="_d3llm14e4")
ap.add_argument("--interactive", action="store_true")
ap.add_argument("--llm-model", default="")
args = ap.parse_args()

eng = Engine(tag=args.tag, verbose=False, llm_model=args.llm_model)
print(f"model set: {eng.tag}" + (f"  (fallback: {eng.fallback_reason})" if eng.fallback_reason else ""))


def show(rows):
    df = pd.DataFrame(rows, columns=["entity", "chunk_text", "subconcept", "note"])
    t = time.time()
    r = eng.map(df[["entity", "chunk_text", "subconcept"]])
    dt = (time.time() - t) / len(df)
    for (_, q), (_, p) in zip(df.iterrows(), r.iterrows()):
        print(f"\n▶ {q.entity!r}   [{q.note}]")
        print(f"   context: {q.chunk_text[:110]}")
        for pre, name in (("icd10cm", "ICD-10-CM"), ("snomed", "SNOMED CT")):
            flag = "" if p[f"{pre}_confidence"] >= 0.5 else "   ⚠ low confidence → review"
            print(f"   {name:9s} {p[f'{pre}_code']:<16} {str(p[f'{pre}_description'])[:70]:<70} "
                  f"conf={p[f'{pre}_confidence']:.2f}{flag}")
            print(f"   {'':9s} alternatives: {p[f'{pre}_alternatives']}")
    print(f"\n{dt*1000:.0f} ms per term (batch of {len(df)})")


if args.interactive:
    print("enter: term || context   (empty line to quit)")
    while True:
        line = input("> ").strip()
        if not line:
            break
        term, _, ctx = line.partition("||")
        show([(term.strip(), ctx.strip(), "DiagnosisCode", "live input")])
else:
    show(HARD_CASES)
