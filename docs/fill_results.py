"""Fill the result placeholders of docs/WRITEUP.md and README.md from the JSON reports.

Usage: python docs/fill_results.py --final _d3llm --rows "Lexical only=_lexC" "+ SapBERT=_sap" ...
Writes docs/WRITEUP.md and README.md in place from docs/templates/*.md.
"""
import argparse
import json
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ART = Path(os.environ.get("TERMMAP_ARTIFACTS_DIR", ROOT / "artifacts"))
ap = argparse.ArgumentParser()
ap.add_argument("--final", required=True)
ap.add_argument("--rows", nargs="+", required=True, help="label=tag pairs, in ablation order")
ap.add_argument("--note", default="")
ap.add_argument("--llm-latency", default="measured in the GPU run")
args = ap.parse_args()


def cv(tag):
    r = json.loads((ART / f"cv_report{tag}.json").read_text())
    out = {}
    for s in ("ICD10CM", "SNOMEDCT_US"):
        x = r[s]
        out[s] = {"recall": x["candidate_recall"],
                  "top1": x.get("final_cv_top1", max(x["stage1_cv_top1"], x["stage2_cv_top1"])),
                  "dx": x.get("by_subconcept", {}).get("DiagnosisCode")}
    return out


ea = json.loads((ROOT / "outputs" / f"error_analysis{args.final}.json").read_text())
rows = [r.split("=", 1) for r in args.rows]
abl = ["| configuration | ICD-10-CM | SNOMED CT | headline (min) | SNOMED, diagnoses only | candidate recall ICD / SNOMED |",
       "|---|---|---|---|---|---|"]
for label, tag in rows:
    c = cv(tag)
    i, s = c["ICD10CM"]["top1"], c["SNOMEDCT_US"]["top1"]
    bold = "**" if tag == args.final else ""
    dx = c["SNOMEDCT_US"]["dx"]
    abl.append(f"| {bold}{label}{bold} | {bold}{i:.3f}{bold} | {bold}{s:.3f}{bold} | {bold}{min(i, s):.3f}{bold} | "
               f"{dx:.3f} | {c['ICD10CM']['recall']:.3f} / {c['SNOMEDCT_US']['recall']:.3f} |")
fin = cv(args.final)
I, S = ea["ICD10CM"], ea["SNOMEDCT_US"]
summary = (f"| metric | ICD-10-CM | SNOMED CT |\n|---|---|---|\n"
           f"| exact match (5-fold CV) | **{I['exact_match']:.3f}** | **{S['exact_match']:.3f}** |\n"
           f"| long tail: gold code seen ≤ 2× elsewhere in train | {I['long_tail_acc_unseen_or_rare']:.3f} | {S['long_tail_acc_unseen_or_rare']:.3f} |\n"
           f"| accuracy on the 70% most confident | {I['accuracy_at_coverage']['70%']:.3f} | {S['accuracy_at_coverage']['70%']:.3f} |\n"
           f"| same 3-character category (ICD partial credit) | {I['same_3char_category']:.3f} | – |\n"
           f"| label self-consistency on near-duplicates (ceiling) | {I['label_consistency']['near_agree']:.3f} | {S['label_consistency']['near_agree']:.3f} |")

modes = {}
for sname, r in (("ICD-10-CM", I), ("SNOMED CT", S)):
    for k, v in r["failure_modes"].items():
        modes.setdefault(k, {})[sname] = v["share_of_rows"]
ft = ["| failure mode | ICD-10-CM | SNOMED CT |", "|---|---|---|"]
def _fmt(v, key):
    return f"{v[key]:.3f}" if key in v else "–"


for k, v in sorted(modes.items(), key=lambda kv: -max(kv[1].values())):
    ft.append(f"| {k} | {_fmt(v, 'ICD-10-CM')} | {_fmt(v, 'SNOMED CT')} |")
lt = ["| gold code seen elsewhere in training | ICD-10-CM acc (n) | SNOMED CT acc (n) |", "|---|---|---|"]
for k in I["long_tail"]:
    a, b = I["long_tail"][k], S["long_tail"][k]
    lt.append(f"| {k} | {a['acc']:.3f} ({a['n']}) | {b['acc']:.3f} ({b['n']}) |")
cov = ["| coverage (most confident first) | ICD-10-CM | SNOMED CT |", "|---|---|---|"]
for k in I["accuracy_at_coverage"]:
    cov.append(f"| {k} | {I['accuracy_at_coverage'][k]:.3f} | {S['accuracy_at_coverage'][k]:.3f} |")
sub = S["by_subconcept"]
subline = ", ".join(f"{k} {v['mean']:.3f} (n={int(v['size'])})" for k, v in sub.items() if v["size"] > 5)
subline = (f"{subline}. Diagnoses are the easiest SNOMED slice; procedures, devices and labs carry more "
           f"granularity conventions. ICD-10-CM: {I['by_subconcept'].get('DiagnosisCode', {}).get('mean', float('nan')):.3f} on diagnoses.")

results = (f"## 4. Results\n\nAll numbers are 5-fold CV exact match on the 9,646 labelled rows. {args.note}\n\n"
           f"**Headline (configuration `{args.final}`):** ICD-10-CM **{fin['ICD10CM']['top1']:.3f}**, "
           f"SNOMED CT **{fin['SNOMEDCT_US']['top1']:.3f}** → min **{min(fin['ICD10CM']['top1'], fin['SNOMEDCT_US']['top1']):.3f}**.\n\n"
           + summary + "\n\n**Ablation** (each row adds one component; the configuration in bold is the submitted one):\n\n"
           + "\n".join(abl) + "\n\n![ablation](figures/ablation.png)\n")

tpl = (ROOT / "docs" / "templates" / "WRITEUP.md").read_text()
tpl = (tpl.replace("RESULTS_SECTION", results).replace("FAILURE_TABLE", "\n".join(ft))
       .replace("LONGTAIL_TABLE", "\n".join(lt)).replace("COVERAGE_TABLE", "\n".join(cov))
       .replace("SUBCONCEPT_LINE", subline).replace("LLM_LATENCY", args.llm_latency)
       .replace("ICD_CAT", f"{I['same_3char_category']:.0%}").replace("ICD_CHAP", f"{I['same_chapter_letter']:.0%}"))
(ROOT / "docs" / "WRITEUP.md").write_text(tpl)
rt = (ROOT / "docs" / "templates" / "README.md").read_text()
(ROOT / "README.md").write_text(rt.replace("RESULTS_TABLE", summary + "\n\n" + "\n".join(abl)))
print("filled with", args.final)
