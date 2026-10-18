# Demo video script (2–3 min screen recording)

Setup before recording (one terminal, large font):

```bash
cd atharv && source scripts/env.sh
# GPU node for the full system (LLM stage):
srun --gres=gpu:1 --cpus-per-task=16 --mem=64G --pty bash
source scripts/env.sh
```

## 0:00–0:40 Architecture (show `docs/figures/architecture.png`)

* "Terms arrive verbatim from CodEx with their chunk. We normalise them: markup, `(hcc)`,
  ICD codes written in the text, abbreviation variants from a lexicon mined from the SNOMED
  dictionary itself, and negation or history cues."
* "Six independent retrievers propose candidates per system: exact match, character n-grams
  for OCR noise, word TF-IDF, three dense encoders, and the nearest **already-coded** training
  terms, which carry the house convention."
* "A LightGBM learning-to-rank model scores every candidate with every signal: about 80
  features, including laterality, stage numbers, specificity and semantic type. A second stage
  couples ICD and SNOMED, and adds a fine-tuned cross-encoder and a local LLM that sees the
  shortlist and five similar coded examples. Out comes one code per system, with a calibrated
  confidence and alternatives."

## 0:40–2:00 Live hard cases

```bash
python scripts/demo.py --tag _d3llm14      # GPU node: full system (LLM loaded automatically)
# or: python scripts/demo.py --tag _d3ce     # no GPU
```

Narrate while it prints:

* **Abbreviation collision.** "MS" in an echo note ("mitral valve area 0.9 cm²") vs "MS" in a
  neurology note: the same string gives two different codes, chosen from context.
  "RA" in a rheumatology note.
* **Nearest string is the wrong concept.** "pulmonary htn" → I27.20, not I10.
* **OCR noise.** "chronickidney disease stag 3" → N18.30 / CKD stage 3.
* **Specificity.** "diabetes" → E11.9 (unspecified: the text supports nothing more), while
  "type 2 dm with diabetic nephropathy" → the combination code E11.21. "left knee oa" →
  laterality-specific M17.12.
* **History.** "h/o dvt" → personal-history code.

Then live input:

```bash
python scripts/demo.py --tag _d3llm14 --interactive
> afib w rvr || cardiology: afib w rvr, rate 140, started diltiazem drip
> sz d/o || neuro: known sz d/o on keppra
```

## 2:00–2:45 Confidence and error analysis

* Point at the `conf=` column and the "⚠ low confidence → review" flag. "Confidence is
  isotonic-calibrated on out-of-fold predictions, so a workflow can abstain: the most confident
  70% of terms are mapped at higher accuracy than the full set" (show the coverage table in
  `outputs/error_analysis_<tag>.md`).
* Show the failure-mode table: retrieval misses vs specificity vs wrong concept, and the
  long-tail table. "Most of the remaining errors are convention calls between near-synonymous
  concepts, and codes never seen in training. That is where the LLM stage helps most."
