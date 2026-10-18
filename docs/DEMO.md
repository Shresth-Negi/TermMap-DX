# Demo video script (2–3 min screen recording)

Setup before recording (one terminal, large font). Run the demo once before recording: the
first run builds the training-memory embedding cache (about 10 min on CPU), later runs start in
about 30 s.

```bash
cd atharv && source scripts/env.sh
# full system: GPU node with >= 31 GB free on the card
srun --gres=gpu:1 --cpus-per-task=16 --mem=64G --pty bash
source scripts/env.sh
python scripts/demo.py --tag _d3llm14e4     # prints "model set: ..." first
# no GPU: python scripts/demo.py --tag _d3ce4
```

The first line printed is the model set that actually ran. If the GPU lacks room for the LLM,
it reads e.g. `model set: _d3ce4 (fallback: _d3llm14e4: ...)`. That makes a good 5-second aside
on robustness.

## 0:00–0:40 Architecture (show `docs/figures/architecture.png`)

* "Terms arrive verbatim from CodEx with their chunk. We normalise them: markup, `(hcc)`, ICD
  codes written in the text, abbreviation variants from a lexicon mined from the SNOMED
  dictionary itself, negation and history cues."
* "Several independent retrievers propose candidates per system: exact match, character
  n-grams for OCR noise, word TF-IDF, three dense encoders, and the nearest **already-coded**
  training terms, which carry the house convention."
* "A LightGBM learning-to-rank model scores every candidate with every signal: about 85
  features, including laterality, stage numbers, specificity and semantic type. A second stage
  couples ICD and SNOMED and adds a fine-tuned cross-encoder and a local LLM that reads the
  shortlist plus five similar coded examples. Out comes one code per system, with a calibrated
  confidence and alternatives."

## 0:40–2:00 Live hard cases (verified output of the no-LLM set `_d3ce4`)

Narrate while `demo.py` prints:

* **Abbreviations.** "HTN" → I10 / *High blood pressure*. "sob" → R06.02 / *Dyspnea*. "RA" in
  a rheumatology note → M06.9 / *Rheumatoid arthritis*.
* **Nearest string is the wrong concept.** "pulmonary htn" → I27.20 *Pulmonary hypertension*,
  not I10.
* **OCR noise.** "chronickidney disease stag 3" → N18.30 / *Chronic kidney disease stage 3*.
* **Specificity.** "diabetes" → E11.9: the text supports nothing more specific. "type 2 dm
  with diabetic nephropathy" → the combination code E11.21. "left knee oa" → the
  laterality-specific M17.12 / *Osteoarthritis of left knee joint*.
* **Honest failures, flagged.** Point at the "⚠ low confidence → review" markers:
  * "h/o dvt": SNOMED gets *History of deep vein thrombosis*, but ICD picks acute DVT
    (I82.409, confidence 0.24) instead of Z86.718. The low confidence routes it to review.
  * "MS" in an echo note ("mitral valve area 0.9 cm²") still comes out as multiple sclerosis
    without the LLM. Mitral stenosis is the 2nd SNOMED alternative and rheumatic mitral
    stenosis (I05.0) the 4th ICD alternative, so the right answer is on the shortlist. **Before
    recording, check `outputs/demo_d3llm14e4.txt` (GPU run).** If the LLM set fixes it, show
    it as "the LLM reads the context"; if not, show it as a known limitation.

Then live input (interactive mode, same tag):

```bash
python scripts/demo.py --tag _d3llm14e4 --interactive
> afib w rvr || cardiology: afib w rvr, rate 140, started diltiazem drip
> sz d/o || neuro: known sz d/o on keppra
```

## 2:00–2:45 Confidence and error analysis

* "Confidence is isotonic-calibrated on out-of-fold predictions, so a workflow can abstain: on
  the most confident half of the terms the engine is right 88% of the time for ICD and 76% for
  SNOMED, against 71% / 58% on all terms." (Coverage table in
  `reports/error_analysis_d3llm14e4.md`.)
* Show the failure-mode table and the label-consistency table: "Near-identical training terms
  get the same SNOMED code only 76% of the time, so part of the remaining gap is annotation
  convention, not model error."
