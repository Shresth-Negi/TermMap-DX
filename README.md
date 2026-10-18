# TermMap-DX — clinical term → ICD-10-CM + SNOMED CT mapping engine

HiLabs Hackathon 2026. Given a clinical entity extracted from a chart and its
chunk of surrounding text, the engine assigns **one ICD-10-CM code and one
SNOMED CT concept**, plus a calibrated confidence and the top-5 alternatives for
each system. Everything runs locally: no chart text leaves the machine.

> **Short write-up (2 pages):** [`docs/SUMMARY.pdf`](docs/SUMMARY.pdf) ·
> **Full documentation:** [`docs/WRITEUP.pdf`](docs/WRITEUP.pdf) (architecture, protocol, error analysis,
> innovations, resources) · **Demo script:** [`docs/DEMO.md`](docs/DEMO.md) · Demo video: _link to be added_.

## Results (5-fold cross-validation on the labelled training set)

| metric | ICD-10-CM | SNOMED CT |
|---|---|---|
| exact match (5-fold CV) | **0.708** | **0.582** |
| long tail: gold code seen ≤ 2× elsewhere in train | 0.587 | 0.518 |
| accuracy on the 70% most confident | 0.826 | 0.694 |
| same 3-character category (ICD partial credit) | 0.812 | – |
| label self-consistency on near-duplicates (ceiling) | 0.937 | 0.758 |

| configuration | ICD-10-CM | SNOMED CT | headline (min) | SNOMED, diagnoses only | candidate recall ICD / SNOMED |
|---|---|---|---|---|---|
| Lexical retrieval + ranker | 0.569 | 0.440 | 0.440 | 0.531 | 0.835 / 0.797 |
| + SapBERT dense retrieval | 0.655 | 0.500 | 0.500 | 0.598 | 0.922 / 0.878 |
| + context / negation / numeric features | 0.653 | 0.504 | 0.504 | 0.600 | 0.922 / 0.878 |
| + contextual kNN, wider candidate pools | 0.655 | 0.507 | 0.507 | 0.604 | 0.929 / 0.889 |
| + official ICD titles (fully CPU set) | 0.658 | 0.507 | 0.507 | 0.604 | 0.929 / 0.889 |
| + gte-base + Qwen3-Embedding signals | 0.666 | 0.515 | 0.515 | 0.614 | 0.942 / 0.915 |
| + fine-tuned cross-encoder, 2 epochs | 0.683 | 0.535 | 0.535 | 0.631 | 0.942 / 0.915 |
| cross-encoder trained 4 epochs (no-LLM set) | 0.684 | 0.543 | 0.543 | 0.638 | 0.942 / 0.915 |
| + Qwen2.5-7B listwise LLM scores | 0.702 | 0.567 | 0.567 | 0.656 | 0.942 / 0.915 |
| **Qwen2.5-14B instead of 7B (default)** | **0.708** | **0.582** | **0.582** | 0.669 | 0.942 / 0.915 |

The headline number is the lower of the two systems, as in the brief. The
long-tail column covers rows whose gold code appears at most twice elsewhere
in training.

## How it works (one paragraph)

```
entity + chunk ──► normalise (html, "(hcc)", ICD codes in text, abbreviation expansion,
                   SNOMED-synonym bridge for ICD)
               ──► candidate generation, per system (union, ~60-120 codes)
                     exact key · char 3-4gram TF-IDF · word TF-IDF · SapBERT · gte-base · Qwen3-Embedding
                     · nearest labelled training terms (local coding convention)
               ──► ~70 features per (term, candidate): every retriever's score for every
                   candidate, fuzzy-string, token coverage, laterality, numbers/stage, negation,
                   history, specificity flags (billable / unspecified / other / 7th char),
                   semantic tag × entity type, code priors, neighbour votes
               ──► Stage 1: LightGBM LambdaRank per system  ──► top-20 shortlist
               ──► Stage 2: + cross-system coupling (ICD↔SNOMED co-occurrence and
                   description similarity of the other system's shortlist)
                   + optional local LLM listwise scores (Qwen2.5-7B-Instruct,
                   retrieval-augmented with the 5 nearest coded training terms)
               ──► LightGBM LambdaRank per system ──► code, calibrated confidence, top-5
```

## Repository layout

| path | what |
|---|---|
| `src/termmap/normalize.py`, `abbrev.py` | cleaning; abbreviation lexicon mined from the SNOMED dictionary (`ABBR - Expansion` descriptions) + curated chart shorthand |
| `src/termmap/dictionaries.py`, `corpus.py` | load the supplied dictionaries, build per-system retrieval corpora |
| `src/termmap/lexical.py`, `encoders.py`, `resources.py` | char/word TF-IDF indexes, dense encoders, per-system lookups |
| `src/termmap/features.py` | candidate generation + feature extraction |
| `src/termmap/pipeline.py` | keys, leave-one-out memory, shortlist, cross-system coupler, SNOMED bridge |
| `src/termmap/ranker.py` | LightGBM LambdaRank wrapper |
| `src/termmap/llm_scorer.py` | local LLM listwise scorer (one forward pass, letter log-probs) |
| `src/termmap/engine.py` | inference engine used by `predict.py` and the demo |
| `scripts/build_index.py` | embed dictionary descriptions |
| `scripts/train.py` | out-of-fold features, 5-fold CV, final models, calibration |
| `scripts/llm_score_train.py` | LLM scores for the out-of-fold training shortlists (GPU) |
| `scripts/predict.py` | map an input file |
| `scripts/error_analysis.py` | breakdown by failure mode, long tail, coverage |
| `scripts/demo.py` | interactive / scripted demo with hard cases |
| `scripts/run_all.sh` | **single command**: build missing pieces and predict |
| `scripts/train_cross_encoder.py` | fine-tune the shortlist cross-encoder (5-fold OOF + final) |
| `scripts/export_oof.py` | cross-validated predictions for every training row |
| `tests/` | unit tests (pytest) |
| `slurm/*.sbatch` | SLURM wrappers (GPU work never runs on the login node) |

## Setup

Requirements: Linux, Python 3.13, ~25 GB of fast disk for model weights and
indexes. A CUDA GPU (≥24 GB) is needed only for the LLM stage and makes
index building ~50× faster; everything else runs on CPU.

```bash
# 1. choose a work dir on fast local disk for venv / HF cache / artifacts
export TERMMAP_WORK=/path/to/fast/disk/termmap_work      # see .env.example
python3.13 -m venv $TERMMAP_WORK/venv
$TERMMAP_WORK/venv/bin/pip install -r requirements.txt
# 2. data: put the three challenge files in data/raw/ (not committed - proprietary)
#    data/raw/hackathon_train.csv, data/raw/icd10cm_codes.xlsx,
#    data/raw/SNOMEDCT_US_codes_and_descriptions.xlsx
# 3. environment for every shell (venv, caches, PYTHONPATH; hides GPUs outside SLURM)
source scripts/env.sh
```

### Environment variables

No API keys are needed. See `.env.example`:

| variable | default | meaning |
|---|---|---|
| `TERMMAP_WORK` | scratch dir | venv, Hugging Face cache, artifacts |
| `TERMMAP_ARTIFACTS_DIR` | `$TERMMAP_WORK/artifacts` | indexes, features, models |
| `TERMMAP_RAW_DIR` | `data/raw` | challenge files |
| `TERMMAP_TAG` | `_d3llm14e4` | model set for `run_all.sh` (see table below) |

## Run end to end (one command)

```bash
# full system on a GPU node (SLURM) - never on the login node
srun --gres=gpu:1 --cpus-per-task=32 --mem=160G bash scripts/run_all.sh data/raw/eval.csv outputs/predictions.csv
# no GPU at inference (indexes / cross-encoder are trained once on a GPU)
TERMMAP_TAG=_d3ce4 bash scripts/run_all.sh data/raw/eval.csv outputs/predictions.csv
# fastest, fully CPU end to end
TERMMAP_TAG=_sap4 bash scripts/run_all.sh data/raw/eval.csv outputs/predictions.csv
```

| `TERMMAP_TAG` | components | CV exact match ICD / SNOMED | inference hardware |
|---|---|---|---|
| `_sap4` | lexical + SapBERT + training memory, LambdaRank | 0.658 / 0.507 | CPU, 50 ms/term |
| `_d3ce4` | + gte-base, Qwen3-Embedding, fine-tuned cross-encoder | 0.684 / 0.543 | CPU, ~0.7 s/term |
| `_d3llm7e4` | + Qwen2.5-7B-Instruct listwise scores | 0.702 / 0.567 | GPU, ≥ 18 GB free |
| `_d3llm14e4` (default) | + Qwen2.5-14B-Instruct instead of 7B | **0.708 / 0.582** | GPU, ≥ 31 GB free |

**Automatic fallback.** GPUs on a shared node are often partly used by other processes.
Before loading the LLM, the engine checks the free GPU memory against the model's weight size.
If the LLM cannot run (no GPU, not enough free memory, out of memory while loading), it falls
back to the next set: `_d3llm14e4 → _d3llm7e4 → _d3ce4`. It prints which set ran and why.
Every set is trained on the same folds, so a fallback costs accuracy but never breaks a run.

`run_all.sh` is idempotent: for the chosen set and its fallbacks it builds the dense indexes,
trains the rankers, cross-encoder and LLM training scores only if the artifacts are missing,
then maps the input. The input may use the training layout
(`entity, chunk_text, coding_system, subconcept`) or the brief's layout
(`chunk_id, clinical_entity[, chunk_text]`); a missing `subconcept` is inferred
by a small classifier (82% CV accuracy). Output has, for **every row**,
`icd10cm_code`, `snomed_code`, descriptions, calibrated confidences, margins and
top-5 alternatives; when the input names a `coding_system` per row, `code`
holds that system's prediction. `predict.py --brief` keeps only the input
columns plus `icd10cm_code` / `snomed_code` (the deliverable schema).

Individual steps (`slurm/gpu_pipeline.sbatch` runs the full GPU sequence):

```bash
python scripts/build_index.py --models sapbert gte qwen3             # dictionary embeddings
python scripts/train.py --dense sapbert gte qwen3 --tag _d3          # stage-1 features, CV, models
python scripts/train_cross_encoder.py --tag _d3 --out-tag _d3e4 --epochs 4   # GPU
python scripts/llm_score_train.py --tag _d3 --model Qwen/Qwen2.5-14B-Instruct --out-suffix _q14  # GPU
python scripts/train.py --dense sapbert gte qwen3 --tag _d3llm14e4 --feats-tag _d3 \
       --extra $TERMMAP_ARTIFACTS_DIR/ce_scores_train_d3e4.parquet \
       --llm $TERMMAP_ARTIFACTS_DIR/llm_scores_train_d3_q14.parquet \
       --llm-model-id Qwen/Qwen2.5-14B-Instruct --fallback-tag _d3llm7e4
python scripts/error_analysis.py --tag _d3llm14e4                    # outputs/error_analysis_d3llm14e4.md
python scripts/export_oof.py --tag _d3llm14e4                        # cross-validated predictions
python scripts/predict.py --input X.csv --output Y.csv --tag _d3llm14e4
python scripts/demo.py --tag _d3llm14e4 [--interactive]              # hard-case walkthrough
```

## Tests

```bash
source scripts/env.sh && $PY -m pytest -q tests     # unit tests for the deterministic components (~2 s)
```

## Determinism

No sampling anywhere. LightGBM runs with `deterministic=True`, fixed seeds and fixed thread
counts; CV folds use `random_state=7`; the LLM is used for a single forward pass whose letter
log-probabilities are read directly (no generation). Query-side encoders always run in fp32.
Training-memory embeddings are cached on disk per model, device and precision, and computed
outside the query cache. So **the same input on the same hardware gives byte-identical output,
on the first run and every run after** (checked with `cmp` on the sample file).

Two caveats, stated plainly. Scores of a term can move by about 1e-7 depending on which other
terms share its encoder batch, which in principle could flip an exact tie. CPU vs GPU, or a
different thread count, can change the last decimals. The cross-encoder and the LLM run in
bf16 on GPU, where these batch effects are larger (~1e-3).

## External resources (all open, local)

| resource | use | licence |
|---|---|---|
| supplied ICD-10-CM and SNOMED CT US dictionaries | answer space, retrieval corpora, abbreviation lexicon | challenge-provided |
| `cambridgeltl/SapBERT-from-PubMedBERT-fulltext` | dense retrieval + features | MIT |
| `thenlper/gte-base` | dense features | MIT |
| `Qwen/Qwen3-Embedding-0.6B` | dense features | Apache-2.0 |
| `Qwen/Qwen2.5-7B-Instruct` | LLM listwise scoring (optional stage) | Apache-2.0 |
| LightGBM, scikit-learn, rapidfuzz, sparse_dot_topn | ranking / retrieval | MIT / BSD |

The official SNOMED→ICD-10-CM map and UMLS were **not** used (licence-gated);
the cross-system prior is learned from training pairs instead.

## Data handling

The challenge data is proprietary: `data/raw/`, every `*.csv` / `*.xlsx`, and
all generated artifacts are git-ignored. Nothing is sent to any external API.
