# TermMap-DX — clinical term → ICD-10-CM + SNOMED CT mapping engine

HiLabs Hackathon 2026. Given a clinical entity extracted from a chart and its
chunk of surrounding text, the engine assigns **one ICD-10-CM code and one
SNOMED CT concept**, plus a calibrated confidence and the top-5 alternatives for
each system. Everything runs locally: no chart text leaves the machine.

> Results, design rationale and error analysis: see [`docs/WRITEUP.md`](docs/WRITEUP.md)
> (also exported as `docs/WRITEUP.pdf`). Demo video: _link to be added_.

## Results (5-fold cross-validation on the labelled training set)

RESULTS_TABLE

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
| `TERMMAP_TAG` | `_d3llm` | model set for `run_all.sh` (see table below) |
| `TERMMAP_LLM` | `Qwen/Qwen2.5-7B-Instruct` | local LLM for the LLM stage |

## Run end to end (one command)

```bash
# full system on a GPU node (SLURM) - never on the login node
srun --gres=gpu:1 --cpus-per-task=32 --mem=160G bash scripts/run_all.sh data/raw/eval.csv outputs/predictions.csv
# CPU-only fallback (no GPU needed at all)
TERMMAP_TAG=_sap4 bash scripts/run_all.sh data/raw/eval.csv outputs/predictions.csv
```

| `TERMMAP_TAG` | components | hardware at inference |
|---|---|---|
| `_sap4` | lexical + SapBERT + training memory, stage-1 LambdaRank | CPU, ~0.2 s/term |
| `_d3` | + gte-base + Qwen3-Embedding signals | CPU ok |
| `_d3ce` | + fine-tuned cross-encoder over the shortlist | CPU ok, GPU faster |
| `_d3llm` (default) | + Qwen2.5-7B-Instruct listwise scores | GPU ≥ 24 GB |
| `_d3llm14` | + Qwen2.5-14B-Instruct instead of 7B | GPU ≥ 40 GB |

`run_all.sh` is idempotent: it builds the dense indexes, trains the rankers
(and cross-encoder / LLM training scores) only if the artifacts are missing, then
maps the input. The input may use the training layout
(`entity, chunk_text, coding_system, subconcept`) or the brief's layout
(`chunk_id, clinical_entity[, chunk_text]`); a missing `subconcept` is inferred
by a small classifier (82% CV accuracy). Output has, for **every row**,
`icd10cm_code`, `snomed_code`, descriptions, calibrated confidences, margins and
top-5 alternatives; when the input names a `coding_system` per row, `code`
holds that system's prediction. `predict.py --brief` keeps only the input
columns plus `icd10cm_code` / `snomed_code` (the deliverable schema).

Individual steps:

```bash
python scripts/build_index.py --models sapbert gte qwen3     # dictionary embeddings
python scripts/train.py --dense sapbert gte qwen3 --tag _d3  # CV report + models
python scripts/error_analysis.py --tag _d3                     # outputs/error_analysis_d3.md
python scripts/llm_score_train.py --tag _d3                    # GPU
python scripts/train.py --dense sapbert gte qwen3 --tag _d3llm --feats-tag _d3 \
       --llm $TERMMAP_ARTIFACTS_DIR/llm_scores_train_d3.parquet
python scripts/predict.py --input X.csv --output Y.csv --tag _d3llm --llm-model Qwen/Qwen2.5-7B-Instruct
python scripts/demo.py --tag _sap4 [--interactive]              # hard-case walkthrough
```

## Determinism

No sampling anywhere. LightGBM runs with `deterministic=True`, fixed seeds and
fixed thread counts; CV folds use `random_state=7`; the LLM is used only for a
single forward pass (argmax-free letter log-probs). Dense encoders run in fp16
on GPU / fp32 on CPU, so scores can differ in the last decimals across hardware;
predictions are stable on the same hardware.

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
