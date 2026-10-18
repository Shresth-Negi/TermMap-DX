---
license: apache-2.0
language: en
tags:
  - medical
  - clinical-coding
  - icd-10-cm
  - snomed-ct
  - entity-linking
  - learning-to-rank
---

# TermMap-DX: trained artifacts

Trained models for **TermMap-DX**, an engine that maps clinical terms extracted from charts to
one ICD-10-CM code and one SNOMED CT concept (HiLabs Hackathon 2026). The code lives in the team's
GitHub repository; `scripts/download_artifacts.py` there pulls these files into place, and
`scripts/run_all.sh` does so automatically before mapping an input file.

## Contents

| files | what |
|---|---|
| `ce_ICD10CM_d3e4/`, `ce_SNOMEDCT_US_d3e4/` | cross-encoders fine-tuned from `cambridgeltl/SapBERT-from-PubMedBERT-fulltext` (4 epochs, listwise loss over candidate shortlists) |
| `ranker_<system>_s1<set>.txt`, `ranker_<system>_s2<set>.txt` (+ `.json` feature lists) | LightGBM LambdaRank models, stage 1 and stage 2, per coding system |
| `calib_<system><set>.json` | isotonic calibration (ranker margin -> P(correct)) |
| `config<set>.json` | model-set configuration: encoders, cross-encoder, LLM, fallback set |
| `emb_<encoder>_<system>.npy` | float16 embeddings of every unique dictionary description (row = description id rebuilt by the code from the supplied dictionaries) for SapBERT, gte-base and Qwen3-Embedding-0.6B |
| `manifest.json` | sha256 and size of every file |

Model sets: `_d3llm14e4` (default; adds Qwen2.5-14B-Instruct scores at inference), `_d3llm7e4`
(Qwen2.5-7B-Instruct), `_d3ce4` (no LLM), `_sap4` (SapBERT only, lightest). The LLMs and encoders
are not included; they are pulled from their public Hugging Face repositories.

## Results (5-fold cross-validation on the training file)

| set | ICD-10-CM exact match | SNOMED CT exact match |
|---|---|---|
| `_d3llm14e4` | 0.708 | 0.582 |
| `_d3llm7e4` | 0.702 | 0.567 |
| `_d3ce4` | 0.684 | 0.543 |
| `_sap4` | 0.658 | 0.507 |

## Training data and licences

Trained on the HiLabs hackathon training file (proprietary; not included here). The supplied
ICD-10-CM and SNOMED CT US dictionaries define the answer space; the embedding matrices are
derived from their descriptions, so use is subject to the SNOMED CT licence terms that apply to
you. Base models: SapBERT (Apache-2.0), gte-base (MIT), Qwen3-Embedding-0.6B (Apache-2.0),
Qwen2.5-7B/14B-Instruct (Apache-2.0, used at inference only).
