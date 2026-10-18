# Reports

Generated from 5-fold cross-validation on `hackathon_train.csv` (folds keyed by
entity + chunk + subconcept, shared across both systems; training-memory features leave-one-out).

| file | what | regenerate with |
|---|---|---|
| `error_analysis_d3llm14e4.md` / `.json` | final model set: exact match, ICD partial credit, long tail, failure modes with sampled errors, accuracy at coverage, label-consistency ceiling | `python scripts/error_analysis.py --tag _d3llm14e4` |
| `cv/cv_report_<tag>.json` | per-set candidate recall and stage-1 / stage-2 CV top-1 for every row of the ablation table | `python scripts/train.py ... --tag <tag>` |
| `cv/config_d3llm14e4.json` | the final set's configuration (encoders, cross-encoder, LLM, fallback, stage choice) | written by `train.py` |

The per-row cross-validated predictions (`cv_predictions.csv`, 9,646 rows, joinable to the
training file by `train_row`) ship in the submission archive, not in git, so the repository holds
no bulk copy of the proprietary training data. The reports above quote a few dozen example terms,
as an error analysis requires.
