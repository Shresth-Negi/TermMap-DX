# TermMap-DX: mapping extracted clinical terms to ICD-10-CM and SNOMED CT

HiLabs Hackathon 2026 · documentation and write-up

## 1. How we read the problem

The supplied training file differs from the schema in the brief, and that changes the problem:

* **Long format, one row per (entity, chunk, coding system).** 9,646 rows: 2,306 ICD-10-CM and
  7,340 SNOMED CT. ICD rows are diagnoses only. SNOMED rows also cover procedures (2,777), labs
  (858), encounter/social items (567) and devices (437). 2,123 diagnosis terms are labelled in
  both systems.
* **Every entity string occurs exactly once per system.** The data is deduplicated by term, so
  the held-out set will consist of *unseen strings*. Memorising terms is worth nothing; the
  engine has to generalise.
* **The long tail is the bulk of the data.** 54% of gold ICD codes and 68% of gold SNOMED
  concepts occur exactly once. For 20% of ICD rows and 40% of SNOMED rows, the gold code
  appears nowhere else in training.
* **Gold labels encode a house convention.** Examples: "gait is slightly unstable" → R26.81, and
  "abnormal movement" → R29.898 (an "other symptoms" residual code). 98% of gold ICD codes are
  billable leaves; for SNOMED, 92% of gold `display` strings are the concept's preferred term.
  Some labels are arguably noisy ("benign prostatic hyperplasia" → *Benign fibroma of
  prostate*). Inter-annotator agreement is not published; §5 estimates the practical ceiling from near-duplicate terms.
* **Exact dictionary matching barely helps.** Only 4.3% of ICD and 18.9% of SNOMED diagnosis
  terms match a description of their gold code exactly after normalisation.

Scope: the engine maps **every** input row to one ICD-10-CM code *and* one SNOMED CT concept.
Both come from the supplied dictionaries, which define the legal answer space. Every prediction
carries a calibrated confidence and top-5 alternatives, so a downstream workflow can abstain or
route low-confidence terms to review.

## 2. Architecture

![architecture](figures/architecture.png)

**Normalisation.** Unescape HTML and strip markup (`<br>`), EHR artefacts (`(hcc)`, `cms-hcc`),
bullets and list numbering. ICD codes written in the text are detected ("primary oa left knee
m17.12") and become candidates. Abbreviation expansion produces *query variants*; it never
replaces the original string. The lexicon has two sources: 7.7k `ABBR - Expansion` descriptions
mined from the supplied SNOMED file ("AR - Aortic regurgitation", "HV - Hallux valgus") and a short curated list of chart shorthand (HTN, LE, bilat, s/p…).
Ambiguous abbreviations (MS, RA, PID…) keep every reading as a separate variant. Each reading
is scored against the context with the mention removed, and every candidate inherits the score
of the reading that explains it best. Negation, history and family-history cues are read from
the 60 characters before the mention; history cues are also read inside it ("h/o dvt").

**SNOMED synonym bridge for ICD.** The SNOMED dictionary has six times as many descriptions as
ICD (985k vs 159k). For ICD queries, the preferred terms of the top SNOMED concepts (dense match)
are added as extra query variants. So "hepatic enlargement" also searches ICD as "hepatomegaly".

**Candidate generation (recall).** Per system, we take the union of: exact normalised-key match;
character 3–4-gram TF-IDF (robust to OCR noise such as "chronickidney disease stag 3"); word
TF-IDF (rewards rare exact tokens); dense retrieval with SapBERT, plus gte-base and
Qwen3-Embedding in the GPU configuration; and the **nearest labelled training terms**, whose gold
codes carry the house convention. All retrieval runs over unique normalised descriptions: 158k
for ICD and 624k for SNOMED, after stripping the FSN semantic tag.

**Stage-1 ranker.** Every candidate is scored by *every* signal, not only by the retriever that
proposed it. That gives ~85 features: max/preferred-term cosine per encoder (original and
expanded query), char/word TF-IDF, fuzzy ratios, query/description token coverage, numbers and
stage agreement ("stage 2" vs N18.30), laterality match/conflict/missing, negation and history
agreement, and context overlap. For ICD it adds billable flag, code depth, "unspecified"/"other"
wording, 7th-character encounter type and chapter. For SNOMED it adds semantic tag (crossed with
the entity's subconcept) and description order. Training-memory features include neighbour votes
at code and, for ICD, 3-character-category level (explicit specificity calibration). There is
also a *contextual* neighbour vote, entity similarity × context similarity, so that a neighbour
from a different clinical setting counts less. Code and category priors complete the set. A LightGBM LambdaRank model per system ranks the candidates.

**Stage-2 ranker.** A second LambdaRank model works on the top-20 shortlist. It uses stage-1
score, rank and gap, plus **cross-system coupling**: the ICD and SNOMED predictions for the same
term inform each other through (a) leave-one-out co-occurrence P(ICD | SNOMED) learned from the
2,123 doubly-labelled terms, and (b) dense similarity between each candidate and the other
system's shortlist, weighted by its probabilities. In the GPU configuration, stage 2 also gets
**LLM listwise scores**. A local Qwen2.5-Instruct model sees the term, its context, its entity
type, the five most similar *already-coded* training terms with their codes, and the top-10
candidates as lettered options. One forward pass gives the log-probability of each option
letter: no generation, no sampling, deterministic. A fine-tuned **cross-encoder** (SapBERT
initialisation, listwise loss over the shortlist, 5-fold out-of-fold) adds a learned
semantic-match score. Both models see SNOMED options with their semantic tag
("International normalized ratio [observable entity]").

**Stage selection and calibration.** Per system, stage 2 is used only if it beats stage 1 out of
fold; without new evidence (no cross-encoder or LLM) it does not, and stage 1 is kept. Isotonic
regression on the out-of-fold margin of the selected stage turns ranker scores into P(correct),
which is reported as `confidence`.

## 3. Evaluation protocol (no leakage)

All numbers are **5-fold cross-validation** on the training file. Folds are keyed by
(entity, chunk, subconcept) and shared by both systems, so cross-system features never see the
other system's label for the same term. Training-memory features (neighbour votes, priors,
co-occurrence) are **leave-one-out**: a term is never scored against its own label. Stage-1
scores feeding stage 2 are out-of-fold. Early stopping uses an inner fold, never the evaluation
fold. Exact-match accuracy is reported per system; the headline is the lower of the two.

RESULTS_SECTION

### Latency and footprint

Measured on the shared node (AMD EPYC 9554, 16 threads, no GPU) for the CPU model set `_sap4`,
mapping both systems per term: **50 ms/term in batch (≈72k terms/hour)**, single-term latency
331 ms median / 372 ms p95, 64 s cold start, 16 GB RAM (dense indexes held as float32). The
cross-encoder adds ~1.5 ms/term on a GPU. The LLM stage costs one ~700-token forward pass per
term per system (LLM_LATENCY). It is the only component
that needs a GPU at inference, and it can be limited to low-confidence terms.

## 5. Error analysis

Numbers in this section are from out-of-fold predictions of the configuration in the headline
row (`outputs/error_analysis_<tag>.md` has the full tables and sampled errors per failure mode).

**The labels set a ceiling well below 95%.** We measured how consistently the gold labels treat
near-identical terms. Training pairs that are identical after normalising case, punctuation,
plurals and word order get the *same* gold code only **92.5% of the time in ICD-10-CM and
66.8% in SNOMED CT**. For nearest-neighbour pairs with character cosine ≥ 0.95 the figures are
93.7% and 75.8%. Example: "diabetes mellitus type 2" → *Type 2 diabetes mellitus*, but
"type 2 diabetes mellitus with other specified complication…" → *Disorder associated with
diabetes mellitus*, and the three "icd-10-cm" variants map to two different reference sets.
Under these labels, any deterministic system is capped at roughly 75% SNOMED exact match. Our
SNOMED gap is therefore half annotation convention, half model error. **Closing it needs
convention consolidation as much as modelling** (see §10).

**Failure modes** (share of all rows):

FAILURE_TABLE

* *Retrieval misses* are mostly paraphrases with no lexical overlap and a convention target:
  "could not df her ankle well" → R29.898, "hemodynamic instability" → R09.89. They are also
  colloquialisms ("belly is swollen" → R19.00).
* *Specificity* errors (ICD, same 3-character category) are dominated by residual-code
  conventions: "nasal dyspnea" → R06.89 (other) vs R06.00 (unspecified); "lymphadenopathy" →
  R59.1 (generalised) vs R59.9; "alcohol dependence" → F10.20 vs F10.29. The predicted code is in the right
  3-character category ICD_CAT of the time (right chapter ICD_CHAP); the leaf choice is where
  convention bites.
* *Semantic-type mismatch* (SNOMED) is the largest SNOMED-specific failure. The same name
  exists as a procedure, an observable and a finding ("inr", "full mask cpap", "acute rash"
  → *Acute eruption of skin* vs *Eruption*). The entity's subconcept helps, but the training
  convention is not uniform.
* *Laterality* errors are rare (≈1% of rows).

**Long tail.** Accuracy falls with how often the gold code was seen elsewhere in training:

LONGTAIL_TABLE

Unseen-code accuracy is about half the frequent-code accuracy in both systems. The engine still
maps a third of never-seen codes exactly, purely from dictionary retrieval plus learned
convention features. This is the number that best predicts production behaviour.

**By entity type (SNOMED).** SUBCONCEPT_LINE

**Abbreviation collisions.** Bare ambiguous abbreviations are only 182 training rows (1.9%).
ICD gets 75% of them right, SNOMED 46%. The context-conditioned reading score identifies the
right reading (e.g. "MS" + "mitral valve area" → mitral stenosis 0.46 vs multiple sclerosis
0.27), but with so few collision examples in training the ranker under-uses it. When a
neighbour shares the exact abbreviation ("ra" → *Breathing room air* from "hypoxic on ra"), the
house convention wins over context. These cases come out with low calibrated confidence and are
routed to review. The LLM stage, which reads the context, is designed to resolve them.

**Calibrated abstention.** Confidence is isotonic-calibrated on out-of-fold margins:

COVERAGE_TABLE


## 6. Design decisions and why

* **Learning-to-rank over hand-tuned score fusion.** No single signal wins. On SNOMED
  shortlists, SapBERT alone picks the gold concept 53% of the time, gte-base 52% and the
  lexical ranker 61%. The ranker learns *when* each signal is trustworthy (abbreviation,
  long phrase, procedure vs finding).
* **The training memory is a retriever, not a lookup.** Because every test string is new,
  labelled neighbours are used as *evidence* (similarity-weighted votes for a code and its
  category), so the convention ("other symptoms" residual codes, preferred SNOMED granularity)
  transfers to paraphrases.
* **Specificity as an explicit feature set.** "Unspecified"/"other" wording, billable leaf,
  code depth, 7th character, numeric and laterality agreement, and *category-level* neighbour
  votes let the ranker separate "right category, wrong leaf" from "wrong concept". It learns
  from data when the text supports a specific code and when E11.9-style unspecified codes are
  correct.
* **Both systems predicted independently, then coupled.** Each system has its own retrieval
  and ranker, so neither is a crosswalk of the other (the brief warns against that). Coupling
  enters only as stage-2 evidence.
* **LLM only where judgement is needed, and only as a feature.** The LLM never sees the whole
  vocabulary; it re-scores a 10-option shortlist in a single forward pass, and the ranker
  decides how much to trust it. Cost: about one 700-token forward pass per term per system.
* **Everything local.** The data is proprietary; no API calls are made, and all models are
  open-weight and named below.

## 7. What we tried and abandoned

* **Using the client's own index model as an oracle.** The SNOMED export notes say it came from
  a `gte_base` semantic index, so we tested whether gold labels follow gte-base's nearest
  neighbour. They do not (52% on shortlists vs 61% for our ranker). gte-base is kept as one
  feature, not as the decider.
* **BioLORD-2023** as an encoder. Dropped because its licence requires UMLS and SNOMED CT
  licences; replaced with Qwen3-Embedding-0.6B (Apache-2.0).
* **Stage 2 without new evidence.** Re-ranking stage-1 scores with only coupling features gives
  ±0.3 pt, which is within noise. Stage 2 earns its place only with the LLM scores.
* **Larger shortlists for stage 2.** Going from 12 to 20 candidates recovers 2–4 pt of recall
  without hurting top-1.
* **Ranker hyper-parameter search.** Six LightGBM variants (learning rate / leaves, XE-NDCG,
  binary objective, stronger regularisation) and rank-averaged ensembles of objectives/seeds all
  landed within ±0.5 pt of the defaults. The
  model is limited by its evidence, not its capacity, so we kept the defaults.
* **A diagnosis-only SNOMED ranker.** Training only on diagnosis rows gave 60.0% on
  diagnoses vs 60.4% when trained on all entity types: procedures, labs and devices transfer
  useful convention.
* **Context alone for abbreviation collisions.** The context-conditioned reading score is
  correct on the showcase cases but barely moves CV (only 1.9% of rows are bare ambiguous
  abbreviations), so it is kept as a feature and the LLM stage handles the rest.
* **Official SNOMED→ICD map / UMLS.** Not used (licence-gated, and the brief notes the map
  cannot be ground truth). A data-driven co-occurrence prior replaces it.

## 8. Innovations

* **A measured label ceiling.** Self-agreement of the gold labels on near-duplicate terms
  (§5) turns "95%" from a target into a diagnosable gap: it separates annotation convention
  from model error, and it lists the conflicting groups for adjudication.
* **The dictionary teaches its own abbreviations.** 7.7k `ABBR - Expansion` descriptions in the
  supplied SNOMED export become the abbreviation lexicon. There is no external resource and it
  is versioned with the dictionary.
* **SNOMED as a thesaurus for ICD.** SNOMED preferred terms of the best dense matches become
  ICD query variants, bridging lay or clinical phrasing to ICD's terse titles.
* **Training memory as a retriever with hierarchy-aware votes.** Leave-one-out neighbour
  votes at the code *and* the ICD 3-character-category level make specificity an explicit,
  learned decision. A contextual variant (entity × context similarity) down-weights
  neighbours from a different clinical setting.
* **Coupled, not crosswalked.** Each system is predicted on its own. Cross-system evidence
  (leave-one-out co-occurrence, shortlist-to-shortlist similarity) only enters stage 2.
* **Deterministic LLM judgement.** One forward pass, letter log-probabilities over a
  10-option shortlist, with the five nearest *coded* examples in the prompt so the model sees
  the house convention. No sampling, no parsing, and a calibrated ranker decides how much to
  trust it.
* **Abstention built in.** Out-of-fold isotonic calibration plus CV-based stage selection give
  each prediction a usable P(correct); the coverage table in §5 shows the trade-off.

## 9. Challenges and learnings

* **The data did not match the brief.** It is long-format with one row per system, includes
  non-diagnosis SNOMED entity types, and has no released evaluation input. We built the engine
  to accept both layouts and to always emit both systems.
* **Compute constraints shaped the design.** On a shared node every GPU was allocated for
  hours, the QOS allowed one job per user, and the home quota was full. The pipeline was
  therefore built CPU-first, with every GPU step an optional, cached add-on that runs
  unattended under SLURM. A useful side effect is a ladder of model sets: `_sap4` is fully CPU, `_d3ce` (68.3% / 53.5%) needs
  a GPU only to build its indexes once, and only the LLM sets need a GPU at inference.
* **Evidence beats capacity.** Hyper-parameter search, ensembles and specialised rankers moved
  nothing (±0.5 pt). Every real gain came from a new, independent signal: dense retrieval,
  then the cross-encoder, then the LLM.
* **Stage 2 must earn its place.** Without new evidence a second ranker slightly *hurt*. That
  is why stage selection is decided per system on out-of-fold accuracy.

## 10. Plan for the remaining gap

1. **Consolidate the convention, then measure against it.** With SNOMED self-agreement at
   67–76% on near-duplicates, the most valuable next step is adjudicating the ~250 conflicting
   near-duplicate groups we surface automatically (`error_analysis.py`) and writing a short
   coding guideline: preferred granularity, residual "other" vs "unspecified", and procedure vs
   observable for labs. Re-labelling those groups also raises the achievable ceiling.
2. **SNOMED hierarchy.** The supplied export has no IS-A relationships. With the RF2 release
   (licensed via UMLS/NLM), we would add ancestor/descendant features and hierarchical partial
   credit, and turn "right concept, wrong granularity" into an explicit choice, as we already do
   for ICD categories.
3. **Contrastively fine-tuned retriever.** Fine-tune SapBERT on gold (term, description) pairs
   with in-batch hard negatives from our shortlists. That targets the retrieval misses
   (10% ICD, 17% SNOMED), which no reranker can recover.
4. **LLM only where it pays.** Route the least-confident ~30% of terms to a larger local
   model with the full shortlist and retrieved examples. The other 70% are already at
   75–80% accuracy and cost ~0.2 s/term on CPU.
5. **One title per ICD code everywhere.** The supplied ICD file mixes official titles, short
   titles and index terms per code. The output uses a refined display rule (85% agreement with
   the gold display), but the cross-encoder and LLM were trained on an earlier rule (79%).
   Retraining them on the refined titles is a one-command GPU job.
6. **Version drift.** Dictionaries are inputs, not code. A new ICD-10-CM (October) or SNOMED
   (biannual) release only needs `build_index.py` plus `train.py` (≈20 min on one GPU). Codes
   deleted in a release can be remapped with the CMS GEM / conversion tables before retraining,
   and training labels that point at retired codes are flagged by the dictionary check that
   already exists (two gold ICD codes in the training file, e.g. `L6.0`, are not in the
   dictionary).


## 11. External resources

| resource | version / id | use | licence |
|---|---|---|---|
| ICD-10-CM codes & descriptions | challenge file `icd10cm_codes.xlsx` (159,306 rows) | answer space, corpus | provided |
| SNOMED CT US codes & descriptions | challenge file (984,585 descriptions, 382,124 concepts; Term Map export) | answer space, corpus, abbreviation lexicon | provided |
| SapBERT | `cambridgeltl/SapBERT-from-PubMedBERT-fulltext` | dense retrieval + features | Apache-2.0 |
| gte-base | `thenlper/gte-base` | dense features | MIT |
| Qwen3-Embedding | `Qwen/Qwen3-Embedding-0.6B` | dense features | Apache-2.0 |
| Qwen2.5-Instruct | `Qwen/Qwen2.5-7B-Instruct`, `Qwen/Qwen2.5-14B-Instruct` | LLM listwise scoring | Apache-2.0 |
| LightGBM 4.7, scikit-learn 1.9, rapidfuzz 3.14, sparse_dot_topn 1.2 | | ranking, retrieval | MIT/BSD |

Not used: UMLS, the official SNOMED→ICD-10-CM map, any hosted API.
