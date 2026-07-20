# Oncology Trial Similarity and Bayesian Prior Borrowing Project Summary

Last updated: 2026-05-20

## 1. Project goal

This project builds an end-to-end local pipeline for oncology clinical trial similarity retrieval and Bayesian prior borrowing support.

The project objective is not ordinary text similarity. The core question is:

> Which historical oncology trials are clinically similar enough in disease, population, treatment, endpoint, design, and result usability to support Bayesian prior borrowing?

The current system supports:

- Parsing ClinicalTrials.gov-style trial JSON.
- Discovering protocol/SAP PDF paths.
- Extracting outcome tables, arm-level counts, denominators, percentages, and proportions.
- Rule-based oncology normalization.
- Multi-aspect trial representation.
- ClinicalBERT retrieval.
- Stage-2 prior-borrowing reranking.
- Bayesian prior-borrowing analysis.
- Local FastAPI web agent with JSON/Markdown output and visualization.

## 2. Main project folders and files

### Core repository

- `README.md`  
  Repository overview, usage examples, and Web agent startup instructions.

- `requirements.txt`  
  Runtime dependencies for pipeline and Web agent.

- `pipeline/oncology_trial_similarity_pipeline.py`  
  Main executable pipeline. It includes indexing, retrieval, reranking, Bayesian analysis, report rendering, and CLI entrypoints.

- `docs/oncology_trial_similarity_pipeline.md`  
  Design document for Trial2Vec + SECRET-style retrieval and prior-borrowing rerank logic.

- `docs/oncology_trial_similarity_final_pipeline_explanation.md`  
  Full implementation-level explanation of the complete retrieval/rerank pipeline.

- `docs/project_work_summary.md`  
  This summary document.

### Web agent

- `web_agent/app.py`  
  FastAPI backend. Accepts uploaded or pasted JSON, runs the pipeline, and returns JSON plus Markdown report.

- `web_agent/static/index.html`  
  Local browser UI. Displays query summary, candidate trials, Bayesian analysis tables, and Bayesian charts.

- `web_agent/README.md`  
  Web agent-specific run instructions and output description.

### Tests

- `tests/test_web_agent.py`  
  Unit tests for health endpoint, invalid JSON handling, search response shape, temp query file passing, Bayesian analysis behavior, endpoint exclusion, prior-only mode, arm selection, endpoint-specific denominator scoring, and HTML chart renderer presence.

### Evaluation support

- `docs/evaluation/random5_manual_evaluation_protocol.md`  
  Manual scoring protocol for random-five top-three retrieval results.

- `docs/evaluation/random5_top3_manual_review_template.csv`  
  Human review template.

- `scripts/build_manual_evaluation_template.py`  
  Helper script for regenerating manual review CSV templates.

## 3. External project context files

These files live one level above the repository and record broader context and statistical design:

- `../context.md`  
  Full project context, environment details, index paths, random-five test results, and design rationale.

- `../bayesian_prior_borrowing_hierarchical_model.md`  
  English Bayesian prior-borrowing design document.

- `../bayesian_prior_borrowing_hierarchical_model_zh.md`  
  Chinese Bayesian prior-borrowing design document. Covers hierarchical model, MAP prior, robust MAP idea, path A for two-arm trials, and go/no-go decision formulation.

- `../oncology_trial_similarity_report.md`  
  Earlier high-level report.

- `../oncology_trial_similarity_pipeline_detailed_report.md`  
  Earlier detailed pipeline report.

- `../TRIAL2VEC_REPRODUCTION.md`  
  Trial2Vec reproduction notes.

## 4. Data and artifacts

### Historical database

Default local database:

```text
/Users/wang/PHD/clinic.gov/Oncology_All_Trials/Oncology_All_Trials
```

The database contains local oncology historical trial folders, usually one folder per NCT ID.

### ClinicalBERT index

Default Web agent index:

```text
../artifacts/oncology_trial_similarity_clinicalbert/
```

Main files:

- `../artifacts/oncology_trial_similarity_clinicalbert/trial_summaries.jsonl`
- `../artifacts/oncology_trial_similarity_clinicalbert/trial_embeddings.npz`

This index uses:

- `embedding_backend = clinicalbert`
- `embedding_model = emilyalsentzer/Bio_ClinicalBERT`

### Hashing index

Fallback/debug index:

```text
../artifacts/oncology_trial_similarity/
```

Main files:

- `../artifacts/oncology_trial_similarity/trial_summaries.jsonl`
- `../artifacts/oncology_trial_similarity/trial_embeddings.npz`

### Random-five test artifacts

Random-five ClinicalBERT top-three test outputs:

```text
../artifacts/random5_clinicalbert_top3_test/
```

Important files:

- `sampled_nct_ids.json`
- `random5_top3_summary.json`
- `random5_clinicalbert_top3_pipeline_report.md`
- Per-query `*_stage2_top3.json`
- Per-query `*_stage2_top3.md`

## 5. End-to-end pipeline

### Step 1. Input

Input is one ClinicalTrials.gov-style oncology trial JSON.

The system can run through:

- CLI: `pipeline/oncology_trial_similarity_pipeline.py search`
- Web agent: upload or paste JSON into the local browser UI

### Step 2. JSON and document extraction

The pipeline extracts:

- NCT ID
- brief title / official title
- summary / description
- phase / status
- intervention/treatment
- study design
- enrollment
- primary and secondary outcomes
- outcome data tables
- measurement rows
- denominator rows
- arm-level result rows
- protocol/SAP PDF paths

The code for this is mainly in:

- `find_trial_json`
- `find_supporting_pdfs`
- `extract_outcomes`
- `extract_trial_record`
- `extract_trial_record_like`

All are in `pipeline/oncology_trial_similarity_pipeline.py`.

### Step 3. Oncology normalization

The rule-based normalizer extracts:

- primary site
- histology
- molecular marker
- stage/risk
- line of therapy
- age group
- drug classes
- regimen backbone
- arm structure
- randomization
- number of arms

Key functions:

- `infer_oncology_concepts`
- `infer_intervention_concepts`
- `infer_design_concepts`
- `make_rule_based_summary`

### Step 4. Multi-aspect representation

Each trial summary is converted into aspect-specific text:

- disease/population
- intervention
- endpoint
- design
- results/safety

Aspect weights:

```python
ASPECT_WEIGHTS = {
    "disease_population": 0.30,
    "intervention": 0.25,
    "endpoint": 0.20,
    "design": 0.15,
    "results_safety": 0.10,
}
```

Key functions:

- `aspect_text`
- `summary_embedding`
- `weighted_similarity`

### Step 5. Embedding backends

Two embedding backends are supported:

- `hashing`  
  Lightweight signed hashing embedding for debugging and reproducibility.

- `clinicalbert`  
  Bio_ClinicalBERT mean-pooling embedding using local HuggingFace cache.

Key classes/functions:

- `HashingEmbedder`
- `ClinicalBertEmbedder`
- `make_embedder`

Important environment note:

ClinicalBERT should be run using:

```bash
../.venv/bin/python
```

### Step 6. Build index

The index builder:

1. Iterates over NCT folders.
2. Parses JSON and supporting documents.
3. Builds rule-based trial summaries.
4. Encodes each trial into multi-aspect embeddings.
5. Writes `trial_summaries.jsonl` and `trial_embeddings.npz`.

CLI example:

```bash
../.venv/bin/python pipeline/oncology_trial_similarity_pipeline.py build-index \
  --db-root /Users/wang/PHD/clinic.gov/Oncology_All_Trials/Oncology_All_Trials \
  --output-dir ../artifacts/oncology_trial_similarity_clinicalbert \
  --embedding-backend clinicalbert \
  --embedding-batch-size 16 \
  --embedding-max-length 256
```

### Step 7. Stage-1 retrieval

The search flow:

1. Parses query JSON into the same trial summary schema.
2. Loads index embeddings.
3. Verifies query backend matches stored index backend.
4. Embeds the query trial.
5. Computes weighted multi-aspect cosine similarity.
6. Returns top-K candidates.

Key function:

- `search`

### Step 8. Stage-2 prior-borrowing rerank

Stage-2 rerank evaluates whether a retrieved historical trial is suitable for prior borrowing.

It outputs six dimension scores:

- `disease_population_match`
- `treatment_regimen_match`
- `endpoint_estimand_match`
- `design_phase_match`
- `result_usability`
- `safety_and_followup_relevance`

It also outputs:

- `overall_similarity_score`
- `prior_borrowing_suitability`
- `suggested_borrowing_discount`
- `dimension_scores`
- `borrowable_quantities`
- `required_adjustments`
- `red_flags`
- `candidate_snapshot`

Discount mapping:

```text
high -> 0.75
medium -> 0.40
low -> 0.15
do_not_borrow -> 0.00
```

Key functions:

- `score_prior_borrowing_pair`
- `rerank_candidates`

## 6. Bayesian prior-borrowing analysis

The Bayesian layer was added on top of existing search output without changing the input API.

Main entrypoint:

- `add_bayesian_analysis`

It adds:

```json
"bayesian_analysis": {
  "status": "available | not_available",
  "model": "weighted_beta_binomial_path_a",
  "endpoint_analyses": [],
  "two_arm_decision_support": {},
  "limitations": []
}
```

### Supported endpoint families

Current Bayesian analysis supports:

- ORR absolute rate
- PFS6 absolute rate when 6-month count/denominator is available

Explicitly excluded:

- DOR / Duration of Response
- OS
- generic survival endpoints without 6-month binary data
- PFS without identifiable 6-month count/denominator

Key function:

- `canonical_bayesian_endpoint`

### Historical borrowing data

For each eligible historical trial, the code extracts:

- candidate NCT ID
- endpoint
- treatment arm
- count
- denominator
- observed rate
- borrowing weight
- suitability category

Key function:

- `historical_endpoint_observations`

### Query data modes

The Bayesian layer supports two modes:

#### Posterior mode

Used when query trial has treatment-arm count/denominator.

The analysis computes:

- borrowed prior
- posterior after observing query data
- ESS
- weighted historical rate
- posterior summaries
- probability grid
- tipping points

#### Prior-only mode

Used when query trial has ORR/PFS6 primary endpoint definition but no posted query result.

The analysis computes:

- borrowed prior
- ESS
- weighted historical rate
- prior summaries
- probability grid based on active prior distribution
- tipping points based on active prior distribution

This is important because many new trials do not yet have posted results.

### Statistical approximation

The current implementation uses a lightweight weighted beta-binomial power-prior approximation:

```text
alpha = 1 + sum_j w_j * y_j
beta  = 1 + sum_j w_j * (n_j - y_j)
```

If query data are available:

```text
alpha_post = alpha + y_0
beta_post  = beta + n_0 - y_0
```

The implementation intentionally does not claim to be a full robust MAP mixture posterior.

### Output summaries

For each endpoint, the Bayesian output includes:

- `analysis_mode`
- `query_endpoint`
- `historical_observations`
- `historical_trial_count`
- `effective_sample_size`
- `weighted_historical_rate`
- `mixture_weight_pi`
- `weight_sensitivity`
- `success_probability_grid`
- `tipping_points`
- `notes`

### Weight sensitivity scenarios

Scenarios:

- `no_borrowing`
- `25pct_weights`
- `50pct_weights`
- `75pct_weights`
- `observed_weights`
- `125pct_capped_weights`

### Effective sample size

ESS is computed as:

```text
ESS = sum_j w_j * n_j
```

### Probability grid

The system does not define `p_target`.

Instead, it provides a threshold grid and computes:

```text
Pr(p >= threshold | active distribution)
```

The user defines the final clinical target and decision probability cutoff.

### Tipping points

The system reports rate thresholds associated with posterior/prior probability levels:

```text
0.5, 0.7, 0.8, 0.9
```

This supports sensitivity analysis without forcing a go/no-go decision.

### Two-arm ORR path A

When query has two-arm ORR data:

- treatment arm uses borrowed absolute-rate prior
- control arm uses weak beta prior
- posterior OR is estimated by Monte Carlo sampling

Output includes:

- posterior OR mean
- posterior OR median
- 95% credible interval
- OR probability grid

The system does not define `OR_target` or `gamma`.

## 7. Bayesian fixes completed after review

The review found and fixed several important issues:

### DOR was incorrectly treated as ORR

Problem:

`Duration of Response` contains the word `Response`, so it was incorrectly mapped to ORR.

Fix:

- Explicitly exclude `DOR` and `duration of response`.
- Only map ORR-like endpoints when text contains `ORR`, `objective response`, or `response rate`.

Test:

- `test_bayesian_analysis_excludes_duration_of_response`

### Query without results could not produce Bayesian output

Problem:

The first implementation required query treatment-arm count/denominator, so new protocol-only trials could not show Bayesian prior information.

Fix:

- Added prior-only analysis mode.
- Query endpoint definition can trigger Bayesian analysis even without query result rows.

Test:

- `test_bayesian_analysis_supports_prior_only_when_query_has_no_results`

### Endpoint denominator credit was too broad

Problem:

The reranker gave endpoint score credit if any candidate denominator existed, even if it belonged to a non-matching endpoint.

Fix:

- Added endpoint-specific denominator matching.
- Denominator credit now requires a matched endpoint family with count/denominator.

Test:

- `test_endpoint_score_requires_matched_endpoint_denominator`

### Active comparator arm could be misused as treatment

Problem:

Arm selection could treat `Active Comparator` as a treatment arm.

Fix:

- Arm role detection now classifies comparator/control/placebo/standard-of-care labels as control.
- Experimental/treatment/investigational labels are prioritized as treatment.

Test:

- `test_treatment_arm_selection_skips_active_comparator`

### Robust MAP wording was too strong

Problem:

The UI wording could imply a full robust MAP mixture posterior was implemented.

Fix:

- Web UI and README now describe the method as a weighted beta-binomial power-prior approximation.
- `mixture_weight_pi` remains a sensitivity descriptor, not a full mixture posterior.

## 8. Web agent

### Backend

File:

- `web_agent/app.py`

Endpoints:

- `GET /`
- `GET /api/health`
- `POST /api/search`

Default index:

```text
../artifacts/oncology_trial_similarity_clinicalbert
```

The backend:

- loads the pipeline module from `pipeline/oncology_trial_similarity_pipeline.py`
- writes pasted/uploaded JSON to a temp file
- calls `pipeline.search`
- returns raw result plus Markdown report

CORS is enabled because users sometimes open `index.html` via `file://`.

### Frontend

File:

- `web_agent/static/index.html`

The UI displays:

- pipeline health status
- input JSON upload/paste
- query summary
- top reranked candidates
- red flags
- dimension scores
- borrowable quantities
- Bayesian analysis
- charts and tables
- JSON/Markdown download buttons

### Bayesian charts

The Web UI adds chart/table pairs:

- prior / borrowed prior / active distribution density overlay
- borrowing weight sensitivity forest-style plot
- probability of success curve
- tipping point curve
- posterior OR probability curve for two-arm ORR

Charts are implemented as inline SVG without external JavaScript chart dependencies.

### File URL support

The UI detects `file://` usage and sends API calls to:

```text
http://127.0.0.1:8000
```

Recommended usage remains:

```text
http://127.0.0.1:8000
```

## 9. Reports and outputs

### JSON output

The JSON output includes:

- query summary
- top matches
- reranked top matches
- Bayesian analysis

### Markdown report

Generated by:

- `render_markdown_report`

The report includes:

- query summary
- reranked top matches
- notes
- Bayesian prior borrowing section when available

## 10. Validation status

Current validation commands:

```bash
../.venv/bin/python -m unittest tests.test_web_agent
```

Current result:

```text
10 tests passed
```

Syntax validation:

```bash
../.venv/bin/python -m py_compile \
  pipeline/oncology_trial_similarity_pipeline.py \
  web_agent/app.py \
  scripts/build_manual_evaluation_template.py
```

Current result:

```text
passed
```

Additional manual checks completed:

- DOR maps to `None`, not ORR.
- Prior-only mode returns `status = available`.
- Prior-only mode returns `analysis_mode = prior_only`.
- Bayesian output is JSON serializable.
- Web agent health endpoint responds.
- ClinicalBERT index is available.

## 11. Known limitations

### Statistical limitations

- The current Bayesian layer is a weighted beta-binomial approximation, not a full hierarchical MAP or robust MAP mixture posterior.
- `mixture_weight_pi` is currently a sensitivity descriptor.
- PFS is only supported when 6-month count/denominator data are available.
- KM estimate + SE support is not yet implemented.
- HR borrowing for PFS is not yet implemented.
- User-defined `p_target`, `OR_target`, and `gamma` are not set by the system.

### Clinical normalization limitations

- Oncology concept extraction is rule-based.
- Line of therapy is often `Not reported`.
- Histology and regimen backbone normalization can miss rare cancers or uncommon regimens.
- Protocol/SAP text extraction depends on PDF tooling and available documents.

### Arm-selection limitations

- Treatment/control arm role is still mostly label-based.
- Complex multi-arm trials may need manual review.
- Future improvement should use intervention overlap and structured arm metadata.

### Frontend limitations

- `web_agent/static/index.html` is a single large file.
- Chart rendering is custom inline SVG.
- Future maintenance would benefit from splitting chart/render helper code into separate JS files.

## 12. Recommended next work

Priority order:

1. Add PFS6 KM estimate + SE support.
2. Improve treatment/control arm selection using intervention overlap and regimen matching.
3. Implement true robust MAP mixture posterior if this becomes a statistical requirement.
4. Add endpoint-specific validation metrics on the random-five set.
5. Add more unit tests for PFS6, no-borrowing, do-not-borrow, and multi-arm edge cases.
6. Split Web UI JavaScript into smaller files if frontend complexity continues to grow.
7. Add an exportable Bayesian summary table for downstream reports.
8. Add optional user inputs for `p_target`, `OR_target`, and `gamma` while keeping defaults unset.

## 13. How to run

Start Web agent:

```bash
cd "/Users/wang/Documents/New project/oncology-trial-similarity"
../.venv/bin/uvicorn web_agent.app:app --reload --port 8000
```

Open:

```text
http://127.0.0.1:8000
```

Run tests:

```bash
cd "/Users/wang/Documents/New project/oncology-trial-similarity"
../.venv/bin/python -m unittest tests.test_web_agent
```

Run syntax check:

```bash
../.venv/bin/python -m py_compile \
  pipeline/oncology_trial_similarity_pipeline.py \
  web_agent/app.py \
  scripts/build_manual_evaluation_template.py
```

Search via CLI:

```bash
../.venv/bin/python pipeline/oncology_trial_similarity_pipeline.py search \
  --query-json /path/to/new_trial.json \
  --index-dir ../artifacts/oncology_trial_similarity_clinicalbert \
  --top-k 10 \
  --rerank \
  --rerank-top-n 100 \
  --output ../artifacts/query_result.json \
  --report-output ../artifacts/query_report.md
```

## 14. Current modified files

At the time this document was written, the main modified files were:

- `pipeline/oncology_trial_similarity_pipeline.py`
- `tests/test_web_agent.py`
- `web_agent/README.md`
- `web_agent/app.py`
- `web_agent/static/index.html`

These contain the Bayesian extension, Web agent improvements, visualization work, and regression tests.

