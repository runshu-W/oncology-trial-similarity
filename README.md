# Oncology Trial Similarity and Prior Borrowing Pipeline

This repository contains a prototype pipeline and documentation for finding clinically similar historical oncology trials from ClinicalTrials.gov, with a focus on Bayesian prior borrowing.

The goal is not only to retrieve textually similar trials, but to identify historical trials that are potentially usable as external evidence based on disease, population, regimen, endpoint, design, results availability, and safety/follow-up relevance.

## Repository Contents

| File | Description |
| --- | --- |
| [`docs/oncology_trial_similarity_pipeline.py`](docs/oncology_trial_similarity_pipeline.py) | Main Python implementation for building an oncology trial similarity index and searching similar trials for a query JSON. |
| [`docs/oncology_trial_similarity_pipeline.md`](docs/oncology_trial_similarity_pipeline.md) | High-level design document describing the Trial2Vec + SECRET-style hybrid retrieval strategy and prior-borrowing rerank logic. |
| [`docs/oncology_trial_similarity_final_pipeline_explanation.md`](docs/oncology_trial_similarity_final_pipeline_explanation.md) | Full pipeline explanation covering offline indexing, online query processing, structured extraction, multi-aspect embeddings, reranking, and output reports. |
| [`docs/random5_clinicalbert_top3_pipeline_report.md`](docs/random5_clinicalbert_top3_pipeline_report.md) | Random five-query ClinicalBERT top-3 test report showing example retrieval and reranking results. |
| [`docs/evaluation/random5_manual_evaluation_protocol.md`](docs/evaluation/random5_manual_evaluation_protocol.md) | Human-review scoring guide for the random five-query top-3 results. |
| [`docs/evaluation/random5_top3_manual_review_template.csv`](docs/evaluation/random5_top3_manual_review_template.csv) | Review template with one row per query-candidate pair. |
| [`scripts/build_manual_evaluation_template.py`](scripts/build_manual_evaluation_template.py) | Helper script for regenerating the manual-review CSV from a random-five summary JSON file. |
| [`web_agent/`](web_agent/) | Local FastAPI web interface for uploading or pasting one trial JSON and running the pipeline. |

## What the Pipeline Does

The pipeline has two major stages:

1. **Offline indexing**
   - Discover historical trial folders by NCT ID.
   - Parse ClinicalTrials.gov JSON records and optional protocol/SAP PDFs.
   - Extract structured oncology trial fields.
   - Normalize disease, population, intervention, endpoint, design, and result fields.
   - Build multi-aspect embeddings for historical trials.

2. **Online search**
   - Accept a new oncology trial JSON as the query.
   - Convert it into the same structured summary schema.
   - Retrieve similar historical trials using weighted multi-aspect similarity.
   - Optionally rerank candidates using deterministic prior-borrowing criteria.
   - Output JSON results and an optional Markdown report.

## Similarity Dimensions

The retrieval and reranking logic emphasizes:

- disease and patient population match
- intervention and regimen similarity
- endpoint and estimand compatibility
- trial phase, arm structure, and randomization design
- result availability and borrowable quantities
- safety and follow-up relevance
- red flags that should discount or prevent borrowing

## Example Usage

Build a local trial index:

```bash
python3 docs/oncology_trial_similarity_pipeline.py build-index \
  --db-root /path/to/Oncology_All_Trials \
  --output-dir artifacts/oncology_trial_similarity
```

Build an index with ClinicalBERT embeddings:

```bash
python3 docs/oncology_trial_similarity_pipeline.py build-index \
  --db-root /path/to/Oncology_All_Trials \
  --output-dir artifacts/oncology_trial_similarity_clinicalbert \
  --embedding-backend clinicalbert
```

Search for similar historical trials:

```bash
python3 docs/oncology_trial_similarity_pipeline.py search \
  --query-json /path/to/new_trial.json \
  --index-dir artifacts/oncology_trial_similarity \
  --top-k 10 \
  --rerank \
  --output artifacts/query_top10.json \
  --report-output artifacts/query_top10_report.md
```

## Suggested Reading Order

1. Read [`oncology_trial_similarity_pipeline.md`](docs/oncology_trial_similarity_pipeline.md) for the core project idea and architecture.
2. Read [`oncology_trial_similarity_final_pipeline_explanation.md`](docs/oncology_trial_similarity_final_pipeline_explanation.md) for the complete implementation-level explanation.
3. Inspect [`oncology_trial_similarity_pipeline.py`](docs/oncology_trial_similarity_pipeline.py) for the executable prototype.
4. Review [`random5_clinicalbert_top3_pipeline_report.md`](docs/random5_clinicalbert_top3_pipeline_report.md) for sample ClinicalBERT retrieval results.

## Notes

The default database path in the script points to a local ClinicalTrials.gov oncology dataset and should be updated with `--db-root` when running on another machine.

ClinicalBERT mode requires the relevant Python ML dependencies and access to the Bio_ClinicalBERT model. The hashing backend is available as a lightweight fallback for local testing.

PDF parsing uses `pdftotext` when available and falls back to the Python `pypdf` package. Install Python dependencies with:

```bash
python3 -m pip install -r requirements.txt
```

## Web Agent

From this directory, start the local web interface with:

```bash
../.venv/bin/uvicorn web_agent.app:app --reload --port 8000
```

Then open `http://127.0.0.1:8000`. The web agent accepts one uploaded or pasted trial JSON and returns the same pipeline-style suitability, discount, red flag, dimension-score, and borrowable-quantity outputs. Suggested borrowing discounts are not final prior weights and require expert clinical/statistical review.
