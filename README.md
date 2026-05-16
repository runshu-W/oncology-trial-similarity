# Oncology Trial Similarity Pipeline

This repository collects Markdown documentation for an oncology clinical trial similarity pipeline focused on Bayesian prior borrowing from historical ClinicalTrials.gov oncology trials.

## Repository Contents

| File | Purpose |
| --- | --- |
| [`docs/oncology_trial_similarity_pipeline.md`](docs/oncology_trial_similarity_pipeline.md) | High-level pipeline design for retrieving historical oncology trials suitable for prior borrowing. |
| [`docs/oncology_trial_similarity_final_pipeline_explanation_source_a.md`](docs/oncology_trial_similarity_final_pipeline_explanation_source_a.md) | Full pipeline explanation, including offline indexing, online query flow, scoring, reranking, and outputs. |
| [`docs/oncology_trial_similarity_final_pipeline_explanation_source_b.md`](docs/oncology_trial_similarity_final_pipeline_explanation_source_b.md) | Duplicate source copy of the final pipeline explanation, preserved because it was provided as a separate input file. |
| [`docs/random5_clinicalbert_top3_pipeline_report.md`](docs/random5_clinicalbert_top3_pipeline_report.md) | Random 5-query ClinicalBERT top-3 similarity test report with retrieval and prior-borrowing rerank interpretation. |

## Suggested Reading Order

1. Start with `oncology_trial_similarity_pipeline.md` for the project objective and overall architecture.
2. Read `oncology_trial_similarity_final_pipeline_explanation_source_a.md` for the complete implementation-level explanation.
3. Review `random5_clinicalbert_top3_pipeline_report.md` to inspect example retrieval behavior and reranking results.

## Pipeline Goal

Given a new oncology clinical trial JSON, the pipeline searches a local historical oncology trial database and returns trials that are not only textually similar, but also clinically and statistically plausible candidates for Bayesian historical borrowing.

The ranking emphasizes:

- disease and population match
- treatment regimen similarity
- endpoint and estimand compatibility
- trial design comparability
- result availability and safety/follow-up usability
- red flags that should discount or prevent borrowing

## Notes

The two final pipeline explanation files are byte-identical in the provided source set. They are both retained with source-specific filenames so that the uploaded repository reflects all four requested files without filename collisions.
