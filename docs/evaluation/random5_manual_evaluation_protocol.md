# Random 5 Manual Evaluation Protocol

This file defines a lightweight human-review pass for the random five-query ClinicalBERT test set. The goal is to assess whether top-ranked historical trials are clinically plausible prior-borrowing candidates, not just embedding neighbors.

## Inputs

- Pipeline report: `docs/random5_clinicalbert_top3_pipeline_report.md`
- Review template: `docs/evaluation/random5_top3_manual_review_template.csv`
- Source summary used to generate the template: `artifacts/random5_clinicalbert_top3_test/random5_top3_summary.json`

## Review Instructions

Score each query-candidate pair on five dimensions:

- `reviewer_disease_population_0_2`: `0` mismatch, `1` partial match, `2` close match.
- `reviewer_regimen_0_2`: `0` mismatch, `1` related class or partial regimen, `2` close regimen.
- `reviewer_endpoint_0_2`: `0` not borrowable, `1` related but different estimand, `2` compatible estimand.
- `reviewer_design_0_2`: `0` incompatible design, `1` partly comparable, `2` close phase/arms/randomization context.
- `reviewer_result_usable_0_2`: `0` unusable, `1` usable only for sensitivity analysis, `2` usable for candidate prior construction.

Fill `reviewer_overall_borrowability` with one of:

- `high`
- `medium`
- `low`
- `do_not_borrow`

Use `reviewer_notes` to record the specific reason for disagreement with the pipeline, especially disease mismatch, treatment mismatch, endpoint mismatch, missing denominator, or protocol/SAP ambiguity.

## Success Criteria

- All 15 query-candidate rows receive human scores.
- Rows where the reviewer category differs from the pipeline category are flagged in `reviewer_notes`.
- Common error modes are summarized before changing reranker weights or normalization rules.
