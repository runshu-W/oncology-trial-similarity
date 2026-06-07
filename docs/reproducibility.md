# Reproducibility Notes

This repository is organized around a lightweight public evidence package in `results/` and heavier local runtime artifacts in `artifacts/`. The heavy artifacts are ignored by Git because they may include large JSONL files, model checkpoints, or local ClinicalTrials.gov exports.

## Environment

Install the basic Python dependencies:

```bash
python -m pip install -r requirements.txt
```

Some retrospective lambda-model training paths require `torch`. Non-torch components include true date extraction, temporal split construction, baseline comparison, simulation operating characteristics, paired backend benchmarking, feature ablation, and manuscript evidence package generation.

## Build the Manuscript Evidence Package

If the required artifact directories are already present, run:

```bash
python scripts/build_manuscript_evidence_package.py
```

This creates:

- `results/tables/*.csv`
- `results/tables/*.tex`
- `results/figures/*.svg`
- `docs/manuscript_evidence_plan.md`

## True Date Extraction

```bash
python scripts/clinicaltrials_dates.py \
  --db-root /path/to/Oncology_All_Trials \
  --output-dir artifacts/temporal_validation_true_dates
```

Expected outputs:

- `clinicaltrials_date_rows.csv`
- `clinicaltrials_date_missingness_report.json`

## Attach Date Metadata to Lambda Examples

```bash
python scripts/attach_temporal_metadata_to_lambda_examples.py \
  --examples-jsonl artifacts/retrospective_lambda_secret_pool_orr_all/lambda_training_examples.jsonl \
  --date-metadata-csv artifacts/temporal_validation_true_dates/clinicaltrials_date_rows.csv \
  --output-jsonl artifacts/temporal_validation_true_dates/lambda_training_examples_with_true_dates.jsonl \
  --report-json artifacts/temporal_validation_true_dates/lambda_training_examples_date_attachment.json
```

## True-Date Temporal NLL Table

```bash
python scripts/run_temporal_borrowing_validation.py \
  --examples-jsonl artifacts/temporal_validation_true_dates/lambda_training_examples_with_true_dates.jsonl \
  --output-dir artifacts/temporal_validation_true_dates \
  --methods weak_only rule fixed_discount commensurate_like rule_sam \
  --learned-nll-csv artifacts/retrospective_lambda_secret_pool_orr_all/lambda_nll_rows.csv \
  --learned-method-name two_head_trained \
  --reference-method rule
```

This is a true-date temporal NLL summary using existing learned NLL rows. It is not the same as full temporal retraining. Full retraining commands are listed in `docs/torch_temporal_validation_commands.md`.

## Paired Stage 1 Backend Benchmark

```bash
python scripts/run_paired_stage1_backend_benchmark.py \
  --results hashing=artifacts/retrospective_lambda_oncology_orr_all/pipeline_results.jsonl \
            secret_pool=artifacts/stage1_secret_pool_rerank_orr_all/pipeline_results.jsonl \
  --baseline-label hashing \
  --output-dir artifacts/paired_stage1_backend_benchmark \
  --endpoint-key ORR \
  --top-k-eval 100 \
  --bootstrap-iterations 1000
```

## Borrowing Baseline Head-to-Head

```bash
python scripts/run_borrowing_baseline_comparison.py \
  --examples-jsonl artifacts/retrospective_lambda_secret_pool_orr_all/lambda_training_examples.jsonl \
  --output-dir artifacts/borrowing_baseline_head_to_head \
  --methods weak_only rule fixed_discount map_like power_prior_like commensurate_like rule_sam \
  --learned-nll-csv artifacts/retrospective_lambda_secret_pool_orr_all/lambda_nll_rows.csv \
  --learned-method-name two_head_trained \
  --reference-method rule
```

## Simulation Operating Characteristics

```bash
python scripts/run_borrowing_operating_characteristics_simulation.py \
  --examples-jsonl artifacts/retrospective_lambda_secret_pool_orr_all/lambda_training_examples.jsonl \
  --output-dir artifacts/operating_characteristics_simulation \
  --iterations 500 \
  --max-examples 400 \
  --methods weak_only rule rule_sam fixed_discount \
  --seed 20260607
```

The current lightweight manuscript evidence package uses 500 iterations with 400 deterministic template examples. On stronger compute environments, repeat with the full template set or 1000 iterations as a sensitivity analysis.

## Feature Ablation and Sensitivity

```bash
python scripts/run_feature_ablation_sensitivity.py \
  --examples-jsonl artifacts/retrospective_lambda_secret_pool_orr_all/lambda_training_examples.jsonl \
  --pipeline-results-jsonl artifacts/stage1_secret_pool_rerank_orr_all/pipeline_results.jsonl \
  --output-dir artifacts/feature_ablation_sensitivity
```

## Verification

Run focused non-torch tests:

```bash
python -m unittest \
  tests/test_clinicaltrials_dates.py \
  tests/test_temporal_validation.py \
  tests/test_temporal_borrowing_validation.py \
  tests/test_borrowing_operating_characteristics_simulation.py \
  tests/test_paired_stage1_backend_benchmark.py \
  tests/test_borrowing_baseline_comparison.py \
  tests/test_feature_ablation_sensitivity.py
```

Compile key scripts:

```bash
python -m py_compile \
  scripts/clinicaltrials_dates.py \
  scripts/temporal_validation.py \
  scripts/run_borrowing_operating_characteristics_simulation.py \
  scripts/run_paired_stage1_backend_benchmark.py \
  scripts/run_borrowing_baseline_comparison.py \
  scripts/run_temporal_borrowing_validation.py \
  scripts/run_feature_ablation_sensitivity.py \
  scripts/build_manuscript_evidence_package.py
```
