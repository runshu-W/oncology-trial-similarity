#!/usr/bin/env bash
set -euo pipefail

# Reproducibility-oriented command skeleton for the no-expert-label validation suite.
# Edit DB_ROOT and artifact paths before running on a new machine.

DB_ROOT="${DB_ROOT:-/path/to/Oncology_All_Trials}"
SECRET_EXAMPLES="${SECRET_EXAMPLES:-artifacts/retrospective_lambda_secret_pool_orr_all/lambda_training_examples.jsonl}"
SECRET_NLL="${SECRET_NLL:-artifacts/retrospective_lambda_secret_pool_orr_all/lambda_nll_rows.csv}"
HASHING_RESULTS="${HASHING_RESULTS:-artifacts/retrospective_lambda_oncology_orr_all/pipeline_results.jsonl}"
SECRET_RESULTS="${SECRET_RESULTS:-artifacts/stage1_secret_pool_rerank_orr_all/pipeline_results.jsonl}"

echo "== Step 1: true ClinicalTrials.gov date metadata =="
python pipeline/clinicaltrials_dates.py \
  --db-root "$DB_ROOT" \
  --output-dir artifacts/temporal_validation_true_dates

echo "== Step 2: attach true dates to lambda examples =="
python pipeline/attach_temporal_metadata_to_lambda_examples.py \
  --examples-jsonl "$SECRET_EXAMPLES" \
  --date-metadata-csv artifacts/temporal_validation_true_dates/clinicaltrials_date_rows.csv \
  --output-jsonl artifacts/temporal_validation_true_dates/lambda_training_examples_with_true_dates.jsonl \
  --report-json artifacts/temporal_validation_true_dates/lambda_training_examples_date_attachment.json

echo "== Step 3: true-date temporal NLL summaries =="
python pipeline/run_temporal_borrowing_validation.py \
  --examples-jsonl artifacts/temporal_validation_true_dates/lambda_training_examples_with_true_dates.jsonl \
  --output-dir artifacts/temporal_validation_true_dates \
  --methods weak_only rule fixed_discount commensurate_like rule_sam \
  --learned-nll-csv "$SECRET_NLL" \
  --learned-method-name two_head_trained \
  --reference-method rule

echo "== Step 4: paired Stage 1 backend benchmark =="
python pipeline/run_paired_stage1_backend_benchmark.py \
  --results "hashing=$HASHING_RESULTS" "secret_pool=$SECRET_RESULTS" \
  --baseline-label hashing \
  --output-dir artifacts/paired_stage1_backend_benchmark \
  --endpoint-key ORR \
  --top-k-eval 100 \
  --bootstrap-iterations 1000

echo "== Step 5: borrowing baseline head-to-head =="
python pipeline/run_borrowing_baseline_comparison.py \
  --examples-jsonl "$SECRET_EXAMPLES" \
  --output-dir artifacts/borrowing_baseline_head_to_head \
  --methods weak_only rule fixed_discount map_like power_prior_like commensurate_like rule_sam \
  --learned-nll-csv "$SECRET_NLL" \
  --learned-method-name two_head_trained \
  --reference-method rule

echo "== Step 6: simulation operating characteristics =="
python simulation/run_borrowing_operating_characteristics_simulation.py \
  --examples-jsonl "$SECRET_EXAMPLES" \
  --output-dir artifacts/operating_characteristics_simulation \
  --iterations "${SIMULATION_ITERATIONS:-500}" \
  --max-examples "${SIMULATION_MAX_EXAMPLES:-400}" \
  --methods weak_only rule rule_sam fixed_discount \
  --seed 20260607

echo "== Step 7: feature ablation and section sensitivity =="
python pipeline/run_feature_ablation_sensitivity.py \
  --examples-jsonl "$SECRET_EXAMPLES" \
  --pipeline-results-jsonl "$SECRET_RESULTS" \
  --output-dir artifacts/feature_ablation_sensitivity

echo "== Step 8: lightweight manuscript evidence package =="
python scripts/build_manuscript_evidence_package.py

echo "No-expert-label validation suite completed. Inspect results/ and artifacts/."
