# Torch Temporal Retraining Commands

The current local environment used to prepare the lightweight evidence package did not have `torch` installed. Therefore the current true-date temporal NLL tables in `results/` are temporal subset summaries using existing learned NLL rows, not full temporal retraining experiments.

Use the commands below in an environment with `torch` installed to run full temporal retraining and evaluation.

## Date-Based Temporal Retraining

```bash
python pipeline/run_oncology_retrospective_lambda_training.py \
  --pipeline-results-jsonl artifacts/stage1_secret_pool_rerank_orr_all/pipeline_results.jsonl \
  --output-dir artifacts/temporal_retraining_date_based_2020 \
  --endpoint-key ORR \
  --lambda0 0.2 \
  --model-type two_head_deepsets \
  --epochs 100 \
  --learning-rate 0.01 \
  --hidden-dim 16 \
  --temporal-split-mode date_based \
  --temporal-train-end-date 2020-12-31 \
  --date-metadata-csv artifacts/temporal_validation_true_dates/clinicaltrials_date_rows.csv \
  --bootstrap-iterations 1000 \
  --simulation-iterations 0
```

Repeat with alternative cutoffs:

```bash
for cutoff in 2019-12-31 2020-12-31 2021-12-31 2022-12-31; do
  python pipeline/run_oncology_retrospective_lambda_training.py \
    --pipeline-results-jsonl artifacts/stage1_secret_pool_rerank_orr_all/pipeline_results.jsonl \
    --output-dir "artifacts/temporal_retraining_date_based_${cutoff}" \
    --endpoint-key ORR \
    --lambda0 0.2 \
    --model-type two_head_deepsets \
    --epochs 100 \
    --learning-rate 0.01 \
    --hidden-dim 16 \
    --temporal-split-mode date_based \
    --temporal-train-end-date "$cutoff" \
    --date-metadata-csv artifacts/temporal_validation_true_dates/clinicaltrials_date_rows.csv \
    --bootstrap-iterations 1000
done
```

## Rolling-Origin Temporal Retraining

```bash
python pipeline/run_oncology_retrospective_lambda_training.py \
  --pipeline-results-jsonl artifacts/stage1_secret_pool_rerank_orr_all/pipeline_results.jsonl \
  --output-dir artifacts/temporal_retraining_rolling_origin \
  --endpoint-key ORR \
  --lambda0 0.2 \
  --model-type two_head_deepsets \
  --epochs 100 \
  --learning-rate 0.01 \
  --hidden-dim 16 \
  --temporal-split-mode rolling_origin \
  --rolling-min-train-count 500 \
  --rolling-eval-window-size 250 \
  --date-metadata-csv artifacts/temporal_validation_true_dates/clinicaltrials_date_rows.csv \
  --bootstrap-iterations 1000
```

## Expected Outputs

Each run should produce:

- `lambda_training_examples.jsonl`
- `lambda_training_summary.json`
- `lambda_evaluation.json`
- `lambda_nll_rows.csv`
- `lambda_temporal_split_evaluation.json`
- `lambda_temporal_prediction_rows.csv`
- `lambda_temporal_nll_rows.csv`
- `retrospective_lambda_training_results.md`

## Reporting Distinction

Use precise language when reporting temporal validation:

- **Current subset table:** existing trained NLL values summarized over true-date subsets.
- **Full temporal retraining:** model retrained using only training-period pseudo-queries and evaluated on future-period pseudo-queries.

The manuscript should prefer full temporal retraining when available. If it is not available, the subset table should be labeled as retrospective true-date temporal subset calibration.
