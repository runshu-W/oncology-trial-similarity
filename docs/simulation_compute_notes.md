# Simulation Compute Notes

The current manuscript evidence package uses:

```text
6 scenarios x 4 methods x 500 iterations x 400 deterministic template examples
```

The template examples are sampled from the ORR retrospective lambda examples with a fixed seed (`20260607`). This provides a higher-iteration operating-characteristics table than the earlier 100-iteration development run while keeping local runtime manageable.

## Why Use Template Subsampling

Full-template simulation over all ORR examples is computationally heavier because each simulated trial requires:

- historical binomial outcome generation for candidate components;
- beta component reconstruction;
- mixture predictive/posterior calculations;
- optional SAM conflict adaptation;
- type I error, power, bias, MSE, coverage, interval width, historical mass, and SAM trigger aggregation.

The deterministic template subset keeps the scenario definitions, candidate-set structure, and simulation logic fixed while reducing runtime. Because the seed is fixed, the result is exactly reproducible.

## Recommended Sensitivity Checks Before Submission

For a final manuscript revision, repeat at least one of the following:

1. Full-template 500-iteration run:

```bash
python simulation/run_borrowing_operating_characteristics_simulation.py \
  --examples-jsonl artifacts/retrospective_lambda_secret_pool_orr_all/lambda_training_examples.jsonl \
  --output-dir artifacts/operating_characteristics_simulation_full_500 \
  --iterations 500 \
  --methods weak_only rule rule_sam fixed_discount \
  --seed 20260607
```

2. Deterministic 1000-iteration subsampling sensitivity:

```bash
python simulation/run_borrowing_operating_characteristics_simulation.py \
  --examples-jsonl artifacts/retrospective_lambda_secret_pool_orr_all/lambda_training_examples.jsonl \
  --output-dir artifacts/operating_characteristics_simulation_subsample_1000 \
  --iterations 1000 \
  --max-examples 400 \
  --methods weak_only rule rule_sam fixed_discount \
  --seed 20260607
```

3. Alternative template subset:

```bash
python simulation/run_borrowing_operating_characteristics_simulation.py \
  --examples-jsonl artifacts/retrospective_lambda_secret_pool_orr_all/lambda_training_examples.jsonl \
  --output-dir artifacts/operating_characteristics_simulation_subsample_seed2 \
  --iterations 500 \
  --max-examples 400 \
  --methods weak_only rule rule_sam fixed_discount \
  --seed 20260608
```

## Reporting Language

Recommended manuscript wording:

> Simulation operating characteristics were estimated using leakage-controlled ORR retrospective pseudo-query templates. The primary simulation table used 500 Monte Carlo iterations and a deterministic seed-fixed subset of 400 pseudo-query templates for computational tractability; full-template or alternative-seed analyses can be reported as sensitivity checks.
