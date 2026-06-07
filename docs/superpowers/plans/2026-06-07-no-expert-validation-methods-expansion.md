# No-Expert Validation Methods Expansion Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Strengthen the oncology historical-borrowing manuscript without expert borrowability labels by adding formal simulation operating characteristics, true temporal validation, paired backend benchmarking, baseline comparisons, and feature ablations.

**Architecture:** Keep the existing retrospective lambda pipeline as the source of leakage-controlled pseudo-query examples. Add focused scripts for date extraction, formal operating-characteristics simulation, paired Stage 1 benchmarking, baseline prior comparison, and feature ablation so each experiment can write a stable artifact directory and be tested independently.

**Tech Stack:** Python standard library, existing `docs/mixture_prior.py`, existing `scripts/train_retrospective_lambda_model.py`, existing `scripts/evaluate_stage1_retrieval.py`, existing JSONL/CSV/Markdown artifact conventions.

---

## Execution Order

1. Formal simulation operating characteristics.
2. True CT.gov date extraction and temporal validation metadata.
3. Paired Stage 1 backend benchmark on common query sets.
4. Traditional borrowing baseline head-to-head.
5. Feature ablation and sensitivity analysis.
6. Reporting updates and figure/table regeneration.

## Parallel Work

- Date extraction can run independently of simulation.
- Paired Stage 1 benchmark can run independently once backend JSONL outputs exist.
- Baseline comparison and feature ablation can share lambda training examples and should reuse the same deterministic splits.

## Deliverables

### Formal Simulation Study

**Files:**
- Create: `scripts/run_borrowing_operating_characteristics_simulation.py`
- Test: `tests/test_borrowing_operating_characteristics_simulation.py`
- Artifacts: `artifacts/operating_characteristics_simulation/`

**Outputs:**
- `simulation_operating_characteristics.csv`
- `simulation_scenarios.json`
- `simulation_operating_characteristics_report.md`

**Scope:**
- Exchangeable, mild conflict, strong conflict, mixture conflict, and heterogeneous historical scenarios.
- Weak-only, rule, learned/model, fixed-discount, and SAM variants.
- Type I error, power, bias, MSE, coverage, interval width, SAM trigger rate, and historical mass.

### True Temporal Validation Dates

**Files:**
- Create: `scripts/clinicaltrials_dates.py`
- Modify: `scripts/run_oncology_retrospective_lambda_training.py`
- Test: `tests/test_clinicaltrials_dates.py`, `tests/test_retrospective_lambda_full_evaluation.py`
- Artifacts: `artifacts/temporal_validation_true_dates/`

**Outputs:**
- `clinicaltrials_date_rows.csv`
- `clinicaltrials_date_missingness_report.json`

**Scope:**
- Extract `primaryCompletionDate`, `completionDate`, `resultsFirstSubmitDate` or posted-date equivalent, and `startDate`.
- Preserve date precision labels: day, month, year, missing, unparseable.
- Prefer primary completion date for temporal sorting.

### Paired Stage 1 Backend Benchmark

**Files:**
- Create: `scripts/run_paired_stage1_backend_benchmark.py`
- Test: `tests/test_paired_stage1_backend_benchmark.py`
- Artifacts: `artifacts/paired_stage1_backend_benchmark/`

**Outputs:**
- `paired_backend_query_metrics.csv`
- `paired_backend_summary.csv`
- `paired_backend_delta_rows.csv`
- `paired_backend_delta_bootstrap_ci.csv`
- `paired_backend_benchmark_report.md`

**Scope:**
- Compare common query IDs with the same endpoint key and topK budget.
- Report component-readiness, endpoint/result-readiness, clinical dimension scores, and paired bootstrap CIs.

### Baseline Prior Head-to-Head

**Files:**
- Create: `scripts/run_borrowing_baseline_comparison.py`
- Test: `tests/test_borrowing_baseline_comparison.py`
- Artifacts: `artifacts/borrowing_baseline_head_to_head/`

**Outputs:**
- `borrowing_baseline_nll.csv`
- `borrowing_baseline_summary.csv`
- `borrowing_baseline_report.md`

**Scope:**
- Weak-only, rule mixture, fixed discount, MAP-like empirical beta, power-prior-like, learned two-head, and SAM variants.

### Feature Ablation and Sensitivity

**Files:**
- Create: `scripts/run_feature_ablation_sensitivity.py`
- Test: `tests/test_feature_ablation_sensitivity.py`
- Artifacts: `artifacts/feature_ablation_sensitivity/`

**Outputs:**
- `feature_ablation_results.csv`
- `section_weight_sensitivity.csv`
- `feature_ablation_report.md`

**Scope:**
- Feature group removal for disease/regimen/endpoint/follow-up/eligibility/result/redflag/log-n.
- Sensitivity for `lambda0`, topK/rerank-topN, fixed discounts, and SECRET section weights.

## Current Implementation Slice

- [x] Add true CT.gov date extraction helper.
- [x] Add formal simulation operating-characteristics runner.
- [x] Add paired Stage 1 backend benchmark helper.
- [x] Extend temporal training CLI with fraction, date-based, and rolling-origin temporal validation modes.
- [x] Add baseline prior comparison runner.
- [x] Add feature ablation runner with feature-group removal and SECRET section-weight sensitivity.
- [ ] Regenerate manuscript-ready reports and figures from the new artifact directories.
