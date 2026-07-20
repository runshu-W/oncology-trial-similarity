# Results Package

This directory contains lightweight, manuscript-oriented result files generated from local runtime artifacts. It is intended for reproducibility checks and manuscript drafting without committing large JSONL files, model checkpoints, or raw ClinicalTrials.gov data.

## Tables

| File | Description |
|---|---|
| `tables/simulation_operating_characteristics.csv` | Simulation operating-characteristics results across exchangeability and conflict scenarios. |
| `tables/table_simulation_operating_characteristics.tex` | LaTeX-ready simulation table. |
| `tables/paired_backend_summary.csv` | Mean metrics for paired Stage 1 backend benchmark. |
| `tables/paired_backend_delta_bootstrap_ci.csv` | Paired bootstrap confidence intervals for backend metric deltas. |
| `tables/table_paired_stage1_benchmark.tex` | LaTeX-ready paired backend benchmark table. |
| `tables/borrowing_baseline_summary.csv` | Head-to-head borrowing prior baseline summary. |
| `tables/table_borrowing_baseline_head_to_head.tex` | LaTeX-ready borrowing baseline table. |
| `tables/temporal_borrowing_nll_table.csv` | True-date temporal borrowing NLL table. |
| `tables/table_true_date_temporal_nll_date_based.tex` | LaTeX-ready date-based temporal NLL table. |
| `tables/table_true_date_temporal_nll_rolling_origin.tex` | LaTeX-ready rolling-origin temporal NLL table. |
| `tables/feature_ablation_results.csv` | Deterministic feature-weight proxy ablation results. |
| `tables/table_feature_ablation.tex` | LaTeX-ready feature ablation table. |
| `tables/clinicaltrials_date_missingness_report.json` | True ClinicalTrials.gov date coverage and precision report. |
| `tables/gold_standard_scenarios.csv` | Gold-standard simulation, estimation scenarios S1-S5: bias, RMSE, coverage, borrowing-decision accuracy, oracle agreement. |
| `tables/gold_standard_design_worlds.csv` | Design-based operating characteristics: type I error (null world) and power (alternative world) across prior-data conflict. |
| `tables/gold_standard_external_control.csv` | External control arm scenario across external-control drift. |
| `tables/gold_standard_external_control_pcomp.csv` | External control sensitivity to the comparable fraction of the candidate pool. |
| `tables/gold_standard_dgm_diagnostics.json` | Trap / hidden-gem composition produced by the data-generating mechanism. |
| `tables/gold_standard_run_config.json` | Replicate counts, seed, design constants and conflict grid for the reported run. |
| `tables/table_gold_standard_design_oc.tex` | LaTeX-ready design-based operating-characteristics table. |
| `tables/table_gold_standard_external_control.tex` | LaTeX-ready external control table. |

## Figures

| File | Description |
|---|---|
| `figures/pipeline_diagram.svg`, `figures/pipeline_diagram.pdf` | Manuscript pipeline diagram. |
| `figures/simulation_oc_power_heatmap.svg`, `figures/simulation_oc_power_heatmap.pdf` | Simulation power heatmap across scenarios and methods. |
| `figures/feature_ablation_heatmap.svg`, `figures/feature_ablation_heatmap.pdf` | Feature ablation heatmap. |
| `figures/gold_standard_design_oc.svg` | Type I error and power against a fixed design null, versus prior-data conflict. |
| `figures/gold_standard_ess_response.svg` | Learned per-source ESS by true donor distance; shows the discount head is inert without the prospective conflict signal. |
| `figures/gold_standard_discrimination.svg` | Borrowing-decision ROC-AUC by scenario, against the parameter-level gold standard. |
| `figures/gold_standard_external_control.svg` | External control type I error, power and hybrid-control bias versus drift. |

## Interpretation

These results are retrospective predictive calibration and simulation evidence without expert borrowability labels. They should not be interpreted as expert validation, clinical validation, or regulatory qualification.

The `gold_standard_*` files come from a simulation in which the correct
borrowing decision is known by construction, and are the load-bearing evidence
for the borrowing layer. The remaining files are retrospective evidence on real
registry text; note the open defect documented in
`docs/KNOWN_ISSUE_endpoint_units.md`, which affects a minority of held-out ORR
values feeding the retrospective aggregates but does not touch the simulations.

To regenerate this directory from local artifacts, run:

```bash
python scripts/build_manuscript_evidence_package.py
python scripts/gold_standard_simulation/build_outputs.py
```
