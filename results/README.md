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

## Figures

| File | Description |
|---|---|
| `figures/pipeline_diagram.svg`, `figures/pipeline_diagram.pdf` | Manuscript pipeline diagram. |
| `figures/simulation_oc_power_heatmap.svg`, `figures/simulation_oc_power_heatmap.pdf` | Simulation power heatmap across scenarios and methods. |
| `figures/feature_ablation_heatmap.svg`, `figures/feature_ablation_heatmap.pdf` | Feature ablation heatmap. |

## Interpretation

These results are retrospective predictive calibration and simulation evidence without expert borrowability labels. They should not be interpreted as expert validation, clinical validation, or regulatory qualification.

To regenerate this directory from local artifacts, run:

```bash
python scripts/build_manuscript_evidence_package.py
```
