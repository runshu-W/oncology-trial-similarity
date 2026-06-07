# Manuscript Evidence Package Plan

Target journals: Pharmaceutical Statistics or Statistics in Medicine.

## Core Figures and Tables

| Item | Output | Source | Intended manuscript role |
|---|---|---|---|
| Figure 1 pipeline diagram | `results/figures/pipeline_diagram.svg`, `results/figures/pipeline_diagram.pdf` | Pipeline architecture | Methods overview |
| Simulation OC table | `results/tables/table_simulation_operating_characteristics.tex` | `artifacts/operating_characteristics_simulation/` | Simulation study |
| Simulation power heatmap | `results/figures/simulation_oc_power_heatmap.svg`, `results/figures/simulation_oc_power_heatmap.pdf` | `simulation_operating_characteristics.csv` | Visual summary of operating characteristics |
| Paired Stage 1 table | `results/tables/table_paired_stage1_benchmark.tex` | `artifacts/paired_stage1_backend_benchmark/` | Retrieval benchmark |
| Borrowing baseline table | `results/tables/table_borrowing_baseline_head_to_head.tex` | `artifacts/borrowing_baseline_head_to_head/` | Predictive calibration baselines |
| True-date temporal NLL tables | `results/tables/table_true_date_temporal_nll_*.tex` | `artifacts/temporal_validation_true_dates/` | Temporal validation |
| Feature ablation table and heatmap | `results/tables/table_feature_ablation.tex`, `results/figures/feature_ablation_heatmap.svg`, `results/figures/feature_ablation_heatmap.pdf` | `artifacts/feature_ablation_sensitivity/` | Sensitivity analysis |

## Framing

All results are retrospective predictive calibration or simulation evidence without expert borrowability labels. They support method development and internal validation, not clinical deployment or regulatory qualification.
