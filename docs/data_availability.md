# Data Availability and Sharing Notes

This repository is designed to share code, methods documentation, and lightweight retrospective calibration results. It does not include the full local ClinicalTrials.gov oncology export, protocol PDFs, statistical analysis plan PDFs, large pipeline JSONL files, model checkpoints, or embedding matrices.

## Public Data Source

The underlying trial records are derived from ClinicalTrials.gov. Users should retrieve trial records directly from ClinicalTrials.gov or an institutionally approved local mirror. The scripts assume a local directory layout with one folder per NCT ID and a JSON file named `NCT*_data.json`.

## Files Not Included in Git

The following are intentionally excluded:

- raw ClinicalTrials.gov JSON exports;
- protocol and SAP PDFs;
- full `pipeline_results.jsonl` files;
- embedding matrices such as `trial_embeddings.npz`;
- model checkpoints such as `lambda_model.pt`;
- complete `artifacts/` runtime directories.

These files are large, environment-specific, and may be subject to separate data management or redistribution considerations.

## Lightweight Results Included

The `results/` directory contains selected lightweight result summaries suitable for manuscript review and reproducibility checks:

- simulation operating-characteristics CSV and LaTeX table;
- paired Stage 1 benchmark CSV and LaTeX table;
- borrowing baseline head-to-head CSV and LaTeX table;
- true-date temporal NLL CSV and LaTeX tables;
- feature ablation CSV, LaTeX table, and SVG heatmap;
- pipeline and simulation SVG figures.

These outputs are generated from local artifacts by:

```bash
python scripts/build_manuscript_evidence_package.py
```

## Manuscript Data Availability Statement Draft

The analysis uses ClinicalTrials.gov trial records and result tables. The raw trial records and associated protocol or statistical analysis plan PDFs are not redistributed in this repository. Code for extracting trial summaries, date metadata, borrowing components, retrospective pseudo-query examples, simulation operating characteristics, temporal validation summaries, paired retrieval benchmarks, and feature ablations is available in this repository. Lightweight summary tables and manuscript figures are provided in `results/`. Users can regenerate the full artifact set from their own ClinicalTrials.gov mirror by following `docs/reproducibility.md`.

## Limitations

Automated extraction from ClinicalTrials.gov records can introduce endpoint mapping, arm parsing, date parsing, and result usability errors. The current repository does not include expert adjudication labels for borrowability. Results should therefore be interpreted as retrospective predictive calibration and simulation evidence rather than expert-validated clinical evidence.
