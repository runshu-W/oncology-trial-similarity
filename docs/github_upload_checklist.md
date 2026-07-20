# GitHub Upload Checklist

This repository should be uploaded as a reproducible methodology prototype and retrospective predictive calibration evidence package. It should not be presented as a clinically validated borrowing recommendation system.

## Include

- `README.md`
- `.gitignore`
- `requirements.txt`
- `pipeline/oncology_trial_similarity_pipeline.py`
- `docs/methods_no_expert_validation.md`
- `docs/reproducibility.md`
- `docs/data_availability.md`
- `docs/manuscript_positioning.md`
- `docs/manuscript_evidence_plan.md`
- `docs/simulation_compute_notes.md`
- `docs/torch_temporal_validation_commands.md`
- `scripts/*.py` needed for extraction, validation, benchmarking, simulation, ablation, and evidence-package generation
- `scripts/run_no_expert_validation_suite.sh`
- `tests/test_*.py`
- `results/README.md`
- `results/tables/*.csv`
- `results/tables/*.tex`
- `results/figures/*.svg`
- `results/figures/*.pdf`

## Exclude

- `artifacts/`
- raw ClinicalTrials.gov JSON exports
- protocol and SAP PDFs
- large JSONL files
- embedding matrices such as `trial_embeddings.npz`
- model checkpoints such as `lambda_model.pt`
- LaTeX build outputs
- legacy manuscript figures in `docs/figures/`; the upload-ready figures are in `results/figures/`
- local virtual environments and caches

## External Storage Candidates

If a reviewer or collaborator needs full runtime artifacts, store them outside GitHub, for example on OSF, Zenodo, institutional storage, or a private data repository:

- full `pipeline_results.jsonl`
- full `lambda_training_examples.jsonl`
- trained model checkpoints
- raw ClinicalTrials.gov mirror
- protocol/SAP PDFs

## Pre-Upload Commands

Check ignored large files:

```bash
find . -maxdepth 3 -type f -size +5M -not -path './.git/*' -not -path './artifacts/*' -print
```

Check that large artifact paths are ignored:

```bash
git check-ignore -v artifacts/example.jsonl docs/figures/example.tiff docs/manuscript.pdf
```

Regenerate the lightweight evidence package:

```bash
python scripts/build_manuscript_evidence_package.py
```

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

## Required README Framing

Use:

> A reproducible methodology prototype and retrospective predictive calibration evidence package for oncology trial similarity and Bayesian historical borrowing.

Avoid:

> A clinically validated borrowing recommendation system.

## Known Current Limitation

The current local environment used to build `results/` did not include `torch`, so full date-based and rolling-origin temporal retraining was not run here. The commands for running those analyses in a torch-enabled environment are documented in `docs/torch_temporal_validation_commands.md`.
