# 1. Pipeline

The end-to-end path from a ClinicalTrials.gov record to a Bayesian mixture
prior, plus the retrospective evaluation that runs on real registry text.

## Core library

| Module | Role |
|---|---|
| `oncology_trial_similarity_pipeline.py` | Record parsing, structured summaries, Stage 1 indexing and retrieval, Stage 2 explainable reranking. |
| `secret_retrieval.py` | SECRET-style section retrieval used to build the high-recall candidate pool. |
| `mixture_prior.py` | Beta-binomial components, mixture weights, ESS discounts, SAM adapter. |

These are imported by scripts here and in `../casestudy/` via
`REPO_ROOT / "pipeline"`.

## Stages

**Retrieval and reranking**

| Script | Purpose |
|---|---|
| `build_trial2vec_index.py` | Build the Stage 1 index. |
| `apply_secret_pool_rerank.py` | Form the SECRET pool and apply Stage 2 reranking. |
| `evaluate_stage1_retrieval.py` | Stage 1 retrieval metrics. |
| `run_paired_stage1_backend_benchmark.py` | Paired backend benchmark on common query IDs. |

**Borrowing model**

| Script | Purpose |
|---|---|
| `train_retrospective_lambda_model.py` | Train the two-head mixture-weight model. |
| `run_oncology_retrospective_lambda_training.py` | Pseudo-query construction, training, temporal validation modes. |
| `evaluate_retrospective_lambda_model.py` | Held-out evaluation of the trained model. |

**Temporal validation**

| Script | Purpose |
|---|---|
| `clinicaltrials_dates.py` | True registry date metadata and precision labels. |
| `temporal_validation.py` | Date-based and rolling-origin split utilities. |
| `attach_temporal_metadata_to_lambda_examples.py` | Attach dates to training examples. |
| `run_temporal_borrowing_validation.py` | True-date temporal borrowing NLL. |
| `run_temporal_split_summary.py` | Split composition summaries. |

**Method comparison and sensitivity on real data**

| Script | Purpose |
|---|---|
| `run_borrowing_baseline_comparison.py` | Head-to-head comparison of weak, rule, classical, SAM and trained two-head priors. |
| `run_feature_ablation_sensitivity.py` | Feature-group ablation and SECRET section-weight sensitivity. |

**Data quality**

| Script | Purpose |
|---|---|
| `fix_endpoint_units.py` | Unit-aware conversion of outcome rows to rates. Dispatches on the reported unit; raises rather than guessing on unrecognised units. |
| `audit_orr_units.py` | Quantifies the impact of the endpoint-unit defect and writes a per-query correction table. |

> **Open defect.** The rate conversion is reimplemented at four call sites and
> is wrong for percentage-reported outcomes. Diagnosed and quantified, not yet
> fixed, because a piecemeal fix would leave the codebase internally
> inconsistent and a correct one requires re-running the pipeline from the raw
> exports. See `../docs/KNOWN_ISSUE_endpoint_units.md`. The simulations in
> `../simulation/` do not touch this path.

## Related sections

- `../simulation/` — controlled evaluation with known borrowing truth
- `../casestudy/` — worked single-query comparisons on real trials
- `../results/` — the lightweight, git-tracked evidence package
