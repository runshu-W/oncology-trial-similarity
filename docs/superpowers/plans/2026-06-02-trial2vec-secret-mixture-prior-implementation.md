# Trial2Vec/SECRET Mixture Prior Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a backend-neutral Stage-1 retrieval path with Trial2Vec support, preserve the current Stage-2 reranker, and add a mixture-prior layer whose component weights can be trained by retrospective prediction.

**Architecture:** Keep the existing pipeline as the orchestration entry point, but split new statistical mixture-prior logic into a focused module. Add Trial2Vec as an optional retrieval backend that consumes the existing `trial_summaries.jsonl` index through a separate Trial2Vec embedding index. Preserve ClinicalBERT as the default path and treat SECRET as an explicit future backend that returns a clear unsupported-backend error until a protocol summarization pipeline exists.

**Tech Stack:** Python 3.12, NumPy, PyTorch, optional `trial2vec`, existing `transformers`/ClinicalBERT path, `unittest`, current FastAPI web agent.

---

## Scope Check

This plan implements the first useful revision of the approved design:

- ClinicalBERT remains the default Stage-1 backend.
- Trial2Vec is added as an optional Stage-1 backend.
- SECRET is represented in the backend interface but is not implemented in this revision.
- The current deterministic Stage-2 reranker is preserved.
- The current weighted beta-binomial output is preserved for backward compatibility.
- A new mixture-prior output is added beside the existing Bayesian output.
- Retrospective lambda training is implemented as an offline script that consumes precomputed query-result JSONL records, avoiding long end-to-end training runs inside unit tests.

## File Structure

Create:

- `docs/mixture_prior.py`  
  Pure statistical utilities for mixture-prior components, lambda normalization, beta-binomial predictive probabilities, posterior component updates, rule-based lambda baselines, and lambda-feature extraction.

- `scripts/build_trial2vec_index.py`  
  Converts existing `trial_summaries.jsonl` into the DataFrame columns expected by the installed Trial2Vec package, encodes summaries, and writes `trial2vec_embeddings.npz`.

- `scripts/train_retrospective_lambda_model.py`  
  Offline training script for a small PyTorch lambda scorer using retrospective beta-binomial predictive loss.

- `tests/test_stage1_backends.py`  
  Unit tests for backend selection, Trial2Vec row conversion, and unsupported SECRET behavior.

- `tests/test_mixture_prior.py`  
  Unit tests for mixture-prior math, lambda normalization, rule-based baseline, and posterior component weights.

- `tests/test_retrospective_lambda_training.py`  
  Unit tests for the retrospective predictive loss and lambda model on small synthetic examples.

Modify:

- `docs/oncology_trial_similarity_pipeline.py`  
  Add retrieval backend selection, Trial2Vec search from a Trial2Vec embedding index, SECRET unsupported-backend error, and mixture-prior integration into `add_bayesian_analysis`.

- `web_agent/app.py`  
  Keep the default backend unchanged. Accept an optional retrieval backend field only if the UI/API explicitly passes it.

- `tests/test_web_agent.py`  
  Adjust fake runner signatures only if `run_pipeline_search` receives the retrieval backend parameter.

- `README.md` and `web_agent/README.md`  
  Document the optional Trial2Vec backend and the new mixture-prior output.

## Task 1: Add Backend Selection Tests Before Touching Retrieval Code

**Files:**
- Create: `tests/test_stage1_backends.py`
- Modify later: `docs/oncology_trial_similarity_pipeline.py`

- [ ] **Step 1: Write failing tests for backend defaults and SECRET unsupported behavior**

Create `tests/test_stage1_backends.py` with this content:

```python
from __future__ import annotations

import importlib
import unittest


pipeline = importlib.import_module("docs.oncology_trial_similarity_pipeline")


class Stage1BackendTests(unittest.TestCase):
    def test_default_retrieval_backend_is_clinicalbert_compatible(self) -> None:
        self.assertEqual(pipeline.DEFAULT_RETRIEVAL_BACKEND, "clinicalbert")
        self.assertIn("clinicalbert", pipeline.RETRIEVAL_BACKENDS)
        self.assertIn("trial2vec", pipeline.RETRIEVAL_BACKENDS)
        self.assertIn("secret", pipeline.RETRIEVAL_BACKENDS)

    def test_secret_backend_is_explicitly_not_implemented(self) -> None:
        with self.assertRaisesRegex(NotImplementedError, "SECRET retrieval"):
            pipeline.ensure_supported_retrieval_backend("secret")

    def test_unknown_backend_has_clear_error(self) -> None:
        with self.assertRaisesRegex(ValueError, "Unsupported retrieval backend"):
            pipeline.ensure_supported_retrieval_backend("not_a_backend")


if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 2: Run the new test and verify it fails**

Run:

```bash
../.venv/bin/python -m unittest tests.test_stage1_backends -v
```

Expected: FAIL because `DEFAULT_RETRIEVAL_BACKEND`, `RETRIEVAL_BACKENDS`, and `ensure_supported_retrieval_backend` do not exist.

- [ ] **Step 3: Add the minimal backend constants and validation helper**

In `docs/oncology_trial_similarity_pipeline.py`, add the following near the existing `ASPECT_WEIGHTS` and `DEFAULT_CLINICALBERT_MODEL` constants:

```python
DEFAULT_RETRIEVAL_BACKEND = "clinicalbert"
RETRIEVAL_BACKENDS = ("clinicalbert", "trial2vec", "secret")


def ensure_supported_retrieval_backend(backend: str) -> None:
    if backend not in RETRIEVAL_BACKENDS:
        raise ValueError(
            f"Unsupported retrieval backend: {backend}. "
            f"Supported backends: {', '.join(RETRIEVAL_BACKENDS)}"
        )
    if backend == "secret":
        raise NotImplementedError(
            "SECRET retrieval is reserved for the protocol-summary backend and is not implemented in this revision."
        )
```

- [ ] **Step 4: Run the backend tests**

Run:

```bash
../.venv/bin/python -m unittest tests.test_stage1_backends -v
```

Expected: PASS for all 3 tests.

- [ ] **Step 5: Commit**

```bash
git add tests/test_stage1_backends.py docs/oncology_trial_similarity_pipeline.py
git commit -m "Add retrieval backend selection guardrails"
```

## Task 2: Add Trial2Vec Summary Conversion and Index Builder

**Files:**
- Modify: `tests/test_stage1_backends.py`
- Create: `scripts/build_trial2vec_index.py`
- Modify: `docs/oncology_trial_similarity_pipeline.py`

- [ ] **Step 1: Add tests for Trial2Vec row conversion**

Append this test method to `Stage1BackendTests` in `tests/test_stage1_backends.py`:

```python
    def test_trial2vec_row_uses_expected_columns(self) -> None:
        summary = {
            "nct_id": "NCT123",
            "brief_title": "A Trial of Drug A in Lung Cancer",
            "brief_summary": "This study evaluates Drug A.",
            "intervention": {
                "experimental_regimen": "Drug A",
                "drug_classes": ["Immunotherapy"],
            },
            "cancer_type": {
                "primary_site": ["Lung"],
                "histology": ["NSCLC"],
            },
            "endpoints": {
                "primary": [{"title": "Objective Response Rate"}],
            },
            "population": {
                "key_inclusion": ["Adults with measurable disease"],
                "key_exclusion": ["Active brain metastases"],
            },
            "status": "Completed",
        }

        row = pipeline.summary_to_trial2vec_row(summary)

        self.assertEqual(
            sorted(row),
            [
                "criteria",
                "description",
                "disease",
                "intervention_name",
                "keyword",
                "nct_id",
                "outcome_measure",
                "overall_status",
                "reference",
                "title",
            ],
        )
        self.assertEqual(row["nct_id"], "NCT123")
        self.assertIn("Drug A", row["intervention_name"])
        self.assertIn("Lung", row["disease"])
        self.assertIn("Objective Response Rate", row["outcome_measure"])
        self.assertIn("Adults with measurable disease", row["criteria"])
```

- [ ] **Step 2: Run the targeted test and verify it fails**

Run:

```bash
../.venv/bin/python -m unittest tests.test_stage1_backends.Stage1BackendTests.test_trial2vec_row_uses_expected_columns -v
```

Expected: FAIL because `summary_to_trial2vec_row` does not exist.

- [ ] **Step 3: Implement Trial2Vec row conversion**

In `docs/oncology_trial_similarity_pipeline.py`, add this function after `aspect_text`:

```python
TRIAL2VEC_COLUMNS = (
    "nct_id",
    "description",
    "title",
    "intervention_name",
    "disease",
    "keyword",
    "outcome_measure",
    "criteria",
    "reference",
    "overall_status",
)


def summary_to_trial2vec_row(summary: dict[str, Any]) -> dict[str, str]:
    endpoints = summary.get("endpoints", {}).get("primary", [])
    endpoint_titles = [endpoint.get("title", "") for endpoint in endpoints]
    population = summary.get("population", {})
    criteria = clean_text(
        {
            "inclusion": population.get("key_inclusion", []),
            "exclusion": population.get("key_exclusion", []),
        }
    )
    row = {
        "nct_id": clean_text(summary.get("nct_id", "")),
        "description": clean_text(
            [
                summary.get("brief_summary", ""),
                summary.get("one_paragraph_summary_for_embedding", ""),
            ]
        ),
        "title": clean_text(summary.get("brief_title", summary.get("title", ""))),
        "intervention_name": clean_text(summary.get("intervention", {})),
        "disease": clean_text(summary.get("cancer_type", {})),
        "keyword": clean_text(
            [
                summary.get("phase", ""),
                summary.get("status", ""),
                summary.get("design", {}),
            ]
        ),
        "outcome_measure": clean_text(endpoint_titles),
        "criteria": criteria,
        "reference": "",
        "overall_status": clean_text(summary.get("status", "")),
    }
    return {column: row[column] for column in TRIAL2VEC_COLUMNS}
```

- [ ] **Step 4: Create the Trial2Vec index builder script**

Create `scripts/build_trial2vec_index.py` with this content:

```python
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT / "docs") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "docs"))

import oncology_trial_similarity_pipeline as pipeline  # noqa: E402


def load_summary_rows(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def encode_trial2vec_index(
    summaries_path: Path,
    output_path: Path,
    model_dir: Path,
    device: str = "cpu",
) -> dict[str, Any]:
    from trial2vec import Trial2Vec

    summaries = load_summary_rows(summaries_path)
    frame = pd.DataFrame([pipeline.summary_to_trial2vec_row(summary) for summary in summaries])
    model = Trial2Vec(device=device)

    original_torch_load = torch.load
    original_load_state_dict = torch.nn.Module.load_state_dict

    def compatible_torch_load(*args: Any, **kwargs: Any) -> Any:
        kwargs.setdefault("weights_only", False)
        return original_torch_load(*args, **kwargs)

    def compatible_load_state_dict(module: torch.nn.Module, state_dict: dict[str, Any], *args: Any, **kwargs: Any) -> Any:
        kwargs.setdefault("strict", False)
        return original_load_state_dict(module, state_dict, *args, **kwargs)

    torch.load = compatible_torch_load
    torch.nn.Module.load_state_dict = compatible_load_state_dict
    try:
        model.from_pretrained(str(model_dir))
    finally:
        torch.load = original_torch_load
        torch.nn.Module.load_state_dict = original_load_state_dict

    tags, embeddings = model.encode({"x": frame}, return_dict=False)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output_path,
        nct_ids=np.array(tags),
        embeddings=embeddings.astype(np.float32),
        retrieval_backend=np.array(["trial2vec"]),
        model_dir=np.array([str(model_dir)]),
    )
    return {
        "summary_count": len(summaries),
        "embedding_shape": list(embeddings.shape),
        "output_path": str(output_path),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Build a Trial2Vec retrieval index from trial_summaries.jsonl.")
    parser.add_argument("--summaries-path", type=Path, required=True)
    parser.add_argument("--output-path", type=Path, required=True)
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()
    summary = encode_trial2vec_index(args.summaries_path, args.output_path, args.model_dir, device=args.device)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
```

- [ ] **Step 5: Run conversion tests and script import check**

Run:

```bash
../.venv/bin/python -m unittest tests.test_stage1_backends -v
../.venv/bin/python -m py_compile scripts/build_trial2vec_index.py
```

Expected: both commands pass.

- [ ] **Step 6: Commit**

```bash
git add tests/test_stage1_backends.py docs/oncology_trial_similarity_pipeline.py scripts/build_trial2vec_index.py
git commit -m "Add Trial2Vec summary conversion and index builder"
```

## Task 3: Add Trial2Vec Retrieval Search Path

**Files:**
- Modify: `tests/test_stage1_backends.py`
- Modify: `docs/oncology_trial_similarity_pipeline.py`

- [ ] **Step 1: Add a test for Trial2Vec index scoring**

Append this test method to `Stage1BackendTests`:

```python
    def test_trial2vec_index_search_uses_cosine_scores(self) -> None:
        import tempfile
        from pathlib import Path

        import numpy as np

        query_vector = np.array([1.0, 0.0], dtype=np.float32)
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "trial2vec_embeddings.npz"
            np.savez_compressed(
                path,
                nct_ids=np.array(["NCT1", "NCT2"]),
                embeddings=np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32),
                retrieval_backend=np.array(["trial2vec"]),
            )

            rows = pipeline.score_trial2vec_index(query_vector, path, excluded_nct_id="NCTQUERY", top_k=2)

        self.assertEqual([row["nct_id"] for row in rows], ["NCT1", "NCT2"])
        self.assertEqual(rows[0]["retrieval_backend"], "trial2vec")
        self.assertEqual(rows[0]["score_0_100"], 100.0)
        self.assertEqual(rows[1]["score_0_100"], 0.0)
```

- [ ] **Step 2: Run the targeted test and verify it fails**

Run:

```bash
../.venv/bin/python -m unittest tests.test_stage1_backends.Stage1BackendTests.test_trial2vec_index_search_uses_cosine_scores -v
```

Expected: FAIL because `score_trial2vec_index` does not exist.

- [ ] **Step 3: Implement Trial2Vec index scoring helper**

Add this function in `docs/oncology_trial_similarity_pipeline.py` near `weighted_similarity`:

```python
def score_trial2vec_index(
    query_vector: np.ndarray,
    trial2vec_index_path: Path,
    excluded_nct_id: str,
    top_k: int,
) -> list[dict[str, Any]]:
    if not trial2vec_index_path.exists():
        raise FileNotFoundError(f"Trial2Vec index not found: {trial2vec_index_path}")
    data = np.load(trial2vec_index_path, allow_pickle=False)
    nct_ids = data["nct_ids"]
    embeddings = data["embeddings"]
    rows = []
    for idx, nct_id_raw in enumerate(nct_ids):
        nct_id = str(nct_id_raw)
        if nct_id == excluded_nct_id:
            continue
        score = cosine(query_vector, embeddings[idx])
        rows.append(
            {
                "nct_id": nct_id,
                "score": score,
                "score_0_100": round(100 * max(0.0, score), 2),
                "aspect_scores": {},
                "retrieval_backend": "trial2vec",
            }
        )
    rows.sort(key=lambda row: row["score"], reverse=True)
    for rank, row in enumerate(rows[:top_k], start=1):
        row["retrieval_rank"] = rank
    return rows[:top_k]
```

- [ ] **Step 4: Extend `search` signature for retrieval backend**

Change the `search` signature to include:

```python
    retrieval_backend: str = DEFAULT_RETRIEVAL_BACKEND,
    trial2vec_index_path: Path | None = None,
    trial2vec_model_dir: Path | None = None,
```

At the start of `search`, add:

```python
    ensure_supported_retrieval_backend(retrieval_backend)
```

The ClinicalBERT branch should keep the existing behavior. The Trial2Vec branch should:

1. Convert `query_summary` to a Trial2Vec row.
2. Load the pretrained Trial2Vec model from `trial2vec_model_dir`.
3. Encode the query row.
4. Call `score_trial2vec_index`.
5. Enrich scored rows with the same candidate summary fields currently appended in the ClinicalBERT branch.

Use this enrichment helper to avoid duplicating fields:

```python
def enrich_candidate_row(row: dict[str, Any], candidate_summary: dict[str, Any]) -> dict[str, Any]:
    borrowing = candidate_summary.get("borrowing_relevance", {})
    return {
        **row,
        "title": candidate_summary.get("brief_title", ""),
        "phase": candidate_summary.get("phase", ""),
        "status": candidate_summary.get("status", ""),
        "cancer_type": candidate_summary.get("cancer_type", {}),
        "population": candidate_summary.get("population", {}),
        "intervention": candidate_summary.get("intervention", {}),
        "design": candidate_summary.get("design", {}),
        "endpoints": candidate_summary.get("endpoints", {}),
        "results": candidate_summary.get("results", {}),
        "result_usability": {
            "has_posted_results": candidate_summary.get("results", {}).get("has_posted_results"),
            "denominators_available": bool(candidate_summary.get("results", {}).get("denominators")),
            "source_documents": candidate_summary.get("source_documents", {}),
        },
        "similarity_drivers": borrowing.get("major_similarity_drivers", []),
        "nonborrowability_risks": borrowing.get("major_nonborrowability_risks", []),
        "borrowable_quantities": borrowing.get("borrowable_quantities", []),
    }
```

- [ ] **Step 5: Add CLI flags**

In the `search` argparse block, add:

```python
    query.add_argument(
        "--retrieval-backend",
        choices=list(RETRIEVAL_BACKENDS),
        default=DEFAULT_RETRIEVAL_BACKEND,
        help="Stage-1 retrieval backend. clinicalbert uses the existing embedding index; trial2vec uses --trial2vec-index-path.",
    )
    query.add_argument("--trial2vec-index-path", type=Path, default=None)
    query.add_argument("--trial2vec-model-dir", type=Path, default=None)
```

Pass the new arguments into `search`.

- [ ] **Step 6: Run backend tests and existing web tests**

Run:

```bash
../.venv/bin/python -m unittest tests.test_stage1_backends -v
../.venv/bin/python -m unittest tests.test_web_agent.WebAgentTests.test_run_pipeline_search_passes_temp_query_file_to_pipeline -v
```

Expected: both commands pass. If the web test fails because the fake `pipeline.search` signature changed, update the fake function to accept the new keyword arguments with explicit defaults:

```python
def fake_search(
    query_json,
    index_dir,
    top_k,
    rerank_top_n,
    embedding_backend=None,
    embedding_model=None,
    embedding_batch_size=16,
    embedding_max_length=256,
    retrieval_backend="clinicalbert",
    trial2vec_index_path=None,
    trial2vec_model_dir=None,
):
    self.assertEqual(retrieval_backend, "clinicalbert")
    return {"query": {"nct_id": "NCTTEMP"}, "reranked_top10": []}
```

- [ ] **Step 7: Commit**

```bash
git add docs/oncology_trial_similarity_pipeline.py tests/test_stage1_backends.py tests/test_web_agent.py
git commit -m "Add Trial2Vec retrieval search path"
```

## Task 4: Add Pure Mixture-Prior Math

**Files:**
- Create: `docs/mixture_prior.py`
- Create: `tests/test_mixture_prior.py`

- [ ] **Step 1: Write tests for beta-binomial predictive probability and lambda normalization**

Create `tests/test_mixture_prior.py` with this content:

```python
from __future__ import annotations

import importlib
import math
import unittest


mixture_prior = importlib.import_module("docs.mixture_prior")


class MixturePriorMathTests(unittest.TestCase):
    def test_beta_binomial_predictive_probability_matches_closed_form(self) -> None:
        value = mixture_prior.beta_binomial_predictive_probability(
            y=2,
            n=4,
            alpha=1.0,
            beta=1.0,
        )

        self.assertAlmostEqual(value, 0.2, places=6)

    def test_normalize_lambdas_reserves_weak_component(self) -> None:
        lambdas = mixture_prior.normalize_lambdas([2.0, 1.0, 0.0], lambda0=0.2)

        self.assertAlmostEqual(lambdas["lambda_0"], 0.2)
        self.assertAlmostEqual(sum(lambdas["lambda_i"]), 0.8)
        self.assertGreater(lambdas["lambda_i"][0], lambdas["lambda_i"][1])
        self.assertEqual(lambdas["lambda_i"][2], 0.0)

    def test_normalize_lambdas_returns_weak_only_when_all_raw_weights_are_zero(self) -> None:
        lambdas = mixture_prior.normalize_lambdas([0.0, 0.0], lambda0=0.2)

        self.assertEqual(lambdas["lambda_0"], 1.0)
        self.assertEqual(lambdas["lambda_i"], [0.0, 0.0])

    def test_mixture_predictive_probability_is_weighted_sum(self) -> None:
        components = [
            {"alpha": 1.0, "beta": 1.0, "lambda": 0.5},
            {"alpha": 3.0, "beta": 1.0, "lambda": 0.3},
        ]

        probability = mixture_prior.mixture_predictive_probability(
            y=2,
            n=4,
            lambda0=0.2,
            weak_alpha=1.0,
            weak_beta=1.0,
            components=components,
        )

        expected = (
            0.2 * mixture_prior.beta_binomial_predictive_probability(2, 4, 1.0, 1.0)
            + 0.5 * mixture_prior.beta_binomial_predictive_probability(2, 4, 1.0, 1.0)
            + 0.3 * mixture_prior.beta_binomial_predictive_probability(2, 4, 3.0, 1.0)
        )
        self.assertTrue(math.isfinite(probability))
        self.assertAlmostEqual(probability, expected, places=8)


if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 2: Run tests and verify they fail**

Run:

```bash
../.venv/bin/python -m unittest tests.test_mixture_prior -v
```

Expected: FAIL because `docs.mixture_prior` does not exist.

- [ ] **Step 3: Implement `docs/mixture_prior.py`**

Create `docs/mixture_prior.py` with this content:

```python
from __future__ import annotations

import math
from typing import Any


def _log_choose(n: int, y: int) -> float:
    return math.lgamma(n + 1) - math.lgamma(y + 1) - math.lgamma(n - y + 1)


def _log_beta(alpha: float, beta: float) -> float:
    return math.lgamma(alpha) + math.lgamma(beta) - math.lgamma(alpha + beta)


def beta_binomial_predictive_probability(y: int, n: int, alpha: float, beta: float) -> float:
    if n < 0:
        raise ValueError("n must be non-negative")
    if y < 0 or y > n:
        raise ValueError("y must satisfy 0 <= y <= n")
    if alpha <= 0 or beta <= 0:
        raise ValueError("alpha and beta must be positive")
    log_probability = (
        _log_choose(n, y)
        + _log_beta(y + alpha, n - y + beta)
        - _log_beta(alpha, beta)
    )
    return float(math.exp(log_probability))


def normalize_lambdas(raw_weights: list[float], lambda0: float = 0.2) -> dict[str, Any]:
    if lambda0 < 0 or lambda0 > 1:
        raise ValueError("lambda0 must be in [0, 1]")
    clipped = [max(0.0, float(value)) for value in raw_weights]
    total = sum(clipped)
    if total <= 0:
        return {"lambda_0": 1.0, "lambda_i": [0.0 for _ in clipped]}
    scale = 1.0 - lambda0
    return {
        "lambda_0": lambda0,
        "lambda_i": [scale * value / total for value in clipped],
    }


def mixture_predictive_probability(
    y: int,
    n: int,
    lambda0: float,
    weak_alpha: float,
    weak_beta: float,
    components: list[dict[str, float]],
) -> float:
    probability = lambda0 * beta_binomial_predictive_probability(y, n, weak_alpha, weak_beta)
    for component in components:
        probability += component["lambda"] * beta_binomial_predictive_probability(
            y,
            n,
            component["alpha"],
            component["beta"],
        )
    return float(probability)


def posterior_component_weights(
    y: int,
    n: int,
    lambda0: float,
    weak_alpha: float,
    weak_beta: float,
    components: list[dict[str, float]],
) -> dict[str, Any]:
    weak_predictive = beta_binomial_predictive_probability(y, n, weak_alpha, weak_beta)
    numerators = [
        component["lambda"] * beta_binomial_predictive_probability(y, n, component["alpha"], component["beta"])
        for component in components
    ]
    weak_numerator = lambda0 * weak_predictive
    denominator = weak_numerator + sum(numerators)
    if denominator <= 0:
        return {"lambda_0_post": 1.0, "lambda_i_post": [0.0 for _ in components]}
    return {
        "lambda_0_post": weak_numerator / denominator,
        "lambda_i_post": [value / denominator for value in numerators],
    }
```

- [ ] **Step 4: Run mixture-prior tests**

Run:

```bash
../.venv/bin/python -m unittest tests.test_mixture_prior -v
```

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add docs/mixture_prior.py tests/test_mixture_prior.py
git commit -m "Add mixture prior math utilities"
```

## Task 5: Add Rule-Based Lambda Components From Reranked Candidates

**Files:**
- Modify: `docs/mixture_prior.py`
- Modify: `tests/test_mixture_prior.py`

- [ ] **Step 1: Add tests for component extraction and rule weights**

Append this class to `tests/test_mixture_prior.py`:

```python
class MixturePriorComponentTests(unittest.TestCase):
    def test_components_from_reranked_rows_use_stage2_fields(self) -> None:
        rows = [
            {
                "candidate_nct_id": "NCTHIST1",
                "overall_similarity_score": 80.0,
                "suggested_borrowing_discount": 0.75,
                "dimension_scores": {
                    "disease_biology_match": 5.0,
                    "endpoint_estimand_match": 5.0,
                    "result_usability": 5.0,
                },
                "red_flags": [],
                "borrowable_quantities": [
                    {
                        "endpoint": "Objective Response Rate",
                        "endpoint_family": "ORR/CR/PR",
                        "arm_results": [{"arm": "Experimental", "count": 20, "denominator": 50}],
                    }
                ],
            },
            {
                "candidate_nct_id": "NCTHIST2",
                "overall_similarity_score": 65.0,
                "suggested_borrowing_discount": 0.4,
                "dimension_scores": {
                    "disease_biology_match": 1.0,
                    "endpoint_estimand_match": 5.0,
                    "result_usability": 5.0,
                },
                "red_flags": ["Low disease biology match."],
                "borrowable_quantities": [
                    {
                        "endpoint": "Objective Response Rate",
                        "endpoint_family": "ORR/CR/PR",
                        "arm_results": [{"arm": "Experimental", "count": 8, "denominator": 40}],
                    }
                ],
            },
        ]

        components = mixture_prior.components_from_reranked_rows(rows, endpoint_key="ORR", lambda0=0.2)

        self.assertEqual(len(components["components"]), 2)
        self.assertAlmostEqual(components["lambda_0"], 0.2)
        self.assertAlmostEqual(sum(row["lambda_rule"] for row in components["components"]), 0.8)
        self.assertGreater(components["components"][0]["lambda_rule"], components["components"][1]["lambda_rule"])
        self.assertEqual(components["components"][0]["alpha"], 16.0)
        self.assertEqual(components["components"][0]["beta"], 23.5)
```

- [ ] **Step 2: Run the targeted component test and verify it fails**

Run:

```bash
../.venv/bin/python -m unittest tests.test_mixture_prior.MixturePriorComponentTests.test_components_from_reranked_rows_use_stage2_fields -v
```

Expected: FAIL because `components_from_reranked_rows` does not exist.

- [ ] **Step 3: Implement component extraction**

Append these functions to `docs/mixture_prior.py`:

```python
def _canonical_endpoint_key(family: str, title: str, time_frame: str = "") -> str | None:
    text = f"{family} {title} {time_frame}".lower()
    if "orr" in text or "response" in text or "complete response" in text:
        return "ORR"
    if "pfs6" in text or ("progression" in text and "6" in text and "month" in text):
        return "PFS6"
    return None


def _selected_treatment_observation(rows: list[dict[str, Any]]) -> dict[str, float] | None:
    for row in rows:
        arm = str(row.get("arm", "")).lower()
        count = row.get("count")
        denominator = row.get("denominator")
        if count is None or denominator is None:
            continue
        if "placebo" in arm or "control" in arm or "comparator" in arm:
            continue
        return {
            "count": float(count),
            "denominator": float(denominator),
            "rate": float(count) / float(denominator) if float(denominator) > 0 else 0.0,
        }
    return None


def conservative_gate(dimension_scores: dict[str, float], red_flags: list[str]) -> float:
    disease = float(dimension_scores.get("disease_biology_match", 0.0))
    endpoint = float(dimension_scores.get("endpoint_estimand_match", 0.0))
    result = float(dimension_scores.get("result_usability", 0.0))
    if endpoint < 1.5 or result <= 0:
        return 0.0
    gate = 1.0
    if disease < 1.5:
        gate *= 0.2
    elif disease < 2.5:
        gate *= 0.6
    if any("Low " in flag for flag in red_flags):
        gate *= 0.5
    return gate


def components_from_reranked_rows(
    rows: list[dict[str, Any]],
    endpoint_key: str,
    lambda0: float = 0.2,
) -> dict[str, Any]:
    components: list[dict[str, Any]] = []
    raw_weights: list[float] = []
    for row in rows[:10]:
        discount = max(0.0, min(1.0, float(row.get("suggested_borrowing_discount") or 0.0)))
        if discount <= 0:
            continue
        selected_observation = None
        selected_endpoint = ""
        for quantity in row.get("borrowable_quantities", []):
            candidate_key = _canonical_endpoint_key(
                str(quantity.get("endpoint_family", "")),
                str(quantity.get("endpoint", "")),
                str(quantity.get("time_frame", "")),
            )
            if candidate_key != endpoint_key:
                continue
            selected_observation = _selected_treatment_observation(quantity.get("arm_results", []))
            selected_endpoint = str(quantity.get("endpoint", ""))
            if selected_observation is not None:
                break
        if selected_observation is None:
            continue
        y_i = selected_observation["count"]
        n_i = selected_observation["denominator"]
        dimension_scores = {
            key: float(value)
            for key, value in (row.get("dimension_scores") or {}).items()
            if isinstance(value, (int, float))
        }
        gate = conservative_gate(dimension_scores, list(row.get("red_flags") or []))
        overall = float(row.get("overall_similarity_score") or 0.0)
        raw_weight = gate * discount * max(0.0, overall) / 100.0 * math.log1p(n_i)
        components.append(
            {
                "nct_id": row.get("candidate_nct_id", row.get("nct_id", "")),
                "endpoint": selected_endpoint,
                "count": y_i,
                "denominator": n_i,
                "rate": selected_observation["rate"],
                "discount": discount,
                "gate": gate,
                "overall_similarity_score": overall,
                "alpha": 1.0 + discount * y_i,
                "beta": 1.0 + discount * (n_i - y_i),
                "raw_rule_weight": raw_weight,
            }
        )
        raw_weights.append(raw_weight)
    normalized = normalize_lambdas(raw_weights, lambda0=lambda0)
    for component, lambda_rule in zip(components, normalized["lambda_i"]):
        component["lambda_rule"] = lambda_rule
    return {"lambda_0": normalized["lambda_0"], "components": components}
```

- [ ] **Step 4: Run mixture-prior tests**

Run:

```bash
../.venv/bin/python -m unittest tests.test_mixture_prior -v
```

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add docs/mixture_prior.py tests/test_mixture_prior.py
git commit -m "Add rule-based mixture prior components"
```

## Task 6: Integrate Mixture Prior Into Bayesian Analysis Output

**Files:**
- Modify: `docs/oncology_trial_similarity_pipeline.py`
- Modify: `tests/test_web_agent.py`

- [ ] **Step 1: Add a test that `add_bayesian_analysis` includes mixture prior output**

In `tests/test_web_agent.py`, add this assertion block inside `test_bayesian_analysis_summarizes_weighted_orr_borrowing` after `endpoint = analysis["endpoint_analyses"][0]`:

```python
        mixture = endpoint["mixture_prior"]
        self.assertIn("lambda_0", mixture)
        self.assertIn("components", mixture)
        self.assertGreaterEqual(mixture["lambda_0"], 0.0)
        self.assertLessEqual(mixture["lambda_0"], 1.0)
        self.assertAlmostEqual(
            mixture["lambda_0"] + sum(component["lambda_rule"] for component in mixture["components"]),
            1.0,
            places=6,
        )
```

- [ ] **Step 2: Run the targeted web-agent test and verify it fails**

Run:

```bash
../.venv/bin/python -m unittest tests.test_web_agent.WebAgentTests.test_bayesian_analysis_summarizes_weighted_orr_borrowing -v
```

Expected: FAIL because `endpoint["mixture_prior"]` is missing.

- [ ] **Step 3: Import the mixture module**

At the top of `docs/oncology_trial_similarity_pipeline.py`, add:

```python
try:
    import mixture_prior
except ImportError:
    mixture_prior = None
```

This works because scripts and the web agent already add `docs/` to the import path.

- [ ] **Step 4: Add mixture prior output in `add_bayesian_analysis`**

Inside the dictionary assigned to `analysis` in `add_bayesian_analysis`, immediately after the `target_defaults` entry, add:

```python
            "mixture_prior": (
                mixture_prior.components_from_reranked_rows(rows, endpoint_key, lambda0=0.2)
                if mixture_prior is not None
                else {"lambda_0": 1.0, "components": []}
            ),
```

- [ ] **Step 5: Run targeted and publication tests**

Run:

```bash
../.venv/bin/python -m unittest tests.test_web_agent.WebAgentTests.test_bayesian_analysis_summarizes_weighted_orr_borrowing -v
../.venv/bin/python -m unittest tests.test_publication_evaluation.PublicationEvaluationTests.test_bayesian_sensitivity_includes_ess_cap -v
```

Expected: both commands pass. Existing weighted power-prior sensitivity rows must remain unchanged.

- [ ] **Step 6: Commit**

```bash
git add docs/oncology_trial_similarity_pipeline.py tests/test_web_agent.py
git commit -m "Expose mixture prior components in Bayesian output"
```

## Task 7: Add Retrospective Lambda Training Utilities

**Files:**
- Create: `scripts/train_retrospective_lambda_model.py`
- Create: `tests/test_retrospective_lambda_training.py`

- [ ] **Step 1: Write tests for predictive loss and lambda model shape**

Create `tests/test_retrospective_lambda_training.py` with this content:

```python
from __future__ import annotations

import importlib.util
import unittest
from pathlib import Path

import torch


SCRIPT_PATH = Path(__file__).resolve().parents[1] / "scripts" / "train_retrospective_lambda_model.py"


def load_training_module():
    spec = importlib.util.spec_from_file_location("train_retrospective_lambda_model", SCRIPT_PATH)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


class RetrospectiveLambdaTrainingTests(unittest.TestCase):
    def test_lambda_model_outputs_one_score_per_candidate(self) -> None:
        module = load_training_module()
        model = module.LambdaScorer(input_dim=4, hidden_dim=6)
        features = torch.zeros((3, 4), dtype=torch.float32)

        scores = model(features)

        self.assertEqual(tuple(scores.shape), (3,))

    def test_predictive_loss_is_finite_for_synthetic_example(self) -> None:
        module = load_training_module()
        example = {
            "query": {"count": 2, "denominator": 4},
            "lambda_0": 0.2,
            "features": [[0.9, 0.8, 1.0, 3.9], [0.1, 0.2, 0.5, 2.0]],
            "components": [
                {"alpha": 3.0, "beta": 3.0, "gate": 1.0},
                {"alpha": 1.2, "beta": 4.8, "gate": 1.0},
            ],
            "lambda_rule": [0.6, 0.2],
        }
        model = module.LambdaScorer(input_dim=4, hidden_dim=6)

        loss = module.predictive_loss_for_example(model, example, rho=0.1, ess_cap=100.0)

        self.assertTrue(torch.isfinite(loss))
        self.assertGreater(float(loss.detach()), 0.0)


if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 2: Run the tests and verify they fail**

Run:

```bash
../.venv/bin/python -m unittest tests.test_retrospective_lambda_training -v
```

Expected: FAIL because the training script does not exist.

- [ ] **Step 3: Implement the training script**

Create `scripts/train_retrospective_lambda_model.py` with this content:

```python
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import torch


class LambdaScorer(torch.nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int = 16) -> None:
        super().__init__()
        self.network = torch.nn.Sequential(
            torch.nn.Linear(input_dim, hidden_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_dim, 1),
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.network(features).squeeze(-1)


def _log_choose(n: int, y: int) -> float:
    return math.lgamma(n + 1) - math.lgamma(y + 1) - math.lgamma(n - y + 1)


def _log_beta(alpha: torch.Tensor, beta: torch.Tensor) -> torch.Tensor:
    return torch.lgamma(alpha) + torch.lgamma(beta) - torch.lgamma(alpha + beta)


def beta_binomial_log_predictive(y: int, n: int, alpha: torch.Tensor, beta: torch.Tensor) -> torch.Tensor:
    log_choose = torch.tensor(_log_choose(n, y), dtype=alpha.dtype, device=alpha.device)
    return log_choose + _log_beta(alpha + y, beta + n - y) - _log_beta(alpha, beta)


def predictive_loss_for_example(
    model: LambdaScorer,
    example: dict[str, Any],
    rho: float = 0.1,
    ess_cap: float = 100.0,
) -> torch.Tensor:
    features = torch.tensor(example["features"], dtype=torch.float32)
    components = example["components"]
    alpha = torch.tensor([component["alpha"] for component in components], dtype=torch.float32)
    beta = torch.tensor([component["beta"] for component in components], dtype=torch.float32)
    gates = torch.tensor([component["gate"] for component in components], dtype=torch.float32)
    denominator = torch.tensor([component.get("denominator", 0.0) for component in components], dtype=torch.float32)
    discount = torch.tensor([component.get("discount", 0.0) for component in components], dtype=torch.float32)
    lambda0 = float(example.get("lambda_0", 0.2))
    scores = model(features)
    raw = gates * torch.exp(scores)
    if float(torch.sum(raw).detach()) <= 0.0:
        lambda_i = torch.zeros_like(raw)
        lambda0_tensor = torch.tensor(1.0, dtype=torch.float32)
    else:
        lambda_i = (1.0 - lambda0) * raw / torch.sum(raw)
        lambda0_tensor = torch.tensor(lambda0, dtype=torch.float32)
    query = example["query"]
    y = int(query["count"])
    n = int(query["denominator"])
    weak_log_pred = beta_binomial_log_predictive(
        y,
        n,
        torch.tensor(1.0, dtype=torch.float32),
        torch.tensor(1.0, dtype=torch.float32),
    )
    component_log_pred = beta_binomial_log_predictive(y, n, alpha, beta)
    mixture_terms = torch.cat(
        [
            torch.log(lambda0_tensor.clamp_min(1e-12)).view(1) + weak_log_pred.view(1),
            torch.log(lambda_i.clamp_min(1e-12)) + component_log_pred,
        ]
    )
    predictive_loss = -torch.logsumexp(mixture_terms, dim=0)
    lambda_rule = torch.tensor(example.get("lambda_rule", [0.0 for _ in components]), dtype=torch.float32)
    if float(torch.sum(lambda_rule).detach()) > 0.0:
        lambda_rule = lambda_rule / torch.sum(lambda_rule) * max(0.0, 1.0 - lambda0)
        kl = torch.sum(lambda_rule * (torch.log(lambda_rule.clamp_min(1e-12)) - torch.log(lambda_i.clamp_min(1e-12))))
    else:
        kl = torch.tensor(0.0, dtype=torch.float32)
    ess = torch.sum(lambda_i * discount * denominator)
    ess_penalty = torch.relu(ess - ess_cap) ** 2
    return predictive_loss + rho * kl + 1e-4 * ess_penalty


def load_examples(path: Path) -> list[dict[str, Any]]:
    examples = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                examples.append(json.loads(line))
    return examples


def train_model(
    examples: list[dict[str, Any]],
    epochs: int,
    learning_rate: float,
    hidden_dim: int,
) -> dict[str, Any]:
    if not examples:
        raise ValueError("At least one training example is required")
    input_dim = len(examples[0]["features"][0])
    model = LambdaScorer(input_dim=input_dim, hidden_dim=hidden_dim)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    history = []
    for epoch in range(epochs):
        optimizer.zero_grad()
        losses = [predictive_loss_for_example(model, example) for example in examples]
        loss = torch.stack(losses).mean()
        loss.backward()
        optimizer.step()
        history.append(float(loss.detach()))
    return {
        "epochs": epochs,
        "final_loss": history[-1],
        "loss_history": history,
        "input_dim": input_dim,
        "hidden_dim": hidden_dim,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Train lambda scorer with retrospective beta-binomial predictive loss.")
    parser.add_argument("--examples-jsonl", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--learning-rate", type=float, default=0.01)
    parser.add_argument("--hidden-dim", type=int, default=16)
    args = parser.parse_args()
    examples = load_examples(args.examples_jsonl)
    summary = train_model(examples, args.epochs, args.learning_rate, args.hidden_dim)
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run training tests**

Run:

```bash
../.venv/bin/python -m unittest tests.test_retrospective_lambda_training -v
../.venv/bin/python -m py_compile scripts/train_retrospective_lambda_model.py
```

Expected: both commands pass.

- [ ] **Step 5: Commit**

```bash
git add scripts/train_retrospective_lambda_model.py tests/test_retrospective_lambda_training.py
git commit -m "Add retrospective lambda training utilities"
```

## Task 8: Add Retrospective Example Builder From Pipeline Outputs

**Files:**
- Modify: `docs/mixture_prior.py`
- Modify: `scripts/train_retrospective_lambda_model.py`
- Modify: `tests/test_retrospective_lambda_training.py`

- [ ] **Step 1: Add a test that pipeline output becomes one training example**

Append this test method to `RetrospectiveLambdaTrainingTests`:

```python
    def test_build_training_example_from_pipeline_result(self) -> None:
        module = load_training_module()
        result = {
            "query_summary": {
                "nct_id": "NCTQUERY",
                "endpoints": {
                    "primary": [
                        {
                            "title": "Objective Response Rate",
                            "endpoint_family": "ORR/CR/PR",
                            "arm_results": [{"arm": "Experimental", "count": 12, "denominator": 40}],
                        }
                    ]
                },
            },
            "reranked_top_matches": [
                {
                    "candidate_nct_id": "NCTHIST",
                    "overall_similarity_score": 82.0,
                    "retrieval_score": 98.0,
                    "suggested_borrowing_discount": 0.75,
                    "dimension_scores": {
                        "disease_biology_match": 5.0,
                        "treatment_regimen_match": 4.0,
                        "endpoint_estimand_match": 5.0,
                        "outcome_assessment_followup": 2.0,
                        "eligibility_criteria_overlap": 1.0,
                        "result_usability": 5.0,
                    },
                    "red_flags": [],
                    "borrowable_quantities": [
                        {
                            "endpoint": "Objective Response Rate",
                            "endpoint_family": "ORR/CR/PR",
                            "arm_results": [{"arm": "Experimental", "count": 20, "denominator": 50}],
                        }
                    ],
                }
            ],
        }

        example = module.build_training_example_from_pipeline_result(result, endpoint_key="ORR")

        self.assertEqual(example["query"], {"count": 12, "denominator": 40})
        self.assertEqual(len(example["features"]), 1)
        self.assertEqual(len(example["components"]), 1)
        self.assertGreater(example["features"][0][0], 0.0)
```

- [ ] **Step 2: Run the targeted test and verify it fails**

Run:

```bash
../.venv/bin/python -m unittest tests.test_retrospective_lambda_training.RetrospectiveLambdaTrainingTests.test_build_training_example_from_pipeline_result -v
```

Expected: FAIL because `build_training_example_from_pipeline_result` does not exist.

- [ ] **Step 3: Implement training-example conversion**

In `scripts/train_retrospective_lambda_model.py`, add imports:

```python
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT / "docs") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "docs"))

import mixture_prior  # noqa: E402
import oncology_trial_similarity_pipeline as pipeline  # noqa: E402
```

Add this function after `load_examples`:

```python
def build_training_example_from_pipeline_result(
    result: dict[str, Any],
    endpoint_key: str = "ORR",
    lambda0: float = 0.2,
) -> dict[str, Any]:
    query_observations = pipeline.query_endpoint_observations(result["query_summary"])
    query = query_observations.get(endpoint_key)
    if query is None:
        raise ValueError(f"Query does not contain endpoint {endpoint_key}")
    count = query.get("treatment_count")
    denominator = query.get("treatment_denominator")
    if count is None or denominator is None:
        raise ValueError(f"Query endpoint {endpoint_key} does not have treatment count/denominator")
    rows = result.get("reranked_top_matches") or result.get("reranked_top10") or []
    mixture = mixture_prior.components_from_reranked_rows(rows, endpoint_key=endpoint_key, lambda0=lambda0)
    features = []
    components = []
    lambda_rule = []
    for component in mixture["components"]:
        overall = float(component.get("overall_similarity_score", 0.0)) / 100.0
        rate = float(component.get("rate", 0.0))
        denominator_i = float(component.get("denominator", 0.0))
        discount = float(component.get("discount", 0.0))
        gate = float(component.get("gate", 0.0))
        features.append([overall, rate, discount, math.log1p(denominator_i), gate])
        components.append(
            {
                "alpha": float(component["alpha"]),
                "beta": float(component["beta"]),
                "gate": gate,
                "denominator": denominator_i,
                "discount": discount,
            }
        )
        lambda_rule.append(float(component.get("lambda_rule", 0.0)))
    return {
        "query": {"count": int(count), "denominator": int(denominator)},
        "lambda_0": float(mixture["lambda_0"]),
        "features": features,
        "components": components,
        "lambda_rule": lambda_rule,
    }
```

- [ ] **Step 4: Extend CLI to build examples from pipeline-result JSONL**

Add this optional argument:

```python
    parser.add_argument("--pipeline-results-jsonl", type=Path, default=None)
```

In `main`, before `examples = load_examples(args.examples_jsonl)`, replace with:

```python
    if args.pipeline_results_jsonl is not None:
        examples = []
        with args.pipeline_results_jsonl.open("r", encoding="utf-8") as handle:
            for line in handle:
                if line.strip():
                    examples.append(build_training_example_from_pipeline_result(json.loads(line)))
    else:
        examples = load_examples(args.examples_jsonl)
```

Change `--examples-jsonl` to `default=None` and add a validation block:

```python
    if args.examples_jsonl is None and args.pipeline_results_jsonl is None:
        raise SystemExit("Pass either --examples-jsonl or --pipeline-results-jsonl")
```

- [ ] **Step 5: Run training tests**

Run:

```bash
../.venv/bin/python -m unittest tests.test_retrospective_lambda_training -v
```

Expected: PASS.

- [ ] **Step 6: Commit**

```bash
git add scripts/train_retrospective_lambda_model.py tests/test_retrospective_lambda_training.py
git commit -m "Build retrospective lambda examples from pipeline outputs"
```

## Task 9: Update Documentation and Run Focused Verification

**Files:**
- Modify: `README.md`
- Modify: `web_agent/README.md`
- Modify: `docs/prior_borrowing_reranker_bayesian_explanation.md`

- [ ] **Step 1: Add README method note**

In `README.md`, add this section under the existing pipeline description:

```markdown
### Optional Trial2Vec Retrieval and Mixture Prior

The default Stage-1 backend remains the local Bio_ClinicalBERT index for reproducibility. A Trial2Vec backend can be used when a Trial2Vec embedding index has been built from `trial_summaries.jsonl`. Trial2Vec and future SECRET retrieval backends are treated as high-recall candidate discovery modules; they do not directly decide Bayesian borrowing weights.

For supported binary endpoints, the Bayesian layer now reports both the original weighted beta-binomial sensitivity summaries and an experimental mixture-prior summary. The mixture prior separates the Stage-2 borrowing discount `a_i`, which controls information discounting inside each historical component, from the mixture weight `lambda_i`, which controls how much prior mass each top candidate receives. Learned `lambda_i` values should be trained only through retrospective prediction or expert labels, and query outcomes must not be used to set prior weights at deployment time.
```

- [ ] **Step 2: Add web-agent README caveat**

In `web_agent/README.md`, add this paragraph under the existing caveats:

```markdown
When available, mixture-prior outputs are exploratory. Component weights are reported for review and sensitivity analysis; they are not expert-validated borrowing decisions. The weak-prior component is retained as a robustness safeguard, and retrospective lambda training must be evaluated with leakage controls before use in a manuscript claim.
```

- [ ] **Step 3: Add method explanation paragraph**

In `docs/prior_borrowing_reranker_bayesian_explanation.md`, add this subsection after the current Bayesian prior-borrowing analysis method:

```markdown
### Mixture-prior extension

The revised design separates two quantities that were previously easy to conflate. The Stage-2 `suggested_borrowing_discount` is an information discount `a_i` inside the candidate-specific beta component:

```text
Beta(1 + a_i y_i, 1 + a_i(n_i - y_i))
```

The mixture-prior component weight `lambda_i` controls how much prior mass is assigned to that candidate component:

```text
p_prior(p) = lambda_0 Beta(1, 1) + sum_i lambda_i Beta(1 + a_i y_i, 1 + a_i(n_i - y_i))
```

Without expert labels, `lambda_i` is trained by retrospective prediction: completed trials are treated as pseudo-queries, their outcomes are hidden during retrieval and weighting, and the model is optimized to maximize the beta-binomial predictive likelihood of the held-out outcome. This training signal evaluates whether the selected historical prior predicts future-like trial results, but it does not replace expert adjudication.
```

- [ ] **Step 4: Run focused verification**

Run:

```bash
../.venv/bin/python -m unittest tests.test_stage1_backends tests.test_mixture_prior tests.test_retrospective_lambda_training -v
../.venv/bin/python -m unittest tests.test_web_agent.WebAgentTests.test_bayesian_analysis_summarizes_weighted_orr_borrowing -v
../.venv/bin/python -m unittest tests.test_publication_evaluation.PublicationEvaluationTests.test_bayesian_sensitivity_includes_ess_cap -v
../.venv/bin/python -m py_compile docs/oncology_trial_similarity_pipeline.py docs/mixture_prior.py scripts/build_trial2vec_index.py scripts/train_retrospective_lambda_model.py
```

Expected: all commands pass.

- [ ] **Step 5: Commit**

```bash
git add README.md web_agent/README.md docs/prior_borrowing_reranker_bayesian_explanation.md
git commit -m "Document Trial2Vec retrieval and mixture prior extension"
```

## Task 10: End-to-End Smoke Run on Existing Artifacts

**Files:**
- No code changes expected.

- [ ] **Step 1: Build a small Trial2Vec index from the existing summaries**

Run this only if the pretrained Trial2Vec model directory exists:

```bash
../.venv/bin/python scripts/build_trial2vec_index.py \
  --summaries-path ../artifacts/oncology_trial_similarity_clinicalbert/trial_summaries.jsonl \
  --output-path ../artifacts/oncology_trial_similarity_clinicalbert/trial2vec_embeddings.npz \
  --model-dir ../artifacts/trial2vec/pretrained_model \
  --device cpu
```

Expected: command prints JSON with `summary_count`, `embedding_shape`, and `output_path`.

- [ ] **Step 2: Run a ClinicalBERT default search to confirm backward compatibility**

Run:

```bash
../.venv/bin/python docs/oncology_trial_similarity_pipeline.py search \
  --query-json /Users/wang/PHD/clinic.gov/Oncology_All_Trials/Oncology_All_Trials/NCT02413320/NCT02413320_data.json \
  --index-dir ../artifacts/oncology_trial_similarity_clinicalbert \
  --top-k 3 \
  --rerank \
  --rerank-top-n 20 \
  --output ../artifacts/smoke_clinicalbert_mixture_prior.json
```

Expected: output JSON contains `reranked_top10` and `bayesian_analysis.endpoint_analyses[0].mixture_prior`.

- [ ] **Step 3: Run a Trial2Vec search if the Trial2Vec index was built**

Run:

```bash
../.venv/bin/python docs/oncology_trial_similarity_pipeline.py search \
  --query-json /Users/wang/PHD/clinic.gov/Oncology_All_Trials/Oncology_All_Trials/NCT02413320/NCT02413320_data.json \
  --index-dir ../artifacts/oncology_trial_similarity_clinicalbert \
  --retrieval-backend trial2vec \
  --trial2vec-index-path ../artifacts/oncology_trial_similarity_clinicalbert/trial2vec_embeddings.npz \
  --trial2vec-model-dir ../artifacts/trial2vec/pretrained_model \
  --top-k 3 \
  --rerank \
  --rerank-top-n 20 \
  --output ../artifacts/smoke_trial2vec_mixture_prior.json
```

Expected: output JSON contains `embedding_backend`, `embedding_model`, `top_matches`, `reranked_top10`, and mixture-prior output. `top_matches[0].retrieval_backend` should be `trial2vec`.

- [ ] **Step 4: Commit smoke-run documentation only if a tracked smoke report is added**

Do not commit large JSON artifacts. If a small markdown smoke report is created, commit it with:

```bash
git add docs/trial2vec_mixture_prior_smoke_report.md
git commit -m "Add Trial2Vec mixture prior smoke report"
```

## Self-Review Checklist

- Spec coverage: Tasks 1-3 cover Stage-1 backend replacement and Trial2Vec support. Tasks 4-6 cover mixture prior output. Tasks 7-8 cover retrospective prediction training. Task 9 covers documentation. Task 10 covers smoke verification.
- Placeholder scan: no task contains `TBD`, `TODO`, or undefined file paths.
- Type consistency: `discount_i`/`a_i` remains inside beta components, while `lambda_i` remains mixture-prior mass. `lambda_i_post` is posterior-only and must not be used to form the prior.
- Scope control: SECRET is intentionally interface-only in this first revision because it requires a separate protocol summarization subsystem.
