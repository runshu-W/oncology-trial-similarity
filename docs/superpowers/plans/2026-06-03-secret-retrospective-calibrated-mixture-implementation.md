# SECRET-Style Retrospective-Calibrated Mixture Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Implement a runnable SECRET-style Stage-1 backend, full 9-feature retrospective lambda training, retrospective no-expert evaluation, and optional retrospective-calibrated mixture-prior search mode.

**Architecture:** Keep the main pipeline in `docs/oncology_trial_similarity_pipeline.py`, but put deterministic SECRET-style section construction and section-vector search into `docs/secret_retrieval.py`. Keep lambda training/evaluation in scripts so normal search remains lightweight. Preserve rule-based mixture prior as the default and make calibrated lambda weights opt-in through a saved model artifact.

**Tech Stack:** Python 3.12, NumPy, PyTorch, existing ClinicalBERT/hash embedder interface, `unittest`, existing JSON/NPZ artifacts.

---

## File Structure

- Create `docs/secret_retrieval.py`  
  Deterministic SECRET-style Q/A sections, bounded protocol/SAP excerpt handling, section-vector index save/load helpers, section-weighted cosine scoring.

- Modify `docs/oncology_trial_similarity_pipeline.py`  
  Add `build-secret-index`, runnable `--retrieval-backend secret`, eligibility score in Stage 2, calibrated mixture-prior CLI options, and model-weight application.

- Modify `docs/mixture_prior.py`  
  Add active lambda mode metadata and helper for replacing `lambda_rule` with model-produced `lambda_model` / `lambda_active`.

- Modify `scripts/train_retrospective_lambda_model.py`  
  Add full feature names, full `x_i` construction, model artifact save/load helpers, rule/learned NLL helpers.

- Create `scripts/evaluate_retrospective_lambda_model.py`  
  Deterministic train/eval split, no-expert retrospective metrics, JSON report.

- Modify `tests/test_stage1_backends.py`  
  Tests for SECRET-style sections, index scoring, and pipeline backend guardrail removal.

- Modify `tests/test_mixture_prior.py`  
  Tests for active lambda metadata and model-lambda replacement.

- Modify `tests/test_retrospective_lambda_training.py`  
  Tests for 9-feature `x_i`, feature order, redflag severity, model artifact save/load.

- Create `tests/test_retrospective_lambda_evaluation.py`  
  Tests for deterministic split, metrics, invalid inputs, and JSON output.

- Modify `docs/trial2vec_secret_mixture_prior_update_report_2026-06-03.md`  
  Update limitation section to reflect implemented SECRET-style MVP, 9-feature lambda training, retrospective evaluation, and calibrated mixture mode.

---

## Task 1: SECRET-Style Section Module

**Files:**
- Create: `docs/secret_retrieval.py`
- Modify: `tests/test_stage1_backends.py`

- [ ] **Step 1: Write failing tests for deterministic SECRET-style sections**

Append to `tests/test_stage1_backends.py`:

```python
class SecretRetrievalTests(unittest.TestCase):
    def test_secret_sections_include_required_qa_fields_and_bounded_excerpts(self) -> None:
        secret = importlib.import_module("docs.secret_retrieval")
        summary = {
            "nct_id": "NCTSECRET",
            "brief_title": "Neoadjuvant HER2 breast cancer trial",
            "brief_summary": "Tests a targeted regimen.",
            "cancer_type": {"primary_site": ["Breast"], "histology": ["HER2-positive"]},
            "intervention": {
                "experimental_regimen": "Trastuzumab plus pertuzumab and docetaxel",
                "drug_classes": ["HER2 therapy", "taxane"],
            },
            "population": {
                "key_inclusion": ["Stage II-III HER2-positive breast cancer"],
                "key_exclusion": ["Active brain metastases"],
            },
            "endpoints": {
                "primary": [
                    {
                        "title": "Pathologic complete response",
                        "endpoint_family": "ORR/CR/PR",
                        "time_frame": "20 weeks",
                    }
                ]
            },
            "design": {"single_or_multi_arm": "Multi Arm", "randomized": "Randomized"},
            "result_usability": {"has_posted_results": True},
            "supporting_documents": {
                "protocol_excerpt": "Eligibility details " * 200,
                "sap_excerpt": "Analysis population details " * 200,
            },
        }

        sections = secret.secret_sections_from_summary(summary, excerpt_char_limit=120)

        self.assertEqual(set(sections), set(secret.SECRET_SECTIONS))
        self.assertIn("Q:", sections["disease_population"])
        self.assertIn("Breast", sections["disease_population"])
        self.assertIn("HER2-positive breast cancer", sections["eligibility"])
        self.assertIn("Active brain metastases", sections["eligibility"])
        self.assertLessEqual(len(sections["eligibility"]), 700)
        self.assertIn("protocol excerpt", sections["eligibility"].lower())
```

- [ ] **Step 2: Run test to verify it fails**

Run:

```bash
"/Users/wang/Documents/New project/.venv/bin/python" -m unittest tests.test_stage1_backends.SecretRetrievalTests.test_secret_sections_include_required_qa_fields_and_bounded_excerpts -v
```

Expected: FAIL with `ModuleNotFoundError` for `docs.secret_retrieval`.

- [ ] **Step 3: Implement `docs/secret_retrieval.py`**

Create the module with these public functions and constants:

```python
from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

SECRET_SECTIONS = (
    "disease_population",
    "intervention",
    "eligibility",
    "endpoint",
    "design",
    "results",
    "safety_followup",
)

SECRET_SECTION_WEIGHTS = {
    "disease_population": 0.22,
    "intervention": 0.20,
    "eligibility": 0.16,
    "endpoint": 0.18,
    "design": 0.10,
    "results": 0.10,
    "safety_followup": 0.04,
}


def _text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, list):
        return "; ".join(_text(item) for item in value if _text(item))
    if isinstance(value, dict):
        return "; ".join(f"{key}: {_text(val)}" for key, val in value.items() if _text(val))
    return " ".join(str(value).split())


def _bounded(value: Any, limit: int) -> str:
    text = _text(value)
    if limit <= 0 or len(text) <= limit:
        return text
    return text[:limit].rstrip() + "..."


def _qa(question: str, answer_parts: list[str]) -> str:
    answer = " ".join(part for part in answer_parts if part).strip()
    if not answer:
        answer = "Not reported."
    return f"Q: {question}\nA: {answer}"


def secret_sections_from_summary(summary: dict[str, Any], excerpt_char_limit: int = 800) -> dict[str, str]:
    cancer = summary.get("cancer_type") if isinstance(summary.get("cancer_type"), dict) else {}
    intervention = summary.get("intervention") if isinstance(summary.get("intervention"), dict) else {}
    population = summary.get("population") if isinstance(summary.get("population"), dict) else {}
    endpoints = summary.get("endpoints") if isinstance(summary.get("endpoints"), dict) else {}
    design = summary.get("design") if isinstance(summary.get("design"), dict) else {}
    result = summary.get("result_usability") if isinstance(summary.get("result_usability"), dict) else {}
    docs = summary.get("supporting_documents") if isinstance(summary.get("supporting_documents"), dict) else {}
    protocol = _bounded(docs.get("protocol_excerpt"), excerpt_char_limit)
    sap = _bounded(docs.get("sap_excerpt"), excerpt_char_limit)

    primary = endpoints.get("primary") if isinstance(endpoints.get("primary"), list) else []
    endpoint_text = _text([
        {
            "title": item.get("title"),
            "family": item.get("endpoint_family"),
            "time_frame": item.get("time_frame"),
        }
        for item in primary
        if isinstance(item, dict)
    ])

    return {
        "disease_population": _qa(
            "What disease and population does the trial study?",
            [_text(cancer), _text(population.get("line_of_therapy")), _text(summary.get("brief_summary"))],
        ),
        "intervention": _qa(
            "What intervention or regimen is tested?",
            [_text(intervention), _text(summary.get("brief_title"))],
        ),
        "eligibility": _qa(
            "What eligibility criteria define the target population?",
            [
                "Inclusion: " + _text(population.get("key_inclusion")),
                "Exclusion: " + _text(population.get("key_exclusion")),
                ("Protocol excerpt: " + protocol) if protocol else "",
            ],
        ),
        "endpoint": _qa(
            "What primary endpoint and estimand are measured?",
            [endpoint_text],
        ),
        "design": _qa(
            "What design, phase, arm structure, and randomization are used?",
            [_text(summary.get("phase")), _text(design)],
        ),
        "results": _qa(
            "What result quantities are available for borrowing?",
            [_text(result), _text(summary.get("borrowable_quantities"))],
        ),
        "safety_followup": _qa(
            "What safety and follow-up information is available?",
            [_text(summary.get("results")), ("SAP excerpt: " + sap) if sap else ""],
        ),
    }


def cosine(a: np.ndarray, b: np.ndarray) -> float:
    denom = float(np.linalg.norm(a) * np.linalg.norm(b))
    if denom == 0.0:
        return 0.0
    return float(np.dot(a, b) / denom)


def score_secret_index(
    query_vectors: dict[str, np.ndarray],
    index_path: Path,
    summaries: dict[str, dict[str, Any]],
    excluded_nct_id: str,
    top_k: int,
) -> list[dict[str, Any]]:
    index = np.load(index_path, allow_pickle=False)
    nct_ids = index["nct_ids"]
    scored = []
    for idx, raw_nct_id in enumerate(nct_ids):
        nct_id = str(raw_nct_id)
        if nct_id == excluded_nct_id:
            continue
        section_scores = {}
        score = 0.0
        for section, weight in SECRET_SECTION_WEIGHTS.items():
            sim = cosine(np.asarray(query_vectors[section], dtype=np.float32), index[section][idx])
            section_scores[section] = sim
            score += weight * sim
        candidate_summary = summaries.get(nct_id, {})
        scored.append(
            {
                "nct_id": nct_id,
                "score": score,
                "score_0_100": round(100 * max(0.0, score), 2),
                "retrieval_backend": "secret",
                "secret_section_scores": {key: round(value, 4) for key, value in section_scores.items()},
                "secret_evidence": secret_sections_from_summary(candidate_summary),
            }
        )
    scored.sort(key=lambda row: row["score"], reverse=True)
    for rank, row in enumerate(scored[:top_k], start=1):
        row["retrieval_rank"] = rank
    return scored[:top_k]
```

- [ ] **Step 4: Run test to verify it passes**

Run:

```bash
"/Users/wang/Documents/New project/.venv/bin/python" -m unittest tests.test_stage1_backends.SecretRetrievalTests.test_secret_sections_include_required_qa_fields_and_bounded_excerpts -v
```

Expected: PASS.

- [ ] **Step 5: Write failing tests for SECRET index scoring**

Append:

```python
    def test_secret_index_search_uses_section_weights_and_returns_evidence(self) -> None:
        secret = importlib.import_module("docs.secret_retrieval")
        query_vectors = {
            section: np.array([1.0, 0.0], dtype=np.float32)
            for section in secret.SECRET_SECTIONS
        }
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "secret_embeddings.npz"
            arrays = {
                section: np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32)
                for section in secret.SECRET_SECTIONS
            }
            np.savez_compressed(
                path,
                nct_ids=np.array(["NCTGOOD", "NCTBAD"]),
                retrieval_backend=np.array(["secret"]),
                embedding_backend=np.array(["hashing"]),
                embedding_model=np.array(["signed-token-hashing-2048"]),
                **arrays,
            )
            summaries = {
                "NCTGOOD": {
                    "nct_id": "NCTGOOD",
                    "cancer_type": {"primary_site": ["Breast"]},
                    "intervention": {"experimental_regimen": "Drug A"},
                }
            }

            rows = secret.score_secret_index(query_vectors, path, summaries, excluded_nct_id="NCTQUERY", top_k=2)

        self.assertEqual([row["nct_id"] for row in rows], ["NCTGOOD", "NCTBAD"])
        self.assertEqual(rows[0]["retrieval_backend"], "secret")
        self.assertEqual(rows[0]["score_0_100"], 100.0)
        self.assertIn("disease_population", rows[0]["secret_section_scores"])
        self.assertIn("disease_population", rows[0]["secret_evidence"])
```

- [ ] **Step 6: Run failing scoring test**

Run:

```bash
"/Users/wang/Documents/New project/.venv/bin/python" -m unittest tests.test_stage1_backends.SecretRetrievalTests.test_secret_index_search_uses_section_weights_and_returns_evidence -v
```

Expected: PASS if Step 3 already included `score_secret_index`; if it fails, fix only the missing scorer behavior.

- [ ] **Step 7: Run stage1 tests and commit**

Run:

```bash
"/Users/wang/Documents/New project/.venv/bin/python" -m unittest tests.test_stage1_backends -v
```

Expected: all tests pass.

Commit:

```bash
git add docs/secret_retrieval.py tests/test_stage1_backends.py
git commit -m "Add SECRET-style section retrieval utilities"
```

---

## Task 2: Pipeline CLI Support for SECRET-Style Backend

**Files:**
- Modify: `docs/oncology_trial_similarity_pipeline.py`
- Modify: `tests/test_stage1_backends.py`

- [ ] **Step 1: Write failing test that `secret` is supported**

Replace the old unsupported test in `tests/test_stage1_backends.py`:

```python
    def test_secret_backend_is_supported_by_backend_guardrail(self) -> None:
        pipeline.ensure_supported_retrieval_backend("secret")
```

Run:

```bash
"/Users/wang/Documents/New project/.venv/bin/python" -m unittest tests.test_stage1_backends.Stage1BackendTests.test_secret_backend_is_supported_by_backend_guardrail -v
```

Expected: FAIL because `ensure_supported_retrieval_backend("secret")` still raises `NotImplementedError`.

- [ ] **Step 2: Make guardrail allow `secret`**

In `docs/oncology_trial_similarity_pipeline.py`, change:

```python
    if backend == "secret":
        raise NotImplementedError(...)
```

to no-op support:

```python
    return None
```

Run the same test. Expected: PASS.

- [ ] **Step 3: Write failing test for building a SECRET index through pipeline helper**

Append to `SecretRetrievalTests`:

```python
    def test_build_secret_index_from_summaries_writes_section_arrays(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            summaries_path = tmp_path / "trial_summaries.jsonl"
            output_path = tmp_path / "secret_embeddings.npz"
            summaries_path.write_text(
                json.dumps(
                    {
                        "nct_id": "NCT1",
                        "cancer_type": {"primary_site": ["Breast"]},
                        "intervention": {"experimental_regimen": "Drug A"},
                    }
                )
                + "\n",
                encoding="utf-8",
            )

            pipeline.build_secret_index(
                summaries_path=summaries_path,
                output_path=output_path,
                embedding_backend="hashing",
            )

            data = np.load(output_path, allow_pickle=False)

        self.assertEqual(list(data["nct_ids"]), ["NCT1"])
        self.assertEqual(str(data["retrieval_backend"][0]), "secret")
        self.assertIn("eligibility", data.files)
        self.assertEqual(data["eligibility"].shape[0], 1)
```

Run:

```bash
"/Users/wang/Documents/New project/.venv/bin/python" -m unittest tests.test_stage1_backends.SecretRetrievalTests.test_build_secret_index_from_summaries_writes_section_arrays -v
```

Expected: FAIL because `build_secret_index` is missing.

- [ ] **Step 4: Implement `build_secret_index`**

Add import near the top:

```python
from . import secret_retrieval
```

Use the same fallback import style as `mixture_prior` if direct package import is not available when the script runs as a file.

Add:

```python
def build_secret_index(
    summaries_path: Path,
    output_path: Path,
    embedding_backend: str = DEFAULT_INDEX_EMBEDDING_BACKEND,
    embedding_model: str = DEFAULT_CLINICALBERT_MODEL,
    embedding_batch_size: int = 16,
    embedding_max_length: int = 256,
) -> None:
    summaries = load_summaries(summaries_path)
    embedder = make_embedder(
        embedding_backend,
        model_name=embedding_model,
        batch_size=embedding_batch_size,
        max_length=embedding_max_length,
    )
    nct_ids = list(summaries)
    section_texts = {section: [] for section in secret_retrieval.SECRET_SECTIONS}
    for nct_id in nct_ids:
        sections = secret_retrieval.secret_sections_from_summary(summaries[nct_id])
        for section in secret_retrieval.SECRET_SECTIONS:
            section_texts[section].append(sections[section])
    arrays = {
        section: embedder.encode(section_texts[section]).astype(np.float32)
        for section in secret_retrieval.SECRET_SECTIONS
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output_path,
        nct_ids=np.array(nct_ids),
        retrieval_backend=np.array(["secret"]),
        embedding_backend=np.array([embedding_backend]),
        embedding_model=np.array([embedding_model if embedding_backend != "hashing" else "signed-token-hashing-2048"]),
        **arrays,
    )
```

- [ ] **Step 5: Write failing test for `search(... retrieval_backend="secret")`**

Add:

```python
    def test_pipeline_search_uses_secret_backend_index(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            index_dir = tmp_path / "index"
            index_dir.mkdir()
            query_path = tmp_path / "query.json"
            query_path.write_text(
                json.dumps({"Study details": {"1. NCT number": "NCTQUERY"}}),
                encoding="utf-8",
            )
            (index_dir / "trial_summaries.jsonl").write_text(
                json.dumps({"nct_id": "NCT1", "brief_title": "Candidate", "cancer_type": {"primary_site": ["Breast"]}})
                + "\n",
                encoding="utf-8",
            )
            secret_path = index_dir / "secret_embeddings.npz"
            secret = importlib.import_module("docs.secret_retrieval")
            arrays = {
                section: np.zeros((1, 2048), dtype=np.float32)
                for section in secret.SECRET_SECTIONS
            }
            np.savez_compressed(
                secret_path,
                nct_ids=np.array(["NCT1"]),
                retrieval_backend=np.array(["secret"]),
                embedding_backend=np.array(["hashing"]),
                embedding_model=np.array(["signed-token-hashing-2048"]),
                **arrays,
            )

            result = pipeline.search(
                query_path,
                index_dir,
                top_k=1,
                retrieval_backend="secret",
                secret_index_path=secret_path,
            )

        self.assertEqual(result["retrieval_backend"], "secret")
        self.assertEqual(result["embedding_backend"], "hashing")
        self.assertEqual(result["top_matches"][0]["retrieval_backend"], "secret")
        self.assertIn("secret_section_scores", result["top_matches"][0])
```

Run:

```bash
"/Users/wang/Documents/New project/.venv/bin/python" -m unittest tests.test_stage1_backends.SecretRetrievalTests.test_pipeline_search_uses_secret_backend_index -v
```

Expected: FAIL because `search` has no `secret_index_path` argument.

- [ ] **Step 6: Implement `secret` branch in `search` and CLI**

Add `secret_index_path: Path | None = None` to `search`.

Validation:

```python
    if retrieval_backend == "secret":
        if secret_index_path is None:
            raise ValueError("--secret-index-path is required when --retrieval-backend=secret.")
        if not summaries_path.exists():
            raise FileNotFoundError(...)
        if not secret_index_path.exists():
            raise FileNotFoundError(f"SECRET-style index file does not exist: {secret_index_path}")
```

Branch:

```python
    elif retrieval_backend == "secret":
        assert secret_index_path is not None
        secret_file = np.load(secret_index_path, allow_pickle=False)
        stored_backend = str(secret_file["embedding_backend"][0]) if "embedding_backend" in secret_file else "hashing"
        stored_model = str(secret_file["embedding_model"][0]) if "embedding_model" in secret_file else "signed-token-hashing-2048"
        query_embedder = make_embedder(
            stored_backend,
            model_name=(stored_model if stored_backend != "hashing" else DEFAULT_CLINICALBERT_MODEL),
            batch_size=embedding_batch_size,
            max_length=embedding_max_length,
        )
        query_sections = secret_retrieval.secret_sections_from_summary(query_summary)
        query_vectors = {
            section: query_embedder.encode([query_sections[section]])[0]
            for section in secret_retrieval.SECRET_SECTIONS
        }
        scored = secret_retrieval.score_secret_index(
            query_vectors,
            secret_index_path,
            summaries,
            excluded_nct_id=query_summary.get("nct_id", ""),
            top_k=max(rerank_top_n, top_k),
        )
        scored = [enrich_candidate_row(row, summaries.get(row["nct_id"], {})) for row in scored]
        result_embedding_backend = stored_backend
        result_embedding_model = stored_model
```

Add CLI subcommand:

```python
secret_build = sub.add_parser("build-secret-index")
secret_build.add_argument("--index-dir", type=Path, default=Path("artifacts/oncology_trial_similarity"))
secret_build.add_argument("--summaries-path", type=Path, default=None)
secret_build.add_argument("--output-path", type=Path, default=None)
secret_build.add_argument("--embedding-backend", choices=["hashing", "clinicalbert"], default=DEFAULT_INDEX_EMBEDDING_BACKEND)
secret_build.add_argument("--embedding-model", default=DEFAULT_CLINICALBERT_MODEL)
secret_build.add_argument("--embedding-batch-size", type=int, default=16)
secret_build.add_argument("--embedding-max-length", type=int, default=256)
```

Add search arg:

```python
query.add_argument("--secret-index-path", type=Path, default=None)
```

Dispatch:

```python
    if args.command == "build-secret-index":
        summaries_path = args.summaries_path or args.index_dir / "trial_summaries.jsonl"
        output_path = args.output_path or args.index_dir / "secret_embeddings.npz"
        build_secret_index(...)
```

- [ ] **Step 7: Run stage1 tests and commit**

Run:

```bash
"/Users/wang/Documents/New project/.venv/bin/python" -m unittest tests.test_stage1_backends -v
```

Expected: all tests pass.

Commit:

```bash
git add docs/oncology_trial_similarity_pipeline.py tests/test_stage1_backends.py
git commit -m "Wire SECRET-style retrieval into pipeline"
```

---

## Task 3: Full 9-Feature Lambda Vector

**Files:**
- Modify: `docs/oncology_trial_similarity_pipeline.py`
- Modify: `scripts/train_retrospective_lambda_model.py`
- Modify: `tests/test_retrospective_lambda_training.py`

- [ ] **Step 1: Write failing tests for feature names and redflag severity**

Add near top of `tests/test_retrospective_lambda_training.py`:

```python
EXPECTED_FEATURE_NAMES = [
    "s_i",
    "disease_match_i",
    "regimen_match_i",
    "endpoint_match_i",
    "followup_match_i",
    "eligibility_match_i",
    "result_quality_i",
    "negative_redflag_severity_i",
    "log_n_i",
]
```

Add tests:

```python
    def test_full_feature_names_match_design_vector(self):
        module = load_training_module()
        self.assertEqual(module.LAMBDA_FEATURE_NAMES, EXPECTED_FEATURE_NAMES)

    def test_redflag_severity_is_normalized(self):
        module = load_training_module()
        flags = [
            "Low disease/population match.",
            "No normalized regimen-backbone overlap.",
            "Candidate has no posted results in indexed JSON.",
        ]
        self.assertAlmostEqual(module.redflag_severity(flags), min((1.0 + 0.8 + 0.8) / 3.0, 1.0))
```

Run:

```bash
"/Users/wang/Documents/New project/.venv/bin/python" -m unittest tests.test_retrospective_lambda_training.RetrospectiveLambdaTrainingTests.test_full_feature_names_match_design_vector tests.test_retrospective_lambda_training.RetrospectiveLambdaTrainingTests.test_redflag_severity_is_normalized -v
```

Expected: FAIL because names/helper do not exist.

- [ ] **Step 2: Implement feature names and severity helper**

In `scripts/train_retrospective_lambda_model.py` add:

```python
LAMBDA_FEATURE_NAMES = [
    "s_i",
    "disease_match_i",
    "regimen_match_i",
    "endpoint_match_i",
    "followup_match_i",
    "eligibility_match_i",
    "result_quality_i",
    "negative_redflag_severity_i",
    "log_n_i",
]


def redflag_severity(red_flags: list[Any]) -> float:
    raw = 0.0
    for flag in red_flags:
        text = str(flag)
        low = text.lower()
        if "low disease" in low or "low endpoint" in low:
            raw += 1.0
        elif "no primary endpoint-family overlap" in low:
            raw += 1.0
        elif "no normalized regimen-backbone overlap" in low:
            raw += 0.8
        elif "no posted results" in low or "no arm-level count/denominator" in low:
            raw += 0.8
        else:
            raw += 0.25
    return min(raw / 3.0, 1.0)
```

Run tests from Step 1. Expected: PASS.

- [ ] **Step 3: Write failing test for 9-feature example construction**

Update `test_build_training_example_from_pipeline_result` assertions:

```python
        self.assertEqual(example["feature_names"], EXPECTED_FEATURE_NAMES)
        self.assertEqual(len(example["features"][0]), 9)
        self.assertAlmostEqual(example["features"][0][0], 0.82)
        self.assertAlmostEqual(example["features"][0][1], 1.0)
        self.assertAlmostEqual(example["features"][0][2], 0.8)
        self.assertAlmostEqual(example["features"][0][3], 1.0)
        self.assertAlmostEqual(example["features"][0][4], 0.4)
        self.assertAlmostEqual(example["features"][0][5], 0.2)
        self.assertAlmostEqual(example["features"][0][6], 1.0)
        self.assertLessEqual(example["features"][0][7], 0.0)
        self.assertAlmostEqual(example["features"][0][8], math.log1p(50.0))
```

Add redflag case:

```python
    def test_build_training_example_uses_negative_redflag_feature(self) -> None:
        module = load_training_module()
        result = self.pipeline_result()
        result["reranked_top_matches"][0]["red_flags"] = ["Low endpoint/estimand match."]

        example = module.build_training_example_from_pipeline_result(result, endpoint_key="ORR")

        self.assertLess(example["features"][0][7], 0.0)
```

Run:

```bash
"/Users/wang/Documents/New project/.venv/bin/python" -m unittest tests.test_retrospective_lambda_training.RetrospectiveLambdaTrainingTests.test_build_training_example_from_pipeline_result tests.test_retrospective_lambda_training.RetrospectiveLambdaTrainingTests.test_build_training_example_uses_negative_redflag_feature -v
```

Expected: FAIL because feature vector remains 5 columns.

- [ ] **Step 4: Implement full feature construction**

In `build_training_example_from_pipeline_result`, keep access to the source reranked rows by `nct_id`. Use a helper:

```python
def _normalized_dimension(dimension_scores: dict[str, Any], *keys: str) -> float:
    for key in keys:
        if key in dimension_scores:
            value = float(dimension_scores.get(key) or 0.0)
            return max(0.0, min(1.0, value / 5.0))
    return 0.0
```

For each component, find its row:

```python
row_by_nct = {
    row.get("candidate_nct_id") or row.get("nct_id"): row
    for row in rows
    if isinstance(row, dict)
}
source_row = row_by_nct.get(component.get("nct_id")) or {}
dimension_scores = source_row.get("dimension_scores") if isinstance(source_row.get("dimension_scores"), dict) else {}
flags = source_row.get("red_flags") if isinstance(source_row.get("red_flags"), list) else []
features.append(
    [
        float(component.get("overall_similarity_score", 0.0)) / 100.0,
        _normalized_dimension(dimension_scores, "disease_population_match", "disease_biology_match"),
        _normalized_dimension(dimension_scores, "treatment_regimen_match"),
        _normalized_dimension(dimension_scores, "endpoint_estimand_match"),
        _normalized_dimension(dimension_scores, "safety_and_followup_relevance", "outcome_assessment_followup"),
        _normalized_dimension(dimension_scores, "eligibility_criteria_overlap"),
        _normalized_dimension(dimension_scores, "result_usability"),
        -redflag_severity(flags),
        math.log1p(denominator_i),
    ]
)
```

Return:

```python
"feature_names": LAMBDA_FEATURE_NAMES,
```

Update `zero_model` and existing tests that hardcode `input_dim=4` if needed by deriving `len(example["features"][0])`.

- [ ] **Step 5: Add Stage-2 eligibility score**

Write a failing test in `tests/test_web_agent.py` or `tests/test_stage1_backends.py` against `pipeline.score_prior_borrowing_pair`:

```python
def test_stage2_adds_eligibility_criteria_overlap_dimension(self):
    query = {
        "population": {"key_inclusion": ["HER2 positive breast cancer"], "key_exclusion": ["brain metastases"]},
        ...
    }
    candidate = {
        "population": {"key_inclusion": ["HER2 positive breast cancer"], "key_exclusion": ["active brain metastases"]},
        ...
    }
    scored = pipeline.score_prior_borrowing_pair(query, candidate)
    self.assertIn("eligibility_criteria_overlap", scored["dimension_scores"])
    self.assertGreater(scored["dimension_scores"]["eligibility_criteria_overlap"], 0.0)
```

Implement helper in `docs/oncology_trial_similarity_pipeline.py`:

```python
ELIGIBILITY_STOPWORDS = {...}

def eligibility_tokens(population: dict[str, Any]) -> set[str]:
    text = clean_text([population.get("key_inclusion", []), population.get("key_exclusion", [])]).lower()
    return {token for token in re.findall(r"[a-z0-9]+", text) if len(token) > 2 and token not in ELIGIBILITY_STOPWORDS}

def score_eligibility_overlap(query_population: dict[str, Any], candidate_population: dict[str, Any]) -> float:
    q_tokens = eligibility_tokens(query_population)
    c_tokens = eligibility_tokens(candidate_population)
    if not q_tokens or not c_tokens:
        return 0.0
    return round(5.0 * len(q_tokens & c_tokens) / len(q_tokens | c_tokens), 2)
```

Add to `dimension_scores`:

```python
"eligibility_criteria_overlap": eligibility_score,
```

Run:

```bash
"/Users/wang/Documents/New project/.venv/bin/python" -m unittest tests.test_retrospective_lambda_training tests.test_web_agent -v
```

Expected: PASS.

- [ ] **Step 6: Commit**

Run:

```bash
"/Users/wang/Documents/New project/.venv/bin/python" -m unittest tests.test_retrospective_lambda_training tests.test_web_agent -v
```

Commit:

```bash
git add docs/oncology_trial_similarity_pipeline.py scripts/train_retrospective_lambda_model.py tests/test_retrospective_lambda_training.py tests/test_web_agent.py
git commit -m "Expand retrospective lambda feature vector"
```

---

## Task 4: Lambda Model Artifact and Calibrated Mixture Mode

**Files:**
- Modify: `docs/mixture_prior.py`
- Modify: `docs/oncology_trial_similarity_pipeline.py`
- Modify: `scripts/train_retrospective_lambda_model.py`
- Modify: `tests/test_mixture_prior.py`
- Modify: `tests/test_retrospective_lambda_training.py`
- Modify: `tests/test_web_agent.py`

- [ ] **Step 1: Write failing model artifact test**

In `tests/test_retrospective_lambda_training.py` add:

```python
    def test_train_model_can_save_and_load_artifact(self):
        module = load_training_module()
        example = self.base_example()
        example["features"] = [[0.1] * 9, [0.2] * 9]
        example["feature_names"] = module.LAMBDA_FEATURE_NAMES
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "lambda_model.pt"
            summary = module.train_model(
                [example],
                epochs=1,
                learning_rate=0.01,
                hidden_dim=6,
                model_output=path,
            )
            loaded = module.load_model_artifact(path)

        self.assertTrue(path.exists())
        self.assertEqual(summary["model_output"], str(path))
        self.assertEqual(loaded["feature_names"], module.LAMBDA_FEATURE_NAMES)
        self.assertEqual(loaded["input_dim"], 9)
```

Run:

```bash
"/Users/wang/Documents/New project/.venv/bin/python" -m unittest tests.test_retrospective_lambda_training.RetrospectiveLambdaTrainingTests.test_train_model_can_save_and_load_artifact -v
```

Expected: FAIL because `model_output` and `load_model_artifact` are missing.

- [ ] **Step 2: Implement artifact save/load**

In training script:

```python
def save_model_artifact(path: str | Path, model: LambdaScorer, input_dim: int, hidden_dim: int, lambda0: float) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "state_dict": model.state_dict(),
            "input_dim": input_dim,
            "hidden_dim": hidden_dim,
            "feature_names": LAMBDA_FEATURE_NAMES,
            "lambda0": float(lambda0),
        },
        output,
    )


def load_model_artifact(path: str | Path) -> dict[str, Any]:
    artifact = torch.load(Path(path), map_location="cpu", weights_only=False)
    if artifact.get("feature_names") != LAMBDA_FEATURE_NAMES:
        raise ValueError("lambda model feature_names do not match current feature order")
    if int(artifact.get("input_dim", -1)) != len(LAMBDA_FEATURE_NAMES):
        raise ValueError("lambda model input_dim does not match current feature order")
    return artifact
```

Change `train_model(..., model_output: str | Path | None = None)` and save after training. Add `--model-output` CLI.

- [ ] **Step 3: Write failing mixture lambda replacement test**

In `tests/test_mixture_prior.py` add:

```python
    def test_apply_model_lambdas_sets_active_weights_and_preserves_rule_weights(self):
        mixture = {
            "lambda_0": 0.2,
            "components": [
                {"lambda_rule": 0.6},
                {"lambda_rule": 0.2},
            ],
        }

        updated = mixture_prior.apply_model_lambdas(mixture, [0.5, 0.3])

        self.assertEqual(updated["mode"], "retrospective_calibrated")
        self.assertAlmostEqual(sum(c["lambda_active"] for c in updated["components"]) + updated["lambda_0"], 1.0)
        self.assertEqual(updated["components"][0]["lambda_rule"], 0.6)
        self.assertIn("lambda_model", updated["components"][0])
        self.assertIn("calibration_note", updated)
```

Run:

```bash
"/Users/wang/Documents/New project/.venv/bin/python" -m unittest tests.test_mixture_prior.MixturePriorTest.test_apply_model_lambdas_sets_active_weights_and_preserves_rule_weights -v
```

Expected: FAIL because helper is missing.

- [ ] **Step 4: Implement `apply_model_lambdas`**

In `docs/mixture_prior.py`:

```python
def apply_model_lambdas(mixture: dict[str, Any], model_lambdas: list[float]) -> dict[str, Any]:
    components = list(mixture.get("components") or [])
    if len(model_lambdas) != len(components):
        raise ValueError("model lambda count must equal component count")
    lambda0 = _validate_lambda0(float(mixture.get("lambda_0", 0.2)))
    normalized = normalize_lambdas(model_lambdas, lambda0=lambda0)
    output = {**mixture, "mode": "retrospective_calibrated"}
    output["components"] = []
    for component, lambda_model in zip(components, normalized["lambda_i"]):
        output["components"].append({**component, "lambda_model": lambda_model, "lambda_active": lambda_model})
    output["calibration_note"] = "No expert labels were used; lambda_model was trained by retrospective predictive loss."
    return output
```

Also set rule mode in `components_from_reranked_rows`:

```python
return {"mode": "rule", "lambda_0": normalized["lambda_0"], "components": components}
```

- [ ] **Step 5: Write failing pipeline calibrated mode test**

In `tests/test_web_agent.py` or a new small test in `tests/test_retrospective_lambda_training.py`, monkeypatch the pipeline lambda model loader:

```python
def test_bayesian_analysis_applies_calibrated_lambda_model(monkeypatch equivalent using mock.patch):
    result = {... existing pipeline-shaped result with reranked_top_matches ...}
    with mock.patch.object(pipeline, "predict_lambda_model_weights", return_value=[10.0]):
        updated = pipeline.add_bayesian_analysis(
            result,
            mixture_prior_mode="retrospective_calibrated",
            lambda_model_path=Path("fake.pt"),
        )
    mixture = updated["bayesian_analysis"]["endpoint_analyses"][0]["mixture_prior"]
    self.assertEqual(mixture["mode"], "retrospective_calibrated")
    self.assertIn("lambda_model", mixture["components"][0])
```

Run the new test. Expected: FAIL because `add_bayesian_analysis` has no calibrated args.

- [ ] **Step 6: Implement calibrated mode in pipeline**

Add constants:

```python
MIXTURE_PRIOR_MODES = ("rule", "retrospective_calibrated")
```

Add helper:

```python
def predict_lambda_model_weights(mixture: dict[str, Any], lambda_model_path: Path) -> list[float]:
    from scripts import train_retrospective_lambda_model as lambda_training
    artifact = lambda_training.load_model_artifact(lambda_model_path)
    model = lambda_training.LambdaScorer(input_dim=artifact["input_dim"], hidden_dim=artifact["hidden_dim"])
    model.load_state_dict(artifact["state_dict"])
    model.eval()
    features = lambda_training.features_from_mixture_components(mixture["components"])
    with torch.no_grad():
        return model(torch.tensor(features, dtype=torch.float32)).tolist()
```

If importing `scripts` directly is awkward, add a local dynamic import by path, following the existing `mixture_prior` load pattern.

Change `add_bayesian_analysis(result, mixture_prior_mode="rule", lambda_model_path=None)`.

Inside endpoint loop:

```python
mixture = mixture_prior.components_from_reranked_rows(...)
if mixture_prior_mode == "retrospective_calibrated":
    if lambda_model_path is None:
        raise ValueError("--lambda-model-path is required when --mixture-prior-mode=retrospective_calibrated")
    model_weights = predict_lambda_model_weights(mixture, lambda_model_path)
    mixture = mixture_prior.apply_model_lambdas(mixture, model_weights)
analysis["mixture_prior"] = mixture
```

Thread args through `search` and CLI:

```python
query.add_argument("--mixture-prior-mode", choices=list(MIXTURE_PRIOR_MODES), default="rule")
query.add_argument("--lambda-model-path", type=Path, default=None)
```

- [ ] **Step 7: Run tests and commit**

Run:

```bash
"/Users/wang/Documents/New project/.venv/bin/python" -m unittest tests.test_mixture_prior tests.test_retrospective_lambda_training tests.test_web_agent -v
```

Expected: PASS.

Commit:

```bash
git add docs/mixture_prior.py docs/oncology_trial_similarity_pipeline.py scripts/train_retrospective_lambda_model.py tests/test_mixture_prior.py tests/test_retrospective_lambda_training.py tests/test_web_agent.py
git commit -m "Add retrospective calibrated mixture mode"
```

---

## Task 5: Retrospective Evaluation Script

**Files:**
- Create: `scripts/evaluate_retrospective_lambda_model.py`
- Create: `tests/test_retrospective_lambda_evaluation.py`

- [ ] **Step 1: Write failing evaluation tests**

Create `tests/test_retrospective_lambda_evaluation.py`:

```python
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest


SCRIPT_PATH = Path(__file__).resolve().parents[1] / "scripts" / "evaluate_retrospective_lambda_model.py"


def load_eval_module():
    spec = importlib.util.spec_from_file_location("evaluate_retrospective_lambda_model", SCRIPT_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class RetrospectiveEvaluationTests(unittest.TestCase):
    def example(self, count):
        return {
            "query": {"count": count, "denominator": 4},
            "lambda_0": 0.2,
            "feature_names": [
                "s_i",
                "disease_match_i",
                "regimen_match_i",
                "endpoint_match_i",
                "followup_match_i",
                "eligibility_match_i",
                "result_quality_i",
                "negative_redflag_severity_i",
                "log_n_i",
            ],
            "features": [[0.8, 1.0, 0.8, 1.0, 0.2, 0.5, 1.0, 0.0, 3.9]],
            "components": [{"alpha": 4.0, "beta": 2.0, "gate": 1.0, "discount": 0.75, "denominator": 20.0}],
            "lambda_rule": [0.8],
        }

    def test_split_indices_are_deterministic_and_disjoint(self):
        module = load_eval_module()
        train_a, eval_a = module.deterministic_split_indices(10, train_fraction=0.6, seed=123)
        train_b, eval_b = module.deterministic_split_indices(10, train_fraction=0.6, seed=123)
        self.assertEqual((train_a, eval_a), (train_b, eval_b))
        self.assertTrue(set(train_a).isdisjoint(eval_a))
        self.assertEqual(len(train_a), 6)
        self.assertEqual(len(eval_a), 4)

    def test_evaluate_examples_reports_required_metrics(self):
        module = load_eval_module()
        examples = [self.example(1), self.example(2), self.example(3), self.example(1)]
        report = module.evaluate_examples(
            examples,
            train_fraction=0.5,
            seed=20260603,
            epochs=1,
            learning_rate=0.01,
            hidden_dim=6,
        )
        self.assertEqual(report["example_count"], 4)
        self.assertEqual(report["train_count"], 2)
        self.assertEqual(report["eval_count"], 2)
        self.assertIn("weak_only_mean_nll", report["metrics"])
        self.assertIn("rule_lambda_mean_nll", report["metrics"])
        self.assertIn("learned_lambda_mean_nll", report["metrics"])
        self.assertIn("leakage_control_assumption", report)

    def test_main_writes_report(self):
        module = load_eval_module()
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            examples_path = tmp_path / "examples.jsonl"
            output_path = tmp_path / "evaluation.json"
            examples = [self.example(1), self.example(2), self.example(3), self.example(1)]
            examples_path.write_text("\\n".join(json.dumps(example) for example in examples) + "\\n", encoding="utf-8")
            module.main([
                "--examples-jsonl",
                str(examples_path),
                "--output-json",
                str(output_path),
                "--train-fraction",
                "0.5",
                "--epochs",
                "1",
                "--hidden-dim",
                "6",
            ])
            report = json.loads(output_path.read_text(encoding="utf-8"))
        self.assertEqual(report["example_count"], 4)
```

- [ ] **Step 2: Run tests to verify failure**

Run:

```bash
"/Users/wang/Documents/New project/.venv/bin/python" -m unittest tests.test_retrospective_lambda_evaluation -v
```

Expected: FAIL because script is missing.

- [ ] **Step 3: Implement evaluation script**

Create `scripts/evaluate_retrospective_lambda_model.py` with:

```python
import argparse
import json
import math
import random
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "scripts"))

import train_retrospective_lambda_model as lambda_training  # noqa: E402
import torch  # noqa: E402

LEAKAGE_CONTROL_ASSUMPTION = (
    "Input examples must have been generated with query outcomes hidden from retrieval, reranking, "
    "feature construction, lambda selection, and model selection. The held-out outcome is used only "
    "for predictive loss and evaluation."
)


def deterministic_split_indices(count: int, train_fraction: float, seed: int) -> tuple[list[int], list[int]]:
    if count < 2:
        raise ValueError("at least two examples are required for retrospective evaluation")
    if train_fraction <= 0.0 or train_fraction >= 1.0:
        raise ValueError("train_fraction must be in (0, 1)")
    indices = list(range(count))
    rng = random.Random(seed)
    rng.shuffle(indices)
    train_count = max(1, min(count - 1, int(round(count * train_fraction))))
    return sorted(indices[:train_count]), sorted(indices[train_count:])


def mean_nll(losses: list[float]) -> float:
    if not losses:
        return math.nan
    return float(sum(losses) / len(losses))


def evaluate_examples(
    examples: list[dict[str, Any]],
    train_fraction: float,
    seed: int,
    epochs: int,
    learning_rate: float,
    hidden_dim: int,
) -> dict[str, Any]:
    if not examples:
        raise ValueError("examples must not be empty")
    train_indices, eval_indices = deterministic_split_indices(len(examples), train_fraction, seed)
    train_examples = [examples[index] for index in train_indices]
    eval_examples = [examples[index] for index in eval_indices]
    training_summary = lambda_training.train_model(
        train_examples,
        epochs=epochs,
        learning_rate=learning_rate,
        hidden_dim=hidden_dim,
    )
    model = training_summary["model"]
    learned_losses = [
        float(lambda_training.predictive_loss_for_example(model, example, rho=0.0).detach().item())
        for example in eval_examples
    ]
    weak_losses = [float(lambda_training.weak_only_loss_for_example(example)) for example in eval_examples]
    rule_losses = [float(lambda_training.rule_lambda_loss_for_example(example)) for example in eval_examples]
    learned = mean_nll(learned_losses)
    rule = mean_nll(rule_losses)
    return {
        "example_count": len(examples),
        "train_count": len(train_examples),
        "eval_count": len(eval_examples),
        "train_indices": train_indices,
        "eval_indices": eval_indices,
        "seed": seed,
        "train_fraction": train_fraction,
        "metrics": {
            "weak_only_mean_nll": mean_nll(weak_losses),
            "rule_lambda_mean_nll": rule,
            "learned_lambda_mean_nll": learned,
            "learned_minus_rule_mean_nll": learned - rule,
        },
        "leakage_control_assumption": LEAKAGE_CONTROL_ASSUMPTION,
    }
```

Also add `main()` supporting `--examples-jsonl` and `--pipeline-results-jsonl`.

If `train_model` currently returns no model object, update Task 4 implementation to include `"model": model` in returned summary while keeping JSON output free of the model object by filtering before write.

- [ ] **Step 4: Add NLL helpers in training script**

In `scripts/train_retrospective_lambda_model.py` add:

```python
def weak_only_loss_for_example(example: dict[str, Any]) -> torch.Tensor:
    tensors = _validated_example_tensors(example)
    count = torch.tensor(tensors["query_count"], dtype=torch.float32)
    denominator = torch.tensor(tensors["query_denominator"], dtype=torch.float32)
    return -beta_binomial_log_predictive(count, denominator, torch.tensor(1.0), torch.tensor(1.0))


def rule_lambda_loss_for_example(example: dict[str, Any]) -> torch.Tensor:
    tensors = _validated_example_tensors(example)
    lambda0 = torch.tensor(tensors["lambda0"], dtype=torch.float32)
    lambda_rule = torch.tensor(tensors["lambda_rule_values"], dtype=torch.float32)
    if lambda_rule.sum().item() > 0:
        lambda_rule = lambda_rule / lambda_rule.sum() * torch.clamp(1.0 - lambda0, min=0.0)
    count = torch.tensor(tensors["query_count"], dtype=torch.float32)
    denominator = torch.tensor(tensors["query_denominator"], dtype=torch.float32)
    weak = beta_binomial_log_predictive(count, denominator, torch.tensor(1.0), torch.tensor(1.0))
    comp = beta_binomial_log_predictive(count, denominator, tensors["alpha"], tensors["beta"])
    terms = torch.cat([(lambda0.clamp_min(1e-12).log() + weak).reshape(1), lambda_rule.clamp_min(1e-12).log() + comp])
    return -torch.logsumexp(terms, dim=0)
```

- [ ] **Step 5: Run evaluation tests and commit**

Run:

```bash
"/Users/wang/Documents/New project/.venv/bin/python" -m unittest tests.test_retrospective_lambda_evaluation tests.test_retrospective_lambda_training -v
```

Expected: PASS.

Commit:

```bash
git add scripts/evaluate_retrospective_lambda_model.py scripts/train_retrospective_lambda_model.py tests/test_retrospective_lambda_evaluation.py tests/test_retrospective_lambda_training.py
git commit -m "Add retrospective lambda evaluation"
```

---

## Task 6: Documentation and Full Verification

**Files:**
- Modify: `docs/trial2vec_secret_mixture_prior_update_report_2026-06-03.md`
- Modify as needed: `README.md`, `web_agent/README.md`

- [ ] **Step 1: Update report limitations and usage**

In `docs/trial2vec_secret_mixture_prior_update_report_2026-06-03.md` update Section 7:

```text
1. SECRET-style backend is now runnable as an MVP.
2. Lambda training now uses the full 9-feature x_i vector.
3. Retrospective evaluation is available through scripts/evaluate_retrospective_lambda_model.py.
5. Mixture prior remains rule-based by default, with optional retrospective_calibrated mode when a lambda model artifact is supplied.
```

Add explicit caveat:

```text
No expert review labels were used. Retrospective predictive calibration is a substitute training signal, not expert validation.
```

- [ ] **Step 2: Add command examples**

Add examples:

```bash
python3 docs/oncology_trial_similarity_pipeline.py build-secret-index \
  --index-dir artifacts/oncology_trial_similarity_clinicalbert \
  --embedding-backend clinicalbert

python3 scripts/train_retrospective_lambda_model.py \
  --pipeline-results-jsonl artifacts/pseudo_query_results.jsonl \
  --output-json artifacts/lambda_training_summary.json \
  --model-output artifacts/lambda_model.pt

python3 scripts/evaluate_retrospective_lambda_model.py \
  --pipeline-results-jsonl artifacts/pseudo_query_results.jsonl \
  --output-json artifacts/retrospective_lambda_evaluation.json
```

- [ ] **Step 3: Run full unit tests**

Run:

```bash
"/Users/wang/Documents/New project/.venv/bin/python" -m unittest discover -s tests -v
```

Expected: all tests pass.

- [ ] **Step 4: Run py_compile**

Run:

```bash
"/Users/wang/Documents/New project/.venv/bin/python" -m py_compile docs/oncology_trial_similarity_pipeline.py docs/mixture_prior.py docs/secret_retrieval.py scripts/build_trial2vec_index.py scripts/train_retrospective_lambda_model.py scripts/evaluate_retrospective_lambda_model.py
```

Expected: exit 0.

- [ ] **Step 5: Optional local smoke commands**

Use hashing for a fast no-model smoke:

```bash
"/Users/wang/Documents/New project/.venv/bin/python" docs/oncology_trial_similarity_pipeline.py build-secret-index \
  --index-dir /Users/wang/Documents/New\ project/artifacts/oncology_trial_similarity_clinicalbert \
  --output-path /Users/wang/Documents/New\ project/artifacts/oncology_trial_similarity_clinicalbert/secret_embeddings_smoke.npz \
  --embedding-backend hashing
```

Then run search with one known query:

```bash
"/Users/wang/Documents/New project/.venv/bin/python" docs/oncology_trial_similarity_pipeline.py search \
  --query-json /Users/wang/PHD/clinic.gov/Oncology_All_Trials/Oncology_All_Trials/NCT02413320/NCT02413320_data.json \
  --index-dir /Users/wang/Documents/New\ project/artifacts/oncology_trial_similarity_clinicalbert \
  --retrieval-backend secret \
  --secret-index-path /Users/wang/Documents/New\ project/artifacts/oncology_trial_similarity_clinicalbert/secret_embeddings_smoke.npz \
  --top-k 10 \
  --rerank \
  --rerank-top-n 10 \
  --output /Users/wang/Documents/New\ project/artifacts/smoke_secret_mixture_prior.json
```

Expected: output JSON contains `retrieval_backend: "secret"`, `secret_section_scores`, `reranked_top10`, and `bayesian_analysis.status`.

- [ ] **Step 6: Request final code review**

Dispatch reviewer with:

```text
Review the implementation of SECRET-style retrieval, full lambda features, retrospective evaluation, and calibrated mixture-prior mode. Check for leakage risks, backwards compatibility, missing validations, and overclaiming.
```

Fix any Critical or Important findings before completion.

- [ ] **Step 7: Commit docs and final verification**

Commit:

```bash
git add docs/trial2vec_secret_mixture_prior_update_report_2026-06-03.md README.md web_agent/README.md
git commit -m "Document SECRET retrospective calibrated pipeline"
```

If final review fixes changed code, commit those separately with a focused message.

---

## Self-Review Checklist

- Spec coverage: Tasks 1-2 cover runnable SECRET-style backend. Task 3 covers full `x_i` and eligibility/redflag features. Task 5 covers retrospective evaluation without expert labels. Task 4 covers optional calibrated mixture-prior mode. Task 6 covers docs and verification.
- Placeholder scan: no task uses TBD/TODO/fill-in placeholders; each behavior has concrete tests and implementation snippets.
- Type consistency: feature names are centralized in `LAMBDA_FEATURE_NAMES`; model artifact validation uses the same list; mixture mode names are `rule` and `retrospective_calibrated`.
- Backward compatibility: rule mixture mode remains default; ClinicalBERT, hashing, and Trial2Vec tests must continue to pass.
