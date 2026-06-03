from __future__ import annotations

import importlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest import mock

import numpy as np


pipeline = importlib.import_module("docs.oncology_trial_similarity_pipeline")


def load_trial2vec_builder_module(module_name: str = "build_trial2vec_index"):
    path = Path(__file__).resolve().parents[1] / "scripts" / "build_trial2vec_index.py"
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load module spec for {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class Stage1BackendTests(unittest.TestCase):
    def test_default_retrieval_backend_is_clinicalbert_compatible(self) -> None:
        self.assertEqual(pipeline.DEFAULT_RETRIEVAL_BACKEND, "clinicalbert")
        self.assertEqual(pipeline.DEFAULT_INDEX_EMBEDDING_BACKEND, "clinicalbert")
        self.assertIn("clinicalbert", pipeline.RETRIEVAL_BACKENDS)
        self.assertIn("trial2vec", pipeline.RETRIEVAL_BACKENDS)
        self.assertIn("secret", pipeline.RETRIEVAL_BACKENDS)

    def test_search_reports_hashing_when_index_uses_hashing_embeddings(self) -> None:
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
                json.dumps({"nct_id": "NCTCANDIDATE", "brief_title": "Candidate"}) + "\n",
                encoding="utf-8",
            )
            embeddings = {
                aspect: np.zeros((1, 2048), dtype=np.float32)
                for aspect in pipeline.ASPECT_WEIGHTS
            }
            np.savez_compressed(
                index_dir / "trial_embeddings.npz",
                nct_ids=np.array(["NCTCANDIDATE"]),
                embedding_backend=np.array(["hashing"]),
                embedding_model=np.array(["signed-token-hashing-2048"]),
                **embeddings,
            )

            result = pipeline.search(query_path, index_dir, top_k=1)

        self.assertEqual(result["retrieval_backend"], "hashing")
        self.assertEqual(result["embedding_backend"], "hashing")
        self.assertEqual(result["top_matches"][0]["retrieval_backend"], "hashing")

    def test_redact_query_outcomes_for_retrieval_removes_counts_but_keeps_endpoint_labels(self) -> None:
        summary = {
            "nct_id": "NCTQUERY",
            "endpoints": {
                "primary": [
                    {
                        "title": "Objective Response Rate",
                        "endpoint_family": "ORR/CR/PR",
                        "time_frame": "24 weeks",
                        "arm_results": [
                            {"arm": "Experimental", "count": 12, "denominator": 40}
                        ],
                        "denominators": [{"arm": "Experimental", "denominator": 40}],
                        "measurements": [{"arm": "Experimental", "count": 12}],
                    }
                ],
                "secondary_or_other": [],
            },
            "results": {
                "has_posted_results": True,
                "primary_results": [{"title": "Objective Response Rate"}],
                "safety_results": [{"title": "Adverse Events"}],
                "denominators": [{"denominator": 40}],
                "follow_up_duration": "24 weeks",
            },
            "borrowing_relevance": {
                "borrowable_quantities": [
                    {"endpoint": "Objective Response Rate", "arm_results": []}
                ]
            },
        }

        redacted = pipeline.redact_query_outcomes_for_retrieval(summary)

        endpoint = redacted["endpoints"]["primary"][0]
        self.assertEqual(endpoint["title"], "Objective Response Rate")
        self.assertEqual(endpoint["endpoint_family"], "ORR/CR/PR")
        self.assertNotIn("arm_results", endpoint)
        self.assertNotIn("denominators", endpoint)
        self.assertNotIn("measurements", endpoint)
        self.assertFalse(redacted["results"]["has_posted_results"])
        self.assertEqual(redacted["results"]["primary_results"], [])
        self.assertEqual(redacted["borrowing_relevance"]["borrowable_quantities"], [])
        self.assertNotIn("12", redacted.get("one_paragraph_summary_for_embedding", ""))
        self.assertNotIn("40", redacted.get("one_paragraph_summary_for_embedding", ""))
        self.assertIn("arm_results", summary["endpoints"]["primary"][0])

    def test_trial2vec_row_uses_redacted_summary_text_when_query_outcomes_are_hidden(self) -> None:
        summary = {
            "nct_id": "NCTQUERY",
            "brief_title": "Query Trial",
            "brief_summary": "Tests treatment in lung cancer.",
            "one_paragraph_summary_for_embedding": (
                "Objective Response Rate arm_results count 12 denominator 40 proportion 0.3"
            ),
            "endpoints": {
                "primary": [
                    {
                        "title": "Objective Response Rate",
                        "endpoint_family": "ORR/CR/PR",
                        "arm_results": [
                            {
                                "arm": "Experimental",
                                "count": 12,
                                "denominator": 40,
                                "proportion": 0.3,
                            }
                        ],
                    }
                ]
            },
            "results": {"has_posted_results": True},
            "borrowing_relevance": {"borrowable_quantities": []},
        }

        row = pipeline.summary_to_trial2vec_row(
            pipeline.redact_query_outcomes_for_retrieval(summary)
        )

        self.assertIn("Objective Response Rate", row["outcome_measure"])
        self.assertNotIn("12", row["description"])
        self.assertNotIn("40", row["description"])
        self.assertNotIn("0.3", row["description"])

    def test_search_can_hide_query_outcomes_from_retrieval_output(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            index_dir = tmp_path / "index"
            index_dir.mkdir()
            query_path = tmp_path / "query.json"
            query_path.write_text(
                json.dumps(
                    {
                        "Study details": {
                            "1. NCT number": "NCTQUERY",
                            "5. Study Overview": {"Brief Title": "Query"},
                        },
                        "Results Posted": {
                            "5. Outcome measures": [
                                {
                                    "Type": "PRIMARY",
                                    "Title": "Objective Response Rate",
                                    "Time Frame": "24 weeks",
                                    "Data Table": [
                                        {"Category": "Measurement", "Experimental Arm": "12"},
                                        {"Category": "Denominator", "Experimental Arm": "40"},
                                    ],
                                }
                            ]
                        },
                    }
                ),
                encoding="utf-8",
            )
            (index_dir / "trial_summaries.jsonl").write_text(
                json.dumps({"nct_id": "NCTCANDIDATE", "brief_title": "Candidate"}) + "\n",
                encoding="utf-8",
            )
            embeddings = {
                aspect: np.zeros((1, 2048), dtype=np.float32)
                for aspect in pipeline.ASPECT_WEIGHTS
            }
            np.savez_compressed(
                index_dir / "trial_embeddings.npz",
                nct_ids=np.array(["NCTCANDIDATE"]),
                embedding_backend=np.array(["hashing"]),
                embedding_model=np.array(["signed-token-hashing-2048"]),
                **embeddings,
            )

            result = pipeline.search(
                query_path,
                index_dir,
                top_k=1,
                hide_query_outcomes_for_retrieval=True,
            )

        self.assertTrue(
            result["retrospective_leakage_control"]["query_outcomes_hidden_from_retrieval"]
        )
        self.assertNotIn("arm_results", result["query_summary"]["endpoints"]["primary"][0])
        self.assertIn("arm_results", result["heldout_query_outcomes"]["endpoints"]["primary"][0])

    def test_secret_backend_is_supported_by_backend_guardrail(self) -> None:
        pipeline.ensure_supported_retrieval_backend("secret")

    def test_unknown_backend_has_clear_error(self) -> None:
        with self.assertRaisesRegex(ValueError, "Unsupported retrieval backend"):
            pipeline.ensure_supported_retrieval_backend("not_a_backend")

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

    def test_trial2vec_row_handles_missing_null_and_malformed_nested_fields(self) -> None:
        malformed_summaries = [
            {},
            {
                "endpoints": None,
                "population": None,
                "cancer_type": None,
                "intervention": None,
            },
            {
                "endpoints": {"primary": [None, "bad endpoint", {"title": None}]},
                "population": {"key_inclusion": None, "key_exclusion": []},
                "cancer_type": None,
                "intervention": None,
            },
        ]

        for summary in malformed_summaries:
            with self.subTest(summary=summary):
                row = pipeline.summary_to_trial2vec_row(summary)

                self.assertEqual(row["criteria"], "")
                self.assertEqual(row["outcome_measure"], "")
                self.assertEqual(row["disease"], "")
                self.assertEqual(row["intervention_name"], "")

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

    def test_pipeline_adds_applymap_alias_when_pandas_dataframe_only_has_map(self) -> None:
        class FakeDataFrame:
            def map(self):
                return None

        class FakePandas:
            DataFrame = FakeDataFrame

        pipeline.ensure_pandas_applymap_compat(FakePandas)

        self.assertIs(FakeDataFrame.applymap, FakeDataFrame.map)


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

            rows = secret.score_secret_index(
                query_vectors, path, summaries, excluded_nct_id="NCTQUERY", top_k=2
            )

        self.assertEqual([row["nct_id"] for row in rows], ["NCTGOOD", "NCTBAD"])
        self.assertEqual(rows[0]["retrieval_backend"], "secret")
        self.assertEqual(rows[0]["score_0_100"], 100.0)
        self.assertIn("disease_population", rows[0]["secret_section_scores"])
        self.assertIn("disease_population", rows[0]["secret_evidence"])

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

    def test_secret_summary_uses_borrowing_relevance_result_quantities(self) -> None:
        secret = importlib.import_module("docs.secret_retrieval")
        loaded_summary = {
            "nct_id": "NCT1",
            "results": {"has_posted_results": True, "denominators": [{"arm": "Experimental"}]},
            "borrowing_relevance": {
                "borrowable_quantities": [
                    {
                        "endpoint": "Pathologic complete response",
                        "endpoint_family": "ORR/CR/PR",
                        "arm_results": [
                            {"arm": "Experimental", "count": 12, "denominator": 40}
                        ],
                    }
                ]
            },
            "source_documents": {"json": "NCT1_data.json"},
        }

        sections = secret.secret_sections_from_summary(
            pipeline.secret_ready_summary(loaded_summary)
        )

        self.assertIn("Pathologic complete response", sections["results"])
        self.assertIn("denominator", sections["results"])
        self.assertIn("has_posted_results", sections["results"])

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
                json.dumps(
                    {
                        "nct_id": "NCT1",
                        "brief_title": "Candidate",
                        "cancer_type": {"primary_site": ["Breast"]},
                    }
                )
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


class Trial2VecIndexBuilderTests(unittest.TestCase):
    def test_builder_import_does_not_require_optional_ml_dependencies(self) -> None:
        original_import = __import__

        def guarded_import(name, *args, **kwargs):
            if name in {"pandas", "torch", "trial2vec"}:
                raise ImportError(f"blocked optional dependency: {name}")
            return original_import(name, *args, **kwargs)

        with mock.patch("builtins.__import__", guarded_import):
            module = load_trial2vec_builder_module("build_trial2vec_index_no_optional_deps")

        self.assertTrue(hasattr(module, "encode_trial2vec_index"))

    def test_optional_dependency_loader_reports_clear_runtime_error(self) -> None:
        module = load_trial2vec_builder_module("build_trial2vec_index_missing_deps")
        original_import = __import__

        def guarded_import(name, *args, **kwargs):
            if name in {"pandas", "torch", "trial2vec"}:
                raise ImportError(f"blocked optional dependency: {name}")
            return original_import(name, *args, **kwargs)

        with mock.patch("builtins.__import__", guarded_import):
            with self.assertRaisesRegex(RuntimeError, "optional Trial2Vec index dependencies"):
                module.load_trial2vec_index_dependencies()

    def test_builder_adds_applymap_alias_when_pandas_dataframe_only_has_map(self) -> None:
        module = load_trial2vec_builder_module("build_trial2vec_index_pandas_compat")

        class FakeDataFrame:
            def map(self):
                return None

        class FakePandas:
            DataFrame = FakeDataFrame

        module.ensure_pandas_applymap_compat(FakePandas)

        self.assertIs(FakeDataFrame.applymap, FakeDataFrame.map)

    def test_compatible_torch_load_only_adds_supported_weights_only_default(self) -> None:
        module = load_trial2vec_builder_module("build_trial2vec_index_torch_load")
        calls = []

        def load_with_weights(path, *, weights_only=None):
            calls.append((path, weights_only))
            return weights_only

        wrapped_with_weights = module.make_compatible_torch_load(load_with_weights)

        self.assertFalse(wrapped_with_weights("model.pt"))
        self.assertTrue(wrapped_with_weights("model.pt", weights_only=True))
        self.assertEqual(calls, [("model.pt", False), ("model.pt", True)])

        def load_without_weights(path):
            return path

        wrapped_without_weights = module.make_compatible_torch_load(load_without_weights)

        self.assertEqual(wrapped_without_weights("legacy.pt"), "legacy.pt")

    def test_compatible_load_state_dict_respects_existing_strict_argument(self) -> None:
        module = load_trial2vec_builder_module("build_trial2vec_index_load_state_dict")
        calls = []

        def load_state_dict(module_arg, state_dict, *args, **kwargs):
            calls.append((module_arg, state_dict, args, kwargs))
            if "strict" in kwargs:
                return kwargs["strict"]
            if args:
                return args[0]
            return None

        wrapped = module.make_compatible_load_state_dict(load_state_dict)

        self.assertFalse(wrapped("module", {"weight": 1}))
        self.assertTrue(wrapped("module", {"weight": 1}, True))
        self.assertTrue(wrapped("module", {"weight": 1}, strict=True))
        self.assertEqual(
            calls,
            [
                ("module", {"weight": 1}, (), {"strict": False}),
                ("module", {"weight": 1}, (True,), {}),
                ("module", {"weight": 1}, (), {"strict": True}),
            ],
        )


if __name__ == "__main__":
    unittest.main()
