from __future__ import annotations

import importlib
from pathlib import Path
import unittest
from unittest import mock


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
        self.assertIn("clinicalbert", pipeline.RETRIEVAL_BACKENDS)
        self.assertIn("trial2vec", pipeline.RETRIEVAL_BACKENDS)
        self.assertIn("secret", pipeline.RETRIEVAL_BACKENDS)

    def test_secret_backend_is_explicitly_not_implemented(self) -> None:
        with self.assertRaisesRegex(NotImplementedError, "SECRET retrieval"):
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
