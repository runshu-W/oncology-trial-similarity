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


if __name__ == "__main__":
    unittest.main()
