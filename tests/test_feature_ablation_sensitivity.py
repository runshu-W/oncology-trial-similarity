import importlib.util
from pathlib import Path
import tempfile
import unittest


SCRIPT_PATH = Path(__file__).resolve().parents[1] / "pipeline" / "run_feature_ablation_sensitivity.py"


def load_ablation_module():
    spec = importlib.util.spec_from_file_location("run_feature_ablation_sensitivity", SCRIPT_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class FeatureAblationSensitivityTests(unittest.TestCase):
    def example(self):
        feature_names = [
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
        return {
            "query_nct_id": "NCTQ",
            "query": {"count": 2, "denominator": 8},
            "lambda_0": 0.2,
            "feature_names": feature_names,
            "features": [
                [0.9, 1.0, 0.8, 1.0, 0.6, 0.5, 1.0, 0.0, 3.0],
                [0.2, 0.1, 0.3, 0.5, 0.4, 0.2, 0.5, -0.5, 2.0],
            ],
            "components": [
                {"alpha": 3.0, "beta": 5.0, "gate": 1.0},
                {"alpha": 1.5, "beta": 3.5, "gate": 1.0},
            ],
        }

    def test_feature_mask_zeroes_named_features_without_reordering(self):
        module = load_ablation_module()
        example = self.example()

        masked = module.mask_example_features(example, drop_features=["disease_match_i", "log_n_i"])

        self.assertEqual(masked["feature_names"], example["feature_names"])
        self.assertEqual(masked["features"][0][1], 0.0)
        self.assertEqual(masked["features"][0][8], 0.0)
        self.assertEqual(masked["features"][0][0], 0.9)

    def test_ablation_rows_include_full_and_drop_groups(self):
        module = load_ablation_module()

        rows = module.ablation_nll_rows([self.example()], ablations=module.default_feature_ablations())

        names = {row["ablation"] for row in rows}
        self.assertIn("full_9_feature_proxy", names)
        self.assertIn("drop_disease", names)
        for row in rows:
            self.assertIn("mean_nll", row)
            self.assertIn("mean_nll_minus_full", row)
            self.assertEqual(row["example_count"], 1)

    def test_section_weight_sensitivity_reweights_secret_section_scores(self):
        module = load_ablation_module()
        result = {
            "query_summary": {"nct_id": "NCTQ"},
            "top_matches": [
                {
                    "nct_id": "NCTC1",
                    "secret_section_scores": {
                        "disease_population": 1.0,
                        "intervention": 0.5,
                    },
                },
                {
                    "nct_id": "NCTC2",
                    "secret_section_scores": {
                        "disease_population": 0.2,
                        "intervention": 1.0,
                    },
                },
            ],
        }

        rows = module.section_weight_sensitivity_rows(
            [result],
            scenarios=[
                {"name": "disease_heavy", "weights": {"disease_population": 0.8, "intervention": 0.2}},
                {"name": "intervention_heavy", "weights": {"disease_population": 0.2, "intervention": 0.8}},
            ],
        )

        disease = next(row for row in rows if row["scenario"] == "disease_heavy")
        intervention = next(row for row in rows if row["scenario"] == "intervention_heavy")
        self.assertEqual(disease["top_candidate_nct_id"], "NCTC1")
        self.assertEqual(intervention["top_candidate_nct_id"], "NCTC2")

    def test_write_ablation_heatmap_svg_creates_svg_artifact(self):
        module = load_ablation_module()
        rows = [
            {"ablation": "full_9_feature_proxy", "mean_nll_minus_full": 0.0},
            {"ablation": "drop_disease", "mean_nll_minus_full": 0.2},
        ]

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "heatmap.svg"
            module.write_ablation_heatmap_svg(path, rows)

            self.assertTrue(path.exists())
            self.assertIn("<svg", path.read_text(encoding="utf-8"))


if __name__ == "__main__":
    unittest.main()
