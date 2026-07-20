import importlib.util
from pathlib import Path
import tempfile
import unittest


SCRIPT_PATH = Path(__file__).resolve().parents[1] / "pipeline" / "run_borrowing_baseline_comparison.py"


def load_baseline_module():
    spec = importlib.util.spec_from_file_location("run_borrowing_baseline_comparison", SCRIPT_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class BorrowingBaselineComparisonTests(unittest.TestCase):
    def example(self):
        return {
            "query_nct_id": "NCTQ",
            "query": {"count": 3, "denominator": 10},
            "lambda_0": 0.2,
            "components": [
                {
                    "candidate_nct_id": "NCTH1",
                    "count": 4,
                    "denominator": 12,
                    "alpha": 3.0,
                    "beta": 5.0,
                    "lambda_rule": 0.6,
                    "lambda_model": 0.5,
                    "discount": 0.5,
                    "discount_model": 0.5,
                },
                {
                    "candidate_nct_id": "NCTH2",
                    "count": 1,
                    "denominator": 8,
                    "alpha": 1.25,
                    "beta": 2.75,
                    "lambda_rule": 0.2,
                    "lambda_model": 0.3,
                    "discount": 0.25,
                    "discount_model": 0.25,
                },
            ],
            "lambda_rule": [0.6, 0.2],
        }

    def test_baseline_nll_rows_include_requested_methods(self):
        module = load_baseline_module()

        rows = module.baseline_nll_rows(
            [self.example()],
            methods=("weak_only", "rule", "fixed_discount", "map_like", "power_prior_like", "model", "model_sam"),
            fixed_discount=0.25,
        )

        self.assertEqual(
            {row["method"] for row in rows},
            {"weak_only", "rule", "fixed_discount", "map_like", "power_prior_like", "model", "model_sam"},
        )
        for row in rows:
            self.assertEqual(row["query_nct_id"], "NCTQ")
            self.assertIn("nll", row)
            self.assertIn("predictive_probability", row)
            self.assertIn("historical_mass", row)
            self.assertGreater(row["predictive_probability"], 0.0)

    def test_summarize_baseline_rows_reports_mean_nll_and_delta_vs_rule(self):
        module = load_baseline_module()
        rows = [
            {"method": "rule", "nll": 2.0},
            {"method": "weak_only", "nll": 2.5},
            {"method": "model", "nll": 1.7},
        ]

        summary = module.summarize_baseline_rows(rows, reference_method="rule")

        model = next(row for row in summary if row["method"] == "model")
        self.assertAlmostEqual(model["mean_nll"], 1.7)
        self.assertAlmostEqual(model["mean_nll_minus_rule"], -0.3)

    def test_learned_nll_rows_from_csv_preserve_trained_two_head_values(self):
        module = load_baseline_module()
        with tempfile.TemporaryDirectory() as tmpdir:
            csv_path = Path(tmpdir) / "lambda_nll_rows.csv"
            csv_path.write_text(
                "example_index,query_nct_id,weak_nll,rule_nll,learned_nll\n"
                "0,NCT1,3.0,2.5,2.0\n"
                "1,NCT2,4.0,3.5,1.5\n",
                encoding="utf-8",
            )

            rows = module.learned_nll_rows_from_csv(
                csv_path,
                method_name="two_head_trained",
                nll_column="learned_nll",
            )

        self.assertEqual(len(rows), 2)
        self.assertEqual(rows[0]["method"], "two_head_trained")
        self.assertEqual(rows[0]["query_nct_id"], "NCT1")
        self.assertAlmostEqual(rows[0]["nll"], 2.0)
        self.assertAlmostEqual(rows[1]["nll"], 1.5)
        self.assertGreater(rows[0]["predictive_probability"], 0.0)
        self.assertEqual(rows[0]["sam_status"], "from_learned_nll_csv")


if __name__ == "__main__":
    unittest.main()
