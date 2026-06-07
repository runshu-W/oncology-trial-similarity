import importlib.util
from pathlib import Path
import tempfile
import unittest


SCRIPT_PATH = Path(__file__).resolve().parents[1] / "scripts" / "run_temporal_borrowing_validation.py"


def load_temporal_borrowing_module():
    spec = importlib.util.spec_from_file_location("run_temporal_borrowing_validation", SCRIPT_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class TemporalBorrowingValidationTests(unittest.TestCase):
    def example(self, nct_id, date, count=3, denominator=10):
        return {
            "query_nct_id": nct_id,
            "query": {"count": count, "denominator": denominator},
            "query_metadata": {
                "temporal_sort_date": date,
                "temporal_sort_source": "primary_completion_date",
            },
            "lambda_0": 0.2,
            "components": [
                {
                    "count": 3,
                    "denominator": 10,
                    "lambda_rule": 0.6,
                    "discount": 0.5,
                }
            ],
            "lambda_rule": [0.6],
        }

    def test_date_based_temporal_nll_rows_include_baseline_and_learned_methods(self):
        module = load_temporal_borrowing_module()
        examples = [
            self.example("NCT1", "2019-01-01"),
            self.example("NCT2", "2021-01-01"),
            self.example("NCT3", "2022-01-01"),
        ]
        learned_rows = {
            "NCT2": {"two_head_trained": 1.2},
            "NCT3": {"two_head_trained": 1.4},
        }

        rows = module.date_based_temporal_nll_rows(
            examples,
            cutoffs=["2020-12-31"],
            methods=("weak_only", "rule"),
            learned_nll_by_query=learned_rows,
        )

        methods = {row["method"] for row in rows}
        self.assertEqual(methods, {"weak_only", "rule", "two_head_trained"})
        for row in rows:
            self.assertEqual(row["split_strategy"], "date_based")
            self.assertEqual(row["eval_count"], 2)
            self.assertEqual(row["train_count"], 1)
            self.assertEqual(row["split_label"], "train_through_2020-12-31")
        learned = next(row for row in rows if row["method"] == "two_head_trained")
        self.assertAlmostEqual(learned["mean_nll"], 1.3)

    def test_read_learned_nll_csv_keys_by_query_id(self):
        module = load_temporal_borrowing_module()
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "lambda_nll_rows.csv"
            path.write_text(
                "example_index,query_nct_id,learned_nll\n"
                "0,NCT1,2.0\n"
                "1,NCT2,1.5\n",
                encoding="utf-8",
            )

            rows = module.read_learned_nll_csv(path)

        self.assertEqual(rows["NCT1"]["two_head_trained"], 2.0)
        self.assertEqual(rows["NCT2"]["two_head_trained"], 1.5)


if __name__ == "__main__":
    unittest.main()
