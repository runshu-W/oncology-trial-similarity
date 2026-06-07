import importlib.util
from pathlib import Path
import tempfile
import unittest


SCRIPT_PATH = Path(__file__).resolve().parents[1] / "scripts" / "temporal_validation.py"


def load_temporal_module():
    spec = importlib.util.spec_from_file_location("temporal_validation", SCRIPT_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class TemporalValidationTests(unittest.TestCase):
    def example(self, nct_id, sort_date):
        return {
            "query_nct_id": nct_id,
            "query_metadata": {
                "temporal_sort_date": sort_date,
                "temporal_sort_source": "primary_completion_date",
            },
        }

    def test_date_based_split_uses_cutoff_date(self):
        module = load_temporal_module()
        examples = [
            self.example("NCT1", "2018-01-01"),
            self.example("NCT2", "2020-01-01"),
            self.example("NCT3", "2021-01-01"),
        ]

        train, eval_, metadata = module.date_based_split_indices(
            examples,
            train_end_date="2020-06-30",
        )

        self.assertEqual(train, [0, 1])
        self.assertEqual(eval_, [2])
        self.assertEqual(metadata["split_mode"], "date_based")
        self.assertEqual(metadata["train_end_date"], "2020-06-30")

    def test_rolling_origin_splits_create_forward_validation_windows(self):
        module = load_temporal_module()
        examples = [
            self.example("NCT1", "2018-01-01"),
            self.example("NCT2", "2019-01-01"),
            self.example("NCT3", "2020-01-01"),
            self.example("NCT4", "2021-01-01"),
        ]

        splits = module.rolling_origin_splits(
            examples,
            min_train_count=2,
            eval_window_size=1,
        )

        self.assertEqual(len(splits), 2)
        self.assertEqual(splits[0]["train_indices"], [0, 1])
        self.assertEqual(splits[0]["eval_indices"], [2])
        self.assertEqual(splits[1]["train_indices"], [0, 1, 2])
        self.assertEqual(splits[1]["eval_indices"], [3])

    def test_attach_date_metadata_to_examples_adds_true_temporal_fields(self):
        module = load_temporal_module()
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "dates.csv"
            path.write_text(
                "nct_id,primary_completion_date,primary_completion_date_precision,"
                "completion_date,completion_date_precision,results_first_posted_date,"
                "results_first_posted_date_precision,start_date,start_date_precision,"
                "temporal_sort_date,temporal_sort_source\n"
                "NCT1,2020-01-15,day,2020-02-15,day,2021-01-01,day,2018-01-01,day,2020-01-15,primary_completion_date\n",
                encoding="utf-8",
            )
            date_rows = module.read_date_metadata_csv(path)

        examples = [{"query_nct_id": "NCT1", "query_metadata": {"disease": "lung"}}]
        report = module.attach_date_metadata_to_examples(examples, date_rows)

        self.assertEqual(report["attached_count"], 1)
        metadata = examples[0]["query_metadata"]
        self.assertEqual(metadata["disease"], "lung")
        self.assertEqual(metadata["primary_completion_date"], "2020-01-15")
        self.assertEqual(metadata["temporal_sort_date"], "2020-01-15")
        self.assertEqual(module.temporal_key_for_example(examples[0])[1], "primary_completion_date")


if __name__ == "__main__":
    unittest.main()
