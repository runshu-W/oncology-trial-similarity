import importlib.util
from pathlib import Path
import unittest


SCRIPT_PATH = Path(__file__).resolve().parents[1] / "pipeline" / "run_oncology_retrospective_lambda_training.py"


def load_run_module():
    spec = importlib.util.spec_from_file_location("run_oncology_retrospective_lambda_training", SCRIPT_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class RetrospectiveLambdaFullEvaluationTests(unittest.TestCase):
    def example(self, nct_id, year=None, disease="Lung"):
        metadata = {
            "nct_id": nct_id,
            "primary_site": disease,
            "histology": "NSCLC",
            "primary_completion_date": str(year) if year is not None else None,
        }
        return {
            "query_nct_id": nct_id,
            "query_metadata": metadata,
            "query": {"count": 2, "denominator": 4},
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
            "features": [[0.8, 1.0, 0.8, 1.0, 0.2, 0.5, 1.0, 0.0, 3.0]],
            "components": [{"alpha": 2.0, "beta": 3.0}],
            "lambda_rule": [0.8],
        }

    def test_temporal_split_prefers_dates_and_falls_back_to_nct_proxy(self):
        module = load_run_module()
        examples = [
            self.example("NCT00000003", year=None),
            self.example("NCT00000001", year=2010),
            self.example("NCT00000002", year=2012),
            self.example("NCT00000004", year=None),
        ]

        train, eval_, metadata = module.temporal_split_indices(examples, train_fraction=0.5)

        self.assertEqual(len(train), 2)
        self.assertEqual(len(eval_), 2)
        self.assertEqual(metadata["temporal_key_sources"]["primary_completion_date"], 2)
        self.assertEqual(metadata["temporal_key_sources"]["nct_id_numeric_proxy"], 2)
        self.assertTrue(set(train).isdisjoint(eval_))

    def test_temporal_key_prefers_true_temporal_sort_date_metadata(self):
        module = load_run_module()
        example = self.example("NCT99999999", year=None)
        example["query_metadata"]["temporal_sort_date"] = "2020-06-15"
        example["query_metadata"]["temporal_sort_source"] = "primary_completion_date"

        key, source = module.temporal_key_for_example(example)

        self.assertAlmostEqual(key, 2020.454, places=3)
        self.assertEqual(source, "primary_completion_date")

    def test_stratified_metric_rows_reports_group_metrics(self):
        module = load_run_module()
        prediction_rows = [
            {
                "split": "eval",
                "example_index": 0,
                "disease_group": "Lung",
                "observed_rate": 0.3,
                "learned_predicted_rate": 0.4,
                "rule_predicted_rate": 0.5,
            },
            {
                "split": "eval",
                "example_index": 1,
                "disease_group": "Breast",
                "observed_rate": 0.6,
                "learned_predicted_rate": 0.55,
                "rule_predicted_rate": 0.5,
            },
        ]
        nll_rows = [
            {"example_index": 0, "weak_nll": 2.0, "rule_nll": 1.5, "learned_nll": 1.2},
            {"example_index": 1, "weak_nll": 2.0, "rule_nll": 1.4, "learned_nll": 1.6},
        ]

        rows = module.stratified_metric_rows(prediction_rows, nll_rows, group_key="disease_group")

        self.assertEqual({row["group"] for row in rows}, {"Breast", "Lung"})
        lung = next(row for row in rows if row["group"] == "Lung")
        self.assertAlmostEqual(lung["learned_minus_rule_nll"], -0.3)

    def test_simulation_operating_characteristics_have_stable_columns(self):
        module = load_run_module()
        prediction_rows = [
            {
                "split": "eval",
                "query_denominator": 5,
                "observed_rate": 0.2,
                "learned_predicted_rate": 0.3,
            },
            {
                "split": "eval",
                "query_denominator": 5,
                "observed_rate": 0.8,
                "learned_predicted_rate": 0.7,
            },
        ]

        rows = module.simulation_operating_characteristics(prediction_rows, iterations=5, seed=123)

        self.assertEqual(len(rows), 3)
        for row in rows:
            self.assertIn("type_i_error", row)
            self.assertIn("power", row)
            self.assertIn("mse", row)
            self.assertEqual(row["iterations"], 5)

    def test_train_model_with_curves_records_listwise_settings(self):
        module = load_run_module()
        examples = [
            self.example("NCT00000001", year=2010),
            self.example("NCT00000002", year=2011),
            self.example("NCT00000003", year=2012),
        ]

        summary = module.train_model_with_curves(
            examples,
            train_indices=[0, 1],
            eval_indices=[2],
            epochs=1,
            learning_rate=0.01,
            hidden_dim=6,
            model_type="mlp",
            seed=123,
            listwise_eta=0.2,
            listwise_temperature=1.5,
        )

        self.assertEqual(summary["listwise_eta"], 0.2)
        self.assertEqual(summary["listwise_temperature"], 1.5)


if __name__ == "__main__":
    unittest.main()
