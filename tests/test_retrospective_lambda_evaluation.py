import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
from unittest import mock


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
            "components": [
                {
                    "alpha": 4.0,
                    "beta": 2.0,
                    "gate": 1.0,
                    "discount": 0.75,
                    "denominator": 20.0,
                }
            ],
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

    def test_evaluate_examples_reports_required_counts_metrics_and_leakage_assumption(self):
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
        self.assertIn("learned_minus_rule_mean_nll", report["metrics"])
        self.assertIn("leakage_control_assumption", report)
        assumption = report["leakage_control_assumption"].lower()
        self.assertIn("query outcomes must be hidden", assumption)
        self.assertIn("retrieval", assumption)
        self.assertIn("reranking", assumption)
        self.assertIn("feature construction", assumption)
        self.assertIn("model selection", assumption)
        self.assertIn("post-retrieval predictive loss/evaluation/analysis", assumption)

    def test_evaluate_examples_uses_pure_learned_predictive_nll(self):
        module = load_eval_module()
        examples = [self.example(1), self.example(2), self.example(3), self.example(1)]

        with mock.patch.object(
            module.lambda_training,
            "learned_lambda_loss_for_example",
            wraps=module.lambda_training.learned_lambda_loss_for_example,
        ) as learned_loss:
            report = module.evaluate_examples(
                examples,
                train_fraction=0.5,
                seed=20260603,
                epochs=1,
                learning_rate=0.01,
                hidden_dim=6,
            )

        self.assertEqual(learned_loss.call_count, report["eval_count"])

    def test_main_writes_report_without_model_object(self):
        module = load_eval_module()
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            examples_path = tmp_path / "examples.jsonl"
            output_path = tmp_path / "evaluation.json"
            examples = [self.example(1), self.example(2), self.example(3), self.example(1)]
            examples_path.write_text(
                "\n".join(json.dumps(example) for example in examples) + "\n",
                encoding="utf-8",
            )

            module.main(
                [
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
                ]
            )

            report = json.loads(output_path.read_text(encoding="utf-8"))

        self.assertEqual(report["example_count"], 4)
        self.assertNotIn("model", report)

    def test_main_rejects_pipeline_results_without_leakage_metadata(self):
        module = load_eval_module()
        result = {
            "query_summary": {
                "endpoints": {
                    "primary": [
                        {
                            "title": "Objective Response Rate",
                            "endpoint_family": "ORR/CR/PR",
                            "arm_results": [
                                {"arm": "Experimental", "count": 12, "denominator": 40}
                            ],
                        }
                    ]
                }
            },
            "reranked_top_matches": [],
        }
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            pipeline_path = tmp_path / "pipeline.jsonl"
            output_path = tmp_path / "evaluation.json"
            pipeline_path.write_text(json.dumps(result) + "\n", encoding="utf-8")

            with self.assertRaisesRegex(ValueError, "query_outcomes_hidden_from_retrieval"):
                module.main(
                    [
                        "--pipeline-results-jsonl",
                        str(pipeline_path),
                        "--output-json",
                        str(output_path),
                    ]
                )


if __name__ == "__main__":
    unittest.main()
