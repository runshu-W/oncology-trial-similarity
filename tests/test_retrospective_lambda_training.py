import importlib.util
import json
import math
from pathlib import Path
import tempfile
import unittest

import torch


SCRIPT_PATH = Path(__file__).resolve().parents[1] / "scripts" / "train_retrospective_lambda_model.py"


def load_training_module():
    spec = importlib.util.spec_from_file_location("train_retrospective_lambda_model", SCRIPT_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class RetrospectiveLambdaTrainingTests(unittest.TestCase):
    def zero_model(self, module):
        model = module.LambdaScorer(input_dim=4, hidden_dim=6)
        with torch.no_grad():
            for parameter in model.parameters():
                parameter.zero_()
        return model

    def base_example(self):
        return {
            "query": {"count": 2, "denominator": 4},
            "lambda_0": 0.2,
            "features": [
                [0.9, 0.8, 1.0, 3.9],
                [0.1, 0.2, 0.5, 2.0],
            ],
            "components": [
                {"alpha": 4.0, "beta": 2.0, "gate": 1.0, "discount": 0.5, "denominator": 10.0},
                {"alpha": 1.5, "beta": 3.5, "gate": 1.0, "discount": 0.25, "denominator": 20.0},
            ],
            "lambda_rule": [0.6, 0.2],
        }

    def test_lambda_scorer_outputs_one_score_per_candidate(self):
        module = load_training_module()
        model = module.LambdaScorer(input_dim=4, hidden_dim=6)
        features = torch.tensor(
            [
                [0.9, 0.8, 1.0, 3.9],
                [0.1, 0.2, 0.5, 2.0],
                [0.4, 0.3, 0.7, 3.0],
            ],
            dtype=torch.float32,
        )

        scores = model(features)

        self.assertEqual(scores.shape, (3,))

    def test_predictive_loss_for_example_returns_finite_positive_loss(self):
        module = load_training_module()
        model = module.LambdaScorer(input_dim=4, hidden_dim=6)
        example = self.base_example()

        loss = module.predictive_loss_for_example(model, example, rho=0.1, ess_cap=100.0)

        self.assertEqual(loss.shape, ())
        self.assertTrue(torch.isfinite(loss).item())
        self.assertGreater(loss.item(), 0.0)
        self.assertTrue(math.isfinite(loss.item()))

    def test_lambda_rule_is_normalized_to_candidate_budget_for_kl(self):
        module = load_training_module()
        model = self.zero_model(module)
        example = self.base_example()
        example["lambda_rule"] = [0.9, 0.1]

        loss = module.predictive_loss_for_example(model, example, rho=0.25, ess_cap=100.0)

        count = torch.tensor(2.0)
        denominator = torch.tensor(4.0)
        weak_log_predictive = module.beta_binomial_log_predictive(
            count,
            denominator,
            torch.tensor(1.0),
            torch.tensor(1.0),
        )
        component_log_predictive = module.beta_binomial_log_predictive(
            count,
            denominator,
            torch.tensor([4.0, 1.5]),
            torch.tensor([2.0, 3.5]),
        )
        lambda_i = torch.tensor([0.4, 0.4])
        mixture_terms = torch.cat(
            [
                (torch.tensor(0.2).log() + weak_log_predictive).reshape(1),
                lambda_i.log() + component_log_predictive,
            ]
        )
        base_loss = -torch.logsumexp(mixture_terms, dim=0)
        normalized_rule = torch.tensor([0.72, 0.08])
        expected_kl = (
            normalized_rule
            * (normalized_rule.clamp_min(1e-12).log() - lambda_i.clamp_min(1e-12).log())
        ).sum()
        expected = base_loss + 0.25 * expected_kl

        self.assertAlmostEqual(loss.item(), expected.item(), places=6)

    def test_ess_penalty_uses_lambda_discount_denominator(self):
        module = load_training_module()
        model = self.zero_model(module)
        example = self.base_example()
        example["lambda_rule"] = [0.0, 0.0]
        for component in example["components"]:
            component["alpha"] = 1000.0
            component["beta"] = 1000.0
            component["discount"] = 0.0
            component["denominator"] = 1000.0

        loss_with_low_cap = module.predictive_loss_for_example(model, example, rho=0.0, ess_cap=1.0)
        loss_with_high_cap = module.predictive_loss_for_example(model, example, rho=0.0, ess_cap=100000.0)

        self.assertAlmostEqual(loss_with_low_cap.item(), loss_with_high_cap.item(), places=6)

    def test_all_zero_gates_fall_back_to_weak_only_loss(self):
        module = load_training_module()
        model = self.zero_model(module)
        example = self.base_example()
        example["components"][0]["gate"] = 0.0
        example["components"][1]["gate"] = 0.0

        loss = module.predictive_loss_for_example(model, example, rho=0.0, ess_cap=100.0)

        expected = -module.beta_binomial_log_predictive(
            torch.tensor(2.0),
            torch.tensor(4.0),
            torch.tensor(1.0),
            torch.tensor(1.0),
        )
        self.assertTrue(torch.isfinite(loss).item())
        self.assertAlmostEqual(loss.item(), expected.item(), places=6)

    def test_large_model_scores_do_not_overflow_to_nan(self):
        module = load_training_module()

        class HugeScoreModel(torch.nn.Module):
            def forward(self, features):
                return torch.full((features.shape[0],), 1000.0)

        loss = module.predictive_loss_for_example(
            HugeScoreModel(),
            self.base_example(),
            rho=0.0,
            ess_cap=100.0,
        )

        self.assertTrue(torch.isfinite(loss).item())

    def test_shape_mismatches_raise_value_error(self):
        module = load_training_module()
        model = self.zero_model(module)

        cases = [
            ("feature rows", {"features": [[1.0, 2.0, 3.0, 4.0]]}),
            ("lambda_rule", {"lambda_rule": [0.8]}),
            ("features must be 2D", {"features": [1.0, 2.0, 3.0, 4.0]}),
        ]
        for message, overrides in cases:
            with self.subTest(message=message):
                example = self.base_example()
                example.update(overrides)
                with self.assertRaisesRegex(ValueError, message):
                    module.predictive_loss_for_example(model, example)

    def test_zero_column_features_raise_value_error(self):
        module = load_training_module()
        model = self.zero_model(module)
        example = self.base_example()
        example["features"] = [[], []]

        with self.assertRaisesRegex(ValueError, "features must have at least one column"):
            module.predictive_loss_for_example(model, example)

    def test_train_model_rejects_empty_examples(self):
        module = load_training_module()

        with self.assertRaisesRegex(ValueError, "examples must not be empty"):
            module.train_model([], epochs=1, learning_rate=0.01, hidden_dim=6)

    def test_train_model_rejects_non_positive_epochs(self):
        module = load_training_module()

        for epochs in (0, -1):
            with self.subTest(epochs=epochs):
                with self.assertRaisesRegex(ValueError, "epochs"):
                    module.train_model([self.base_example()], epochs=epochs, learning_rate=0.01, hidden_dim=6)

    def test_main_creates_output_parent_directories(self):
        module = load_training_module()
        example = self.base_example()
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            examples_path = tmp_path / "examples.jsonl"
            output_path = tmp_path / "nested" / "reports" / "summary.json"
            examples_path.write_text(json.dumps(example) + "\n", encoding="utf-8")

            module.main(
                [
                    "--examples-jsonl",
                    str(examples_path),
                    "--output-json",
                    str(output_path),
                    "--epochs",
                    "1",
                    "--hidden-dim",
                    "6",
                ]
            )

            self.assertTrue(output_path.exists())
            summary = json.loads(output_path.read_text(encoding="utf-8"))
            self.assertEqual(summary["epochs"], 1)

    def pipeline_result(self):
        return {
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
            "reranked_top_matches": [
                {
                    "candidate_nct_id": "NCTHIST",
                    "overall_similarity_score": 82.0,
                    "retrieval_score": 98.0,
                    "suggested_borrowing_discount": 0.75,
                    "dimension_scores": {
                        "disease_biology_match": 5.0,
                        "treatment_regimen_match": 4.0,
                        "endpoint_estimand_match": 5.0,
                        "outcome_assessment_followup": 2.0,
                        "eligibility_criteria_overlap": 1.0,
                        "result_usability": 5.0,
                    },
                    "red_flags": [],
                    "borrowable_quantities": [
                        {
                            "endpoint": "Objective Response Rate",
                            "endpoint_family": "ORR/CR/PR",
                            "arm_results": [
                                {"arm": "Experimental", "count": 20, "denominator": 50}
                            ],
                        }
                    ],
                }
            ],
        }

    def test_build_training_example_from_pipeline_result(self) -> None:
        module = load_training_module()
        result = self.pipeline_result()

        example = module.build_training_example_from_pipeline_result(result, endpoint_key="ORR")

        self.assertEqual(example["query"], {"count": 12, "denominator": 40})
        self.assertEqual(len(example["features"]), 1)
        self.assertEqual(len(example["components"]), 1)
        self.assertGreater(example["features"][0][0], 0.0)

    def test_main_builds_examples_from_pipeline_results_jsonl(self):
        module = load_training_module()
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            pipeline_path = tmp_path / "pipeline.jsonl"
            output_path = tmp_path / "nested" / "reports" / "summary.json"
            pipeline_path.write_text(
                json.dumps(self.pipeline_result()) + "\n",
                encoding="utf-8",
            )

            module.main(
                [
                    "--pipeline-results-jsonl",
                    str(pipeline_path),
                    "--output-json",
                    str(output_path),
                    "--epochs",
                    "1",
                    "--hidden-dim",
                    "6",
                ]
            )

            self.assertTrue(output_path.exists())
            summary = json.loads(output_path.read_text(encoding="utf-8"))
            self.assertEqual(summary["epochs"], 1)
            self.assertEqual(summary["input_dim"], 5)


if __name__ == "__main__":
    unittest.main()
