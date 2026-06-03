import importlib.util
import json
import math
from pathlib import Path
import tempfile
import unittest

import torch


SCRIPT_PATH = Path(__file__).resolve().parents[1] / "scripts" / "train_retrospective_lambda_model.py"

EXPECTED_FEATURE_NAMES = [
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


def load_training_module():
    spec = importlib.util.spec_from_file_location("train_retrospective_lambda_model", SCRIPT_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class RetrospectiveLambdaTrainingTests(unittest.TestCase):
    def zero_model(self, module):
        model = module.LambdaScorer(input_dim=len(EXPECTED_FEATURE_NAMES), hidden_dim=6)
        with torch.no_grad():
            for parameter in model.parameters():
                parameter.zero_()
        return model

    def base_example(self):
        return {
            "query": {"count": 2, "denominator": 4},
            "lambda_0": 0.2,
            "feature_names": EXPECTED_FEATURE_NAMES,
            "features": [
                [0.9, 0.8, 1.0, 0.7, 0.6, 0.5, 1.0, 0.0, 3.9],
                [0.1, 0.2, 0.5, 0.3, 0.4, 0.2, 0.5, -0.25, 2.0],
            ],
            "components": [
                {"alpha": 4.0, "beta": 2.0, "gate": 1.0, "discount": 0.5, "denominator": 10.0},
                {"alpha": 1.5, "beta": 3.5, "gate": 1.0, "discount": 0.25, "denominator": 20.0},
            ],
            "lambda_rule": [0.6, 0.2],
        }

    def test_lambda_scorer_outputs_one_score_per_candidate(self):
        module = load_training_module()
        model = module.LambdaScorer(input_dim=len(EXPECTED_FEATURE_NAMES), hidden_dim=6)
        features = torch.tensor(
            [
                [0.9, 0.8, 1.0, 0.7, 0.6, 0.5, 1.0, 0.0, 3.9],
                [0.1, 0.2, 0.5, 0.3, 0.4, 0.2, 0.5, -0.25, 2.0],
                [0.4, 0.3, 0.7, 0.2, 0.5, 0.1, 0.8, -0.1, 3.0],
            ],
            dtype=torch.float32,
        )

        scores = model(features)

        self.assertEqual(scores.shape, (3,))

    def test_full_feature_names_match_design_vector(self):
        module = load_training_module()

        self.assertEqual(module.LAMBDA_FEATURE_NAMES, EXPECTED_FEATURE_NAMES)

    def test_redflag_severity_is_normalized(self):
        module = load_training_module()

        severity = module.redflag_severity(
            [
                "Low disease/population match.",
                "Low endpoint/estimand match.",
                "No primary endpoint-family overlap.",
                "No normalized regimen-backbone overlap.",
                "Candidate has no posted results in indexed JSON.",
                "No arm-level count/denominator pair found for primary borrowable quantities.",
                "Treatment line mismatch: query=1L candidate=2L.",
            ]
        )

        self.assertEqual(severity, 1.0)
        self.assertAlmostEqual(module.redflag_severity(["Treatment line mismatch."]), 0.25 / 3.0)

    def test_predictive_loss_for_example_returns_finite_positive_loss(self):
        module = load_training_module()
        model = module.LambdaScorer(input_dim=len(EXPECTED_FEATURE_NAMES), hidden_dim=6)
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
            ("feature rows", {"features": [[1.0] * len(EXPECTED_FEATURE_NAMES)]}),
            ("lambda_rule", {"lambda_rule": [0.8]}),
            ("features must be 2D", {"features": [1.0] * len(EXPECTED_FEATURE_NAMES)}),
            ("features must have 9 columns", {"features": [[1.0, 2.0, 3.0, 4.0], [1.0, 2.0, 3.0, 4.0]]}),
            ("feature_names must match", {"feature_names": EXPECTED_FEATURE_NAMES[:-1]}),
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

    def test_predictive_loss_rejects_invalid_lambda0(self):
        module = load_training_module()
        model = self.zero_model(module)

        cases = [
            (-0.1, "lambda_0 must be in \\[0, 1\\]"),
            (1.1, "lambda_0 must be in \\[0, 1\\]"),
            (float("nan"), "lambda_0 must be finite"),
        ]
        for lambda0, message in cases:
            with self.subTest(lambda0=lambda0):
                example = self.base_example()
                example["lambda_0"] = lambda0
                with self.assertRaisesRegex(ValueError, message):
                    module.predictive_loss_for_example(model, example)

    def test_predictive_loss_rejects_invalid_query_count_denominator(self):
        module = load_training_module()
        model = self.zero_model(module)

        cases = [
            ({"count": 2, "denominator": 0}, "query denominator must be greater than 0"),
            ({"count": 5, "denominator": 4}, "query count must be less than or equal to denominator"),
            ({"count": -1, "denominator": 4}, "query count must be non-negative"),
            ({"count": float("nan"), "denominator": 4}, "query count must be finite"),
            ({"count": float("inf"), "denominator": 4}, "query count must be finite"),
            ({"count": 2, "denominator": float("nan")}, "query denominator must be finite"),
            ({"count": 2, "denominator": float("inf")}, "query denominator must be finite"),
        ]
        for query, message in cases:
            with self.subTest(query=query):
                example = self.base_example()
                example["query"] = query
                with self.assertRaisesRegex(ValueError, message):
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

    def test_train_model_can_save_and_load_artifact(self):
        module = load_training_module()
        example = self.base_example()
        example["features"] = [[0.1] * 9, [0.2] * 9]
        example["feature_names"] = module.LAMBDA_FEATURE_NAMES
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "lambda_model.pt"
            summary = module.train_model(
                [example],
                epochs=1,
                learning_rate=0.01,
                hidden_dim=6,
                model_output=path,
            )
            loaded = module.load_model_artifact(path)

            self.assertTrue(path.exists())

        self.assertEqual(summary["model_output"], str(path))
        self.assertEqual(loaded["feature_names"], module.LAMBDA_FEATURE_NAMES)
        self.assertEqual(loaded["input_dim"], 9)
        self.assertEqual(loaded["hidden_dim"], 6)

    def test_load_model_artifact_rejects_feature_metadata_mismatch(self):
        module = load_training_module()
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "lambda_model.pt"
            torch.save(
                {
                    "state_dict": {},
                    "input_dim": 9,
                    "hidden_dim": 6,
                    "feature_names": ["wrong"],
                    "lambda0": 0.2,
                },
                path,
            )

            with self.assertRaisesRegex(ValueError, "feature_names"):
                module.load_model_artifact(path)

            torch.save(
                {
                    "state_dict": {},
                    "input_dim": 8,
                    "hidden_dim": 6,
                    "feature_names": module.LAMBDA_FEATURE_NAMES,
                    "lambda0": 0.2,
                },
                path,
            )

            with self.assertRaisesRegex(ValueError, "input_dim"):
                module.load_model_artifact(path)

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

    def test_main_rejects_both_example_sources(self):
        module = load_training_module()
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            examples_path = tmp_path / "examples.jsonl"
            pipeline_path = tmp_path / "pipeline.jsonl"
            output_path = tmp_path / "summary.json"
            examples_path.write_text(json.dumps(self.base_example()) + "\n", encoding="utf-8")
            pipeline_path.write_text(json.dumps(self.pipeline_result()) + "\n", encoding="utf-8")

            with self.assertRaises(SystemExit):
                module.main(
                    [
                        "--examples-jsonl",
                        str(examples_path),
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

    def test_main_rejects_missing_example_source(self):
        module = load_training_module()
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "summary.json"

            with self.assertRaises(SystemExit):
                module.main(["--output-json", str(output_path)])

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
        self.assertEqual(example["feature_names"], EXPECTED_FEATURE_NAMES)
        self.assertEqual(len(example["features"]), 1)
        self.assertEqual(len(example["components"]), 1)
        self.assertEqual(len(example["features"][0]), 9)
        self.assertEqual(
            example["features"][0],
            [
                0.82,
                1.0,
                0.8,
                1.0,
                0.4,
                0.2,
                1.0,
                -0.0,
                math.log1p(50.0),
            ],
        )
        self.assertAlmostEqual(example["components"][0]["alpha"], 16.0)
        self.assertAlmostEqual(example["components"][0]["beta"], 23.5)
        self.assertAlmostEqual(example["components"][0]["denominator"], 50.0)
        self.assertAlmostEqual(example["components"][0]["discount"], 0.75)
        self.assertAlmostEqual(sum(example["lambda_rule"]) + example["lambda_0"], 1.0)

    def test_build_training_example_uses_negative_redflag_feature(self) -> None:
        module = load_training_module()
        result = self.pipeline_result()
        result["reranked_top_matches"][0]["red_flags"] = [
            "Low disease/population match.",
            "No normalized regimen-backbone overlap.",
        ]

        example = module.build_training_example_from_pipeline_result(result, endpoint_key="ORR")

        self.assertAlmostEqual(example["features"][0][7], -(1.8 / 3.0))

    def test_build_training_example_raises_for_missing_endpoint_key(self) -> None:
        module = load_training_module()
        result = self.pipeline_result()

        with self.assertRaisesRegex(ValueError, "endpoint DCR"):
            module.build_training_example_from_pipeline_result(result, endpoint_key="DCR")

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
            self.assertEqual(summary["input_dim"], 9)


if __name__ == "__main__":
    unittest.main()
