import importlib.util
import math
from pathlib import Path
import unittest

import torch


SCRIPT_PATH = Path(__file__).resolve().parents[1] / "scripts" / "train_retrospective_lambda_model.py"


def load_training_module():
    spec = importlib.util.spec_from_file_location("train_retrospective_lambda_model", SCRIPT_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class RetrospectiveLambdaTrainingTest(unittest.TestCase):
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
        example = {
            "query": {"count": 2, "denominator": 4},
            "lambda_0": 0.2,
            "features": [
                [0.9, 0.8, 1.0, 3.9],
                [0.1, 0.2, 0.5, 2.0],
            ],
            "components": [
                {"alpha": 4.0, "beta": 2.0, "gate": 1.0},
                {"alpha": 1.5, "beta": 3.5, "gate": 0.5},
            ],
            "lambda_rule": [0.6, 0.2],
        }

        loss = module.predictive_loss_for_example(model, example, rho=0.1, ess_cap=100.0)

        self.assertEqual(loss.shape, ())
        self.assertTrue(torch.isfinite(loss).item())
        self.assertGreater(loss.item(), 0.0)
        self.assertTrue(math.isfinite(loss.item()))


if __name__ == "__main__":
    unittest.main()
