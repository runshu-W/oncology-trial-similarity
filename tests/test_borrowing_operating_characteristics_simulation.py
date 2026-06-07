import importlib.util
from pathlib import Path
import unittest


SCRIPT_PATH = (
    Path(__file__).resolve().parents[1]
    / "scripts"
    / "run_borrowing_operating_characteristics_simulation.py"
)


def load_simulation_module():
    spec = importlib.util.spec_from_file_location("run_borrowing_operating_characteristics_simulation", SCRIPT_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class BorrowingOperatingCharacteristicsSimulationTests(unittest.TestCase):
    def example(self):
        return {
            "query_nct_id": "NCTQUERY",
            "query": {"count": 3, "denominator": 12},
            "lambda_0": 0.2,
            "components": [
                {
                    "candidate_nct_id": "NCTHIST1",
                    "count": 4,
                    "denominator": 10,
                    "alpha": 1.0 + 0.5 * 4,
                    "beta": 1.0 + 0.5 * 6,
                    "lambda_rule": 0.5,
                    "lambda_model": 0.55,
                    "discount_model": 0.5,
                },
                {
                    "candidate_nct_id": "NCTHIST2",
                    "count": 1,
                    "denominator": 8,
                    "alpha": 1.0 + 0.25 * 1,
                    "beta": 1.0 + 0.25 * 7,
                    "lambda_rule": 0.3,
                    "lambda_model": 0.25,
                    "discount_model": 0.25,
                },
            ],
            "lambda_rule": [0.5, 0.3],
        }

    def test_default_scenarios_include_conflict_and_mixture_settings(self):
        module = load_simulation_module()

        names = [scenario.name for scenario in module.default_scenarios()]

        self.assertIn("exchangeable", names)
        self.assertIn("mild_optimistic_conflict", names)
        self.assertIn("strong_optimistic_conflict", names)
        self.assertIn("mixture_historical_conflict", names)

    def test_simulation_outputs_stable_operating_characteristic_columns(self):
        module = load_simulation_module()
        scenarios = [module.default_scenarios()[0]]

        rows = module.simulate_operating_characteristics(
            examples=[self.example()],
            scenarios=scenarios,
            methods=("weak_only", "rule", "model", "rule_sam"),
            iterations=10,
            seed=7,
            null_rate=0.2,
            alternative_rate=0.35,
            decision_threshold=0.3,
            posterior_probability_cutoff=0.8,
        )

        self.assertEqual({row["method"] for row in rows}, {"weak_only", "rule", "model", "rule_sam"})
        for row in rows:
            self.assertEqual(row["scenario"], "exchangeable")
            self.assertIn("type_i_error", row)
            self.assertIn("power", row)
            self.assertIn("bias", row)
            self.assertIn("mse", row)
            self.assertIn("coverage", row)
            self.assertIn("mean_interval_width", row)
            self.assertIn("sam_trigger_rate", row)
            self.assertIn("mean_historical_mass", row)
            self.assertEqual(row["iterations"], 10)

    def test_binomial_sampler_returns_count_within_denominator(self):
        module = load_simulation_module()
        rng = module.random.Random(11)

        count = module._binomial(rng, 1000, 0.25)

        self.assertGreaterEqual(count, 0)
        self.assertLessEqual(count, 1000)

    def test_select_template_examples_is_seeded_and_size_limited(self):
        module = load_simulation_module()
        examples = [{"query_nct_id": f"NCT{i:04d}"} for i in range(10)]

        first = module.select_template_examples(examples, max_examples=4, seed=7)
        second = module.select_template_examples(examples, max_examples=4, seed=7)

        self.assertEqual(first, second)
        self.assertEqual(len(first), 4)
        self.assertNotEqual(first, examples[:4])


if __name__ == "__main__":
    unittest.main()
