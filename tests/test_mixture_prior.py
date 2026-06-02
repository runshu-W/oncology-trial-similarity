import unittest

from docs import mixture_prior


class MixturePriorTest(unittest.TestCase):
    def test_beta_binomial_predictive_probability_matches_uniform_prior(self):
        probability = mixture_prior.beta_binomial_predictive_probability(
            y=2,
            n=4,
            alpha=1,
            beta=1,
        )

        self.assertAlmostEqual(probability, 0.2)

    def test_normalize_lambdas_reserves_weak_weight_and_preserves_order(self):
        lambda0, lambdas = mixture_prior.normalize_lambdas([2, 1, 0], lambda0=0.2)

        self.assertAlmostEqual(lambda0, 0.2)
        self.assertAlmostEqual(sum(lambdas), 0.8)
        self.assertGreater(lambdas[0], lambdas[1])
        self.assertEqual(lambdas[2], 0)

    def test_normalize_lambdas_uses_weak_only_when_all_weights_zero(self):
        lambda0, lambdas = mixture_prior.normalize_lambdas([0, 0], lambda0=0.2)

        self.assertEqual(lambda0, 1.0)
        self.assertEqual(lambdas, [0.0, 0.0])

    def test_mixture_predictive_probability_is_weighted_sum(self):
        components = [
            {"lambda": 0.3, "alpha": 2, "beta": 3},
            {"lambda": 0.2, "alpha": 4, "beta": 1},
        ]
        weak = mixture_prior.beta_binomial_predictive_probability(1, 3, 1, 1)
        candidate1 = mixture_prior.beta_binomial_predictive_probability(1, 3, 2, 3)
        candidate2 = mixture_prior.beta_binomial_predictive_probability(1, 3, 4, 1)

        probability = mixture_prior.mixture_predictive_probability(
            y=1,
            n=3,
            lambda0=0.5,
            weak_alpha=1,
            weak_beta=1,
            components=components,
        )

        self.assertAlmostEqual(probability, 0.5 * weak + 0.3 * candidate1 + 0.2 * candidate2)


if __name__ == "__main__":
    unittest.main()
