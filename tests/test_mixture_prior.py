import unittest

from pipeline import mixture_prior


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
        normalized = mixture_prior.normalize_lambdas([2, 1, 0], lambda0=0.2)

        self.assertEqual(set(normalized), {"lambda_0", "lambda_i"})
        self.assertAlmostEqual(normalized["lambda_0"], 0.2)
        lambdas = normalized["lambda_i"]
        self.assertAlmostEqual(sum(lambdas), 0.8)
        self.assertGreater(lambdas[0], lambdas[1])
        self.assertEqual(lambdas[2], 0)

    def test_normalize_lambdas_uses_weak_only_when_all_weights_zero(self):
        normalized = mixture_prior.normalize_lambdas([0, 0], lambda0=0.2)

        self.assertEqual(
            normalized,
            {"lambda_0": 1.0, "lambda_i": [0.0, 0.0]},
        )

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

    def test_posterior_component_weights_returns_named_weights_summing_to_one(self):
        posterior = mixture_prior.posterior_component_weights(
            y=1,
            n=3,
            lambda0=0.5,
            weak_alpha=1,
            weak_beta=1,
            components=[
                {"lambda": 0.3, "alpha": 2, "beta": 3},
                {"lambda": 0.2, "alpha": 4, "beta": 1},
            ],
        )

        self.assertEqual(set(posterior), {"lambda_0_post", "lambda_i_post"})
        self.assertAlmostEqual(
            posterior["lambda_0_post"] + sum(posterior["lambda_i_post"]),
            1.0,
        )

    def test_posterior_component_weights_zero_normalizer_falls_back_to_weak_only(self):
        posterior = mixture_prior.posterior_component_weights(
            y=1,
            n=3,
            lambda0=0.0,
            weak_alpha=1,
            weak_beta=1,
            components=[
                {"lambda": 0.0, "alpha": 2, "beta": 3},
                {"lambda": 0.0, "alpha": 4, "beta": 1},
            ],
        )

        self.assertEqual(
            posterior,
            {"lambda_0_post": 1.0, "lambda_i_post": [0.0, 0.0]},
        )

    def test_invalid_lambda0_values_raise(self):
        for lambda0 in (float("nan"), -0.1, 1.1):
            with self.subTest(lambda0=lambda0):
                with self.assertRaises(ValueError):
                    mixture_prior.normalize_lambdas([1], lambda0=lambda0)

    def test_infinite_raw_weight_raises(self):
        with self.assertRaises(ValueError):
            mixture_prior.normalize_lambdas([float("inf")], lambda0=0.2)

    def test_non_integer_observation_raises(self):
        with self.assertRaises(ValueError):
            mixture_prior.beta_binomial_predictive_probability(1.5, 3, 1, 1)

        with self.assertRaises(ValueError):
            mixture_prior.beta_binomial_predictive_probability(True, 3, 1, 1)

    def test_invalid_beta_parameters_raise(self):
        with self.assertRaises(ValueError):
            mixture_prior.beta_binomial_predictive_probability(1, 3, float("nan"), 1)

    def test_invalid_component_lambda_raises(self):
        with self.assertRaises(ValueError):
            mixture_prior.mixture_predictive_probability(
                y=1,
                n=3,
                lambda0=0.5,
                weak_alpha=1,
                weak_beta=1,
                components=[{"lambda": -0.1, "alpha": 2, "beta": 3}],
            )


class MixturePriorComponentTests(unittest.TestCase):
    def _component_row(self, **overrides):
        row = {
            "candidate_nct_id": "NCTHIST",
            "overall_similarity_score": 80.0,
            "suggested_borrowing_discount": 0.5,
            "dimension_scores": {
                "disease_population_match": 5.0,
                "endpoint_estimand_match": 5.0,
                "result_usability": 5.0,
            },
            "red_flags": [],
            "borrowable_quantities": [
                {
                    "endpoint": "Objective Response Rate",
                    "endpoint_family": "ORR/CR/PR",
                    "arm_results": [{"arm": "Experimental", "count": 10, "denominator": 40}],
                }
            ],
        }
        row.update(overrides)
        return row

    def test_components_from_reranked_rows_use_stage2_fields(self) -> None:
        rows = [
            {
                "candidate_nct_id": "NCTHIST1",
                "overall_similarity_score": 80.0,
                "suggested_borrowing_discount": 0.75,
                "dimension_scores": {
                    "disease_biology_match": 5.0,
                    "endpoint_estimand_match": 5.0,
                    "result_usability": 5.0,
                },
                "red_flags": [],
                "borrowable_quantities": [
                    {
                        "endpoint": "Objective Response Rate",
                        "endpoint_family": "ORR/CR/PR",
                        "arm_results": [{"arm": "Experimental", "count": 20, "denominator": 50}],
                    }
                ],
            },
            {
                "candidate_nct_id": "NCTHIST2",
                "overall_similarity_score": 65.0,
                "suggested_borrowing_discount": 0.4,
                "dimension_scores": {
                    "disease_biology_match": 1.0,
                    "endpoint_estimand_match": 5.0,
                    "result_usability": 5.0,
                },
                "red_flags": ["Low disease biology match."],
                "borrowable_quantities": [
                    {
                        "endpoint": "Objective Response Rate",
                        "endpoint_family": "ORR/CR/PR",
                        "arm_results": [{"arm": "Experimental", "count": 8, "denominator": 40}],
                    }
                ],
            },
        ]

        components = mixture_prior.components_from_reranked_rows(rows, endpoint_key="ORR", lambda0=0.2)

        self.assertEqual(components["mode"], "rule")
        self.assertEqual(len(components["components"]), 2)
        self.assertAlmostEqual(components["lambda_0"], 0.2)
        self.assertAlmostEqual(sum(row["lambda_rule"] for row in components["components"]), 0.8)
        self.assertGreater(components["components"][0]["lambda_rule"], components["components"][1]["lambda_rule"])
        self.assertEqual(components["components"][0]["alpha"], 16.0)
        self.assertEqual(components["components"][0]["beta"], 23.5)

    def test_components_include_planned_downstream_schema_fields(self) -> None:
        row = self._component_row(
            candidate_nct_id=None,
            nct_id="NCTFALLBACK",
            overall_similarity_score=77.0,
        )

        components = mixture_prior.components_from_reranked_rows([row], endpoint_key="ORR")
        component = components["components"][0]

        expected_fields = {
            "nct_id",
            "endpoint",
            "count",
            "denominator",
            "rate",
            "discount",
            "gate",
            "overall_similarity_score",
            "alpha",
            "beta",
            "raw_rule_weight",
            "lambda_rule",
        }
        self.assertTrue(expected_fields.issubset(component))
        self.assertEqual(component["nct_id"], "NCTFALLBACK")
        self.assertEqual(component["endpoint"], "Objective Response Rate")
        self.assertEqual(component["overall_similarity_score"], 77.0)
        self.assertIn("raw_rule_weight", component)

    def test_conservative_gate_prefers_disease_population_match(self) -> None:
        gate = mixture_prior.conservative_gate(
            {
                "disease_population_match": 5.0,
                "disease_biology_match": 1.0,
                "endpoint_estimand_match": 5.0,
                "result_usability": 5.0,
            },
            [],
        )

        self.assertEqual(gate, 1.0)

    def test_selected_treatment_observation_skips_standard_of_care(self) -> None:
        observation = mixture_prior._selected_treatment_observation(
            [
                {"arm": "Standard of Care", "count": 1, "denominator": 20},
                {"arm": "Experimental", "count": 12, "denominator": 30},
            ]
        )

        self.assertEqual(observation, (12.0, 30.0, 0.4))

    def test_dor_quantity_is_ignored_for_orr_endpoint(self) -> None:
        row = self._component_row(
            borrowable_quantities=[
                {
                    "endpoint": "Duration of Response",
                    "endpoint_family": "DOR",
                    "arm_results": [{"arm": "Experimental", "count": 10, "denominator": 40}],
                }
            ]
        )

        components = mixture_prior.components_from_reranked_rows([row], endpoint_key="ORR")

        self.assertEqual(components, {"mode": "rule", "lambda_0": 1.0, "components": []})

    def test_pfs6_canonicalization_matches_progression_six_month_endpoint(self) -> None:
        row = self._component_row(
            borrowable_quantities=[
                {
                    "endpoint": "Progression-free survival",
                    "endpoint_family": "PFS",
                    "time_frame": "6 months",
                    "arm_results": [{"arm": "Experimental", "count": 18, "denominator": 50}],
                }
            ]
        )

        components = mixture_prior.components_from_reranked_rows([row], endpoint_key="PFS6")

        self.assertEqual(len(components["components"]), 1)
        self.assertEqual(components["components"][0]["endpoint"], "Progression-free survival")

    def test_apply_model_lambdas_sets_active_weights_and_preserves_rule_weights(self):
        mixture = {
            "mode": "rule",
            "lambda_0": 0.2,
            "components": [
                {"lambda_rule": 0.6, "nct_id": "NCT1"},
                {"lambda_rule": 0.2, "nct_id": "NCT2"},
            ],
        }

        updated = mixture_prior.apply_model_lambdas(mixture, [0.5, 0.3])

        self.assertEqual(updated["mode"], "retrospective_calibrated")
        self.assertAlmostEqual(
            sum(component["lambda_active"] for component in updated["components"]) + updated["lambda_0"],
            1.0,
        )
        self.assertEqual(updated["components"][0]["lambda_rule"], 0.6)
        self.assertEqual(updated["components"][1]["lambda_rule"], 0.2)
        self.assertIn("lambda_model", updated["components"][0])
        self.assertIn("lambda_active", updated["components"][0])
        self.assertIn("retrospective predictive loss", updated["calibration_note"])
        self.assertIn("No expert labels", updated["calibration_note"])

    def test_apply_model_lambdas_can_apply_two_head_model_discounts(self):
        mixture = {
            "mode": "rule",
            "lambda_0": 0.2,
            "components": [
                {
                    "lambda_rule": 0.6,
                    "nct_id": "NCT1",
                    "count": 20.0,
                    "denominator": 50.0,
                    "discount": 0.25,
                    "alpha": 6.0,
                    "beta": 8.5,
                },
                {
                    "lambda_rule": 0.2,
                    "nct_id": "NCT2",
                    "count": 4.0,
                    "denominator": 10.0,
                    "discount": 0.5,
                    "alpha": 3.0,
                    "beta": 4.0,
                },
            ],
        }

        updated = mixture_prior.apply_model_lambdas(
            mixture,
            [0.5, 0.3],
            model_discounts=[0.4, 0.1],
            model_type="two_head_deepsets",
        )

        first = updated["components"][0]
        self.assertEqual(updated["lambda_model_type"], "two_head_deepsets")
        self.assertIn("discount_calibration_note", updated)
        self.assertAlmostEqual(first["discount_rule"], 0.25)
        self.assertAlmostEqual(first["discount_model"], 0.4)
        self.assertAlmostEqual(first["discount_active"], 0.4)
        self.assertAlmostEqual(first["alpha_rule"], 6.0)
        self.assertAlmostEqual(first["beta_rule"], 8.5)
        self.assertAlmostEqual(first["alpha"], 1.0 + 0.4 * 20.0)
        self.assertAlmostEqual(first["beta"], 1.0 + 0.4 * 30.0)
        self.assertIn("lambda_active", first)

    def test_apply_model_lambdas_rejects_length_mismatch(self):
        mixture = {
            "lambda_0": 0.2,
            "components": [
                {"lambda_rule": 0.6},
                {"lambda_rule": 0.2},
            ],
        }

        with self.assertRaisesRegex(ValueError, "model lambda count"):
            mixture_prior.apply_model_lambdas(mixture, [0.5])

        with self.assertRaisesRegex(ValueError, "model discount count"):
            mixture_prior.apply_model_lambdas(mixture, [0.5, 0.3], model_discounts=[0.1])

    def test_sam_conflict_adapter_preserves_candidate_mass_when_historical_predicts_well(self):
        mixture = {
            "mode": "rule",
            "lambda_0": 0.2,
            "components": [
                {"lambda_active": 0.8, "alpha": 21.0, "beta": 31.0},
            ],
        }

        updated = mixture_prior.apply_sam_conflict_adapter(mixture, y=4, n=10)

        self.assertEqual(updated["sam_prior_data_conflict"]["status"], "no_conflict")
        self.assertAlmostEqual(updated["sam_prior_data_conflict"]["borrowing_multiplier"], 1.0)
        self.assertAlmostEqual(updated["lambda_0"], 0.2)
        self.assertAlmostEqual(updated["components"][0]["lambda_active"], 0.8)
        self.assertAlmostEqual(updated["components"][0]["lambda_pre_sam"], 0.8)

    def test_sam_conflict_adapter_downweights_candidate_mass_when_historical_conflicts(self):
        mixture = {
            "mode": "rule",
            "lambda_0": 0.2,
            "components": [
                {"lambda_active": 0.8, "alpha": 46.0, "beta": 6.0},
            ],
        }

        updated = mixture_prior.apply_sam_conflict_adapter(mixture, y=0, n=10)

        self.assertEqual(updated["sam_prior_data_conflict"]["status"], "conflict_downweighted")
        self.assertLess(updated["sam_prior_data_conflict"]["borrowing_multiplier"], 1.0)
        self.assertGreater(updated["lambda_0"], 0.2)
        self.assertLess(updated["components"][0]["lambda_active"], 0.8)
        self.assertAlmostEqual(
            updated["lambda_0"] + sum(component["lambda_active"] for component in updated["components"]),
            1.0,
        )


if __name__ == "__main__":
    unittest.main()
