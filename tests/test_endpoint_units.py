"""Regression tests for unit-aware endpoint conversion (KNOWN_ISSUE_endpoint_units).

ClinicalTrials.gov reports an outcome value together with a unit of measure.
For ORR the unit is sometimes ``Participants`` (the value is a responder
count) and sometimes ``Percentage of Participants`` (the value is already a
percentage). These tests pin the corrected behaviour at every conversion
site: ``extract_outcomes`` proportions, arm-level observation selection,
historical endpoint observations, and mixture-prior component construction.
"""

from __future__ import annotations

import unittest

from pipeline import mixture_prior
from pipeline import oncology_trial_similarity_pipeline as pipeline
from pipeline.fix_endpoint_units import UnitError, rate_from_row, responders_from_row


def _results_posted(unit: str, value: float, denominator: float) -> dict:
    return {
        "5. Outcome measures": [
            {
                "Type": "Primary",
                "Title": "Objective Response Rate",
                "Description": "ORR",
                "Time Frame": "24 months",
                "Unit of Measure": unit,
                "Data Table": [
                    {"Category": "Measurement", "Arm A": str(value)},
                    {"Category": "Denominator: participants analysed", "Arm A": str(denominator)},
                ],
            }
        ]
    }


class RateFromRowTests(unittest.TestCase):
    def test_participant_units_divide_by_denominator(self) -> None:
        self.assertAlmostEqual(rate_from_row("Participants", 8, 31), 8 / 31)

    def test_percentage_units_divide_by_100(self) -> None:
        self.assertAlmostEqual(rate_from_row("Percentage of Participants", 26.0, 31), 0.26)

    def test_proportion_units_pass_through(self) -> None:
        self.assertAlmostEqual(rate_from_row("Proportion of participants", 0.26, 31), 0.26)

    def test_missing_unit_raises(self) -> None:
        with self.assertRaises(UnitError):
            rate_from_row(None, 26.0, 31)

    def test_impossible_count_raises(self) -> None:
        with self.assertRaises(UnitError):
            rate_from_row("Participants", 45, 31)

    def test_percentage_reconstruction_rounds_to_nearest_responder(self) -> None:
        responders, n, reconstructed = responders_from_row("Percentage of Participants", 26.0, 31)
        self.assertEqual((responders, n), (8, 31))
        self.assertTrue(reconstructed)

    def test_count_units_are_not_flagged_as_reconstructed(self) -> None:
        responders, n, reconstructed = responders_from_row("Participants", 8, 31)
        self.assertEqual((responders, n), (8, 31))
        self.assertFalse(reconstructed)


class ExtractOutcomesUnitTests(unittest.TestCase):
    def test_participant_unit_proportion_divides_by_denominator(self) -> None:
        outcomes = pipeline.extract_outcomes(_results_posted("Participants", 8, 31))
        row = outcomes[0]["arm_results"][0]
        self.assertAlmostEqual(row["proportion"], round(8 / 31, 6))

    def test_percentage_unit_proportion_divides_by_100(self) -> None:
        # The historical defect: 26.0 under "Percentage of Participants" with
        # denominator 31 was converted to 26/31 = 0.839 instead of 0.26.
        outcomes = pipeline.extract_outcomes(
            _results_posted("Percentage of Participants", 26.0, 31)
        )
        row = outcomes[0]["arm_results"][0]
        self.assertAlmostEqual(row["proportion"], 0.26)

    def test_unrecognised_unit_leaves_proportion_unset(self) -> None:
        outcomes = pipeline.extract_outcomes(_results_posted("Months", 5.2, 31))
        row = outcomes[0]["arm_results"][0]
        self.assertNotIn("proportion", row)

    def test_impossible_participant_count_leaves_proportion_unset(self) -> None:
        outcomes = pipeline.extract_outcomes(_results_posted("Participants", 45, 31))
        row = outcomes[0]["arm_results"][0]
        self.assertNotIn("proportion", row)


class ObservationConversionTests(unittest.TestCase):
    def test_percentage_row_reconstructs_responders(self) -> None:
        observation = pipeline.endpoint_observation_from_row(
            {"arm": "Experimental", "count": 26.0, "denominator": 31},
            unit="Percentage of Participants",
        )
        self.assertIsNotNone(observation)
        self.assertEqual(observation["count"], 8.0)
        self.assertEqual(observation["denominator"], 31.0)
        self.assertAlmostEqual(observation["rate"], 8.0 / 31.0)
        self.assertTrue(observation.get("count_reconstructed_from_unit"))

    def test_count_row_passes_through(self) -> None:
        observation = pipeline.endpoint_observation_from_row(
            {"arm": "Experimental", "count": 8, "denominator": 31},
            unit="Participants",
        )
        self.assertEqual(observation["count"], 8.0)
        self.assertNotIn("count_reconstructed_from_unit", observation)

    def test_missing_unit_drops_row(self) -> None:
        observation = pipeline.endpoint_observation_from_row(
            {"arm": "Experimental", "count": 8, "denominator": 31}
        )
        self.assertIsNone(observation)

    def test_query_endpoint_observations_use_endpoint_unit(self) -> None:
        query_summary = {
            "endpoints": {
                "primary": [
                    {
                        "title": "Objective Response Rate",
                        "endpoint_family": "ORR/CR/PR",
                        "unit": "Percentage of Participants",
                        "arm_results": [
                            {"arm": "Experimental", "count": 26.0, "denominator": 31}
                        ],
                    }
                ]
            }
        }
        observations = pipeline.query_endpoint_observations(query_summary)
        self.assertIn("ORR", observations)
        self.assertEqual(observations["ORR"]["treatment_count"], 8.0)
        self.assertEqual(observations["ORR"]["treatment_denominator"], 31.0)
        self.assertAlmostEqual(observations["ORR"]["treatment_rate"], round(8 / 31, 6))

    def test_historical_endpoint_observations_use_quantity_unit(self) -> None:
        rows = [
            {
                "candidate_nct_id": "NCTHIST",
                "suggested_borrowing_discount": 0.4,
                "prior_borrowing_suitability": "medium",
                "borrowable_quantities": [
                    {
                        "endpoint": "Objective Response Rate",
                        "endpoint_family": "ORR/CR/PR",
                        "unit": "Percentage of Participants",
                        "arm_results": [
                            {"arm": "Experimental", "count": 26.0, "denominator": 31}
                        ],
                    }
                ],
            }
        ]
        observations = pipeline.historical_endpoint_observations(rows, "ORR")
        self.assertEqual(len(observations), 1)
        self.assertEqual(observations[0]["count"], 8.0)
        self.assertEqual(observations[0]["denominator"], 31.0)
        self.assertAlmostEqual(observations[0]["rate"], round(8 / 31, 6))

    def test_mixture_components_use_quantity_unit(self) -> None:
        rows = [
            {
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
                        "unit": "Percentage of Participants",
                        "arm_results": [
                            {"arm": "Experimental", "count": 26.0, "denominator": 31}
                        ],
                    }
                ],
            }
        ]
        components = mixture_prior.components_from_reranked_rows(rows, endpoint_key="ORR")
        self.assertEqual(len(components["components"]), 1)
        component = components["components"][0]
        self.assertEqual(component["count"], 8.0)
        self.assertEqual(component["denominator"], 31.0)
        # alpha = 1 + a*y, beta = 1 + a*(n - y) with the reconstructed responders.
        self.assertAlmostEqual(
            component["beta"] - 1.0,
            (component["denominator"] - component["count"])
            * ((component["alpha"] - 1.0) / component["count"]),
        )


if __name__ == "__main__":
    unittest.main()
