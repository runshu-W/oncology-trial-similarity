from __future__ import annotations

import json
import unittest

from fastapi.testclient import TestClient

from web_agent import app as app_module


class WebAgentTests(unittest.TestCase):
    def setUp(self) -> None:
        self.client = TestClient(app_module.app)
        self.original_runner = app_module.run_pipeline_search

    def tearDown(self) -> None:
        app_module.run_pipeline_search = self.original_runner

    def test_health_reports_index_status(self) -> None:
        response = self.client.get("/api/health")

        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertIn("index_dir", data)
        self.assertIn("index_available", data)
        self.assertTrue(data["default_rerank"])

    def test_search_rejects_invalid_json_text(self) -> None:
        response = self.client.post("/api/search", data={"json_text": "{not json"})

        self.assertEqual(response.status_code, 400)
        self.assertIn("Invalid JSON", response.json()["detail"])

    def test_search_accepts_pasted_json_and_returns_pipeline_output(self) -> None:
        def fake_runner(query_payload, index_dir, top_k, rerank_top_n):
            self.assertEqual(query_payload["Study details"]["1. NCT number"], "NCTTEST")
            self.assertEqual(top_k, 10)
            self.assertEqual(rerank_top_n, 100)
            return {
                "query": {
                    "nct_id": "NCTTEST",
                    "brief_title": "Test Trial",
                    "phase": "Phase 2",
                    "oncology": {"primary_site": ["Lung"]},
                    "intervention": {"experimental_regimen": "Drug: Test"},
                },
                "reranked_top10": [
                    {
                        "rank": 1,
                        "nct_id": "NCT00000001",
                        "brief_title": "Historical Trial",
                        "overall_similarity_score": 72.5,
                        "retrieval_score": 98.1,
                        "prior_borrowing_suitability": "medium",
                        "suggested_borrowing_discount": 0.4,
                        "red_flags": ["Low treatment-regimen match."],
                        "dimension_scores": {"disease_population_match": 3.5},
                        "borrowable_quantities": [],
                    }
                ],
            }

        app_module.run_pipeline_search = fake_runner
        payload = {"Study details": {"1. NCT number": "NCTTEST"}}

        response = self.client.post("/api/search", data={"json_text": json.dumps(payload)})

        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertEqual(data["result"]["query"]["nct_id"], "NCTTEST")
        self.assertIn("markdown_report", data)
        self.assertEqual(data["result"]["reranked_top10"][0]["nct_id"], "NCT00000001")

    def test_run_pipeline_search_passes_temp_query_file_to_pipeline(self) -> None:
        original_search = app_module.pipeline.search

        def fake_search(query_json, index_dir, top_k, rerank_top_n):
            self.assertTrue(query_json.exists())
            self.assertEqual(json.loads(query_json.read_text())["id"], "query")
            self.assertEqual(top_k, 3)
            self.assertEqual(rerank_top_n, 7)
            return {"query": {"nct_id": "NCTTEMP"}, "reranked_top10": []}

        app_module.pipeline.search = fake_search
        try:
            result = app_module.run_pipeline_search({"id": "query"}, app_module.DEFAULT_INDEX_DIR, 3, 7)
        finally:
            app_module.pipeline.search = original_search

        self.assertEqual(result["query"]["nct_id"], "NCTTEMP")

    def test_bayesian_analysis_summarizes_weighted_orr_borrowing(self) -> None:
        result = {
            "query_summary": {
                "nct_id": "NCTQUERY",
                "endpoints": {
                    "primary": [
                        {
                            "title": "Objective Response Rate",
                            "endpoint_family": "ORR/CR/PR",
                            "arm_results": [
                                {"arm": "Experimental Drug", "count": 12, "denominator": 40},
                                {"arm": "Placebo Comparator", "count": 6, "denominator": 40},
                            ],
                        }
                    ]
                },
            },
            "reranked_top_matches": [
                {
                    "candidate_nct_id": "NCTHIST1",
                    "overall_similarity_score": 82.0,
                    "suggested_borrowing_discount": 0.4,
                    "dimension_scores": {
                        "endpoint_estimand_match": 3.0,
                        "result_usability": 3.0,
                        "disease_population_match": 3.0,
                    },
                    "borrowable_quantities": [
                        {
                            "endpoint": "Objective Response Rate",
                            "endpoint_family": "ORR/CR/PR",
                            "arm_results": [
                                {"arm": "Experimental Arm", "count": 10, "denominator": 50},
                            ],
                        }
                    ],
                },
                {
                    "candidate_nct_id": "NCTHIST2",
                    "overall_similarity_score": 76.0,
                    "suggested_borrowing_discount": 0.75,
                    "dimension_scores": {
                        "endpoint_estimand_match": 3.0,
                        "result_usability": 3.0,
                        "disease_population_match": 3.0,
                    },
                    "borrowable_quantities": [
                        {
                            "endpoint": "Objective Response Rate",
                            "endpoint_family": "ORR/CR/PR",
                            "arm_results": [
                                {"arm": "Treatment Arm", "count": 18, "denominator": 60},
                            ],
                        }
                    ],
                },
            ],
        }

        enriched = app_module.pipeline.add_bayesian_analysis(result)
        analysis = enriched["bayesian_analysis"]

        self.assertEqual(analysis["status"], "available")
        endpoint = analysis["endpoint_analyses"][0]
        self.assertEqual(endpoint["endpoint_family"], "ORR")
        self.assertEqual(endpoint["analysis_mode"], "posterior")
        self.assertGreater(endpoint["effective_sample_size"], 0)
        mixture = endpoint["mixture_prior"]
        self.assertIn("lambda_0", mixture)
        self.assertIn("components", mixture)
        self.assertAlmostEqual(mixture["lambda_0"], 0.2)
        self.assertTrue(mixture["components"])
        first_component = mixture["components"][0]
        self.assertIn("nct_id", first_component)
        self.assertIn("lambda_rule", first_component)
        self.assertIn("alpha", first_component)
        self.assertIn("beta", first_component)
        self.assertAlmostEqual(
            mixture["lambda_0"] + sum(component["lambda_rule"] for component in mixture["components"]),
            1.0,
            places=6,
        )
        self.assertIn("observed_weights", {row["scenario"] for row in endpoint["weight_sensitivity"]})
        self.assertTrue(endpoint["success_probability_grid"])
        self.assertTrue(endpoint["tipping_points"])
        self.assertGreater(analysis["two_arm_decision_support"]["orr"]["posterior_or_mean"], 1.0)

    def test_bayesian_analysis_excludes_duration_of_response(self) -> None:
        result = {
            "query_summary": {
                "nct_id": "NCTQUERY",
                "endpoints": {
                    "primary": [
                        {
                            "title": "Duration of Response",
                            "endpoint_family": "DOR",
                            "arm_results": [{"arm": "Experimental", "count": 5, "denominator": 10}],
                        }
                    ]
                },
            },
            "reranked_top_matches": [
                {
                    "candidate_nct_id": "NCTHIST",
                    "suggested_borrowing_discount": 0.4,
                    "borrowable_quantities": [
                        {
                            "endpoint": "Duration of Response",
                            "endpoint_family": "DOR",
                            "arm_results": [{"arm": "Experimental", "count": 4, "denominator": 10}],
                        }
                    ],
                }
            ],
        }

        analysis = app_module.pipeline.add_bayesian_analysis(result)["bayesian_analysis"]

        self.assertEqual(analysis["status"], "not_available")
        self.assertEqual(analysis["endpoint_analyses"], [])

    def test_bayesian_analysis_supports_prior_only_when_query_has_no_results(self) -> None:
        result = {
            "query_summary": {
                "nct_id": "NCTQUERY",
                "endpoints": {
                    "primary": [
                        {
                            "title": "Objective Response Rate",
                            "endpoint_family": "ORR/CR/PR",
                            "arm_results": [],
                        }
                    ]
                },
            },
            "reranked_top_matches": [
                {
                    "candidate_nct_id": "NCTHIST",
                    "suggested_borrowing_discount": 0.75,
                    "borrowable_quantities": [
                        {
                            "endpoint": "Objective Response Rate",
                            "endpoint_family": "ORR/CR/PR",
                            "arm_results": [{"arm": "Experimental", "count": 15, "denominator": 50}],
                        }
                    ],
                }
            ],
        }

        analysis = app_module.pipeline.add_bayesian_analysis(result)["bayesian_analysis"]

        self.assertEqual(analysis["status"], "available")
        endpoint = analysis["endpoint_analyses"][0]
        self.assertEqual(endpoint["analysis_mode"], "prior_only")
        observed = next(row for row in endpoint["weight_sensitivity"] if row["scenario"] == "observed_weights")
        self.assertEqual(observed["posterior"], None)
        self.assertIsNotNone(observed["prior"])

    def test_treatment_arm_selection_skips_active_comparator(self) -> None:
        rows = [
            {"arm": "Active Comparator: Standard Therapy", "count": 4, "denominator": 20},
            {"arm": "Experimental: Drug X", "count": 10, "denominator": 20},
        ]

        selected, observation = app_module.pipeline.select_arm_observation(rows, "treatment")

        self.assertEqual(selected["arm"], "Experimental: Drug X")
        self.assertEqual(observation["count"], 10.0)

    def test_endpoint_score_requires_matched_endpoint_denominator(self) -> None:
        query = {
            "phase": "Phase 2",
            "cancer_type": {"histology": ["NSCLC"], "primary_site": ["Lung"], "molecular_marker": ["Not reported"], "line_of_therapy": "Not reported"},
            "intervention": {"backbone_regimen": ["Not normalized"], "drug_classes": []},
            "design": {"single_or_multi_arm": "Single-arm", "randomized": "No"},
            "endpoints": {"primary": [{"endpoint_family": "ORR/CR/PR"}]},
        }
        candidate = {
            "nct_id": "NCTHIST",
            "score_0_100": 50,
            "phase": "Phase 2",
            "cancer_type": {"histology": ["NSCLC"], "primary_site": ["Lung"], "molecular_marker": ["Not reported"], "line_of_therapy": "Not reported"},
            "intervention": {"backbone_regimen": ["Not normalized"], "drug_classes": []},
            "design": {"single_or_multi_arm": "Single-arm", "randomized": "No"},
            "result_usability": {"has_posted_results": True, "denominators_available": True},
            "borrowable_quantities": [
                {
                    "endpoint": "Overall Survival",
                    "endpoint_family": "OS",
                    "arm_results": [{"arm": "Experimental", "count": 10, "denominator": 20}],
                }
            ],
        }

        scored = app_module.pipeline.score_prior_borrowing_pair(query, candidate)

        self.assertEqual(scored["dimension_scores"]["endpoint_estimand_match"], 0.0)

    def test_root_serves_html(self) -> None:
        response = self.client.get("/")

        self.assertEqual(response.status_code, 200)
        self.assertIn("text/html", response.headers["content-type"])
        html = response.text
        self.assertIn("renderDensityChart", html)
        self.assertIn("renderWeightSensitivityChart", html)
        self.assertIn("renderTippingPointChart", html)


if __name__ == "__main__":
    unittest.main()
