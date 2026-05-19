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

    def test_root_serves_html(self) -> None:
        response = self.client.get("/")

        self.assertEqual(response.status_code, 200)
        self.assertIn("text/html", response.headers["content-type"])


if __name__ == "__main__":
    unittest.main()
