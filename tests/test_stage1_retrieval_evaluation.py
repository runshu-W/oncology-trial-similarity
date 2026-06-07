import importlib.util
import json
from pathlib import Path
import tempfile
import unittest


SCRIPT_PATH = (
    Path(__file__).resolve().parents[1] / "scripts" / "evaluate_stage1_retrieval.py"
)


def load_eval_module():
    spec = importlib.util.spec_from_file_location("evaluate_stage1_retrieval", SCRIPT_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class Stage1RetrievalEvaluationTests(unittest.TestCase):
    def result(self, backend="secret", query_id="NCTQ", candidate_id="NCTC"):
        return {
            "query_summary": {
                "nct_id": query_id,
                "endpoints": {
                    "primary": [
                        {
                            "title": "Objective Response Rate",
                            "endpoint_family": "ORR/CR/PR",
                        }
                    ]
                },
            },
            "retrieval_backend": backend,
            "embedding_backend": "hashing",
            "top_matches": [
                {
                    "nct_id": candidate_id,
                    "result_usability": {
                        "has_posted_results": True,
                        "denominators_available": True,
                    },
                    "endpoints": {
                        "primary": [
                            {
                                "title": "Objective Response Rate",
                                "endpoint_family": "ORR/CR/PR",
                                "arm_results": [
                                    {"count": 3, "denominator": 10},
                                ],
                            }
                        ]
                    },
                },
                {
                    "nct_id": "NCTOTHER",
                    "result_usability": {
                        "has_posted_results": False,
                        "denominators_available": False,
                    },
                    "endpoints": {
                        "primary": [
                            {"title": "Overall Survival", "endpoint_family": "OS"}
                        ]
                    },
                },
            ],
            "reranked_top_matches": [
                {
                    "candidate_nct_id": candidate_id,
                    "overall_similarity_score": 80.0,
                    "dimension_scores": {
                        "disease_population_match": 4.0,
                        "treatment_regimen_match": 3.0,
                        "endpoint_estimand_match": 5.0,
                        "eligibility_criteria_overlap": 2.0,
                        "result_usability": 5.0,
                    },
                    "borrowable_quantities": [
                        {
                            "endpoint_family": "ORR/CR/PR",
                            "arm_results": [
                                {"count": 3, "denominator": 10},
                            ],
                        }
                    ],
                    "red_flags": [],
                }
            ],
        }

    def test_query_metrics_reports_recall_and_quality_proxies(self):
        module = load_eval_module()

        row = module.query_metrics(self.result(), label="secret", endpoint_key="ORR")

        self.assertEqual(row["label"], "secret")
        self.assertEqual(row["query_nct_id"], "NCTQ")
        self.assertEqual(row["topk_count"], 2)
        self.assertEqual(row["topk_endpoint_hit_count"], 1)
        self.assertEqual(row["topk_result_ready_count"], 1)
        self.assertEqual(row["rerank_component_ready_count"], 1)
        self.assertAlmostEqual(row["rerank_mean_overall_similarity"], 80.0)
        self.assertAlmostEqual(row["rerank_mean_endpoint_match"], 5.0)

    def test_query_metrics_can_apply_common_topk_cutoff(self):
        module = load_eval_module()

        row = module.query_metrics(
            self.result(),
            label="secret",
            endpoint_key="ORR",
            top_k_eval=1,
        )

        self.assertEqual(row["topk_count"], 1)
        self.assertEqual(row["topk_endpoint_hit_count"], 1)
        self.assertEqual(row["topk_result_ready_count"], 1)

    def test_candidate_overlap_compares_same_query_between_backends(self):
        module = load_eval_module()
        baseline = self.result(backend="hashing", candidate_id="NCTA")
        candidate = self.result(backend="secret", candidate_id="NCTA")
        candidate["top_matches"].append({"nct_id": "NCTB"})

        rows = module.overlap_rows(
            {"hashing": [baseline], "secret": [candidate]},
            baseline_label="hashing",
            top_k_eval=1,
        )

        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["query_nct_id"], "NCTQ")
        self.assertAlmostEqual(rows[0]["topk_jaccard_vs_baseline"], 1.0)
        self.assertEqual(rows[0]["shared_topk_count"], 1)

    def test_main_writes_summary_query_and_report_files(self):
        module = load_eval_module()
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            results_path = tmp_path / "secret.jsonl"
            output_dir = tmp_path / "eval"
            results_path.write_text(
                json.dumps(self.result()) + "\n",
                encoding="utf-8",
            )

            module.main(
                [
                    "--results",
                    f"secret={results_path}",
                    "--output-dir",
                    str(output_dir),
                    "--endpoint-key",
                    "ORR",
                ]
            )

            self.assertTrue((output_dir / "stage1_retrieval_summary.csv").exists())
            self.assertTrue((output_dir / "stage1_retrieval_query_metrics.csv").exists())
            self.assertTrue((output_dir / "stage1_retrieval_report.md").exists())


if __name__ == "__main__":
    unittest.main()
