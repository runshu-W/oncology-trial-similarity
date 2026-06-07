import importlib.util
import json
from pathlib import Path
import tempfile
import unittest


SCRIPT_PATH = Path(__file__).resolve().parents[1] / "scripts" / "apply_secret_pool_rerank.py"


def load_module():
    spec = importlib.util.spec_from_file_location("apply_secret_pool_rerank", SCRIPT_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class SecretPoolRerankTests(unittest.TestCase):
    def result(self):
        return {
            "query_summary": {
                "nct_id": "NCTQ",
                "brief_title": "Drug A objective response in lung cancer",
                "brief_summary": "A study of Drug A for lung cancer.",
                "cancer_type": {"primary_site": ["Lung"], "histology": ["NSCLC"]},
                "population": {"key_inclusion": ["Adults with measurable disease"]},
                "intervention": {"experimental_regimen": "Drug A"},
                "endpoints": {
                    "primary": [
                        {"title": "Objective Response Rate", "endpoint_family": "ORR/CR/PR"}
                    ]
                },
                "design": {"single_or_multi_arm": "Single-arm"},
            },
            "retrieval_backend": "hashing",
            "embedding_backend": "hashing",
            "top_matches": [
                {
                    "nct_id": "NCTBAD",
                    "brief_title": "Overall survival in breast cancer",
                    "brief_summary": "Breast cancer study.",
                    "cancer_type": {"primary_site": ["Breast"]},
                    "population": {},
                    "intervention": {"experimental_regimen": "Drug Z"},
                    "endpoints": {
                        "primary": [{"title": "Overall Survival", "endpoint_family": "OS"}]
                    },
                    "design": {"single_or_multi_arm": "Randomized"},
                    "borrowable_quantities": [],
                },
                {
                    "nct_id": "NCTGOOD",
                    "brief_title": "Drug A objective response in lung cancer",
                    "brief_summary": "Lung cancer study of Drug A.",
                    "cancer_type": {"primary_site": ["Lung"], "histology": ["NSCLC"]},
                    "population": {"key_inclusion": ["Adults with measurable disease"]},
                    "intervention": {"experimental_regimen": "Drug A"},
                    "endpoints": {
                        "primary": [
                            {
                                "title": "Objective Response Rate",
                                "endpoint_family": "ORR/CR/PR",
                                "arm_results": [{"count": 5, "denominator": 10}],
                            }
                        ]
                    },
                    "design": {"single_or_multi_arm": "Single-arm"},
                    "result_usability": {
                        "has_posted_results": True,
                        "denominators_available": True,
                    },
                    "borrowable_quantities": [
                        {
                            "endpoint_family": "ORR/CR/PR",
                            "arm_results": [{"count": 5, "denominator": 10}],
                        }
                    ],
                },
            ],
            "heldout_query_outcomes": {"nct_id": "NCTQ", "endpoints": {}},
            "retrospective_leakage_control": {
                "query_outcomes_hidden_from_retrieval": True,
                "heldout_query_outcomes_for_post_retrieval_analysis": True,
            },
        }

    def test_secret_pool_rerank_promotes_semantically_closer_candidate(self):
        module = load_module()

        updated = module.rerank_pipeline_result(self.result(), top_k=2, rerank_top_n=2)

        self.assertEqual(updated["retrieval_backend"], "secret_pool_rerank")
        self.assertEqual(updated["top_matches"][0]["nct_id"], "NCTGOOD")
        self.assertIn("secret_pool_score", updated["top_matches"][0])
        self.assertIn("secret_section_scores", updated["top_matches"][0])
        self.assertTrue(updated["retrospective_leakage_control"]["query_outcomes_hidden_from_retrieval"])
        self.assertEqual(updated["heldout_query_outcomes"]["nct_id"], "NCTQ")
        self.assertEqual(len(updated["reranked_top_matches"]), 2)

    def test_main_writes_jsonl(self):
        module = load_module()
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            input_path = tmp_path / "input.jsonl"
            output_path = tmp_path / "output.jsonl"
            input_path.write_text(json.dumps(self.result()) + "\n", encoding="utf-8")

            module.main(
                [
                    "--input-jsonl",
                    str(input_path),
                    "--output-jsonl",
                    str(output_path),
                    "--top-k",
                    "2",
                    "--rerank-top-n",
                    "2",
                ]
            )

            rows = [json.loads(line) for line in output_path.read_text(encoding="utf-8").splitlines()]
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["top_matches"][0]["nct_id"], "NCTGOOD")


if __name__ == "__main__":
    unittest.main()
