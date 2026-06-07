import importlib.util
import json
from pathlib import Path
import tempfile
import unittest


SCRIPT_PATH = Path(__file__).resolve().parents[1] / "scripts" / "run_paired_stage1_backend_benchmark.py"


def load_benchmark_module():
    spec = importlib.util.spec_from_file_location("run_paired_stage1_backend_benchmark", SCRIPT_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class PairedStage1BackendBenchmarkTests(unittest.TestCase):
    def test_paired_metric_delta_rows_compare_common_queries_to_baseline(self):
        module = load_benchmark_module()
        rows = [
            {"label": "hashing", "query_nct_id": "NCT1", "rerank_component_ready_rate": 0.4},
            {"label": "secret_pool", "query_nct_id": "NCT1", "rerank_component_ready_rate": 0.7},
            {"label": "hashing", "query_nct_id": "NCT2", "rerank_component_ready_rate": 0.6},
            {"label": "secret_pool", "query_nct_id": "NCT2", "rerank_component_ready_rate": 0.5},
        ]

        deltas = module.paired_metric_delta_rows(
            rows,
            baseline_label="hashing",
            metric_keys=["rerank_component_ready_rate"],
        )

        self.assertEqual(len(deltas), 2)
        self.assertAlmostEqual(deltas[0]["secret_pool_minus_hashing_rerank_component_ready_rate"], 0.3)
        self.assertAlmostEqual(deltas[1]["secret_pool_minus_hashing_rerank_component_ready_rate"], -0.1)

    def test_bootstrap_delta_ci_is_paired_and_has_stable_columns(self):
        module = load_benchmark_module()
        delta_rows = [
            {"secret_pool_minus_hashing_learned_nll": -0.2},
            {"secret_pool_minus_hashing_learned_nll": -0.1},
            {"secret_pool_minus_hashing_learned_nll": 0.05},
        ]

        rows = module.bootstrap_delta_ci(
            delta_rows,
            delta_key="secret_pool_minus_hashing_learned_nll",
            iterations=20,
            seed=11,
        )

        self.assertEqual(rows["delta_key"], "secret_pool_minus_hashing_learned_nll")
        self.assertIn("mean_delta", rows)
        self.assertIn("ci_lower", rows)
        self.assertIn("ci_upper", rows)
        self.assertEqual(rows["paired_query_count"], 3)

    def test_metric_rows_from_result_path_streams_jsonl(self):
        module = load_benchmark_module()
        result = {
            "query_summary": {"nct_id": "NCTQ", "endpoints": {"primary": [{"endpoint_family": "ORR/CR/PR"}]}},
            "top_matches": [],
            "reranked_top10": [],
        }

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "results.jsonl"
            path.write_text(json.dumps(result) + "\n", encoding="utf-8")

            rows = module.metric_rows_from_result_path(
                label="hashing",
                path=path,
                endpoint_key="ORR",
                top_k_eval=None,
            )

        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["label"], "hashing")
        self.assertEqual(rows[0]["query_nct_id"], "NCTQ")


if __name__ == "__main__":
    unittest.main()
