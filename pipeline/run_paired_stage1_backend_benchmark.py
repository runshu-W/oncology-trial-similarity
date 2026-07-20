from __future__ import annotations

import argparse
import csv
import json
import math
import random
import sys
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT / "pipeline") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "pipeline"))

import evaluate_stage1_retrieval as stage1_eval  # noqa: E402


DEFAULT_METRIC_KEYS = [
    "topk_endpoint_and_result_ready_rate",
    "rerank_component_ready_rate",
    "rerank_mean_overall_similarity",
    "rerank_mean_disease_match",
    "rerank_mean_endpoint_match",
]


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def iter_jsonl(path: Path):
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                yield json.loads(line)


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames = list(rows[0].keys())
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _finite_float(value: Any) -> float | None:
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return None
    return numeric if math.isfinite(numeric) else None


def paired_metric_delta_rows(
    metric_rows: list[dict[str, Any]],
    baseline_label: str,
    metric_keys: list[str] | None = None,
) -> list[dict[str, Any]]:
    metric_keys = metric_keys or DEFAULT_METRIC_KEYS
    labels = sorted({str(row.get("label")) for row in metric_rows if row.get("label")})
    comparison_labels = [label for label in labels if label != baseline_label]
    by_query: dict[str, dict[str, dict[str, Any]]] = {}
    for row in metric_rows:
        query_id = str(row.get("query_nct_id") or "")
        label = str(row.get("label") or "")
        if query_id and label:
            by_query.setdefault(query_id, {})[label] = row

    output = []
    for query_id, label_rows in sorted(by_query.items()):
        baseline = label_rows.get(baseline_label)
        if baseline is None:
            continue
        row_out: dict[str, Any] = {"query_nct_id": query_id, "baseline_label": baseline_label}
        has_delta = False
        for label in comparison_labels:
            comparison = label_rows.get(label)
            if comparison is None:
                continue
            for metric_key in metric_keys:
                baseline_value = _finite_float(baseline.get(metric_key))
                comparison_value = _finite_float(comparison.get(metric_key))
                if baseline_value is None or comparison_value is None:
                    continue
                row_out[f"{label}_minus_{baseline_label}_{metric_key}"] = comparison_value - baseline_value
                has_delta = True
        if has_delta:
            output.append(row_out)
    return output


def bootstrap_delta_ci(
    delta_rows: list[dict[str, Any]],
    delta_key: str,
    iterations: int = 1000,
    seed: int = 20260607,
) -> dict[str, Any]:
    values = []
    for row in delta_rows:
        value = _finite_float(row.get(delta_key))
        if value is not None:
            values.append(value)
    if not values:
        return {
            "delta_key": delta_key,
            "paired_query_count": 0,
            "mean_delta": math.nan,
            "ci_lower": math.nan,
            "ci_upper": math.nan,
            "iterations": iterations,
        }
    rng = random.Random(seed)
    means = []
    for _ in range(iterations):
        sample = [values[rng.randrange(len(values))] for _ in values]
        means.append(sum(sample) / len(sample))
    means.sort()
    lower_index = int(0.025 * (len(means) - 1))
    upper_index = int(0.975 * (len(means) - 1))
    return {
        "delta_key": delta_key,
        "paired_query_count": len(values),
        "mean_delta": sum(values) / len(values),
        "ci_lower": means[lower_index],
        "ci_upper": means[upper_index],
        "iterations": iterations,
    }


def parse_labeled_path(value: str) -> tuple[str, Path]:
    if "=" not in value:
        raise argparse.ArgumentTypeError("expected LABEL=/path/to/results.jsonl")
    label, path = value.split("=", 1)
    if not label:
        raise argparse.ArgumentTypeError("label must be non-empty")
    return label, Path(path)


def metric_rows_from_result_path(
    label: str,
    path: Path,
    endpoint_key: str,
    top_k_eval: int | None,
) -> list[dict[str, Any]]:
    return [
        stage1_eval.query_metrics(
            result,
            label=label,
            endpoint_key=endpoint_key,
            top_k_eval=top_k_eval,
        )
        for result in iter_jsonl(path)
    ]


def write_report(path: Path, summary_rows: list[dict[str, Any]], ci_rows: list[dict[str, Any]]) -> None:
    lines = [
        "# Paired Stage 1 Backend Benchmark",
        "",
        "All backends are compared on common query IDs with the same endpoint key and candidate budget.",
        "",
        "## Mean Metrics",
        "",
        "| Label | Query count | Component-ready | Endpoint+result ready | Endpoint match |",
        "|---|---:|---:|---:|---:|",
    ]
    for row in summary_rows:
        lines.append(
            "| {label} | {query_count} | {mean_rerank_component_ready_rate:.4f} | {mean_topk_endpoint_and_result_ready_rate:.4f} | {mean_rerank_mean_endpoint_match:.4f} |".format(
                **row
            )
        )
    lines.extend(["", "## Paired Delta Bootstrap CIs", "", "| Delta | Mean | CI lower | CI upper | n |", "|---|---:|---:|---:|---:|"])
    for row in ci_rows:
        lines.append(
            "| {delta_key} | {mean_delta:.4f} | {ci_lower:.4f} | {ci_upper:.4f} | {paired_query_count} |".format(
                **row
            )
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Run a paired Stage 1 retrieval backend benchmark.")
    parser.add_argument("--results", nargs="+", type=parse_labeled_path, required=True)
    parser.add_argument("--baseline-label", required=True)
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts/paired_stage1_backend_benchmark"))
    parser.add_argument("--endpoint-key", default="ORR")
    parser.add_argument("--top-k-eval", type=int, default=None)
    parser.add_argument("--bootstrap-iterations", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=20260607)
    args = parser.parse_args(argv)

    metric_rows = []
    for label, path in args.results:
        metric_rows.extend(
            metric_rows_from_result_path(
                label=label,
                path=path,
                endpoint_key=args.endpoint_key,
                top_k_eval=args.top_k_eval,
            )
        )
    summary_rows = stage1_eval.summarize_query_metrics(metric_rows)
    delta_rows = paired_metric_delta_rows(
        metric_rows,
        baseline_label=args.baseline_label,
        metric_keys=DEFAULT_METRIC_KEYS,
    )
    delta_keys = sorted({key for row in delta_rows for key in row if key.endswith(tuple(DEFAULT_METRIC_KEYS))})
    ci_rows = [
        bootstrap_delta_ci(delta_rows, key, iterations=args.bootstrap_iterations, seed=args.seed)
        for key in delta_keys
    ]

    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(args.output_dir / "paired_backend_query_metrics.csv", metric_rows)
    write_csv(args.output_dir / "paired_backend_summary.csv", summary_rows)
    write_csv(args.output_dir / "paired_backend_delta_rows.csv", delta_rows)
    write_csv(args.output_dir / "paired_backend_delta_bootstrap_ci.csv", ci_rows)
    write_report(args.output_dir / "paired_backend_benchmark_report.md", summary_rows, ci_rows)


if __name__ == "__main__":
    main()
