import argparse
import csv
import json
import math
from pathlib import Path
from typing import Any


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


def _candidate_id(row: dict[str, Any]) -> str:
    return str(row.get("candidate_nct_id") or row.get("nct_id") or "")


def _endpoint_families_from_summary(summary: dict[str, Any]) -> set[str]:
    families = set()
    endpoints = summary.get("endpoints")
    if isinstance(endpoints, dict):
        endpoint_groups = []
        for value in endpoints.values():
            if isinstance(value, list):
                endpoint_groups.extend(value)
        for endpoint in endpoint_groups:
            if isinstance(endpoint, dict):
                family = endpoint.get("endpoint_family")
                if family:
                    families.add(str(family).upper())
    return families


def _endpoint_matches_key(family: str, endpoint_key: str) -> bool:
    family_upper = str(family or "").upper()
    key_upper = str(endpoint_key or "").upper()
    if not family_upper or not key_upper:
        return False
    if key_upper == "ORR":
        return "ORR" in family_upper or "CR/PR" in family_upper
    return key_upper in family_upper


def _row_endpoint_hit(row: dict[str, Any], query_families: set[str], endpoint_key: str) -> bool:
    row_families = _endpoint_families_from_summary(row)
    for item in row.get("borrowable_quantities") or []:
        if isinstance(item, dict) and item.get("endpoint_family"):
            row_families.add(str(item["endpoint_family"]).upper())
    if query_families and row_families.intersection(query_families):
        return True
    return any(_endpoint_matches_key(family, endpoint_key) for family in row_families)


def _has_arm_result(row: dict[str, Any]) -> bool:
    for item in row.get("borrowable_quantities") or []:
        if isinstance(item, dict) and item.get("arm_results"):
            return True
    endpoints = row.get("endpoints")
    if isinstance(endpoints, dict):
        for group in endpoints.values():
            if isinstance(group, list):
                for endpoint in group:
                    if isinstance(endpoint, dict) and endpoint.get("arm_results"):
                        return True
    return False


def _result_ready(row: dict[str, Any]) -> bool:
    usability = row.get("result_usability")
    if isinstance(usability, dict):
        if usability.get("has_posted_results") and usability.get("denominators_available"):
            return True
    return _has_arm_result(row)


def _mean(values: list[float]) -> float:
    finite = [value for value in values if math.isfinite(value)]
    return sum(finite) / len(finite) if finite else math.nan


def _dimension_mean(rows: list[dict[str, Any]], key: str) -> float:
    values = []
    for row in rows:
        dimension_scores = row.get("dimension_scores")
        if isinstance(dimension_scores, dict):
            try:
                values.append(float(dimension_scores.get(key)))
            except (TypeError, ValueError):
                pass
    return _mean(values)


def query_metrics(
    result: dict[str, Any],
    label: str,
    endpoint_key: str,
    top_k_eval: int | None = None,
) -> dict[str, Any]:
    query_summary = result.get("query_summary") if isinstance(result.get("query_summary"), dict) else {}
    query_id = str(query_summary.get("nct_id") or "")
    query_families = _endpoint_families_from_summary(query_summary)
    top_matches = result.get("top_matches") if isinstance(result.get("top_matches"), list) else []
    if top_k_eval is not None:
        if top_k_eval <= 0:
            raise ValueError("top_k_eval must be greater than 0")
        top_matches = top_matches[:top_k_eval]
    reranked = (
        result.get("reranked_top_matches")
        if isinstance(result.get("reranked_top_matches"), list)
        else result.get("reranked_top10")
    )
    if not isinstance(reranked, list):
        reranked = []

    endpoint_hits = [
        row for row in top_matches if isinstance(row, dict) and _row_endpoint_hit(row, query_families, endpoint_key)
    ]
    result_ready = [row for row in top_matches if isinstance(row, dict) and _result_ready(row)]
    ready_endpoint_hits = [
        row for row in endpoint_hits if isinstance(row, dict) and _result_ready(row)
    ]
    component_ready = [
        row
        for row in reranked
        if isinstance(row, dict)
        and _row_endpoint_hit(row, query_families, endpoint_key)
        and _result_ready(row)
    ]
    overall_values = []
    redflag_counts = []
    for row in reranked:
        if not isinstance(row, dict):
            continue
        try:
            overall_values.append(float(row.get("overall_similarity_score")))
        except (TypeError, ValueError):
            pass
        red_flags = row.get("red_flags")
        if isinstance(red_flags, list):
            redflag_counts.append(float(len(red_flags)))

    return {
        "label": label,
        "query_nct_id": query_id,
        "retrieval_backend": result.get("retrieval_backend", ""),
        "embedding_backend": result.get("embedding_backend", ""),
        "topk_count": len(top_matches),
        "topk_endpoint_hit_count": len(endpoint_hits),
        "topk_endpoint_hit_rate": len(endpoint_hits) / len(top_matches) if top_matches else math.nan,
        "topk_result_ready_count": len(result_ready),
        "topk_result_ready_rate": len(result_ready) / len(top_matches) if top_matches else math.nan,
        "topk_endpoint_and_result_ready_count": len(ready_endpoint_hits),
        "topk_endpoint_and_result_ready_rate": (
            len(ready_endpoint_hits) / len(top_matches) if top_matches else math.nan
        ),
        "rerank_count": len(reranked),
        "rerank_component_ready_count": len(component_ready),
        "rerank_component_ready_rate": len(component_ready) / len(reranked) if reranked else math.nan,
        "rerank_mean_overall_similarity": _mean(overall_values),
        "rerank_mean_disease_match": _dimension_mean(reranked, "disease_population_match"),
        "rerank_mean_regimen_match": _dimension_mean(reranked, "treatment_regimen_match"),
        "rerank_mean_endpoint_match": _dimension_mean(reranked, "endpoint_estimand_match"),
        "rerank_mean_eligibility_match": _dimension_mean(reranked, "eligibility_criteria_overlap"),
        "rerank_mean_result_usability": _dimension_mean(reranked, "result_usability"),
        "rerank_mean_redflag_count": _mean(redflag_counts),
    }


def summarize_query_metrics(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    labels = sorted({row["label"] for row in rows})
    summary_rows = []
    for label in labels:
        subset = [row for row in rows if row["label"] == label]
        metric_keys = [
            key for key in subset[0] if key not in {"label", "query_nct_id", "retrieval_backend", "embedding_backend"}
        ]
        output = {"label": label, "query_count": len(subset)}
        for key in metric_keys:
            values = []
            for row in subset:
                try:
                    values.append(float(row[key]))
                except (TypeError, ValueError):
                    pass
            output[f"mean_{key}"] = _mean(values)
        summary_rows.append(output)
    return summary_rows


def topk_ids(result: dict[str, Any], top_k_eval: int | None = None) -> set[str]:
    top_matches = result.get("top_matches") if isinstance(result.get("top_matches"), list) else []
    if top_k_eval is not None:
        top_matches = top_matches[:top_k_eval]
    return {_candidate_id(row) for row in top_matches if isinstance(row, dict) and _candidate_id(row)}


def results_by_query(results: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    output = {}
    for result in results:
        query = result.get("query_summary") if isinstance(result.get("query_summary"), dict) else {}
        query_id = str(query.get("nct_id") or "")
        if query_id:
            output[query_id] = result
    return output


def overlap_rows(
    result_sets: dict[str, list[dict[str, Any]]],
    baseline_label: str | None = None,
    top_k_eval: int | None = None,
) -> list[dict[str, Any]]:
    if not result_sets:
        return []
    baseline_label = baseline_label or sorted(result_sets)[0]
    baseline = results_by_query(result_sets[baseline_label])
    rows = []
    for label, results in sorted(result_sets.items()):
        if label == baseline_label:
            continue
        by_query = results_by_query(results)
        for query_id, baseline_result in sorted(baseline.items()):
            if query_id not in by_query:
                continue
            baseline_ids = topk_ids(baseline_result, top_k_eval=top_k_eval)
            candidate_ids = topk_ids(by_query[query_id], top_k_eval=top_k_eval)
            union = baseline_ids | candidate_ids
            shared = baseline_ids & candidate_ids
            rows.append(
                {
                    "baseline_label": baseline_label,
                    "label": label,
                    "query_nct_id": query_id,
                    "baseline_topk_count": len(baseline_ids),
                    "candidate_topk_count": len(candidate_ids),
                    "shared_topk_count": len(shared),
                    "topk_jaccard_vs_baseline": len(shared) / len(union) if union else math.nan,
                }
            )
    return rows


def overlap_rows_from_top_ids(
    top_ids_by_label: dict[str, dict[str, set[str]]],
    baseline_label: str | None = None,
) -> list[dict[str, Any]]:
    if not top_ids_by_label:
        return []
    baseline_label = baseline_label or sorted(top_ids_by_label)[0]
    baseline = top_ids_by_label[baseline_label]
    rows = []
    for label, by_query in sorted(top_ids_by_label.items()):
        if label == baseline_label:
            continue
        for query_id, baseline_ids in sorted(baseline.items()):
            if query_id not in by_query:
                continue
            candidate_ids = by_query[query_id]
            union = baseline_ids | candidate_ids
            shared = baseline_ids & candidate_ids
            rows.append(
                {
                    "baseline_label": baseline_label,
                    "label": label,
                    "query_nct_id": query_id,
                    "baseline_topk_count": len(baseline_ids),
                    "candidate_topk_count": len(candidate_ids),
                    "shared_topk_count": len(shared),
                    "topk_jaccard_vs_baseline": len(shared) / len(union) if union else math.nan,
                }
            )
    return rows


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def _fmt(value: Any) -> str:
    if value is None:
        return "NA"
    if isinstance(value, float) and not math.isfinite(value):
        return "NA"
    try:
        return f"{float(value):.4f}"
    except (TypeError, ValueError):
        return str(value)


def markdown_report(summary_rows: list[dict[str, Any]], overlap: list[dict[str, Any]]) -> str:
    lines = [
        "# Stage 1 Retrieval Evaluation",
        "",
        "These metrics are no-expert proxy diagnostics. They measure whether Stage 1 retrieves candidates that are endpoint/result-ready and whether Stage 2 receives higher-quality top candidates.",
        "",
        "## Backend Summary",
        "",
        "| Label | Queries | TopK endpoint+result ready rate | Rerank component-ready rate | Rerank overall similarity | Endpoint match | Disease match | Regimen match |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in summary_rows:
        lines.append(
            "| {label} | {queries} | {ready} | {component} | {overall} | {endpoint} | {disease} | {regimen} |".format(
                label=row["label"],
                queries=row["query_count"],
                ready=_fmt(row.get("mean_topk_endpoint_and_result_ready_rate")),
                component=_fmt(row.get("mean_rerank_component_ready_rate")),
                overall=_fmt(row.get("mean_rerank_mean_overall_similarity")),
                endpoint=_fmt(row.get("mean_rerank_mean_endpoint_match")),
                disease=_fmt(row.get("mean_rerank_mean_disease_match")),
                regimen=_fmt(row.get("mean_rerank_mean_regimen_match")),
            )
        )
    if overlap:
        lines.extend(
            [
                "",
                "## Candidate Overlap",
                "",
                "| Baseline | Label | Mean topK Jaccard | Mean shared candidates | Compared queries |",
                "|---|---|---:|---:|---:|",
            ]
        )
        grouped = {}
        for row in overlap:
            key = (row["baseline_label"], row["label"])
            grouped.setdefault(key, []).append(row)
        for (baseline, label), rows in sorted(grouped.items()):
            lines.append(
                f"| {baseline} | {label} | {_fmt(_mean([float(row['topk_jaccard_vs_baseline']) for row in rows]))} | {_fmt(_mean([float(row['shared_topk_count']) for row in rows]))} | {len(rows)} |"
            )
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "- Higher endpoint/result-ready rates mean Stage 1 is more likely to recall candidates that can become mixture prior components.",
            "- Higher rerank dimension means the Stage 2 borrowing scorer is receiving clinically closer candidates.",
            "- Low overlap against a baseline is not automatically bad; it means the backend is surfacing different candidates, which must be judged by the quality proxies and downstream NLL.",
        ]
    )
    return "\n".join(lines) + "\n"


def parse_result_arg(value: str) -> tuple[str, Path]:
    if "=" not in value:
        raise argparse.ArgumentTypeError("--results must use label=/path/to/results.jsonl")
    label, path = value.split("=", 1)
    label = label.strip()
    if not label:
        raise argparse.ArgumentTypeError("result label must not be empty")
    return label, Path(path)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--results",
        action="append",
        type=parse_result_arg,
        required=True,
        help="Named pipeline result JSONL in label=path form. Can be repeated.",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--endpoint-key", default="ORR")
    parser.add_argument("--baseline-label", default=None)
    parser.add_argument(
        "--top-k-eval",
        type=int,
        default=None,
        help="Optional common Stage-1 cutoff for top_matches metrics and overlap.",
    )
    args = parser.parse_args(argv)

    query_rows = []
    top_ids_by_label: dict[str, dict[str, set[str]]] = {}
    for label, path in args.results:
        top_ids_by_label[label] = {}
        for result in iter_jsonl(path):
            query_rows.append(
                query_metrics(
                    result,
                    label=label,
                    endpoint_key=args.endpoint_key,
                    top_k_eval=args.top_k_eval,
                )
            )
            query = result.get("query_summary") if isinstance(result.get("query_summary"), dict) else {}
            query_id = str(query.get("nct_id") or "")
            if query_id:
                top_ids_by_label[label][query_id] = topk_ids(
                    result,
                    top_k_eval=args.top_k_eval,
                )
    summary_rows = summarize_query_metrics(query_rows)
    overlap = overlap_rows_from_top_ids(
        top_ids_by_label,
        baseline_label=args.baseline_label,
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(args.output_dir / "stage1_retrieval_query_metrics.csv", query_rows)
    write_csv(args.output_dir / "stage1_retrieval_summary.csv", summary_rows)
    write_csv(args.output_dir / "stage1_retrieval_overlap.csv", overlap)
    (args.output_dir / "stage1_retrieval_report.md").write_text(
        markdown_report(summary_rows, overlap),
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "query_metrics": str(args.output_dir / "stage1_retrieval_query_metrics.csv"),
                "summary": str(args.output_dir / "stage1_retrieval_summary.csv"),
                "overlap": str(args.output_dir / "stage1_retrieval_overlap.csv"),
                "report": str(args.output_dir / "stage1_retrieval_report.md"),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
