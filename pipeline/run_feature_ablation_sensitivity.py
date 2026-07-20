from __future__ import annotations

import argparse
import csv
import html
import json
import math
import sys
from pathlib import Path
from typing import Any, Iterable


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT / "pipeline") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "pipeline"))

import mixture_prior  # noqa: E402


FEATURE_GROUPS = {
    "disease": ["disease_match_i"],
    "regimen": ["regimen_match_i"],
    "endpoint": ["endpoint_match_i"],
    "followup": ["followup_match_i"],
    "eligibility": ["eligibility_match_i"],
    "result_quality": ["result_quality_i"],
    "redflag": ["negative_redflag_severity_i"],
    "information_size": ["log_n_i"],
}

DEFAULT_SECTION_WEIGHT_SCENARIOS = [
    {
        "name": "default_secret_weights",
        "weights": {
            "disease_population": 0.22,
            "intervention": 0.20,
            "eligibility": 0.16,
            "endpoint": 0.18,
            "design": 0.10,
            "results": 0.10,
            "safety_followup": 0.04,
        },
    },
    {
        "name": "disease_endpoint_heavy",
        "weights": {
            "disease_population": 0.35,
            "intervention": 0.15,
            "eligibility": 0.10,
            "endpoint": 0.30,
            "design": 0.05,
            "results": 0.03,
            "safety_followup": 0.02,
        },
    },
    {
        "name": "regimen_result_heavy",
        "weights": {
            "disease_population": 0.15,
            "intervention": 0.35,
            "eligibility": 0.10,
            "endpoint": 0.15,
            "design": 0.05,
            "results": 0.18,
            "safety_followup": 0.02,
        },
    },
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


def default_feature_ablations() -> list[dict[str, Any]]:
    ablations = [{"name": "full_9_feature_proxy", "drop_features": []}]
    for group, features in FEATURE_GROUPS.items():
        ablations.append({"name": f"drop_{group}", "drop_features": features})
    return ablations


def mask_example_features(example: dict[str, Any], drop_features: list[str]) -> dict[str, Any]:
    feature_names = list(example.get("feature_names") or [])
    drop_indices = {index for index, name in enumerate(feature_names) if name in set(drop_features)}
    masked = {**example, "features": []}
    for feature_row in example.get("features") or []:
        masked["features"].append(
            [0.0 if index in drop_indices else float(value) for index, value in enumerate(feature_row)]
        )
    return masked


def _proxy_raw_weights(example: dict[str, Any]) -> list[float]:
    weights = []
    feature_names = list(example.get("feature_names") or [])
    log_n_index = feature_names.index("log_n_i") if "log_n_i" in feature_names else None
    redflag_index = feature_names.index("negative_redflag_severity_i") if "negative_redflag_severity_i" in feature_names else None
    for features, component in zip(example.get("features") or [], example.get("components") or []):
        positive_values = []
        for index, value in enumerate(features):
            if index == log_n_index:
                continue
            if index == redflag_index:
                positive_values.append(max(0.0, 1.0 + float(value)))
            else:
                positive_values.append(max(0.0, float(value)))
        score = sum(positive_values) / len(positive_values) if positive_values else 0.0
        log_n = max(1.0, float(features[log_n_index])) if log_n_index is not None else 1.0
        gate = float(component.get("gate", 1.0))
        weights.append(max(0.0, gate) * max(0.0, score) * log_n)
    return weights


def _normalize(values: list[float], lambda0: float) -> tuple[float, list[float]]:
    total = sum(values)
    if total <= 0.0:
        return 1.0, [0.0 for _ in values]
    lambda0 = max(0.0, min(1.0, float(lambda0)))
    return lambda0, [(1.0 - lambda0) * value / total for value in values]


def proxy_nll_for_example(example: dict[str, Any]) -> float:
    y = int(float(example["query"]["count"]))
    n = int(float(example["query"]["denominator"]))
    lambda0, lambdas = _normalize(_proxy_raw_weights(example), float(example.get("lambda_0", 0.2)))
    components = []
    for component, lambda_i in zip(example.get("components") or [], lambdas):
        components.append(
            {
                "alpha": float(component["alpha"]),
                "beta": float(component["beta"]),
                "lambda": lambda_i,
            }
        )
    probability = mixture_prior.mixture_predictive_probability(
        y=y,
        n=n,
        lambda0=lambda0,
        weak_alpha=1.0,
        weak_beta=1.0,
        components=components,
    )
    return -math.log(max(probability, 1e-300))


def ablation_nll_rows(
    examples: list[dict[str, Any]],
    ablations: list[dict[str, Any]] | None = None,
) -> list[dict[str, Any]]:
    ablations = ablations or default_feature_ablations()
    full_mean = math.nan
    rows = []
    for ablation in ablations:
        masked_examples = [
            mask_example_features(example, list(ablation.get("drop_features") or []))
            for example in examples
        ]
        nll_values = [proxy_nll_for_example(example) for example in masked_examples]
        mean_nll = _mean(nll_values)
        if ablation["name"] == "full_9_feature_proxy":
            full_mean = mean_nll
        rows.append(
            {
                "ablation": ablation["name"],
                "dropped_features": ";".join(ablation.get("drop_features") or []),
                "example_count": len(masked_examples),
                "mean_nll": mean_nll,
                "mean_nll_minus_full": math.nan,
            }
        )
    if math.isfinite(full_mean):
        for row in rows:
            row["mean_nll_minus_full"] = row["mean_nll"] - full_mean
    return rows


def section_weight_score(section_scores: dict[str, Any], weights: dict[str, float]) -> float:
    weighted_sum = 0.0
    total_weight = 0.0
    for section, weight in weights.items():
        try:
            score = float(section_scores.get(section, 0.0))
        except (TypeError, ValueError):
            score = 0.0
        weighted_sum += float(weight) * score
        total_weight += float(weight)
    return weighted_sum / total_weight if total_weight > 0.0 else math.nan


def section_weight_sensitivity_rows(
    pipeline_results: Iterable[dict[str, Any]],
    scenarios: list[dict[str, Any]] | None = None,
) -> list[dict[str, Any]]:
    scenarios = scenarios or DEFAULT_SECTION_WEIGHT_SCENARIOS
    rows = []
    for result in pipeline_results:
        query = result.get("query_summary") if isinstance(result.get("query_summary"), dict) else {}
        query_id = str(query.get("nct_id") or result.get("query_nct_id") or "")
        candidates = result.get("top_matches") if isinstance(result.get("top_matches"), list) else []
        for scenario in scenarios:
            weights = dict(scenario.get("weights") or {})
            scored = []
            for candidate in candidates:
                if not isinstance(candidate, dict):
                    continue
                section_scores = candidate.get("secret_section_scores")
                if not isinstance(section_scores, dict):
                    continue
                scored.append(
                    (
                        section_weight_score(section_scores, weights),
                        str(candidate.get("nct_id") or candidate.get("candidate_nct_id") or ""),
                    )
                )
            scored = [(score, nct_id) for score, nct_id in scored if math.isfinite(score)]
            scored.sort(reverse=True)
            rows.append(
                {
                    "query_nct_id": query_id,
                    "scenario": str(scenario.get("name") or "unnamed"),
                    "candidate_count": len(scored),
                    "top_candidate_nct_id": scored[0][1] if scored else "",
                    "top_section_weighted_score": scored[0][0] if scored else math.nan,
                    "mean_section_weighted_score": _mean([score for score, _nct_id in scored]),
                }
            )
    return rows


def _mean(values: list[float]) -> float:
    finite = [value for value in values if math.isfinite(value)]
    return sum(finite) / len(finite) if finite else math.nan


def write_report(path: Path, rows: list[dict[str, Any]]) -> None:
    lines = [
        "# Feature Ablation Sensitivity",
        "",
        "This report uses a deterministic feature-weight proxy to test how the 9-feature schema",
        "affects held-out predictive NLL without requiring expert labels or model retraining.",
        "",
        "| Ablation | Dropped features | Mean NLL | Delta vs full | Examples |",
        "|---|---|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            f"| {row['ablation']} | {row['dropped_features']} | {row['mean_nll']:.6f} | {row['mean_nll_minus_full']:.6f} | {row['example_count']} |"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _svg_text(value: Any) -> str:
    return html.escape(str(value), quote=True)


def write_ablation_heatmap_svg(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    width = 820
    row_height = 34
    top = 56
    label_x = 24
    cell_x = 300
    cell_width = 360
    height = top + row_height * max(1, len(rows)) + 40
    finite_deltas = [abs(float(row.get("mean_nll_minus_full", 0.0))) for row in rows if math.isfinite(float(row.get("mean_nll_minus_full", 0.0)))]
    max_abs = max(finite_deltas) if finite_deltas else 1.0
    max_abs = max(max_abs, 1e-9)
    body = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="white"/>',
        '<text x="24" y="30" font-family="Arial" font-size="20" font-weight="bold">Feature ablation sensitivity</text>',
        '<text x="300" y="50" font-family="Arial" font-size="12" fill="#555">Delta NLL vs full 9-feature proxy</text>',
    ]
    for index, row in enumerate(rows):
        y = top + index * row_height
        delta = float(row.get("mean_nll_minus_full", 0.0))
        intensity = min(1.0, abs(delta) / max_abs)
        if delta >= 0:
            red = 255
            green = int(245 - 120 * intensity)
            blue = int(235 - 140 * intensity)
        else:
            red = int(235 - 120 * intensity)
            green = int(245 - 70 * intensity)
            blue = 255
        fill = f"rgb({red},{green},{blue})"
        body.extend(
            [
                f'<text x="{label_x}" y="{y + 22}" font-family="Arial" font-size="13">{_svg_text(row.get("ablation", ""))}</text>',
                f'<rect x="{cell_x}" y="{y + 6}" width="{cell_width}" height="24" rx="3" fill="{fill}" stroke="#ccc"/>',
                f'<text x="{cell_x + cell_width + 12}" y="{y + 23}" font-family="Arial" font-size="13">{delta:.4f}</text>',
            ]
        )
    body.append("</svg>")
    path.write_text("\n".join(body) + "\n", encoding="utf-8")


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Run feature mask ablation sensitivity on lambda examples.")
    parser.add_argument("--examples-jsonl", type=Path, required=True)
    parser.add_argument(
        "--pipeline-results-jsonl",
        type=Path,
        default=None,
        help="Optional pipeline results containing SECRET section scores for section-weight sensitivity.",
    )
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts/feature_ablation_sensitivity"))
    args = parser.parse_args(argv)

    examples = read_jsonl(args.examples_jsonl)
    rows = ablation_nll_rows(examples, default_feature_ablations())
    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(args.output_dir / "feature_ablation_results.csv", rows)
    write_ablation_heatmap_svg(args.output_dir / "feature_ablation_heatmap.svg", rows)
    if args.pipeline_results_jsonl is not None:
        section_rows = section_weight_sensitivity_rows(iter_jsonl(args.pipeline_results_jsonl))
        write_csv(args.output_dir / "section_weight_sensitivity.csv", section_rows)
    write_report(args.output_dir / "feature_ablation_report.md", rows)


if __name__ == "__main__":
    main()
