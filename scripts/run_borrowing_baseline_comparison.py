from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT / "docs") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "docs"))

import mixture_prior  # noqa: E402


DEFAULT_METHODS = (
    "weak_only",
    "rule",
    "fixed_discount",
    "map_like",
    "power_prior_like",
    "commensurate_like",
    "model",
    "two_head",
    "rule_sam",
    "model_sam",
    "two_head_sam",
)


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows


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


def learned_nll_rows_from_csv(
    path: Path,
    method_name: str = "two_head_trained",
    nll_column: str = "learned_nll",
) -> list[dict[str, Any]]:
    """Load held-out NLL values emitted by the trained retrospective lambda model.

    The lambda example JSONL is sufficient for recomputing rule-based and
    classical borrowing baselines, but the trained two-head DeepSets predictive
    likelihood is stored in the model evaluation CSV. Treating these rows as a
    separate method keeps the head-to-head table aligned with the actual
    learned model rather than a zero-lambda fallback.
    """
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        if nll_column not in (reader.fieldnames or []):
            raise ValueError(f"{path} does not contain required NLL column {nll_column!r}")
        for csv_index, row in enumerate(reader):
            raw_nll = row.get(nll_column, "")
            if raw_nll == "":
                continue
            nll = float(raw_nll)
            rows.append(
                {
                    "example_index": int(float(row.get("example_index") or csv_index)),
                    "query_nct_id": row.get("query_nct_id", ""),
                    "method": method_name,
                    "predictive_probability": math.exp(-min(nll, 745.0)),
                    "nll": nll,
                    "lambda_0": math.nan,
                    "historical_mass": math.nan,
                    "component_count": math.nan,
                    "sam_status": "from_learned_nll_csv",
                }
            )
    return rows


def _component_count_denominator(component: dict[str, Any]) -> tuple[float, float]:
    count = component.get("count")
    denominator = component.get("denominator")
    if count is not None and denominator is not None:
        return float(count), float(denominator)
    alpha = float(component.get("alpha", 1.0))
    beta = float(component.get("beta", 1.0))
    denominator = max(0.0, alpha + beta - 2.0)
    count = max(0.0, alpha - 1.0)
    return count, denominator


def _normalize(values: list[float], lambda0: float) -> tuple[float, list[float]]:
    lambda0 = max(0.0, min(1.0, float(lambda0)))
    total = sum(max(0.0, value) for value in values)
    if total <= 0.0:
        return 1.0, [0.0 for _ in values]
    return lambda0, [(1.0 - lambda0) * max(0.0, value) / total for value in values]


def _method_lambdas(example: dict[str, Any], method: str) -> tuple[float, list[float]]:
    components = list(example.get("components") or [])
    lambda0 = float(example.get("lambda_0", 0.2))
    if method == "weak_only":
        return 1.0, [0.0 for _ in components]
    if method in {"map_like", "power_prior_like"}:
        weights = []
        for component in components:
            count, denominator = _component_count_denominator(component)
            weights.append(max(0.0, denominator) * max(0.01, (count + 1.0) / (denominator + 2.0)))
        return _normalize(weights, lambda0=lambda0)
    if method == "commensurate_like":
        means = []
        for component in components:
            count, denominator = _component_count_denominator(component)
            means.append((count + 1.0) / (denominator + 2.0))
        center = sum(means) / len(means) if means else 0.5
        weights = []
        for component, mean in zip(components, means):
            denominator = _component_count_denominator(component)[1]
            weights.append(max(0.0, denominator) * math.exp(-abs(mean - center) / 0.10))
        return _normalize(weights, lambda0=lambda0)
    if method.startswith("model") or method.startswith("two_head"):
        weights = []
        for component in components:
            raw = component.get("lambda_model", component.get("lambda_active"))
            weights.append(float(raw) if raw is not None else 0.0)
        return _normalize(weights, lambda0=lambda0)
    weights = []
    rule_values = list(example.get("lambda_rule") or [])
    for index, component in enumerate(components):
        raw = component.get("lambda_rule")
        if raw is None and index < len(rule_values):
            raw = rule_values[index]
        weights.append(float(raw) if raw is not None else 0.0)
    return _normalize(weights, lambda0=lambda0)


def _method_components(example: dict[str, Any], method: str, fixed_discount: float) -> list[dict[str, float]]:
    output = []
    for component in example.get("components") or []:
        count, denominator = _component_count_denominator(component)
        if method in {"fixed_discount", "power_prior_like"}:
            discount = fixed_discount
        elif method.startswith("model") or method.startswith("two_head"):
            discount = component.get("discount_model", component.get("discount_active", component.get("discount", fixed_discount)))
        else:
            discount = component.get("discount", fixed_discount)
        discount = max(0.0, min(1.0, float(discount)))
        output.append(
            {
                "alpha": 1.0 + discount * count,
                "beta": 1.0 + discount * (denominator - count),
            }
        )
    return output


def prior_for_method(example: dict[str, Any], method: str, fixed_discount: float = 0.25) -> dict[str, Any]:
    base_method = method.replace("_sam", "")
    lambda0, lambdas = _method_lambdas(example, base_method)
    components = _method_components(example, base_method, fixed_discount=fixed_discount)
    for component, lambda_i in zip(components, lambdas):
        component["lambda"] = lambda_i
        component["lambda_active"] = lambda_i
    return {"lambda_0": lambda0, "components": components, "mode": method}


def predictive_probability_for_method(
    example: dict[str, Any],
    method: str,
    fixed_discount: float = 0.25,
) -> tuple[float, dict[str, Any]]:
    y = int(float(example["query"]["count"]))
    n = int(float(example["query"]["denominator"]))
    prior = prior_for_method(example, method, fixed_discount=fixed_discount)
    if method.endswith("_sam"):
        prior = mixture_prior.apply_sam_conflict_adapter(prior, y=y, n=n)
    probability = mixture_prior.mixture_predictive_probability(
        y=y,
        n=n,
        lambda0=float(prior.get("lambda_0", 1.0)),
        weak_alpha=1.0,
        weak_beta=1.0,
        components=[
            {
                "alpha": float(component["alpha"]),
                "beta": float(component["beta"]),
                "lambda": float(component.get("lambda_active", component.get("lambda", 0.0))),
            }
            for component in prior.get("components") or []
        ],
    )
    return probability, prior


def baseline_nll_rows(
    examples: list[dict[str, Any]],
    methods: tuple[str, ...] = DEFAULT_METHODS,
    fixed_discount: float = 0.25,
) -> list[dict[str, Any]]:
    rows = []
    for index, example in enumerate(examples):
        for method in methods:
            probability, prior = predictive_probability_for_method(
                example,
                method=method,
                fixed_discount=fixed_discount,
            )
            historical_mass = sum(
                float(component.get("lambda_active", component.get("lambda", 0.0)))
                for component in prior.get("components") or []
            )
            rows.append(
                {
                    "example_index": index,
                    "query_nct_id": example.get("query_nct_id", ""),
                    "method": method,
                    "predictive_probability": probability,
                    "nll": -math.log(max(probability, 1e-300)),
                    "lambda_0": float(prior.get("lambda_0", 1.0)),
                    "historical_mass": historical_mass,
                    "component_count": len(prior.get("components") or []),
                    "sam_status": (prior.get("sam_prior_data_conflict") or {}).get("status", ""),
                }
            )
    return rows


def summarize_baseline_rows(rows: list[dict[str, Any]], reference_method: str = "rule") -> list[dict[str, Any]]:
    by_method: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        by_method.setdefault(str(row["method"]), []).append(row)
    reference_mean = _mean([float(row["nll"]) for row in by_method.get(reference_method, [])])
    output = []
    for method, method_rows in sorted(by_method.items()):
        mean_nll = _mean([float(row["nll"]) for row in method_rows])
        output.append(
            {
                "method": method,
                "example_count": len(method_rows),
                "mean_nll": mean_nll,
                f"mean_nll_minus_{reference_method}": mean_nll - reference_mean,
                "mean_predictive_probability": _mean(
                    [float(row["predictive_probability"]) for row in method_rows if "predictive_probability" in row]
                ),
                "mean_historical_mass": _mean(
                    [float(row["historical_mass"]) for row in method_rows if "historical_mass" in row]
                ),
            }
        )
    return output


def _mean(values: list[float]) -> float:
    finite = [value for value in values if math.isfinite(value)]
    return sum(finite) / len(finite) if finite else math.nan


def _format_table_float(value: Any) -> str:
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return "NA"
    return f"{numeric:.6f}" if math.isfinite(numeric) else "NA"


def write_report(path: Path, summary_rows: list[dict[str, Any]], reference_method: str) -> None:
    lines = [
        "# Borrowing Baseline Head-to-Head",
        "",
        "This comparison uses held-out beta-binomial predictive NLL without expert labels.",
        "",
        "| Method | Examples | Mean NLL | Delta vs reference | Mean historical mass |",
        "|---|---:|---:|---:|---:|",
    ]
    delta_key = f"mean_nll_minus_{reference_method}"
    for row in summary_rows:
        lines.append(
            "| {method} | {example_count} | {mean_nll} | {delta} | {historical_mass} |".format(
                method=row["method"],
                example_count=row["example_count"],
                mean_nll=_format_table_float(row["mean_nll"]),
                delta=_format_table_float(row[delta_key]),
                historical_mass=_format_table_float(row["mean_historical_mass"]),
            )
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Compare historical-borrowing prior baselines on lambda examples.")
    parser.add_argument("--examples-jsonl", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts/borrowing_baseline_head_to_head"))
    parser.add_argument("--methods", nargs="+", default=list(DEFAULT_METHODS))
    parser.add_argument("--fixed-discount", type=float, default=0.25)
    parser.add_argument("--reference-method", default="rule")
    parser.add_argument(
        "--learned-nll-csv",
        type=Path,
        help="Optional lambda_nll_rows.csv from a trained two-head evaluation to include as a true learned method.",
    )
    parser.add_argument("--learned-method-name", default="two_head_trained")
    parser.add_argument("--learned-nll-column", default="learned_nll")
    args = parser.parse_args(argv)

    examples = read_jsonl(args.examples_jsonl)
    rows = baseline_nll_rows(
        examples,
        methods=tuple(args.methods),
        fixed_discount=args.fixed_discount,
    )
    if args.learned_nll_csv:
        rows.extend(
            learned_nll_rows_from_csv(
                args.learned_nll_csv,
                method_name=args.learned_method_name,
                nll_column=args.learned_nll_column,
            )
        )
    summary = summarize_baseline_rows(rows, reference_method=args.reference_method)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(args.output_dir / "borrowing_baseline_nll.csv", rows)
    write_csv(args.output_dir / "borrowing_baseline_summary.csv", summary)
    write_report(args.output_dir / "borrowing_baseline_report.md", summary, args.reference_method)


if __name__ == "__main__":
    main()
