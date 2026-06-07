from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "scripts"))

import run_borrowing_baseline_comparison as baseline_comparison  # noqa: E402
import temporal_validation  # noqa: E402


DEFAULT_METHODS = (
    "weak_only",
    "rule",
    "fixed_discount",
    "commensurate_like",
    "rule_sam",
)


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
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


def read_learned_nll_csv(
    path: Path,
    method_name: str = "two_head_trained",
    nll_column: str = "learned_nll",
) -> dict[str, dict[str, float]]:
    by_query: dict[str, dict[str, float]] = {}
    with path.open(encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        if nll_column not in (reader.fieldnames or []):
            raise ValueError(f"{path} does not contain required NLL column {nll_column!r}")
        for row in reader:
            query_id = str(row.get("query_nct_id") or "").strip()
            if not query_id or not row.get(nll_column):
                continue
            by_query.setdefault(query_id, {})[method_name] = float(row[nll_column])
    return by_query


def _mean(values: list[float]) -> float:
    finite = [value for value in values if math.isfinite(value)]
    return sum(finite) / len(finite) if finite else math.nan


def _nll_for_examples(examples: list[dict[str, Any]], method: str, fixed_discount: float) -> list[float]:
    nlls: list[float] = []
    for example in examples:
        probability, _prior = baseline_comparison.predictive_probability_for_method(
            example,
            method=method,
            fixed_discount=fixed_discount,
        )
        nlls.append(-math.log(max(probability, 1e-300)))
    return nlls


def _learned_nll_for_examples(
    examples: list[dict[str, Any]],
    learned_nll_by_query: dict[str, dict[str, float]],
    method: str,
) -> list[float]:
    values: list[float] = []
    for example in examples:
        query_id = str(example.get("query_nct_id") or "").strip()
        value = learned_nll_by_query.get(query_id, {}).get(method)
        if value is not None:
            values.append(float(value))
    return values


def _temporal_row(
    split_strategy: str,
    split_label: str,
    train_count: int,
    eval_count: int,
    method: str,
    nlls: list[float],
) -> dict[str, Any]:
    return {
        "split_strategy": split_strategy,
        "split_label": split_label,
        "train_count": train_count,
        "eval_count": eval_count,
        "method": method,
        "nll_count": len(nlls),
        "mean_nll": _mean(nlls),
        "median_nll": _median(nlls),
        "min_nll": min(nlls) if nlls else math.nan,
        "max_nll": max(nlls) if nlls else math.nan,
    }


def _median(values: list[float]) -> float:
    finite = sorted(value for value in values if math.isfinite(value))
    if not finite:
        return math.nan
    midpoint = len(finite) // 2
    if len(finite) % 2:
        return finite[midpoint]
    return (finite[midpoint - 1] + finite[midpoint]) / 2.0


def _rows_for_split(
    examples: list[dict[str, Any]],
    eval_indices: list[int],
    split_strategy: str,
    split_label: str,
    train_count: int,
    methods: tuple[str, ...],
    fixed_discount: float,
    learned_nll_by_query: dict[str, dict[str, float]] | None = None,
) -> list[dict[str, Any]]:
    eval_examples = [examples[index] for index in eval_indices]
    rows: list[dict[str, Any]] = []
    for method in methods:
        rows.append(
            _temporal_row(
                split_strategy=split_strategy,
                split_label=split_label,
                train_count=train_count,
                eval_count=len(eval_examples),
                method=method,
                nlls=_nll_for_examples(eval_examples, method, fixed_discount=fixed_discount),
            )
        )
    learned_nll_by_query = learned_nll_by_query or {}
    learned_methods = sorted({method for row in learned_nll_by_query.values() for method in row})
    for method in learned_methods:
        nlls = _learned_nll_for_examples(eval_examples, learned_nll_by_query, method)
        rows.append(
            _temporal_row(
                split_strategy=split_strategy,
                split_label=split_label,
                train_count=train_count,
                eval_count=len(eval_examples),
                method=method,
                nlls=nlls,
            )
        )
    return rows


def date_based_temporal_nll_rows(
    examples: list[dict[str, Any]],
    cutoffs: list[str],
    methods: tuple[str, ...] = DEFAULT_METHODS,
    fixed_discount: float = 0.25,
    learned_nll_by_query: dict[str, dict[str, float]] | None = None,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for cutoff in cutoffs:
        train_indices, eval_indices, _metadata = temporal_validation.date_based_split_indices(
            examples,
            train_end_date=cutoff,
        )
        rows.extend(
            _rows_for_split(
                examples,
                eval_indices=eval_indices,
                split_strategy="date_based",
                split_label=f"train_through_{cutoff}",
                train_count=len(train_indices),
                methods=methods,
                fixed_discount=fixed_discount,
                learned_nll_by_query=learned_nll_by_query,
            )
        )
    return rows


def rolling_origin_temporal_nll_rows(
    examples: list[dict[str, Any]],
    min_train_count: int,
    eval_window_size: int,
    methods: tuple[str, ...] = DEFAULT_METHODS,
    fixed_discount: float = 0.25,
    learned_nll_by_query: dict[str, dict[str, float]] | None = None,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for split in temporal_validation.rolling_origin_splits(
        examples,
        min_train_count=min_train_count,
        eval_window_size=eval_window_size,
    ):
        rows.extend(
            _rows_for_split(
                examples,
                eval_indices=split["eval_indices"],
                split_strategy="rolling_origin",
                split_label=f"window_{split['split_id']}",
                train_count=int(split["train_count"]),
                methods=methods,
                fixed_discount=fixed_discount,
                learned_nll_by_query=learned_nll_by_query,
            )
        )
    return rows


def add_delta_vs_reference(rows: list[dict[str, Any]], reference_method: str) -> list[dict[str, Any]]:
    reference_by_split = {
        (row["split_strategy"], row["split_label"]): float(row["mean_nll"])
        for row in rows
        if row["method"] == reference_method and math.isfinite(float(row["mean_nll"]))
    }
    output = []
    for row in rows:
        copied = dict(row)
        reference = reference_by_split.get((row["split_strategy"], row["split_label"]))
        mean_nll = float(row["mean_nll"])
        copied[f"mean_nll_minus_{reference_method}"] = (
            mean_nll - reference
            if reference is not None and math.isfinite(mean_nll)
            else math.nan
        )
        output.append(copied)
    return output


def _fmt(value: Any) -> str:
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return "NA"
    return f"{numeric:.6f}" if math.isfinite(numeric) else "NA"


def write_report(path: Path, rows: list[dict[str, Any]], reference_method: str) -> None:
    delta_key = f"mean_nll_minus_{reference_method}"
    lines = [
        "# True-Date Temporal Borrowing Validation",
        "",
        "Rows are evaluated on pseudo-queries ordered by true CT.gov date metadata. This is retrospective predictive calibration, not expert borrowability validation.",
        "",
        "| Strategy | Split | Method | Train | Eval | NLL rows | Mean NLL | Delta vs reference |",
        "|---|---|---|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            "| {split_strategy} | {split_label} | {method} | {train_count} | {eval_count} | {nll_count} | {mean_nll} | {delta} |".format(
                split_strategy=row["split_strategy"],
                split_label=row["split_label"],
                method=row["method"],
                train_count=row["train_count"],
                eval_count=row["eval_count"],
                nll_count=row["nll_count"],
                mean_nll=_fmt(row["mean_nll"]),
                delta=_fmt(row.get(delta_key)),
            )
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Run true-date temporal borrowing NLL validation tables.")
    parser.add_argument("--examples-jsonl", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts/temporal_validation_true_dates"))
    parser.add_argument("--methods", nargs="+", default=list(DEFAULT_METHODS))
    parser.add_argument("--fixed-discount", type=float, default=0.25)
    parser.add_argument("--cutoffs", nargs="+", default=["2019-12-31", "2020-12-31", "2021-12-31", "2022-12-31"])
    parser.add_argument("--rolling-min-train-count", type=int, default=500)
    parser.add_argument("--rolling-eval-window-size", type=int, default=250)
    parser.add_argument("--learned-nll-csv", type=Path, default=None)
    parser.add_argument("--learned-method-name", default="two_head_trained")
    parser.add_argument("--learned-nll-column", default="learned_nll")
    parser.add_argument("--reference-method", default="rule")
    args = parser.parse_args(argv)

    examples = read_jsonl(args.examples_jsonl)
    learned_nll_by_query = (
        read_learned_nll_csv(
            args.learned_nll_csv,
            method_name=args.learned_method_name,
            nll_column=args.learned_nll_column,
        )
        if args.learned_nll_csv
        else {}
    )
    rows = date_based_temporal_nll_rows(
        examples,
        cutoffs=args.cutoffs,
        methods=tuple(args.methods),
        fixed_discount=args.fixed_discount,
        learned_nll_by_query=learned_nll_by_query,
    )
    rows.extend(
        rolling_origin_temporal_nll_rows(
            examples,
            min_train_count=args.rolling_min_train_count,
            eval_window_size=args.rolling_eval_window_size,
            methods=tuple(args.methods),
            fixed_discount=args.fixed_discount,
            learned_nll_by_query=learned_nll_by_query,
        )
    )
    rows = add_delta_vs_reference(rows, reference_method=args.reference_method)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(args.output_dir / "temporal_borrowing_nll_table.csv", rows)
    write_report(args.output_dir / "temporal_borrowing_nll_report.md", rows, reference_method=args.reference_method)


if __name__ == "__main__":
    main()
