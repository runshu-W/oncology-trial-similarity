from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "scripts"))

import temporal_validation  # noqa: E402


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


def _metadata_source_summary(examples: list[dict[str, Any]]) -> str:
    counts: dict[str, int] = {}
    for example in examples:
        _key, source = temporal_validation.temporal_key_for_example(example)
        counts[source] = counts.get(source, 0) + 1
    return json.dumps(dict(sorted(counts.items())), sort_keys=True)


def temporal_split_summary_rows(
    examples: list[dict[str, Any]],
    cutoffs: list[str],
    rolling_min_train_count: int,
    rolling_eval_window_size: int,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for cutoff in cutoffs:
        try:
            train_indices, eval_indices, metadata = temporal_validation.date_based_split_indices(
                examples,
                train_end_date=cutoff,
            )
            rows.append(
                {
                    "split_strategy": "date_based",
                    "split_label": f"train_through_{cutoff}",
                    "train_count": len(train_indices),
                    "eval_count": len(eval_indices),
                    "status": "ready",
                    "temporal_key_sources": json.dumps(metadata["temporal_key_sources"], sort_keys=True),
                }
            )
        except ValueError as exc:
            rows.append(
                {
                    "split_strategy": "date_based",
                    "split_label": f"train_through_{cutoff}",
                    "train_count": 0,
                    "eval_count": 0,
                    "status": f"skipped: {exc}",
                    "temporal_key_sources": _metadata_source_summary(examples),
                }
            )
    rolling_splits = temporal_validation.rolling_origin_splits(
        examples,
        min_train_count=rolling_min_train_count,
        eval_window_size=rolling_eval_window_size,
    )
    for split in rolling_splits:
        rows.append(
            {
                "split_strategy": "rolling_origin",
                "split_label": f"window_{split['split_id']}",
                "train_count": split["train_count"],
                "eval_count": split["eval_count"],
                "status": "ready",
                "temporal_key_sources": _metadata_source_summary(examples),
            }
        )
    return rows


def write_report(path: Path, rows: list[dict[str, Any]]) -> None:
    lines = [
        "# True Temporal Validation Split Summary",
        "",
        "Splits use true CT.gov date metadata attached to query examples before model fitting.",
        "",
        "| Strategy | Split | Train | Eval | Status |",
        "|---|---|---:|---:|---|",
    ]
    for row in rows:
        lines.append(
            f"| {row['split_strategy']} | {row['split_label']} | {row['train_count']} | {row['eval_count']} | {row['status']} |"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Summarize true-date temporal validation split feasibility.")
    parser.add_argument("--examples-jsonl", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts/temporal_validation_true_dates"))
    parser.add_argument("--cutoffs", nargs="+", default=["2019-12-31", "2020-12-31", "2021-12-31", "2022-12-31"])
    parser.add_argument("--rolling-min-train-count", type=int, default=500)
    parser.add_argument("--rolling-eval-window-size", type=int, default=250)
    args = parser.parse_args(argv)

    examples = read_jsonl(args.examples_jsonl)
    rows = temporal_split_summary_rows(
        examples,
        cutoffs=args.cutoffs,
        rolling_min_train_count=args.rolling_min_train_count,
        rolling_eval_window_size=args.rolling_eval_window_size,
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(args.output_dir / "temporal_split_summary.csv", rows)
    write_report(args.output_dir / "temporal_split_summary_report.md", rows)


if __name__ == "__main__":
    main()
