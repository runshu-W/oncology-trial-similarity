from __future__ import annotations

import csv
import math
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Any


DATE_METADATA_FIELDS = (
    "primary_completion_date",
    "primary_completion_date_precision",
    "completion_date",
    "completion_date_precision",
    "results_first_posted_date",
    "results_first_posted_date_precision",
    "start_date",
    "start_date_precision",
    "temporal_sort_date",
    "temporal_sort_source",
)


def parse_temporal_sort_value(value: Any) -> float | None:
    if value is None:
        return None
    text = str(value).strip()
    if not text:
        return None
    for fmt in ("%Y-%m-%d", "%Y/%m/%d"):
        try:
            parsed = datetime.strptime(text, fmt)
            return float(parsed.year) + (parsed.timetuple().tm_yday - 1) / 366.0
        except ValueError:
            pass
    if len(text) >= 4 and text[:4].isdigit():
        return float(text[:4])
    return None


def nct_numeric_id(nct_id: str) -> int | None:
    digits = "".join(char for char in str(nct_id) if char.isdigit())
    return int(digits) if digits else None


def temporal_key_for_example(example: dict[str, Any]) -> tuple[float, str]:
    metadata = example.get("query_metadata") or {}
    temporal_sort_date = parse_temporal_sort_value(metadata.get("temporal_sort_date"))
    if temporal_sort_date is not None:
        return temporal_sort_date, str(metadata.get("temporal_sort_source") or "temporal_sort_date")
    for field in ("primary_completion_date", "completion_date", "results_first_posted_date", "start_date"):
        value = parse_temporal_sort_value(metadata.get(field))
        if value is not None:
            return value, field
    nct_value = nct_numeric_id(example.get("query_nct_id") or metadata.get("nct_id", ""))
    if nct_value is not None:
        return float(nct_value), "nct_id_numeric_proxy"
    return math.inf, "missing"


def read_date_metadata_csv(path: Path) -> dict[str, dict[str, str]]:
    """Read trial-level true date metadata keyed by NCT ID."""
    by_nct: dict[str, dict[str, str]] = {}
    with path.open(encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            nct_id = str(row.get("nct_id") or "").strip()
            if not nct_id:
                continue
            by_nct[nct_id] = {
                field: str(row.get(field) or "").strip()
                for field in DATE_METADATA_FIELDS
                if row.get(field)
            }
    return by_nct


def attach_date_metadata_to_examples(
    examples: list[dict[str, Any]],
    date_metadata_by_nct: dict[str, dict[str, str]],
) -> dict[str, Any]:
    """Merge true CT.gov date metadata into example query metadata in place."""
    attached_count = 0
    missing_count = 0
    for example in examples:
        query_id = str(example.get("query_nct_id") or example.get("query", {}).get("nct_id") or "").strip()
        date_metadata = date_metadata_by_nct.get(query_id)
        if not date_metadata:
            missing_count += 1
            continue
        metadata = example.setdefault("query_metadata", {})
        for field in DATE_METADATA_FIELDS:
            value = date_metadata.get(field)
            if value:
                metadata[field] = value
        attached_count += 1
    return {
        "example_count": len(examples),
        "attached_count": attached_count,
        "missing_count": missing_count,
        "attached_fraction": attached_count / len(examples) if examples else math.nan,
        "note": "True CT.gov date metadata was merged by query NCT ID before temporal validation.",
    }


def ordered_temporal_indices(examples: list[dict[str, Any]]) -> list[int]:
    return sorted(
        range(len(examples)),
        key=lambda index: (temporal_key_for_example(examples[index])[0], examples[index].get("query_nct_id", "")),
    )


def fraction_split_indices(
    examples: list[dict[str, Any]],
    train_fraction: float,
) -> tuple[list[int], list[int], dict[str, Any]]:
    if len(examples) < 2:
        raise ValueError("at least two examples are required for temporal split")
    ordered = ordered_temporal_indices(examples)
    train_count = max(1, min(len(examples) - 1, int(round(len(examples) * train_fraction))))
    train_indices = sorted(ordered[:train_count])
    eval_indices = sorted(ordered[train_count:])
    return train_indices, eval_indices, _metadata(
        examples,
        split_mode="fraction",
        train_count=len(train_indices),
        eval_count=len(eval_indices),
        train_fraction=train_fraction,
    )


def date_based_split_indices(
    examples: list[dict[str, Any]],
    train_end_date: str,
    eval_start_date: str | None = None,
) -> tuple[list[int], list[int], dict[str, Any]]:
    cutoff = parse_temporal_sort_value(train_end_date)
    if cutoff is None:
        raise ValueError("train_end_date must be parseable")
    eval_start = parse_temporal_sort_value(eval_start_date) if eval_start_date else cutoff
    if eval_start is None:
        raise ValueError("eval_start_date must be parseable")
    train_indices = []
    eval_indices = []
    for index, example in enumerate(examples):
        key, _source = temporal_key_for_example(example)
        if key <= cutoff:
            train_indices.append(index)
        elif key > eval_start:
            eval_indices.append(index)
    if not train_indices or not eval_indices:
        raise ValueError("date-based split requires at least one train and one eval example")
    return sorted(train_indices), sorted(eval_indices), _metadata(
        examples,
        split_mode="date_based",
        train_count=len(train_indices),
        eval_count=len(eval_indices),
        train_end_date=train_end_date,
        eval_start_date=eval_start_date or train_end_date,
    )


def rolling_origin_splits(
    examples: list[dict[str, Any]],
    min_train_count: int,
    eval_window_size: int,
) -> list[dict[str, Any]]:
    if min_train_count <= 0:
        raise ValueError("min_train_count must be positive")
    if eval_window_size <= 0:
        raise ValueError("eval_window_size must be positive")
    ordered = ordered_temporal_indices(examples)
    splits = []
    start = min_train_count
    split_id = 1
    while start < len(ordered):
        end = min(len(ordered), start + eval_window_size)
        train_indices = sorted(ordered[:start])
        eval_indices = sorted(ordered[start:end])
        if eval_indices:
            splits.append(
                {
                    "split_id": split_id,
                    "split_mode": "rolling_origin",
                    "train_indices": train_indices,
                    "eval_indices": eval_indices,
                    "train_count": len(train_indices),
                    "eval_count": len(eval_indices),
                }
            )
            split_id += 1
        start = end
    return splits


def _metadata(examples: list[dict[str, Any]], **extra: Any) -> dict[str, Any]:
    sources = Counter(temporal_key_for_example(example)[1] for example in examples)
    return {
        **extra,
        "temporal_key_sources": dict(sorted(sources.items())),
        "note": (
            "Temporal validation uses true CT.gov date metadata when available and falls back "
            "to NCT numeric registry proxy only when dates are missing."
        ),
    }
