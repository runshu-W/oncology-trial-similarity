from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT / "pipeline") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "pipeline"))

import temporal_validation  # noqa: E402


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def attach_temporal_metadata(
    examples_jsonl: Path,
    date_metadata_csv: Path,
    output_jsonl: Path,
    report_json: Path,
) -> dict[str, Any]:
    examples = read_jsonl(examples_jsonl)
    date_metadata = temporal_validation.read_date_metadata_csv(date_metadata_csv)
    report = temporal_validation.attach_date_metadata_to_examples(examples, date_metadata)
    report = {
        **report,
        "examples_jsonl": str(examples_jsonl),
        "date_metadata_csv": str(date_metadata_csv),
        "output_jsonl": str(output_jsonl),
    }
    write_jsonl(output_jsonl, examples)
    report_json.parent.mkdir(parents=True, exist_ok=True)
    report_json.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    return report


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Attach true CT.gov temporal metadata to lambda examples.")
    parser.add_argument("--examples-jsonl", type=Path, required=True)
    parser.add_argument("--date-metadata-csv", type=Path, required=True)
    parser.add_argument(
        "--output-jsonl",
        type=Path,
        default=Path("artifacts/temporal_validation_true_dates/lambda_training_examples_with_true_dates.jsonl"),
    )
    parser.add_argument(
        "--report-json",
        type=Path,
        default=Path("artifacts/temporal_validation_true_dates/lambda_training_examples_date_attachment.json"),
    )
    args = parser.parse_args(argv)
    report = attach_temporal_metadata(
        examples_jsonl=args.examples_jsonl,
        date_metadata_csv=args.date_metadata_csv,
        output_jsonl=args.output_jsonl,
        report_json=args.report_json,
    )
    print(json.dumps(report, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
