from __future__ import annotations

import argparse
import csv
import json
import re
from collections import Counter
from datetime import date, datetime
from pathlib import Path
from typing import Any


DATE_FIELDS = (
    "primary_completion_date",
    "completion_date",
    "results_first_posted_date",
    "start_date",
)


def _nested_get(raw: dict[str, Any], *keys: str) -> Any:
    value: Any = raw
    for key in keys:
        if not isinstance(value, dict):
            return None
        value = value.get(key)
    return value


def _date_text(value: Any) -> str | None:
    if isinstance(value, dict):
        value = value.get("date")
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def _first_present(*values: Any) -> Any:
    for value in values:
        if _date_text(value) is not None:
            return value
    return None


def normalize_clinicaltrials_date(value: Any) -> tuple[str | None, str]:
    """Normalize CT.gov date strings while retaining precision.

    Month-only and year-only dates are intentionally mapped to mid-month and
    mid-year anchors. The precision field makes these approximations explicit
    for sensitivity analyses.
    """
    text = _date_text(value)
    if text is None:
        return None, "missing"
    text = text.strip()

    # Local CT.gov exports often append status qualifiers, e.g.
    # "2019-08-29 (Actual)" or "2022-06 (Estimated)".
    leading_iso_day = re.match(r"^\s*(\d{4}[-/]\d{1,2}[-/]\d{1,2})\b", text)
    if leading_iso_day:
        text = leading_iso_day.group(1)
    else:
        leading_iso_month = re.match(r"^\s*(\d{4}[-/]\d{1,2})\b", text)
        if leading_iso_month:
            text = leading_iso_month.group(1)

    for fmt in ("%Y-%m-%d", "%Y/%m/%d", "%B %d, %Y", "%b %d, %Y"):
        try:
            parsed = datetime.strptime(text, fmt).date()
            return parsed.isoformat(), "day"
        except ValueError:
            pass

    for fmt in ("%B %Y", "%b %Y", "%Y-%m", "%Y/%m"):
        try:
            parsed = datetime.strptime(text, fmt)
            return date(parsed.year, parsed.month, 15).isoformat(), "month"
        except ValueError:
            pass

    year_match = re.search(r"\b(19|20|21)\d{2}\b", text)
    if year_match:
        year = int(year_match.group(0))
        return date(year, 6, 30).isoformat(), "year"

    return None, "unparseable"


def extract_trial_date_metadata(raw: dict[str, Any], nct_id: str | None = None) -> dict[str, Any]:
    status = _nested_get(raw, "protocolSection", "statusModule") or {}
    identification = _nested_get(raw, "protocolSection", "identificationModule") or {}
    legacy_details = raw.get("Study details") if isinstance(raw.get("Study details"), dict) else {}
    legacy_record_dates = _nested_get(raw, "Results Posted", "4. Study Record Dates") or {}

    nct_id = (
        nct_id
        or _date_text(identification.get("nctId"))
        or _date_text(legacy_details.get("1. NCT number"))
        or _date_text(raw.get("nct_id"))
        or ""
    )
    raw_values = {
        "primary_completion_date": _first_present(
            status.get("primaryCompletionDateStruct")
            or status.get("primaryCompletionDate"),
            legacy_details.get("Primary Completion Date"),
            legacy_record_dates.get("Primary Completion Date"),
        ),
        "completion_date": _first_present(
            status.get("completionDateStruct")
            or status.get("completionDate"),
            legacy_details.get("Completion Date"),
            legacy_details.get("Study Completion Date"),
            legacy_record_dates.get("Completion Date"),
            legacy_record_dates.get("Study Completion Date"),
        ),
        "results_first_posted_date": _first_present(
            status.get("resultsFirstPostDateStruct")
            or status.get("resultsFirstSubmitDate")
            or status.get("resultsFirstPostDate"),
            legacy_details.get("Results First Posted"),
            legacy_details.get("Results First Posted Date"),
            legacy_record_dates.get("Results First Posted"),
            legacy_record_dates.get("Results First Posted Date"),
        ),
        "start_date": _first_present(
            status.get("startDateStruct")
            or status.get("startDate"),
            legacy_details.get("Start Date"),
            legacy_details.get("Study Start Date"),
            legacy_record_dates.get("Start Date"),
            legacy_record_dates.get("Study Start Date"),
        ),
    }

    output: dict[str, Any] = {"nct_id": str(nct_id)}
    for field, raw_value in raw_values.items():
        normalized, precision = normalize_clinicaltrials_date(raw_value)
        output[field] = normalized
        output[f"{field}_precision"] = precision
        output[f"{field}_raw"] = _date_text(raw_value)

    temporal_source = "missing"
    temporal_date = None
    for field in ("primary_completion_date", "completion_date", "results_first_posted_date", "start_date"):
        if output.get(field):
            temporal_source = field
            temporal_date = output[field]
            break
    output["temporal_sort_date"] = temporal_date
    output["temporal_sort_source"] = temporal_source
    return output


def date_missingness_report(rows: list[dict[str, Any]]) -> dict[str, Any]:
    field_missing_counts: dict[str, int] = {}
    precision_counts: dict[str, dict[str, int]] = {}
    for field in DATE_FIELDS:
        field_missing_counts[field] = sum(1 for row in rows if not row.get(field))
        precision_counts[field] = dict(
            sorted(Counter(str(row.get(f"{field}_precision") or "missing") for row in rows).items())
        )
    return {
        "trial_count": len(rows),
        "temporal_sort_sources": dict(
            sorted(Counter(str(row.get("temporal_sort_source") or "missing") for row in rows).items())
        ),
        "field_missing_counts": field_missing_counts,
        "precision_counts": precision_counts,
        "note": (
            "Month/year dates are normalized to deterministic anchor dates and retain precision labels; "
            "primary completion date is preferred for temporal validation."
        ),
    }


def iter_trial_jsons(db_root: Path) -> list[Path]:
    return sorted(db_root.glob("NCT*/NCT*_data.json"))


def read_date_rows(db_root: Path) -> list[dict[str, Any]]:
    rows = []
    for path in iter_trial_jsons(db_root):
        with path.open(encoding="utf-8") as handle:
            rows.append(extract_trial_date_metadata(json.load(handle), nct_id=path.parent.name))
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


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Extract true CT.gov temporal validation dates.")
    parser.add_argument("--db-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts/temporal_validation_true_dates"))
    args = parser.parse_args(argv)

    rows = read_date_rows(args.db_root)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(args.output_dir / "clinicaltrials_date_rows.csv", rows)
    report = date_missingness_report(rows)
    (args.output_dir / "clinicaltrials_date_missingness_report.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
