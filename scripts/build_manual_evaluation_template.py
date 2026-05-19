from __future__ import annotations

import csv
import json
import sys
from pathlib import Path


FIELDS = [
    "query_nct_id",
    "query_title",
    "candidate_rank",
    "candidate_nct_id",
    "candidate_title",
    "pipeline_suitability",
    "pipeline_score",
    "pipeline_discount",
    "disease_population_match",
    "treatment_regimen_match",
    "endpoint_estimand_match",
    "design_phase_match",
    "result_usability",
    "red_flags",
    "reviewer_disease_population_0_2",
    "reviewer_regimen_0_2",
    "reviewer_endpoint_0_2",
    "reviewer_design_0_2",
    "reviewer_result_usable_0_2",
    "reviewer_overall_borrowability",
    "reviewer_notes",
]


def build_rows(summary_path: Path) -> list[dict[str, object]]:
    data = json.loads(summary_path.read_text(encoding="utf-8"))
    rows = []
    for result in data["results"]:
        for candidate in result["top3"]:
            dimension_scores = candidate.get("dimension_scores", {})
            rows.append(
                {
                    "query_nct_id": result["query_nct_id"],
                    "query_title": result["query_title"],
                    "candidate_rank": candidate["rank"],
                    "candidate_nct_id": candidate["nct_id"],
                    "candidate_title": candidate["title"],
                    "pipeline_suitability": candidate["suitability"],
                    "pipeline_score": candidate["score"],
                    "pipeline_discount": candidate["discount"],
                    "disease_population_match": dimension_scores.get("disease_population_match", ""),
                    "treatment_regimen_match": dimension_scores.get("treatment_regimen_match", ""),
                    "endpoint_estimand_match": dimension_scores.get("endpoint_estimand_match", ""),
                    "design_phase_match": dimension_scores.get("design_phase_match", ""),
                    "result_usability": dimension_scores.get("result_usability", ""),
                    "red_flags": "; ".join(candidate.get("red_flags") or []),
                    "reviewer_disease_population_0_2": "",
                    "reviewer_regimen_0_2": "",
                    "reviewer_endpoint_0_2": "",
                    "reviewer_design_0_2": "",
                    "reviewer_result_usable_0_2": "",
                    "reviewer_overall_borrowability": "",
                    "reviewer_notes": "",
                }
            )
    return rows


def main() -> None:
    if len(sys.argv) != 3:
        raise SystemExit(
            "Usage: python scripts/build_manual_evaluation_template.py "
            "/path/to/random5_top3_summary.json docs/evaluation/random5_top3_manual_review_template.csv"
        )
    summary_path = Path(sys.argv[1])
    output_path = Path(sys.argv[2])
    rows = build_rows(summary_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    print(f"Wrote {len(rows)} rows to {output_path}")


if __name__ == "__main__":
    main()
