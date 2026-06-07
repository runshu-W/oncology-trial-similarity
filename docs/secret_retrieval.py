from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np


SECRET_SECTIONS = (
    "disease_population",
    "intervention",
    "eligibility",
    "endpoint",
    "design",
    "results",
    "safety_followup",
)

SECRET_SECTION_WEIGHTS = {
    "disease_population": 0.22,
    "intervention": 0.20,
    "eligibility": 0.16,
    "endpoint": 0.18,
    "design": 0.10,
    "results": 0.10,
    "safety_followup": 0.04,
}


def _text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, list):
        return "; ".join(text for item in value if (text := _text(item)))
    if isinstance(value, dict):
        return "; ".join(
            f"{key}: {text}" for key, val in value.items() if (text := _text(val))
        )
    return " ".join(str(value).split())


def _bounded(value: Any, limit: int) -> str:
    text = _text(value)
    if limit <= 0 or len(text) <= limit:
        return text
    return text[:limit].rstrip() + "..."


def _qa(question: str, answer_parts: list[str]) -> str:
    answer = " ".join(part for part in answer_parts if part).strip()
    if not answer:
        answer = "Not reported."
    return f"Q: {question}\nA: {answer}"


def secret_sections_from_summary(
    summary: dict[str, Any], excerpt_char_limit: int = 800
) -> dict[str, str]:
    cancer = summary.get("cancer_type") if isinstance(summary.get("cancer_type"), dict) else {}
    intervention = (
        summary.get("intervention") if isinstance(summary.get("intervention"), dict) else {}
    )
    population = summary.get("population") if isinstance(summary.get("population"), dict) else {}
    endpoints = summary.get("endpoints") if isinstance(summary.get("endpoints"), dict) else {}
    design = summary.get("design") if isinstance(summary.get("design"), dict) else {}
    result = (
        summary.get("result_usability")
        if isinstance(summary.get("result_usability"), dict)
        else {}
    )
    docs = (
        summary.get("supporting_documents")
        if isinstance(summary.get("supporting_documents"), dict)
        else {}
    )
    protocol = _bounded(docs.get("protocol_excerpt"), excerpt_char_limit)
    sap = _bounded(docs.get("sap_excerpt"), excerpt_char_limit)

    primary = endpoints.get("primary") if isinstance(endpoints.get("primary"), list) else []
    endpoint_text = _text(
        [
            {
                "title": item.get("title"),
                "family": item.get("endpoint_family"),
                "time_frame": item.get("time_frame"),
            }
            for item in primary
            if isinstance(item, dict)
        ]
    )

    return {
        "disease_population": _qa(
            "What disease and population does the trial study?",
            [
                _text(cancer),
                _text(population.get("line_of_therapy")),
                _text(summary.get("brief_summary")),
            ],
        ),
        "intervention": _qa(
            "What intervention or regimen is tested?",
            [_text(intervention), _text(summary.get("brief_title"))],
        ),
        "eligibility": _qa(
            "What eligibility criteria define the target population?",
            [
                "Inclusion: " + _text(population.get("key_inclusion")),
                "Exclusion: " + _text(population.get("key_exclusion")),
                ("Protocol excerpt: " + protocol) if protocol else "",
            ],
        ),
        "endpoint": _qa(
            "What primary endpoint and estimand are measured?",
            [endpoint_text],
        ),
        "design": _qa(
            "What design, phase, arm structure, and randomization are used?",
            [_text(summary.get("phase")), _text(design)],
        ),
        "results": _qa(
            "What result quantities are available for borrowing?",
            [_text(result), _text(summary.get("borrowable_quantities"))],
        ),
        "safety_followup": _qa(
            "What safety and follow-up information is available?",
            [_text(summary.get("results")), ("SAP excerpt: " + sap) if sap else ""],
        ),
    }


def cosine(a: np.ndarray, b: np.ndarray) -> float:
    denom = float(np.linalg.norm(a) * np.linalg.norm(b))
    if denom == 0.0:
        return 0.0
    return float(np.dot(a, b) / denom)


def score_secret_index(
    query_vectors: dict[str, np.ndarray],
    index_path: Path,
    summaries: dict[str, dict[str, Any]],
    excluded_nct_id: str,
    top_k: int,
) -> list[dict[str, Any]]:
    index = np.load(index_path, allow_pickle=False)
    nct_ids = index["nct_ids"]
    if top_k <= 0:
        return []

    weighted_scores = np.zeros(len(nct_ids), dtype=np.float32)
    section_score_arrays: dict[str, np.ndarray] = {}
    for section, weight in SECRET_SECTION_WEIGHTS.items():
        query = np.asarray(query_vectors[section], dtype=np.float32).reshape(-1)
        matrix = np.asarray(index[section], dtype=np.float32)
        query_norm = float(np.linalg.norm(query))
        row_norms = np.linalg.norm(matrix, axis=1)
        denom = row_norms * query_norm
        raw_scores = matrix @ query
        similarities = np.divide(
            raw_scores,
            denom,
            out=np.zeros_like(raw_scores, dtype=np.float32),
            where=denom > 0.0,
        )
        section_score_arrays[section] = similarities
        weighted_scores += np.float32(weight) * similarities

    order = np.argsort(-weighted_scores)
    scored = []
    for idx in order:
        nct_id = str(nct_ids[idx])
        if nct_id == excluded_nct_id:
            continue
        section_scores = {
            section: float(scores[idx]) for section, scores in section_score_arrays.items()
        }
        candidate_summary = summaries.get(nct_id, {})
        scored.append(
            {
                "nct_id": nct_id,
                "score": float(weighted_scores[idx]),
                "score_0_100": round(100 * max(0.0, float(weighted_scores[idx])), 2),
                "retrieval_backend": "secret",
                "secret_section_scores": {
                    key: round(value, 4) for key, value in section_scores.items()
                },
                "secret_evidence": secret_sections_from_summary(candidate_summary),
            }
        )
        if len(scored) >= top_k:
            break

    for rank, row in enumerate(scored, start=1):
        row["retrieval_rank"] = rank
    return scored
