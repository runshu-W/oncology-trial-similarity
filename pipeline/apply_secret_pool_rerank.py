import argparse
import copy
import json
import math
import sys
from pathlib import Path
from typing import Any

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT / "pipeline") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "pipeline"))

import oncology_trial_similarity_pipeline as pipeline  # noqa: E402
import secret_retrieval  # noqa: E402


def _cosine_batch(query: np.ndarray, matrix: np.ndarray) -> np.ndarray:
    query = np.asarray(query, dtype=np.float32).reshape(-1)
    matrix = np.asarray(matrix, dtype=np.float32)
    query_norm = float(np.linalg.norm(query))
    row_norms = np.linalg.norm(matrix, axis=1)
    denom = row_norms * query_norm
    raw = matrix @ query
    return np.divide(
        raw,
        denom,
        out=np.zeros_like(raw, dtype=np.float32),
        where=denom > 0.0,
    )


def score_pool_with_secret_sections(
    query_summary: dict[str, Any],
    candidates: list[dict[str, Any]],
    embedder: pipeline.TextEmbedder | None = None,
) -> list[dict[str, Any]]:
    if not candidates:
        return []
    embedder = embedder or pipeline.HashingEmbedder()
    query_sections = secret_retrieval.secret_sections_from_summary(
        pipeline.secret_ready_summary(query_summary)
    )
    candidate_sections = [
        secret_retrieval.secret_sections_from_summary(pipeline.secret_ready_summary(candidate))
        for candidate in candidates
    ]
    weighted_scores = np.zeros(len(candidates), dtype=np.float32)
    section_scores: dict[str, np.ndarray] = {}

    for section, weight in secret_retrieval.SECRET_SECTION_WEIGHTS.items():
        query_vector = embedder.encode([query_sections[section]])[0]
        candidate_vectors = embedder.encode(
            [sections[section] for sections in candidate_sections]
        )
        similarities = _cosine_batch(query_vector, candidate_vectors)
        section_scores[section] = similarities
        weighted_scores += np.float32(weight) * similarities

    scored = []
    for index, candidate in enumerate(candidates):
        row = copy.deepcopy(candidate)
        row["original_retrieval_backend"] = candidate.get("retrieval_backend", "")
        row["retrieval_backend"] = "secret_pool_rerank"
        row["secret_pool_score"] = float(weighted_scores[index])
        row["secret_pool_score_0_100"] = round(100 * max(0.0, float(weighted_scores[index])), 2)
        row["secret_section_scores"] = {
            section: round(float(scores[index]), 4)
            for section, scores in section_scores.items()
        }
        scored.append(row)

    scored.sort(key=lambda row: row["secret_pool_score"], reverse=True)
    for rank, row in enumerate(scored, start=1):
        row["retrieval_rank"] = rank
    return scored


def rerank_pipeline_result(
    result: dict[str, Any],
    top_k: int | None = None,
    rerank_top_n: int = 10,
    embedder: pipeline.TextEmbedder | None = None,
) -> dict[str, Any]:
    if rerank_top_n <= 0:
        raise ValueError("rerank_top_n must be greater than 0")
    query_summary = result.get("query_summary")
    if not isinstance(query_summary, dict):
        raise ValueError("pipeline result must include query_summary")
    candidates = result.get("top_matches")
    if not isinstance(candidates, list):
        raise ValueError("pipeline result must include top_matches")

    scored = score_pool_with_secret_sections(query_summary, candidates, embedder=embedder)
    if top_k is None:
        top_k = len(scored)
    if top_k <= 0:
        raise ValueError("top_k must be greater than 0")
    top_matches = scored[:top_k]
    rerank_input = scored[: max(rerank_top_n, top_k)]
    reranked = pipeline.rerank_candidates(query_summary, rerank_input, rerank_top_n)

    output = copy.deepcopy(result)
    output["retrieval_backend"] = "secret_pool_rerank"
    output["embedding_backend"] = getattr(embedder or pipeline.HashingEmbedder(), "backend_name", "hashing")
    output["embedding_model"] = getattr(
        embedder or pipeline.HashingEmbedder(),
        "model_name",
        "signed-token-hashing-2048",
    )
    output["top_matches"] = top_matches
    output["top10"] = top_matches[: min(10, len(top_matches))]
    output["reranked_top_matches"] = reranked
    output["reranked_top10"] = reranked[:10]
    if "bayesian_analysis" in output:
        output.pop("bayesian_analysis")
        output = pipeline.add_bayesian_analysis(output)
    return output


def transform_jsonl(
    input_jsonl: Path,
    output_jsonl: Path,
    top_k: int | None,
    rerank_top_n: int,
    max_results: int | None = None,
) -> dict[str, Any]:
    output_jsonl.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    embedder = pipeline.HashingEmbedder()
    with input_jsonl.open(encoding="utf-8") as source, output_jsonl.open("w", encoding="utf-8") as sink:
        for line in source:
            if not line.strip():
                continue
            result = rerank_pipeline_result(
                json.loads(line),
                top_k=top_k,
                rerank_top_n=rerank_top_n,
                embedder=embedder,
            )
            sink.write(json.dumps(result, ensure_ascii=False, allow_nan=False) + "\n")
            count += 1
            if max_results is not None and count >= max_results:
                break
    return {
        "input_jsonl": str(input_jsonl),
        "output_jsonl": str(output_jsonl),
        "result_count": count,
        "top_k": top_k,
        "rerank_top_n": rerank_top_n,
    }


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-jsonl", type=Path, required=True)
    parser.add_argument("--output-jsonl", type=Path, required=True)
    parser.add_argument("--top-k", type=int, default=None)
    parser.add_argument("--rerank-top-n", type=int, default=10)
    parser.add_argument("--max-results", type=int, default=None)
    args = parser.parse_args(argv)

    summary = transform_jsonl(
        input_jsonl=args.input_jsonl,
        output_jsonl=args.output_jsonl,
        top_k=args.top_k,
        rerank_top_n=args.rerank_top_n,
        max_results=args.max_results,
    )
    print(json.dumps(summary, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
