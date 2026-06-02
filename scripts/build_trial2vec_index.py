from __future__ import annotations

import argparse
import inspect
import json
import sys
from pathlib import Path
from typing import Any, Callable

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT / "docs") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "docs"))

import oncology_trial_similarity_pipeline as pipeline  # noqa: E402


def load_summary_rows(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def make_compatible_torch_load(original_torch_load: Callable[..., Any]) -> Callable[..., Any]:
    try:
        supports_weights_only = "weights_only" in inspect.signature(original_torch_load).parameters
    except (TypeError, ValueError):
        supports_weights_only = False

    def compatible_torch_load(*args: Any, **kwargs: Any) -> Any:
        if supports_weights_only and "weights_only" not in kwargs:
            kwargs["weights_only"] = False
        return original_torch_load(*args, **kwargs)

    return compatible_torch_load


def make_compatible_load_state_dict(original_load_state_dict: Callable[..., Any]) -> Callable[..., Any]:
    def compatible_load_state_dict(
        module: Any,
        state_dict: dict[str, Any],
        *args: Any,
        **kwargs: Any,
    ) -> Any:
        if not args and "strict" not in kwargs:
            kwargs["strict"] = False
        return original_load_state_dict(module, state_dict, *args, **kwargs)

    return compatible_load_state_dict


def load_trial2vec_index_dependencies() -> tuple[Any, Any, Any]:
    try:
        import pandas as pd
        import torch
        from trial2vec import Trial2Vec
    except ImportError as exc:
        raise RuntimeError(
            "Building a Trial2Vec index requires optional Trial2Vec index dependencies: "
            "pandas, torch, and trial2vec."
        ) from exc
    return pd, torch, Trial2Vec


def encode_trial2vec_index(
    summaries_path: Path,
    output_path: Path,
    model_dir: Path,
    device: str = "cpu",
) -> dict[str, Any]:
    pd, torch, Trial2Vec = load_trial2vec_index_dependencies()

    summaries = load_summary_rows(summaries_path)
    frame = pd.DataFrame([pipeline.summary_to_trial2vec_row(summary) for summary in summaries])
    model = Trial2Vec(device=device)

    original_torch_load = torch.load
    original_load_state_dict = torch.nn.Module.load_state_dict

    torch.load = make_compatible_torch_load(original_torch_load)
    torch.nn.Module.load_state_dict = make_compatible_load_state_dict(original_load_state_dict)
    try:
        model.from_pretrained(str(model_dir))
    finally:
        torch.load = original_torch_load
        torch.nn.Module.load_state_dict = original_load_state_dict

    tags, embeddings = model.encode({"x": frame}, return_dict=False)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output_path,
        nct_ids=np.array(tags),
        embeddings=embeddings.astype(np.float32),
        retrieval_backend=np.array(["trial2vec"]),
        model_dir=np.array([str(model_dir)]),
    )
    return {
        "summary_count": len(summaries),
        "embedding_shape": list(embeddings.shape),
        "output_path": str(output_path),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Build a Trial2Vec retrieval index from trial_summaries.jsonl.")
    parser.add_argument("--summaries-path", type=Path, required=True)
    parser.add_argument("--output-path", type=Path, required=True)
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()
    summary = encode_trial2vec_index(args.summaries_path, args.output_path, args.model_dir, device=args.device)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
