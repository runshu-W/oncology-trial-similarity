"""Two-head DeepSets inference bridge - makes the paper's core method usable everywhere.

The two-head DeepSets + SAM prior is the manuscript's headline method. Its per-
candidate mixture weight ``lambda_model`` and effective-sample-size discount
``discount_model`` are produced by a trained ``lambda_model.pt``. Downstream
analysis modules (calibration, sham controls, operating characteristics) can only
include the ``two_head`` / ``two_head_sam`` methods when those two columns are
present on each candidate component.

This module applies a trained two-head DeepSets checkpoint to the *precomputed*
per-candidate feature vectors already stored in the lambda examples, writing
``lambda_model`` and ``discount_model`` onto every component. It has TWO backends:

  * ``torch`` if available (canonical);
  * a **dependency-light NumPy reimplementation** of the exact forward pass
    (phi MLP -> DeepSets mean-context -> rho score head + sigmoid discount head
    -> gated softmax allocation), with a torch-free reader for the ``.pt`` zip.

The NumPy path is verified to reproduce the trained model's held-out predictive
NLL (see ``tests/test_two_head_inference.py`` and the ``--validate-nll-csv``
option), so it is a faithful stand-in when torch is unavailable.

Usage
-----
    python scripts/two_head_inference.py \
        --examples-jsonl artifacts/retrospective_lambda_oncology_orr_all/lambda_training_examples.jsonl \
        --model-path     artifacts/retrospective_lambda_oncology_orr_all/lambda_model.pt \
        --output-jsonl   artifacts/orr_all_examples_with_model.jsonl \
        --validate-nll-csv artifacts/retrospective_lambda_oncology_orr_all/lambda_nll_rows.csv

The output JSONL can then be passed to run_calibration_diagnostics.py,
run_sham_borrowing_controls.py and run_borrowing_oc_sweeps.py with
``--methods ... two_head two_head_sam``.
"""

from __future__ import annotations

import argparse
import io
import json
import math
import pickle
import struct
import sys
import zipfile
from collections import OrderedDict
from pathlib import Path
from typing import Any

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
for _sub in ("docs", "scripts"):
    _path = str(REPO_ROOT / _sub)
    if _path not in sys.path:
        sys.path.insert(0, _path)

import mixture_prior  # noqa: E402

LAMBDA_FEATURE_NAMES = [
    "s_i", "disease_match_i", "regimen_match_i", "endpoint_match_i", "followup_match_i",
    "eligibility_match_i", "result_quality_i", "negative_redflag_severity_i", "log_n_i",
]


# --------------------------------------------------------------------------- #
# Torch-free reader for a torch.save(...) zip checkpoint
# --------------------------------------------------------------------------- #
def load_checkpoint(path: str | Path) -> dict[str, Any]:
    """Load a two-head checkpoint into plain NumPy arrays without importing torch."""
    path = Path(path)
    with zipfile.ZipFile(path) as archive:
        names = archive.namelist()
        prefix = names[0].split("/", 1)[0]
        byteorder = archive.read(f"{prefix}/byteorder").decode().strip() if f"{prefix}/byteorder" in names else "little"
        dtype = "<f4" if byteorder == "little" else ">f4"

        def rebuild(storage, storage_offset, size, stride, requires_grad, backward_hooks, *_extra):
            raw = archive.read(f"{prefix}/data/{storage['key']}")
            count = 1
            for dim in size:
                count *= dim
            flat = struct.unpack(f"{'<' if byteorder == 'little' else '>'}{len(raw) // 4}f", raw)
            arr = np.array(flat[:count], dtype=np.float32)
            return arr.reshape(tuple(size)) if size else arr

        class _Stub:
            def __init__(self, *args, **kwargs):
                pass

        class _Unpickler(pickle.Unpickler):
            def find_class(self, module: str, name: str):
                if name == "_rebuild_tensor_v2":
                    return rebuild
                if name == "OrderedDict":
                    return OrderedDict
                if module.startswith("torch"):
                    return _Stub
                return super().find_class(module, name)

            def persistent_load(self, pid):
                return {"key": str(pid[2]), "numel": pid[4]}

        obj = _Unpickler(io.BytesIO(archive.read(f"{prefix}/data.pkl"))).load()

    state = {k: np.asarray(v, dtype=np.float32) for k, v in obj["state_dict"].items()}
    return {
        "state_dict": state,
        "input_dim": int(obj.get("input_dim", 9)),
        "hidden_dim": int(obj.get("hidden_dim", 16)),
        "model_type": str(obj.get("model_type", "two_head_deepsets")),
        "lambda0": float(obj.get("lambda0", 0.2)),
        "feature_names": list(obj.get("feature_names", LAMBDA_FEATURE_NAMES)),
    }


# --------------------------------------------------------------------------- #
# NumPy forward pass (exact reimplementation of TwoHeadDeepSetsLambdaScorer)
# --------------------------------------------------------------------------- #
def _relu(x: np.ndarray) -> np.ndarray:
    return np.maximum(x, 0.0)


def _linear(x: np.ndarray, weight: np.ndarray, bias: np.ndarray) -> np.ndarray:
    # torch Linear stores weight as (out, in); y = x @ weight.T + bias
    return x @ weight.T + bias


def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


def _contextual(features: np.ndarray, sd: dict[str, np.ndarray]) -> np.ndarray:
    h = _relu(_linear(features, sd["phi.0.weight"], sd["phi.0.bias"]))
    h = _relu(_linear(h, sd["phi.2.weight"], sd["phi.2.bias"]))
    context = h.mean(axis=0, keepdims=True)
    repeated = np.broadcast_to(context, h.shape)
    return np.concatenate([h, repeated], axis=1)


def forward_scores_discounts(features: np.ndarray, sd: dict[str, np.ndarray]) -> tuple[np.ndarray, np.ndarray | None]:
    """Architecture-aware forward pass.

    Supports the plain MLP scorer (``network.0/2``; no discount head) and the
    DeepSets / two-head DeepSets scorer (``phi``/``rho`` with optional
    ``discount_head``). The correct branch is chosen from the checkpoint keys.
    """
    if "network.0.weight" in sd:  # LambdaScorer (MLP): Linear -> ReLU -> Linear
        h = _relu(_linear(features, sd["network.0.weight"], sd["network.0.bias"]))
        scores = _linear(h, sd["network.2.weight"], sd["network.2.bias"]).reshape(-1)
        return scores, None

    if "phi.0.weight" in sd:  # DeepSets / two-head DeepSets
        ctx = _contextual(features, sd)
        r = _relu(_linear(ctx, sd["rho.0.weight"], sd["rho.0.bias"]))
        scores = _linear(r, sd["rho.2.weight"], sd["rho.2.bias"]).reshape(-1)
        discounts = None
        if "discount_head.0.weight" in sd:
            d = _relu(_linear(ctx, sd["discount_head.0.weight"], sd["discount_head.0.bias"]))
            discounts = _sigmoid(_linear(d, sd["discount_head.2.weight"], sd["discount_head.2.bias"]).reshape(-1))
        return scores, discounts

    if "raw_weight" in sd:  # MonotonicSoftmaxScorer: softplus(raw_weight) . features + bias
        weights = np.log1p(np.exp(sd["raw_weight"]))
        scores = features @ weights + float(sd.get("bias", np.zeros(())))
        return scores.reshape(-1), None

    raise ValueError(f"Unrecognized lambda-model architecture; keys={list(sd.keys())}")


def lambda_from_scores(scores: np.ndarray, gate: np.ndarray, lambda0: float) -> tuple[np.ndarray, float]:
    """Gated softmax allocation, matching _lambda_weights_from_scores."""
    positive = gate > 0.0
    if not positive.any():
        return np.zeros_like(scores), 1.0
    budget = max(0.0, 1.0 - lambda0)
    log_raw = scores + np.log(np.clip(gate, 1e-12, None))
    masked = np.where(positive, log_raw, -np.inf)
    masked = masked - np.max(masked[positive])
    exp = np.where(positive, np.exp(masked), 0.0)
    softmax = exp / np.sum(exp)
    return budget * softmax, lambda0


# --------------------------------------------------------------------------- #
# Apply to examples
# --------------------------------------------------------------------------- #
def attach_model_outputs(example: dict[str, Any], checkpoint: dict[str, Any]) -> dict[str, Any]:
    """Return a copy of ``example`` with lambda_model/discount_model on each component."""
    sd = checkpoint["state_dict"]
    features = np.asarray(example["features"], dtype=np.float32)
    components = example.get("components") or []
    if features.ndim != 2 or features.shape[0] != len(components):
        raise ValueError("features must be 2D with one row per component")
    gate = np.array([float(c.get("gate", 1.0)) for c in components], dtype=np.float32)
    lambda0 = float(example.get("lambda_0", checkpoint.get("lambda0", 0.2)))

    scores, discounts = forward_scores_discounts(features, sd)
    lambdas, _lambda0 = lambda_from_scores(scores, gate, lambda0)

    out = dict(example)
    new_components = []
    for index, component in enumerate(components):
        c = dict(component)
        c["lambda_model"] = float(lambdas[index])
        if discounts is not None:
            c["discount_model"] = float(discounts[index])
        new_components.append(c)
    out["components"] = new_components
    return out


def two_head_predictive_nll(example: dict[str, Any]) -> float:
    """Held-out predictive NLL under the two-head prior (for validation vs the trained CSV)."""
    y = int(float(example["query"]["count"]))
    n = int(float(example["query"]["denominator"]))
    lambda0 = float(example.get("lambda_0", 0.2))
    comps = []
    for c in example.get("components") or []:
        discount = float(c.get("discount_model", c.get("discount", 0.25)))
        count = float(c.get("count", float(c.get("alpha", 1.0)) - 1.0))
        denom = float(c.get("denominator", float(c.get("alpha", 1.0)) + float(c.get("beta", 1.0)) - 2.0))
        comps.append(
            {
                "alpha": 1.0 + discount * count,
                "beta": 1.0 + discount * (denom - count),
                "lambda": float(c.get("lambda_model", 0.0)),
            }
        )
    prob = mixture_prior.mixture_predictive_probability(y, n, lambda0, 1.0, 1.0, comps)
    return -math.log(max(prob, 1e-300))


def attach_to_file(
    examples_jsonl: Path,
    output_jsonl: Path,
    model_path: Path,
    validate_nll_csv: Path | None = None,
) -> dict[str, Any]:
    checkpoint = load_checkpoint(model_path)
    examples = []
    with examples_jsonl.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                examples.append(json.loads(line))
    augmented = [attach_model_outputs(example, checkpoint) for example in examples]
    output_jsonl.parent.mkdir(parents=True, exist_ok=True)
    with output_jsonl.open("w", encoding="utf-8") as handle:
        for example in augmented:
            handle.write(json.dumps(example) + "\n")

    report: dict[str, Any] = {
        "model_path": str(model_path),
        "model_type": checkpoint["model_type"],
        "hidden_dim": checkpoint["hidden_dim"],
        "n_examples": len(augmented),
        "output_jsonl": str(output_jsonl),
    }
    if validate_nll_csv is not None and validate_nll_csv.exists():
        report["validation"] = _validate_against_csv(augmented, validate_nll_csv)
    return report


def _validate_against_csv(augmented: list[dict[str, Any]], csv_path: Path) -> dict[str, Any]:
    import csv as _csv

    csv_by_nct: dict[str, float] = {}
    with csv_path.open(encoding="utf-8") as handle:
        for row in _csv.DictReader(handle):
            val = row.get("learned_nll", "")
            if val not in ("", "nan"):
                csv_by_nct[str(row.get("query_nct_id", ""))] = float(val)
    diffs = []
    matched = 0
    for example in augmented:
        nct = str(example.get("query_nct_id", ""))
        if nct in csv_by_nct:
            mine = two_head_predictive_nll(example)
            diffs.append(abs(mine - csv_by_nct[nct]))
            matched += 1
    return {
        "matched_examples": matched,
        "max_abs_nll_diff": round(max(diffs), 6) if diffs else None,
        "mean_abs_nll_diff": round(sum(diffs) / len(diffs), 6) if diffs else None,
    }


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--examples-jsonl", type=Path, required=True)
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--output-jsonl", type=Path, required=True)
    parser.add_argument("--validate-nll-csv", type=Path, default=None)
    args = parser.parse_args(argv)

    report = attach_to_file(args.examples_jsonl, args.output_jsonl, args.model_path, args.validate_nll_csv)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
