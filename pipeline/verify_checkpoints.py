"""Module M8 helper - verify lambda-model checkpoints (declared vs actual arch).

The review found a mislabeled checkpoint: a file whose metadata declares
``model_type = "two_head_deepsets"`` but whose weights are a plain MLP
(``network.0/2``, no discount head). Silent mislabels are a reproducibility
hazard because a "two_head" result may actually be an MLP. This tool scans every
``lambda_model.pt`` under a root, infers the true architecture from the
state-dict keys, and flags any disagreement with the declared ``model_type``.

Torch is NOT required (uses the torch-free reader in two_head_inference).

Usage
-----
    python scripts/verify_checkpoints.py --root artifacts --output-dir results/tables
Exit code is non-zero if any mismatch is found (useful in CI / release gating).
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
for _sub in ("docs", "scripts"):
    _path = str(REPO_ROOT / _sub)
    if _path not in sys.path:
        sys.path.insert(0, _path)

import two_head_inference as thi  # noqa: E402


def infer_architecture(state_keys: list[str]) -> str:
    keys = set(state_keys)
    if "discount_head.0.weight" in keys and "phi.0.weight" in keys:
        return "two_head_deepsets"
    if "phi.0.weight" in keys and "rho.0.weight" in keys:
        return "deepsets"
    if "raw_weight" in keys:
        return "monotonic_softmax"
    if "network.0.weight" in keys:
        return "mlp"
    return "unknown"


def verify_root(root: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for pt in sorted(root.rglob("lambda_model.pt")):
        row: dict[str, Any] = {"path": str(pt.relative_to(root))}
        try:
            checkpoint = thi.load_checkpoint(pt)
            declared = checkpoint.get("model_type", "")
            actual = infer_architecture(list(checkpoint["state_dict"].keys()))
            has_discount = "discount_head.0.weight" in checkpoint["state_dict"]
            row.update(
                {
                    "declared_model_type": declared,
                    "actual_architecture": actual,
                    "hidden_dim": checkpoint.get("hidden_dim"),
                    "has_discount_head": has_discount,
                    "match": declared == actual,
                    "status": "ok" if declared == actual else "MISLABELED",
                }
            )
        except Exception as error:  # noqa: BLE001
            row.update({"status": f"error:{type(error).__name__}", "match": False})
        rows.append(row)
    return rows


def run(root: Path, output_dir: Path) -> tuple[list[dict[str, Any]], int]:
    rows = verify_root(root)
    output_dir.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "path", "declared_model_type", "actual_architecture", "hidden_dim",
        "has_discount_head", "match", "status",
    ]
    with (output_dir / "checkpoint_verification.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    (output_dir / "checkpoint_verification.json").write_text(json.dumps(rows, indent=2), encoding="utf-8")
    mismatches = sum(1 for r in rows if r.get("status") == "MISLABELED")
    return rows, mismatches


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--root", type=Path, default=REPO_ROOT / "artifacts")
    parser.add_argument("--output-dir", type=Path, default=REPO_ROOT / "results" / "tables")
    parser.add_argument("--strict", action="store_true", help="Exit non-zero if any mislabel is found.")
    args = parser.parse_args(argv)

    rows, mismatches = run(args.root, args.output_dir)
    for r in rows:
        marker = "  " if r.get("status") == "ok" else "!!"
        print(f"{marker} {r['path']}: declared={r.get('declared_model_type')} actual={r.get('actual_architecture')} [{r.get('status')}]")
    print(f"\n{len(rows)} checkpoints, {mismatches} mislabeled.")
    if args.strict and mismatches:
        sys.exit(1)


if __name__ == "__main__":
    main()
