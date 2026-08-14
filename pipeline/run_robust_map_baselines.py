"""Module M9 - External robust-prior baselines (RBesT-style robust-MAP).

Reviewers of a Bayesian borrowing paper will ask "why not just use a robust
meta-analytic-predictive (MAP) prior?" (Schmidli et al. 2014). This module adds
a genuine robust-MAP baseline and benchmarks it head-to-head against the paper's
two_head / two_head_sam methods on identical held-out beta-binomial predictive
NLL and 95% interval coverage. No expert labels are used.

Robust-MAP construction (per pseudo-query, from its historical components):
  1. Build a meta-analytic-predictive (MAP) Beta for the new trial's rate:
       - pooled mean m from the historical (y_i, n_i);
       - between-trial heterogeneity tau^2 via a DerSimonian-Laird moment
         estimator (k >= 2) or a single-source fallback (k = 1);
       - MAP effective sample size ESS_map = m(1-m)/(tau^2 + eps) - 1, capped;
       - MAP Beta(a_map, b_map) with mean m and that ESS.
  2. Robustify: prior = (1 - w) * Beta(1,1) + w * MAP, i.e. a two-component
     mixture with robustification weight ``w`` on the informative MAP component.

This is expressed in the SAME mixture form the rest of the code uses, so it is
evaluated through the identical predictive pipeline as every other method.

The comparison methods (weak_only, rule, rule_sam, two_head, two_head_sam,
map_like, power_prior_like, commensurate_like) are scored via the shared
calibration helpers so the head-to-head table is apples-to-apples.

Example
-------
    python scripts/run_robust_map_baselines.py \
        --examples-jsonl artifacts/twohead_deepsets_examples_with_model.jsonl \
        --output-dir artifacts/robust_map_orr_all \
        --robust-weights 0.5 0.8 0.9 \
        --compare-methods weak_only rule rule_sam two_head two_head_sam map_like power_prior_like commensurate_like
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
for _sub in ("docs", "scripts"):
    _path = str(REPO_ROOT / _sub)
    if _path not in sys.path:
        sys.path.insert(0, _path)

import mixture_prior  # noqa: E402
import run_borrowing_baseline_comparison as baseline  # noqa: E402
import run_calibration_diagnostics as calib  # noqa: E402

DEFAULT_COMPARE = (
    "weak_only", "rule", "rule_sam", "two_head", "two_head_sam",
    "map_like", "power_prior_like", "commensurate_like",
)
ESS_CAP = 200.0
EPS = 1e-4


# --------------------------------------------------------------------------- #
# Robust-MAP prior construction
# --------------------------------------------------------------------------- #
def _historical_rates(example: dict[str, Any]) -> list[tuple[float, float]]:
    """Return [(y_i, n_i)] for components with usable denominators."""
    pairs = []
    for component in example.get("components") or []:
        count, denom = baseline._component_count_denominator(component)
        if denom > 0 and 0.0 <= count <= denom:
            pairs.append((count, denom))
    return pairs


def map_beta(pairs: list[tuple[float, float]]) -> tuple[float, float, float]:
    """Return (alpha_map, beta_map, ess_map) for a meta-analytic-predictive Beta."""
    if not pairs:
        return 1.0, 1.0, 0.0
    ns = [n for _, n in pairs]
    ps = [(y + 0.5) / (n + 1.0) for y, n in pairs]  # Haldane-smoothed rates
    total_n = sum(ns)
    m = sum(y for y, _ in pairs) / total_n  # pooled event rate
    m = min(max(m, 1e-4), 1 - 1e-4)

    if len(pairs) == 1:
        # Single historical source: MAP = that trial, ESS = its denominator (capped).
        ess = min(ns[0], ESS_CAP)
        return m * ess + EPS, (1 - m) * ess + EPS, ess

    # DerSimonian-Laird between-trial heterogeneity on the rate scale.
    within_var = m * (1 - m)
    weights = [n / within_var for n in ns]  # inverse within-trial variance
    sw = sum(weights)
    q = sum(w * (p - m) ** 2 for w, p in zip(weights, ps))
    df = len(pairs) - 1
    c = sw - sum(w * w for w in weights) / sw
    tau2 = max(0.0, (q - df) / c) if c > 0 else 0.0

    predictive_var = tau2 + EPS
    ess = min(max(within_var / predictive_var - 1.0, 1.0), ESS_CAP)
    return m * ess + EPS, (1 - m) * ess + EPS, ess


def robust_map_prior(example: dict[str, Any], robust_weight: float) -> dict[str, Any]:
    """Robust-MAP mixture prior: (1-w) Beta(1,1) + w MAP."""
    pairs = _historical_rates(example)
    alpha_map, beta_map, ess_map = map_beta(pairs)
    if not pairs or ess_map <= 0:
        return {"lambda_0": 1.0, "components": [], "ess_map": 0.0}
    return {
        "lambda_0": 1.0 - robust_weight,
        "components": [{"alpha": alpha_map, "beta": beta_map, "lambda": robust_weight}],
        "ess_map": ess_map,
    }


def robust_map_pmf(example: dict[str, Any], robust_weight: float) -> tuple[int, int, list[float], float]:
    y = int(float(example["query"]["count"]))
    n = int(float(example["query"]["denominator"]))
    prior = robust_map_prior(example, robust_weight)
    pmf = [
        mixture_prior.mixture_predictive_probability(
            k, n, float(prior["lambda_0"]), 1.0, 1.0, prior["components"]
        )
        for k in range(n + 1)
    ]
    total = math.fsum(pmf)
    if total > 0:
        pmf = [v / total for v in pmf]
    return y, n, pmf, prior["ess_map"]


# --------------------------------------------------------------------------- #
# Evaluation
# --------------------------------------------------------------------------- #
def _mean(values: list[float]) -> float:
    finite = [v for v in values if v is not None and math.isfinite(v)]
    return math.fsum(finite) / len(finite) if finite else float("nan")


def evaluate(
    examples_jsonl: Path,
    output_dir: Path,
    robust_weights: tuple[float, ...],
    compare_methods: tuple[str, ...],
    fixed_discount: float,
) -> dict[str, Any]:
    examples = calib.read_jsonl(examples_jsonl)
    output_dir.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, Any]] = []

    # Robust-MAP at each robustification weight.
    for w in robust_weights:
        nlls, covers, esss = [], [], []
        for example in examples:
            try:
                y, n, pmf, ess = robust_map_pmf(example, w)
            except Exception:  # noqa: BLE001
                continue
            if n <= 0 or not (0 <= y <= n):
                continue
            nlls.append(-math.log(max(pmf[y], 1e-300)))
            lo, hi = calib.central_interval(pmf, 0.95)
            covers.append(int(lo <= y <= hi))
            esss.append(ess)
        rows.append(
            {
                "method": f"robust_map_w{w:g}",
                "family": "robust_map",
                "n": len(nlls),
                "mean_nll": round(_mean(nlls), 6),
                "coverage95": round(_mean(covers), 6),
                "mean_ess": round(_mean(esss), 4),
            }
        )

    # Comparison methods through the identical predictive pipeline.
    for method in compare_methods:
        nlls, covers = [], []
        for example in examples:
            try:
                y, n, pmf = calib.predictive_pmf(example, method, fixed_discount=fixed_discount)
            except Exception:  # noqa: BLE001
                continue
            if n <= 0 or not (0 <= y <= n):
                continue
            nlls.append(-math.log(max(pmf[y], 1e-300)))
            lo, hi = calib.central_interval(pmf, 0.95)
            covers.append(int(lo <= y <= hi))
        rows.append(
            {
                "method": method,
                "family": "learned" if method.startswith("two_head") else "other",
                "n": len(nlls),
                "mean_nll": round(_mean(nlls), 6),
                "coverage95": round(_mean(covers), 6),
                "mean_ess": "",
            }
        )

    rows.sort(key=lambda r: (r["mean_nll"] if isinstance(r["mean_nll"], float) and math.isfinite(r["mean_nll"]) else 1e9))
    _write_csv(output_dir / "robust_map_head_to_head.csv", rows)
    _write_markdown(output_dir / "robust_map_head_to_head.md", rows)
    payload = {
        "examples_jsonl": str(examples_jsonl),
        "n_examples": len(examples),
        "robust_weights": list(robust_weights),
        "compare_methods": list(compare_methods),
        "rows": rows,
    }
    (output_dir / "robust_map_head_to_head.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return payload


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ["method", "family", "n", "mean_nll", "coverage95", "mean_ess"]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def _write_markdown(path: Path, rows: list[dict[str, Any]]) -> None:
    lines = [
        "# Robust-MAP head-to-head (label-free held-out predictive NLL)",
        "",
        "Genuine RBesT-style robust-MAP (meta-analytic-predictive Beta + vague, "
        "robustification weight w) benchmarked against the paper's two_head / "
        "two_head_sam and other priors on identical held-out beta-binomial NLL "
        "and 95% coverage. Sorted best NLL first.",
        "",
        "| Method | Family | N | Mean NLL | Cov95 | MAP ESS |",
        "|---|---|---:|---:|---:|---:|",
    ]
    for r in rows:
        ess = r["mean_ess"] if r["mean_ess"] != "" else "-"
        lines.append(
            "| {m} | {fam} | {n} | {nll:.4f} | {cov:.4f} | {ess} |".format(
                m=r["method"], fam=r["family"], n=r["n"], nll=r["mean_nll"], cov=r["coverage95"], ess=ess
            )
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--examples-jsonl", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts/robust_map"))
    parser.add_argument("--robust-weights", nargs="+", type=float, default=[0.5, 0.8, 0.9])
    parser.add_argument("--compare-methods", nargs="+", default=list(DEFAULT_COMPARE))
    parser.add_argument("--fixed-discount", type=float, default=0.25)
    args = parser.parse_args(argv)

    payload = evaluate(
        examples_jsonl=args.examples_jsonl,
        output_dir=args.output_dir,
        robust_weights=tuple(args.robust_weights),
        compare_methods=tuple(args.compare_methods),
        fixed_discount=args.fixed_discount,
    )
    print(json.dumps({"rows": payload["rows"]}, indent=2))


if __name__ == "__main__":
    main()
