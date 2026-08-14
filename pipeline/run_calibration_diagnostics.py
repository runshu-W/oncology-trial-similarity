"""Module M4 - Label-free calibration diagnostics for Bayesian borrowing priors.

This script quantifies HOW WELL each borrowing prior predicts the real held-out
pseudo-query endpoint outcome (y0 successes in n0 patients), using only the
observed count as ground truth. No expert labels are required.

For every held-out example and every borrowing method it reconstructs the exact
same mixture prior used by ``run_borrowing_baseline_comparison`` (including the
optional SAM conflict adapter), evaluates the *full* beta-binomial mixture
predictive PMF over y in {0, ..., n0}, and derives:

  * Probability Integral Transform (PIT), randomized and non-randomized.
    Under a calibrated predictive distribution the PIT is Uniform(0, 1).
  * Central predictive-interval coverage at the 50 / 80 / 95% levels
    (a genuine label-free validity check: does a nominal 95% interval contain
    the real outcome 95% of the time?).
  * Sharpness (mean interval width as a fraction of n0) at matched coverage.
  * A reliability diagram: predicted mean response rate vs observed rate,
    binned by predicted rate.
  * Kolmogorov-Smirnov distance of the PIT to Uniform(0, 1).

Outputs (all written to ``--output-dir``):
  calibration_per_example.csv         one row per (method, example)
  calibration_summary.csv             one row per method
  calibration_summary.json            machine-readable summary
  calibration_summary.md              markdown table for the manuscript
  calibration_reliability.csv         reliability-diagram bins per method
  pit_hist_<method>.svg               PIT histogram per method
  reliability_<method>.svg            reliability diagram per method

This module has NO third-party dependencies beyond the repository's own
``mixture_prior`` and ``run_borrowing_baseline_comparison`` modules
(numpy/torch not required), so it runs anywhere the pure-Python stats layer runs.

Example
-------
    python scripts/run_calibration_diagnostics.py \
        --examples-jsonl artifacts/retrospective_lambda_oncology_orr_all/lambda_training_examples.jsonl \
        --output-dir artifacts/calibration_diagnostics_orr_all \
        --methods weak_only rule rule_sam two_head two_head_sam map_like power_prior_like \
        --seed 20260707
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import random
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

DEFAULT_METHODS = (
    "weak_only",
    "rule",
    "rule_sam",
    "map_like",
    "power_prior_like",
    "commensurate_like",
    "fixed_discount",
)
COVERAGE_LEVELS = (0.50, 0.80, 0.95)


# --------------------------------------------------------------------------- #
# Predictive distribution
# --------------------------------------------------------------------------- #
def predictive_pmf(
    example: dict[str, Any],
    method: str,
    fixed_discount: float = 0.25,
) -> tuple[int, int, list[float]]:
    """Return (y_obs, n, pmf) where pmf[y] = P(Y = y | n, prior) under ``method``.

    The prior is reconstructed with the *same* code path as the head-to-head
    baseline so the diagnostics describe exactly the priors reported elsewhere.
    """
    y_obs = int(float(example["query"]["count"]))
    n = int(float(example["query"]["denominator"]))
    prior = baseline.prior_for_method(example, method, fixed_discount=fixed_discount)
    if method.endswith("_sam"):
        prior = mixture_prior.apply_sam_conflict_adapter(prior, y=y_obs, n=n)
    lambda0 = float(prior.get("lambda_0", 1.0))
    components = [
        {
            "alpha": float(component["alpha"]),
            "beta": float(component["beta"]),
            "lambda": float(component.get("lambda_active", component.get("lambda", 0.0))),
        }
        for component in prior.get("components") or []
    ]
    pmf = [
        mixture_prior.mixture_predictive_probability(y, n, lambda0, 1.0, 1.0, components)
        for y in range(n + 1)
    ]
    total = math.fsum(pmf)
    if total > 0:  # defensive renormalization; a proper mixture already sums to 1
        pmf = [value / total for value in pmf]
    return y_obs, n, pmf


# --------------------------------------------------------------------------- #
# Calibration primitives
# --------------------------------------------------------------------------- #
def _cdf_below_and_at(pmf: list[float], y: int) -> tuple[float, float]:
    """Return (P(Y < y), P(Y = y))."""
    below = math.fsum(pmf[:y]) if y > 0 else 0.0
    at = pmf[y] if 0 <= y < len(pmf) else 0.0
    return below, at


def pit_values(pmf: list[float], y_obs: int, rng: random.Random) -> tuple[float, float]:
    """Return (randomized_pit, mean_pit) for the observed count.

    Randomized PIT = F(y-1) + U * P(Y = y), U ~ Uniform(0, 1); Uniform under
    calibration. Mean (non-randomized) PIT = F(y-1) + 0.5 * P(Y = y) is the
    deterministic, reproducible analogue used for the reported summaries.
    """
    below, at = _cdf_below_and_at(pmf, y_obs)
    randomized = below + rng.random() * at
    mean_pit = below + 0.5 * at
    return min(max(randomized, 0.0), 1.0), min(max(mean_pit, 0.0), 1.0)


def central_interval(pmf: list[float], level: float) -> tuple[int, int]:
    """Equal-tailed central predictive interval [lo, hi] at coverage ``level``."""
    tail = (1.0 - level) / 2.0
    cumulative = 0.0
    lo = 0
    for y, mass in enumerate(pmf):
        cumulative += mass
        if cumulative > tail:
            lo = y
            break
    cumulative = 0.0
    hi = len(pmf) - 1
    for y in range(len(pmf) - 1, -1, -1):
        cumulative += pmf[y]
        if cumulative > tail:
            hi = y
            break
    if hi < lo:
        lo, hi = min(lo, hi), max(lo, hi)
    return lo, hi


def predicted_mean_rate(pmf: list[float], n: int) -> float:
    if n <= 0:
        return float("nan")
    expected = math.fsum(y * mass for y, mass in enumerate(pmf))
    return expected / n


def ks_uniform(sorted_pits: list[float]) -> float:
    """Kolmogorov-Smirnov distance of a PIT sample to Uniform(0, 1)."""
    m = len(sorted_pits)
    if m == 0:
        return float("nan")
    d = 0.0
    for i, u in enumerate(sorted_pits, start=1):
        d = max(d, abs(u - (i - 1) / m), abs(i / m - u))
    return d


# --------------------------------------------------------------------------- #
# Core evaluation
# --------------------------------------------------------------------------- #
def evaluate_method(
    examples: list[dict[str, Any]],
    method: str,
    fixed_discount: float,
    seed: int,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    rng = random.Random(seed)
    per_example: list[dict[str, Any]] = []
    pits: list[float] = []
    coverage_hits = {level: 0 for level in COVERAGE_LEVELS}
    widths = {level: [] for level in COVERAGE_LEVELS}
    nlls: list[float] = []
    rate_pairs: list[tuple[float, float]] = []  # (predicted_rate, observed_rate)

    for index, example in enumerate(examples):
        try:
            y_obs, n, pmf = predictive_pmf(example, method, fixed_discount=fixed_discount)
        except Exception as error:  # noqa: BLE001 - skip malformed rows, keep count
            per_example.append(
                {
                    "example_index": index,
                    "query_nct_id": example.get("query_nct_id", ""),
                    "method": method,
                    "status": f"skipped:{type(error).__name__}",
                }
            )
            continue
        if n <= 0 or not (0 <= y_obs <= n):
            continue
        randomized_pit, mean_pit = pit_values(pmf, y_obs, rng)
        pits.append(mean_pit)
        p_hat = predicted_mean_rate(pmf, n)
        rate_pairs.append((p_hat, y_obs / n))
        nll = -math.log(max(pmf[y_obs], 1e-300))
        nlls.append(nll)
        row = {
            "example_index": index,
            "query_nct_id": example.get("query_nct_id", ""),
            "method": method,
            "y_obs": y_obs,
            "n": n,
            "predicted_mean_rate": round(p_hat, 6),
            "observed_rate": round(y_obs / n, 6),
            "predictive_prob_at_obs": round(pmf[y_obs], 8),
            "nll": round(nll, 6),
            "pit_mean": round(mean_pit, 6),
            "pit_randomized": round(randomized_pit, 6),
            "status": "ok",
        }
        for level in COVERAGE_LEVELS:
            lo, hi = central_interval(pmf, level)
            inside = int(lo <= y_obs <= hi)
            coverage_hits[level] += inside
            widths[level].append((hi - lo) / n)
            tag = f"{int(level * 100)}"
            row[f"cover{tag}"] = inside
            row[f"width{tag}"] = round((hi - lo) / n, 6)
        per_example.append(row)

    n_ok = len(pits)
    sorted_pits = sorted(pits)
    summary = {
        "method": method,
        "n_examples": n_ok,
        "mean_nll": round(_mean(nlls), 6),
        "pit_mean": round(_mean(pits), 6),
        "pit_variance": round(_variance(pits), 6),
        "pit_ks_distance": round(ks_uniform(sorted_pits), 6),
        "reliability_slope": round(_reliability_slope(rate_pairs), 6),
    }
    for level in COVERAGE_LEVELS:
        tag = f"{int(level * 100)}"
        summary[f"coverage{tag}"] = round(coverage_hits[level] / n_ok, 6) if n_ok else float("nan")
        summary[f"coverage{tag}_error"] = (
            round(coverage_hits[level] / n_ok - level, 6) if n_ok else float("nan")
        )
        summary[f"mean_width{tag}"] = round(_mean(widths[level]), 6)
    return per_example, summary


def reliability_bins(
    examples: list[dict[str, Any]],
    method: str,
    fixed_discount: float,
    n_bins: int = 10,
) -> list[dict[str, Any]]:
    pairs: list[tuple[float, float, int]] = []  # (predicted, observed, n)
    for example in examples:
        try:
            y_obs, n, pmf = predictive_pmf(example, method, fixed_discount=fixed_discount)
        except Exception:  # noqa: BLE001
            continue
        if n <= 0 or not (0 <= y_obs <= n):
            continue
        pairs.append((predicted_mean_rate(pmf, n), y_obs / n, n))
    bins: list[dict[str, Any]] = []
    for b in range(n_bins):
        lo = b / n_bins
        hi = (b + 1) / n_bins
        members = [p for p in pairs if (lo <= p[0] < hi) or (b == n_bins - 1 and p[0] == hi)]
        if not members:
            continue
        bins.append(
            {
                "method": method,
                "bin_lower": round(lo, 3),
                "bin_upper": round(hi, 3),
                "count": len(members),
                "mean_predicted_rate": round(_mean([m[0] for m in members]), 6),
                "mean_observed_rate": round(_mean([m[1] for m in members]), 6),
                "total_patients": sum(m[2] for m in members),
            }
        )
    return bins


# --------------------------------------------------------------------------- #
# Small stats helpers (stdlib only)
# --------------------------------------------------------------------------- #
def _mean(values: list[float]) -> float:
    finite = [v for v in values if v is not None and math.isfinite(v)]
    return math.fsum(finite) / len(finite) if finite else float("nan")


def _variance(values: list[float]) -> float:
    finite = [v for v in values if v is not None and math.isfinite(v)]
    if len(finite) < 2:
        return float("nan")
    mu = math.fsum(finite) / len(finite)
    return math.fsum((v - mu) ** 2 for v in finite) / (len(finite) - 1)


def _reliability_slope(rate_pairs: list[tuple[float, float]]) -> float:
    """OLS slope of observed rate on predicted rate; 1.0 is perfectly calibrated."""
    pairs = [(p, o) for p, o in rate_pairs if math.isfinite(p) and math.isfinite(o)]
    if len(pairs) < 2:
        return float("nan")
    mx = _mean([p for p, _ in pairs])
    my = _mean([o for _, o in pairs])
    sxx = math.fsum((p - mx) ** 2 for p, _ in pairs)
    sxy = math.fsum((p - mx) * (o - my) for p, o in pairs)
    # A near-constant predictor (e.g. the weak Beta(1,1) prior always predicts
    # rate 0.5) has no spread on the x-axis; the slope is undefined, not huge.
    if sxx <= 1e-8:
        return float("nan")
    return sxy / sxx


# --------------------------------------------------------------------------- #
# Minimal dependency-free SVG figures
# --------------------------------------------------------------------------- #
def _svg_pit_histogram(pits: list[float], method: str, n_bins: int = 10) -> str:
    counts = [0] * n_bins
    for u in pits:
        idx = min(int(u * n_bins), n_bins - 1)
        counts[idx] += 1
    total = max(sum(counts), 1)
    freqs = [c / total for c in counts]
    expected = 1.0 / n_bins
    w, h, pad = 460, 300, 40
    plot_w, plot_h = w - 2 * pad, h - 2 * pad
    bar_w = plot_w / n_bins
    ymax = max(max(freqs), expected) * 1.25
    parts = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {w} {h}" font-family="sans-serif">']
    parts.append(f'<rect x="0" y="0" width="{w}" height="{h}" fill="white"/>')
    for i, f in enumerate(freqs):
        bar_h = (f / ymax) * plot_h
        x = pad + i * bar_w
        y = pad + plot_h - bar_h
        parts.append(
            f'<rect x="{x + 1:.1f}" y="{y:.1f}" width="{bar_w - 2:.1f}" height="{bar_h:.1f}" '
            f'fill="#4C72B0" opacity="0.85"/>'
        )
    y_exp = pad + plot_h - (expected / ymax) * plot_h
    parts.append(
        f'<line x1="{pad}" y1="{y_exp:.1f}" x2="{pad + plot_w}" y2="{y_exp:.1f}" '
        f'stroke="#C44E52" stroke-width="2" stroke-dasharray="6 4"/>'
    )
    parts.append(f'<line x1="{pad}" y1="{pad}" x2="{pad}" y2="{pad + plot_h}" stroke="#333"/>')
    parts.append(f'<line x1="{pad}" y1="{pad + plot_h}" x2="{pad + plot_w}" y2="{pad + plot_h}" stroke="#333"/>')
    parts.append(f'<text x="{w/2:.0f}" y="20" text-anchor="middle" font-size="15">PIT histogram - {method}</text>')
    parts.append(f'<text x="{w/2:.0f}" y="{h-8}" text-anchor="middle" font-size="12">PIT (uniform if calibrated)</text>')
    parts.append("</svg>")
    return "\n".join(parts)


def _svg_reliability(bins: list[dict[str, Any]], method: str) -> str:
    w, h, pad = 340, 340, 45
    plot = w - 2 * pad
    parts = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {w} {h}" font-family="sans-serif">']
    parts.append(f'<rect x="0" y="0" width="{w}" height="{h}" fill="white"/>')
    parts.append(f'<line x1="{pad}" y1="{pad+plot}" x2="{pad+plot}" y2="{pad}" stroke="#999" stroke-dasharray="5 4"/>')
    parts.append(f'<line x1="{pad}" y1="{pad}" x2="{pad}" y2="{pad+plot}" stroke="#333"/>')
    parts.append(f'<line x1="{pad}" y1="{pad+plot}" x2="{pad+plot}" y2="{pad+plot}" stroke="#333"/>')

    def sx(v: float) -> float:
        return pad + v * plot

    def sy(v: float) -> float:
        return pad + plot - v * plot

    pts = [(b["mean_predicted_rate"], b["mean_observed_rate"]) for b in bins]
    if pts:
        path = " ".join(f'{"M" if i == 0 else "L"}{sx(p):.1f},{sy(o):.1f}' for i, (p, o) in enumerate(pts))
        parts.append(f'<path d="{path}" fill="none" stroke="#4C72B0" stroke-width="2"/>')
        for p, o in pts:
            parts.append(f'<circle cx="{sx(p):.1f}" cy="{sy(o):.1f}" r="3.5" fill="#4C72B0"/>')
    parts.append(f'<text x="{w/2:.0f}" y="20" text-anchor="middle" font-size="14">Reliability - {method}</text>')
    parts.append(f'<text x="{w/2:.0f}" y="{h-10}" text-anchor="middle" font-size="11">predicted response rate</text>')
    parts.append("</svg>")
    return "\n".join(parts)


# --------------------------------------------------------------------------- #
# IO
# --------------------------------------------------------------------------- #
def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames: list[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _write_markdown(path: Path, summary_rows: list[dict[str, Any]]) -> None:
    lines = [
        "# Calibration Diagnostics (label-free)",
        "",
        "Predictive calibration against the real held-out pseudo-query outcome. "
        "A well-calibrated method has PIT mean ~0.5, low PIT KS distance, coverage "
        "close to nominal, reliability slope ~1, and low mean NLL at matched coverage.",
        "",
        "| Method | N | Mean NLL | PIT mean | PIT KS | Cov50 | Cov80 | Cov95 | Width95 | Rel. slope |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in summary_rows:
        lines.append(
            "| {method} | {n} | {nll:.4f} | {pit:.3f} | {ks:.3f} | {c50:.3f} | {c80:.3f} | "
            "{c95:.3f} | {w95:.3f} | {slope:.3f} |".format(
                method=row["method"],
                n=row["n_examples"],
                nll=row["mean_nll"],
                pit=row["pit_mean"],
                ks=row["pit_ks_distance"],
                c50=row["coverage50"],
                c80=row["coverage80"],
                c95=row["coverage95"],
                w95=row["mean_width95"],
                slope=row["reliability_slope"],
            )
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def run(
    examples_jsonl: Path,
    output_dir: Path,
    methods: tuple[str, ...],
    fixed_discount: float,
    seed: int,
    limit: int | None = None,
) -> dict[str, Any]:
    examples = read_jsonl(examples_jsonl)
    if limit is not None:
        examples = examples[:limit]
    output_dir.mkdir(parents=True, exist_ok=True)

    all_rows: list[dict[str, Any]] = []
    summaries: list[dict[str, Any]] = []
    reliability_rows: list[dict[str, Any]] = []
    for method in methods:
        per_example, summary = evaluate_method(examples, method, fixed_discount, seed)
        all_rows.extend(per_example)
        summaries.append(summary)
        bins = reliability_bins(examples, method, fixed_discount)
        reliability_rows.extend(bins)
        pits = [r["pit_mean"] for r in per_example if r.get("status") == "ok"]
        (output_dir / f"pit_hist_{method}.svg").write_text(
            _svg_pit_histogram(pits, method), encoding="utf-8"
        )
        (output_dir / f"reliability_{method}.svg").write_text(
            _svg_reliability(bins, method), encoding="utf-8"
        )

    _write_csv(output_dir / "calibration_per_example.csv", all_rows)
    _write_csv(output_dir / "calibration_summary.csv", summaries)
    _write_csv(output_dir / "calibration_reliability.csv", reliability_rows)
    _write_markdown(output_dir / "calibration_summary.md", summaries)
    payload = {
        "examples_jsonl": str(examples_jsonl),
        "n_examples": len(examples),
        "methods": list(methods),
        "fixed_discount": fixed_discount,
        "seed": seed,
        "coverage_levels": list(COVERAGE_LEVELS),
        "summary": summaries,
    }
    (output_dir / "calibration_summary.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return payload


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--examples-jsonl", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts/calibration_diagnostics"))
    parser.add_argument("--methods", nargs="+", default=list(DEFAULT_METHODS))
    parser.add_argument("--fixed-discount", type=float, default=0.25)
    parser.add_argument("--seed", type=int, default=20260707)
    parser.add_argument("--limit", type=int, default=None, help="Optional cap on #examples (for smoke tests).")
    args = parser.parse_args(argv)

    payload = run(
        examples_jsonl=args.examples_jsonl,
        output_dir=args.output_dir,
        methods=tuple(args.methods),
        fixed_discount=args.fixed_discount,
        seed=args.seed,
        limit=args.limit,
    )
    print(json.dumps({"n_examples": payload["n_examples"], "summary": payload["summary"]}, indent=2))


if __name__ == "__main__":
    main()
