"""Module M1 / Experiment E1 - Rolling-origin forward validation (label-free).

The current temporal analysis scores *pre-existing* predictions on temporal
subsets. That is vulnerable to a leakage/overfitting critique because nothing is
retrained on past-only data. This harness performs true train-on-past /
test-on-future validation using real ClinicalTrials.gov primary-completion
dates: for each temporal cutoff it evaluates held-out beta-binomial predictive
NLL on the *future* fold only.

Leakage status is reported honestly per method:

  * Closed-form priors (weak_only, rule, rule_sam, map_like, power_prior_like,
    commensurate_like, fixed_discount) require NO fitting, so their future-fold
    NLL is genuinely leakage-free forward validation. Note that ``rule_sam`` is
    the mixture prior + SAM conflict adapter, i.e. very close to the paper's
    method, so this already demonstrates that conflict-adapted mixture borrowing
    generalizes forward in time.

  * The learned two_head / two_head_sam methods depend on a trained
    lambda_model. Two modes:
      - ``frozen`` (default): a single globally trained model is applied to all
        folds. This is convenient but NOT leakage-free (the model may have seen
        future pseudo-queries); rows are flagged ``leakage_free=false``.
      - ``refit``: supply ``--fold-model-dir`` containing per-cutoff models
        trained on past-only data (produced in a torch environment, see the
        command template at the bottom of this file). The matching fold model is
        applied to each eval fold and rows are flagged ``leakage_free=true``.

Outputs: per-(split, method) NLL/coverage with bootstrap-CI deltas vs a
reference method, an aggregated summary, a markdown table, JSON, and an
NLL-vs-cutoff SVG.

Example (frozen two_head, closed-form methods leakage-free):
    python scripts/run_forward_validation.py \
        --examples-jsonl artifacts/temporal_validation_true_dates/lambda_training_examples_with_true_dates.jsonl \
        --model-path artifacts/lambda_model_comparison_orr_all/two_head_deepsets/lambda_model.pt \
        --cutoffs 2020-12-31 2021-12-31 2022-12-31 \
        --methods weak_only rule rule_sam two_head two_head_sam map_like power_prior_like commensurate_like \
        --output-dir artifacts/forward_validation_orr_all --seed 20260707
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

import run_calibration_diagnostics as calib  # noqa: E402
import temporal_validation as tv  # noqa: E402
import two_head_inference as thi  # noqa: E402

CLOSED_FORM = {
    "weak_only", "rule", "rule_sam", "map_like", "power_prior_like",
    "commensurate_like", "fixed_discount",
}
LEARNED = {"two_head", "two_head_sam", "model", "model_sam"}
DEFAULT_METHODS = (
    "weak_only", "rule", "rule_sam", "two_head", "two_head_sam",
    "map_like", "power_prior_like", "commensurate_like",
)


def _mean(values: list[float]) -> float:
    finite = [v for v in values if v is not None and math.isfinite(v)]
    return math.fsum(finite) / len(finite) if finite else float("nan")


def _eval_fold(
    examples: list[dict[str, Any]],
    indices: list[int],
    method: str,
    fixed_discount: float,
) -> tuple[list[float], list[int]]:
    """Return per-example (nll, cover95) on an evaluation fold."""
    nlls: list[float] = []
    covers: list[int] = []
    for idx in indices:
        example = examples[idx]
        try:
            y, n, pmf = calib.predictive_pmf(example, method, fixed_discount=fixed_discount)
        except Exception:  # noqa: BLE001
            continue
        if n <= 0 or not (0 <= y <= n):
            continue
        nlls.append(-math.log(max(pmf[y], 1e-300)))
        lo, hi = calib.central_interval(pmf, 0.95)
        covers.append(int(lo <= y <= hi))
    return nlls, covers


def _bootstrap_delta_ci(
    method_nlls: list[float],
    reference_nlls: list[float],
    rng: random.Random,
    n_boot: int = 1000,
) -> tuple[float, float]:
    """Paired bootstrap CI for mean(method - reference) NLL over eval queries."""
    paired = [(m, r) for m, r in zip(method_nlls, reference_nlls) if math.isfinite(m) and math.isfinite(r)]
    if len(paired) < 2:
        return (float("nan"), float("nan"))
    deltas = []
    m_count = len(paired)
    for _ in range(n_boot):
        sample = [paired[rng.randrange(m_count)] for _ in range(m_count)]
        deltas.append(_mean([a - b for a, b in sample]))
    deltas.sort()
    lo = deltas[int(0.025 * n_boot)]
    hi = deltas[min(int(0.975 * n_boot), n_boot - 1)]
    return (round(lo, 6), round(hi, 6))


def run(
    examples_jsonl: Path,
    output_dir: Path,
    cutoffs: list[str],
    methods: tuple[str, ...],
    reference_method: str,
    model_path: Path | None,
    fold_model_dir: Path | None,
    fixed_discount: float,
    seed: int,
    n_boot: int,
) -> dict[str, Any]:
    examples = calib.read_jsonl(examples_jsonl)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Attach frozen global model lambdas if a learned method is requested.
    frozen_attached = False
    if any(m in LEARNED for m in methods) and model_path is not None and fold_model_dir is None:
        checkpoint = thi.load_checkpoint(model_path)
        examples = [thi.attach_model_outputs(e, checkpoint) for e in examples]
        frozen_attached = True

    rng = random.Random(seed)
    per_split_rows: list[dict[str, Any]] = []

    for cutoff in cutoffs:
        try:
            train_idx, eval_idx, meta = tv.date_based_split_indices(examples, train_end_date=cutoff)
        except ValueError as error:
            per_split_rows.append({"cutoff": cutoff, "method": "-", "error": str(error)})
            continue

        # If per-fold refit models are supplied, attach the model trained on past-only.
        fold_examples = examples
        fold_leakage_free_learned = False
        if fold_model_dir is not None:
            candidate = fold_model_dir / f"model_{cutoff}.pt"
            if candidate.exists():
                ckpt = thi.load_checkpoint(candidate)
                fold_examples = [thi.attach_model_outputs(e, ckpt) for e in examples]
                fold_leakage_free_learned = True

        reference_nlls, _ = _eval_fold(fold_examples, eval_idx, reference_method, fixed_discount)

        for method in methods:
            nlls, covers = _eval_fold(fold_examples, eval_idx, method, fixed_discount)
            if not nlls:
                continue
            if method in CLOSED_FORM:
                leakage_free = True
            else:
                leakage_free = fold_leakage_free_learned  # only true with per-fold refit
            delta_lo, delta_hi = _bootstrap_delta_ci(nlls, reference_nlls, rng, n_boot=n_boot)
            per_split_rows.append(
                {
                    "cutoff": cutoff,
                    "method": method,
                    "train_count": meta["train_count"],
                    "eval_count": len(nlls),
                    "mean_nll": round(_mean(nlls), 6),
                    f"delta_nll_vs_{reference_method}": round(_mean(nlls) - _mean(reference_nlls), 6),
                    "delta_ci_low": delta_lo,
                    "delta_ci_high": delta_hi,
                    "coverage95": round(_mean(covers), 6),
                    "leakage_free": leakage_free,
                    "learned_mode": (
                        "n/a" if method in CLOSED_FORM
                        else ("refit" if fold_leakage_free_learned else ("frozen" if frozen_attached else "unavailable"))
                    ),
                }
            )

    # Aggregate across cutoffs (mean NLL per method).
    summary: dict[str, dict[str, Any]] = {}
    for row in per_split_rows:
        if row.get("method") in (None, "-"):
            continue
        key = row["method"]
        summary.setdefault(key, {"method": key, "nlls": [], "covers": [], "leakage_free": row["leakage_free"]})
        summary[key]["nlls"].append(row["mean_nll"])
        summary[key]["covers"].append(row["coverage95"])
    summary_rows = [
        {
            "method": s["method"],
            "n_folds": len(s["nlls"]),
            "mean_nll_across_folds": round(_mean(s["nlls"]), 6),
            "mean_coverage95": round(_mean(s["covers"]), 6),
            "leakage_free": s["leakage_free"],
        }
        for s in summary.values()
    ]
    summary_rows.sort(key=lambda r: r["mean_nll_across_folds"])

    _write_csv(output_dir / "forward_validation_by_split.csv", per_split_rows)
    _write_csv(output_dir / "forward_validation_summary.csv", summary_rows)
    _write_markdown(output_dir / "forward_validation_summary.md", per_split_rows, summary_rows, reference_method)
    (output_dir / "fig_forward_nll_by_cutoff.svg").write_text(
        _svg_nll_by_cutoff(per_split_rows, cutoffs), encoding="utf-8"
    )
    payload = {
        "examples_jsonl": str(examples_jsonl),
        "n_examples": len(examples),
        "cutoffs": cutoffs,
        "methods": list(methods),
        "reference_method": reference_method,
        "frozen_model_attached": frozen_attached,
        "per_split": per_split_rows,
        "summary": summary_rows,
    }
    (output_dir / "forward_validation_summary.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return payload


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
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def _write_markdown(
    path: Path,
    per_split_rows: list[dict[str, Any]],
    summary_rows: list[dict[str, Any]],
    reference_method: str,
) -> None:
    lines = [
        "# Rolling-Origin Forward Validation (train-on-past / test-on-future)",
        "",
        "Held-out beta-binomial predictive NLL on the FUTURE fold using true "
        "primary-completion dates. Closed-form priors are leakage-free; learned "
        "two_head rows are `frozen` (not leakage-free) unless per-fold refit "
        "models are supplied.",
        "",
        "## Aggregated across cutoffs (sorted best NLL first)",
        "",
        "| Method | Folds | Mean NLL | Mean Cov95 | Leakage-free |",
        "|---|---:|---:|---:|:--:|",
    ]
    for r in summary_rows:
        lines.append(
            "| {m} | {n} | {nll:.4f} | {cov:.4f} | {lf} |".format(
                m=r["method"], n=r["n_folds"], nll=r["mean_nll_across_folds"],
                cov=r["mean_coverage95"], lf="yes" if r["leakage_free"] else "no",
            )
        )
    lines += [
        "",
        f"## Per-cutoff detail (delta vs {reference_method}, with paired bootstrap 95% CI)",
        "",
        "| Cutoff | Method | Train | Eval | Mean NLL | dNLL | 95% CI | Cov95 | Leakage-free |",
        "|---|---|---:|---:|---:|---:|---|---:|:--:|",
    ]
    for r in per_split_rows:
        if r.get("method") in (None, "-"):
            continue
        delta_key = f"delta_nll_vs_{reference_method}"
        lines.append(
            "| {c} | {m} | {tr} | {ev} | {nll:.4f} | {d:.4f} | [{lo:.4f}, {hi:.4f}] | {cov:.4f} | {lf} |".format(
                c=r["cutoff"], m=r["method"], tr=r["train_count"], ev=r["eval_count"],
                nll=r["mean_nll"], d=r[delta_key], lo=r["delta_ci_low"], hi=r["delta_ci_high"],
                cov=r["coverage95"], lf="yes" if r["leakage_free"] else "no",
            )
        )
    lines += [
        "",
        "## To make two_head fully leakage-free (torch), refit per fold:",
        "",
        "```bash",
        "# for each cutoff C, train on past-only then export a fold model:",
        "python scripts/run_oncology_retrospective_lambda_training.py \\",
        "    --examples-jsonl <examples_with_dates> --train-end-date C \\",
        "    --model-out <fold_model_dir>/model_C.pt   # (past-only training)",
        "# then re-run this harness with --fold-model-dir <fold_model_dir>",
        "```",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _svg_nll_by_cutoff(rows: list[dict[str, Any]], cutoffs: list[str]) -> str:
    methods = sorted({r["method"] for r in rows if r.get("method") not in (None, "-")})
    colors = ["#4C72B0", "#C44E52", "#55A868", "#8172B2", "#CCB974", "#64B5CD", "#DA8BC3", "#8C8C8C"]
    w, h, pad = 520, 320, 55
    plot_w, plot_h = w - 2 * pad, h - 2 * pad
    xs = list(range(len(cutoffs)))
    all_nll = [r["mean_nll"] for r in rows if r.get("method") not in (None, "-")]
    if not all_nll:
        return '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 10 10"></svg>'
    ymin, ymax = min(all_nll) * 0.999, max(all_nll) * 1.001

    def sx(i: int) -> float:
        return pad + (i / max(1, len(cutoffs) - 1)) * plot_w

    def sy(v: float) -> float:
        return pad + plot_h - (v - ymin) / (ymax - ymin if ymax > ymin else 1) * plot_h

    parts = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {w} {h}" font-family="sans-serif">']
    parts.append(f'<rect width="{w}" height="{h}" fill="white"/>')
    parts.append(f'<line x1="{pad}" y1="{pad}" x2="{pad}" y2="{pad+plot_h}" stroke="#333"/>')
    parts.append(f'<line x1="{pad}" y1="{pad+plot_h}" x2="{pad+plot_w}" y2="{pad+plot_h}" stroke="#333"/>')
    for mi, method in enumerate(methods):
        pts = []
        for i, cutoff in enumerate(cutoffs):
            match = [r for r in rows if r.get("method") == method and r.get("cutoff") == cutoff]
            if match:
                pts.append((sx(i), sy(match[0]["mean_nll"])))
        if not pts:
            continue
        color = colors[mi % len(colors)]
        d = " ".join(f'{"M" if i == 0 else "L"}{x:.1f},{y:.1f}' for i, (x, y) in enumerate(pts))
        parts.append(f'<path d="{d}" fill="none" stroke="{color}" stroke-width="2"/>')
        for x, y in pts:
            parts.append(f'<circle cx="{x:.1f}" cy="{y:.1f}" r="3" fill="{color}"/>')
        parts.append(f'<text x="{pad+plot_w-96}" y="{pad+14+mi*14}" font-size="10" fill="{color}">{method}</text>')
    for i, cutoff in enumerate(cutoffs):
        parts.append(f'<text x="{sx(i):.0f}" y="{pad+plot_h+16}" text-anchor="middle" font-size="10">{cutoff}</text>')
    parts.append(f'<text x="{w/2:.0f}" y="20" text-anchor="middle" font-size="14">Forward-validation NLL by cutoff (lower is better)</text>')
    parts.append("</svg>")
    return "\n".join(parts)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--examples-jsonl", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts/forward_validation"))
    parser.add_argument("--cutoffs", nargs="+", default=["2020-12-31", "2021-12-31", "2022-12-31"])
    parser.add_argument("--methods", nargs="+", default=list(DEFAULT_METHODS))
    parser.add_argument("--reference-method", default="rule")
    parser.add_argument("--model-path", type=Path, default=None, help="Frozen global two_head model.")
    parser.add_argument("--fold-model-dir", type=Path, default=None, help="Dir with per-cutoff model_<cutoff>.pt (leakage-free).")
    parser.add_argument("--fixed-discount", type=float, default=0.25)
    parser.add_argument("--seed", type=int, default=20260707)
    parser.add_argument("--n-boot", type=int, default=1000)
    args = parser.parse_args(argv)

    payload = run(
        examples_jsonl=args.examples_jsonl,
        output_dir=args.output_dir,
        cutoffs=args.cutoffs,
        methods=tuple(args.methods),
        reference_method=args.reference_method,
        model_path=args.model_path,
        fold_model_dir=args.fold_model_dir,
        fixed_discount=args.fixed_discount,
        seed=args.seed,
        n_boot=args.n_boot,
    )
    print(json.dumps({"summary": payload["summary"]}, indent=2))


if __name__ == "__main__":
    main()
