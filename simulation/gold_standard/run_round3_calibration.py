"""Round-3 calibration study: threshold recalibration, borrowing cap, 2-D grid.

Answers review round-3 Major 1. For a set of selective designs this runner
stores the PER-REPLICATE posterior probability P(theta > theta_null | y0, n0),
so that the rejection rate at ANY success threshold is computable offline:

  - full-conflict design cells reuse the released per-cell seeds (identical
    datasets; rejection at 0.975 must reproduce the round-2 numbers), and
  - NEW partial-conflict cells extend the grid to conflict_fraction in
    {0.25, 0.5, 0.75} x shift in {0.5, 1.0, 1.5} (fresh seeds via seed_for).

The pre-specified calibration grid for worst-case type I error is ALL null
cells: shifts {0,...,1.5} at fraction 1.0 plus the partial-conflict cells.
tau* = smallest threshold whose worst-case type I over that grid <= 0.025.

Methods: selective (pro), selective+SAM (anchored), capped selective
(borrowed mass <= 0.5) with and without SAM, fixed-budget+SAM, rule+SAM,
EB-only and no-borrowing references.

Usage:
  python3 run_round3_calibration.py --replicates 2000 --output <dir>
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from dgm import SimConfig, simulate_dataset  # noqa: E402
from metrics import SUCCESS_THRESHOLD, THETA_NULL, DESIGN_DELTA  # noqa: E402
from methods import (  # noqa: E402
    TwoHeadDeepSets, apply_sam, rule_components, two_head_prior, weak_only,
)
from run_simulation import CONFLICT_GRID, seed_for  # noqa: E402
from selective_methods import (  # noqa: E402
    AnchoredMixturePrior, apply_sam_anchored, eb_only, load_selective_npz,
    selective_prior,
)

FRACTIONS = (0.25, 0.5, 0.75)
PARTIAL_SHIFTS = (0.5, 1.0, 1.5)
CAP = 0.5


def cap_prior(prior: AnchoredMixturePrior, cap: float = CAP) -> AnchoredMixturePrior:
    """Project the selective prior onto {historical mass <= cap}."""
    mass = prior.historical_mass()
    if mass <= cap:
        return prior
    scale = cap / mass
    w = prior.weights * scale
    out = AnchoredMixturePrior(1.0 - float(w.sum()), w, prior.alphas, prior.betas,
                               prior.ess_terms, scores=prior.scores,
                               weak_ab=prior.weak_ab)
    out.discounts = prior.discounts
    return out


def design_cfg(world: str, shift: float, fraction: float) -> SimConfig:
    theta = THETA_NULL if world == "null" else THETA_NULL + DESIGN_DELTA
    if fraction >= 1.0:
        # identical name -> identical seed -> identical dataset as released
        return SimConfig(name=f"{world}_shift{shift}", conflict_shift=shift,
                         theta_query_fixed=theta)
    return SimConfig(name=f"{world}_shift{shift}_frac{fraction}",
                     conflict_shift=shift, conflict_fraction=fraction,
                     theta_query_fixed=theta)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--replicates", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=20260720)
    ap.add_argument("--models-dir", type=str, required=True,
                    help="dir holding the round-2 goldsim model npz files")
    ap.add_argument("--output", type=str, required=True)
    args = ap.parse_args()

    out = Path(args.output).resolve()
    (out / "pvals").mkdir(parents=True, exist_ok=True)
    mdir = Path(args.models_dir).resolve()

    anchor = tuple(json.loads((mdir / "anchor.json").read_text())["gold"])
    sel = load_selective_npz(mdir / "selective_sel_pro.npz", 10)
    pro = TwoHeadDeepSets(input_dim=10)
    z = np.load(mdir / "two_head_pro.npz")
    pro.load([z[f"p{i}"] for i in range(len(pro.params))])

    def build(name, cands, y0, n0):
        if name == "weak_only":
            return weak_only(cands)
        if name == "eb_only":
            return eb_only(cands, anchor)
        if name == "rule_sam":
            p = rule_components(cands)
            return apply_sam(p, y0, n0)[0]
        if name == "two_head_pro_sam":
            p = two_head_prior(pro, cands, prospective=True)
            return apply_sam(p, y0, n0)[0]
        if name == "selective":
            return selective_prior(sel, cands, anchor, prospective=True)
        if name == "selective_sam":
            p = selective_prior(sel, cands, anchor, prospective=True)
            return apply_sam_anchored(p, y0, n0)[0]
        if name == "selective_cap50":
            return cap_prior(selective_prior(sel, cands, anchor, prospective=True))
        if name == "selective_cap50_sam":
            p = cap_prior(selective_prior(sel, cands, anchor, prospective=True))
            return apply_sam_anchored(p, y0, n0)[0]
        raise KeyError(name)

    METHODS = ("weak_only", "eb_only", "rule_sam", "two_head_pro_sam",
               "selective", "selective_sam", "selective_cap50",
               "selective_cap50_sam")

    cells = []
    for world in ("null", "alt"):
        for s in CONFLICT_GRID:
            cells.append((world, float(s), 1.0))
        for s in PARTIAL_SHIFTS:
            for f in FRACTIONS:
                cells.append((world, float(s), float(f)))

    for world, s, f in cells:
        tag = f"{world}_s{s}_f{f}"
        part = out / "pvals" / f"{tag}.npz"
        if part.exists():
            print(f"{tag} cached", flush=True)
            continue
        cfg = design_cfg(world, s, f)
        data = simulate_dataset(cfg, args.replicates, seed_for(cfg.name, args.seed))
        store = {m: np.empty(len(data)) for m in METHODS}
        for i, q in enumerate(data):
            cands, y0, n0 = q["candidates"], q["y_query"], q["n_query"]
            for m in METHODS:
                prior = build(m, cands, y0, n0)
                store[m][i] = prior.posterior_prob_greater(y0, n0, THETA_NULL)
        np.savez_compressed(part, **store)
        r975 = {m: float((store[m] > SUCCESS_THRESHOLD).mean()) for m in METHODS}
        print(f"{tag}: reject@0.975 " + " ".join(f"{m}={v:.3f}" for m, v in r975.items()
                                                 if m.startswith("select")), flush=True)

    # ---- offline calibration --------------------------------------------
    grid = np.round(np.arange(0.975, 0.9996, 0.0005), 4)
    null_cells = [(w, s, f) for (w, s, f) in cells if w == "null"]
    alt_cells = [(w, s, f) for (w, s, f) in cells if w == "alt"]
    summary = {"threshold_grid": grid.tolist(), "nominal": 0.025,
               "cells_null": [f"s{s}_f{f}" for _, s, f in null_cells],
               "methods": {}}
    for m in METHODS:
        t1 = {}
        for _, s, f in null_cells:
            p = np.load(out / "pvals" / f"null_s{s}_f{f}.npz")[m]
            t1[f"s{s}_f{f}"] = [float((p > t).mean()) for t in grid]
        worst = np.max(np.array(list(t1.values())), axis=0)
        ok = np.where(worst <= 0.025)[0]
        tau_star = float(grid[ok[0]]) if ok.size else None
        pw = {}
        for _, s, f in alt_cells:
            p = np.load(out / "pvals" / f"alt_s{s}_f{f}.npz")[m]
            pw[f"s{s}_f{f}"] = {
                "at_0.975": float((p > SUCCESS_THRESHOLD).mean()),
                "at_tau_star": float((p > tau_star).mean()) if tau_star else None,
            }
        summary["methods"][m] = {
            "worst_type1_at_0.975": float(worst[0]),
            "tau_star": tau_star,
            "worst_type1_at_tau_star": float(worst[ok[0]]) if ok.size else None,
            "type1_by_cell_at_0.975": {k: v[0] for k, v in t1.items()},
            "power": pw,
        }
        n = args.replicates
        mcse = (0.025 * 0.975 / n) ** 0.5
        summary["mcse_at_nominal"] = mcse
        print(f"{m:20s} worst@0.975={worst[0]:.3f}  tau*={tau_star}  "
              f"power(no-conflict)@0.975={pw['s0.0_f1.0']['at_0.975']:.3f}"
              f" @tau*={pw['s0.0_f1.0']['at_tau_star'] if tau_star else float('nan'):.3f}",
              flush=True)
    (out / "calibration_summary.json").write_text(json.dumps(summary, indent=1))
    print("done", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
