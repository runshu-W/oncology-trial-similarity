"""Round-3 EC anchor-misspecification battery (review round-3 Major 4).

The round-2 external-control result showed the selective prior withholding
external controls and riding a highly informative EB anchor. This runner asks
what happens when that anchor is WRONG, and when the true control rates are
heterogeneous:

  Anchor variants (evaluation-side; identical datasets, released seeds):
    canonical          Beta(a0, b0) as fitted on the EC training draw
    mean_up / mean_dn  anchor mean shifted by +/-0.4 on the logit scale,
                       precision (a0+b0) preserved
    prec_x4 / prec_q4  anchor precision scaled x4 / x0.25, mean preserved
    flat               Beta(1, 1) (no marginal information)

  Heterogeneity cells (new seeds): true control rate drawn per replicate with
  SD 0.3 on the logit scale around 0.20 (theta_control_sd), drift grid subset.

Methods per variant: EB-only (anchor alone) and the FROZEN round-2 selective
model evaluated with that anchor; internal-only as the fixed reference.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from dgm import ExternalControlConfig, simulate_external_control_dataset  # noqa: E402
from metrics import SUCCESS_THRESHOLD  # noqa: E402
from methods import weak_only  # noqa: E402
from run_external_control import DRIFT_GRID, EFFECT_ALT, EFFECT_NULL, seed_for  # noqa: E402
from selective_methods import eb_only, load_selective_npz, selective_prior  # noqa: E402


def perturb(anchor, kind):
    a0, b0 = anchor
    if kind == "canonical":
        return (a0, b0)
    if kind == "flat":
        return (1.0, 1.0)
    n = a0 + b0
    mu = a0 / n
    lo = math.log(mu / (1 - mu))
    if kind == "mean_up":
        mu2 = 1 / (1 + math.exp(-(lo + 0.4)))
        return (mu2 * n, (1 - mu2) * n)
    if kind == "mean_dn":
        mu2 = 1 / (1 + math.exp(-(lo - 0.4)))
        return (mu2 * n, (1 - mu2) * n)
    if kind == "prec_x4":
        return (a0 * 4, b0 * 4)
    if kind == "prec_q4":
        return (a0 / 4, b0 / 4)
    raise KeyError(kind)


VARIANTS = ("canonical", "mean_up", "mean_dn", "prec_x4", "prec_q4", "flat")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--replicates", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=20260720)
    ap.add_argument("--models-dir", type=str, required=True)
    ap.add_argument("--output", type=str, required=True)
    args = ap.parse_args()
    out = Path(args.output).resolve()
    out.mkdir(parents=True, exist_ok=True)
    mdir = Path(args.models_dir).resolve()

    anchor = tuple(json.loads((mdir / "anchor_ec.json").read_text())["ec"])
    sel = load_selective_npz(mdir / "ec_selective_sel_pro.npz", 10)

    rows = []
    cells = [("null", EFFECT_NULL, d, 0.0) for d in DRIFT_GRID] + \
            [("alt", EFFECT_ALT, d, 0.0) for d in DRIFT_GRID] + \
            [("null", EFFECT_NULL, d, 0.3) for d in (-0.8, 0.0)] + \
            [("alt", EFFECT_ALT, d, 0.3) for d in (-0.8, 0.0)]
    for world, effect, drift, het in cells:
        if het > 0:
            cfg = ExternalControlConfig(name=f"EC_het{het}_{world}_drift{drift}",
                                        treatment_effect=effect, drift_shift=drift,
                                        theta_control_sd=het)
        else:
            cfg = ExternalControlConfig(name=f"EC_{world}_drift{drift}",
                                        treatment_effect=effect, drift_shift=drift)
        data = simulate_external_control_dataset(
            cfg, args.replicates, seed_for(cfg.name, args.seed))
        table = {}
        for v in VARIANTS:
            ab = perturb(anchor, v)
            table[f"eb_only_{v}"] = lambda c, y0, n0, ab=ab: eb_only(c, ab)
            table[f"selective_{v}"] = lambda c, y0, n0, ab=ab: selective_prior(
                sel, c, ab, prospective=True)
        table["internal_only"] = lambda c, y0, n0: weak_only(c)
        for name, build in table.items():
            rej = 0
            for q in data:
                y_ctl, n_ctl = q["y_control"], q["n_control"]
                y_trt, n_trt = q["y_treatment"], q["n_treatment"]
                prior = build(q["candidates"], y_ctl, n_ctl)
                _eff, _lo, _hi, p_sup, _cm = prior.treatment_effect(
                    y_ctl, n_ctl, y_trt, n_trt)
                rej += p_sup > SUCCESS_THRESHOLD
            rows.append({"world": world, "drift": drift, "het": het,
                         "method": name, "reject_rate": rej / len(data),
                         "n": len(data)})
        done = [r for r in rows if r["world"] == world and r["drift"] == drift
                and r["het"] == het]
        print(f"{world} drift={drift} het={het}: " + " ".join(
            f"{r['method']}={r['reject_rate']:.3f}" for r in done
            if "selective_canonical" in r["method"] or "flat" in r["method"]
            or r["method"] == "internal_only"), flush=True)
        (out / "ec_anchor_battery.json").write_text(json.dumps(rows, indent=1))
    print("done", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
