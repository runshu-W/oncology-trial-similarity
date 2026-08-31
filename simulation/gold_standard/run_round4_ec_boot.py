"""Round-4 EC anchor-fit uncertainty propagation (review round-4 Major 7).

The round-3 battery perturbs the EC anchor SYSTEMATICALLY (a stress test).
This runner adds the sampling-uncertainty side:

  1. regenerate the EC training draw (300 replicates, seed base+1, exactly as
     released; the refit anchor must reproduce anchor_ec.json);
  2. nonparametric bootstrap over the draw's internal control arms -> B=500
     anchors; report the anchor's logit-mean shift and precision-ratio
     sampling distribution, and where the round-3 battery perturbations sit
     relative to it;
  3. propagate: per evaluation replicate draw one bootstrap anchor and
     evaluate EB-only and the frozen selective model with it, over the
     released drift grid plus the heterogeneity cells.

Output: artifacts_round4/ec_boot/ec_boot.json
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
from run_external_control import DRIFT_GRID, EFFECT_ALT, EFFECT_NULL, seed_for  # noqa: E402
from selective_methods import eb_only, fit_eb_anchor, load_selective_npz, selective_prior  # noqa: E402

NBOOT = 500


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--replicates", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=20260720)
    ap.add_argument("--train-size", type=int, default=300)
    ap.add_argument("--models-dir", type=str, required=True)
    ap.add_argument("--output", type=str, required=True)
    args = ap.parse_args()
    out = Path(args.output).resolve()
    out.mkdir(parents=True, exist_ok=True)
    mdir = Path(args.models_dir).resolve()

    released = tuple(json.loads((mdir / "anchor_ec.json").read_text())["ec"])
    sel = load_selective_npz(mdir / "ec_selective_sel_pro.npz", 10)

    cfg = ExternalControlConfig(name="ec_train", drift_shift=-0.6, p_comparable=0.5)
    train = simulate_external_control_dataset(cfg, args.train_size, args.seed + 1)
    arms = [(q["y_control"], q["n_control"]) for q in train]
    refit = fit_eb_anchor(arms)
    repro = max(abs(refit[0] - released[0]), abs(refit[1] - released[1]))

    rng = np.random.default_rng(20260901)
    def logit(a, b):
        return math.log(a / b)
    boots = []
    for _ in range(NBOOT):
        idx = rng.integers(0, len(arms), size=len(arms))
        ab = fit_eb_anchor([arms[i] for i in idx])
        boots.append(ab)
    mu0 = logit(released[0], released[1])
    prec0 = released[0] + released[1]
    shifts = np.array([logit(a, b) - mu0 for a, b in boots])
    ratios = np.array([(a + b) / prec0 for a, b in boots])
    dist = {
        "anchor_refit_reproduction_maxdiff": repro,
        "n_boot": NBOOT,
        "logit_mean_shift_q": {q: float(np.quantile(shifts, float(q)))
                               for q in ("0.025", "0.5", "0.975")},
        "precision_ratio_q": {q: float(np.quantile(ratios, float(q)))
                              for q in ("0.025", "0.5", "0.975")},
        "battery_mean_shift": 0.4, "battery_precision_ratio": 4.0,
    }

    boot_idx_seed = 20260902
    cells = [("null", EFFECT_NULL, d, 0.0) for d in DRIFT_GRID] + \
            [("alt", EFFECT_ALT, d, 0.0) for d in DRIFT_GRID] + \
            [("null", EFFECT_NULL, d, 0.3) for d in (-0.8, 0.0)] + \
            [("alt", EFFECT_ALT, d, 0.3) for d in (-0.8, 0.0)]
    rows = []
    for world, effect, drift, het in cells:
        if het > 0:
            c = ExternalControlConfig(name=f"EC_het{het}_{world}_drift{drift}",
                                      treatment_effect=effect, drift_shift=drift,
                                      theta_control_sd=het)
        else:
            c = ExternalControlConfig(name=f"EC_{world}_drift{drift}",
                                      treatment_effect=effect, drift_shift=drift)
        data = simulate_external_control_dataset(
            c, args.replicates, seed_for(c.name, args.seed))
        pick = np.random.default_rng(boot_idx_seed).integers(0, NBOOT, size=len(data))
        rej = {"eb_only_boot": 0, "selective_boot": 0}
        for i, q in enumerate(data):
            ab = boots[pick[i]]
            y_ctl, n_ctl = q["y_control"], q["n_control"]
            y_trt, n_trt = q["y_treatment"], q["n_treatment"]
            for name, prior in (
                ("eb_only_boot", eb_only(q["candidates"], ab)),
                ("selective_boot", selective_prior(sel, q["candidates"], ab,
                                                   prospective=True))):
                p_sup = prior.treatment_effect(y_ctl, n_ctl, y_trt, n_trt)[3]
                rej[name] += p_sup > SUCCESS_THRESHOLD
        for name, v in rej.items():
            rows.append({"world": world, "drift": drift, "het": het,
                         "method": name, "reject_rate": v / len(data)})
        print(f"{world} drift={drift} het={het}: " + " ".join(
            f"{k}={v/len(data):.3f}" for k, v in rej.items()), flush=True)
        (out / "ec_boot.json").write_text(json.dumps(
            {"anchor_sampling_distribution": dist, "propagated": rows}, indent=1))
    print("done", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
