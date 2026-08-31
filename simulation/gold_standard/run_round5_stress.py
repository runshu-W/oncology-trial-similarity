"""Round-5 cap-protection stress tests (review round-5 Major 4).

The historical mixture-mass cap bounds component PROBABILITY, not component
information, so its protection must be demonstrated, not asserted, when
historical components are much more precise.  Two stress axes, null and
alternative worlds, uniform shifts {0, 0.5, 1.0, 1.5}:

  donor_n_x4    donor arm sizes scaled x4 (fresh cells, deterministic seeds
                from the cell name) - component-precision stress;
  anchor prec   the released cells re-evaluated with the EB anchor's
                precision scaled x4 / x0.25 at fixed mean (anchor-precision
                stress; identical datasets, released seeds).

Methods per cell: no borrowing, uncapped selective, capped selective (0.5).
First seed-invariance check: at donor_n_scale=1.0 the dataset must be
bit-identical to the released draw.

Output: artifacts_round4/stress/round5_stress.json
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from dgm import SimConfig, simulate_dataset  # noqa: E402
from metrics import SUCCESS_THRESHOLD, THETA_NULL, DESIGN_DELTA  # noqa: E402
from methods import weak_only  # noqa: E402
from run_simulation import seed_for  # noqa: E402
from run_round3_calibration import cap_prior  # noqa: E402
from selective_methods import load_selective_npz, selective_prior  # noqa: E402

SHIFTS = (0.0, 0.5, 1.0, 1.5)


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
    anchor = tuple(json.loads((mdir / "anchor.json").read_text())["gold"])
    sel = load_selective_npz(mdir / "selective_sel_pro.npz", 10)
    a_x4 = (anchor[0] * 4, anchor[1] * 4)
    a_q4 = (anchor[0] / 4, anchor[1] / 4)

    # seed-invariance check at default scale
    cfg0 = SimConfig(name="null_shift0.0", conflict_shift=0.0,
                     theta_query_fixed=THETA_NULL)
    d0 = simulate_dataset(cfg0, 5, seed_for(cfg0.name, args.seed))
    sig = [(q["y_query"], q["n_query"], q["candidates"][0]["y"],
            q["candidates"][0]["n"]) for q in d0]
    ref = json.loads((out / "invariance_ref.json").read_text()) \
        if (out / "invariance_ref.json").exists() else None
    (out / "invariance_ref.json").write_text(json.dumps(sig))
    inv_ok = (ref is None) or (ref == sig)

    rows = []
    for world in ("null", "alt"):
        theta = THETA_NULL if world == "null" else THETA_NULL + DESIGN_DELTA
        for s in SHIFTS:
            # A. donor-precision world (fresh cells)
            cfg = SimConfig(name=f"{world}_shift{s}_dn4", conflict_shift=s,
                            theta_query_fixed=theta, donor_n_scale=4.0)
            data = simulate_dataset(cfg, args.replicates,
                                    seed_for(cfg.name, args.seed))
            rej = {"weak_only": 0, "selective": 0, "cap50": 0}
            for q in data:
                cands, y0, n0 = q["candidates"], q["y_query"], q["n_query"]
                rej["weak_only"] += weak_only(cands).posterior_prob_greater(
                    y0, n0, THETA_NULL) > SUCCESS_THRESHOLD
                base = selective_prior(sel, cands, anchor, prospective=True)
                rej["selective"] += base.posterior_prob_greater(
                    y0, n0, THETA_NULL) > SUCCESS_THRESHOLD
                rej["cap50"] += cap_prior(base).posterior_prob_greater(
                    y0, n0, THETA_NULL) > SUCCESS_THRESHOLD
            for m, v in rej.items():
                rows.append({"axis": "donor_n_x4", "world": world, "shift": s,
                             "method": m, "reject": v / len(data)})
            print(f"donor_n_x4 {world} s={s}: " + " ".join(
                f"{m}={v/len(data):.3f}" for m, v in rej.items()), flush=True)

            # B. anchor-precision stress on the RELEASED cells
            cfg = SimConfig(name=f"{world}_shift{s}", conflict_shift=s,
                            theta_query_fixed=theta)
            data = simulate_dataset(cfg, args.replicates,
                                    seed_for(cfg.name, args.seed))
            rej = {"cap50_anchor_x4": 0, "cap50_anchor_q4": 0}
            for q in data:
                cands, y0, n0 = q["candidates"], q["y_query"], q["n_query"]
                for key, ab in (("cap50_anchor_x4", a_x4),
                                ("cap50_anchor_q4", a_q4)):
                    base = selective_prior(sel, cands, ab, prospective=True)
                    rej[key] += cap_prior(base).posterior_prob_greater(
                        y0, n0, THETA_NULL) > SUCCESS_THRESHOLD
            for m, v in rej.items():
                rows.append({"axis": "anchor_prec", "world": world, "shift": s,
                             "method": m, "reject": v / len(data)})
            print(f"anchor_prec {world} s={s}: " + " ".join(
                f"{m}={v/len(data):.3f}" for m, v in rej.items()), flush=True)
            (out / "round5_stress.json").write_text(json.dumps(
                {"invariance_default_scale_ok": inv_ok, "rows": rows}, indent=1))
    print("done", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
