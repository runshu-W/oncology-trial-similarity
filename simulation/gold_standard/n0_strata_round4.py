"""Round-4: query-size-conditional operating characteristics.

n_query is ancillary (fixed by design in a real trial), so conditional OCs
by n0 stratum are the decision-relevant view: borrowing should matter most
where the internal arm is smallest.  Re-simulates the uniform cells (released
seed base -> identical datasets, replicate order matches the stored pvals),
recovers each replicate's n0, and reports mixture-average type I / power per
n0 tertile at each design's GLOBAL mixture-calibrated threshold.

Usage: python3 n0_strata_round4.py [--batch selection|validation]
Output: artifacts_round4/calibration/n0_strata_<batch>.json
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from dgm import simulate_dataset  # noqa: E402
from run_simulation import seed_for  # noqa: E402
from run_round3_calibration import design_cfg  # noqa: E402

ROOT = Path("/root/work/osim")
SHIFTS = (-1.5, -1.25, -1.0, -0.75, -0.5, -0.25,
          0.0, 0.25, 0.5, 0.75, 1.0, 1.25, 1.5)
STRATA = ((20, 28), (29, 37), (38, 45))
DESIGNS = {  # method -> (store hint handled by loader), tau_mix from selection
    "weak_only": 0.9805, "eb_only": 0.975,
    "selective_cap50": 0.9755, "selective_cap50_sam": 0.978,
}


def loader(batch):
    R3 = ROOT / "artifacts_round3" / "calibration" / "pvals"
    R4N = ROOT / "artifacts_round4" / "negcells" / "pvals"
    RV = ROOT / "artifacts_round4" / "validation" / "pvals"

    def load(m, world, s):
        tag = f"{world}_s{s}_f1.0"
        if batch == "validation":
            key = m.replace("selective_cap50", "cap50")
            return np.load(RV / f"{tag}.npz")[key]
        if s < 0:
            return np.load(R4N / f"{tag}.npz")[m.replace("selective_cap50", "cap50")]
        return np.load(R3 / f"{tag}.npz")[m]
    return load


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--batch", choices=("selection", "validation"),
                    default="selection")
    ap.add_argument("--replicates", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=20260720)
    args = ap.parse_args()
    if args.batch == "validation":
        args.replicates, args.seed = 10000, 20260901

    W = json.loads((ROOT / "artifacts_round4" / "calibration" /
                    "mixture_calibration.json").read_text())["mixture_weights"]
    load = loader(args.batch)

    n0 = {}
    for world in ("null", "alt"):
        for s in SHIFTS:
            cfg = design_cfg(world, float(s), 1.0)
            data = simulate_dataset(cfg, args.replicates,
                                    seed_for(cfg.name, args.seed))
            n0[(world, s)] = np.array([q["n_query"] for q in data])

    out = {"batch": args.batch, "strata": {}, "tau_mix": DESIGNS}
    for lo, hi in STRATA:
        key = f"n{lo}-{hi}"
        entry = {}
        for m, tau in DESIGNS.items():
            t1 = pw = wt = 0.0
            for s in SHIFTS:
                w = W[str(float(s))]
                msk_n = (n0[("null", s)] >= lo) & (n0[("null", s)] <= hi)
                msk_a = (n0[("alt", s)] >= lo) & (n0[("alt", s)] <= hi)
                t1 += w * float((load(m, "null", s)[msk_n] > tau).mean())
                pw += w * float((load(m, "alt", s)[msk_a] > tau).mean())
                wt += w
            entry[m] = {"avg_t1": round(t1 / wt, 4), "mix_power": round(pw / wt, 4)}
        entry["net_benefit_cap50_vs_weak"] = round(
            entry["selective_cap50"]["mix_power"] - entry["weak_only"]["mix_power"], 4)
        entry["net_benefit_cap50_vs_eb"] = round(
            entry["selective_cap50"]["mix_power"] - entry["eb_only"]["mix_power"], 4)
        out["strata"][key] = entry
        print(key, json.dumps(entry))
    p = (ROOT / "artifacts_round4" / "calibration" /
         f"n0_strata_{args.batch}.json")
    p.write_text(json.dumps(out, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
