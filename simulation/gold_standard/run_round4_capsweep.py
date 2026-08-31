"""Round-4 cap sweep (review round-4 Major 1).

For each borrowed-mass cap c in {0.1, ..., 0.9} this runner stores the
PER-REPLICATE posterior probability for the capped selective prior with and
without the anchored SAM companion, over the same 32-cell 2-D conflict grid
and the SAME released per-cell seeds as the round-3 calibration study
(identical datasets; the c=0.5 columns must reproduce the round-3
selective_cap50 / selective_cap50_sam pvals bit for bit).

The base selective prior is built ONCE per replicate and the cap projection /
SAM adapter are applied per cap, so the sweep costs little more than one
selective pass.  Offline calibration (tau*, restricted-set tau, tipping
point, power) is done by calibrate_round4.py.

Usage:
  python3 run_round4_capsweep.py --replicates 2000 --models-dir <dir> \
      --output <dir> [--shard k --nshards N]
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from dgm import simulate_dataset  # noqa: E402
from metrics import THETA_NULL  # noqa: E402
from run_simulation import CONFLICT_GRID, seed_for  # noqa: E402
from run_round3_calibration import (  # noqa: E402
    FRACTIONS, PARTIAL_SHIFTS, cap_prior, design_cfg,
)
from selective_methods import load_selective_npz, selective_prior, apply_sam_anchored  # noqa: E402

CAPS = (0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--replicates", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=20260720)
    ap.add_argument("--models-dir", type=str, required=True)
    ap.add_argument("--output", type=str, required=True)
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--nshards", type=int, default=1)
    args = ap.parse_args()

    out = Path(args.output).resolve()
    (out / "pvals").mkdir(parents=True, exist_ok=True)
    mdir = Path(args.models_dir).resolve()
    anchor = tuple(json.loads((mdir / "anchor.json").read_text())["gold"])
    sel = load_selective_npz(mdir / "selective_sel_pro.npz", 10)

    cells = []
    for world in ("null", "alt"):
        for s in CONFLICT_GRID:
            cells.append((world, float(s), 1.0))
        for s in PARTIAL_SHIFTS:
            for f in FRACTIONS:
                cells.append((world, float(s), float(f)))
    cells = [c for i, c in enumerate(cells) if i % args.nshards == args.shard]

    keys = [f"cap{int(round(c * 100))}{suf}" for c in CAPS for suf in ("", "_sam")]
    t0 = time.time()
    for world, s, f in cells:
        tag = f"{world}_s{s}_f{f}"
        part = out / "pvals" / f"{tag}.npz"
        if part.exists():
            print(f"{tag} cached", flush=True)
            continue
        cfg = design_cfg(world, s, f)
        data = simulate_dataset(cfg, args.replicates, seed_for(cfg.name, args.seed))
        store = {k: np.empty(len(data)) for k in keys}
        for i, q in enumerate(data):
            cands, y0, n0 = q["candidates"], q["y_query"], q["n_query"]
            base = selective_prior(sel, cands, anchor, prospective=True)
            for c in CAPS:
                capped = cap_prior(base, c)
                key = f"cap{int(round(c * 100))}"
                store[key][i] = capped.posterior_prob_greater(y0, n0, THETA_NULL)
                sam = apply_sam_anchored(capped, y0, n0)[0]
                store[key + "_sam"][i] = sam.posterior_prob_greater(y0, n0, THETA_NULL)
        np.savez_compressed(part, **store)
        print(f"[{time.time()-t0:6.0f}s] {tag}: cap30_sam@0.975="
              f"{(store['cap30_sam'] > 0.975).mean():.3f} cap50_sam@0.975="
              f"{(store['cap50_sam'] > 0.975).mean():.3f}", flush=True)
    print("done", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
