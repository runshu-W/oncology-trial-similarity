"""Round-4 INDEPENDENT validation batch (review round-4 Major 2).

Thresholds and the final design are selected on the 2000-replicate SELECTION
batch (round-3 store + round-4 sweep, seed base 20260720).  This runner
re-simulates every cell with a FRESH seed base and (by default) 5x more
replicates, and stores per-replicate posterior probabilities for the base
designs and the requested cap variants.  Offline validation
(validate_round4.py) then reports, at the PRE-SELECTED thresholds:
per-cell type I error, the worst cell with its Clopper-Pearson 95% upper
bound, mixture-average error, and power - all computed on data never used
for selection.

Cells: uniform two-sided shifts {-1.5..1.5} x {null, alt} plus the 9
partial-conflict cells per world (44 cells).
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
from methods import TwoHeadDeepSets, apply_sam, rule_components, two_head_prior, weak_only  # noqa: E402
from run_simulation import seed_for  # noqa: E402
from run_round3_calibration import FRACTIONS, PARTIAL_SHIFTS, cap_prior, design_cfg  # noqa: E402
from run_round4_negcells import NEG_SHIFTS  # noqa: E402
from selective_methods import (  # noqa: E402
    apply_sam_anchored, eb_only, load_selective_npz, selective_prior,
)

POS_SHIFTS = (0.0, 0.25, 0.5, 0.75, 1.0, 1.25, 1.5)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--replicates", type=int, default=10000)
    ap.add_argument("--seed", type=int, default=20260901, help="FRESH base")
    ap.add_argument("--caps", type=str, default="30,40,50,60")
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
    pro = TwoHeadDeepSets(input_dim=10)
    z = np.load(mdir / "two_head_pro.npz")
    pro.load([z[f"p{i}"] for i in range(len(pro.params))])
    caps = tuple(int(c) / 100 for c in args.caps.split(","))

    cells = []
    for world in ("null", "alt"):
        for s in tuple(sorted(NEG_SHIFTS)) + POS_SHIFTS:
            cells.append((world, float(s), 1.0))
        for s in PARTIAL_SHIFTS:
            for f in FRACTIONS:
                cells.append((world, float(s), float(f)))
    cells = [c for i, c in enumerate(cells) if i % args.nshards == args.shard]

    keys = (["weak_only", "eb_only", "rule_sam", "two_head_pro_sam",
             "selective", "selective_sam"]
            + [f"cap{int(round(c * 100))}{suf}" for c in caps for suf in ("", "_sam")])
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
            store["weak_only"][i] = weak_only(cands).posterior_prob_greater(y0, n0, THETA_NULL)
            store["eb_only"][i] = eb_only(cands, anchor).posterior_prob_greater(y0, n0, THETA_NULL)
            store["rule_sam"][i] = apply_sam(rule_components(cands), y0, n0)[0] \
                .posterior_prob_greater(y0, n0, THETA_NULL)
            store["two_head_pro_sam"][i] = apply_sam(
                two_head_prior(pro, cands, prospective=True), y0, n0)[0] \
                .posterior_prob_greater(y0, n0, THETA_NULL)
            base = selective_prior(sel, cands, anchor, prospective=True)
            store["selective"][i] = base.posterior_prob_greater(y0, n0, THETA_NULL)
            store["selective_sam"][i] = apply_sam_anchored(base, y0, n0)[0] \
                .posterior_prob_greater(y0, n0, THETA_NULL)
            for c in caps:
                capped = cap_prior(base, c)
                key = f"cap{int(round(c * 100))}"
                store[key][i] = capped.posterior_prob_greater(y0, n0, THETA_NULL)
                store[key + "_sam"][i] = apply_sam_anchored(capped, y0, n0)[0] \
                    .posterior_prob_greater(y0, n0, THETA_NULL)
        np.savez_compressed(part, **store)
        print(f"[{time.time()-t0:6.0f}s] {tag} n={len(data)}", flush=True)
    print("done", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
