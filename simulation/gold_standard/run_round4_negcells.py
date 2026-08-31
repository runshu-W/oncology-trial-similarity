"""Round-4 negative-shift cells (two-sided drift for the empirical-mixture
calibration).

The round-2/3 conflict grid tested only UPWARD donor drift (worst case for
type I error).  The round-4 empirical drift distribution is two-sided, so the
mixture-average operating characteristics need the downward cells too:
uniform conflict_shift in {-1.5,...,-0.25}, both worlds, fresh deterministic
seeds via the cell name (never released before).

Stores per-replicate posterior probabilities for the 6 round-3 base designs
plus the 18 cap-sweep variants, one npz per cell, same layout as the other
pval stores.
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
from run_round3_calibration import cap_prior, design_cfg  # noqa: E402
from run_round4_capsweep import CAPS  # noqa: E402
from selective_methods import (  # noqa: E402
    apply_sam_anchored, eb_only, load_selective_npz, selective_prior,
)

NEG_SHIFTS = (-1.5, -1.25, -1.0, -0.75, -0.5, -0.25)


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
    pro = TwoHeadDeepSets(input_dim=10)
    z = np.load(mdir / "two_head_pro.npz")
    pro.load([z[f"p{i}"] for i in range(len(pro.params))])

    cells = [(w, float(s), 1.0) for w in ("null", "alt") for s in NEG_SHIFTS]
    cells = [c for i, c in enumerate(cells) if i % args.nshards == args.shard]

    keys = (["weak_only", "eb_only", "rule_sam", "two_head_pro_sam",
             "selective", "selective_sam"]
            + [f"cap{int(round(c * 100))}{suf}" for c in CAPS for suf in ("", "_sam")])
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
            for c in CAPS:
                capped = cap_prior(base, c)
                key = f"cap{int(round(c * 100))}"
                store[key][i] = capped.posterior_prob_greater(y0, n0, THETA_NULL)
                store[key + "_sam"][i] = apply_sam_anchored(capped, y0, n0)[0] \
                    .posterior_prob_greater(y0, n0, THETA_NULL)
        np.savez_compressed(part, **store)
        print(f"[{time.time()-t0:6.0f}s] {tag}: sel@0.975="
              f"{(store['selective'] > 0.975).mean():.3f} cap50_sam@0.975="
              f"{(store['cap50_sam'] > 0.975).mean():.3f} weak@0.975="
              f"{(store['weak_only'] > 0.975).mean():.3f}", flush=True)
    print("done", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
