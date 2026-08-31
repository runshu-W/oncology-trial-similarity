"""Round-7: repeated random cross-fitting for the size-adjusted difference
(review round-7 item 3), plus the pointwise diagnostic extended to negative
and partial-conflict cells (item 4).

A single two-fold split leaves split- and threshold-estimation uncertainty
uncharacterised.  This script repeats the cross-fit under R random
half-splits of each cell's validation replicates: per split, size-matching
thresholds (average type I = no-borrowing's validated 0.02513 under the
canonical mixing) are solved by bisection on one half and the paired power
difference is evaluated on the other, folds averaged; the distribution of
the R estimates characterises split/threshold-estimation variability.

Output: artifacts_round4/calibration/round7_crossfit.json
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

ROOT = Path("/root/work/osim")
RV = ROOT / "artifacts_round4" / "validation" / "pvals"
CAL = ROOT / "artifacts_round4" / "calibration" / "mixture_calibration.json"
OUT = ROOT / "artifacts_round4" / "calibration" / "round7_crossfit.json"
SHIFTS = (-1.5, -1.25, -1.0, -0.75, -0.5, -0.25,
          0.0, 0.25, 0.5, 0.75, 1.0, 1.25, 1.5)
PARTIAL = [(s, f) for s in (0.5, 1.0, 1.5) for f in (0.25, 0.5, 0.75)]
R = 200
TARGET = 0.02513


def main() -> int:
    W = {float(k): v for k, v in json.loads(CAL.read_text())["mixture_weights"].items()}
    nul, alt = {}, {}
    for m in ("weak_only", "cap50"):
        nul[m] = {s: np.load(RV / f"null_s{s}_f1.0.npz")[m] for s in SHIFTS}
        alt[m] = {s: np.load(RV / f"alt_s{s}_f1.0.npz")[m] for s in SHIFTS}
    n = len(nul["weak_only"][0.0])
    half = n // 2

    def tau_half(m, idxmap, target=TARGET):
        """Bisection for the tau whose weighted avg type I = target on the
        given per-cell index subsets."""
        srt = {s: np.sort(nul[m][s][idxmap[s]]) for s in SHIFTS}
        nn = len(next(iter(idxmap.values())))

        def size_at(tau):
            return sum(
                W[s] * (nn - np.searchsorted(srt[s], tau, side="right")) / nn
                for s in SHIFTS)

        lo, hi = 0.970, 0.99995
        if size_at(lo) < target or size_at(hi) > target:
            return None
        for _ in range(50):
            mid = 0.5 * (lo + hi)
            if size_at(mid) >= target:
                lo = mid
            else:
                hi = mid
        return 0.5 * (lo + hi)

    rng = np.random.default_rng(20260908)
    ests = []
    for _ in range(R):
        perm = {s: rng.permutation(n) for s in SHIFTS}
        h0 = {s: perm[s][:half] for s in SHIFTS}
        h1 = {s: perm[s][half:] for s in SHIFTS}
        fold_vals = []
        for a, b in ((h0, h1), (h1, h0)):
            t_w = tau_half("weak_only", a)
            t_m = tau_half("cap50", a)
            if t_w is None or t_m is None:
                continue
            d = sum(
                W[s] * float(((alt["cap50"][s][b[s]] > t_m).astype(float)
                              - (alt["weak_only"][s][b[s]] > t_w)
                              .astype(float)).mean())
                for s in SHIFTS)
            fold_vals.append(d)
        if fold_vals:
            ests.append(float(np.mean(fold_vals)))
    ests = np.array(ests)
    res = {"R": int(len(ests)),
           "mean": round(float(ests.mean()), 5),
           "sd": round(float(ests.std(ddof=1)), 5),
           "q": {q: round(float(np.quantile(ests, float(q))), 5)
                 for q in ("0.025", "0.5", "0.975")},
           "frac_positive": round(float((ests > 0).mean()), 4),
           "full_batch_estimate": 0.00242,
           "single_crossfit_estimate": 0.00574}

    # ---- pointwise diagnostic over ALL cells (uniform + partial) ----------
    prng = np.random.default_rng(20260909)
    rows = {}
    for s, f in [(x, 1.0) for x in SHIFTS] + PARTIAL:
        nl = {m: np.load(RV / f"null_s{s}_f{f}.npz")[m]
              for m in ("weak_only", "cap50")}
        al = {m: np.load(RV / f"alt_s{s}_f{f}.npz")[m]
              for m in ("weak_only", "cap50")}
        perm = prng.permutation(n)
        vals = []
        for a, b in ((perm[:half], perm[half:]), (perm[half:], perm[:half])):
            t_m = float(np.quantile(nl["cap50"][a], 0.975))
            t_w = float(np.quantile(nl["weak_only"][a], 0.975))
            dd = ((al["cap50"][b] > t_m).astype(float)
                  - (al["weak_only"][b] > t_w).astype(float))
            vals.append(float(dd.mean()))
        tag = f"s{s}" if f == 1.0 else f"s{s}_f{f}"
        rows[tag] = round(float(np.mean(vals)), 5)
    res["pointwise_all_cells"] = rows
    OUT.write_text(json.dumps(res, indent=1))
    print(json.dumps({k: v for k, v in res.items()
                      if k != "pointwise_all_cells"}, indent=1))
    print("pointwise:", rows)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
