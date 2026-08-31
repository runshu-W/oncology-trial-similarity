"""Round-8: distribution summary of the repeated cross-fit estimates
(review round-8 item 3).

The round-7 script reported mean, SD, and the 2.5/50/97.5% quantiles of the
R = 200 repeated random-split estimates.  This script re-runs the identical
estimation loop with the identical RNG seed (20260908), so the 200 split
estimates are reproduced bit-identically, and emits the fuller summary the
review asks for: the 2.5/25/50/75/97.5% quantiles, the median stated
alongside the mean, and 20-bin histogram counts for the supplement.  The
recomputed mean/SD are cross-checked against the stored round-7 artifact.

Output: artifacts_round4/calibration/round8_crossfit_dist.json
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
RV = ROOT / "artifacts_round4" / "validation" / "pvals"
CAL = ROOT / "artifacts_round4" / "calibration" / "mixture_calibration.json"
R7 = ROOT / "artifacts_round4" / "calibration" / "round7_crossfit.json"
OUT = ROOT / "artifacts_round4" / "calibration" / "round8_crossfit_dist.json"
SHIFTS = (-1.5, -1.25, -1.0, -0.75, -0.5, -0.25,
          0.0, 0.25, 0.5, 0.75, 1.0, 1.25, 1.5)
R = 200
TARGET = 0.02513


def main() -> int:
    W = {float(k): v
         for k, v in json.loads(CAL.read_text())["mixture_weights"].items()}
    nul, alt = {}, {}
    for m in ("weak_only", "cap50"):
        nul[m] = {s: np.load(RV / f"null_s{s}_f1.0.npz")[m] for s in SHIFTS}
        alt[m] = {s: np.load(RV / f"alt_s{s}_f1.0.npz")[m] for s in SHIFTS}
    n = len(nul["weak_only"][0.0])
    half = n // 2

    def tau_half(m, idxmap, target=TARGET):
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

    counts, edges = np.histogram(ests, bins=20)
    res = {"R": int(len(ests)),
           "mean": round(float(ests.mean()), 5),
           "sd": round(float(ests.std(ddof=1)), 5),
           "median": round(float(np.median(ests)), 5),
           "q": {q: round(float(np.quantile(ests, float(q))), 5)
                 for q in ("0.025", "0.25", "0.5", "0.75", "0.975")},
           "frac_positive": round(float((ests > 0).mean()), 4),
           "hist": {"bin_edges": [round(float(e), 6) for e in edges],
                    "counts": [int(c) for c in counts]}}
    r7 = json.loads(R7.read_text())
    res["check_round7"] = {
        "stored_mean": r7["mean"], "stored_sd": r7["sd"],
        "match": bool(res["mean"] == r7["mean"] and res["sd"] == r7["sd"])}
    OUT.write_text(json.dumps(res, indent=1))
    print(json.dumps({k: v for k, v in res.items() if k != "hist"}, indent=1))
    print("hist counts:", res["hist"]["counts"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
