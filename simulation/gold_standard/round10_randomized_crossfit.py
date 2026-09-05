"""Round-10: repeated cross-fit with EXACT expected-size matching via
randomized boundaries, at a common target across mixing measures (review
round-10 item 1).

The round-7/8/9 repeated cross-fit solved a single deterministic threshold
per training half by bisection.  Because the no-borrowing test statistic is
discrete (four distinct threshold atoms over 400 fold-halves), the achieved
evaluation-half size differed from the target by up to a few thousandths
--- comparable to the effect being estimated, and the round-9 diagnostics
showed the split estimate tracking that residual gap (r = 0.91).  This
script removes the granularity: on each training half and for each design
it constructs the classical randomized test

    reject if v > c;  if v = c reject with probability gamma,

with (c, gamma) chosen so the weighted average type I on the training half
equals the target EXACTLY, and applies (c, gamma) to the evaluation half in
expectation (the rejection functional is P(v > c) + gamma P(v = c), i.e.
the convex combination between the two adjacent achievable tests --- no
Monte Carlo randomization noise is added).  Both designs are matched to
the SAME target under every mixing:

    targets 0.02513 (the paper's canonical convention) and 0.025
    (nominal), each applied to the canonical, N(0,1), and N(0,0.5)
    mixings on the identical 200 splits (RNG 20260908).

Output: artifacts_round4/calibration/round10_randomized_crossfit.json
"""
from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
RV = ROOT / "artifacts_round4" / "validation" / "pvals"
CAL = ROOT / "artifacts_round4" / "calibration" / "mixture_calibration.json"
OUT = ROOT / "artifacts_round4" / "calibration" / "round10_randomized_crossfit.json"
SHIFTS = (-1.5, -1.25, -1.0, -0.75, -0.5, -0.25,
          0.0, 0.25, 0.5, 0.75, 1.0, 1.25, 1.5)
R = 200
TARGETS = (0.02513, 0.025)


def normal_weights(mu, sd):
    def Phi(x):
        return 0.5 * (1 + math.erf(x / math.sqrt(2)))
    step = 0.25
    w = []
    for i, s in enumerate(SHIFTS):
        lo = -np.inf if i == 0 else s - step / 2
        hi = np.inf if i == len(SHIFTS) - 1 else s + step / 2
        plo = 0.0 if lo == -np.inf else Phi((lo - mu) / sd)
        phi = 1.0 if hi == np.inf else Phi((hi - mu) / sd)
        w.append(phi - plo)
    w = np.array(w)
    return {s: float(x) for s, x in zip(SHIFTS, w / w.sum())}


def rand_threshold(vals_by_cell, W, target):
    """(c, gamma) with training-half weighted expected size == target."""
    vs = np.concatenate([vals_by_cell[s] for s in SHIFTS])
    ws = np.concatenate([np.full(len(vals_by_cell[s]), W[s] / len(vals_by_cell[s]))
                         for s in SHIFTS])
    order = np.argsort(-vs, kind="stable")
    v_sorted = vs[order]
    csum = np.cumsum(ws[order])
    idx0 = int(np.searchsorted(csum, target, side="right"))
    if idx0 >= len(v_sorted):
        idx0 = len(v_sorted) - 1
    u = v_sorted[idx0]
    neg = -v_sorted
    left = int(np.searchsorted(neg, -u, side="left"))
    right = int(np.searchsorted(neg, -u, side="right"))
    s_above = float(csum[left - 1]) if left > 0 else 0.0
    atom = float(csum[right - 1] - s_above)
    gamma = 0.0 if atom <= 0 else (target - s_above) / atom
    return float(u), float(min(max(gamma, 0.0), 1.0))


def expected_reject(arrs_by_cell, idxmap, W, c, gamma):
    tot = 0.0
    for s in SHIFTS:
        a = arrs_by_cell[s][idxmap[s]]
        tot += W[s] * (float((a > c).mean()) + gamma * float((a == c).mean()))
    return tot


def main() -> int:
    Wcan = {float(k): v
            for k, v in json.loads(CAL.read_text())["mixture_weights"].items()}
    mixings = {"canonical": Wcan,
               "N(0,1)": normal_weights(0.0, 1.0),
               "N(0,0.5)": normal_weights(0.0, 0.5)}
    nul, alt = {}, {}
    for m in ("weak_only", "cap50"):
        nul[m] = {s: np.load(RV / f"null_s{s}_f1.0.npz")[m] for s in SHIFTS}
        alt[m] = {s: np.load(RV / f"alt_s{s}_f1.0.npz")[m] for s in SHIFTS}
    n = len(nul["weak_only"][0.0])
    half = n // 2

    res = {"R": R, "targets": list(TARGETS), "runs": {}}
    for tgt in TARGETS:
        for name, W in mixings.items():
            rng = np.random.default_rng(20260908)
            ests, sw, sm = [], [], []
            for _ in range(R):
                perm = {s: rng.permutation(n) for s in SHIFTS}
                h0 = {s: perm[s][:half] for s in SHIFTS}
                h1 = {s: perm[s][half:] for s in SHIFTS}
                fold_vals, fsw, fsm = [], [], []
                for a, b in ((h0, h1), (h1, h0)):
                    trn_w = {s: nul["weak_only"][s][a[s]] for s in SHIFTS}
                    trn_m = {s: nul["cap50"][s][a[s]] for s in SHIFTS}
                    c_w, g_w = rand_threshold(trn_w, W, tgt)
                    c_m, g_m = rand_threshold(trn_m, W, tgt)
                    pw_m = expected_reject(alt["cap50"], b, W, c_m, g_m)
                    pw_w = expected_reject(alt["weak_only"], b, W, c_w, g_w)
                    fold_vals.append(pw_m - pw_w)
                    fsw.append(expected_reject(nul["weak_only"], b, W, c_w, g_w))
                    fsm.append(expected_reject(nul["cap50"], b, W, c_m, g_m))
                ests.append(float(np.mean(fold_vals)))
                sw.append(float(np.mean(fsw)))
                sm.append(float(np.mean(fsm)))
            e = np.array(ests)
            sw, sm = np.array(sw), np.array(sm)
            gap = sm - sw
            key = f"{name}@{tgt}"
            res["runs"][key] = {
                "mean": round(float(e.mean()), 5),
                "median": round(float(np.median(e)), 5),
                "sd": round(float(e.std(ddof=1)), 5),
                "q": {q: round(float(np.quantile(e, float(q))), 5)
                      for q in ("0.025", "0.25", "0.5", "0.75", "0.975")},
                "frac_positive": round(float((e > 0).mean()), 4),
                "eval_size_weak_mean": round(float(sw.mean()), 5),
                "eval_size_cap50_mean": round(float(sm.mean()), 5),
                "eval_size_gap_mean": round(float(gap.mean()), 6),
                "eval_size_gap_sd": round(float(gap.std(ddof=1)), 6),
                "corr_estimate_gap": round(float(np.corrcoef(e, gap)[0, 1]), 3),
            }
            if name == "canonical" and tgt == 0.02513:
                res["canonical_estimates"] = [round(float(x), 6) for x in e]
            print(key, res["runs"][key], flush=True)
    OUT.write_text(json.dumps(res, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
