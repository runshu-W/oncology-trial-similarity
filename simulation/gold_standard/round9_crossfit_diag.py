"""Round-9: diagnostics for the repeated cross-fit distribution (review
round-9 item 3), and the repeated cross-fit under alternative mixing
measures.

Part A re-runs the canonical repeated cross-fit with the identical RNG
(20260908), so the 200 split estimates are bit-identical to rounds 7/8,
and records per split and per fold: the size-matching thresholds of both
designs, the achieved weighted size on the estimation half (bisection
residual) and --- the quantity that can actually fail --- the achieved
weighted size of both designs on the *evaluation* half at those
thresholds.  Tail splits (estimate $> +0.001$; the 8 of round 8) are then
characterised: their evaluation-half size gap (cap50 minus no-borrowing),
the correlation between that gap and the estimate over all 200 splits,
and the per-cell decomposition of tail-split estimates versus the rest.

Part B repeats the cross-fit under two narrower mixing measures on the
same 200 splits (identical permutations): N(0,1) and N(0,0.5), with the
size-matching target defined as no borrowing's full-batch weighted
average type I at its frozen threshold (tau_mix = 0.9805) under that
mixing --- the same convention as the round-5 plug-in analysis.

Output: artifacts_round4/calibration/round9_crossfit_diag.json
"""
from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
RV = ROOT / "artifacts_round4" / "validation" / "pvals"
CAL = ROOT / "artifacts_round4" / "calibration" / "mixture_calibration.json"
OUT = ROOT / "artifacts_round4" / "calibration" / "round9_crossfit_diag.json"
SHIFTS = (-1.5, -1.25, -1.0, -0.75, -0.5, -0.25,
          0.0, 0.25, 0.5, 0.75, 1.0, 1.25, 1.5)
R = 200
TAU_FROZEN = 0.9805
TAIL_CUT = 0.001


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


def main() -> int:
    Wcan = {float(k): v
            for k, v in json.loads(CAL.read_text())["mixture_weights"].items()}
    nul, alt = {}, {}
    for m in ("weak_only", "cap50"):
        nul[m] = {s: np.load(RV / f"null_s{s}_f1.0.npz")[m] for s in SHIFTS}
        alt[m] = {s: np.load(RV / f"alt_s{s}_f1.0.npz")[m] for s in SHIFTS}
    n = len(nul["weak_only"][0.0])
    half = n // 2

    def wavg_size(m, W, tau, idxmap=None):
        tot = 0.0
        for s in SHIFTS:
            v = nul[m][s] if idxmap is None else nul[m][s][idxmap[s]]
            tot += W[s] * float((v > tau).mean())
        return tot

    def tau_half(m, W, idxmap, target):
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

    def run(W, target, collect=False):
        rng = np.random.default_rng(20260908)
        ests, diag = [], []
        for r in range(R):
            perm = {s: rng.permutation(n) for s in SHIFTS}
            h0 = {s: perm[s][:half] for s in SHIFTS}
            h1 = {s: perm[s][half:] for s in SHIFTS}
            fold_vals, fdiag = [], []
            for a, b in ((h0, h1), (h1, h0)):
                t_w = tau_half("weak_only", W, a, target)
                t_m = tau_half("cap50", W, a, target)
                if t_w is None or t_m is None:
                    continue
                cells = {}
                d = 0.0
                for s in SHIFTS:
                    cs = W[s] * float(((alt["cap50"][s][b[s]] > t_m).astype(float)
                                       - (alt["weak_only"][s][b[s]] > t_w)
                                       .astype(float)).mean())
                    cells[s] = cs
                    d += cs
                fold_vals.append(d)
                if collect:
                    fdiag.append({
                        "tau_w": t_w, "tau_m": t_m,
                        "size_w_est": wavg_size("weak_only", W, t_w, a),
                        "size_m_est": wavg_size("cap50", W, t_m, a),
                        "size_w_eval": wavg_size("weak_only", W, t_w, b),
                        "size_m_eval": wavg_size("cap50", W, t_m, b),
                        "cells": cells})
            if fold_vals:
                ests.append(float(np.mean(fold_vals)))
                if collect:
                    diag.append(fdiag)
        return np.array(ests), diag

    # ---- part A: canonical, with diagnostics ------------------------------
    target_can = 0.02513
    ests, diag = run(Wcan, target_can, collect=True)
    assert len(ests) == R
    tail = ests > TAIL_CUT
    gap = np.array([np.mean([f["size_m_eval"] - f["size_w_eval"] for f in fd])
                    for fd in diag])
    sw_eval = np.array([np.mean([f["size_w_eval"] for f in fd]) for fd in diag])
    sm_eval = np.array([np.mean([f["size_m_eval"] for f in fd]) for fd in diag])
    taus_w = np.array([f["tau_w"] for fd in diag for f in fd])
    taus_m = np.array([f["tau_m"] for fd in diag for f in fd])
    cells_tail = {str(s): float(np.mean(
        [f["cells"][s] for fd, t in zip(diag, tail) if t for f in fd]))
        for s in SHIFTS}
    cells_rest = {str(s): float(np.mean(
        [f["cells"][s] for fd, t in zip(diag, tail) if not t for f in fd]))
        for s in SHIFTS}
    res = {"R": int(R), "target_canonical": target_can,
           "tail_cut": TAIL_CUT, "n_tail": int(tail.sum()),
           "tail_estimates": [round(float(x), 5) for x in sorted(ests[tail])],
           "corr_estimate_evalsizegap": round(float(np.corrcoef(ests, gap)[0, 1]), 3),
           "eval_size_gap": {
               "tail_mean": round(float(gap[tail].mean()), 5),
               "rest_mean": round(float(gap[~tail].mean()), 5)},
           "eval_size_weak": {
               "tail_mean": round(float(sw_eval[tail].mean()), 5),
               "rest_mean": round(float(sw_eval[~tail].mean()), 5)},
           "eval_size_cap50": {
               "tail_mean": round(float(sm_eval[tail].mean()), 5),
               "rest_mean": round(float(sm_eval[~tail].mean()), 5)},
           "est_size_residual_max": round(float(max(
               abs(f[k] - target_can) for fd in diag for f in fd
               for k in ("size_w_est", "size_m_est"))), 6),
           "tau_w": {"min": round(float(taus_w.min()), 5),
                     "max": round(float(taus_w.max()), 5),
                     "n_unique": int(len(np.unique(np.round(taus_w, 6))))},
           "tau_m": {"min": round(float(taus_m.min()), 5),
                     "max": round(float(taus_m.max()), 5),
                     "n_unique": int(len(np.unique(np.round(taus_m, 6))))},
           "cells_tail_mean_contrib": cells_tail,
           "cells_rest_mean_contrib": cells_rest}

    # ---- part B: alternative mixings on the identical splits --------------
    res["alt_mixings"] = {}
    for name, W in (("N(0,1)", normal_weights(0.0, 1.0)),
                    ("N(0,0.5)", normal_weights(0.0, 0.5))):
        tgt = wavg_size("weak_only", W, TAU_FROZEN)
        e, _ = run(W, tgt, collect=False)
        res["alt_mixings"][name] = {
            "target": round(float(tgt), 5), "R": int(len(e)),
            "mean": round(float(e.mean()), 5),
            "median": round(float(np.median(e)), 5),
            "sd": round(float(e.std(ddof=1)), 5),
            "q": {q: round(float(np.quantile(e, float(q))), 5)
                  for q in ("0.025", "0.25", "0.5", "0.75", "0.975")},
            "frac_positive": round(float((e > 0).mean()), 4)}
        print(name, res["alt_mixings"][name], flush=True)

    # store the canonical estimates for the supplement figure
    res["canonical_estimates"] = [round(float(x), 6) for x in ests]
    OUT.write_text(json.dumps(res, indent=1))
    print(json.dumps({k: v for k, v in res.items()
                      if k not in ("canonical_estimates", "cells_tail_mean_contrib",
                                   "cells_rest_mean_contrib")}, indent=1))
    print("tail cells:", {k: round(v, 4) for k, v in cells_tail.items()})
    print("rest cells:", {k: round(v, 4) for k, v in cells_rest.items()})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
