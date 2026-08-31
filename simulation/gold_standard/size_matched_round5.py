"""Round-5: size-matched power comparison on the independent validation batch
(review round-5 Major 2).

The frozen thresholds gave the final design a validated average type I of
0.0258 versus 0.0251 for calibrated no borrowing --- so part of the +0.0066
mixture-power difference could be bought by the extra 0.0007 of size.  This
script eliminates that objection by comparing on the SIZE-POWER CURVE:

  for each design, sweep the threshold over a fine grid, computing the
  drift-weighted average type I (size) and mixture power on the validation
  batch; report power at interpolated thresholds where every design's
  average type I EQUALS (a) no-borrowing's validated 0.0251, (b) the nominal
  0.025, and (c) the largest tau whose MC 95% UCB of average type I is
  <= 0.025 (a conservative calibration); paired within-replicate power
  differences are recomputed at the matched thresholds.

Nothing here re-selects the design; it is a fairness re-read of the same
frozen validation data.  Output: artifacts_round4/calibration/
size_matched_round5.json and figure F15 source values.
"""
from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np

ROOT = Path("/root/work/osim")
RV = ROOT / "artifacts_round4" / "validation" / "pvals"
CAL = ROOT / "artifacts_round4" / "calibration" / "mixture_calibration.json"
OUT = ROOT / "artifacts_round4" / "calibration" / "size_matched_round5.json"

SHIFTS = (-1.5, -1.25, -1.0, -0.75, -0.5, -0.25,
          0.0, 0.25, 0.5, 0.75, 1.0, 1.25, 1.5)
MAP = {"selective_cap50": "cap50", "selective_cap50_sam": "cap50_sam"}
DESIGNS = ("weak_only", "eb_only", "cap10", "cap30", "selective_cap50",
           "selective_cap50_sam", "selective_sam")


def main() -> int:
    W = {float(k): v for k, v in json.loads(CAL.read_text())["mixture_weights"].items()}
    nul, alt = {}, {}
    for m in DESIGNS:
        key = MAP.get(m, m)
        nul[m] = {s: np.load(RV / f"null_s{s}_f1.0.npz")[key] for s in SHIFTS}
        alt[m] = {s: np.load(RV / f"alt_s{s}_f1.0.npz")[key] for s in SHIFTS}
    n = len(nul[DESIGNS[0]][0.0])

    grid = np.unique(np.round(np.concatenate([
        np.arange(0.970, 0.9990, 0.00025), np.arange(0.9990, 0.99991, 0.00005)]), 6))

    def size_power(m, tau):
        t1 = sum(W[s] * float((nul[m][s] > tau).mean()) for s in SHIFTS)
        pw = sum(W[s] * float((alt[m][s] > tau).mean()) for s in SHIFTS)
        return t1, pw

    curves = {}
    for m in DESIGNS:
        arr = [(float(t),) + size_power(m, float(t)) for t in grid]
        curves[m] = arr

    def tau_at_size(m, target):
        """Largest tau with avg T1 >= target crossing; linear interp on curve."""
        arr = curves[m]
        # avg T1 decreases in tau; find bracketing pair
        for i in range(len(arr) - 1):
            t0, s0, _ = arr[i]
            t1_, s1, _ = arr[i + 1]
            if s0 >= target >= s1:
                if s0 == s1:
                    return t0
                w = (s0 - target) / (s0 - s1)
                return t0 + w * (t1_ - t0)
        return None

    def power_at_tau(m, tau):
        return sum(W[s] * float((alt[m][s] > tau).mean()) for s in SHIFTS)

    def size_at_tau(m, tau):
        return sum(W[s] * float((nul[m][s] > tau).mean()) for s in SHIFTS)

    def ucb_size(m, tau):
        # MC 95% (one-sided) upper bound on the weighted average via normal
        # approx with exact per-cell binomial variances (weights fixed).
        t1 = 0.0
        var = 0.0
        for s in SHIFTS:
            p = float((nul[m][s] > tau).mean())
            t1 += W[s] * p
            var += (W[s] ** 2) * p * (1 - p) / n
        return t1 + 1.645 * math.sqrt(var)

    def tau_at_ucb(m, target=0.025):
        arr = curves[m]
        for i in range(len(arr)):
            t = arr[i][0]
            if ucb_size(m, t) <= target:
                return t
        return None

    weak_valid = size_at_tau("weak_only", 0.9805)
    targets = {"match_weak_validated": weak_valid, "nominal_0.025": 0.025}
    res = {"n_per_cell": n, "weak_validated_size": round(weak_valid, 5),
           "at": {}, "ucb_calibrated": {}, "curves_coarse": {}}

    for name, target in targets.items():
        block = {}
        for m in DESIGNS:
            tau = tau_at_size(m, target)
            if tau is None:
                block[m] = None
                continue
            pw = power_at_tau(m, tau)
            block[m] = {"tau": round(tau, 6), "size": round(size_at_tau(m, tau), 5),
                        "power": round(pw, 5)}
        # paired differences vs weak_only at the matched taus
        tw = block["weak_only"]["tau"]
        for m in DESIGNS:
            if m == "weak_only" or block.get(m) is None:
                continue
            tm = block[m]["tau"]
            d, v = 0.0, 0.0
            for s in SHIFTS:
                dd = (alt[m][s] > tm).astype(float) - (alt["weak_only"][s] > tw).astype(float)
                d += W[s] * float(dd.mean())
                v += (W[s] ** 2) * float(dd.var(ddof=1)) / n
            block[m]["power_minus_weak"] = round(d, 5)
            block[m]["mc_se"] = round(math.sqrt(v), 5)
        res["at"][name] = block

    for m in DESIGNS:
        tau = tau_at_ucb(m)
        if tau is None:
            res["ucb_calibrated"][m] = None
            continue
        res["ucb_calibrated"][m] = {
            "tau": round(float(tau), 6),
            "size": round(size_at_tau(m, tau), 5),
            "ucb": round(ucb_size(m, tau), 5),
            "power": round(power_at_tau(m, tau), 5)}

    for m in DESIGNS:
        res["curves_coarse"][m] = [
            [round(t, 5), round(s, 5), round(p, 5)]
            for t, s, p in curves[m] if abs((t * 1000) % 1) < 1e-9 or t > 0.998]

    OUT.write_text(json.dumps(res, indent=1))
    for name in res["at"]:
        b = res["at"][name]
        print(f"== {name} (target {targets[name]:.5f})")
        for m in DESIGNS:
            e = b.get(m)
            if not e:
                print(f"  {m:20s} unreachable")
                continue
            extra = (f" dW={e.get('power_minus_weak')}+-{e.get('mc_se')}"
                     if "power_minus_weak" in e else "")
            print(f"  {m:20s} tau={e['tau']:.5f} size={e['size']:.5f} "
                  f"power={e['power']:.4f}{extra}")
    print("== UCB<=0.025 calibrated")
    for m in DESIGNS:
        e = res["ucb_calibrated"][m]
        if e:
            print(f"  {m:20s} tau={e['tau']:.5f} size={e['size']:.5f} "
                  f"ucb={e['ucb']:.5f} power={e['power']:.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
