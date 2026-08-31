"""Round-4 offline validation report (review round-4 Major 2).

Evaluates, on the INDEPENDENT 10,000-replicate validation batch, the
thresholds selected on the 2,000-replicate selection batch (read from
mixture_calibration.json; nothing is re-selected here):

  per design: mixture-average type I at tau_mix (+ paired MC se), worst null
  cell at tau_mix with Clopper-Pearson 95% UCB (pointwise at the worst cell,
  and a Bonferroni-simultaneous version over all null cells), mixture power,
  per-shift profiles, tipping shifts (last uniform shift with T1 <= 0.025 /
  <= 0.05), and the PAIRED mixture-power difference vs weak_only and eb_only
  with MC se (same replicates, so the difference is paired within cell).

Output: artifacts_round4/calibration/validation_report.json
"""
from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np

ROOT = Path("/root/work/osim")
RV = ROOT / "artifacts_round4" / "validation" / "pvals"
CAL = ROOT / "artifacts_round4" / "calibration" / "mixture_calibration.json"
OUT = ROOT / "artifacts_round4" / "calibration" / "validation_report.json"

SHIFTS = (-1.5, -1.25, -1.0, -0.75, -0.5, -0.25,
          0.0, 0.25, 0.5, 0.75, 1.0, 1.25, 1.5)
PARTIAL = [(s, f) for s in (0.5, 1.0, 1.5) for f in (0.25, 0.5, 0.75)]
MAP = {"selective_cap50": "cap50", "selective_cap50_sam": "cap50_sam"}


def cp_ucb(k: int, n: int, conf: float = 0.95) -> float:
    """Clopper-Pearson upper bound via beta quantile (bisection, no scipy)."""
    if k >= n:
        return 1.0
    a, b = k + 1, n - k
    lo, hi = k / n, 1.0
    # regularised incomplete beta via continued fraction is overkill; use
    # bisection on the Binomial cdf directly (exact enough at n=1e4).
    from math import lgamma, log, exp

    logC = [lgamma(n + 1) - lgamma(i + 1) - lgamma(n - i + 1) for i in range(k + 1)]

    def cdf_le_k(p):
        if p <= 0:
            return 1.0
        if p >= 1:
            return 0.0
        lp, lq = log(p), log(1 - p)
        return sum(exp(c + i * lp + (n - i) * lq) for i, c in enumerate(logC))

    for _ in range(60):
        mid = 0.5 * (lo + hi)
        if cdf_le_k(mid) > 1 - conf:
            lo = mid
        else:
            hi = mid
    return hi


def main() -> int:
    cal = json.loads(CAL.read_text())
    W = {float(k): v for k, v in cal["mixture_weights"].items()}
    designs = ["weak_only", "eb_only", "rule_sam", "two_head_pro_sam",
               "selective", "selective_sam", "cap10", "cap10_sam",
               "cap30", "cap30_sam", "selective_cap50", "selective_cap50_sam"]

    def load(m, world, s, f=1.0):
        key = MAP.get(m, m)
        return np.load(RV / f"{world}_s{s}_f{f}.npz")[key]

    def calentry(m):
        return cal["methods"][m]

    report = {"batch": "validation", "replicates": None, "methods": {}}
    rej_cache = {}

    for m in designs:
        e = calentry(m)
        tau_mix, tau_star = e.get("tau_mix"), e["tau_star_exact"]
        entry = {"tau_mix_selected": tau_mix, "tau_star_selected": tau_star}
        if tau_mix is None:
            report["methods"][m] = entry
            continue
        # null side
        t1u, ucbs = {}, {}
        for s in SHIFTS:
            p = load(m, "null", s)
            k = int((p > tau_mix).sum())
            n = len(p)
            t1u[s] = k / n
            ucbs[("u", s)] = cp_ucb(k, n)
            report["replicates"] = n
        for s, f in PARTIAL:
            p = load(m, "null", s, f)
            k = int((p > tau_mix).sum())
            t1u[(s, f)] = k / len(p)
            ucbs[("p", (s, f))] = cp_ucb(k, len(p))
        worst_cell = max(t1u, key=lambda k_: t1u[k_])
        n = report["replicates"]
        kworst = int(round(t1u[worst_cell] * n))
        avg = sum(W[s] * t1u[s] for s in SHIFTS)
        avg_se = math.sqrt(sum((W[s] ** 2) * t1u[s] * (1 - t1u[s]) / n
                               for s in SHIFTS))
        # power side + paired diff storage
        rej = {s: (load(m, "alt", s) > tau_mix) for s in SHIFTS}
        rej_cache[m] = rej
        pw = {s: float(rej[s].mean()) for s in SHIFTS}
        mixpow = sum(W[s] * pw[s] for s in SHIFTS)
        tip25 = max([s for s in SHIFTS if all(
            t1u[s2] <= 0.025 for s2 in SHIFTS if 0 <= s2 <= s)] or [None],
            key=lambda v: -1e9 if v is None else v)
        tip50 = max([s for s in SHIFTS if all(
            t1u[s2] <= 0.05 for s2 in SHIFTS if 0 <= s2 <= s)] or [None],
            key=lambda v: -1e9 if v is None else v)
        entry.update({
            "avg_t1": round(avg, 5), "avg_t1_mc_se": round(avg_se, 5),
            "worst_cell": str(worst_cell), "worst_t1": round(t1u[worst_cell], 5),
            "worst_t1_cp95_ucb": round(cp_ucb(kworst, n), 5),
            "worst_t1_bonf_ucb": round(cp_ucb(kworst, n, 1 - 0.05 / len(ucbs)), 5),
            "mix_power": round(mixpow, 5),
            "t1_profile": {str(s): round(t1u[s], 4) for s in SHIFTS},
            "power_profile": {str(s): round(pw[s], 4) for s in SHIFTS},
            "tipping_shift_le_0.025": tip25, "tipping_shift_le_0.05": tip50,
        })
        report["methods"][m] = entry

    # paired mixture-power differences vs references
    for m in designs:
        if m in ("weak_only",) or "mix_power" not in report["methods"].get(m, {}):
            continue
        for ref in ("weak_only", "eb_only"):
            if ref == m or "mix_power" not in report["methods"][ref]:
                continue
            dmean, dvar = 0.0, 0.0
            for s in SHIFTS:
                d = rej_cache[m][s].astype(float) - rej_cache[ref][s].astype(float)
                dmean += W[s] * float(d.mean())
                dvar += (W[s] ** 2) * float(d.var(ddof=1)) / len(d)
            report["methods"][m][f"mix_power_minus_{ref}"] = {
                "diff": round(dmean, 5), "mc_se": round(math.sqrt(dvar), 5)}

    OUT.write_text(json.dumps(report, indent=1))
    for m in ("weak_only", "eb_only", "selective", "selective_sam",
              "cap30", "selective_cap50", "selective_cap50_sam"):
        e = report["methods"][m]
        if "avg_t1" not in e:
            print(f"{m:20s} tau_mix=None")
            continue
        d = e.get("mix_power_minus_weak_only", {})
        print(f"{m:20s} tau={e['tau_mix_selected']} avgT1={e['avg_t1']:.4f}"
              f"(se {e['avg_t1_mc_se']:.4f}) worst={e['worst_t1']:.4f}"
              f"@{e['worst_cell']} UCB={e['worst_t1_cp95_ucb']:.4f}"
              f" mixPow={e['mix_power']:.4f}"
              f" dWeak={d.get('diff')}+-{d.get('mc_se')}"
              f" tip25={e['tipping_shift_le_0.025']} tip50={e['tipping_shift_le_0.05']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
