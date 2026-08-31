"""Round-6 analyses (review round-6 Majors 1, 3, 4; minors 1-2).

All offline, on the stored validation-batch per-replicate posteriors.

  (1) anchor-x4 stress cell: MC standard error and one-sided 95%
      Clopper-Pearson upper bound for the worst observed cell (0.0595 at
      shift 1.0, 2,000 replicates) - Major 1.
  (2) cross-fitted size matching - Major 3: the validation batch's 10,000
      replicates per cell are split in half; thresholds that equalise the
      drift-weighted average type I to no-borrowing's achieved size are
      interpolated on ONE half and the power difference is evaluated on the
      OTHER half (both folds, averaged), so threshold-estimation and
      interpolation error propagate into the reported difference; achieved
      sizes on the evaluation half are reported.
  (3) pointwise size matching - Major 4: at every uniform drift cell,
      thresholds are interpolated per design so the NULL rejection rate in
      THAT cell equals 0.025 exactly, and the alternative-world rejection
      difference vs no borrowing is evaluated at those per-cell thresholds
      (cross-fitted the same way: match on half A, evaluate on half B).

Output: artifacts_round4/calibration/round6_analysis.json
"""
from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np

ROOT = Path("/root/work/osim")
RV = ROOT / "artifacts_round4" / "validation" / "pvals"
CAL = ROOT / "artifacts_round4" / "calibration" / "mixture_calibration.json"
OUT = ROOT / "artifacts_round4" / "calibration" / "round6_analysis.json"
SHIFTS = (-1.5, -1.25, -1.0, -0.75, -0.5, -0.25,
          0.0, 0.25, 0.5, 0.75, 1.0, 1.25, 1.5)
MAP = {"selective_cap50": "cap50", "selective_cap50_sam": "cap50_sam"}
GRID = np.unique(np.round(np.concatenate([
    np.arange(0.970, 0.9990, 0.00025), np.arange(0.9990, 0.99991, 0.00005)]), 6))


def cp_ucb_onesided(k: int, n: int, conf: float = 0.95) -> float:
    from math import lgamma, log, exp
    if k >= n:
        return 1.0
    logC = [lgamma(n + 1) - lgamma(i + 1) - lgamma(n - i + 1) for i in range(k + 1)]

    def cdf_le_k(p):
        lp, lq = log(p), log(1 - p)
        return sum(exp(c + i * lp + (n - i) * lq) for i, c in enumerate(logC))

    lo, hi = k / n, 1.0
    for _ in range(60):
        m = 0.5 * (lo + hi)
        if cdf_le_k(m) > 1 - conf:
            lo = m
        else:
            hi = m
    return hi


def main() -> int:
    res = {}
    # ---- (1) anchor-x4 worst stress cell ---------------------------------
    st = json.loads((ROOT / "artifacts_round4" / "stress" /
                     "round5_stress.json").read_text())["rows"]
    worst = max((r for r in st if r["axis"] == "anchor_prec"
                 and r["world"] == "null" and r["method"] == "cap50_anchor_x4"),
                key=lambda r: r["reject"])
    n_st = 2000
    k = int(round(worst["reject"] * n_st))
    res["anchor_x4_worst_cell"] = {
        "shift": worst["shift"], "estimate": worst["reject"],
        "mc_se": round(math.sqrt(worst["reject"] * (1 - worst["reject"]) / n_st), 5),
        "cp95_onesided_ucb": round(cp_ucb_onesided(k, n_st), 5), "n": n_st}

    # ---- load validation pvals -------------------------------------------
    W = {float(kk): v for kk, v in json.loads(CAL.read_text())["mixture_weights"].items()}
    designs = ("weak_only", "eb_only", "cap10", "cap30", "selective_cap50",
               "selective_cap50_sam")
    nul, alt = {}, {}
    for m in designs:
        key = MAP.get(m, m)
        nul[m] = {s: np.load(RV / f"null_s{s}_f1.0.npz")[key] for s in SHIFTS}
        alt[m] = {s: np.load(RV / f"alt_s{s}_f1.0.npz")[key] for s in SHIFTS}
    n = len(nul["weak_only"][0.0])
    half = n // 2
    folds = [(slice(0, half), slice(half, n)), (slice(half, n), slice(0, half))]

    def avg_size(m, tau, sl):
        return sum(W[s] * float((nul[m][s][sl] > tau).mean()) for s in SHIFTS)

    def mix_power(m, tau, sl):
        return sum(W[s] * float((alt[m][s][sl] > tau).mean()) for s in SHIFTS)

    def tau_for_size(m, target, sl):
        sizes = np.array([avg_size(m, float(t), sl) for t in GRID])
        for i in range(len(GRID) - 1):
            if sizes[i] >= target >= sizes[i + 1]:
                if sizes[i] == sizes[i + 1]:
                    return float(GRID[i])
                w = (sizes[i] - target) / (sizes[i] - sizes[i + 1])
                return float(GRID[i] + w * (GRID[i + 1] - GRID[i]))
        return None

    # ---- (2) cross-fitted average size matching --------------------------
    xfit = {}
    for m in designs:
        diffs, ach_m, ach_w = [], [], []
        for a, b in folds:
            t_w = tau_for_size("weak_only", 0.02513, a)
            t_m = tau_for_size(m, 0.02513, a)
            if t_w is None or t_m is None:
                continue
            # evaluate on the OTHER half, paired within replicate
            d, v = 0.0, 0.0
            for s in SHIFTS:
                dd = ((alt[m][s][b] > t_m).astype(float)
                      - (alt["weak_only"][s][b] > t_w).astype(float))
                d += W[s] * float(dd.mean())
                v += (W[s] ** 2) * float(dd.var(ddof=1)) / len(dd)
            diffs.append((d, math.sqrt(v), t_m))
            ach_m.append(avg_size(m, t_m, b))
            ach_w.append(avg_size("weak_only", t_w, b))
        est = float(np.mean([d for d, _, _ in diffs]))
        se = float(np.sqrt(np.mean([v ** 2 for _, v, _ in diffs])))  # per-fold se scale
        xfit[m] = {
            "crossfit_diff_vs_weak": round(est, 5),
            "per_fold": [[round(d, 5), round(v, 5), round(t, 6)] for d, v, t in diffs],
            "achieved_size_eval_half": [round(x, 5) for x in ach_m],
            "weak_achieved_size_eval_half": [round(x, 5) for x in ach_w],
            "approx_se": round(se, 5)}
    res["crossfit_size_matched"] = xfit

    # ---- (3) pointwise per-cell size matching (cross-fitted) --------------
    def tau_cell(m, s, target, sl):
        p = nul[m][s][sl]
        # empirical quantile threshold achieving rejection = target
        q = np.quantile(p, 1 - target)
        return float(q)

    pw = {}
    for m in ("selective_cap50", "cap30", "eb_only"):
        rows = {}
        for s in [x for x in SHIFTS if x >= 0.0]:
            vals = []
            for a, b in folds:
                t_m = tau_cell(m, s, 0.025, a)
                t_w = tau_cell("weak_only", s, 0.025, a)
                dd = ((alt[m][s][b] > t_m).astype(float)
                      - (alt["weak_only"][s][b] > t_w).astype(float))
                ach = float((nul[m][s][b] > t_m).mean())
                achw = float((nul["weak_only"][s][b] > t_w).mean())
                vals.append((float(dd.mean()),
                             float(dd.std(ddof=1) / math.sqrt(len(dd))), ach, achw))
            rows[str(s)] = {
                "power_diff": round(float(np.mean([v[0] for v in vals])), 5),
                "mc_se": round(float(np.mean([v[1] for v in vals])), 5),
                "achieved_null": round(float(np.mean([v[2] for v in vals])), 5),
                "weak_achieved_null": round(float(np.mean([v[3] for v in vals])), 5)}
        pw[m] = rows
    res["pointwise_matched"] = pw

    OUT.write_text(json.dumps(res, indent=1))
    print(json.dumps(res["anchor_x4_worst_cell"]))
    print("crossfit cap50:", xfit["selective_cap50"]["crossfit_diff_vs_weak"],
          "se", xfit["selective_cap50"]["approx_se"],
          "ach", xfit["selective_cap50"]["achieved_size_eval_half"])
    for s, r in pw["selective_cap50"].items():
        print(f"pointwise s={s}: dPow={r['power_diff']:+.4f} (se {r['mc_se']:.4f}) "
              f"achNull={r['achieved_null']:.4f} weakNull={r['weak_achieved_null']:.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
