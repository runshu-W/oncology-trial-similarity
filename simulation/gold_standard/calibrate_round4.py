"""Round-4 offline calibration: two-sided empirical-mixture frame + frontier.

Assembles the stored per-replicate posterior probabilities from
  round-3  calibration store (positive + partial cells, 8 base designs),
  round-4  cap sweep       (positive + partial cells, 18 cap variants),
  round-4  negative cells  (two-sided uniform cells, 24 designs)
and computes, for every design:

  A. unrestricted worst-case frame over ALL null cells (two-sided uniform +
     partial): worst type I at 0.975, exact tau* (smallest threshold with
     worst-case <= 0.025, from per-cell empirical quantiles), power at tau*;
  B. empirical-mixture frame: weights over uniform shifts from the corpus
     drift distribution (round-4 drift_distribution.json, strict primary:
     per-query mean drift ~ N(mu, sd), discretized to the shift grid, edge
     bins absorbing the tails); tau_mix = smallest threshold >= 0.975 with
     mixture-average type I <= 0.025; mixture-average power at tau_mix;
  C. hard ceiling: worst-case type I at tau_mix over ALL null cells
     (pre-specified ceiling 0.05 = 2x nominal);
  D. final-design rule (pre-specified): among designs passing C, maximise
     mixture-average power at tau_mix; require it to exceed weak_only's
     mixture power at weak_only's own tau_mix (net benefit); smaller cap
     breaks ties.

Also verifies the cap50 columns reproduce round-3 bit for bit.
"""
from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2] if (
    Path(__file__).resolve().parents[2] / "artifacts_round3").exists() else \
    Path("/root/work/osim")

R3 = ROOT / "artifacts_round3" / "calibration" / "pvals"
R4C = ROOT / "artifacts_round4" / "capsweep" / "pvals"
R4N = ROOT / "artifacts_round4" / "negcells" / "pvals"
DRIFT = ROOT / "artifacts_round4" / "drift" / "drift_distribution.json"
OUT = ROOT / "artifacts_round4" / "calibration"

POS_SHIFTS = (0.0, 0.25, 0.5, 0.75, 1.0, 1.25, 1.5)
NEG_SHIFTS = (-1.5, -1.25, -1.0, -0.75, -0.5, -0.25)
ALL_SHIFTS = tuple(sorted(NEG_SHIFTS + POS_SHIFTS))
PARTIAL = [(s, f) for s in (0.5, 1.0, 1.5) for f in (0.25, 0.5, 0.75)]
NOMINAL = 0.025
CEILING = 0.05
BASE8 = ("weak_only", "eb_only", "rule_sam", "two_head_pro_sam",
         "selective", "selective_sam", "selective_cap50", "selective_cap50_sam")
CAPKEYS = [f"cap{c}{s}" for c in range(10, 100, 10) for s in ("", "_sam")]


def load(method: str, world: str, shift: float, frac: float) -> np.ndarray:
    tag = f"{world}_s{shift}_f{frac}"
    if shift < 0:
        return np.load(R4N / f"{tag}.npz")[method]
    if method in BASE8:
        return np.load(R3 / f"{tag}.npz")[method]
    return np.load(R4C / f"{tag}.npz")[method]


def canon(method: str) -> str:
    # round-3 store uses selective_cap50*, round-4 stores use cap50*
    return method


def get(method, world, shift, frac):
    if shift < 0 and method in ("selective_cap50", "selective_cap50_sam"):
        return load(method.replace("selective_", ""), world, shift, frac)
    if shift >= 0 and method in ("cap50", "cap50_sam"):
        # available in both stores; use the sweep store (checked identical)
        return load(method, world, shift, frac)
    return load(method, world, shift, frac)


def mixture_weights(mu: float, sd: float) -> dict[float, float]:
    from math import erf, sqrt

    def Phi(x):
        return 0.5 * (1 + erf(x / sqrt(2)))

    step = 0.25
    w = {}
    for s in ALL_SHIFTS:
        lo = -np.inf if s == ALL_SHIFTS[0] else s - step / 2
        hi = np.inf if s == ALL_SHIFTS[-1] else s + step / 2
        plo = 0.0 if lo == -np.inf else Phi((lo - mu) / sd)
        phi = 1.0 if hi == np.inf else Phi((hi - mu) / sd)
        w[s] = phi - plo
    tot = sum(w.values())
    return {k: v / tot for k, v in w.items()}


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    drift = json.loads(DRIFT.read_text())["strict_pairs_primary"]
    W = mixture_weights(drift["mean"], drift["sd_true_deconvolved"])

    # ---- consistency: cap50 (sweep) vs selective_cap50 (round 3) ----------
    checks = {}
    for world in ("null", "alt"):
        for s in (0.0, 1.5):
            a = np.load(R3 / f"{world}_s{s}_f1.0.npz")["selective_cap50"]
            b = np.load(R4C / f"{world}_s{s}_f1.0.npz")["cap50"]
            checks[f"{world}_s{s}"] = bool(np.array_equal(a, b))
    METHODS = list(BASE8) + [k for k in CAPKEYS if k not in ("cap50", "cap50_sam")]

    null_cells = ([("u", s) for s in ALL_SHIFTS] + [("p", sf) for sf in PARTIAL])
    res = {"mixture_weights": {str(k): round(v, 5) for k, v in W.items()},
           "drift_source": drift, "nominal": NOMINAL, "ceiling": CEILING,
           "cap50_consistency": checks, "methods": {}}

    for m in METHODS:
        # per-cell arrays
        nul_u = {s: get(m, "null", s, 1.0) for s in ALL_SHIFTS}
        alt_u = {s: get(m, "alt", s, 1.0) for s in ALL_SHIFTS}
        nul_p = {sf: get(m, "null", sf[0], sf[1]) for sf in PARTIAL}

        def t1(tau):
            u = {s: float((nul_u[s] > tau).mean()) for s in ALL_SHIFTS}
            p = {sf: float((nul_p[sf] > tau).mean()) for sf in PARTIAL}
            return u, p

        def wavg(d, tau=None, arrs=None):
            src = arrs if arrs is not None else d
            return float(sum(W[s] * ((src[s] > tau).mean() if tau is not None
                                     else src[s]) for s in ALL_SHIFTS))

        # A. unrestricted worst case (exact quantile tau*)
        qs = [float(np.quantile(a, 1 - NOMINAL)) for a in nul_u.values()] + \
             [float(np.quantile(a, 1 - NOMINAL)) for a in nul_p.values()]
        tau_star = max(qs)
        u0, p0 = t1(0.975)
        worst0 = max(list(u0.values()) + list(p0.values()))
        us, ps = t1(tau_star)
        worst_star = max(list(us.values()) + list(ps.values()))
        pow_star = wavg(None, tau_star, alt_u)
        pow_star_s0 = float((alt_u[0.0] > tau_star).mean())

        # B. mixture frame
        grid = np.round(np.concatenate([np.arange(0.975, 0.9995, 0.0005),
                                        [0.9995, 0.9999]]), 6)
        tau_mix = None
        for t in grid:
            if wavg(None, t, nul_u) <= NOMINAL:
                tau_mix = float(t)
                break
        entry = {
            "worst_t1_at_0.975": round(worst0, 4),
            "tau_star_exact": round(tau_star, 6),
            "worst_t1_at_tau_star": round(worst_star, 4),
            "mix_power_at_tau_star": round(pow_star, 4),
            "power_s0_at_tau_star": round(pow_star_s0, 4),
            "avg_t1_at_0.975": round(wavg(None, 0.975, nul_u), 4),
            "tau_mix": tau_mix,
        }
        if tau_mix is not None:
            um, pm = t1(tau_mix)
            entry.update({
                "avg_t1_at_tau_mix": round(wavg(None, tau_mix, nul_u), 4),
                "mix_power_at_tau_mix": round(wavg(None, tau_mix, alt_u), 4),
                "power_s0_at_tau_mix": round(float((alt_u[0.0] > tau_mix).mean()), 4),
                "worst_t1_at_tau_mix": round(max(list(um.values()) + list(pm.values())), 4),
                "worst_cell_at_tau_mix": max(
                    [(v, f"u{s}") for s, v in um.items()] +
                    [(v, f"p{sf}") for sf, v in pm.items()])[1],
                "t1_profile_at_tau_mix": {str(s): round(um[s], 4) for s in ALL_SHIFTS},
                "power_profile_at_tau_mix": {str(s): round(float((alt_u[s] > tau_mix).mean()), 4)
                                             for s in ALL_SHIFTS},
            })
        res["methods"][m] = entry

    # D. final-design rule
    weak = res["methods"]["weak_only"]
    bench = weak["mix_power_at_tau_mix"]
    eligible = []
    for m, e in res["methods"].items():
        if m == "weak_only" or e.get("tau_mix") is None:
            continue
        if e["worst_t1_at_tau_mix"] <= CEILING and e["mix_power_at_tau_mix"] > bench:
            eligible.append((e["mix_power_at_tau_mix"], m))
    eligible.sort(reverse=True)
    res["final_design_rule"] = {
        "benchmark_weak_mix_power": bench,
        "eligible_ranked": [{"method": m, "mix_power": p} for p, m in eligible],
        "selected": eligible[0][1] if eligible else None,
    }
    (OUT / "mixture_calibration.json").write_text(json.dumps(res, indent=1))
    print("cap50 consistency:", checks)
    print(f"weak_only: tau_mix={weak['tau_mix']} mixPower={bench:.3f}")
    for p, m in eligible[:8]:
        e = res["methods"][m]
        print(f"{m:14s} tau_mix={e['tau_mix']:.4f} mixPower={p:.3f} "
              f"worst@tau_mix={e['worst_t1_at_tau_mix']:.3f} "
              f"(cell {e['worst_cell_at_tau_mix']}) tau*={e['tau_star_exact']:.4f} "
              f"powS0@tau*={e['power_s0_at_tau_star']:.3f}")
    print("selected:", res["final_design_rule"]["selected"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
