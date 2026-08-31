#!/usr/bin/env python3
"""Triage-utility analysis (reviewer round-2, point 7).

Question: does the learned borrowed mass rank the realized incremental
benefit of borrowing on unseen data — not merely correlate with its own
inputs? Benefit for query i is b_i = NLL_EB(i) - NLL_selective(i) (positive =
selective beat the marginal reference), computed on the pooled disjoint
forward windows where each query is scored once by its own-cutoff model.

Outputs artifacts_round2/triage_utility.json:
  - rank correlations: Spearman(mass, b), Spearman(info, b) with bootstrap CIs
  - mass-tertile paired deltas vs EB and vs the fixed-budget predecessor
  - decision curve: cumulative mean benefit when borrowing only top-q by mass
  - per-window and per-disease tertile consistency
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
REWRITE = ROOT / "artifacts_pathfinding" / "rewrite_numbers_perexample.json"
EXAMPLES = ROOT / "artifacts_rerun" / "examples_unitfix_with_true_dates.jsonl"
OUT = ROOT / "artifacts_round2"
B = 4000
RNG = np.random.default_rng(20260818)


def spearman(x: np.ndarray, y: np.ndarray) -> float:
    rx = np.argsort(np.argsort(x)).astype(float)
    ry = np.argsort(np.argsort(y)).astype(float)
    rx = (rx - rx.mean()) / rx.std()
    ry = (ry - ry.mean()) / ry.std()
    return float((rx * ry).mean())


def boot_stat(fn, arrays, b=B) -> dict:
    n = len(arrays[0])
    point = fn(*arrays)
    idx = RNG.integers(0, n, size=(b, n))
    vals = np.array([fn(*[a[ix] for a in arrays]) for ix in idx])
    lo, hi = np.quantile(vals, [0.025, 0.975])
    return {"point": float(point), "ci95": [float(lo), float(hi)], "n": int(n)}


def mean_ci(d: np.ndarray) -> dict:
    return boot_stat(lambda a: a.mean(), [d])


def main() -> None:
    OUT.mkdir(exist_ok=True)
    rw = json.load(open(REWRITE))
    with open(EXAMPLES) as f:
        examples = [json.loads(l) for l in f if l.strip()]

    rows = rw["fwd"]
    i_arr = np.array([r["i"] for r in rows])
    mass = np.array([r["sel_mass"] for r in rows])
    info = np.array([r["info"] for r in rows])
    b_eb = np.array([r["eb"] - r["selective"] for r in rows])          # benefit vs EB
    b_old = np.array([r["two_head_old"] - r["selective"] for r in rows])
    d_old_eb = np.array([r["two_head_old"] - r["eb"] for r in rows])   # old model harm vs EB
    dates = []
    site = []
    for r in rows:
        ex = examples[r["i"]]
        md = ex.get("query_metadata") or {}
        dates.append(md.get("temporal_sort_date") or "")
        site.append(md.get("primary_site") or "Unknown")
    dates = np.array(dates)
    site = np.array(site)
    wname = np.where(dates <= "2021-12-31", "w1",
                     np.where(dates <= "2022-12-31", "w2", "w3"))

    res: dict = {"n_fwd": len(rows)}

    # 1. rank correlations -----------------------------------------------------
    res["spearman_mass_benefit"] = boot_stat(spearman, [mass, b_eb])
    res["spearman_info_benefit"] = boot_stat(spearman, [info, b_eb])
    res["spearman_mass_info"] = {"point": spearman(mass, info)}

    # 2. mass tertiles ---------------------------------------------------------
    q1, q2 = np.quantile(mass, [1 / 3, 2 / 3])
    res["mass_tertile_cuts"] = [float(q1), float(q2)]
    tert = np.where(mass <= q1, "low", np.where(mass <= q2, "mid", "high"))
    res["tertiles"] = {}
    for t in ("low", "mid", "high"):
        m = tert == t
        res["tertiles"][t] = {
            "n": int(m.sum()),
            "mean_mass": float(mass[m].mean()),
            "delta_vs_eb": mean_ci(-b_eb[m]),           # negative = beats EB
            "delta_old_vs_eb": mean_ci(d_old_eb[m]),    # old model in same bin
            "delta_vs_old": mean_ci(-b_old[m]),
        }

    # 3. decision curve: borrow only top-q by mass -----------------------------
    order = np.argsort(-mass)
    res["decision_curve"] = []
    for q in (0.1, 0.2, 0.3, 0.4, 0.5, 0.75, 1.0):
        k = max(1, int(round(q * len(rows))))
        sel = order[:k]
        res["decision_curve"].append({
            "top_frac_by_mass": q,
            "mean_benefit_vs_eb": mean_ci(b_eb[sel]),
            "mass_threshold": float(mass[order[k - 1]]),
        })

    # 4. consistency across windows and diseases ------------------------------
    res["per_window"] = {}
    for w in ("w1", "w2", "w3"):
        m = wname == w
        hi = m & (tert == "high")
        lo = m & (tert == "low")
        res["per_window"][w] = {
            "n": int(m.sum()),
            "spearman_mass_benefit": {"point": spearman(mass[m], b_eb[m])},
            "high_delta_vs_eb": mean_ci(-b_eb[hi]) if hi.sum() > 10 else None,
            "low_delta_vs_eb": mean_ci(-b_eb[lo]) if lo.sum() > 10 else None,
        }
    res["per_site"] = {}
    for s in sorted(set(site)):
        m = site == s
        if m.sum() < 40:
            continue
        res["per_site"][s] = {
            "n": int(m.sum()),
            "spearman_mass_benefit": {"point": spearman(mass[m], b_eb[m])},
            "mean_benefit_high_tertile": float(b_eb[m & (tert == "high")].mean())
            if (m & (tert == "high")).sum() else None,
        }

    # 5. comparison against the pre-specified I>=8 rule ------------------------
    hi_rule = info >= 8
    res["rule_I8"] = {
        "n_selected": int(hi_rule.sum()),
        "mean_benefit_vs_eb": mean_ci(b_eb[hi_rule]),
    }
    k = int(hi_rule.sum())
    res["mass_top_matched"] = {
        "n_selected": k,
        "mean_benefit_vs_eb": mean_ci(b_eb[order[:k]]),
    }

    json.dump(res, open(OUT / "triage_utility.json", "w"), indent=1)

    def f(x):
        return f"{x['point']:+.4f} [{x['ci95'][0]:+.4f},{x['ci95'][1]:+.4f}]" if "ci95" in x else f"{x['point']:+.4f}"

    print("Spearman(mass, benefit):", f(res["spearman_mass_benefit"]))
    print("Spearman(info, benefit):", f(res["spearman_info_benefit"]))
    print("Spearman(mass, info):   ", f(res["spearman_mass_info"]))
    for t in ("low", "mid", "high"):
        r = res["tertiles"][t]
        print(f"tertile {t:4s} n={r['n']:3d} mass={r['mean_mass']:.2f} "
              f"dEB={r['delta_vs_eb']['point']:+.4f} "
              f"[{r['delta_vs_eb']['ci95'][0]:+.4f},{r['delta_vs_eb']['ci95'][1]:+.4f}] "
              f"old_dEB={r['delta_old_vs_eb']['point']:+.4f}")
    for row in res["decision_curve"]:
        print(f"top {row['top_frac_by_mass']:.2f} by mass: benefit "
              f"{row['mean_benefit_vs_eb']['point']:+.4f} "
              f"[{row['mean_benefit_vs_eb']['ci95'][0]:+.4f},{row['mean_benefit_vs_eb']['ci95'][1]:+.4f}]")
    print("rule I>=8   :", f(res["rule_I8"]["mean_benefit_vs_eb"]), f"n={res['rule_I8']['n_selected']}")
    print("mass top-k  :", f(res["mass_top_matched"]["mean_benefit_vs_eb"]), f"n={k}")
    for w in ("w1", "w2", "w3"):
        r = res["per_window"][w]
        print(f"{w}: rho={r['spearman_mass_benefit']['point']:+.3f} "
              f"high={f(r['high_delta_vs_eb']) if r['high_delta_vs_eb'] else 'n/a'} "
              f"low={f(r['low_delta_vs_eb']) if r['low_delta_vs_eb'] else 'n/a'}")


if __name__ == "__main__":
    main()
