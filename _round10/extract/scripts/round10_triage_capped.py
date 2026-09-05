#!/usr/bin/env python3
"""Round-10: the Section-4.4 triage analysis recomputed for the FINAL
DESIGN (review round-10 item 2).

The released Section-4.4 numbers (rank correlation +0.16, mass quartiles
0.15/0.36/0.65, tertile deltas) use the uncapped selective model's learned
mass and NLL.  This script recomputes the analysis with the actual
post-cap borrowed mass, min(mass, 0.5), and the final-design (capped) NLL:

  - reproduction of the uncapped numbers (quartiles, rank correlation,
    per-window correlations, tertile deltas);
  - capped rank correlation between post-cap mass and realized final-design
    benefit NLL_EB - NLL_capped, with bootstrap CI and per-window values
    (average-rank Spearman; the cap creates a 281-row tie atom at 0.5);
  - groups by post-cap mass: the non-binding rows split at their median
    mass (low/mid) and the 0.5 atom (binding rows) as its own group ---
    tertile membership inside a 281-row tie atom would be arbitrary, so
    the atom is reported as one group --- with paired capped-vs-EB deltas
    and bootstrap CIs;
  - strict tertiles by post-cap mass (ties broken by stable input order)
    for comparability with the released tertiles.

Inputs: artifacts_pathfinding/rewrite_numbers_perexample.json (fwd rows:
        sel_mass, info, eb, selective), artifacts_round4/stress/
        round9_rows.json (fwd: nll_cap per query).
Output: artifacts_round4/stress/round10_triage_capped.json
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
RW = ROOT / "artifacts_pathfinding" / "rewrite_numbers_perexample.json"
R9 = ROOT / "artifacts_round4" / "stress" / "round9_rows.json"
OUT = ROOT / "artifacts_round4" / "stress" / "round10_triage_capped.json"
B = 4000
RNG = np.random.default_rng(20260913)
CAP = 0.5


def rankdata_avg(x):
    x = np.asarray(x, dtype=float)
    order = np.argsort(x, kind="stable")
    ranks = np.empty(len(x))
    sx = x[order]
    i = 0
    while i < len(x):
        j = i
        while j + 1 < len(x) and sx[j + 1] == sx[i]:
            j += 1
        ranks[order[i:j + 1]] = 0.5 * (i + j) + 1.0
        i = j + 1
    return ranks


def spearman(x, y):
    rx, ry = rankdata_avg(x), rankdata_avg(y)
    rx = rx - rx.mean()
    ry = ry - ry.mean()
    return float((rx * ry).sum() / np.sqrt((rx ** 2).sum() * (ry ** 2).sum()))


def boot_corr(x, y):
    x, y = np.asarray(x), np.asarray(y)
    vals = np.empty(B)
    for b in range(B):
        idx = RNG.integers(0, len(x), size=len(x))
        vals[b] = spearman(x[idx], y[idx])
    return [float(v) for v in np.quantile(vals, [0.025, 0.975])]


def boot_mean(d):
    d = np.asarray(d, dtype=float)
    idx = RNG.integers(0, len(d), size=(B, len(d)))
    m = d[idx].mean(axis=1)
    return {"mean": float(d.mean()),
            "ci95": [float(x) for x in np.quantile(m, [0.025, 0.975])],
            "n": int(len(d))}


def main():
    fwd = json.load(open(RW))["fwd"]
    cap_rows = {r["i"]: r for r in json.load(open(R9))["fwd"]}
    mass_u = np.array([r["sel_mass"] for r in fwd])
    eb = np.array([r["eb"] for r in fwd])
    nll_u = np.array([r["selective"] for r in fwd])
    nll_c = np.array([cap_rows[r["i"]]["nll_cap"] for r in fwd])
    info = np.array([r["info"] for r in fwd])
    win = np.array([cap_rows[r["i"]]["window"] for r in fwd])
    mass_c = np.minimum(mass_u, CAP)
    ben_u = eb - nll_u
    ben_c = eb - nll_c

    res = {"n": len(fwd), "n_tie_at_cap": int((mass_u > CAP).sum())}

    # ---- reproduction: uncapped ------------------------------------------
    rep = {"mass_quartiles": [round(float(np.quantile(mass_u, q)), 2)
                              for q in (0.25, 0.5, 0.75)],
           "spearman": round(spearman(mass_u, ben_u), 3),
           "spearman_ci": [round(v, 3) for v in boot_corr(mass_u, ben_u)],
           "spearman_by_window": {w: round(spearman(mass_u[win == w],
                                                    ben_u[win == w]), 3)
                                  for w in ("w1", "w2", "w3")}}
    t = np.quantile(mass_u, [1 / 3, 2 / 3])
    lo, mid, hi = mass_u < t[0], (mass_u >= t[0]) & (mass_u < t[1]), mass_u >= t[1]
    rep["tertile_mean_mass"] = [round(float(mass_u[g].mean()), 2)
                                for g in (lo, mid, hi)]
    rep["tertile_delta"] = {k: boot_mean((nll_u - eb)[g])
                            for k, g in (("low", lo), ("mid", mid), ("high", hi))}
    res["uncapped_reproduction"] = rep

    # ---- final design: post-cap mass, capped NLL -------------------------
    fin = {"spearman": round(spearman(mass_c, ben_c), 3),
           "spearman_ci": [round(v, 3) for v in boot_corr(mass_c, ben_c)],
           "spearman_by_window": {w: round(spearman(mass_c[win == w],
                                                    ben_c[win == w]), 3)
                                  for w in ("w1", "w2", "w3")},
           "spearman_capped_mass_uncapped_benefit":
               round(spearman(mass_c, ben_u), 3)}
    # groups: non-binding split at median; the 0.5 atom as one group
    bind = mass_u > CAP
    nb_med = float(np.median(mass_c[~bind]))
    g_lo = (~bind) & (mass_c < nb_med)
    g_mid = (~bind) & (mass_c >= nb_med)
    fin["groups"] = {
        "nonbinding_median_mass": round(nb_med, 3),
        "low": {"mean_mass": round(float(mass_c[g_lo].mean()), 2),
                **boot_mean((nll_c - eb)[g_lo])},
        "mid": {"mean_mass": round(float(mass_c[g_mid].mean()), 2),
                **boot_mean((nll_c - eb)[g_mid])},
        "atom_0.5": {"mean_mass": 0.5, **boot_mean((nll_c - eb)[bind])},
    }
    # strict tertiles by post-cap mass (stable tie-breaking) for
    # comparability with the released tertiles
    order = np.argsort(mass_c, kind="stable")
    thirds = np.array_split(order, 3)
    fin["strict_tertile_delta"] = {
        k: boot_mean((nll_c - eb)[idx])
        for k, idx in (("low", thirds[0]), ("mid", thirds[1]),
                       ("high", thirds[2]))}
    fin["strict_tertile_mean_mass"] = [round(float(mass_c[idx].mean()), 2)
                                       for idx in thirds]
    # selection comparison: the 0.5 atom vs the pre-specified I>=8 rule
    sel_rule = info >= 8
    fin["selection"] = {
        "n_rule": int(sel_rule.sum()),
        "benefit_rule_capped": boot_mean(ben_c[sel_rule]),
        "n_atom": int(bind.sum()),
        "benefit_atom_capped": boot_mean(ben_c[bind])}
    res["final_design"] = fin

    OUT.write_text(json.dumps(res, indent=1))
    print(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
