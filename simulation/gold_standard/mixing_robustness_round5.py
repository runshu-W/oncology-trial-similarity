"""Round-5: robustness of the corpus-specific decision analysis to the
mixing measure (review round-5 Major 1).

For a family of alternative mixing distributions over the uniform-shift grid,
recomputes on the FROZEN 10,000-replicate validation batch:

  - the size-matched mixture-power difference (final design cap 0.5, and
    cap 0.3, versus calibrated no borrowing), where BOTH designs' thresholds
    are interpolated so their weighted average type I equals no-borrowing's
    validated size under THAT mixing (so liberality can never pay);

mixings:
  canonical            deconvolved N(+0.124, 1.328^2) (the paper's)
  nonparametric_raw    empirical histogram of per-query mean drift (no
                       deconvolution, no normality)
  nonparametric_shrunk per-query shrinkage D * var_true/(var_true+v_q), then
                       histogram (removes noise without normality)
  all_pairs            N(+0.125, 1.391^2) (no title screening)
  strat_era            era-stratified: normals fitted separately to queries
                       sorted before/after 2021-12-31, mixed by stratum share
  strat_avail          availability-stratified: queries with >=1 available
                       gated donor vs without
  N(0,0.5) N(0,1) U(-1.5,1.5) point0   canonical alternatives
  bootstrap            500 resamples of the queries -> weight uncertainty ->
                       interval for the canonical-mixing net benefit

Output: artifacts_round4/calibration/mixing_robustness_round5.json
"""
from __future__ import annotations

import csv
import json
import math
import sys
from pathlib import Path

import numpy as np

ROOT = Path("/root/work/osim")
sys.path.insert(0, str(ROOT / "scripts"))
from endpoint_canonicalizer import is_strict_orr  # noqa: E402
from drift_distribution import elogit  # noqa: E402

RV = ROOT / "artifacts_round4" / "validation" / "pvals"
EX = ROOT / "artifacts_round2" / "examples_unitfix_ids_ep.jsonl"
META = ROOT / "artifacts_rerun" / "examples_unitfix_with_true_dates.jsonl"
DATES = ROOT / "artifacts_rerun" / "clinicaltrials_date_rows.csv"
OUT = ROOT / "artifacts_round4" / "calibration" / "mixing_robustness_round5.json"

SHIFTS = np.array([-1.5, -1.25, -1.0, -0.75, -0.5, -0.25,
                   0.0, 0.25, 0.5, 0.75, 1.0, 1.25, 1.5])
STEP = 0.25
DESIGNS = {"weak_only": "weak_only", "cap50": "cap50", "cap30": "cap30"}


def normal_weights(mu, sd):
    from math import erf, sqrt

    def Phi(x):
        return 0.5 * (1 + erf(x / sqrt(2)))

    w = []
    for i, s in enumerate(SHIFTS):
        lo = -np.inf if i == 0 else s - STEP / 2
        hi = np.inf if i == len(SHIFTS) - 1 else s + STEP / 2
        plo = 0.0 if lo == -np.inf else Phi((lo - mu) / sd)
        phi = 1.0 if hi == np.inf else Phi((hi - mu) / sd)
        w.append(phi - plo)
    w = np.array(w)
    return w / w.sum()


def hist_weights(vals):
    w = np.zeros(len(SHIFTS))
    for v in vals:
        j = int(np.argmin(np.abs(SHIFTS - np.clip(v, SHIFTS[0], SHIFTS[-1]))))
        w[j] += 1
    return w / w.sum()


def main() -> int:
    # ---- per-query drift values with per-query noise and strata -----------
    ex = [json.loads(l) for l in open(EX)]
    meta = [json.loads(l) for l in open(META)]
    dates = {r["nct_id"]: r for r in csv.DictReader(open(DATES))}
    posted = {k: (v.get("results_first_posted_date") or "") for k, v in dates.items()}
    sortd = [(m["query_metadata"].get("temporal_sort_date") or "") for m in meta]
    startd = [(m["query_metadata"].get("start_date") or "") for m in meta]
    D, V, era, avail = [], [], [], []
    for i, e in enumerate(ex):
        q = e["query"]
        if not is_strict_orr(q.get("endpoint")):
            continue
        lq, vq = elogit(q["count"], q["denominator"])
        ds, vs, has_av = [], [], False
        qs = startd[i]
        for c in e["components"]:
            if c["gate"] <= 0 or not is_strict_orr(c.get("endpoint")):
                continue
            ld, vd = elogit(c["count"], c["denominator"])
            ds.append(ld - lq)
            vs.append(vd + vq)
            if qs and posted.get(c["nct_id"], "") and posted[c["nct_id"]] < qs:
                has_av = True
        if len(ds) >= 2:
            D.append(float(np.mean(ds)))
            V.append(float(np.mean(vs)) / len(ds))
            era.append(0 if (sortd[i] and sortd[i] <= "2021-12-31") else 1)
            avail.append(1 if has_av else 0)
    D, V = np.array(D), np.array(V)
    era, avail = np.array(era), np.array(avail)
    var_true = max(0.0, float(np.var(D, ddof=1)) - float(np.mean(V)))

    def fit(mask):
        d, v = D[mask], V[mask]
        vt = max(1e-6, float(np.var(d, ddof=1)) - float(np.mean(v)))
        return float(np.mean(d)), math.sqrt(vt), int(mask.sum())

    mixings = {}
    mixings["canonical"] = normal_weights(float(np.mean(D)), math.sqrt(var_true))
    mixings["nonparametric_raw"] = hist_weights(D)
    shrink = D * (var_true / (var_true + V))
    mixings["nonparametric_shrunk"] = hist_weights(shrink)
    # era-stratified
    parts = []
    for g in (0, 1):
        mu, sd, ng = fit(era == g)
        parts.append((ng, normal_weights(mu, sd)))
    tot = sum(p[0] for p in parts)
    mixings["strat_era"] = sum(p[0] / tot * p[1] for p in parts)
    parts = []
    for g in (0, 1):
        mu, sd, ng = fit(avail == g)
        parts.append((ng, normal_weights(mu, sd)))
    tot = sum(p[0] for p in parts)
    mixings["strat_avail"] = sum(p[0] / tot * p[1] for p in parts)
    mixings["N(0,0.5)"] = normal_weights(0.0, 0.5)
    mixings["N(0,1)"] = normal_weights(0.0, 1.0)
    mixings["U(-1.5,1.5)"] = np.ones(len(SHIFTS)) / len(SHIFTS)
    p0 = np.zeros(len(SHIFTS)); p0[np.where(SHIFTS == 0.0)[0][0]] = 1.0
    mixings["point0"] = p0

    # all-pairs
    allD, allV = [], []
    for i, e in enumerate(ex):
        q = e["query"]
        lq, vq = elogit(q["count"], q["denominator"])
        ds, vs = [], []
        for c in e["components"]:
            if c["gate"] <= 0:
                continue
            ld, vd = elogit(c["count"], c["denominator"])
            ds.append(ld - lq); vs.append(vd + vq)
        if len(ds) >= 2:
            allD.append(float(np.mean(ds))); allV.append(float(np.mean(vs)) / len(ds))
    allD, allV = np.array(allD), np.array(allV)
    vt = max(0.0, float(np.var(allD, ddof=1)) - float(np.mean(allV)))
    mixings["all_pairs"] = normal_weights(float(np.mean(allD)), math.sqrt(vt))

    # ---- validation pvals --------------------------------------------------
    nul = {m: {s: np.load(RV / f"null_s{s}_f1.0.npz")[k] for s in SHIFTS}
           for m, k in DESIGNS.items()}
    alt = {m: {s: np.load(RV / f"alt_s{s}_f1.0.npz")[k] for s in SHIFTS}
           for m, k in DESIGNS.items()}
    n = len(nul["weak_only"][0.0])
    grid = np.unique(np.round(np.concatenate([
        np.arange(0.970, 0.9990, 0.00025), np.arange(0.9990, 0.99991, 0.00005)]), 6))
    # precompute per-design per-shift rejection rates on the tau grid
    R_nul = {m: {s: np.array([float((nul[m][s] > t).mean()) for t in grid])
                 for s in SHIFTS} for m in DESIGNS}
    R_alt = {m: {s: np.array([float((alt[m][s] > t).mean()) for t in grid])
                 for s in SHIFTS} for m in DESIGNS}

    def net(mix, m, ref="weak_only"):
        w = {s: mix[j] for j, s in enumerate(SHIFTS)}
        size = {mm: sum(w[s] * R_nul[mm][s] for s in SHIFTS) for mm in (m, ref)}
        pw = {mm: sum(w[s] * R_alt[mm][s] for s in SHIFTS) for mm in (m, ref)}
        tgt = None
        # target: ref's size at its frozen tau 0.9805 under THIS mixing
        j = int(np.argmin(np.abs(grid - 0.9805)))
        tgt = size[ref][j]

        def at(mm):
            sarr = size[mm]
            for i in range(len(grid) - 1):
                if sarr[i] >= tgt >= sarr[i + 1]:
                    if sarr[i] == sarr[i + 1]:
                        frac = 0.0
                    else:
                        frac = (sarr[i] - tgt) / (sarr[i] - sarr[i + 1])
                    return pw[mm][i] + frac * (pw[mm][i + 1] - pw[mm][i])
            return None
        pm, pr = at(m), at(ref)
        if pm is None or pr is None:
            return None, tgt
        return pm - pr, tgt

    res = {"n_queries_strict": int(len(D)), "results": {}}
    for name, mix in mixings.items():
        row = {"weights": [round(float(x), 4) for x in mix]}
        for m in ("cap50", "cap30"):
            d, tgt = net(mix, m)
            row[f"net_{m}_size_matched"] = None if d is None else round(float(d), 5)
        row["target_size"] = round(float(tgt), 5)
        res["results"][name] = row
        print(f"{name:22s} net cap50={row['net_cap50_size_matched']} "
              f"cap30={row['net_cap30_size_matched']} (size {row['target_size']})")

    # bootstrap of the canonical mixing
    rng = np.random.default_rng(20260905)
    boots = []
    for _ in range(500):
        idx = rng.integers(0, len(D), size=len(D))
        d, v = D[idx], V[idx]
        vt = max(1e-6, float(np.var(d, ddof=1)) - float(np.mean(v)))
        mix = normal_weights(float(np.mean(d)), math.sqrt(vt))
        val, _ = net(mix, "cap50")
        if val is not None:
            boots.append(val)
    boots = np.array(boots)
    res["bootstrap_cap50"] = {
        "n": int(len(boots)),
        "mean": round(float(boots.mean()), 5),
        "q": {q: round(float(np.quantile(boots, float(q))), 5)
              for q in ("0.025", "0.5", "0.975")}}
    print("bootstrap cap50 net:", res["bootstrap_cap50"])
    OUT.write_text(json.dumps(res, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
