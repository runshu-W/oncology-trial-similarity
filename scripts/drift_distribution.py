#!/usr/bin/env python3
"""Empirical donor-query drift distribution (round 4).

Pre-specifies the bounded-conflict set for the restricted calibration of the
final design.  The simulation's conflict_shift is a SYSTEMATIC logit shift
applied to all donors of a query, so its real-data analogue is the PER-QUERY
MEAN drift between the selected donors and the query:

    d_qj = logit~(p_donor_j) - logit~(p_query)   (empirical logit, +0.5)
    D_q  = mean_j d_qj  over the query's gated donor components

The observed spread of D_q across queries overstates the true systematic
drift because of sampling noise; a method-of-moments deconvolution removes
the average squared standard error.  The restricted-set bound is the
smallest simulation grid shift >= the noise-adjusted 95th percentile of |D|.

Population: gated donor components of title-screened (strict-ORR) queries,
restricted to strict donor components - exactly the records the capped
selective prior would borrow from in the primary analysis.  The all-pairs
version is reported as sensitivity.

Output: artifacts_round4/drift/drift_distribution.json
"""
from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from endpoint_canonicalizer import is_strict_orr  # noqa: E402

EX = ROOT / "artifacts_round2" / "examples_unitfix_ids_ep.jsonl"
OUT = ROOT / "artifacts_round4" / "drift" / "drift_distribution.json"
GRID = (0.25, 0.5, 0.75, 1.0, 1.25, 1.5)


def elogit(y: float, n: float) -> tuple[float, float]:
    """Empirical logit and its variance (+0.5 correction)."""
    a, b = y + 0.5, n - y + 0.5
    return math.log(a / b), 1.0 / a + 1.0 / b


def summarize(D: np.ndarray, V: np.ndarray) -> dict:
    var_obs = float(np.var(D, ddof=1))
    noise = float(np.mean(V))
    var_true = max(0.0, var_obs - noise)
    mu = float(np.mean(D))
    sd_true = math.sqrt(var_true)
    q95_abs_adj = abs(mu) + 1.96 * sd_true
    return {
        "n_queries": int(len(D)),
        "mean": round(mu, 4),
        "sd_observed": round(math.sqrt(var_obs), 4),
        "mean_squared_se": round(noise, 4),
        "sd_true_deconvolved": round(sd_true, 4),
        "raw_abs_quantiles": {str(q): round(float(np.quantile(np.abs(D), q)), 4)
                              for q in (0.5, 0.8, 0.9, 0.95)},
        "adj_abs_q95_normal": round(q95_abs_adj, 4),
        "bound_on_grid": next((g for g in GRID if g >= q95_abs_adj), GRID[-1]),
    }


def main() -> int:
    OUT.parent.mkdir(parents=True, exist_ok=True)
    strictD, strictV, allD, allV = [], [], [], []
    with open(EX) as f:
        for line in f:
            e = json.loads(line)
            q = e["query"]
            lq, vq = elogit(q["count"], q["denominator"])
            qs = is_strict_orr(q.get("endpoint"))
            for keep_strict, Dl, Vl in ((True, strictD, strictV),
                                        (False, allD, allV)):
                ds, vs = [], []
                for c in e["components"]:
                    if c["gate"] <= 0:
                        continue
                    if keep_strict and not (qs and is_strict_orr(c.get("endpoint"))):
                        continue
                    ld, vd = elogit(c["count"], c["denominator"])
                    ds.append(ld - lq)
                    vs.append(vd + vq)
                if len(ds) >= 2:
                    Dl.append(float(np.mean(ds)))
                    Vl.append(float(np.mean(vs)) / len(ds))
    # note: the noise correction for D_q uses only the sampling variance of
    # the mean, mean(v)/m.  Idiosyncratic (true) donor-to-donor variation is
    # deliberately NOT subtracted, so it stays inside the "systematic"
    # estimate -> larger sd_true -> larger (more conservative) bound.
    res = {
        "strict_pairs_primary": summarize(np.array(strictD), np.array(strictV)),
        "all_pairs_sensitivity": summarize(np.array(allD), np.array(allV)),
        "spec": "bound = smallest grid shift >= |mean| + 1.96*sd_true of "
                "per-query mean donor drift (empirical logit, +0.5), gated "
                "components, deconvolved for sampling noise",
    }
    OUT.write_text(json.dumps(res, indent=1))
    print(json.dumps(res, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
