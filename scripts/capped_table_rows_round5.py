#!/usr/bin/env python3
"""Round-5: final-design (capped) rows for the main comparison tables
(review round-5 Major 6).

Computes, for the capped selective prior (c = 0.5):
  - held-out test NLL and development-set NLL (tab:headtohead row),
  - pooled forward NLL (tab:forward row),
  - test-split PIT KS and 50/80/95% prediction-interval coverage
    (tab:calib row).

PIT and intervals use the same conventions as the released evaluation
(randomized PIT seed 20260707; equal-tailed intervals on the exact mixture
pmf).  Output: artifacts_round4/stress/capped_table_rows.json
"""
from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "pipeline"))
sys.path.insert(0, str(ROOT / "scripts"))

import two_head_selective as TS  # noqa: E402
from capped_real_data import bb, filt, read_jsonl  # noqa: E402

EX = ROOT / "artifacts_round2" / "examples_unitfix_ids_ep.jsonl"
RW = ROOT / "artifacts_pathfinding" / "rewrite_numbers_perexample.json"
SEL_DIR = ROOT / "artifacts_pathfinding" / "selective"
OUT = ROOT / "artifacts_round4" / "stress" / "capped_table_rows.json"
CAP = 0.5
CUTOFFS = ("2020-12-31", "2021-12-31", "2022-12-31")


def capped_mixture(model, fe, anchor):
    """Return (weights, alphas, betas) of the CAPPED prior incl. anchor."""
    y0, n0 = fe["query"]["count"], fe["query"]["denominator"]
    if not any(c["gate"] > 0 for c in fe["components"]):
        return np.array([1.0]), np.array([anchor[0]]), np.array([anchor[1]])
    with torch.no_grad():
        t = TS.LT._validated_example_tensors(fe)
        lam0, lam_i = TS.selective_lambda_weights(model, t["features"], t["gate"])
        a, b_, _ = TS.LT._component_alpha_beta_for_model(model, t["features"], t)
        lam_i = lam_i.numpy().astype(float)
        a = a.numpy().astype(float); b_ = b_.numpy().astype(float)
        lam0 = float(lam0)
    m = lam_i.sum()
    if m > CAP:
        lam_i = lam_i * (CAP / m)
        lam0 = 1.0 - lam_i.sum()
    return (np.concatenate([[lam0], lam_i]),
            np.concatenate([[anchor[0]], a]),
            np.concatenate([[anchor[1]], b_]))


def mix_pmf(w, a, b_, n0):
    ks = np.arange(0, int(n0) + 1)
    pmf = np.zeros(len(ks))
    for wi, ai, bi in zip(w, a, b_):
        if wi <= 0:
            continue
        pmf += wi * np.array([math.exp(bb(k, n0, ai, bi)) for k in ks])
    return pmf / pmf.sum()


def main():
    OUT.parent.mkdir(parents=True, exist_ok=True)
    ex = read_jsonl(EX)
    rw = json.load(open(RW))
    pm, panchor = TS.load_selective_artifact(SEL_DIR / "selective_model_primary.pt")

    # test + development NLL, PIT, coverage
    rng = np.random.default_rng(20260707)
    res = {}
    for setting, rows in (("test", rw["test"]), ("full", rw["full"])):
        nlls = []
        pits, covs = [], {0.5: 0, 0.8: 0, 0.95: 0}
        for r in rows:
            i = r["i"]
            fe = filt(ex[i], [True] * len(ex[i]["components"]))
            y0, n0 = fe["query"]["count"], fe["query"]["denominator"]
            w, a, b_ = capped_mixture(pm, fe, panchor)
            pmf = mix_pmf(w, a, b_, n0)
            y0i = int(y0)
            nlls.append(-math.log(max(pmf[y0i], 1e-300)))
            if setting == "test":
                lo = pmf[:y0i].sum()
                pits.append(lo + rng.uniform() * pmf[y0i])
                cdf = np.cumsum(pmf)
                for lvl in covs:
                    a_t = (1 - lvl) / 2
                    lo_k = int(np.searchsorted(cdf, a_t))
                    hi_k = int(np.searchsorted(cdf, 1 - a_t))
                    covs[lvl] += (lo_k <= y0i <= hi_k)
        res[f"{setting}_nll"] = float(np.mean(nlls))
        if setting == "test":
            u = np.sort(np.array(pits))
            grid = (np.arange(len(u)) + 1) / len(u)
            ks = float(np.max(np.abs(u - grid + 1 / len(u))))
            ks2 = float(np.max(np.abs(u - (np.arange(len(u))) / len(u))))
            res["test_pit_ks"] = max(ks, ks2)
            res["test_coverage"] = {str(k): v / len(rows) for k, v in covs.items()}

    # forward NLL (fold models, capped)
    folds = {c: TS.load_selective_artifact(SEL_DIR / f"selective_model_fold_{c}.pt")
             for c in CUTOFFS}
    meta = read_jsonl(ROOT / "artifacts_rerun" / "examples_unitfix_with_true_dates.jsonl")
    sortd = [(m["query_metadata"].get("temporal_sort_date") or "") for m in meta]
    win = {c: [i for i, d in enumerate(sortd) if lo < d <= hi]
           for c, (lo, hi) in (("2020-12-31", ("2020-12-31", "2021-12-31")),
                               ("2021-12-31", ("2021-12-31", "2022-12-31")),
                               ("2022-12-31", ("2022-12-31", "9999")))}
    nlls = []
    for c in CUTOFFS:
        model, anchor = folds[c]
        for i in win[c]:
            fe = filt(ex[i], [True] * len(ex[i]["components"]))
            y0, n0 = fe["query"]["count"], fe["query"]["denominator"]
            w, a, b_ = capped_mixture(model, fe, anchor)
            pmf = mix_pmf(w, a, b_, n0)
            nlls.append(-math.log(max(pmf[int(y0)], 1e-300)))
    res["forward_nll"] = float(np.mean(nlls))
    res["forward_n"] = len(nlls)
    OUT.write_text(json.dumps(res, indent=1))
    print(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
