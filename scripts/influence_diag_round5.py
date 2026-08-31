#!/usr/bin/env python3
"""Round-5: influence diagnostics for the historical mixture-mass cap
(review round-5 Major 4).

The cap bounds prior component PROBABILITY, not information, so this script
measures actual influence on real data, per example, for the capped design:

  - posterior historical weight: total posterior mixture weight on
    historical components (prior mass <= 0.5 by the cap, but the posterior
    weight can exceed it when donors fit the outcome better - Bayes updating,
    quantified rather than hidden);
  - posterior mean shift: |E(theta | capped mixture) - E(theta | anchor only)|;
  - weighted nominal borrowed size after the cap (bookkeeping).

Tiers: forward (canonical fold models, full-information upper bound) and the
primary availability rows.  Output:
artifacts_round4/stress/influence_diag.json
"""
from __future__ import annotations

import csv
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
from endpoint_canonicalizer import is_strict_orr  # noqa: E402
from capped_real_data import bb, filt, read_jsonl  # noqa: E402

EX = ROOT / "artifacts_round2" / "examples_unitfix_ids_ep.jsonl"
META = ROOT / "artifacts_rerun" / "examples_unitfix_with_true_dates.jsonl"
DATES = ROOT / "artifacts_rerun" / "clinicaltrials_date_rows.csv"
SEL_DIR = ROOT / "artifacts_pathfinding" / "selective"
STRICT = ROOT / "artifacts_round3" / "strict_orr"
OUT = ROOT / "artifacts_round4" / "stress" / "influence_diag.json"
CUTOFFS = ("2020-12-31", "2021-12-31", "2022-12-31")
CAP = 0.5


def capped_posterior_stats(model, fe, anchor):
    y0, n0 = fe["query"]["count"], fe["query"]["denominator"]
    a0, b0 = anchor
    anchor_mean = (a0 + y0) / (a0 + b0 + n0)
    if not any(c["gate"] > 0 for c in fe["components"]):
        return {"post_hist_weight": 0.0, "mean_shift": 0.0, "borrowed_size": 0.0,
                "prior_mass": 0.0}
    with torch.no_grad():
        t = TS.LT._validated_example_tensors(fe)
        lam0, lam_i = TS.selective_lambda_weights(model, t["features"], t["gate"])
        a, b_, disc = TS.LT._component_alpha_beta_for_model(model, t["features"], t)
        lam_i = lam_i.numpy().astype(float)
        a = a.numpy().astype(float)
        b_ = b_.numpy().astype(float)
        disc = disc.numpy().astype(float)
        lam0 = float(lam0)
    mass = lam_i.sum()
    if mass > CAP:
        lam_i = lam_i * (CAP / mass)
        lam0 = 1.0 - lam_i.sum()
    logws = [math.log(max(lam0, 1e-300)) + bb(y0, n0, a0, b0)]
    means = [anchor_mean]
    for w, ai, bi in zip(lam_i, a, b_):
        logws.append(math.log(max(w, 1e-300)) + bb(y0, n0, ai, bi))
        means.append((ai + y0) / (ai + bi + n0))
    mx = max(logws)
    ws = np.exp(np.array(logws) - mx)
    ws = ws / ws.sum()
    post_mean = float((ws * np.array(means)).sum())
    ns = [c["denominator"] for c in fe["components"]]
    return {"post_hist_weight": float(ws[1:].sum()),
            "mean_shift": abs(post_mean - anchor_mean),
            "borrowed_size": float(sum(w * d * n for w, d, n
                                       in zip(lam_i, disc, ns))),
            "prior_mass": float(lam_i.sum())}


def summarize(vals):
    v = np.array(vals)
    return {"n": int(len(v)), "mean": round(float(v.mean()), 4),
            "q50": round(float(np.quantile(v, 0.5)), 4),
            "q90": round(float(np.quantile(v, 0.9)), 4),
            "q99": round(float(np.quantile(v, 0.99)), 4),
            "max": round(float(v.max()), 4)}


def main():
    OUT.parent.mkdir(parents=True, exist_ok=True)
    ex = read_jsonl(EX)
    meta = read_jsonl(META)
    dates = {r["nct_id"]: r for r in csv.DictReader(open(DATES))}
    posted = {k: (v.get("results_first_posted_date") or "") for k, v in dates.items()}
    sortd = [(m["query_metadata"].get("temporal_sort_date") or "") for m in meta]
    startd = [(m["query_metadata"].get("start_date") or "") for m in meta]
    q_strict = [is_strict_orr(e["query"].get("endpoint")) for e in ex]
    c_strict = [[is_strict_orr(c.get("endpoint")) for c in e["components"]] for e in ex]
    win = {w: [i for i, d in enumerate(sortd) if lo < d <= hi]
           for w, (lo, hi) in (("w1", ("2020-12-31", "2021-12-31")),
                               ("w2", ("2021-12-31", "2022-12-31")),
                               ("w3", ("2022-12-31", "9999")))}
    folds = {c: TS.load_selective_artifact(SEL_DIR / f"selective_model_fold_{c}.pt")
             for c in CUTOFFS}

    res = {}
    # forward tier (upper bound)
    stats = []
    for w, c in zip(("w1", "w2", "w3"), CUTOFFS):
        model, anchor = folds[c]
        for i in win[w]:
            fe = filt(ex[i], [True] * len(ex[i]["components"]))
            stats.append(capped_posterior_stats(model, fe, anchor))
    res["forward_upper_bound"] = {
        k: summarize([s[k] for s in stats])
        for k in ("post_hist_weight", "mean_shift", "borrowed_size", "prior_mass")}
    res["forward_upper_bound"]["frac_post_weight_gt_0.5"] = round(
        float(np.mean([s["post_hist_weight"] > 0.5 for s in stats])), 4)
    res["forward_upper_bound"]["frac_post_weight_gt_0.8"] = round(
        float(np.mean([s["post_hist_weight"] > 0.8 for s in stats])), 4)

    # primary availability rows
    ssum = json.load(open(STRICT / "summary.json"))
    from capped_real_data import load_strict_model
    mB = {c: load_strict_model(STRICT / "models" / f"strict_avail_{c}.pt",
                               ssum["B_strict_availability"][c]["anchor"])
          for c in CUTOFFS}
    stats = []
    for w, c in zip(("w1", "w2", "w3"), CUTOFFS):
        model, anchor = mB[c]
        for i in win[w]:
            if not q_strict[i]:
                continue
            qs = startd[i]
            keep = [cs and qs != "" and posted.get(cc["nct_id"], "") != ""
                    and posted[cc["nct_id"]] < qs
                    for cc, cs in zip(ex[i]["components"], c_strict[i])]
            fe = filt(ex[i], keep)
            stats.append(capped_posterior_stats(model, fe, anchor))
    res["primary"] = {
        k: summarize([s[k] for s in stats])
        for k in ("post_hist_weight", "mean_shift", "borrowed_size", "prior_mass")}
    res["primary"]["frac_post_weight_gt_0.5"] = round(
        float(np.mean([s["post_hist_weight"] > 0.5 for s in stats])), 4)

    OUT.write_text(json.dumps(res, indent=1))
    print(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
