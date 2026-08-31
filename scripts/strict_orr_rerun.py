#!/usr/bin/env python3
"""Strict-ORR primary reruns (round 3, Major 3, path 1).

Estimand: queries whose extracted primary endpoint is strict ORR under the
audit-validated canonicaliser; donor components restricted the same way.

Three analyses:
  A. STRICT PRIMARY (forward): selective retrained per fold on the strict
     subset; EB anchors refit on strict pools; strict windows scored once.
  B. STRICT + AVAILABILITY (primary real-data analysis of the paper):
     World-A training (results posted by cutoff) intersected with strict;
     scoring donors under the registry-availability proxy (posted before the
     query's start date) intersected with strict.
  C. FROZEN-MODEL SENSITIVITY: the released canonical models scored on
     strict-filtered examples (no retraining), for continuity with round 2.

Outputs artifacts_round3/strict_orr/{summary.json, models/}.
"""
from __future__ import annotations

import csv
import json
import math
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "pipeline"))
sys.path.insert(0, str(ROOT / "scripts"))

import two_head_selective as TS  # noqa: E402
from endpoint_canonicalizer import is_strict_orr  # noqa: E402

EX = ROOT / "artifacts_round2" / "examples_unitfix_ids_ep.jsonl"
META = ROOT / "artifacts_rerun" / "examples_unitfix_with_true_dates.jsonl"
DATES = ROOT / "artifacts_rerun" / "clinicaltrials_date_rows.csv"
SEL_DIR = ROOT / "artifacts_pathfinding" / "selective"
OUT = ROOT / "artifacts_round3" / "strict_orr"
CUTOFFS = ("2020-12-31", "2021-12-31", "2022-12-31")
SEED = 20260603
B = 4000


def bb(y, n, a, b_):
    y, n = float(y), float(n)
    return (math.lgamma(n + 1) - math.lgamma(y + 1) - math.lgamma(n - y + 1)
            + math.lgamma(y + a) + math.lgamma(n - y + b_) - math.lgamma(n + a + b_)
            + math.lgamma(a + b_) - math.lgamma(a) - math.lgamma(b_))


def read_jsonl(p):
    with open(p) as f:
        return [json.loads(l) for l in f if l.strip()]


def filt(ex, keep):
    comps = [c for c, k in zip(ex["components"], keep) if k]
    feats = [f for f, k in zip(ex["features"], keep) if k]
    lr = [v for v, k in zip(ex["lambda_rule"], keep) if k]
    s = sum(lr)
    if s > 0:
        lr = [v * 0.8 / s for v in lr]
    return {"query": ex["query"], "feature_names": ex["feature_names"],
            "features": feats, "components": comps, "lambda_rule": lr}


def sel_nll(model, anchor, fe):
    y0, n0 = fe["query"]["count"], fe["query"]["denominator"]
    if not any(c["gate"] > 0 for c in fe["components"]):
        return -bb(y0, n0, anchor[0], anchor[1]), 1.0
    d = TS.selective_details(model, fe, anchor)
    return d["nll"], d["lambda0"]


def boot(diffs, rng):
    d = np.asarray(diffs)
    idx = rng.integers(0, len(d), size=(B, len(d)))
    m = d[idx].mean(axis=1)
    lo, hi = np.quantile(m, [0.025, 0.975])
    return {"mean": float(d.mean()), "ci95": [float(lo), float(hi)],
            "n": int(len(d)), "win": float((d < 0).mean())}


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "models").mkdir(exist_ok=True)
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
    win_s = {w: [i for i in v if q_strict[i]] for w, v in win.items()}
    rng = np.random.default_rng(20260824)
    summary = {"subset_sizes": {
        "queries_strict": int(sum(q_strict)),
        "windows_strict": {w: len(v) for w, v in win_s.items()},
        "components_strict": int(sum(sum(m) for m in c_strict)),
    }, "A_strict_primary": {}, "B_strict_availability": {},
       "C_frozen_sensitivity": {}}

    t0 = time.time()

    # ---------- A: strict primary (retrained) ------------------------------
    rows_A = {}
    for w, c in zip(("w1", "w2", "w3"), CUTOFFS):
        pool = [i for i, d in enumerate(sortd) if d and d <= c and q_strict[i]]
        train = []
        for i in pool:
            fe = filt(ex[i], c_strict[i])
            if any(cc["gate"] > 0 for cc in fe["components"]):
                train.append(fe)
        anchor = TS.fit_eb_anchor([(ex[i]["query"]["count"],
                                    ex[i]["query"]["denominator"]) for i in pool])
        res = TS.train_selective(train, list(range(len(train))), [], anchor,
                                 epochs=100, learning_rate=0.01, hidden_dim=16,
                                 seed=SEED)
        model = res["model"]
        torch.save(model.state_dict(), OUT / "models" / f"strict_{c}.pt")
        summary["A_strict_primary"][c] = {
            "train_pool": len(pool), "trainable": len(train),
            "anchor": [round(anchor[0], 4), round(anchor[1], 4)]}
        for i in win_s[w]:
            fe = filt(ex[i], c_strict[i])
            nll, lam0 = sel_nll(model, anchor, fe)
            rows_A[i] = {"sel": nll, "eb": -bb(ex[i]["query"]["count"],
                                              ex[i]["query"]["denominator"],
                                              anchor[0], anchor[1]),
                         "lam0": lam0}
        print(f"[{time.time()-t0:5.0f}s] A fold {c}: pool {len(pool)} "
              f"anchor Beta({anchor[0]:.2f},{anchor[1]:.2f})", flush=True)
    allA = [i for w in ("w1", "w2", "w3") for i in win_s[w]]
    summary["A_strict_primary"]["pooled_sel_vs_eb"] = boot(
        [rows_A[i]["sel"] - rows_A[i]["eb"] for i in allA], rng)
    summary["A_strict_primary"]["w3_sel_vs_eb"] = boot(
        [rows_A[i]["sel"] - rows_A[i]["eb"] for i in win_s["w3"]], rng)
    summary["A_strict_primary"]["mean_nll"] = {
        "selective": float(np.mean([rows_A[i]["sel"] for i in allA])),
        "eb": float(np.mean([rows_A[i]["eb"] for i in allA]))}

    # ---------- B: strict + availability (primary analysis) ----------------
    rows_B = {}
    for w, c in zip(("w1", "w2", "w3"), CUTOFFS):
        pool = [i for i, d in enumerate(sortd)
                if d and d <= c and q_strict[i]
                and posted.get(ex[i]["query_nct_id"], "")
                and posted[ex[i]["query_nct_id"]] <= c]
        train = []
        for i in pool:
            keep = [cs and posted.get(cc["nct_id"], "") != ""
                    and posted[cc["nct_id"]] <= c
                    for cc, cs in zip(ex[i]["components"], c_strict[i])]
            fe = filt(ex[i], keep)
            if any(cc["gate"] > 0 for cc in fe["components"]):
                train.append(fe)
        anchor = TS.fit_eb_anchor([(ex[i]["query"]["count"],
                                    ex[i]["query"]["denominator"]) for i in pool])
        res = TS.train_selective(train, list(range(len(train))), [], anchor,
                                 epochs=100, learning_rate=0.01, hidden_dim=16,
                                 seed=SEED)
        model = res["model"]
        torch.save(model.state_dict(), OUT / "models" / f"strict_avail_{c}.pt")
        summary["B_strict_availability"][c] = {
            "train_pool": len(pool), "trainable": len(train),
            "anchor": [round(anchor[0], 4), round(anchor[1], 4)]}
        n_avail = 0
        for i in win_s[w]:
            qs = startd[i]
            keep = [cs and qs != "" and posted.get(cc["nct_id"], "") != ""
                    and posted[cc["nct_id"]] < qs
                    for cc, cs in zip(ex[i]["components"], c_strict[i])]
            fe = filt(ex[i], keep)
            nll, lam0 = sel_nll(model, anchor, fe)
            has = any(cc["gate"] > 0 for cc in fe["components"])
            n_avail += has
            rows_B[i] = {"sel": nll, "eb": -bb(ex[i]["query"]["count"],
                                               ex[i]["query"]["denominator"],
                                               anchor[0], anchor[1]),
                         "has_donor": has}
        summary["B_strict_availability"][c]["queries_with_avail_donor"] = n_avail
        print(f"[{time.time()-t0:5.0f}s] B fold {c}: pool {len(pool)}", flush=True)
    allB = [i for w in ("w1", "w2", "w3") for i in win_s[w]]
    have = [i for i in allB if rows_B[i]["has_donor"]]
    summary["B_strict_availability"]["pooled_sel_vs_eb"] = boot(
        [rows_B[i]["sel"] - rows_B[i]["eb"] for i in allB], rng)
    summary["B_strict_availability"]["subgroup_with_donor"] = boot(
        [rows_B[i]["sel"] - rows_B[i]["eb"] for i in have], rng) if len(have) >= 10 else None
    summary["B_strict_availability"]["n_with_avail_donor"] = len(have)

    # ---------- C: frozen canonical models, strict filter -------------------
    rw = json.load(open(ROOT / "artifacts_pathfinding" / "rewrite_numbers_perexample.json"))
    fwd = {r["i"]: r for r in rw["fwd"]}
    folds = {c: TS.load_selective_artifact(SEL_DIR / f"selective_model_fold_{c}.pt")
             for c in CUTOFFS}
    diffs = []
    for w, c in zip(("w1", "w2", "w3"), CUTOFFS):
        model, anchor = folds[c]
        for i in win_s[w]:
            fe = filt(ex[i], c_strict[i])
            nll, _ = sel_nll(model, anchor, fe)
            diffs.append(nll - fwd[i]["eb"])
    summary["C_frozen_sensitivity"]["pooled_sel_vs_eb_strictfiltered"] = boot(diffs, rng)

    json.dump(summary, open(OUT / "summary.json", "w"), indent=1)
    A = summary["A_strict_primary"]; Bx = summary["B_strict_availability"]
    print("\nA strict primary pooled:", A["pooled_sel_vs_eb"])
    print("A strict w3:", A["w3_sel_vs_eb"])
    print("B strict+avail pooled:", Bx["pooled_sel_vs_eb"])
    print("B subgroup with donor:", Bx["subgroup_with_donor"], "n=", Bx["n_with_avail_donor"])
    print("C frozen strict-filtered:", summary["C_frozen_sensitivity"])


if __name__ == "__main__":
    main()
