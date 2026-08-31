#!/usr/bin/env python3
"""Donor-availability-controlled rerun of the forward validation (round 2, point 1).

Two availability rules, applied to every prior that uses candidates:

  World A (analysis-time, per fold cutoff c):
      training queries   : completed <= c AND query results first posted <= c
      training donors    : results first posted <= c
      EB anchor          : fitted on the World-A training pool
      (answers: could this whole analysis have been run at time c?)

  Design-time scoring rule (deployment replay, per query q):
      scoring donors     : results first posted < q's start date
      (answers: what could the sponsor designing q actually have borrowed?)

Window membership and scored query sets are unchanged (181/204/390); the
filters change what each prior may use, not who is scored. Queries left with
no usable gated donor fall back to their anchor (selective/EB -> EB anchor,
rule/weak -> flat Beta(1,1)), which is exactly what would have happened at
design time.

Outputs artifacts_round2/availability/{availability_rows.json, summary.json,
models/}, plus a printed digest.
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

import two_head_selective as TS  # noqa: E402

EX_IDS = ROOT / "artifacts_round2" / "examples_unitfix_ids.jsonl"
EX_META = ROOT / "artifacts_rerun" / "examples_unitfix_with_true_dates.jsonl"
DATES = ROOT / "artifacts_rerun" / "clinicaltrials_date_rows.csv"
REWRITE = ROOT / "artifacts_pathfinding" / "rewrite_numbers_perexample.json"
OUT = ROOT / "artifacts_round2" / "availability"
CUTOFFS = ("2020-12-31", "2021-12-31", "2022-12-31")
WINDOWS = {"w1": ("2020-12-31", "2021-12-31"), "w2": ("2021-12-31", "2022-12-31"),
           "w3": ("2022-12-31", "9999-12-31")}
SEED = 20260603
B = 4000


def bb_log_pmf(y, n, a, b):
    y, n = float(y), float(n)
    return (math.lgamma(n + 1) - math.lgamma(y + 1) - math.lgamma(n - y + 1)
            + math.lgamma(y + a) + math.lgamma(n - y + b) - math.lgamma(n + a + b)
            + math.lgamma(a + b) - math.lgamma(a) - math.lgamma(b))


def read_jsonl(p):
    with open(p) as f:
        return [json.loads(l) for l in f if l.strip()]


def filtered_example(ex, keep_mask):
    comps = [c for c, k in zip(ex["components"], keep_mask) if k]
    feats = [f for f, k in zip(ex["features"], keep_mask) if k]
    lr = [v for v, k in zip(ex["lambda_rule"], keep_mask) if k]
    s = sum(lr)
    if s > 0:
        lr = [v * 0.8 / s for v in lr]  # renormalise rule allocation to the 0.8 budget
    return {"query": ex["query"], "feature_names": ex["feature_names"],
            "features": feats, "components": comps, "lambda_rule": lr,
            "lambda_0": ex.get("lambda_0", 0.2)}


def rule_nll(ex_f):
    y0, n0 = float(ex_f["query"]["count"]), float(ex_f["query"]["denominator"])
    comps = [c for c in ex_f["components"] if c["gate"] > 0]
    lr = [l for c, l in zip(ex_f["components"], ex_f["lambda_rule"]) if c["gate"] > 0]
    if not comps or sum(lr) <= 0:
        return -bb_log_pmf(y0, n0, 1.0, 1.0)
    terms = [math.log(0.2) + bb_log_pmf(y0, n0, 1.0, 1.0)]
    s = sum(lr)
    for c, l in zip(comps, lr):
        lam = l * 0.8 / s if abs(s - 0.8) > 1e-12 else l
        a = c["discount"]
        terms.append(math.log(max(lam, 1e-300))
                     + bb_log_pmf(y0, n0, 1 + a * c["count"], 1 + a * (c["denominator"] - c["count"])))
    m = max(terms)
    return -(m + math.log(sum(math.exp(t - m) for t in terms)))


def selective_nll(model, anchor, ex_f):
    y0, n0 = float(ex_f["query"]["count"]), float(ex_f["query"]["denominator"])
    gated = [c for c in ex_f["components"] if c["gate"] > 0]
    if not gated:
        return -bb_log_pmf(y0, n0, anchor[0], anchor[1]), 1.0
    d = TS.selective_details(model, ex_f, anchor)
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
    ex_ids = read_jsonl(EX_IDS)
    meta = read_jsonl(EX_META)
    dates = {r["nct_id"]: r for r in csv.DictReader(open(DATES))}
    rw = json.load(open(REWRITE))
    pub_fwd = {r["i"]: r for r in rw["fwd"]}

    posted = {k: (v.get("results_first_posted_date") or "") for k, v in dates.items()}
    sortd, startd = [], []
    for m in meta:
        md = m["query_metadata"]
        sortd.append(md.get("temporal_sort_date") or "")
        startd.append(md.get("start_date") or "")

    win_idx = {w: [i for i, d in enumerate(sortd)
                   if lo < d <= hi] for w, (lo, hi) in
               ((w, WINDOWS[w]) for w in WINDOWS)}
    assert [len(win_idx[w]) for w in ("w1", "w2", "w3")] == [181, 204, 390]

    def comp_posted(c):
        return posted.get(c["nct_id"], "")

    models = {}
    summary = {"train_pools": {}, "windows": {}, "notes": {
        "worldA": "training queries and donors restricted to results posted <= cutoff",
        "design_rule": "scoring donors restricted to results posted < query start date",
    }}
    t0 = time.time()
    for c in CUTOFFS:
        pool = [i for i, d in enumerate(sortd)
                if d and d <= c and posted.get(ex_ids[i]["query_nct_id"], "") and
                posted[ex_ids[i]["query_nct_id"]] <= c]
        # World-A donor filter on training examples
        f_train, kept, comp_tot, comp_ok = [], 0, 0, 0
        for i in pool:
            ex = ex_ids[i]
            mask = [comp_posted(cc) != "" and comp_posted(cc) <= c for cc in ex["components"]]
            comp_tot += len(mask); comp_ok += sum(mask)
            fe = filtered_example(ex, mask)
            if any(cc["gate"] > 0 for cc in fe["components"]):
                f_train.append(fe); kept += 1
        anchor = TS.fit_eb_anchor(
            [(ex_ids[i]["query"]["count"], ex_ids[i]["query"]["denominator"]) for i in pool])
        summary["train_pools"][c] = {
            "pool_completed": sum(1 for d in sortd if d and d <= c),
            "pool_posted": len(pool), "trainable_with_gated_donor": kept,
            "donor_comps": comp_tot, "donor_comps_surviving": comp_ok,
            "anchor": [round(anchor[0], 4), round(anchor[1], 4)],
        }
        res = TS.train_selective(f_train, list(range(len(f_train))), [], anchor,
                                 epochs=100, learning_rate=0.01, hidden_dim=16, seed=SEED)
        models[c] = (res["model"], anchor)
        torch.save(res["model"].state_dict(), OUT / "models" / f"selective_avail_{c}.pt")
        print(f"[{time.time()-t0:5.0f}s] fold {c}: pool {len(pool)} "
              f"(trainable {kept}), anchor Beta({anchor[0]:.3f},{anchor[1]:.3f})", flush=True)

    rows = {}
    rng = np.random.default_rng(20260819)
    for w, c in zip(("w1", "w2", "w3"), CUTOFFS):
        model, anchor = models[c]
        for i in win_idx[w]:
            ex = ex_ids[i]
            y0, n0 = ex["query"]["count"], ex["query"]["denominator"]
            qs = startd[i]
            mask_design = [qs != "" and comp_posted(cc) != "" and comp_posted(cc) < qs
                           for cc in ex["components"]]
            mask_cut = [comp_posted(cc) != "" and comp_posted(cc) <= c
                        for cc in ex["components"]]
            fe_d = filtered_example(ex, mask_design)
            fe_c = filtered_example(ex, mask_cut)
            sel_d, lam0_d = selective_nll(model, anchor, fe_d)
            sel_c, _ = selective_nll(model, anchor, fe_c)
            rows[i] = {
                "i": i, "window": w,
                "eb_avail": -bb_log_pmf(y0, n0, anchor[0], anchor[1]),
                "weak": -bb_log_pmf(y0, n0, 1.0, 1.0),
                "rule_design": rule_nll(fe_d),
                "selective_design": sel_d,
                "selective_cutoffrule": sel_c,
                "lambda0_design": lam0_d,
                "n_gated_design": sum(1 for cc, k in zip(ex["components"], mask_design)
                                      if k and cc["gate"] > 0),
                "n_gated_orig": sum(1 for cc in ex["components"] if cc["gate"] > 0),
                "eb_published": pub_fwd[i]["eb"],
                "selective_published": pub_fwd[i]["selective"],
            }
    json.dump({"rows": {str(k): v for k, v in rows.items()}},
              open(OUT / "availability_rows.json", "w"))

    def agg(idxs, key_a, key_b):
        return boot([rows[i][key_a] - rows[i][key_b] for i in idxs], rng)

    for w in ("w1", "w2", "w3", "pooled"):
        idxs = (win_idx[w] if w != "pooled"
                else [i for ww in ("w1", "w2", "w3") for i in win_idx[ww]])
        have = [i for i in idxs if rows[i]["n_gated_design"] > 0]
        s = {
            "n": len(idxs), "n_with_design_avail_donor": len(have),
            "mean_nll": {k: float(np.mean([rows[i][k] for i in idxs]))
                         for k in ("eb_avail", "selective_design",
                                   "selective_cutoffrule", "rule_design",
                                   "eb_published", "selective_published")},
            "selective_design_vs_eb_avail": agg(idxs, "selective_design", "eb_avail"),
            "selective_cutoff_vs_eb_avail": agg(idxs, "selective_cutoffrule", "eb_avail"),
            "rule_design_vs_eb_avail": agg(idxs, "rule_design", "eb_avail"),
            "eb_avail_vs_eb_published": agg(idxs, "eb_avail", "eb_published"),
            "subgroup_with_avail_donor": {
                "selective_design_vs_eb_avail": agg(have, "selective_design", "eb_avail")
                if len(have) >= 10 else None,
                "mean_lambda0_design": float(np.mean([rows[i]["lambda0_design"] for i in have]))
                if have else None,
            },
        }
        summary["windows"][w] = s
    json.dump(summary, open(OUT / "summary.json", "w"), indent=1)

    print(json.dumps(summary["train_pools"], indent=1))
    for w, s in summary["windows"].items():
        d = s["selective_design_vs_eb_avail"]
        sub = s["subgroup_with_avail_donor"]["selective_design_vs_eb_avail"]
        print(f"{w}: n={s['n']} avail_donor={s['n_with_design_avail_donor']} | "
              f"sel_design vs eb_avail {d['mean']:+.4f} [{d['ci95'][0]:+.4f},{d['ci95'][1]:+.4f}] | "
              f"subgroup {('%+.4f [%+.4f,%+.4f] n=%d' % (sub['mean'], sub['ci95'][0], sub['ci95'][1], sub['n'])) if sub else 'n/a'}")


if __name__ == "__main__":
    main()
