#!/usr/bin/env python3
"""Round-2 point 4/8: donor-clustered bootstrap + count-only sensitivity.

1. Donor-clustered bootstrap: queries sharing any donor are dependent, so the
   paired bootstrap resamples CLUSTERS (connected components of the shared-donor
   graph) instead of queries. Recomputes the headline CIs:
     - forward pooled: selective vs EB, selective vs fixed-budget two-head
     - window 3:       selective vs EB
     - test split:     selective vs EB, selective vs two-head
2. Count-only sensitivity: restrict to queries whose OWN outcome is
   participant-count-reported (not percentage-derived), and separately score
   priors built after dropping percentage-derived donor observations.

Outputs artifacts_round2/cluster_sensitivity.json + digest.
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
import two_head_selective as TS  # noqa: E402

EX_IDS = ROOT / "artifacts_round2" / "examples_unitfix_ids.jsonl"
EX_META = ROOT / "artifacts_rerun" / "examples_unitfix_with_true_dates.jsonl"
REWRITE = ROOT / "artifacts_pathfinding" / "rewrite_numbers_perexample.json"
SEL_DIR = ROOT / "artifacts_pathfinding" / "selective"
OUT = ROOT / "artifacts_round2"
B = 4000


def read_jsonl(p):
    with open(p) as f:
        return [json.loads(l) for l in f if l.strip()]


def bb_log_pmf(y, n, a, b):
    y, n = float(y), float(n)
    return (math.lgamma(n + 1) - math.lgamma(y + 1) - math.lgamma(n - y + 1)
            + math.lgamma(y + a) + math.lgamma(n - y + b) - math.lgamma(n + a + b)
            + math.lgamma(a + b) - math.lgamma(a) - math.lgamma(b))


def clusters_of(idxs, ex_ids):
    """Connected components of the shared-donor graph over the given queries."""
    parent = {i: i for i in idxs}

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a, b):
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[ra] = rb

    donor_to_q = {}
    for i in idxs:
        for c in ex_ids[i]["components"]:
            if c["gate"] <= 0:
                continue
            d = c["nct_id"]
            if d in donor_to_q:
                union(i, donor_to_q[d])
            else:
                donor_to_q[d] = i
    groups = {}
    for i in idxs:
        groups.setdefault(find(i), []).append(i)
    return list(groups.values())


def cluster_boot(diff_by_i, clusters, rng):
    means = []
    K = len(clusters)
    csums = [np.sum([diff_by_i[i] for i in cl]) for cl in clusters]
    csizes = [len(cl) for cl in clusters]
    csums = np.asarray(csums); csizes = np.asarray(csizes, dtype=float)
    for _ in range(B):
        pick = rng.integers(0, K, size=K)
        tot = csums[pick].sum(); n = csizes[pick].sum()
        means.append(tot / n)
    means = np.asarray(means)
    lo, hi = np.quantile(means, [0.025, 0.975])
    all_i = [i for cl in clusters for i in cl]
    return {"mean": float(np.mean([diff_by_i[i] for i in all_i])),
            "ci95_cluster": [float(lo), float(hi)],
            "n_queries": len(all_i), "n_clusters": K,
            "max_cluster": int(csizes.max())}


def main():
    ex_ids = read_jsonl(EX_IDS)
    meta = read_jsonl(EX_META)
    rw = json.load(open(REWRITE))
    rng = np.random.default_rng(20260820)

    fwd = {r["i"]: r for r in rw["fwd"]}
    test = {r["i"]: r for r in rw["test"]}
    sortd = [ (m["query_metadata"].get("temporal_sort_date") or "") for m in meta]
    w3 = [i for i in fwd if sortd[i] > "2022-12-31"]

    res = {"cluster_bootstrap": {}, "count_only": {}}

    # ---- 1. donor-clustered bootstrap ------------------------------------
    for name, idxs, a, b in [
        ("fwd_selective_vs_eb", list(fwd), "selective", "eb"),
        ("fwd_selective_vs_two_head", list(fwd), "selective", "two_head_old"),
        ("w3_selective_vs_eb", w3, "selective", "eb"),
        ("test_selective_vs_eb", list(test), "selective", "eb"),
        ("test_selective_vs_two_head", list(test), "selective", "two_head_old"),
    ]:
        src = fwd if idxs is not w3 and idxs == list(fwd) else (fwd if idxs is w3 else test)
        src = fwd if (name.startswith("fwd") or name.startswith("w3")) else test
        diffs = {i: src[i][a] - src[i][b] for i in idxs}
        cls = clusters_of(idxs, ex_ids)
        res["cluster_bootstrap"][name] = cluster_boot(diffs, cls, rng)

    # ---- 2. count-only sensitivity ---------------------------------------
    # (a) query-side: keep only count-reported query outcomes, published rows
    def is_pct(u):
        return isinstance(u, str) and "percent" in u.lower()

    q_count_fwd = [i for i in fwd if not is_pct(ex_ids[i]["query"].get("unit"))]
    q_count_test = [i for i in test if not is_pct(ex_ids[i]["query"].get("unit"))]

    def plain_boot(vals):
        d = np.asarray(vals)
        idx = rng.integers(0, len(d), size=(B, len(d)))
        m = d[idx].mean(axis=1)
        lo, hi = np.quantile(m, [0.025, 0.975])
        return {"mean": float(d.mean()), "ci95": [float(lo), float(hi)], "n": len(d)}

    res["count_only"]["query_side_fwd_selective_vs_eb"] = plain_boot(
        [fwd[i]["selective"] - fwd[i]["eb"] for i in q_count_fwd])
    res["count_only"]["query_side_test_selective_vs_eb"] = plain_boot(
        [test[i]["selective"] - test[i]["eb"] for i in q_count_test])

    # (b) donor-side: drop percentage-derived donor observations, rescore
    prim_model, prim_anchor = TS.load_selective_artifact(SEL_DIR / "selective_model_primary.pt")
    folds = {c: TS.load_selective_artifact(SEL_DIR / f"selective_model_fold_{c}.pt")
             for c in ("2020-12-31", "2021-12-31", "2022-12-31")}

    def filt(ex):
        mask = [not is_pct(c.get("unit")) for c in ex["components"]]
        comps = [c for c, k in zip(ex["components"], mask) if k]
        feats = [f for f, k in zip(ex["features"], mask) if k]
        lr = [v for v, k in zip(ex["lambda_rule"], mask) if k]
        s = sum(lr)
        if s > 0:
            lr = [v * 0.8 / s for v in lr]
        return {"query": ex["query"], "feature_names": ex["feature_names"],
                "features": feats, "components": comps, "lambda_rule": lr}

    def sel_nll(model, anchor, fe):
        y0, n0 = fe["query"]["count"], fe["query"]["denominator"]
        if not any(c["gate"] > 0 for c in fe["components"]):
            return -bb_log_pmf(y0, n0, anchor[0], anchor[1])
        return TS.selective_details(model, fe, anchor)["nll"]

    diffs_fwd, dropped = [], 0
    for i in fwd:
        c = ("2020-12-31" if sortd[i] <= "2021-12-31"
             else "2021-12-31" if sortd[i] <= "2022-12-31" else "2022-12-31")
        model, anchor = folds[c]
        fe = filt(ex_ids[i])
        dropped += len(ex_ids[i]["components"]) - len(fe["components"])
        diffs_fwd.append(sel_nll(model, anchor, fe) - fwd[i]["eb"])
    res["count_only"]["donor_side_fwd_selective_vs_eb"] = plain_boot(diffs_fwd)
    res["count_only"]["donor_side_fwd_components_dropped"] = dropped

    diffs_t = []
    for i in test:
        fe = filt(ex_ids[i])
        diffs_t.append(sel_nll(prim_model, prim_anchor, fe) - test[i]["eb"])
    res["count_only"]["donor_side_test_selective_vs_eb"] = plain_boot(diffs_t)

    json.dump(res, open(OUT / "cluster_sensitivity.json", "w"), indent=1)
    for k, v in res["cluster_bootstrap"].items():
        print(f"{k}: mean {v['mean']:+.4f} cluster-CI {v['ci95_cluster'][0]:+.4f},"
              f"{v['ci95_cluster'][1]:+.4f} (clusters {v['n_clusters']}, max {v['max_cluster']})")
    for k, v in res["count_only"].items():
        if isinstance(v, dict):
            print(f"{k}: {v['mean']:+.4f} [{v['ci95'][0]:+.4f},{v['ci95'][1]:+.4f}] n={v['n']}")
        else:
            print(k, v)


if __name__ == "__main__":
    main()
