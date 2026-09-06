"""Round-11: inference-scope computations for the randomized-boundary
analysis (review round-11 items 1, 2, and the location/dispersion
qualifier).

  (a) evaluation-half size-gap DISTRIBUTIONS: the round-10 artifact
      reported only the mean/SD of the per-split evaluation-half size gap
      (cap50 minus no borrowing); this script re-runs every randomized
      configuration (identical RNG 20260908, estimates asserted equal to
      the round-10 artifact) and stores the per-split gaps with their
      quantiles;
  (b) location-held mixings: N(+0.124, 1) and N(+0.124, 0.5) --- the
      corpus mean with narrowed SD --- at the common target, so the
      dispersion gradient is separated from the location difference of
      the N(0, sd) runs;
  (c) full-batch randomized plug-in at the common target with its
      within-replicate Monte Carlo standard error recomputed for the
      randomized estimator (not inherited from the deterministic one);
  (d) the mixing-weight bootstrap re-run under randomized matching at
      the fixed common target (500 resamples of the 939 strict drift
      queries; RNG 20260905 as in round 5);
  (e) the overlap of the two selection rules of Section 4.4 (post-cap
      0.5 atom vs the pre-specified I >= 8 rule).

Output: artifacts_round4/calibration/round11_inference_scope.json
"""
from __future__ import annotations

import csv
import json
import math
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "simulation" / "gold_standard"))
sys.path.insert(0, str(ROOT / "scripts"))
from round10_randomized_crossfit import (  # noqa: E402
    SHIFTS, normal_weights, rand_threshold, expected_reject)
from endpoint_canonicalizer import is_strict_orr  # noqa: E402
from drift_distribution import elogit  # noqa: E402

RV = ROOT / "artifacts_round4" / "validation" / "pvals"
CAL = ROOT / "artifacts_round4" / "calibration" / "mixture_calibration.json"
R10 = ROOT / "artifacts_round4" / "calibration" / "round10_randomized_crossfit.json"
RW = ROOT / "artifacts_pathfinding" / "rewrite_numbers_perexample.json"
EX = ROOT / "artifacts_round2" / "examples_unitfix_ids_ep.jsonl"
META = ROOT / "artifacts_rerun" / "examples_unitfix_with_true_dates.jsonl"
DATES = ROOT / "artifacts_rerun" / "clinicaltrials_date_rows.csv"
OUT = ROOT / "artifacts_round4" / "calibration" / "round11_inference_scope.json"
R = 200
TGT = 0.02513


def main() -> int:
    Wcan = {float(k): v
            for k, v in json.loads(CAL.read_text())["mixture_weights"].items()}
    nul, alt = {}, {}
    for m in ("weak_only", "cap50"):
        nul[m] = {s: np.load(RV / f"null_s{s}_f1.0.npz")[m] for s in SHIFTS}
        alt[m] = {s: np.load(RV / f"alt_s{s}_f1.0.npz")[m] for s in SHIFTS}
    n = len(nul["weak_only"][0.0])
    half = n // 2
    r10 = json.loads(R10.read_text())["runs"]

    def run(W, tgt):
        rng = np.random.default_rng(20260908)
        ests, gaps = [], []
        for _ in range(R):
            perm = {s: rng.permutation(n) for s in SHIFTS}
            h0 = {s: perm[s][:half] for s in SHIFTS}
            h1 = {s: perm[s][half:] for s in SHIFTS}
            fv, fg = [], []
            for a, b in ((h0, h1), (h1, h0)):
                c_w, g_w = rand_threshold(
                    {s: nul["weak_only"][s][a[s]] for s in SHIFTS}, W, tgt)
                c_m, g_m = rand_threshold(
                    {s: nul["cap50"][s][a[s]] for s in SHIFTS}, W, tgt)
                fv.append(expected_reject(alt["cap50"], b, W, c_m, g_m)
                          - expected_reject(alt["weak_only"], b, W, c_w, g_w))
                fg.append(expected_reject(nul["cap50"], b, W, c_m, g_m)
                          - expected_reject(nul["weak_only"], b, W, c_w, g_w))
            ests.append(float(np.mean(fv)))
            gaps.append(float(np.mean(fg)))
        return np.array(ests), np.array(gaps)

    res = {"R": R, "common_target": TGT, "runs": {}}
    configs = [("canonical", Wcan, 0.02513), ("canonical", Wcan, 0.025),
               ("N(0,1)", normal_weights(0.0, 1.0), 0.02513),
               ("N(0,0.5)", normal_weights(0.0, 0.5), 0.02513),
               ("N(+0.124,1)", normal_weights(0.124, 1.0), 0.02513),
               ("N(+0.124,0.5)", normal_weights(0.124, 0.5), 0.02513)]
    for name, W, tgt in configs:
        e, g = run(W, tgt)
        key = f"{name}@{tgt}"
        entry = {
            "est": {"mean": round(float(e.mean()), 5),
                    "median": round(float(np.median(e)), 5),
                    "sd": round(float(e.std(ddof=1)), 5),
                    "q": {q: round(float(np.quantile(e, float(q))), 5)
                          for q in ("0.025", "0.25", "0.5", "0.75", "0.975")},
                    "frac_positive": round(float((e > 0).mean()), 4)},
            "gap": {"mean": round(float(g.mean()), 6),
                    "sd": round(float(g.std(ddof=1)), 6),
                    "q": {q: round(float(np.quantile(g, float(q))), 6)
                          for q in ("0.025", "0.25", "0.5", "0.75", "0.975")},
                    "min": round(float(g.min()), 6),
                    "max": round(float(g.max()), 6),
                    "mean_abs": round(float(np.abs(g).mean()), 6),
                    "max_abs": round(float(np.abs(g).max()), 6)},
            "gaps_per_split": [round(float(x), 6) for x in g],
        }
        if key in r10:
            same = (entry["est"]["mean"] == r10[key]["mean"]
                    and entry["est"]["q"]["0.025"] == r10[key]["q"]["0.025"]
                    and entry["est"]["q"]["0.975"] == r10[key]["q"]["0.975"])
            entry["matches_round10"] = bool(same)
            assert same, key
        res["runs"][key] = entry
        print(key, "est", entry["est"]["mean"], entry["est"]["q"],
              "| gap mean", entry["gap"]["mean"], "q",
              entry["gap"]["q"], flush=True)

    # ---- (c) full-batch randomized plug-in + within-replicate MCSE --------
    def full_batch(W, tgt):
        c_w, g_w = rand_threshold(
            {s: nul["weak_only"][s] for s in SHIFTS}, W, tgt)
        c_m, g_m = rand_threshold(
            {s: nul["cap50"][s] for s in SHIFTS}, W, tgt)
        est, var = 0.0, 0.0
        for s in SHIFTS:
            am = alt["cap50"][s]
            aw = alt["weak_only"][s]
            rm = (am > c_m).astype(float) + g_m * (am == c_m)
            rw = (aw > c_w).astype(float) + g_w * (aw == c_w)
            d = rm - rw
            est += W[s] * float(d.mean())
            var += (W[s] ** 2) * float(d.var(ddof=1)) / len(d)
        size_w = expected_reject(nul["weak_only"],
                                 {s: np.arange(n) for s in SHIFTS}, W, c_w, g_w)
        size_m = expected_reject(nul["cap50"],
                                 {s: np.arange(n) for s in SHIFTS}, W, c_m, g_m)
        return est, math.sqrt(var), size_w, size_m

    est, mcse, sw, sm = full_batch(Wcan, TGT)
    res["full_batch_randomized"] = {
        "target": TGT, "estimate": round(est, 5), "mcse": round(mcse, 5),
        "size_weak": round(sw, 6), "size_cap50": round(sm, 6)}
    print("full-batch randomized:", res["full_batch_randomized"], flush=True)

    # ---- (d) mixing-weight bootstrap under randomized matching ------------
    ex = [json.loads(l) for l in open(EX)]
    meta = [json.loads(l) for l in open(META)]
    dates = {r_["nct_id"]: r_ for r_ in csv.DictReader(open(DATES))}
    posted = {k: (v.get("results_first_posted_date") or "")
              for k, v in dates.items()}
    startd = [(m["query_metadata"].get("start_date") or "") for m in meta]
    D, V = [], []
    for i, e in enumerate(ex):
        q = e["query"]
        if not is_strict_orr(q.get("endpoint")):
            continue
        lq, vq = elogit(q["count"], q["denominator"])
        ds, vs = [], []
        for c in e["components"]:
            if c["gate"] <= 0 or not is_strict_orr(c.get("endpoint")):
                continue
            ld, vd = elogit(c["count"], c["denominator"])
            ds.append(ld - lq)
            vs.append(vd + vq)
        if len(ds) >= 2:
            D.append(float(np.mean(ds)))
            V.append(float(np.mean(vs)) / len(ds))
    D, V = np.array(D), np.array(V)
    res["n_drift_queries"] = int(len(D))
    rng = np.random.default_rng(20260905)
    boots = []
    for _ in range(500):
        idx = rng.integers(0, len(D), size=len(D))
        d, v = D[idx], V[idx]
        vt = max(1e-6, float(np.var(d, ddof=1)) - float(np.mean(v)))
        Wb = normal_weights(float(np.mean(d)), math.sqrt(vt))
        val, _, _, _ = full_batch(Wb, TGT)
        boots.append(val)
    boots = np.array(boots)
    res["weight_bootstrap_randomized"] = {
        "n": int(len(boots)), "target": TGT,
        "mean": round(float(boots.mean()), 5),
        "q": {q: round(float(np.quantile(boots, float(q))), 5)
              for q in ("0.025", "0.5", "0.975")},
        "frac_positive": round(float((boots > 0).mean()), 4)}
    print("weight bootstrap:", res["weight_bootstrap_randomized"], flush=True)

    # ---- (e) selection-rule overlap ---------------------------------------
    fwd = json.load(open(RW))["fwd"]
    atom = {r_["i"] for r_ in fwd if r_["sel_mass"] > 0.5}
    rule = {r_["i"] for r_ in fwd if r_["info"] >= 8}
    res["selection_overlap"] = {
        "n_atom": len(atom), "n_rule": len(rule),
        "n_overlap": len(atom & rule),
        "atom_share_of_fwd": round(len(atom) / len(fwd), 4)}
    print("overlap:", res["selection_overlap"])

    OUT.write_text(json.dumps(res, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
