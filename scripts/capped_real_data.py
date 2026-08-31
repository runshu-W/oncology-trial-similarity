#!/usr/bin/env python3
"""Round-4: capped selective prior scored on EVERY real-data tier (Major 1),
plus the dedicated primary-analysis table numbers and its dependence-adjusted
interval (Major 4).

Tiers:
  test      held-out 20% split, primary model (hypothetical upper bound)
  fwd       forward windows w1-w3, fold models (hypothetical upper bound)
  strictA   title-screened retrained fold models, strict windows
  strictB   PRIMARY: title-screened x availability (train + score), incl.
            per-window table numbers, borrowing events, subgroup, and
            strongest-donor block bootstrap
  strictC   frozen canonical fold models on title-screened filter

Caps: none, 0.3, 0.5, 0.7 (projection lambda_i <- lambda_i * cap/mass when
mass > cap; excess returns to the anchor component).  SAM is NOT applied on
real data (it consumes the scored outcome; stated in the paper).

Sanity: uncapped NLLs must reproduce the stored per-example numbers.
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
RW = ROOT / "artifacts_pathfinding" / "rewrite_numbers_perexample.json"
STRICT = ROOT / "artifacts_round3" / "strict_orr"
OUT = ROOT / "artifacts_round4" / "real_capped"
CUTOFFS = ("2020-12-31", "2021-12-31", "2022-12-31")
CAPS = (None, 0.3, 0.5, 0.7)
B = 4000
RNG = np.random.default_rng(20260901)


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


def capped_nll(model, fe, anchor, cap):
    """Selective NLL with the borrowed-mass cap projection; also mass/lam0."""
    y0, n0 = fe["query"]["count"], fe["query"]["denominator"]
    if not any(c["gate"] > 0 for c in fe["components"]):
        return -bb(y0, n0, anchor[0], anchor[1]), 1.0, 0.0
    with torch.no_grad():
        t = TS.LT._validated_example_tensors(fe)
        lam0, lam_i = TS.selective_lambda_weights(model, t["features"], t["gate"])
        a, b_, _ = TS.LT._component_alpha_beta_for_model(model, t["features"], t)
        mass = float(lam_i.sum().item())
        if cap is not None and mass > cap:
            lam_i = lam_i * (cap / mass)
            lam0 = 1.0 - lam_i.sum()
        cnt = torch.tensor(float(y0), dtype=torch.float32)
        den = torch.tensor(float(n0), dtype=torch.float32)
        weak = TS.LT.beta_binomial_log_predictive(
            cnt, den, torch.tensor(float(anchor[0])), torch.tensor(float(anchor[1])))
        comp = TS.LT.beta_binomial_log_predictive(cnt, den, a, b_)
        terms = torch.cat([(lam0.clamp_min(1e-12).log() + weak).reshape(1),
                           lam_i.clamp_min(1e-12).log() + comp])
        nll = float(-torch.logsumexp(terms, dim=0).item())
    return nll, float(lam0), mass


def boot(diffs, blocks=None):
    d = np.asarray(diffs, dtype=float)
    if blocks is None:
        idx = RNG.integers(0, len(d), size=(B, len(d)))
        m = d[idx].mean(axis=1)
    else:
        uniq = {}
        for i, g in enumerate(blocks):
            uniq.setdefault(g, []).append(i)
        groups = [np.array(v) for v in uniq.values()]
        m = np.empty(B)
        for b in range(B):
            pick = RNG.integers(0, len(groups), size=len(groups))
            sel = np.concatenate([groups[j] for j in pick])
            m[b] = d[sel].mean()
    lo, hi = np.quantile(m, [0.025, 0.975])
    return {"mean": float(d.mean()), "ci95": [float(lo), float(hi)],
            "n": int(len(d)), "win": float((d < 0).mean())}


def load_strict_model(path, anchor):
    sd = torch.load(path, map_location="cpu", weights_only=True)
    m = TS.SelectiveTwoHeadDeepSetsLambdaScorer(
        input_dim=int(sd["phi.0.weight"].shape[1]), hidden_dim=16)
    m.load_state_dict(sd)
    m.eval()
    return m, tuple(anchor)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    ex = read_jsonl(EX)
    meta = read_jsonl(META)
    rw = json.load(open(RW))
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
    strict_sum = json.load(open(STRICT / "summary.json"))
    summary = {"caps": [c if c else "none" for c in CAPS], "tiers": {}}
    t0 = time.time()

    def full_keep(i):
        return [True] * len(ex[i]["components"])

    def strict_keep(i):
        return c_strict[i]

    def avail_keep(i):
        qs = startd[i]
        return [cs and qs != "" and posted.get(cc["nct_id"], "") != ""
                and posted[cc["nct_id"]] < qs
                for cc, cs in zip(ex[i]["components"], c_strict[i])]

    def score_tier(name, model_for, idx_iter, keepf, eb_ref=None):
        rows = {c if c else "none": {} for c in CAPS}
        meta_rows = {}
        for w, i in idx_iter:
            model, anchor = model_for(w)
            fe = filt(ex[i], keepf(i))
            ebv = (eb_ref[i] if eb_ref else
                   -bb(ex[i]["query"]["count"], ex[i]["query"]["denominator"],
                       anchor[0], anchor[1]))
            kept = [c for c, k in zip(ex[i]["components"], keepf(i)) if k]
            lrs = [v for v, k in zip(ex[i]["lambda_rule"], keepf(i)) if k]
            strongest = (kept[int(np.argmax(lrs))]["nct_id"]
                         if kept and max(lrs, default=0) > 0 else f"none_{i}")
            for c in CAPS:
                nll, lam0, mass = capped_nll(model, fe, anchor, c)
                rows[c if c else "none"][i] = {"sel": nll, "eb": ebv}
                if c is None:
                    meta_rows[i] = {"lam0": lam0, "mass": mass,
                                    "has_donor": any(cc["gate"] > 0 for cc in fe["components"]),
                                    "strongest": strongest, "window": w}
        return rows, meta_rows

    # ---------------- tier: test (primary model) ---------------------------
    pm, panchor = TS.load_selective_artifact(SEL_DIR / "selective_model_primary.pt")
    test_ref = {r["i"]: r for r in rw["test"]}
    rows, mrows = score_tier("test", lambda w: (pm, panchor),
                             [("test", i) for i in sorted(test_ref)], full_keep)
    chk = max(abs(rows["none"][i]["sel"] - test_ref[i]["selective"]) for i in test_ref)
    summary["tiers"]["test"] = {
        "uncapped_reproduction_maxdiff": chk,
        **{f"cap_{k}": boot([v[i]["sel"] - test_ref[i]["eb"] for i in v])
           for k, v in rows.items()}}
    print(f"[{time.time()-t0:5.0f}s] test done (repro {chk:.2e})", flush=True)

    # ---------------- tier: forward (fold models) --------------------------
    folds = {c: TS.load_selective_artifact(SEL_DIR / f"selective_model_fold_{c}.pt")
             for c in CUTOFFS}
    fwd_ref = {r["i"]: r for r in rw["fwd"]}
    wof = {i: w for w in ("w1", "w2", "w3") for i in win[w]}
    it = [(dict(zip(("w1", "w2", "w3"), CUTOFFS))[wof[i]], i) for i in sorted(fwd_ref)]
    rows, mrows = score_tier("fwd", lambda c: folds[c], it, full_keep)
    chk = max(abs(rows["none"][i]["sel"] - fwd_ref[i]["selective"]) for i in fwd_ref)
    summary["tiers"]["fwd"] = {
        "uncapped_reproduction_maxdiff": chk,
        **{f"cap_{k}": boot([v[i]["sel"] - fwd_ref[i]["eb"] for i in v])
           for k, v in rows.items()}}
    print(f"[{time.time()-t0:5.0f}s] fwd done (repro {chk:.2e})", flush=True)

    # ---------------- tier: strict A (retrained) ---------------------------
    A = strict_sum["A_strict_primary"]
    mA = {c: load_strict_model(STRICT / "models" / f"strict_{c}.pt", A[c]["anchor"])
          for c in CUTOFFS}
    it = [(c, i) for w, c in zip(("w1", "w2", "w3"), CUTOFFS) for i in win_s[w]]
    rows, _ = score_tier("strictA", lambda c: mA[c], it, strict_keep)
    summary["tiers"]["strictA"] = {
        f"cap_{k}": boot([r["sel"] - r["eb"] for r in v.values()])
        for k, v in rows.items()}
    print(f"[{time.time()-t0:5.0f}s] strictA done", flush=True)

    # ---------------- tier: strict B (PRIMARY) -----------------------------
    Bsum = strict_sum["B_strict_availability"]
    mB = {c: load_strict_model(STRICT / "models" / f"strict_avail_{c}.pt",
                               Bsum[c]["anchor"]) for c in CUTOFFS}
    it = [(c, i) for w, c in zip(("w1", "w2", "w3"), CUTOFFS) for i in win_s[w]]
    rows, mrows = score_tier("strictB", lambda c: mB[c], it, avail_keep)
    tierB = {}
    for k, v in rows.items():
        diffs = [v[i]["sel"] - v[i]["eb"] for i in sorted(v)]
        idxs = sorted(v)
        blocks = [mrows[i]["strongest"] for i in idxs]
        have = [i for i in idxs if mrows[i]["has_donor"]]
        tierB[f"cap_{k}"] = {
            "pooled": boot(diffs),
            "pooled_block": boot(diffs, blocks),
            "subgroup_with_donor": boot([v[i]["sel"] - v[i]["eb"] for i in have]),
            "subgroup_block": boot([v[i]["sel"] - v[i]["eb"] for i in have],
                                   [mrows[i]["strongest"] for i in have]),
        }
    # per-window primary table numbers (uncapped model diagnostics)
    tab = {}
    for w, c in zip(("w1", "w2", "w3"), CUTOFFS):
        ids = win_s[w]
        have = [i for i in ids if mrows[i]["has_donor"]]
        borrow = [i for i in have if mrows[i]["mass"] >= 0.01]
        tab[w] = {"n_queries": len(ids), "n_with_avail_donor": len(have),
                  "n_borrowing_ge1pct": len(borrow),
                  "mean_eb_nll": float(np.mean([rows["none"][i]["eb"] for i in ids])),
                  "mean_sel_nll_uncapped": float(np.mean([rows["none"][i]["sel"] for i in ids])),
                  "mean_mass_subgroup": float(np.mean([mrows[i]["mass"] for i in have])) if have else 0.0}
        for k, v in rows.items():
            tab[w][f"delta_{k}"] = boot([v[i]["sel"] - v[i]["eb"] for i in ids])
            tab[w][f"mean_sel_nll_{k}"] = float(np.mean([v[i]["sel"] for i in ids]))
    tierB["table_by_window"] = tab
    n_blocks = len({mrows[i]["strongest"] for i in sorted(rows["none"])})
    mx = max(np.bincount(np.unique([mrows[i]["strongest"] for i in sorted(rows["none"])],
                                   return_inverse=True)[1]))
    tierB["blocks"] = {"n_blocks": int(n_blocks), "max_block": int(mx)}
    summary["tiers"]["strictB"] = tierB
    json.dump({str(i): {**rows["none"][i], **mrows[i],
                        "sel_cap05": rows[0.5][i]["sel"],
                        "sel_cap03": rows[0.3][i]["sel"]}
               for i in rows["none"]},
              open(OUT / "rows_strictB.json", "w"))
    print(f"[{time.time()-t0:5.0f}s] strictB done", flush=True)

    # ---------------- tier: strict C (frozen canonical) --------------------
    it = [(c, i) for w, c in zip(("w1", "w2", "w3"), CUTOFFS) for i in win_s[w]]
    rows, _ = score_tier("strictC", lambda c: folds[c], it, strict_keep,
                         eb_ref={i: fwd_ref[i]["eb"] for i in fwd_ref})
    summary["tiers"]["strictC"] = {
        f"cap_{k}": boot([r["sel"] - r["eb"] for r in v.values()])
        for k, v in rows.items()}
    print(f"[{time.time()-t0:5.0f}s] strictC done", flush=True)

    json.dump(summary, open(OUT / "summary.json", "w"), indent=1)
    for tier, d in summary["tiers"].items():
        keys = [k for k in d if k.startswith("cap_")]
        line = " ".join(
            f"{k[4:]}:{(d[k]['pooled']['mean'] if 'pooled' in d[k] else d[k]['mean']):+.4f}"
            for k in keys)
        print(tier, line)


if __name__ == "__main__":
    main()
