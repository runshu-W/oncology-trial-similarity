#!/usr/bin/env python3
"""Round-8: DIRECT paired capped-versus-uncapped comparisons (review round-8
item 1).

Previous rounds compared each design with the EB reference and contrasted
those two intervals informally.  The reviewer asks for the within-query
paired contrast

    Delta_i = NLL_capped(0.5),i - NLL_uncapped,i

with ordinary and strongest-donor block bootstrap intervals, for

  primaryB   pooled (n=589) and engaged subgroup (n=119); computed offline
             from the stored per-example rows of the released primary
             scoring run (artifacts_round4/real_capped/rows_strictB.json)
  fwd        canonical fold models, full-information donors (n=775)
  strictA    retrained title-screened fold models (n=589)
  strictC    frozen canonical fold models, title-screened filter (n=589)

The cap projection changes a query's prior only when its uncapped borrowed
mass exceeds 0.5, so Delta_i = 0 exactly for every other query; those rows
are retained in the pooled analyses (the contrast is defined for every
decision-time row) and the number of binding rows is reported.  Blocks are
the released convention: strongest gated donor by rule weight; queries with
no gated donor are singleton blocks.

Output: artifacts_round4/stress/round8_direct_paired.json
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "pipeline"))
sys.path.insert(0, str(ROOT / "scripts"))

import two_head_selective as TS  # noqa: E402
from endpoint_canonicalizer import is_strict_orr  # noqa: E402
from capped_real_data import filt, read_jsonl, capped_nll, load_strict_model  # noqa: E402

EX = ROOT / "artifacts_round2" / "examples_unitfix_ids_ep.jsonl"
META = ROOT / "artifacts_rerun" / "examples_unitfix_with_true_dates.jsonl"
RW = ROOT / "artifacts_pathfinding" / "rewrite_numbers_perexample.json"
SEL_DIR = ROOT / "artifacts_pathfinding" / "selective"
STRICT = ROOT / "artifacts_round3" / "strict_orr"
ROWS_B = ROOT / "artifacts_round4" / "real_capped" / "rows_strictB.json"
OUT = ROOT / "artifacts_round4" / "stress" / "round8_direct_paired.json"
CUTOFFS = ("2020-12-31", "2021-12-31", "2022-12-31")
CAP = 0.5
B = 4000
RNG = np.random.default_rng(20260910)


def both_boots(diffs, blocks):
    d = np.asarray(diffs, dtype=float)
    idx = RNG.integers(0, len(d), size=(B, len(d)))
    mo = d[idx].mean(axis=1)
    uniq = {}
    for i, g in enumerate(blocks):
        uniq.setdefault(g, []).append(i)
    groups = [np.array(v) for v in uniq.values()]
    mb = np.empty(B)
    for b in range(B):
        pick = RNG.integers(0, len(groups), size=len(groups))
        sel = np.concatenate([groups[j] for j in pick])
        mb[b] = d[sel].mean()
    return {"mean": float(d.mean()),
            "ci95_ordinary": [float(x) for x in np.quantile(mo, [0.025, 0.975])],
            "ci95_block": [float(x) for x in np.quantile(mb, [0.025, 0.975])],
            "n": int(len(d)), "n_blocks": len(groups),
            "max_block": int(max(len(g) for g in groups)),
            "n_binding": int((np.abs(d) > 1e-12).sum())}


def main():
    res = {}

    # ---- primary tier B: offline from stored per-example rows -------------
    rows = json.load(open(ROWS_B))
    ids = sorted(rows, key=int)
    diffs = [rows[i]["sel_cap05"] - rows[i]["sel"] for i in ids]
    blocks = [rows[i]["strongest"] for i in ids]
    res["primaryB_pooled"] = both_boots(diffs, blocks)
    eng = [i for i in ids if rows[i]["has_donor"]]
    res["primaryB_engaged"] = both_boots(
        [rows[i]["sel_cap05"] - rows[i]["sel"] for i in eng],
        [rows[i]["strongest"] for i in eng])
    # context + cross-checks against the released summary
    res["primaryB_context"] = {
        "mean_capped_minus_eb": float(np.mean(
            [rows[i]["sel_cap05"] - rows[i]["eb"] for i in ids])),
        "mean_uncapped_minus_eb": float(np.mean(
            [rows[i]["sel"] - rows[i]["eb"] for i in ids])),
        "n_engaged": len(eng)}
    print("primaryB pooled", {k: res["primaryB_pooled"][k]
                              for k in ("mean", "ci95_ordinary", "ci95_block",
                                        "n", "n_binding")}, flush=True)
    print("primaryB engaged", {k: res["primaryB_engaged"][k]
                               for k in ("mean", "ci95_ordinary", "ci95_block",
                                         "n", "n_binding")}, flush=True)

    # ---- scored tiers: fwd / strictA / strictC ----------------------------
    ex = read_jsonl(EX)
    meta = read_jsonl(META)
    rw = json.load(open(RW))
    fwd_ref = {r["i"]: r for r in rw["fwd"]}
    sortd = [(m["query_metadata"].get("temporal_sort_date") or "") for m in meta]
    q_strict = [is_strict_orr(e["query"].get("endpoint")) for e in ex]
    c_strict = [[is_strict_orr(c.get("endpoint")) for c in e["components"]]
                for e in ex]
    win = {c: [i for i, d in enumerate(sortd) if lo < d <= hi]
           for c, (lo, hi) in (("2020-12-31", ("2020-12-31", "2021-12-31")),
                               ("2021-12-31", ("2021-12-31", "2022-12-31")),
                               ("2022-12-31", ("2022-12-31", "9999")))}
    folds = {c: TS.load_selective_artifact(SEL_DIR / f"selective_model_fold_{c}.pt")
             for c in CUTOFFS}
    ssum = json.load(open(STRICT / "summary.json"))
    mA = {c: load_strict_model(STRICT / "models" / f"strict_{c}.pt",
                               ssum["A_strict_primary"][c]["anchor"])
          for c in CUTOFFS}

    def strongest(i, keep):
        kept = [c for c, k in zip(ex[i]["components"], keep) if k]
        lrs = [v for v, k in zip(ex[i]["lambda_rule"], keep) if k]
        return (kept[int(np.argmax(lrs))]["nct_id"]
                if kept and max(lrs, default=0) > 0 else f"none_{i}")

    def run_tier(name, model_for, strict_only, keep_for):
        diffs, blocks, ncap, nunc = [], [], [], []
        for c in CUTOFFS:
            model, anchor = model_for(c)
            for i in win[c]:
                if strict_only and not q_strict[i]:
                    continue
                keep = keep_for(i)
                fe = filt(ex[i], keep)
                nc, _, _ = capped_nll(model, fe, anchor, CAP)
                nu, _, _ = capped_nll(model, fe, anchor, None)
                diffs.append(nc - nu)
                ncap.append(nc)
                nunc.append(nu)
                blocks.append(strongest(i, keep))
        r = both_boots(diffs, blocks)
        r["mean_capped_nll"] = float(np.mean(ncap))
        r["mean_uncapped_nll"] = float(np.mean(nunc))
        res[name] = r
        print(name, {k: r[k] for k in ("mean", "ci95_ordinary", "ci95_block",
                                       "n", "n_blocks", "n_binding")},
              f"NLL cap {r['mean_capped_nll']:.4f} unc {r['mean_uncapped_nll']:.4f}",
              flush=True)

    run_tier("fwd", lambda c: folds[c], False,
             lambda i: [True] * len(ex[i]["components"]))
    run_tier("strictA", lambda c: mA[c], True, lambda i: c_strict[i])
    run_tier("strictC", lambda c: folds[c], True, lambda i: c_strict[i])

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(res, indent=1))
    print("wrote", OUT)


if __name__ == "__main__":
    main()
