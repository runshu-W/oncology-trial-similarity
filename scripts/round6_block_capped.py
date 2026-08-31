#!/usr/bin/env python3
"""Round-6: strongest-donor block-bootstrap intervals for the CAPPED
upper-bound comparisons (review round-6 Major 2).

The round-4/5 text reported block-bootstrap intervals for the uncapped and
architectural comparisons but not for the capped forward / title-screened
deltas that are now called the strongest real-data evidence.  This script
computes them: per-example capped-selective minus EB differences with blocks
defined by each query's strongest gated donor (same convention as the
released dependence analysis), for

  fwd      canonical fold models, full-information donors (n=775)
  strictA  retrained title-screened fold models (n=589)
  strictC  frozen canonical models, title-screened filter (n=589)

Output: artifacts_round4/stress/round6_block_capped.json
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
from capped_real_data import bb, filt, read_jsonl, capped_nll, load_strict_model  # noqa: E402

EX = ROOT / "artifacts_round2" / "examples_unitfix_ids_ep.jsonl"
META = ROOT / "artifacts_rerun" / "examples_unitfix_with_true_dates.jsonl"
RW = ROOT / "artifacts_pathfinding" / "rewrite_numbers_perexample.json"
SEL_DIR = ROOT / "artifacts_pathfinding" / "selective"
STRICT = ROOT / "artifacts_round3" / "strict_orr"
OUT = ROOT / "artifacts_round4" / "stress" / "round6_block_capped.json"
CUTOFFS = ("2020-12-31", "2021-12-31", "2022-12-31")
CAP = 0.5
B = 4000


def block_boot(diffs, blocks, rng):
    d = np.asarray(diffs, dtype=float)
    uniq = {}
    for i, g in enumerate(blocks):
        uniq.setdefault(g, []).append(i)
    groups = [np.array(v) for v in uniq.values()]
    m = np.empty(B)
    for b in range(B):
        pick = rng.integers(0, len(groups), size=len(groups))
        sel = np.concatenate([groups[j] for j in pick])
        m[b] = d[sel].mean()
    lo, hi = np.quantile(m, [0.025, 0.975])
    return {"mean": float(d.mean()), "ci95_block": [float(lo), float(hi)],
            "n": int(len(d)), "n_blocks": len(groups),
            "max_block": int(max(len(g) for g in groups))}


def main():
    rng = np.random.default_rng(20260906)
    ex = read_jsonl(EX)
    meta = read_jsonl(META)
    rw = json.load(open(RW))
    fwd_ref = {r["i"]: r for r in rw["fwd"]}
    sortd = [(m["query_metadata"].get("temporal_sort_date") or "") for m in meta]
    q_strict = [is_strict_orr(e["query"].get("endpoint")) for e in ex]
    c_strict = [[is_strict_orr(c.get("endpoint")) for c in e["components"]] for e in ex]
    win = {c: [i for i, d in enumerate(sortd) if lo < d <= hi]
           for c, (lo, hi) in (("2020-12-31", ("2020-12-31", "2021-12-31")),
                               ("2021-12-31", ("2021-12-31", "2022-12-31")),
                               ("2022-12-31", ("2022-12-31", "9999")))}
    folds = {c: TS.load_selective_artifact(SEL_DIR / f"selective_model_fold_{c}.pt")
             for c in CUTOFFS}
    ssum = json.load(open(STRICT / "summary.json"))
    mA = {c: load_strict_model(STRICT / "models" / f"strict_{c}.pt",
                               ssum["A_strict_primary"][c]["anchor"]) for c in CUTOFFS}

    def strongest(i, keep):
        kept = [c for c, k in zip(ex[i]["components"], keep) if k]
        lrs = [v for v, k in zip(ex[i]["lambda_rule"], keep) if k]
        return (kept[int(np.argmax(lrs))]["nct_id"]
                if kept and max(lrs, default=0) > 0 else f"none_{i}")

    res = {}
    # fwd (full keep, canonical folds, EB from stored fwd rows)
    diffs, blocks = [], []
    for c in CUTOFFS:
        model, anchor = folds[c]
        for i in win[c]:
            keep = [True] * len(ex[i]["components"])
            fe = filt(ex[i], keep)
            nll, _, _ = capped_nll(model, fe, anchor, CAP)
            diffs.append(nll - fwd_ref[i]["eb"])
            blocks.append(strongest(i, keep))
    res["fwd_capped_vs_eb"] = block_boot(diffs, blocks, rng)

    # strictA (title-screened keep, retrained models, fold-anchor EB)
    diffs, blocks = [], []
    for c in CUTOFFS:
        model, anchor = mA[c]
        for i in win[c]:
            if not q_strict[i]:
                continue
            keep = c_strict[i]
            fe = filt(ex[i], keep)
            nll, _, _ = capped_nll(model, fe, anchor, CAP)
            ebv = -bb(ex[i]["query"]["count"], ex[i]["query"]["denominator"],
                      anchor[0], anchor[1])
            diffs.append(nll - ebv)
            blocks.append(strongest(i, keep))
    res["strictA_capped_vs_eb"] = block_boot(diffs, blocks, rng)

    # strictC (title-screened keep, frozen canonical folds, stored fwd EB)
    diffs, blocks = [], []
    for c in CUTOFFS:
        model, anchor = folds[c]
        for i in win[c]:
            if not q_strict[i]:
                continue
            keep = c_strict[i]
            fe = filt(ex[i], keep)
            nll, _, _ = capped_nll(model, fe, anchor, CAP)
            diffs.append(nll - fwd_ref[i]["eb"])
            blocks.append(strongest(i, keep))
    res["strictC_capped_vs_eb"] = block_boot(diffs, blocks, rng)

    OUT.write_text(json.dumps(res, indent=1))
    for k, v in res.items():
        print(k, f"{v['mean']:+.4f} block[{v['ci95_block'][0]:+.4f},"
                 f"{v['ci95_block'][1]:+.4f}] blocks={v['n_blocks']} "
                 f"max={v['max_block']}")


if __name__ == "__main__":
    main()
