#!/usr/bin/env python3
"""Round-4: is the primary-analysis harm a training-pool artefact? (Major 4)

Round 3 attributed the availability-consistent model's subgroup harm to
"training-pool starvation" on the evidence of one frozen-model control.  This
script adds the missing mechanism experiments.  All models are scored on the
IDENTICAL primary scoring rows (title-screened windows, availability-filtered
donors), against the same fold EB anchors as the primary analysis:

  1. seeds:   the availability-trained fold models retrained under 4 extra
              seeds - is the harm seed-stable?
  2. equal_n: models trained on random subsamples of the FULL title-screened
              pool downsampled to the availability pool's size (3 seeds) -
              same n, unrestricted composition.  Harm here => size is the
              driver; neutrality here => availability composition matters.
  3. curve:   models trained on {25%, 50%, 100%} of the full title-screened
              pool (seed 0) - the size dose-response.

Output: artifacts_round4/starvation/summary.json
"""
from __future__ import annotations

import csv
import json
import math
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "pipeline"))
sys.path.insert(0, str(ROOT / "scripts"))

import two_head_selective as TS  # noqa: E402
from endpoint_canonicalizer import is_strict_orr  # noqa: E402
from capped_real_data import bb, filt, read_jsonl, capped_nll  # noqa: E402

EX = ROOT / "artifacts_round2" / "examples_unitfix_ids_ep.jsonl"
META = ROOT / "artifacts_rerun" / "examples_unitfix_with_true_dates.jsonl"
DATES = ROOT / "artifacts_rerun" / "clinicaltrials_date_rows.csv"
OUT = ROOT / "artifacts_round4" / "starvation"
CUTOFFS = ("2020-12-31", "2021-12-31", "2022-12-31")
SEED = 20260603
BOOT = 2000


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    ex = read_jsonl(EX)
    meta = read_jsonl(META)
    dates = {r["nct_id"]: r for r in csv.DictReader(open(DATES))}
    posted = {k: (v.get("results_first_posted_date") or "") for k, v in dates.items()}
    sortd = [(m["query_metadata"].get("temporal_sort_date") or "") for m in meta]
    startd = [(m["query_metadata"].get("start_date") or "") for m in meta]
    q_strict = [is_strict_orr(e["query"].get("endpoint")) for e in ex]
    c_strict = [[is_strict_orr(c.get("endpoint")) for c in e["components"]] for e in ex]
    win = {w: [i for i, d in enumerate(sortd) if lo < d <= hi and q_strict[i]]
           for w, (lo, hi) in (("w1", ("2020-12-31", "2021-12-31")),
                               ("w2", ("2021-12-31", "2022-12-31")),
                               ("w3", ("2022-12-31", "9999")))}

    def avail_keep(i):
        qs = startd[i]
        return [cs and qs != "" and posted.get(cc["nct_id"], "") != ""
                and posted[cc["nct_id"]] < qs
                for cc, cs in zip(ex[i]["components"], c_strict[i])]

    pools_full, pools_av = {}, {}
    for c in CUTOFFS:
        pools_full[c] = [i for i, d in enumerate(sortd) if d and d <= c and q_strict[i]]
        pools_av[c] = [i for i in pools_full[c]
                       if posted.get(ex[i]["query_nct_id"], "")
                       and posted[ex[i]["query_nct_id"]] <= c]

    def build_train(pool, availability):
        out = []
        for i in pool:
            if availability:
                cutoff = availability
                keep = [cs and posted.get(cc["nct_id"], "") != ""
                        and posted[cc["nct_id"]] <= cutoff
                        for cc, cs in zip(ex[i]["components"], c_strict[i])]
            else:
                keep = c_strict[i]
            fe = filt(ex[i], keep)
            if any(cc["gate"] > 0 for cc in fe["components"]):
                out.append(fe)
        return out

    def train_score(pool, availability, seed, w, c):
        train = build_train(pool, availability)
        anchor = TS.fit_eb_anchor([(ex[i]["query"]["count"],
                                    ex[i]["query"]["denominator"]) for i in pool])
        res = TS.train_selective(train, list(range(len(train))), [], anchor,
                                 epochs=100, learning_rate=0.01, hidden_dim=16,
                                 seed=seed)
        model = res["model"]
        rows = []
        for i in win[w]:
            fe = filt(ex[i], avail_keep(i))
            nll, lam0, mass = capped_nll(model, fe, anchor, None)
            ebv = -bb(ex[i]["query"]["count"], ex[i]["query"]["denominator"],
                      anchor[0], anchor[1])
            rows.append({"i": i, "d": nll - ebv,
                         "has": any(cc["gate"] > 0 for cc in fe["components"])})
        return rows, len(train), anchor

    def agg(allrows):
        d = np.array([r["d"] for r in allrows])
        sub = np.array([r["d"] for r in allrows if r["has"]])
        rng = np.random.default_rng(1)
        m = d[rng.integers(0, len(d), size=(BOOT, len(d)))].mean(axis=1)
        return {"pooled_mean": float(d.mean()),
                "pooled_ci": [float(np.quantile(m, q)) for q in (0.025, 0.975)],
                "subgroup_mean": float(sub.mean()) if len(sub) else None,
                "n": len(d), "n_sub": int(len(sub))}

    summary = {"pool_sizes": {c: {"full": len(pools_full[c]), "avail": len(pools_av[c])}
                              for c in CUTOFFS},
               "seeds": {}, "equal_n": {}, "curve": {}}
    t0 = time.time()

    # 1. seed stability of the availability-trained primary model
    for k, seed in enumerate((SEED + 1, SEED + 2, SEED + 3, SEED + 4)):
        rows = []
        for w, c in zip(("w1", "w2", "w3"), CUTOFFS):
            r, ntr, _ = train_score(pools_av[c], c, seed, w, c)
            rows += r
        summary["seeds"][f"seed{k+1}"] = agg(rows)
        print(f"[{time.time()-t0:6.0f}s] seeds {k+1}: {summary['seeds'][f'seed{k+1}']}",
              flush=True)
        json.dump(summary, open(OUT / "summary.json", "w"), indent=1)

    # 2. equal-n composition control
    for k in range(3):
        rng = np.random.default_rng(500 + k)
        rows = []
        for w, c in zip(("w1", "w2", "w3"), CUTOFFS):
            sub = list(rng.choice(pools_full[c], size=len(pools_av[c]), replace=False))
            r, ntr, _ = train_score(sub, None, SEED + k, w, c)
            rows += r
        summary["equal_n"][f"draw{k}"] = agg(rows)
        print(f"[{time.time()-t0:6.0f}s] equal_n {k}: {summary['equal_n'][f'draw{k}']}",
              flush=True)
        json.dump(summary, open(OUT / "summary.json", "w"), indent=1)

    # 3. size dose-response on the full pool
    for frac in (0.25, 0.5, 1.0):
        rng = np.random.default_rng(900)
        rows = []
        ns = []
        for w, c in zip(("w1", "w2", "w3"), CUTOFFS):
            pool = pools_full[c] if frac >= 1.0 else \
                list(rng.choice(pools_full[c], size=int(len(pools_full[c]) * frac),
                                replace=False))
            r, ntr, _ = train_score(pool, None, SEED, w, c)
            rows += r
            ns.append(ntr)
        summary["curve"][f"frac{frac}"] = {**agg(rows), "trainable": ns}
        print(f"[{time.time()-t0:6.0f}s] curve {frac}: {summary['curve'][f'frac{frac}']}",
              flush=True)
        json.dump(summary, open(OUT / "summary.json", "w"), indent=1)

    print("done", flush=True)


if __name__ == "__main__":
    main()
