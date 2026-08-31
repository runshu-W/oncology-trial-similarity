#!/usr/bin/env python3
"""Round-5: five additional equal-n downsampling draws (review round-5
minor 4) - same construction as starvation_controls.py equal_n block,
draws 3..7, with per-draw bootstrap CIs, appended to a separate JSON.

Output: artifacts_round4/starvation/equal_n_extra.json
"""
from __future__ import annotations

import csv
import json
import time
from pathlib import Path
import sys

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
OUT = ROOT / "artifacts_round4" / "starvation" / "equal_n_extra.json"
CUTOFFS = ("2020-12-31", "2021-12-31", "2022-12-31")
SEED = 20260603


def main():
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

    results = json.loads(OUT.read_text()) if OUT.exists() else {}
    t0 = time.time()
    for k in range(3, 8):
        key = f"draw{k}"
        if key in results:
            continue
        rng = np.random.default_rng(500 + k)
        rows = []
        for w, c in zip(("w1", "w2", "w3"), CUTOFFS):
            sub = list(rng.choice(pools_full[c], size=len(pools_av[c]), replace=False))
            train = []
            for i in sub:
                fe = filt(ex[i], c_strict[i])
                if any(cc["gate"] > 0 for cc in fe["components"]):
                    train.append(fe)
            anchor = TS.fit_eb_anchor([(ex[i]["query"]["count"],
                                        ex[i]["query"]["denominator"]) for i in sub])
            res = TS.train_selective(train, list(range(len(train))), [], anchor,
                                     epochs=100, learning_rate=0.01, hidden_dim=16,
                                     seed=SEED + k)
            model = res["model"]
            for i in win[w]:
                fe = filt(ex[i], avail_keep(i))
                nll, _, _ = capped_nll(model, fe, anchor, None)
                ebv = -bb(ex[i]["query"]["count"], ex[i]["query"]["denominator"],
                          anchor[0], anchor[1])
                rows.append({"d": nll - ebv,
                             "has": any(cc["gate"] > 0 for cc in fe["components"])})
        d = np.array([r["d"] for r in rows])
        sub_ = np.array([r["d"] for r in rows if r["has"]])
        rg = np.random.default_rng(1)
        m = d[rg.integers(0, len(d), size=(2000, len(d)))].mean(axis=1)
        ms = sub_[rg.integers(0, len(sub_), size=(2000, len(sub_)))].mean(axis=1)
        results[key] = {
            "pooled_mean": float(d.mean()),
            "pooled_ci": [float(np.quantile(m, q)) for q in (0.025, 0.975)],
            "subgroup_mean": float(sub_.mean()),
            "subgroup_ci": [float(np.quantile(ms, q)) for q in (0.025, 0.975)],
            "n": int(len(d)), "n_sub": int(len(sub_))}
        OUT.write_text(json.dumps(results, indent=1))
        print(f"[{time.time()-t0:6.0f}s] {key}: {results[key]}", flush=True)
    print("done", flush=True)


if __name__ == "__main__":
    main()
