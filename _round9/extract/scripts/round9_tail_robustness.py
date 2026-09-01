#!/usr/bin/env python3
"""Round-9: tail-robustness analysis of the direct paired
capped-versus-uncapped contrasts (review round-9 item 1).

The round-8 direct contrasts reported means with ordinary and block
bootstrap intervals.  Because the cap projection binds on a minority of
rows (primary: 50/589), the mean could in principle be driven by a few
extreme NLL improvements.  This script characterises the tails, for every
tier of Table `tab:directpaired`:

  - per-example rows saved for all four tiers (primary offline from the
    released primary artifact; fwd/strictA/strictC rescored, means
    cross-checked against `round8_direct_paired.json`);
  - binding-row distribution: n, mean, median, IQR, 5%/95% quantiles,
    min/max;
  - trimmed (10%) and winsorized (5%) means of the binding rows, plus the
    pooled mean winsorized at the 1st/99th percentiles of all rows;
  - leave-one-query-out and leave-one-donor-block-out ranges of the
    pooled mean;
  - per-window direct contrasts (mean, ordinary and block bootstrap CI,
    binding counts), and per-window capped-vs-EB deltas for the forward
    tier (for the updated forward-validation figure);
  - the explicit marginal (all rows) vs conditional (binding rows only)
    distinction.

Outputs: artifacts_round4/stress/round9_tail_robustness.json
         artifacts_round4/stress/round9_rows.json
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
R8 = ROOT / "artifacts_round4" / "stress" / "round8_direct_paired.json"
OUT = ROOT / "artifacts_round4" / "stress" / "round9_tail_robustness.json"
OUT_ROWS = ROOT / "artifacts_round4" / "stress" / "round9_rows.json"
CUTOFFS = ("2020-12-31", "2021-12-31", "2022-12-31")
WNAME = dict(zip(CUTOFFS, ("w1", "w2", "w3")))
CAP = 0.5
B = 4000
RNG = np.random.default_rng(20260912)


def boot_ci(d, blocks=None):
    d = np.asarray(d, dtype=float)
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
            m[b] = d[np.concatenate([groups[j] for j in pick])].mean()
    return [float(x) for x in np.quantile(m, [0.025, 0.975])]


def tier_stats(rows):
    """rows: list of dicts with d, block, window (and optionally eb/nll)."""
    d = np.array([r["d"] for r in rows])
    blocks = [r["block"] for r in rows]
    bind = d[np.abs(d) > 1e-12]
    res = {"n": int(len(d)), "mean": float(d.mean()),
           "n_binding": int(len(bind))}
    # conditional (binding-row) distribution
    q = np.quantile(bind, [0.05, 0.25, 0.5, 0.75, 0.95])
    k = max(1, int(round(0.10 * len(bind))))
    srt = np.sort(bind)
    res["binding"] = {
        "mean": float(bind.mean()), "median": float(q[2]),
        "q05": float(q[0]), "q25": float(q[1]), "q75": float(q[3]),
        "q95": float(q[4]), "min": float(bind.min()), "max": float(bind.max()),
        "n_negative": int((bind < 0).sum()), "n_positive": int((bind > 0).sum()),
        "trimmed10_mean": float(srt[k:-k].mean()),
        "winsor5_mean": float(np.clip(bind, np.quantile(bind, 0.05),
                                      np.quantile(bind, 0.95)).mean()),
    }
    # pooled winsorized at 1%/99% of all rows
    res["pooled_winsor1_mean"] = float(
        np.clip(d, np.quantile(d, 0.01), np.quantile(d, 0.99)).mean())
    # leave-one-query-out; removing the most negative d maximises the LOO
    # mean, i.e. is the least favourable single removal for the claim
    s, n = d.sum(), len(d)
    loo = (s - d) / (n - 1)
    jmax = int(np.argmax(loo))
    res["loo_query"] = {
        "min": float(loo.min()), "max": float(loo.max()),
        "n_positive": int((loo > 0).sum()),
        "all_negative": bool((loo < 0).all()),
        "most_influential_d": float(d[jmax]),
        "mean_without_most_influential": float(loo[jmax]),
    }
    # leave-one-block-out
    uniq = {}
    for i, g in enumerate(blocks):
        uniq.setdefault(g, []).append(i)
    lobo = []
    for g, idx in uniq.items():
        rest = np.delete(d, idx)
        lobo.append((float(rest.mean()), g, float(d[idx].sum())))
    lm = [x[0] for x in lobo]
    jb = int(np.argmax(lm))
    res["loo_block"] = {
        "n_blocks": len(lobo), "min": float(min(lm)), "max": float(max(lm)),
        "n_positive": int(sum(1 for x in lm if x > 0)),
        "all_negative": bool(all(x < 0 for x in lm)),
        "most_influential_block": lobo[jb][1],
        "mean_without_most_influential": float(lobo[jb][0]),
    }
    # per-window
    res["windows"] = {}
    for w in ("w1", "w2", "w3"):
        rw_ = [r for r in rows if r["window"] == w]
        dw = np.array([r["d"] for r in rw_])
        res["windows"][w] = {
            "n": int(len(dw)), "mean": float(dw.mean()),
            "n_binding": int((np.abs(dw) > 1e-12).sum()),
            "ci95": boot_ci(dw),
            "ci95_block": boot_ci(dw, [r["block"] for r in rw_]),
        }
    return res


def main():
    r8 = json.load(open(R8))
    all_rows = {}
    res = {}

    # ---- primary (offline) ------------------------------------------------
    rb = json.load(open(ROWS_B))
    wmap = {"2020-12-31": "w1", "2021-12-31": "w2", "2022-12-31": "w3"}
    rows = [{"i": int(i), "window": wmap[v["window"]],
             "d": v["sel_cap05"] - v["sel"], "block": v["strongest"],
             "has_donor": v["has_donor"]} for i, v in rb.items()]
    rows.sort(key=lambda r: r["i"])
    all_rows["primaryB"] = rows
    res["primaryB_pooled"] = tier_stats(rows)
    res["primaryB_engaged"] = tier_stats([r for r in rows if r["has_donor"]])
    assert abs(res["primaryB_pooled"]["mean"] - r8["primaryB_pooled"]["mean"]) < 1e-9
    assert abs(res["primaryB_engaged"]["mean"] - r8["primaryB_engaged"]["mean"]) < 1e-9

    # ---- rescore fwd / strictA / strictC, keeping rows --------------------
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

    def run_tier(name, model_for, strict_only, keep_for, eb_of):
        rows = []
        for c in CUTOFFS:
            model, anchor = model_for(c)
            for i in win[c]:
                if strict_only and not q_strict[i]:
                    continue
                keep = keep_for(i)
                fe = filt(ex[i], keep)
                nc, _, _ = capped_nll(model, fe, anchor, CAP)
                nu, _, _ = capped_nll(model, fe, anchor, None)
                rows.append({"i": i, "window": WNAME[c], "d": nc - nu,
                             "nll_cap": nc, "nll_unc": nu,
                             "eb": eb_of(i, anchor), "block": strongest(i, keep)})
        all_rows[name] = rows
        res[name] = tier_stats(rows)
        assert abs(res[name]["mean"] - r8[name]["mean"]) < 1e-6, name
        print(name, "rescored ok", flush=True)

    from capped_real_data import bb  # noqa: E402
    run_tier("fwd", lambda c: folds[c], False,
             lambda i: [True] * len(ex[i]["components"]),
             lambda i, a: fwd_ref[i]["eb"])
    run_tier("strictA", lambda c: mA[c], True, lambda i: c_strict[i],
             lambda i, a: -bb(ex[i]["query"]["count"],
                              ex[i]["query"]["denominator"], a[0], a[1]))
    run_tier("strictC", lambda c: folds[c], True, lambda i: c_strict[i],
             lambda i, a: fwd_ref[i]["eb"])

    # per-window capped-vs-EB deltas for the forward tier (figure F2 update)
    fw = {}
    for w in ("w1", "w2", "w3"):
        rw_ = [r for r in all_rows["fwd"] if r["window"] == w]
        dc = np.array([r["nll_cap"] - r["eb"] for r in rw_])
        du = np.array([r["nll_unc"] - r["eb"] for r in rw_])
        fw[w] = {"n": len(rw_),
                 "capped_delta": float(dc.mean()), "capped_ci": boot_ci(dc),
                 "uncapped_delta": float(du.mean()), "uncapped_ci": boot_ci(du),
                 "capped_nll": float(np.mean([r["nll_cap"] for r in rw_]))}
    res["fwd_windows_vs_eb"] = fw

    OUT.write_text(json.dumps(res, indent=1))
    OUT_ROWS.write_text(json.dumps(all_rows))
    for k in ("primaryB_pooled", "primaryB_engaged", "fwd", "strictA", "strictC"):
        b = res[k]["binding"]
        print(f"{k}: mean {res[k]['mean']:+.4f} | binding n={res[k]['n_binding']} "
              f"mean {b['mean']:+.3f} med {b['median']:+.3f} "
              f"IQR [{b['q25']:+.3f},{b['q75']:+.3f}] "
              f"q5/q95 [{b['q05']:+.3f},{b['q95']:+.3f}] "
              f"rng [{b['min']:+.3f},{b['max']:+.3f}] "
              f"trim10 {b['trimmed10_mean']:+.4f} wins5 {b['winsor5_mean']:+.4f} | "
              f"LOOq [{res[k]['loo_query']['min']:+.4f},{res[k]['loo_query']['max']:+.4f}] "
              f"LOOb [{res[k]['loo_block']['min']:+.4f},{res[k]['loo_block']['max']:+.4f}]",
              flush=True)
        print("   windows:", {w: (round(v['mean'], 4), v['n_binding'], [round(x, 4) for x in v['ci95_block']])
                              for w, v in res[k]["windows"].items()}, flush=True)
    print("fwd windows vs EB:", {w: (round(v['capped_delta'], 4),
                                     [round(x, 4) for x in v['capped_ci']])
                                 for w, v in fw.items()})


if __name__ == "__main__":
    main()
