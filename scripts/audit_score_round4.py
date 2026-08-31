#!/usr/bin/env python3
"""Score the returned round-4 out-of-sample endpoint audit.

Joins the completed CSV with the withheld-prediction key, reports:
  - confusion (rows with human yes/no; 'ambiguous' set aside),
  - sensitivity / specificity / accuracy with exact Clopper-Pearson 95% CIs,
  - per-role breakdown,
  - corpus-weighted accuracy: strata are (role x frozen prediction); each
    stratum's error rate is weighted by its share of the corpus pool
    (pool sizes stored in the key at sampling time).

Usage: python3 audit_score_round4.py <completed_csv>
Output: artifacts_round4/audit/oos_audit_results.json
"""
from __future__ import annotations

import csv
import json
import sys
from math import lgamma, log, exp
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
KEY = ROOT / "artifacts_round4" / "audit" / "endpoint_audit_key_round4.json"
OUT = ROOT / "artifacts_round4" / "audit" / "oos_audit_results.json"


def cp_ci(k: int, n: int, conf: float = 0.95):
    if n == 0:
        return (0.0, 1.0)

    def cdf_le(kk, n, p):
        if p <= 0:
            return 1.0 if kk >= 0 else 0.0
        if p >= 1:
            return 1.0 if kk >= n else 0.0
        lp, lq = log(p), log(1 - p)
        tot = 0.0
        for i in range(kk + 1):
            c = lgamma(n + 1) - lgamma(i + 1) - lgamma(n - i + 1)
            tot += exp(c + i * lp + (n - i) * lq)
        return tot

    lo, hi = 0.0, 1.0
    if k > 0:
        a, b = 0.0, k / n
        for _ in range(60):
            m = 0.5 * (a + b)
            if 1 - cdf_le(k - 1, n, m) < (1 - conf) / 2:
                a = m
            else:
                b = m
        lo = 0.5 * (a + b)
    if k < n:
        a, b = k / n, 1.0
        for _ in range(60):
            m = 0.5 * (a + b)
            if cdf_le(k, n, m) < (1 - conf) / 2:
                b = m
            else:
                a = m
        hi = 0.5 * (a + b)
    return (lo, hi)


def main() -> int:
    csv_path = Path(sys.argv[1])
    key = json.loads(KEY.read_text())
    preds = key["predictions"]
    rows = list(csv.DictReader(open(csv_path)))
    joined, ambiguous = [], []
    for r in rows:
        aid = r["audit_id"].strip()
        k = preds[aid]
        assert k["nct_id"] == r["nct_id"].strip(), aid
        h = (r.get("human_strict_orr") or "").strip().lower()
        rec = {"aid": aid, "role": k["role"], "pred": bool(k["pred_strict"]),
               "human_raw": h,
               "actual_type": (r.get("human_actual_endpoint_type") or "").strip(),
               "notes": (r.get("auditor_notes") or "").strip()}
        if h in ("yes", "y", "true", "是"):
            rec["human"] = True
            joined.append(rec)
        elif h in ("no", "n", "false", "否"):
            rec["human"] = False
            joined.append(rec)
        else:
            ambiguous.append(rec)

    def confusion(sub):
        tp = sum(1 for r in sub if r["pred"] and r["human"])
        tn = sum(1 for r in sub if not r["pred"] and not r["human"])
        fp = sum(1 for r in sub if r["pred"] and not r["human"])
        fn = sum(1 for r in sub if not r["pred"] and r["human"])
        n = len(sub)
        out = {"n": n, "tp": tp, "tn": tn, "fp": fp, "fn": fn}
        if n:
            out["accuracy"] = (tp + tn) / n
            out["accuracy_ci"] = cp_ci(tp + tn, n)
        if tp + fn:
            out["sensitivity"] = tp / (tp + fn)
            out["sensitivity_ci"] = cp_ci(tp, tp + fn)
        if tn + fp:
            out["specificity"] = tn / (tn + fp)
            out["specificity_ci"] = cp_ci(tn, tn + fp)
        # PPV against human truth: P(truly strict | predicted strict)
        if tp + fp:
            out["ppv"] = tp / (tp + fp)
            out["ppv_ci"] = cp_ci(tp, tp + fp)
        if tn + fn:
            out["npv"] = tn / (tn + fn)
            out["npv_ci"] = cp_ci(tn, tn + fn)
        return out

    res = {"n_returned": len(rows), "n_unambiguous": len(joined),
           "n_ambiguous": len(ambiguous),
           "ambiguous_rows": [{k: r[k] for k in ("aid", "role", "pred",
                                                 "actual_type", "notes")}
                              for r in ambiguous],
           "overall": confusion(joined),
           "by_role": {role: confusion([r for r in joined if r["role"] == role])
                       for role in ("query", "component")},
           "false_calls": [{k: r[k] for k in ("aid", "role", "pred",
                                              "actual_type", "notes")}
                           for r in joined if r["pred"] != r["human"]]}

    # corpus-weighted accuracy over (role x prediction) strata
    pool = key["pool_sizes"]
    tot = sum(pool.values())
    wacc = 0.0
    detail = {}
    for role in ("query", "component"):
        for pred in (True, False):
            sk = f"{role}_{'strict' if pred else 'nonstrict'}"
            sub = [r for r in joined if r["role"] == role and r["pred"] == pred]
            if not sub:
                continue
            acc = sum(1 for r in sub if r["pred"] == r["human"]) / len(sub)
            w = pool[sk] / tot
            wacc += w * acc
            detail[sk] = {"n_audited": len(sub), "stratum_accuracy": acc,
                          "corpus_weight": round(w, 4)}
    res["corpus_weighted_accuracy"] = {"estimate": round(wacc, 4),
                                       "strata": detail,
                                       "note": "weights = corpus pool shares "
                                               "at sampling time"}
    OUT.write_text(json.dumps(res, indent=1))
    o = res["overall"]
    print(f"unambiguous n={o['n']}  acc={o.get('accuracy'):.3f} "
          f"CI[{o['accuracy_ci'][0]:.3f},{o['accuracy_ci'][1]:.3f}]")
    if "sensitivity" in o:
        print(f"sens={o['sensitivity']:.3f} CI[{o['sensitivity_ci'][0]:.3f},"
              f"{o['sensitivity_ci'][1]:.3f}]  spec={o.get('specificity'):.3f} "
              f"CI[{o['specificity_ci'][0]:.3f},{o['specificity_ci'][1]:.3f}]")
    print("weighted corpus accuracy:", res["corpus_weighted_accuracy"]["estimate"])
    print("false calls:", len(res["false_calls"]), "ambiguous:", len(ambiguous))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
