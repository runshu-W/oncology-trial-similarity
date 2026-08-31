#!/usr/bin/env python3
"""Round-4 out-of-sample endpoint audit sample (review round-4 Major 3).

Draws a NEW audit sample for validating the FROZEN endpoint canonicaliser
out of sample:

  - the canonicaliser rules are frozen first (sha256 of the script recorded);
  - the sampling universe is all unique (nct_id, endpoint title) records in
    the corpus, EXCLUDING every trial already audited in the round-2/3
    80-row sample (any partial familiarity would contaminate independence);
  - stratified by role (query / component) x FROZEN canonicaliser prediction
    (strict / non-strict), 15 records per stratum = 60 rows;
  - the delivered CSV WITHHOLDS the canonicaliser prediction (single-rater
    but prediction-blind adjudication); predictions are stored separately in
    the key file and joined only after the human labels are returned.

Outputs:
  docs/endpoint_audit_sample_round4.csv       (for the human auditor)
  artifacts_round4/audit/endpoint_audit_key_round4.json  (predictions + spec)
"""
from __future__ import annotations

import csv
import hashlib
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from endpoint_canonicalizer import is_strict_orr  # noqa: E402

EX = ROOT / "artifacts_round2" / "examples_unitfix_ids_ep.jsonl"
OLD_AUDIT = ROOT / "docs" / "manual_audit_sample_2026-08-31.csv"
OUT_CSV = ROOT / "docs" / "endpoint_audit_sample_round4.csv"
OUT_KEY = ROOT / "artifacts_round4" / "audit" / "endpoint_audit_key_round4.json"
SEED = 20260901
PER_STRATUM = 15


def main() -> int:
    OUT_KEY.parent.mkdir(parents=True, exist_ok=True)
    canon_src = (ROOT / "scripts" / "endpoint_canonicalizer.py").read_bytes()
    frozen_hash = hashlib.sha256(canon_src).hexdigest()

    audited_ncts = set()
    with open(OLD_AUDIT) as f:
        for r in csv.DictReader(f):
            audited_ncts.add(r["nct_id"])
            if r.get("appears_in_query"):
                audited_ncts.add(r["appears_in_query"])

    records: dict[tuple[str, str], dict] = {}
    with open(EX) as f:
        for line in f:
            e = json.loads(line)
            qn = e["query_nct_id"]
            qe = e["query"].get("endpoint") or ""
            key = (qn, qe)
            if key not in records:
                records[key] = {"nct_id": qn, "endpoint": qe, "role": "query",
                                "count": e["query"]["count"],
                                "denominator": e["query"]["denominator"]}
            for c in e["components"]:
                cn, ce = c["nct_id"], c.get("endpoint") or ""
                k2 = (cn, ce)
                if k2 not in records:
                    records[k2] = {"nct_id": cn, "endpoint": ce,
                                   "role": "component", "count": c["count"],
                                   "denominator": c["denominator"]}

    pool = [r for (n, _), r in records.items() if n not in audited_ncts]
    strata: dict[tuple[str, bool], list[dict]] = {}
    for r in pool:
        r["pred_strict"] = bool(is_strict_orr(r["endpoint"]))
        strata.setdefault((r["role"], r["pred_strict"]), []).append(r)

    rng = np.random.default_rng(SEED)
    chosen = []
    sizes = {}
    for k in sorted(strata, key=str):
        arr = sorted(strata[k], key=lambda r: (r["nct_id"], r["endpoint"]))
        sizes["%s_%s" % (k[0], "strict" if k[1] else "nonstrict")] = len(arr)
        idx = rng.choice(len(arr), size=min(PER_STRATUM, len(arr)), replace=False)
        chosen.extend(arr[i] for i in sorted(idx))
    order = rng.permutation(len(chosen))
    chosen = [chosen[i] for i in order]

    with open(OUT_CSV, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["audit_id", "role", "nct_id", "registry_link",
                    "endpoint_title", "extracted_responders",
                    "extracted_denominator", "human_strict_orr",
                    "human_actual_endpoint_type", "auditor_notes"])
        for i, r in enumerate(chosen, 1):
            w.writerow([f"R4-{i:02d}", r["role"], r["nct_id"],
                        f"https://clinicaltrials.gov/study/{r['nct_id']}",
                        r["endpoint"], r["count"], r["denominator"], "", "", ""])

    key = {"canonicalizer_sha256": frozen_hash, "seed": SEED,
           "per_stratum": PER_STRATUM, "pool_sizes": sizes,
           "n_excluded_ncts": len(audited_ncts),
           "predictions": {f"R4-{i:02d}": {
               "nct_id": r["nct_id"], "endpoint": r["endpoint"],
               "role": r["role"], "pred_strict": r["pred_strict"]}
               for i, r in enumerate(chosen, 1)}}
    OUT_KEY.write_text(json.dumps(key, indent=1))
    print("frozen canonicalizer sha256:", frozen_hash)
    print("pool sizes:", sizes)
    print("rows:", len(chosen), "->", OUT_CSV)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
