#!/usr/bin/env python3
"""Rule-based endpoint canonicaliser for the strict-ORR estimand (round 3, Major 3).

Classifies an extracted endpoint TITLE as strict ORR (objective/overall
response, CR+PR or a disease-appropriate >=PR equivalent) or not (CR-only,
clinical-benefit/disease-control, pathological response, composite remission,
molecular/cytogenetic response, PFS-defined, non-tumour, ...).

Exclusions are checked before inclusions; an unmatched title is NOT strict
(conservative). Titles only — measure descriptions are not extracted — so
title/definition mismatches (an ORR-titled endpoint defined as PFS >= 6
months) are known misses; the classifier is therefore VALIDATED against the
80-row expert audit, and its confusion matrix is reported rather than assumed.

Usage:
  python3 endpoint_canonicalizer.py --validate   # confusion vs expert audit
"""
from __future__ import annotations

import re

_EXCLUDE = [
    r"clinical benefit", r"\bcbr\b", r"disease control", r"\bdcr\b",
    r"patholog", r"\bpcr\b",
    r"molecular", r"cytogenetic", r"\bmmr\b", r"\bmcyr\b", r"\bmrd\b",
    r"minimal residual",
    r"progression[- ]free", r"\bpfs\b", r"survival",
    r"emesis", r"nausea", r"vomit",
    r"csf", r"clearance",
    r"remission",          # composite CR/CRi remission endpoints
    r"cr *\+ *cri", r"\bcrc\b", r"composite",
    r"immune[- ]related response",   # nonstandard irRC-like criteria
    r"major response",               # >=50% paraprotein landmark analogue
    r"clinical (complete )?response",  # exam-based response at primary site
    r"duration", r"time to", r"landmark",
]
_CR_ONLY = re.compile(r"complete (response|remission)", re.I)
_PARTIAL = re.compile(r"partial|objective|overall|\borr\b|\bbor\b", re.I)
_INCLUDE = [
    r"objective (radiographic )?response",
    r"overall (hematologic )?response",
    r"best overall response",
    r"\borr\b",
    r"confirmed (tumor |objective )?response",
    r"(tumor )?response rate",
    r"number of (participants|patients) with (a |an )?(objective )?response\b",
    r"(participants|patients) with (a |an )?response\b",
    r"with response\b",
]
_EXC_RE = [re.compile(p, re.I) for p in _EXCLUDE]
_INC_RE = [re.compile(p, re.I) for p in _INCLUDE]


def is_strict_orr(title: str | None) -> bool:
    if not title or not title.strip():
        return False
    t = " ".join(str(title).split())
    for p in _EXC_RE:
        if p.search(t):
            return False
    # CR-only: mentions complete response without any partial/objective/overall
    if _CR_ONLY.search(t) and not _PARTIAL.search(t):
        return False
    return any(p.search(t) for p in _INC_RE)


def _validate() -> None:
    import csv
    import json
    from pathlib import Path
    root = Path(__file__).resolve().parents[1]
    ex = [json.loads(l) for l in
          open(root / "artifacts_round2" / "examples_unitfix_ids_ep.jsonl")]
    donor_ep = {}
    for e in ex:
        for c in e["components"]:
            donor_ep.setdefault((c["nct_id"], int(round(c["count"])),
                                 int(round(c["denominator"]))), c.get("endpoint"))
    query_ep = {e["query_nct_id"]: e["query"].get("endpoint") for e in ex}

    rows = list(csv.DictReader(open(
        root / "docs" / "manual_audit_sample_2026-08-31_completed.csv")))
    tp = tn = fp = fn = excl = miss = 0
    misses = []
    for r in rows:
        d = r["audit_orr_definition_ok"].strip().lower()
        m = r["audit_endpoint_mapping_ok"].strip().lower()
        if d == "unclear" or m == "unclear":
            excl += 1
            continue
        truth = (d == "yes") and (m == "yes")
        if r["row_type"] == "query_outcome":
            title = query_ep.get(r["nct_id"])
        else:
            key = (r["nct_id"], int(r["extracted_responders"]),
                   int(r["extracted_denominator"]))
            title = donor_ep.get(key)
        if title is None:
            miss += 1
            continue
        pred = is_strict_orr(title)
        if pred and truth:
            tp += 1
        elif (not pred) and (not truth):
            tn += 1
        elif pred and not truth:
            fp += 1
            misses.append(("FP", r["nct_id"], title[:70]))
        else:
            fn += 1
            misses.append(("FN", r["nct_id"], title[:70]))
    n = tp + tn + fp + fn
    print(f"validated on {n} audited rows (excluded unclear: {excl}, unmatched: {miss})")
    print(f"accuracy {(tp+tn)/n:.1%}  sens(strict) {tp/(tp+fn):.1%}  "
          f"spec(non-strict) {tn/(tn+fp):.1%}  (tp={tp} tn={tn} fp={fp} fn={fn})")
    for kind, nct, title in misses:
        print(f"  {kind}: {nct}  {title!r}")


if __name__ == "__main__":
    _validate()
