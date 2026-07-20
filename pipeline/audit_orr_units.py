"""Audit the ORR corpus for the endpoint-unit bug and quantify its impact.

Writes a machine-readable correction table so that the affected pseudo-queries
can be re-run or excluded, and so the manuscript can state the impact precisely
rather than in the abstract.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from fix_endpoint_units import UnitError, rate_from_row  # noqa: E402

ORR = re.compile(r"(objective response|overall response|response rate|\bORR\b|"
                 r"best overall response)", re.I)

REPO = Path(__file__).resolve().parents[1]

# Both inputs are large runtime artifacts and are therefore not tracked in git.
# Paths are overridable so the audit can be pointed at whichever run is being
# checked; the defaults match the layout produced by the pipeline scripts.
DEFAULT_CORPUS = REPO / "artifacts/oncology_trial_similarity/trial_summaries.jsonl"
DEFAULT_LAMBDA = (REPO / "artifacts/retrospective_lambda_oncology_orr_120"
                  / "lambda_component_features.csv")
# Earlier runs produced the lambda dataset inside a git worktree; fall back to
# that location so previously generated results remain auditable.
LEGACY_LAMBDA = (REPO / ".worktrees/trial2vec-secret-mixture-prior/artifacts"
                 / "retrospective_lambda_oncology_orr_120/lambda_component_features.csv")
DEFAULT_OUT = REPO / "artifacts/orr_unit_audit"


def corpus_rows(corpus):
    with open(corpus, encoding="utf-8") as fh:
        for line in fh:
            yield json.loads(line)


def resolve_inputs(args):
    corpus = Path(args.corpus) if args.corpus else DEFAULT_CORPUS
    if not corpus.exists():
        raise SystemExit(
            f"Corpus not found: {corpus}\n"
            "This is an untracked runtime artifact. Build it with the pipeline "
            "scripts, or pass --corpus to point at an existing copy.")
    if args.lambda_features:
        lam = Path(args.lambda_features)
    elif DEFAULT_LAMBDA.exists():
        lam = DEFAULT_LAMBDA
    elif LEGACY_LAMBDA.exists():
        lam = LEGACY_LAMBDA
    else:
        lam = None
        print("note: lambda component features not found; reporting the "
              "corpus-wide audit only. Pass --lambda-features to include the "
              "per-query impact table.")
    return corpus, lam


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--corpus", default=None,
                    help="trial_summaries.jsonl (default: artifacts/...)")
    ap.add_argument("--lambda-features", default=None,
                    help="lambda_component_features.csv from a borrowing run")
    ap.add_argument("--output", default=None)
    args = ap.parse_args()

    corpus, lambda_path = resolve_inputs(args)
    OUT = Path(args.output) if args.output else DEFAULT_OUT
    OUT.mkdir(parents=True, exist_ok=True)

    # ---- 1. corpus-wide audit of arm-level ORR rows --------------------
    total = failed = 0
    impossible = 0
    by_unit = {}
    for d in corpus_rows(corpus):
        for r in d.get("results", {}).get("primary_results", []) or []:
            if not isinstance(r, dict) or not ORR.search(r.get("title") or ""):
                continue
            unit = r.get("unit")
            for a in r.get("arm_results") or []:
                val, den = a.get("count"), a.get("denominator")
                if val is None or not den:
                    continue
                total += 1
                stored = a.get("proportion")
                if stored is not None and stored > 1.0:
                    impossible += 1
                key = (unit or "").strip().lower()
                slot = by_unit.setdefault(key, {"n": 0, "changed": 0, "max_abs": 0.0})
                slot["n"] += 1
                try:
                    correct = rate_from_row(unit, val, den)
                except UnitError:
                    failed += 1
                    continue
                buggy = float(val) / float(den)
                if abs(buggy - correct) > 1e-9:
                    slot["changed"] += 1
                    slot["max_abs"] = max(slot["max_abs"], abs(buggy - correct))

    summary = {
        "arm_level_orr_rows": total,
        "rows_with_stored_proportion_above_1": impossible,
        "rows_unconvertible_under_strict_units": failed,
        "by_unit": {k: v for k, v in sorted(by_unit.items(), key=lambda x: -x[1]["n"])},
    }

    # ---- 2. impact on the 120-query borrowing dataset ------------------
    used = {}
    if lambda_path and lambda_path.exists():
        with open(lambda_path, newline="", encoding="utf-8") as fh:
            for row in csv.DictReader(fh):
                used.setdefault(row["query_nct_id"],
                                (float(row["query_count"]), float(row["query_denominator"])))

    index = {d["nct_id"]: d for d in corpus_rows(corpus) if d["nct_id"] in used}
    records = []
    for nct, (qc, qd) in sorted(used.items()):
        d = index.get(nct)
        if not d:
            records.append({"nct_id": nct, "status": "not_in_corpus"})
            continue
        res = d.get("results", {})
        pool = list(res.get("primary_results") or [])
        so = res.get("secondary_or_other")
        if isinstance(so, list):
            pool += [x for x in so if isinstance(x, dict)]
        hit = None
        for r in pool:
            for a in r.get("arm_results") or []:
                if a.get("denominator") == qd and a.get("count") == qc:
                    hit = (r.get("unit"), r.get("title"))
                    break
            if hit:
                break
        if not hit:
            records.append({"nct_id": nct, "status": "row_not_located",
                            "used_count": qc, "used_denominator": qd})
            continue
        unit, title = hit
        buggy = qc / qd if qd else float("nan")
        try:
            correct = rate_from_row(unit, qc, qd)
            status = "ok" if abs(buggy - correct) <= 1e-9 else "CORRECTED"
        except UnitError as exc:
            correct, status = float("nan"), f"unconvertible: {exc}"
        records.append({
            "nct_id": nct, "status": status, "unit": unit,
            "endpoint_title": (title or "")[:80],
            "used_count": qc, "used_denominator": qd,
            "rate_as_used": round(buggy, 6),
            "rate_corrected": None if correct != correct else round(correct, 6),
            "abs_error": None if correct != correct else round(abs(buggy - correct), 6),
        })

    corrected = [r for r in records if r["status"] == "CORRECTED"]
    material = [r for r in corrected if (r.get("abs_error") or 0) > 0.10]
    summary["borrowing_dataset"] = {
        "queries_with_components": len(used),
        "rows_located_in_corpus": sum(1 for r in records
                                      if r["status"] in ("ok", "CORRECTED")),
        "queries_corrected": len(corrected),
        "queries_with_abs_error_gt_0.10": len(material),
        "max_abs_error": max([r.get("abs_error") or 0 for r in records], default=0),
    }

    (OUT / "unit_audit_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8")
    keys = ["nct_id", "status", "unit", "endpoint_title", "used_count",
            "used_denominator", "rate_as_used", "rate_corrected", "abs_error"]
    with open(OUT / "query_corrections.csv", "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=keys, extrasaction="ignore")
        w.writeheader()
        w.writerows(records)

    print(json.dumps(summary, indent=2)[:1800])
    print(f"\nwrote {OUT}")
    print(f"\nMaterially wrong held-out ORR ({len(material)} queries):")
    for r in sorted(material, key=lambda x: -(x["abs_error"] or 0)):
        print(f"  {r['nct_id']}  used {r['rate_as_used']:.3f} -> "
              f"correct {r['rate_corrected']:.3f}  (error {r['abs_error']:.3f})")


if __name__ == "__main__":
    main()
