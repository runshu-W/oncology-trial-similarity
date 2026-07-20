"""Worked case study: borrow for one real query trial, compare methods.

Takes a pseudo-query result produced by the retrospective pipeline, rebuilds the
historical mixture prior under every borrowing method, and scores each against
the query's held-out outcome. This is the real-data counterpart to the
gold-standard simulation: there is no known borrowing truth here, so methods are
compared on held-out predictive performance and on how much they borrow, not on
whether they picked the "right" donors.

The prior implementations are imported from ``simulation/gold_standard`` rather
than reimplemented, so the case study and the simulation are guaranteed to be
running identical code.

Endpoint values are converted to rates with ``pipeline/fix_endpoint_units.py``,
which dispatches on the reported unit. The legacy pipeline path divides by the
denominator unconditionally and is wrong for percentage-reported outcomes; see
``docs/KNOWN_ISSUE_endpoint_units.md``. Candidates whose unit cannot be
interpreted are dropped and counted rather than guessed at.

Usage:
    python3 run_case_study.py --query-json path/to/NCT01234567.json
    python3 run_case_study.py --query-dir path/to/pseudo_query_results --top 2
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "simulation" / "gold_standard"))
sys.path.insert(0, str(REPO / "pipeline"))

from fix_endpoint_units import UnitError, responders_from_row  # noqa: E402
from methods import (  # noqa: E402
    TwoHeadDeepSets, apply_sam, pooling, power_prior, robust_map,
    rule_components, two_head_prior, uip_dirichlet, uip_js, weak_only,
)

ORR_PATTERN = re.compile(
    r"(objective response|overall response|best overall response|response rate|\bORR\b)",
    re.I)

# The reranker reports seven named dimensions; the borrowing model consumes the
# nine-dimensional vector used throughout the rest of the codebase.
DIMENSION_MAP = {
    "disease": "disease_population_match",
    "regimen": "treatment_regimen_match",
    "endpoint": "endpoint_estimand_match",
    "followup": "safety_and_followup_relevance",
    "eligibility": "eligibility_criteria_overlap",
    "result_quality": "result_usability",
}


def extract_rate(results_block, pattern=ORR_PATTERN):
    """Find an endpoint matching ``pattern`` and return (y, n, unit, title).

    Returns None when no usable arm-level row exists. Raises nothing: an
    unreadable unit is treated as "not usable", which is the conservative
    reading and is counted by the caller.
    """
    if not isinstance(results_block, dict):
        return None
    pool = list(results_block.get("primary_results") or [])
    secondary = results_block.get("secondary_or_other")
    if isinstance(secondary, list):
        pool += [x for x in secondary if isinstance(x, dict)]
    for row in pool:
        if not isinstance(row, dict) or not pattern.search(row.get("title") or ""):
            continue
        unit = row.get("unit")
        for arm in row.get("arm_results") or []:
            value, denom = arm.get("count"), arm.get("denominator")
            if value is None or not denom:
                continue
            try:
                y, n, derived = responders_from_row(unit, value, denom)
            except UnitError:
                continue
            if n > 0:
                return {"y": y, "n": n, "unit": unit,
                        "title": row.get("title"), "derived_from_percentage": derived}
    return None


def load_corpus_index(corpus_path, wanted):
    """Map nct_id -> results block for the ids we actually need.

    The retrieval snapshot inside a pseudo-query file carries only denominators,
    not arm-level rows, so candidate outcomes are read from the structured
    corpus instead.
    """
    index = {}
    if not corpus_path:
        return index
    with open(corpus_path, encoding="utf-8") as fh:
        for line in fh:
            doc = json.loads(line)
            if doc.get("nct_id") in wanted:
                index[doc["nct_id"]] = doc.get("results")
    return index


def build_candidates(query_doc, corpus_index=None):
    """Join reranked dimension scores with candidate outcome tables."""
    results_by_id = dict(corpus_index or {})
    for match in query_doc.get("top_matches") or []:
        if isinstance(match, dict) and match.get("nct_id"):
            # Only fall back to the trimmed snapshot when the corpus lacks the id.
            results_by_id.setdefault(match["nct_id"], match.get("results"))

    candidates, dropped = [], {"no_outcome": 0, "no_scores": 0}
    for entry in query_doc.get("reranked_top_matches") or []:
        nct = entry.get("candidate_nct_id")
        scores = entry.get("dimension_scores") or {}
        if not nct or not scores:
            dropped["no_scores"] += 1
            continue
        outcome = extract_rate(results_by_id.get(nct))
        if outcome is None:
            dropped["no_outcome"] += 1
            continue

        dim = {key: float(scores.get(src, 0.0) or 0.0)
               for key, src in DIMENSION_MAP.items()}
        # Red-flag severity is not scored directly; derive it from the count of
        # flags the reranker raised, capped at 1.
        dim["redflag"] = min(len(entry.get("red_flags") or []) / 3.0, 1.0)
        dim["overall100"] = float(entry.get("overall_similarity_score") or 0.0)

        features = np.array([
            dim["overall100"] / 100.0, dim["disease"] / 5.0, dim["regimen"] / 5.0,
            dim["endpoint"] / 5.0, dim["followup"] / 5.0, dim["eligibility"] / 5.0,
            dim["result_quality"] / 5.0, -dim["redflag"],
            float(np.log1p(outcome["n"])),
        ], dtype=np.float64)

        candidates.append({
            "nct_id": nct, "features": features, "dim": dim,
            "y": outcome["y"], "n": outcome["n"],
            "endpoint_title": outcome["title"], "unit": outcome["unit"],
            "derived_from_percentage": outcome["derived_from_percentage"],
            "rank": entry.get("rank"),
        })
    return candidates, dropped


def method_table(model):
    """name -> (builder, uses SAM, available at design time)."""
    table = {
        "no_borrowing":   (lambda c, y, n: weak_only(c), False, True),
        "rule":           (lambda c, y, n: rule_components(c), False, True),
        "rule_sam":       (lambda c, y, n: rule_components(c), True, False),
        "robust_map_w0.5": (lambda c, y, n: robust_map(c, w=0.5), False, True),
        "power_prior":    (lambda c, y, n: power_prior(c, a0=0.30), False, True),
        "uip_dirichlet":  (lambda c, y, n: uip_dirichlet(c, y, n), False, True),
        "uip_js":         (lambda c, y, n: uip_js(c, y, n), False, False),
        "pooling":        (lambda c, y, n: pooling(c), False, True),
    }
    if model is not None:
        table["two_head_pro"] = (
            lambda c, y, n: two_head_prior(model, c, prospective=True), False, True)
        table["two_head_pro_sam"] = (
            lambda c, y, n: two_head_prior(model, c, prospective=True), True, False)
    return table


def run_one(query_doc, model, nct_id, corpus_index=None):
    candidates, dropped = build_candidates(query_doc, corpus_index)
    held = extract_rate((query_doc.get("heldout_query_outcomes") or {}).get("endpoints")
                        and {"primary_results":
                             (query_doc["heldout_query_outcomes"]["endpoints"]
                              .get("primary") or []),
                             "secondary_or_other":
                             (query_doc["heldout_query_outcomes"]["endpoints"]
                              .get("secondary_or_other") or [])})
    if held is None:
        return None, f"{nct_id}: no usable held-out ORR outcome"
    if not candidates:
        return None, f"{nct_id}: no candidates with usable outcomes"

    y0, n0 = held["y"], held["n"]
    observed = y0 / n0
    rows = []
    for name, (build, use_sam, design_time) in method_table(model).items():
        prior = build(candidates, y0, n0)
        if use_sam:
            prior, _ = apply_sam(prior, y0, n0)
        nll = -prior.log_predictive(y0, n0)
        post_mean, lo, hi, _, _ = prior.posterior_summary(y0, n0, observed, observed)
        rows.append({
            "query_nct_id": nct_id,
            "method": name,
            "design_time_available": design_time,
            "n_candidates": len(candidates),
            "heldout_y": y0, "heldout_n": n0,
            "heldout_rate": round(observed, 4),
            "posterior_mean": round(post_mean, 4),
            "abs_error": round(abs(post_mean - observed), 4),
            "heldout_nll": round(nll, 4),
            "ci95_low": round(lo, 4), "ci95_high": round(hi, 4),
            "ci95_covers_observed": bool(lo <= observed <= hi),
            "prior_ess": round(prior.prior_ess(), 2),
            "historical_mass": round(prior.historical_mass(), 4),
        })
    meta = {
        "query_nct_id": nct_id,
        "heldout_endpoint": held["title"],
        "heldout_unit": held["unit"],
        "heldout_derived_from_percentage": held["derived_from_percentage"],
        "n_candidates_used": len(candidates),
        "candidates_dropped_no_outcome": dropped["no_outcome"],
        "candidates_dropped_no_scores": dropped["no_scores"],
        "candidates_derived_from_percentage": sum(
            c["derived_from_percentage"] for c in candidates),
    }
    return (rows, meta), None


def write_tex(rows, path):
    by_query = {}
    for r in rows:
        by_query.setdefault(r["query_nct_id"], []).append(r)
    lines = []
    for nct, rs in by_query.items():
        head = rs[0]
        lines += [
            r"\begin{table}[htbp]", r"\centering", r"\small",
            rf"\caption{{Worked case study ({nct}, held-out ORR "
            rf"${head['heldout_y']}/{head['heldout_n']} = "
            rf"{head['heldout_rate']:.3f}$). Methods marked $\dagger$ use the "
            rf"held-out outcome and are not available at design time.}}",
            rf"\label{{tab:casestudy-{nct}}}",
            r"\begin{tabular}{lrrrrr}", r"\toprule",
            r"Method & Posterior mean & $|$error$|$ & Held-out NLL & 95\% CrI & Prior ESS \\",
            r"\midrule",
        ]
        for r in rs:
            dagger = "" if r["design_time_available"] else r"$^\dagger$"
            lines.append(
                f"{r['method'].replace('_', chr(92)+'_')}{dagger} & "
                f"{r['posterior_mean']:.3f} & {r['abs_error']:.3f} & "
                f"{r['heldout_nll']:.3f} & "
                f"[{r['ci95_low']:.3f}, {r['ci95_high']:.3f}] & "
                f"{r['prior_ess']:.1f} \\\\")
        lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}", ""]
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--query-json", default=None, help="one pseudo-query result JSON")
    ap.add_argument("--query-dir", default=None, help="directory of pseudo-query JSONs")
    ap.add_argument("--top", type=int, default=None,
                    help="with --query-dir, run only the first N queries")
    ap.add_argument("--corpus", default=None,
                    help="trial_summaries.jsonl providing candidate arm-level outcomes")
    ap.add_argument("--model", default=None,
                    help="two-head weights (.npz) from the gold-standard simulation")
    ap.add_argument("--output", default=str(REPO / "artifacts" / "case_studies"))
    args = ap.parse_args()

    paths = []
    if args.query_json:
        paths = [Path(args.query_json)]
    elif args.query_dir:
        paths = sorted(Path(args.query_dir).glob("*.json"))
        if args.top:
            paths = paths[:args.top]
    else:
        raise SystemExit("pass --query-json or --query-dir")
    if not paths:
        raise SystemExit("no query JSON files found")

    model = None
    if args.model:
        mp = Path(args.model)
        if not mp.exists():
            raise SystemExit(f"model not found: {mp}")
        model = TwoHeadDeepSets(input_dim=10)
        z = np.load(mp)
        model.load([z[f"p{i}"] for i in range(len(model.params))])
    else:
        print("note: --model not given, the learned two-head prior is skipped; "
              "closed-form comparators still run.")

    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)

    docs = []
    wanted = set()
    for p in paths:
        doc = json.loads(p.read_text(encoding="utf-8"))
        docs.append((p, doc))
        for e in doc.get("reranked_top_matches") or []:
            if e.get("candidate_nct_id"):
                wanted.add(e["candidate_nct_id"])
    corpus_index = load_corpus_index(args.corpus, wanted)
    if args.corpus:
        print(f"corpus: resolved {len(corpus_index)}/{len(wanted)} candidate ids")
    else:
        print("note: --corpus not given; candidate outcomes will usually be "
              "unavailable because the retrieval snapshot omits arm-level rows.")

    all_rows, all_meta, skipped = [], [], []
    for p, doc in docs:
        nct = (doc.get("query_summary") or {}).get("nct_id") or p.stem
        result, err = run_one(doc, model, nct, corpus_index)
        if err:
            skipped.append(err)
            continue
        rows, meta = result
        all_rows.extend(rows)
        all_meta.append(meta)

    if not all_rows:
        print("No case study could be built:")
        for s in skipped:
            print("  ", s)
        return 1

    with open(out / "case_study_comparison.csv", "w", newline="",
              encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(all_rows[0].keys()))
        w.writeheader()
        w.writerows(all_rows)
    (out / "case_study_metadata.json").write_text(
        json.dumps({"queries": all_meta, "skipped": skipped}, indent=2),
        encoding="utf-8")
    write_tex(all_rows, out / "table_case_studies.tex")

    for meta in all_meta:
        rs = [r for r in all_rows if r["query_nct_id"] == meta["query_nct_id"]]
        head = rs[0]
        print(f"\n=== {meta['query_nct_id']} | held-out ORR "
              f"{head['heldout_y']}/{head['heldout_n']} = {head['heldout_rate']:.3f} "
              f"| {meta['n_candidates_used']} candidates ===")
        print(f"{'method':<20}{'post.mean':>10}{'|err|':>8}{'NLL':>9}"
              f"{'ESS':>8}  design-time")
        for r in sorted(rs, key=lambda x: x["heldout_nll"]):
            print(f"{r['method']:<20}{r['posterior_mean']:>10.3f}"
                  f"{r['abs_error']:>8.3f}{r['heldout_nll']:>9.3f}"
                  f"{r['prior_ess']:>8.1f}  {'yes' if r['design_time_available'] else 'NO'}")
    if skipped:
        print(f"\nskipped {len(skipped)} quer{'y' if len(skipped)==1 else 'ies'}:")
        for s in skipped[:5]:
            print("  ", s)
    print(f"\nwrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
