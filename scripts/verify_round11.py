#!/usr/bin/env python3
"""Round-11 programmatic verification."""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
M = (ROOT / "manuscript" / "manuscript_full_draft_pharm_stats.tex").read_text()
S = (ROOT / "manuscript" / "supplement.tex").read_text()
A = json.load(open(ROOT / "artifacts_round4" / "calibration" / "round11_inference_scope.json"))

ok = fail = 0


def chk(name, cond):
    global ok, fail
    if cond:
        ok += 1
    else:
        fail += 1
        print("FAIL:", name)


def r(x, d):
    return round(x, d)


runs = A["runs"]
c = runs["canonical@0.02513"]
chk("gap mean -7e-6", c["gap"]["mean"] == -7e-06)
chk("gap central range", c["gap"]["q"]["0.025"] == -0.000215
    and c["gap"]["q"]["0.975"] == 0.000207)
chk("gap max_abs 0.00058", r(c["gap"]["max_abs"], 5) == 0.00058)
chk("all gaps within 0.00075",
    max(runs[k]["gap"]["max_abs"] for k in runs) <= 0.00075)
for k in ("canonical@0.02513", "canonical@0.025", "N(0,1)@0.02513",
          "N(0,0.5)@0.02513"):
    chk(f"{k} matches round10", runs[k].get("matches_round10") is True)
l1, l5 = runs["N(+0.124,1)@0.02513"], runs["N(+0.124,0.5)@0.02513"]
chk("loc-held N1 +0.0117", r(l1["est"]["mean"], 4) == 0.0117
    and l1["est"]["frac_positive"] == 1.0)
chk("loc-held N05 +0.0362", r(l5["est"]["mean"], 4) == 0.0362
    and l5["est"]["frac_positive"] == 1.0)
chk("loc-held N1 q", l1["est"]["q"]["0.025"] == 0.01105
    and l1["est"]["q"]["0.975"] == 0.01249)
chk("loc-held N05 q", l5["est"]["q"]["0.025"] == 0.03512
    and l5["est"]["q"]["0.975"] == 0.03725)
fb = A["full_batch_randomized"]
chk("fb +0.0025 mcse 0.0007", r(fb["estimate"], 4) == 0.0025
    and fb["mcse"] == 0.0007)
chk("fb sizes", fb["size_weak"] == 0.02513 and fb["size_cap50"] == 0.02513)
wb = A["weight_bootstrap_randomized"]
chk("wb mean +0.0024", r(wb["mean"], 4) == 0.0024)
chk("wb interval", r(wb["q"]["0.025"], 4) == 0.0005
    and r(wb["q"]["0.975"], 4) == 0.0045)
chk("wb 499/500", wb["frac_positive"] == 0.998 and wb["n"] == 500)
so = A["selection_overlap"]
chk("overlap 169", so["n_atom"] == 281 and so["n_rule"] == 291
    and so["n_overlap"] == 169)
chk("atom share 36%", r(so["atom_share_of_fwd"] * 100, 0) == 36)

for frag in (
        "positive in this conditional analysis",
        "\\emph{training-half} empirical size exactly",
        "central $95\\%$ range $[-0.000215, +0.000207]$",
        "maximum magnitude $0.00058$",
        "recomputed for the randomized estimator",
        "$[+0.0005, +0.0045]$", "$499$ of $500$",
        "not jointly\npropagated",
        "no independent replicate batch generated",
        "population-level or cross-batch claim is not made",
        "$+0.0117$ and\n$+0.0362$ on the same splits",
        "stable across the examined splits",
        "a different estimand",
        "does not attach to this frozen rule",
        "neither superiority nor\nequivalence was established",
        "neither superiority nor equivalence of the two",
        "coarse risk grouping",
        "overlaps the rule's selection in $169$ queries",
        "$36\\%$ of forward queries",
        "value is graded triage",
        "significant against the EB",
        "the exploratory five-seed ensemble is as",
        "(alongside the exploratory five-seed ensemble)",
        "location-held\n$N(+0.124,1)$: $+0.0117$; $N(+0.124,0.5)$: $+0.0362$",
        "conditional,\nbatch-specific comparison",
        "to be assigned upon acceptance",
        "verify\\_round*.py",
        "training-half size exact,\nevaluation-half average size approximately matched"):
    chk(f"M has {frag[:44]!r}", frag in M)
for frag in ("only significant", "exactly size-matched",
             "size-matched exactly", "exactly in expectation",
             "robust under repetition", "matches, but does not beat",
             "still matches, and does not beat", "is real but"):
    chk(f"M lacks {frag!r}", frag not in M)
for frag in ("Round-11 additions (this revision)",
             "round11\\_\\allowbreak inference\\_scope.json",
             "$[-0.000215, +0.000207]$", "$0.000577$",
             "$+0.00252$", "$0.00070$",
             "$[+0.00052, +0.00449]$", "$499/500$ positive",
             "$N(+0.124, 1)$ gives $+0.01171$ $[+0.01105,+0.01249]$",
             "$N(+0.124, 0.5)$ gives $+0.03615$ $[+0.03512,+0.03725]$",
             "share $169$ queries",
             "uncapped-architecture mechanism diagnostic), and the structure",
             "final-design claims rest on the final-design",
             "coarse risk\ngrouping, not a per-query ranking",
             "as is the exploratory five-seed ensemble",
             "positive in this conditional\nanalysis"):
    chk(f"S has {frag[:44]!r}", frag in S)
for frag in ("only significant", "robust under repetition",
             "exactly size-matched",
             "per-query value is harm avoidance"):
    chk(f"S lacks {frag!r}", frag not in S)

print(f"\n{ok} OK, {fail} FAIL")
sys.exit(1 if fail else 0)
