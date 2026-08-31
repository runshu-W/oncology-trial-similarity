#!/usr/bin/env python3
"""Round-8 programmatic verification: every number cited in the round-8
edits must match the artifact JSONs, and the mandated wording changes must
be present (and the banned wordings absent) in the tex sources."""
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
M = (ROOT / "manuscript" / "manuscript_full_draft_pharm_stats.tex").read_text()
S = (ROOT / "manuscript" / "supplement.tex").read_text()
DP = json.load(open(ROOT / "artifacts_round4" / "stress" / "round8_direct_paired.json"))
CF = json.load(open(ROOT / "artifacts_round4" / "calibration" / "round8_crossfit_dist.json"))
R7 = json.load(open(ROOT / "artifacts_round4" / "calibration" / "round7_crossfit.json"))

ok = fail = 0


def chk(name, cond):
    global ok, fail
    if cond:
        ok += 1
    else:
        fail += 1
        print("FAIL:", name)


def near(a, b, tol=5e-5):
    return abs(a - b) <= tol


def r(x, d):
    return round(x, d)


# ---- direct paired artifact values vs the numbers cited ------------------
exp = {
    "primaryB_pooled": (-0.0289, [-0.0530, -0.0091], [-0.0527, -0.0101], 589, 50),
    "primaryB_engaged": (-0.1428, [-0.2584, -0.0480], [-0.2581, -0.0556], 119, 50),
    "fwd": (-0.0148, [-0.0301, -0.0011], [-0.0299, -0.0014], 775, 281),
    "strictA": (-0.0147, [-0.0355, 0.0037], [-0.0371, 0.0054], 589, 156),
    "strictC": (-0.0108, [-0.0254, 0.0027], [-0.0260, 0.0029], 589, 176),
}
for k, (m, ci, cb, n, nb) in exp.items():
    d = DP[k]
    chk(f"{k} mean", near(r(d["mean"], 4), m))
    chk(f"{k} ci_ord", all(near(r(a, 4), b) for a, b in zip(d["ci95_ordinary"], ci)))
    chk(f"{k} ci_block", all(near(r(a, 4), b) for a, b in zip(d["ci95_block"], cb)))
    chk(f"{k} n", d["n"] == n)
    chk(f"{k} binding", d["n_binding"] == nb)
chk("fwd blocks 365", DP["fwd"]["n_blocks"] == 365)
chk("strictA blocks 267", DP["strictA"]["n_blocks"] == 267)
chk("fwd capped NLL 2.8027", near(r(DP["fwd"]["mean_capped_nll"], 4), 2.8027))
chk("fwd uncapped NLL 2.8175", near(r(DP["fwd"]["mean_uncapped_nll"], 4), 2.8175))
chk("fwd diff = NLLcap-NLLunc",
    near(DP["fwd"]["mean"],
         DP["fwd"]["mean_capped_nll"] - DP["fwd"]["mean_uncapped_nll"], 1e-9))
ctx = DP["primaryB_context"]
chk("primary diff = capdelta-uncdelta",
    near(DP["primaryB_pooled"]["mean"],
         ctx["mean_capped_minus_eb"] - ctx["mean_uncapped_minus_eb"], 1e-9))
chk("primary capped vs EB +0.006", near(r(ctx["mean_capped_minus_eb"], 3), 0.006))
chk("primary uncapped vs EB +0.035", near(r(ctx["mean_uncapped_minus_eb"], 3), 0.035))
chk("engaged n 119", ctx["n_engaged"] == 119)

# 3dp versions used in abstract/S1/conclusion
chk("pooled 3dp -0.029", near(r(DP["primaryB_pooled"]["mean"], 3), -0.029))
chk("pooled block 3dp", all(near(r(a, 3), b) for a, b in
                            zip(DP["primaryB_pooled"]["ci95_block"], [-0.053, -0.010])))
chk("pooled ord 3dp", all(near(r(a, 3), b) for a, b in
                          zip(DP["primaryB_pooled"]["ci95_ordinary"], [-0.053, -0.009])))
chk("engaged 3dp -0.143", near(r(DP["primaryB_engaged"]["mean"], 3), -0.143))
chk("engaged ord 3dp", all(near(r(a, 3), b) for a, b in
                           zip(DP["primaryB_engaged"]["ci95_ordinary"], [-0.258, -0.048])))
chk("engaged block 3dp", all(near(r(a, 3), b) for a, b in
                             zip(DP["primaryB_engaged"]["ci95_block"], [-0.258, -0.056])))

# ---- cross-fit distribution ---------------------------------------------
chk("cf mean", CF["mean"] == -0.00073)
chk("cf median", CF["median"] == -0.00094)
chk("cf q2.5", CF["q"]["0.025"] == -0.00144)
chk("cf q25", CF["q"]["0.25"] == -0.00114)
chk("cf q50", CF["q"]["0.5"] == -0.00094)
chk("cf q75", CF["q"]["0.75"] == -0.00072)
chk("cf q97.5", CF["q"]["0.975"] == 0.0039)
chk("cf frac_pos 4%", CF["frac_positive"] == 0.04)
chk("cf matches round7", CF["check_round7"]["match"] is True)
cnt, edges = CF["hist"]["counts"], CF["hist"]["bin_edges"]
low = sum(c for c, e in zip(cnt, edges) if e < 0.0001)
chk("hist 192 below +0.0001", low == 192)
tail = [(edges[i], edges[i + 1]) for i, c in enumerate(cnt)
        if c > 0 and edges[i] > 0.001]
chk("hist tail 8 splits", sum(c for i, c in enumerate(cnt) if edges[i] > 0.001) == 8)
chk("tail range 0.0022-0.0057",
    min(e[0] for e in tail) >= 0.0022 and max(e[1] for e in tail) <= 0.0057)
chk("mean at ~75th pct", CF["q"]["0.75"] >= CF["mean"] >= CF["q"]["0.5"])

# ---- pointwise asymmetry values cited in the new sentence ----------------
pw = R7["pointwise_all_cells"]
for tag, v in (("s-0.5", 0.057), ("s0.5", 0.020), ("s-0.75", 0.022),
               ("s0.75", 0.007), ("s0.0", 0.069), ("s-0.25", 0.074),
               ("s0.25", 0.054)):
    chk(f"pointwise {tag}", near(r(pw[tag], 3), v))

# ---- textual presence / absence -----------------------------------------
for frag in (
        "$-0.0289$", "$[-0.0530,-0.0091]$", "$[-0.0527,-0.0101]$",
        "$-0.1428$", "$[-0.2584,-0.0480]$", "$[-0.2581,-0.0556]$",
        "$-0.0148$", "$[-0.0301,-0.0011]$", "$[-0.0299,-0.0014]$",
        "$-0.0147$", "$[-0.0355,+0.0037]$", "$[-0.0371,+0.0054]$",
        "$-0.0108$", "$[-0.0254,+0.0027]$", "$[-0.0260,+0.0029]$",
        "tab:directpaired", "assessed by repeated random splitting",
        "median $-0.0009$", "not a confidence interval",
        "asymmetric magnitude", "fades faster on the positive side",
        "all non-repeated plug-in", "does not\ndemonstrate a",
        "observed local ROC-efficiency advantage",
        "no\nestablished mixture-average gain",
        "carry\n(limited) confirmatory weight",
        "not detectably different"):
    chk(f"manuscript has {frag[:40]!r}", frag in M)
for frag in ("neutral", "roughly symmetric", "is propagated",
             "carry the confirmatory weight", "costs nothing on the favourable",
             "exactly neutral", "power gain concentrates"):
    chk(f"manuscript lacks {frag!r}", frag not in M)
for frag in ("$-0.00144/-0.00114/-0.00094/-0.00072/+0.00390$",
             "round8\\_direct\\_paired.json",
             "round8\\_crossfit\\_dist.json",
             "median $-0.0009$", "not a confidence interval",
             "$281/775$ binding", "Round-8 additions (this revision)"):
    chk(f"supplement has {frag[:40]!r}", frag in S)
for frag in ("improves, not taxes", "neutral"):
    chk(f"supplement lacks {frag!r}", frag not in S)
# every capped-vs-EB number reused this round is unchanged
for frag in ("$-0.0413$", "$[-0.0625,-0.0202]$", "$-0.0485$",
             "$[-0.0742,-0.0233]$", "$-0.0440$", "$[-0.0667,-0.0223]$"):
    chk(f"manuscript keeps {frag}", frag in M)

print(f"\n{ok} OK, {fail} FAIL")
sys.exit(1 if fail else 0)
