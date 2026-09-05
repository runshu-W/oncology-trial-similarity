#!/usr/bin/env python3
"""Round-10 programmatic verification."""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
M = (ROOT / "manuscript" / "manuscript_full_draft_pharm_stats.tex").read_text()
S = (ROOT / "manuscript" / "supplement.tex").read_text()
RC = json.load(open(ROOT / "artifacts_round4" / "calibration" / "round10_randomized_crossfit.json"))
TR = json.load(open(ROOT / "artifacts_round4" / "stress" / "round10_triage_capped.json"))
FB = (ROOT / "pipeline" / "build_main_figures_selective.py").read_text()

ok = fail = 0


def chk(name, cond):
    global ok, fail
    if cond:
        ok += 1
    else:
        fail += 1
        print("FAIL:", name)


def near(a, b, tol=5e-4):
    return abs(a - b) <= tol


def r(x, d):
    return round(x, d)


# ---- randomized-boundary cross-fit ---------------------------------------
c = RC["runs"]["canonical@0.02513"]
chk("can mean +0.0025", near(r(c["mean"], 4), 0.0025, 1e-9))
chk("can median +0.0025", near(r(c["median"], 4), 0.0025, 1e-9))
chk("can sd ~0.0004", 0.00030 <= c["sd"] <= 0.00040)
chk("can range", near(r(c["q"]["0.025"], 4), 0.0018, 1e-9)
    and near(r(c["q"]["0.975"], 4), 0.0032, 1e-9))
chk("can all positive", c["frac_positive"] == 1.0)
chk("can eval sizes 0.02514", near(c["eval_size_weak_mean"], 0.02514, 1e-5)
    and near(c["eval_size_cap50_mean"], 0.02514, 1e-5))
chk("can@0.025 +0.0025", near(r(RC["runs"]["canonical@0.025"]["mean"], 4), 0.0025, 1e-9))
n1 = RC["runs"]["N(0,1)@0.02513"]
n5 = RC["runs"]["N(0,0.5)@0.02513"]
chk("N01 +0.0128", near(r(n1["mean"], 4), 0.0128, 1e-9))
chk("N01 q", n1["q"]["0.025"] == 0.01218 and n1["q"]["0.975"] == 0.01359)
chk("N05 +0.0397", near(r(n5["mean"], 4), 0.0397, 1e-9))
chk("N05 q", n5["q"]["0.025"] == 0.03827 and n5["q"]["0.975"] == 0.04069)
chk("alt all positive", n1["frac_positive"] == 1.0 and n5["frac_positive"] == 1.0)
chk("plug-in consistency", abs(c["mean"] - 0.0024) < 3e-4)
ests = RC["canonical_estimates"]
chk("200 estimates all pos", len(ests) == 200 and min(ests) > 0)

# ---- triage recomputation -------------------------------------------------
rep = TR["uncapped_reproduction"]
chk("quartiles 0.15/0.36/0.65", rep["mass_quartiles"] == [0.15, 0.36, 0.65])
chk("rep rho +0.16", near(r(rep["spearman"], 2), 0.16, 1e-9))
chk("ties 281", TR["n_tie_at_cap"] == 281)
f = TR["final_design"]
chk("fin rho +0.14", near(r(f["spearman"], 2), 0.14, 1e-9))
chk("fin rho CI", near(r(f["spearman_ci"][0], 2), 0.06, 1e-9)
    and near(r(f["spearman_ci"][1], 2), 0.21, 1e-9))
chk("fin rho windows", near(r(f["spearman_by_window"]["w1"], 2), 0.14, 1e-9)
    and near(r(f["spearman_by_window"]["w2"], 2), 0.15, 1e-9)
    and near(r(f["spearman_by_window"]["w3"], 2), 0.13, 1e-9))
g = f["groups"]
chk("low +0.005", near(r(g["low"]["mean"], 3), 0.005, 1e-9)
    and near(r(g["low"]["ci95"][0], 3), -0.004, 1e-9)
    and near(r(g["low"]["ci95"][1], 3), 0.014, 1e-9))
chk("mid -0.051", near(r(g["mid"]["mean"], 3), -0.051, 1e-9)
    and near(r(g["mid"]["ci95"][0], 3), -0.086, 1e-9)
    and near(r(g["mid"]["ci95"][1], 3), -0.016, 1e-9))
chk("atom -0.074", near(r(g["atom_0.5"]["mean"], 3), -0.074, 1e-9)
    and near(r(g["atom_0.5"]["ci95"][0], 3), -0.121, 1e-9)
    and near(r(g["atom_0.5"]["ci95"][1], 3), -0.025, 1e-9)
    and g["atom_0.5"]["n"] == 281)
st = f["strict_tertile_delta"]
chk("strict tertiles", near(r(st["low"]["mean"], 3), 0.003, 1e-9)
    and near(r(st["mid"]["mean"], 3), -0.052, 1e-9)
    and near(r(st["high"]["mean"], 3), -0.075, 1e-9))
sel = f["selection"]
chk("atom sel +0.074", near(r(sel["benefit_atom_capped"]["mean"], 3), 0.074, 1e-9)
    and near(r(sel["benefit_atom_capped"]["ci95"][0], 3), 0.027, 1e-9)
    and near(r(sel["benefit_atom_capped"]["ci95"][1], 3), 0.120, 1e-3))
chk("rule sel +0.080", near(r(sel["benefit_rule_capped"]["mean"], 3), 0.080, 1e-9)
    and sel["n_rule"] == 291 and near(r(sel["benefit_rule_capped"]["ci95"][0], 3), 0.040, 1e-9))

# ---- manuscript text -------------------------------------------------------
for frag in (
        "randomized-boundary tests at a common target",
        "the size-matched difference is $+0.0025$",
        "$[+0.0018, +0.0032]$", "positive in \\emph{all} $200$ splits",
        "evaluation-half sizes average $0.02514$",
        "target $0.025$ ($+0.0025$)",
        "quarter of a percentage point of power",
        "$+0.0128$ under $N(0,1)$", "$+0.0397$ under $N(0,0.5)$",
        "approximately size-adjusted at best",
        "size-matched exactly\nin expectation by randomized-boundary tests",
        "exactly size-matched}, the advantage is small",
        "small but consistent once size is matched exactly in expectation",
        "mechanism analysis of the \\emph{uncapped}",
        "average-rank Spearman correlation is $+0.14$",
        "$[+0.06,+0.21]$", "$+0.14/+0.15/+0.13$",
        "$281$-query tie atom at $0.5$",
        "($-0.074$, CI $[-0.121,-0.025]$)",
        "$+0.003/-0.052/-0.075$",
        "$+0.074$ (CI $[+0.027,+0.120]$)",
        "$+0.080$ (CI $[+0.040,+0.120]$)",
        "uncapped-architecture\n  mechanism diagnostic",
        "the two selective variants --- the final design (capped,\n  $2.877$)",
        "final design's own KS is $0.097$",
        "(both\nuncapped in this diagnostic)",
        "recomputed under the cap for the final\ndesign itself",
        "(dispersion dependence robust under repetition; see text)"):
    chk(f"M has {frag[:42]!r}", frag in M)
for frag in ("no average net power difference at matched size has been",
             "is established in either direction",
             "sign is not established once",
             "$95\\%$ split interval",
             "Selective (ours)"):
    chk(f"M lacks {frag!r}", frag not in M)
chk("builder lacks (ours)", "(ours)" not in FB)
# ---- supplement text --------------------------------------------------------
for frag in ("Round-10 additions (this revision)",
             "randomized\\_crossfit.json",
             "$+0.00247$ (SD $0.00035$", "$[+0.00184,+0.00322]$",
             "$N(0,1)$: $+0.01276$ $[+0.01218,+0.01359]$",
             "$+0.03970$ $[+0.03827,+0.04069]$",
             "round10\\_triage\\_capped.json",
             "$+0.136$ (CI $[+0.064,+0.207]$",
             "$+0.144/+0.151/+0.127$",
             "Wins$_{5}$", "$-0.273$", "$-0.016$",
             "Figure~2)", "exactly size-matched average advantage",
             "superseded by these common-target\nruns"):
    chk(f"S has {frag[:40]!r}", frag in S)
for frag in ("Figure~13", "sign not established"):
    chk(f"S lacks {frag!r}", frag not in S)

print(f"\n{ok} OK, {fail} FAIL")
sys.exit(1 if fail else 0)
