#!/usr/bin/env python3
"""Round-12 programmatic verification (numbers, wording, figure text,
page geometry)."""
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
M = (ROOT / "manuscript" / "manuscript_full_draft_pharm_stats.tex").read_text()
S = (ROOT / "manuscript" / "supplement.tex").read_text()
FIG = (ROOT / "scripts" / "build_round9_figures.py").read_text()
E = json.load(open(ROOT / "artifacts_round4" / "calibration" / "round12_eval_sizes.json"))
R11 = json.load(open(ROOT / "artifacts_round4" / "calibration" / "round11_inference_scope.json"))

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


# ---- Table 8 upper block rows against artifacts -----------------------------
rows = {
    "canonical@0.02513": ("+0.0025", "[+0.0018,+0.0032]", "$0.02514$/$0.02514$"),
    "canonical@0.025": ("+0.0025", "[+0.0017,+0.0032]", "$0.02502$/$0.02501$"),
    "N(0,1)@0.02513": ("+0.0128", "[+0.0122,+0.0136]", "$0.02514$/$0.02513$"),
    "N(0,0.5)@0.02513": ("+0.0397", "[+0.0383,+0.0407]", "$0.02515$/$0.02515$"),
    "N(+0.124,1)@0.02513": ("+0.0117", "[+0.0111,+0.0125]", "$0.02514$/$0.02513$"),
    "N(+0.124,0.5)@0.02513": ("+0.0362", "[+0.0351,+0.0372]", "$0.02515$/$0.02514$"),
}
for k, (m, rng, sz) in rows.items():
    e, q = E["runs"][k], R11["runs"][k]["est"]["q"]
    lo, hi = rng.strip("[]").split(",")
    chk(f"{k} mean", r(e["mean"], 4) == float(m))
    chk(f"{k} range", r(q["0.025"], 4) == float(lo) and r(q["0.975"], 4) == float(hi))
    sw, sm = [float(x.strip("$")) for x in sz.split("/")]
    chk(f"{k} sizes", e["eval_size_weak_mean"] == sw and e["eval_size_cap50_mean"] == sm)
    chk(f"{k} all positive", e["frac_positive"] == 1.0)
    flat = " ".join(M.split())
    chk(f"{k} row in tex", f"${m}$ & ${rng}$ &" in flat and sz in flat)

# ---- wording / provenance ---------------------------------------------------
for frag in ("of that legacy interpolated analysis $[+0.0006, +0.0044]$",
             "$[+0.0005, +0.0045]$ for the randomized estimator above",
             "legacy\ninterpolated estimator: $95\\%$ interval $[+0.0006, +0.0044]$",
             "(randomized estimator, common target:\n$[+0.0005, +0.0045]$",
             "accompanies the submission as supplementary\nmaterial",
             "upon acceptance the corresponding tagged version will be\ndeposited in Zenodo",
             "Legacy full-batch deterministic-threshold",
             "Upper table:\nthe main analysis", "loc.\\ held"):
    chk(f"M has {frag[:40]!r}", frag in M)
for frag in ("DOI: to be assigned", "\\multicolumn{5}{l}{corpus mixing $+0.0025$"):
    chk(f"M lacks {frag!r}", frag not in M)
chk("figure builder title", "Randomized-boundary size standardization, common target" in FIG)
chk("figure builder legend", "(exact size)" not in FIG and "Exact expected-size" not in FIG)
# the PDF figure itself carries the new text
txt = subprocess.run(["pdftotext", str(ROOT / "manuscript" / "figures" / "FS1_crossfit_dist.pdf"), "-"],
                     capture_output=True, text=True).stdout
chk("FS1 pdf text new title", "Randomized-boundary size standardization" in txt)
chk("FS1 pdf text no exact", "exact size" not in txt and "Exact expected" not in txt)

# ---- page geometry ------------------------------------------------------------
geo = subprocess.run([sys.executable, str(ROOT / "scripts" / "check_page_geometry.py"),
                      str(ROOT / "manuscript" / "manuscript_full_draft_pharm_stats.pdf"),
                      str(ROOT / "manuscript" / "supplement.pdf")],
                     capture_output=True, text=True)
chk("page geometry OK", geo.returncode == 0)
# numeric overfull parse of the logs
for log in ("manuscript_full_draft_pharm_stats.log", "supplement.log"):
    txt = (ROOT / "manuscript" / log).read_text(errors="ignore")
    import re
    vals = [float(v) for v in re.findall(r"Overfull \\hbox \(([\d.]+)pt", txt)]
    chk(f"{log} max overfull < 10pt", (max(vals) if vals else 0.0) < 10.0)

print(f"\n{ok} OK, {fail} FAIL")
sys.exit(1 if fail else 0)
