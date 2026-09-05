#!/usr/bin/env python3
"""Round-9/10 supplement figure.

FS1_crossfit_dist.pdf  (a) 20-bin histogram of the 200 deterministic-
                       threshold split estimates under the canonical
                       mixing, with mean/median marked and the 8-split
                       positive tail annotated (the round-9 granularity
                       diagnosis); (b) ECDFs of the deterministic (grey)
                       and randomized-boundary exactly size-matched
                       (green) canonical estimates on the identical 200
                       splits, plus the medians and central 95%
                       split-quantile ranges of the randomized runs
                       under N(0,1) and N(0,0.5) at the common target.

Inputs:  artifacts_round4/calibration/round9_crossfit_diag.json
         artifacts_round4/calibration/round8_crossfit_dist.json
         artifacts_round4/calibration/round10_randomized_crossfit.json
Outputs: manuscript/figures/ and results/figures/final_v4_selective/
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
OUTS = [ROOT / "results" / "figures" / "final_v4_selective",
        ROOT / "manuscript" / "figures"]
D9 = json.loads((ROOT / "artifacts_round4" / "calibration" /
                 "round9_crossfit_diag.json").read_text())
D8 = json.loads((ROOT / "artifacts_round4" / "calibration" /
                 "round8_crossfit_dist.json").read_text())
D10 = json.loads((ROOT / "artifacts_round4" / "calibration" /
                  "round10_randomized_crossfit.json").read_text())


def main():
    ests = np.array(D9["canonical_estimates"])
    assert len(ests) == 200
    assert abs(round(float(ests.mean()), 5) - D8["mean"]) < 1e-9
    assert abs(round(float(np.median(ests)), 5) - D8["median"]) < 1e-9

    fig, (ax, ax2) = plt.subplots(1, 2, figsize=(9.6, 3.6))
    ax.hist(ests, bins=20, color="#a6bddb", edgecolor="white")
    ax.axvline(D8["mean"], color="#1b9e77", lw=1.6,
               label=f"mean {D8['mean']:+.5f}")
    ax.axvline(D8["median"], color="#1b9e77", lw=1.6, ls="--",
               label=f"median {D8['median']:+.5f}")
    ax.axvline(0.0, color="#333", lw=0.9, ls=":")
    ax.annotate("8 splits: +0.0023 to +0.0057\n(no-borrowing threshold\n"
                "undersized on the evaluation half)",
                xy=(0.0045, 2.2), xytext=(0.0016, 26),
                fontsize=7.5, ha="left",
                arrowprops=dict(arrowstyle="->", lw=0.9, color="#555"))
    ax.set_xlabel("Size-adjusted difference per split (canonical mixing)")
    ax.set_ylabel("Splits (of 200)")
    ax.set_title("a  Repeated cross-fit: split distribution", fontsize=10)
    ax.legend(frameon=False, fontsize=8)

    # panel b: deterministic vs randomized-boundary (exactly size-matched)
    # canonical ECDFs on the identical splits, plus the randomized runs
    # under the narrower mixings at the common target (median dot,
    # central 95% split-quantile range).
    rnd = np.array(D10["canonical_estimates"])
    run = D10["runs"]["canonical@0.02513"]
    assert len(rnd) == 200
    assert abs(round(float(rnd.mean()), 5) - run["mean"]) < 1e-9
    x = np.sort(ests)
    y = np.arange(1, len(x) + 1) / len(x)
    ax2.step(x, y, where="post", color="#8a8f94", lw=1.4,
             label="deterministic thresholds (granularity-biased)")
    xr = np.sort(rnd)
    ax2.step(xr, np.arange(1, len(xr) + 1) / len(xr), where="post",
             color="#1b9e77", lw=1.9,
             label="randomized boundary, common target (exact size)")
    for name, col, yy in (("N(0,1)", "#7570b3", 0.55),
                          ("N(0,0.5)", "#d95f02", 0.35)):
        a = D10["runs"][f"{name}@0.02513"]
        lo, hi = a["q"]["0.025"], a["q"]["0.975"]
        ax2.plot([lo, hi], [yy, yy], color=col, lw=2.2)
        ax2.plot([a["median"]], [yy], "o", color=col, ms=5)
        ax2.text(hi + 0.001, yy, f"{name}: all 200 splits > 0",
                 va="center", fontsize=7.5, color=col)
    ax2.axvline(0.0, color="#333", lw=0.9, ls=":")
    ax2.set_xlim(-0.004, 0.055)
    ax2.set_ylim(0, 1.02)
    ax2.set_xlabel("Size-adjusted difference per split")
    ax2.set_ylabel("ECDF (canonical mixing)")
    ax2.set_title("b  Exact expected-size matching, common target\n"
                  "(randomized boundary; medians and central 95% ranges)",
                  fontsize=10)
    ax2.legend(frameon=False, fontsize=7.5, loc="lower right")
    for a_ in (ax, ax2):
        for spine in ("top", "right"):
            a_.spines[spine].set_visible(False)
    fig.tight_layout()
    for d in OUTS:
        d.mkdir(parents=True, exist_ok=True)
        fig.savefig(d / "FS1_crossfit_dist.pdf", bbox_inches="tight")
    print("saved FS1_crossfit_dist")


if __name__ == "__main__":
    main()
