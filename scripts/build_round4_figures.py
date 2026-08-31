#!/usr/bin/env python3
"""Round-4 figures.

F13_primary.pdf   Forest plot of the primary analysis (registry-availability
                  proxy x title-screened ORR subset): per-window deltas of
                  the final design, pooled rows for final design / uncapped
                  ablation / frozen canonical control, engaged subgroups.
F14_profiles.pdf  Mixture-calibration operating profile: type I error and
                  power across the two-sided uniform drift grid at each
                  design's mixture-calibrated threshold, with the empirical
                  drift weights shown underneath.

Inputs: artifacts_round4/real_capped/summary.json,
        artifacts_round4/calibration/mixture_calibration.json
Outputs land in results/figures/final_v4_selective/ and manuscript/figures/.
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
OUTS = [ROOT / "results" / "figures" / "final_v4_selective",
        ROOT / "manuscript" / "figures"]
HL = "#1b9e77"
BAD = "#d95f02"
GRY = "#666666"


def save(fig, name):
    for d in OUTS:
        d.mkdir(parents=True, exist_ok=True)
        fig.savefig(d / f"{name}.pdf", bbox_inches="tight")
    plt.close(fig)
    print("saved", name)


def f13():
    s = json.loads((ROOT / "artifacts_round4" / "real_capped" /
                    "summary.json").read_text())
    B = s["tiers"]["strictB"]
    tab = B["table_by_window"]
    rows = []
    for w, lab in (("w1", "w1 (2021)  n=128"), ("w2", "w2 (2022)  n=155"),
                   ("w3", "w3 (>2022)  n=306")):
        d = tab[w]["delta_0.5"]
        rows.append((lab, d["mean"], d["ci95"], HL))
    rows.append((None, None, None, None))
    d = B["cap_0.5"]["pooled"]
    rows.append(("Final design, pooled  n=589", d["mean"], d["ci95"], HL))
    d = B["cap_0.5"]["subgroup_with_donor"]
    rows.append(("Final design, engaged  n=119", d["mean"], d["ci95"], HL))
    d = B["cap_none"]["pooled"]
    rows.append(("Uncapped ablation, pooled", d["mean"], d["ci95"], BAD))
    d = B["cap_none"]["subgroup_with_donor"]
    rows.append(("Uncapped ablation, engaged", d["mean"], d["ci95"], BAD))
    rows.append(("Frozen canonical control, pooled", -0.002,
                 [-0.015, 0.010], GRY))

    fig, ax = plt.subplots(figsize=(7.6, 3.9))
    fig.subplots_adjust(left=0.36)
    ys = list(range(len(rows), 0, -1))
    for y, (lab, m, ci, col) in zip(ys, rows):
        if lab is None:
            continue
        ax.plot(ci, [y, y], color=col, lw=1.8)
        ax.plot([m], [y], "o", color=col, ms=5)
        ax.annotate(lab, xy=(0, y), xycoords=("axes fraction", "data"),
                    xytext=(-8, 0), textcoords="offset points",
                    ha="right", va="center", fontsize=8.5)
    ax.axvline(0.0, color="#333", ls=":", lw=1.0)
    ax.set_yticks([])
    ax.set_ylim(0.3, len(rows) + 0.7)
    ax.set_xlim(-0.11, 0.36)
    ax.set_xlabel("Paired $\\Delta$NLL vs availability-consistent EB "
                  "(negative favours the design)")
    for spine in ("left", "top", "right"):
        ax.spines[spine].set_visible(False)
    ax.set_title("Primary analysis: capped neutral, uncapped harmful\n"
                 "(post hoc exploratory redesign comparison; "
                 "registry-availability proxy $\\times$ title-screened "
                 "binary-response subset)", fontsize=10,
                 fontweight="bold")
    save(fig, "F13_primary")


def f14():
    r = json.loads((ROOT / "artifacts_round4" / "calibration" /
                    "mixture_calibration.json").read_text())
    v = json.loads((ROOT / "artifacts_round4" / "calibration" /
                    "validation_report.json").read_text())
    W = {float(k): vv for k, vv in r["mixture_weights"].items()}
    shifts = sorted(W)
    fig, axs = plt.subplots(
        3, 1, figsize=(7.2, 6.4), sharex=True,
        gridspec_kw={"height_ratios": [3, 3, 1.2]})
    series = [("weak_only", "no borrowing", GRY, ":"),
              ("eb_only", "EB anchor only", "#7570b3", "--"),
              ("selective_cap50", "final design (cap 0.5)", HL, "-"),
              ("selective_cap50_sam", "cap 0.5 + SAM (sensitivity)",
               "#66a61e", "-.")]
    for m, lab, col, ls in series:
        e = v["methods"][m]
        t1 = [e["t1_profile"][str(s)] for s in shifts]
        pw = [e["power_profile"][str(s)] for s in shifts]
        axs[0].plot(shifts, t1, ls, color=col, lw=1.7, label=lab)
        axs[1].plot(shifts, pw, ls, color=col, lw=1.7, label=lab)
    axs[0].axhline(0.025, color="#333", ls=":", lw=0.9)
    axs[0].axhline(0.05, color=BAD, ls=":", lw=0.9)
    axs[0].text(-1.48, 0.051, "pre-specified ceiling 0.05", fontsize=7.5,
                color=BAD, va="bottom")
    axs[0].set_ylabel("Null-world rejection rate")
    axs[0].legend(frameon=False, fontsize=8, ncol=2)
    axs[1].set_ylabel("Alt-world rejection rate")
    axs[2].bar(shifts, [W[s] for s in shifts], width=0.21, color="#a6bddb")
    axs[2].set_ylabel("Drift\nweight")
    axs[2].set_xlabel("Uniform donor drift (logit shift), two-sided grid")
    for a in axs:
        for spine in ("top", "right"):
            a.spines[spine].set_visible(False)
    fig.suptitle("Operating profile at the frozen mixture-calibrated thresholds\n"
                 "(independent validation batch, 10,000 replicates/cell; "
                 "drift weights below)",
                 fontsize=10.5, fontweight="bold")
    fig.tight_layout()
    save(fig, "F14_profiles")


if __name__ == "__main__":
    f13()
    f14()
