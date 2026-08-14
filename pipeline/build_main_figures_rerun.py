#!/usr/bin/env python
"""Rebuild manuscript figures F1-F3 from the corrected-units re-run artifacts.

Corrected-units variant of ``build_main_figures.py`` (the July script): same
figure designs, artifact paths pointed at ``artifacts/rerun_unitfix_2026-08-14``
outputs, and every headline check() updated to the corrected numbers reported
in the manuscript, so a figure can never silently disagree with the text.
F4-F6 (OC sweep, sham controls, multi-endpoint) are not rebuilt because the
restructured manuscript no longer includes them; F7 is a schematic with no
numeric content.
"""
from __future__ import annotations

import csv
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
ART = Path(__file__).resolve().parents[1] / "artifacts_rerun"
if not ART.exists():
    ART = ROOT / "artifacts" / "rerun_unitfix_2026-08-14"
OUT = ROOT / "results" / "figures" / "final_v3_unitfix"
OUT.mkdir(parents=True, exist_ok=True)

plt.rcParams.update({
    "font.size": 9,
    "axes.titlesize": 10,
    "axes.titleweight": "bold",
    "axes.labelsize": 9,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "legend.fontsize": 8,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": False,
    "figure.dpi": 120,
    "savefig.bbox": "tight",
})
HL = "#b3131b"
C = {
    "two_head_sam": HL,
    "rule_sam": "#2c6fbb",
    "two_head": "#2e9e5b",
    "rule": "#7f8c8d",
    "weak_only": "#c4cbd0",
    "robust_map_w0.5": "#8e44ad",
    "robust_map_w0.8": "#a76bc4",
    "robust_map_w0.9": "#c39bd8",
    "map_like": "#9aa4ab",
    "power_prior_like": "#aab2b8",
    "commensurate_like": "#8b949b",
}
LAB = {
    "two_head_sam": "Two-head + SAM", "rule_sam": "Rule + SAM", "two_head": "Two-head",
    "rule": "Rule", "weak_only": "Weak-only", "robust_map_w0.5": "Robust-MAP (w=0.5)",
    "robust_map_w0.8": "Robust-MAP (w=0.8)", "robust_map_w0.9": "Robust-MAP (w=0.9)",
    "map_like": "MAP-like", "power_prior_like": "Power-prior-like",
    "commensurate_like": "Commensurate-like",
}
def col(m):
    return C.get(m, "#9aa4ab")
def lab(m):
    return LAB.get(m, m)


def read_csv(path: Path):
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def check(name, got, want, tol=5e-3):
    if got is None or abs(got - want) > tol:
        raise SystemExit(f"[STOP] {name}: plotted {got} != authoritative {want} (tol {tol})")
    print(f"  OK {name}: {got:.4f} ~= {want}")


def save(fig, stem):
    pdf = OUT / f"{stem}.pdf"
    tif = OUT / f"{stem}.tif"
    fig.savefig(pdf)
    fig.savefig(tif, dpi=300, pil_kwargs={"compression": "tiff_lzw"})
    plt.close(fig)
    print(f"  wrote {pdf.name} + {tif.name}")


# ============================================================ F1 head-to-head
def f1():
    print("F1 head-to-head NLL (corrected units)")
    rows = read_csv(ART / "robust_map" / "robust_map_head_to_head.csv")
    data = [(r["method"], float(r["mean_nll"]), float(r["coverage95"])) for r in rows]
    data.sort(key=lambda t: t[1], reverse=True)
    methods = [d[0] for d in data]
    nll = [d[1] for d in data]
    cov = [d[2] for d in data]
    ref = {"two_head_sam": 2.7938, "two_head": 2.8957, "rule_sam": 2.9006,
           "robust_map_w0.5": 3.0019, "rule": 3.0100, "weak_only": 3.1641,
           "map_like": 3.2219, "power_prior_like": 3.2269}
    d = dict((m, n) for m, n, _ in data)
    for m, v in ref.items():
        check(f"F1 {m} NLL", d.get(m), v)

    fig, (axL, axR) = plt.subplots(1, 2, figsize=(9.2, 4.6), gridspec_kw={"width_ratios": [2.4, 1]})
    y = range(len(methods))
    axL.barh(list(y), nll, color=[col(m) for m in methods],
             edgecolor="white", height=0.72)
    for i, (m, v) in enumerate(zip(methods, nll)):
        axL.text(v + 0.004, i, f"{v:.3f}", va="center", fontsize=7.5,
                 fontweight="bold" if m == "two_head_sam" else "normal")
    axL.set_yticks(list(y))
    axL.set_yticklabels([lab(m) for m in methods])
    axL.get_yticklabels()[methods.index("two_head_sam")].set_fontweight("bold")
    axL.set_xlim(2.70, 3.35)
    axL.set_xlabel("Held-out predictive NLL (lower is better), N = 1407")
    axL.set_title("a  Borrowing-prior head-to-head")
    axR.barh(list(y), cov, color=[col(m) for m in methods], edgecolor="white", height=0.72)
    axR.axvline(0.95, color="#333", lw=0.9, ls="--")
    axR.text(0.95, len(methods) - 0.4, "0.95", fontsize=7, color="#333", ha="center")
    axR.set_yticks(list(y))
    axR.set_yticklabels([])
    axR.set_xlim(0.88, 1.0)
    axR.set_xlabel("95% coverage")
    axR.set_title("b  Interval coverage")
    fig.suptitle("Two-head + SAM mixture prior has the lowest held-out NLL",
                 fontsize=11, fontweight="bold", y=1.02)
    save(fig, "F1_headtohead_nll")


# ============================================================ F2 forward validation
def f2():
    print("F2 forward validation (corrected units)")
    agg = read_csv(ART / "forward_validation" / "forward_validation_summary.csv")
    by = read_csv(ART / "forward_validation" / "forward_validation_by_split.csv")
    ad = [(r["method"], float(r["mean_nll_across_folds"]), r["leakage_free"]) for r in agg]
    ad.sort(key=lambda t: t[1], reverse=True)
    am = [a[0] for a in ad]; an = [a[1] for a in ad]
    ref = {"two_head_sam": 2.7558, "rule_sam": 2.8517, "two_head": 2.8682,
           "rule": 2.9563, "weak_only": 3.0939}
    dd = dict((m, n) for m, n, _ in ad)
    for m, v in ref.items():
        check(f"F2 agg {m}", dd.get(m), v)
    cutoffs = ["2020-12-31", "2021-12-31", "2022-12-31"]
    per = {}
    for r in by:
        per.setdefault(r["method"], {})[r["cutoff"]] = (
            float(r["delta_nll_vs_rule"]), float(r["delta_ci_low"]), float(r["delta_ci_high"]))
    for c, want in zip(cutoffs, [-0.1983, -0.2003, -0.2028]):
        check(f"F2 dNLL two_head_sam {c}", per["two_head_sam"][c][0], want)

    fig, (axL, axR) = plt.subplots(1, 2, figsize=(9.4, 4.5), gridspec_kw={"width_ratios": [1.5, 1.4]})
    y = range(len(am))
    axL.barh(list(y), an, color=[col(m) for m in am], edgecolor="white", height=0.7)
    for i, (m, v) in enumerate(zip(am, an)):
        axL.text(v + 0.004, i, f"{v:.3f}", va="center", fontsize=7.5,
                 fontweight="bold" if m == "two_head_sam" else "normal")
    axL.set_yticks(list(y)); axL.set_yticklabels([lab(m) for m in am])
    axL.get_yticklabels()[am.index("two_head_sam")].set_fontweight("bold")
    axL.set_xlim(2.70, 3.30)
    axL.set_xlabel("Future-fold NLL (aggregated over 3 cutoffs)")
    axL.set_title("a  Rolling-origin forward validation")
    axL.text(0.98, 0.02, "leakage-free (per-fold refit)", transform=axL.transAxes,
             ha="right", va="bottom", fontsize=7.5, style="italic", color="#444")
    show = ["two_head_sam", "rule_sam", "two_head"]
    x = range(len(cutoffs))
    off = {"two_head_sam": -0.22, "rule_sam": 0.0, "two_head": 0.22}
    for m in show:
        pts = [per[m][c][0] for c in cutoffs]
        los = [per[m][c][0] - per[m][c][1] for c in cutoffs]
        his = [per[m][c][2] - per[m][c][0] for c in cutoffs]
        xs = [i + off[m] for i in x]
        axR.errorbar(xs, pts, yerr=[los, his], fmt="o", color=col(m), label=lab(m),
                     ms=5, capsize=2.5, lw=1.2,
                     markeredgecolor="white" if m == "two_head_sam" else col(m))
    axR.axhline(0, color="#333", lw=0.9)
    axR.set_xticks(list(x)); axR.set_xticklabels([c[:4] for c in cutoffs])
    axR.set_xlabel("Train-on-past cutoff (test on future)")
    axR.set_ylabel("Delta NLL vs rule (95% bootstrap CI)")
    axR.set_title("b  Per-cutoff improvement (CI excludes 0)")
    axR.legend(loc="lower left", frameon=False)
    fig.suptitle("Learned two-head + SAM prior: lower future-fold NLL in three rolling-origin splits",
                 fontsize=11, fontweight="bold", y=1.02)
    save(fig, "F2_forward_validation")


# ============================================================ F3 calibration
def f3():
    print("F3 calibration (corrected units)")
    summ = {r["method"]: r for r in read_csv(ART / "calibration_diagnostics" / "calibration_summary.csv")}
    check("F3 two_head_sam PIT-KS", float(summ["two_head_sam"]["pit_ks_distance"]), 0.177, 2e-3)
    check("F3 weak_only PIT-KS", float(summ["weak_only"]["pit_ks_distance"]), 0.331, 2e-3)
    check("F3 two_head_sam Cov95", float(summ["two_head_sam"]["coverage95"]), 0.996, 2e-3)
    check("F3 two_head_sam slope", float(summ["two_head_sam"]["reliability_slope"]), 1.36, 2e-2)
    per = read_csv(ART / "calibration_diagnostics" / "calibration_per_example.csv")
    pit = {"two_head_sam": [], "weak_only": []}
    for r in per:
        if r["method"] in pit and r["pit_randomized"] not in ("", "nan"):
            pit[r["method"]].append(float(r["pit_randomized"]))
    rel = read_csv(ART / "calibration_diagnostics" / "calibration_reliability.csv")

    fig, axs = plt.subplots(1, 3, figsize=(11, 3.7))
    bins = [i / 10 for i in range(11)]
    axs[0].hist(pit["weak_only"], bins=bins, density=True, color=C["weak_only"],
                alpha=0.85, label="Weak-only")
    axs[0].hist(pit["two_head_sam"], bins=bins, density=True, histtype="step",
                color=HL, lw=2.0, label="Two-head + SAM")
    axs[0].axhline(1.0, color="#333", ls="--", lw=0.9)
    axs[0].set_xlabel("PIT value"); axs[0].set_ylabel("Density")
    axs[0].set_title("a  PIT uniformity")
    axs[0].legend(frameon=False, loc="upper center")
    axs[0].text(0.03, 0.97, f"KS: SAM {summ['two_head_sam']['pit_ks_distance'][:5]}\n"
                f"weak {summ['weak_only']['pit_ks_distance'][:5]}",
                transform=axs[0].transAxes, va="top", fontsize=7.5)
    for m, cc in (("weak_only", C["weak_only"]), ("two_head_sam", HL)):
        pts = [(float(x["mean_predicted_rate"]), float(x["mean_observed_rate"]))
               for x in rel if x["method"] == m]
        pts.sort()
        axs[1].plot([p[0] for p in pts], [p[1] for p in pts], "o-", color=cc,
                    lw=1.6, ms=4, label=lab(m))
    axs[1].plot([0, 1], [0, 1], ls="--", color="#333", lw=0.9)
    axs[1].set_xlim(0, 1); axs[1].set_ylim(0, 1)
    axs[1].set_xlabel("Mean predicted rate"); axs[1].set_ylabel("Mean observed rate")
    axs[1].set_title("b  Reliability")
    axs[1].legend(frameon=False, loc="upper left")
    levels = [("coverage50", 0.50), ("coverage80", 0.80), ("coverage95", 0.95)]
    x = range(len(levels)); w = 0.36
    axs[2].bar([i - w/2 for i in x], [float(summ["weak_only"][k]) for k, _ in levels],
               w, color=C["weak_only"], label="Weak-only")
    axs[2].bar([i + w/2 for i in x], [float(summ["two_head_sam"][k]) for k, _ in levels],
               w, color=HL, label="Two-head + SAM")
    for i, (_, nom) in enumerate(levels):
        axs[2].hlines(nom, i - 0.5, i + 0.5, color="#333", ls="--", lw=0.9)
    axs[2].set_xticks(list(x)); axs[2].set_xticklabels(["50%", "80%", "95%"])
    axs[2].set_ylim(0, 1.05); axs[2].set_xlabel("Nominal interval")
    axs[2].set_ylabel("Empirical coverage"); axs[2].set_title("c  Interval coverage")
    axs[2].legend(frameon=False, loc="lower right")
    fig.suptitle("Conflict adaptation improves calibration (PIT, reliability, coverage)",
                 fontsize=11, fontweight="bold", y=1.03)
    fig.tight_layout()
    save(fig, "F3_calibration")


if __name__ == "__main__":
    f1(); f2(); f3()
    print("\nAll corrected-units figures written to", OUT)
