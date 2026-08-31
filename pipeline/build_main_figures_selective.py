#!/usr/bin/env python
"""Build manuscript figures F1-F3 and F12 for the selective-borrowing revision.

Every plotted headline value is asserted against the numbers reported in the
rewritten manuscript (EB-referenced tables) before any file is written, so a
figure cannot silently disagree with the text. Inputs are the per-example
evaluation rows produced for the rewrite
(artifacts_pathfinding/rewrite_numbers_perexample.json) plus the pipeline
modules for calibration primitives.

F1  EB-referenced head-to-head on the clean test split (design-time priors).
F2  Leakage-free forward validation on disjoint future windows.
F3  Calibration: PIT, reliability, prediction-interval coverage (test split).
F12 Learned borrowing triage: borrowed mass versus candidate-set information.
"""
from __future__ import annotations

import json
import math
import random
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "pipeline"))
import mixture_prior  # noqa: E402
from run_borrowing_baseline_comparison import prior_for_method  # noqa: E402
from run_calibration_diagnostics import pit_values, central_interval  # noqa: E402
from two_head_selective import load_selective_artifact, selective_details  # noqa: E402

OUT = ROOT / "results" / "figures" / "final_v4_selective"
OUT.mkdir(parents=True, exist_ok=True)

plt.rcParams.update({
    "font.size": 9, "axes.titlesize": 10, "axes.titleweight": "bold",
    "axes.labelsize": 9, "xtick.labelsize": 8, "ytick.labelsize": 8,
    "legend.fontsize": 8, "axes.spines.top": False, "axes.spines.right": False,
    "figure.dpi": 120, "savefig.bbox": "tight",
})
HL = "#b3131b"
C = {
    "selective": HL, "eb": "#1a1a1a", "eb_strat": "#5b5b5b",
    "two_head_old": "#2e9e5b", "rule": "#7f8c8d", "weak_only": "#c4cbd0",
    "robust_map_w0.5": "#8e44ad", "robust_map_w0.9": "#c39bd8",
    "map_like": "#9aa4ab", "power_prior_like": "#aab2b8",
    "commensurate_like": "#8b949b", "fixed_discount": "#a3866a",
}
LAB = {
    "selective": "Selective (ours)", "eb": "EB (intercept-only)",
    "eb_strat": "EB (disease-stratified)", "two_head_old": "Two-head (fixed budget)",
    "rule": "Rule", "weak_only": "Weak-only Beta(1,1)",
    "robust_map_w0.5": "Robust-MAP (w=0.5)", "robust_map_w0.9": "Robust-MAP (w=0.9)",
    "map_like": "MAP-like", "power_prior_like": "Power-prior-like",
    "commensurate_like": "Commensurate-like", "fixed_discount": "Fixed-discount",
}

DATA = json.load(open(ROOT / "artifacts_pathfinding" / "rewrite_numbers_perexample.json"))
TEST, FWD = DATA["test"], DATA["fwd"]
BASE = [json.loads(l) for l in open(ROOT / "artifacts_rerun" / "examples_unitfix_with_true_dates.jsonl")]


def check(name, got, want, tol=5e-3):
    if got is None or abs(got - want) > tol:
        raise SystemExit(f"[STOP] {name}: plotted {got} != manuscript {want} (tol {tol})")
    print(f"  OK {name}: {got:.4f} ~= {want}")


def save(fig, stem):
    fig.savefig(OUT / f"{stem}.pdf")
    fig.savefig(OUT / f"{stem}.tif", dpi=300, pil_kwargs={"compression": "tiff_lzw"})
    plt.close(fig)
    print(f"  wrote {stem}.pdf/.tif")


def mean(rows, m):
    return sum(r[m] for r in rows) / len(rows)


def paired(rows, m, ref="eb", nboot=4000, seed=23):
    d = [r[m] - r[ref] for r in rows]
    n = len(d)
    mu = sum(d) / n
    random.seed(seed)
    bs = []
    for _ in range(nboot):
        s = [d[random.randrange(n)] for _ in range(n)]
        bs.append(sum(s) / n)
    bs.sort()
    return mu, bs[int(0.025 * nboot)], bs[int(0.975 * nboot) - 1]


DESIGN = ["selective", "eb", "eb_strat", "two_head_old", "fixed_discount",
          "robust_map_w0.5", "rule", "commensurate_like", "weak_only",
          "robust_map_w0.9", "map_like", "power_prior_like"]


def barh_panel(ax, rows, methods, ref_val, xlab):
    vals = sorted(((m, mean(rows, m)) for m in methods), key=lambda t: t[1], reverse=True)
    names = [v[0] for v in vals]
    xs = [v[1] for v in vals]
    y = range(len(names))
    ax.barh(list(y), xs, color=[C.get(m, "#9aa4ab") for m in names], edgecolor="white", height=0.72)
    for i, (m, v) in enumerate(vals):
        ax.text(v + 0.004, i, f"{v:.3f}", va="center", fontsize=7.5,
                fontweight="bold" if m == "selective" else "normal")
    ax.axvline(ref_val, color="#333", lw=1.0, ls="--")
    ax.text(ref_val, len(names) - 0.25, " EB ref.", fontsize=7, color="#333", va="bottom")
    ax.set_yticks(list(y))
    ax.set_yticklabels([LAB.get(m, m) for m in names])
    ax.get_yticklabels()[names.index("selective")].set_fontweight("bold")
    lo = min(xs) - 0.05
    ax.set_xlim(lo, max(xs) + 0.06)
    ax.set_xlabel(xlab)


# ------------------------------------------------------------------ F1
def f1():
    print("F1 EB-referenced head-to-head (test split)")
    check("F1 selective", mean(TEST, "selective"), 2.8716)
    check("F1 eb", mean(TEST, "eb"), 2.9012)
    check("F1 two_head_old", mean(TEST, "two_head_old"), 2.9297)
    check("F1 rule", mean(TEST, "rule"), 3.0542)
    check("F1 robust w0.5", mean(TEST, "robust_map_w0.5"), 3.0427)
    d, lo, hi = paired(TEST, "selective")
    check("F1 delta selective", d, -0.0295, 2e-3)

    fig, (axL, axR) = plt.subplots(1, 2, figsize=(9.6, 4.8), gridspec_kw={"width_ratios": [2.1, 1.3]})
    barh_panel(axL, TEST, DESIGN, mean(TEST, "eb"),
               "Mean predictive NLL on clean test split (n = 281, lower is better)")
    axL.set_title("a  Design-time priors vs the EB reference")
    show = ["selective", "eb_strat", "two_head_old", "robust_map_w0.5", "rule"]
    ys = range(len(show))
    for i, m in enumerate(show):
        d, lo, hi = paired(TEST, m)
        axR.errorbar([d], [i], xerr=[[d - lo], [hi - d]], fmt="o", color=C.get(m, "#888"),
                     ms=5.5, capsize=3, lw=1.4)
        axR.text(hi + 0.006, i, f"{d:+.3f}", va="center", fontsize=7.5)
    axR.axvline(0, color="#333", lw=1.0)
    axR.set_yticks(list(ys))
    axR.set_yticklabels([LAB.get(m, m) for m in show])
    axR.invert_yaxis()
    axR.set_xlabel("Paired $\\Delta$NLL vs EB (95% bootstrap CI)")
    axR.set_title("b  Paired deltas vs EB")
    fig.suptitle("Only the selective prior improves on the intercept-only EB reference",
                 fontsize=11, fontweight="bold", y=1.02)
    save(fig, "F1_headtohead_nll")


# ------------------------------------------------------------------ F2
def f2():
    print("F2 forward validation (disjoint windows)")
    check("F2 selective", mean(FWD, "selective"), 2.8175)
    check("F2 eb", mean(FWD, "eb"), 2.8440)
    check("F2 eb_strat", mean(FWD, "eb_strat"), 2.8152)
    check("F2 two_head_old", mean(FWD, "two_head_old"), 2.8839)
    d_all, _, _ = paired(FWD, "selective")
    check("F2 pooled delta", d_all, -0.0266, 2e-3)

    def sortdate(i):
        return (BASE[i].get("query_metadata") or {}).get("temporal_sort_date") or ""

    windows = [("2020-12-31", "2021-12-31"), ("2021-12-31", "2022-12-31"), ("2022-12-31", "9999-99-99")]
    per = []
    for lo_, hi_ in windows:
        sub = [r for r in FWD if lo_ < sortdate(r["i"]) <= hi_]
        per.append((f"({lo_[:4]},{hi_[:4] if hi_ != '9999-99-99' else 'now'}]", len(sub), *paired(sub, "selective")))
    check("F2 window3 delta", per[2][2], -0.0461, 2e-3)

    fig, (axL, axR) = plt.subplots(1, 2, figsize=(9.6, 4.8), gridspec_kw={"width_ratios": [2.1, 1.3]})
    barh_panel(axL, FWD, DESIGN, mean(FWD, "eb"),
               "Pooled future-window NLL (n = 775, lower is better)")
    axL.set_title("a  Disjoint-window forward validation")
    axL.text(0.98, 0.02, "leakage-free (past-only refits)", transform=axL.transAxes,
             ha="right", va="bottom", fontsize=7.5, style="italic", color="#444")
    x = range(len(per))
    for i, (label, n, d, lo, hi) in enumerate(per):
        axR.errorbar([i], [d], yerr=[[d - lo], [hi - d]], fmt="o", color=HL, ms=6, capsize=3, lw=1.5)
        axR.text(i, hi + 0.004, f"n={n}", ha="center", fontsize=7)
    axR.axhline(0, color="#333", lw=1.0)
    axR.set_xticks(list(x))
    axR.set_xticklabels([p[0] for p in per])
    axR.set_xlabel("Future window (primary completion date)")
    axR.set_ylabel("Selective vs EB: paired $\\Delta$NLL (95% CI)")
    axR.set_title("b  Per-window paired delta")
    fig.suptitle("Selective borrowing matches or exceeds the EB reference out of time,\n"
                 "significantly in the best-trained window", fontsize=11, fontweight="bold", y=1.05)
    save(fig, "F2_forward_validation")


# ------------------------------------------------------------------ F3
def f3():
    print("F3 calibration (test split)")
    sel_model, sel_anchor = load_selective_artifact(
        ROOT / "artifacts_pathfinding" / "selective" / "selective_model_primary.pt")
    gm = [json.loads(l) for l in open(ROOT / "artifacts_rerun" / "examples_unitfix_with_model.jsonl")]
    eb = tuple(DATA["eb_train"])

    def pmf_of(lam0, anchor, comps, n0):
        pmf = [mixture_prior.mixture_predictive_probability(
            y, n0, lam0, anchor[0], anchor[1], comps) for y in range(n0 + 1)]
        t = math.fsum(pmf)
        return [v / t for v in pmf]

    def prior_of(i, method):
        e = BASE[i]
        y0, n0 = int(e["query"]["count"]), int(e["query"]["denominator"])
        if method == "eb":
            return 1.0, eb, [], y0, n0
        if method == "selective":
            det = selective_details(sel_model, e, sel_anchor)
            comps = [{"alpha": 1.0 + d * c["count"], "beta": 1.0 + d * (c["denominator"] - c["count"]),
                      "lambda": l} for c, l, d in zip(e["components"], det["lambda_i"], det["discount_i"])]
            return det["lambda0"], sel_anchor, comps, y0, n0
        pr = prior_for_method(gm[i], method)
        comps = [{"alpha": c["alpha"], "beta": c["beta"],
                  "lambda": c.get("lambda_active", c.get("lambda", 0.0))}
                 for c in pr["components"]]
        return pr["lambda_0"], (1.0, 1.0), comps, y0, n0

    methods = ["eb", "selective", "two_head", "weak_only"]
    fig_lab = {"eb": "EB reference", "selective": "Selective (ours)",
               "two_head": "Two-head (fixed budget)", "weak_only": "Weak-only Beta(1,1)"}
    fig_col = {"eb": "#1a1a1a", "selective": HL, "two_head": "#2e9e5b", "weak_only": "#c4cbd0"}
    ev_idx = [r["i"] for r in TEST]
    pits, covs, rel = {}, {}, {}
    for m in methods:
        rng = random.Random(20260707)
        pv, cv = [], {0.5: 0, 0.8: 0, 0.95: 0}
        pairs = []
        for i in ev_idx:
            lam0, anchor, comps, y0, n0 = prior_of(i, m)
            pmf = pmf_of(lam0, anchor, comps, n0)
            rp, _ = pit_values(pmf, y0, rng)
            pv.append(rp)
            for lvl in cv:
                a, b = central_interval(pmf, lvl)
                cv[lvl] += (a <= y0 <= b)
            pred = sum(y * p for y, p in enumerate(pmf)) / n0
            pairs.append((pred, y0 / n0))
        pits[m], rel[m] = pv, pairs
        covs[m] = {lvl: cv[lvl] / len(ev_idx) for lvl in cv}

    def ks(v):
        s = sorted(v)
        n = len(s)
        return max(max(abs((k + 1) / n - x), abs(k / n - x)) for k, x in enumerate(s))

    check("F3 selective KS", ks(pits["selective"]), 0.094, 4e-3)
    check("F3 eb KS", ks(pits["eb"]), 0.035, 4e-3)
    check("F3 selective cov95", covs["selective"][0.95], 0.975, 4e-3)
    check("F3 weak cov50", covs["weak_only"][0.5], 0.459, 4e-3)

    fig, axs = plt.subplots(1, 3, figsize=(11, 3.7))
    bins = [i / 10 for i in range(11)]
    axs[0].hist(pits["weak_only"], bins=bins, density=True, color=fig_col["weak_only"],
                alpha=0.85, label=fig_lab["weak_only"])
    axs[0].hist(pits["eb"], bins=bins, density=True, histtype="step",
                color=fig_col["eb"], lw=1.6, ls="--", label=fig_lab["eb"])
    axs[0].hist(pits["selective"], bins=bins, density=True, histtype="step",
                color=HL, lw=2.0, label=fig_lab["selective"])
    axs[0].axhline(1.0, color="#333", ls=":", lw=0.9)
    axs[0].set_xlabel("Randomized PIT")
    axs[0].set_ylabel("Density")
    axs[0].set_title("a  PIT uniformity")
    axs[0].legend(frameon=False, loc="upper center", fontsize=7)
    # (b) reliability: binned means
    for m in ["weak_only", "eb", "selective"]:
        pairs = sorted(rel[m])
        k = 8
        step = max(1, len(pairs) // k)
        xs, ys = [], []
        for j in range(0, len(pairs), step):
            chunk = pairs[j:j + step]
            xs.append(sum(p for p, _ in chunk) / len(chunk))
            ys.append(sum(o for _, o in chunk) / len(chunk))
        axs[1].plot(xs, ys, "o-", color=fig_col[m], lw=1.5, ms=4,
                    ls="--" if m == "eb" else "-", label=fig_lab[m])
    axs[1].plot([0, 0.8], [0, 0.8], ls=":", color="#333", lw=0.9)
    axs[1].set_xlim(0, 0.8)
    axs[1].set_ylim(0, 0.8)
    axs[1].set_xlabel("Mean predicted rate (binned)")
    axs[1].set_ylabel("Mean observed rate")
    axs[1].set_title("b  Reliability")
    axs[1].legend(frameon=False, loc="upper left", fontsize=7)
    # (c) coverage
    levels = [0.5, 0.8, 0.95]
    x = range(len(levels))
    w = 0.27
    offs = {"weak_only": -w, "eb": 0.0, "selective": w}
    for m in ["weak_only", "eb", "selective"]:
        axs[2].bar([i + offs[m] for i in x], [covs[m][l] for l in levels], w,
                   color=fig_col[m], label=fig_lab[m])
    for i, l in enumerate(levels):
        axs[2].hlines(l, i - 0.45, i + 0.45, color="#333", ls="--", lw=0.9)
    axs[2].set_xticks(list(x))
    axs[2].set_xticklabels(["50%", "80%", "95%"])
    axs[2].set_ylim(0, 1.05)
    axs[2].set_xlabel("Nominal prediction interval")
    axs[2].set_ylabel("Empirical coverage")
    axs[2].set_title("c  Interval coverage")
    axs[2].legend(frameon=False, loc="lower right", fontsize=7)
    fig.suptitle("The selective prior approaches EB-level calibration at lower NLL",
                 fontsize=11, fontweight="bold", y=1.03)
    fig.tight_layout()
    save(fig, "F3_calibration")


# ------------------------------------------------------------------ F12
def f12():
    print("F12 learned borrowing triage")
    import statistics
    lo_m = statistics.mean(r["sel_mass"] for r in FWD if r["info"] < 8)
    hi_m = statistics.mean(r["sel_mass"] for r in FWD if r["info"] >= 8)
    check("F12 mass info<8", lo_m, 0.310, 3e-3)
    check("F12 mass info>=8", hi_m, 0.585, 3e-3)

    fig, ax = plt.subplots(figsize=(6.4, 4.2))
    xs = [max(r["info"], 0.05) for r in FWD]
    ys = [r["sel_mass"] for r in FWD]
    ax.scatter(xs, ys, s=9, alpha=0.35, color=HL, edgecolors="none")
    ax.set_xscale("log")
    ax.axvline(8, color="#333", ls="--", lw=0.9)
    ax.text(8, 1.01, " pre-specified $I{=}8$", fontsize=7.5, color="#333")
    ax.hlines(0.79, ax.get_xlim()[0], ax.get_xlim()[1], color="#2e9e5b", ls=":", lw=1.4)
    ax.text(0.06, 0.805, "fixed-budget predecessor ($\\approx$0.79 forced)", fontsize=7.5, color="#2e9e5b")
    for lab_, xv, yv in [("mean 0.31", 1.0, lo_m), ("mean 0.59", 30.0, hi_m)]:
        ax.plot([xv / 2.2, xv * 2.2], [yv, yv], color="#1a1a1a", lw=2.0)
        ax.text(xv, yv - 0.07, lab_, ha="center", fontsize=8, fontweight="bold")
    ax.set_xlabel("Design-time candidate-set information $I=\\sum_i \\mathrm{gate}_i\\, a_i^{\\mathrm{rule}} n_i$ (log scale)")
    ax.set_ylabel("Learned borrowed mass $1-\\lambda_0$")
    ax.set_ylim(-0.02, 1.05)
    ax.set_title("Learned total borrowing tracks candidate-set information\n(Spearman +0.59, forward windows)",
                 fontsize=10, fontweight="bold")
    save(fig, "F12_triage")


if __name__ == "__main__":
    f1()
    f2()
    f3()
    f12()
    print("\nAll selective-revision figures written to", OUT)
