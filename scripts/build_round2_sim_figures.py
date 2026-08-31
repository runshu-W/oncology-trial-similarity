#!/usr/bin/env python3
"""Rebuild F8-F11 (simulation figures) from the round-2 run with selective rows.

Adapted from simulation/gold_standard/build_outputs.py; same visual language,
method lists extended to the selective architecture and the EB-only reference,
discrimination shown under the primary (parameter-level) labels.
"""
from __future__ import annotations

import csv
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
GS = ROOT / "artifacts_round2" / "goldsim"
OUT_MAN = ROOT / "manuscript" / "figures"
OUT_RES = ROOT / "results" / "figures" / "final_v4_selective"
NOMINAL = 0.025

LABEL = {
    "weak_only": "No borrowing", "eb_only": "EB-only reference",
    "internal_only": "Internal control only",
    "rule": "Rule mixture", "rule_sam": "Rule + SAM",
    "two_head": "Two-head (no pro.)", "two_head_pro": "Two-head + pro.",
    "two_head_pro_sam": "Two-head + pro. + SAM",
    "two_head_pro_fixdisc": "Two-head, fixed disc.",
    "selective_pro": "Selective", "selective_base": "Selective (no pro.)",
    "selective_pro_fixdisc": "Selective, fixed disc.",
    "selective_pro_sam": "Selective + SAM",
    "robust_map_w0.5": "Robust-MAP (w=0.5)", "uip_dirichlet": "UIP-Dirichlet",
    "uip_js": "UIP-JS (uses outcome)", "power_prior": "Power prior",
    "pooling": "Full pooling",
}
SEL_COLOR = "#b01c2e"
SAM_COLOR = "#e08214"
TH_COLOR = "#1f4e79"


def read_csv(p):
    with open(p, newline="", encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def f(row, key):
    try:
        return float(row[key])
    except (KeyError, TypeError, ValueError):
        return np.nan


def _series(rows, method, xkey, ykey, xs):
    v, e = [], []
    for x in xs:
        r = next((r for r in rows if r["method"] == method and f(r, xkey) == x), None)
        v.append(f(r, ykey) if r else np.nan)
        e.append(f(r, ykey + "_mcse") if r else np.nan)
    return v, e


def style_for(m):
    if m == "selective_pro":
        return dict(color=SEL_COLOR, lw=2.6, zorder=6)
    if m == "selective_pro_sam":
        return dict(color=SAM_COLOR, lw=2.2, zorder=5)
    if m == "two_head_pro":
        return dict(color=TH_COLOR, lw=1.8, zorder=4)
    if m in ("weak_only", "internal_only"):
        return dict(color="black", lw=1.2, zorder=3)
    if m == "eb_only":
        return dict(color="#4d9221", lw=1.4, zorder=3)
    if m == "uip_js":
        return dict(color="#a6761d", lw=1.0, zorder=2)
    return dict(lw=1.0, zorder=2)


def fig_design_oc(design, path):
    show = ["weak_only", "eb_only", "rule", "two_head_pro", "two_head_pro_sam",
            "selective_pro", "selective_pro_fixdisc", "selective_pro_sam",
            "robust_map_w0.5", "uip_js"]
    nulls = [r for r in design if r["world"] == "null"]
    alts = [r for r in design if r["world"] == "alt"]
    shifts = sorted({f(r, "conflict_shift") for r in nulls})
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.8))
    for ax, rows, title in [
            (axes[0], nulls, "Type I error (null world, $\\theta_0=0.20$)"),
            (axes[1], alts, "Power (alternative world, $\\theta_0=0.35$)")]:
        for m in show:
            v, e = _series(rows, m, "conflict_shift", "reject_rate", shifts)
            ax.errorbar(shifts, v, yerr=e, marker="o", ms=4, capsize=2,
                        label=LABEL[m], **style_for(m))
        ax.set_xlabel("Historical-vs-current logit shift")
        ax.set_ylabel("Rejection rate")
        ax.set_title(title, fontsize=11)
        ax.grid(alpha=0.25)
    axes[0].axhline(NOMINAL, color="crimson", ls="--", lw=1.2)
    axes[0].annotate("nominal 0.025", (0.02, NOMINAL + 0.005), color="crimson", fontsize=8)
    ref = next((f(r, "reject_rate") for r in nulls
                if r["method"] == "weak_only" and f(r, "conflict_shift") == 0.0), np.nan)
    if ref == ref:
        axes[0].axhline(ref, color="grey", ls=":", lw=1.2)
    axes[0].set_ylim(0, 0.26)
    axes[1].legend(fontsize=7.5, loc="lower right", ncol=2)
    fig.suptitle("Operating characteristics against a fixed design null. The standalone "
                 "selective prior's type I error escapes under uniform conflict;\n"
                 "the anchored SAM companion mitigates the inflation "
                 "(full pooling and power prior omitted: type I error 0.48-0.96)",
                 y=1.05, fontsize=11)
    fig.tight_layout()
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print("wrote", path)


def fig_ess_response(rows, path):
    bins = ["ess_d000_005", "ess_d005_010", "ess_d010_020", "ess_d020_plus"]
    xlabs = ["<0.05", "0.05-0.10", "0.10-0.20", ">0.20"]
    scen = ["S1_exchangeable", "S2_trap_heavy"]
    styles = {"two_head": ("#777777", "o", "--"),
              "two_head_pro": (TH_COLOR, "s", "-"),
              "selective_pro": (SEL_COLOR, "D", "-"),
              "selective_pro_fixdisc": ("#c0504d", "^", ":")}
    fig, axes = plt.subplots(1, len(scen), figsize=(10, 4.2), sharey=True)
    for ax, s in zip(axes, scen):
        for m, (col, mk, ls) in styles.items():
            r = next((r for r in rows if r["scenario"] == s and r["method"] == m), None)
            if not r:
                continue
            big = m == "two_head"  # coincides with the selective curve; draw as
            ax.plot(range(4), [f(r, b) for b in bins], marker=mk, ls=ls, color=col,
                    lw=(1.4 if big else 2.0), ms=(12 if big else 7),
                    markerfacecolor=("none" if big else col),
                    zorder=(1 if big else 3), label=LABEL[m] + (" (coincides)" if big else ""))
        ax.set_xticks(range(4))
        ax.set_xticklabels(xlabs, fontsize=9)
        ax.set_xlabel("True $|\\theta_k-\\theta_0|$")
        ax.set_title(s.replace("_", " "), fontsize=10)
        ax.grid(alpha=0.25)
    axes[0].set_ylabel("Learned per-source borrowed sample size")
    axes[0].legend(fontsize=8, loc="center left")
    fig.suptitle("Does the discount head adapt to true comparability? Flat = inert.\n"
                 "The fixed-budget model responds only with the prospective signal; "
                 "the selective model is flat even with it ---\nits adaptivity migrates "
                 "to the set-level total-borrowing head.", y=1.10, fontsize=10.5)
    fig.tight_layout()
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print("wrote", path)


def fig_discrimination(rows, path):
    scenarios = sorted({r["scenario"] for r in rows})
    methods = ["rule", "two_head_pro", "selective_pro", "uip_js"]
    colors = {"rule": "#8da0cb", "two_head_pro": TH_COLOR,
              "selective_pro": SEL_COLOR, "uip_js": "#66a61e"}
    fig, ax = plt.subplots(figsize=(10, 4.6))
    width = 0.8 / len(methods)
    x = np.arange(len(scenarios))
    for i, m in enumerate(methods):
        vals, errs = [], []
        for s in scenarios:
            r = next((r for r in rows if r["scenario"] == s and r["method"] == m), None)
            vals.append(f(r, "roc_auc") if r else np.nan)
            errs.append(f(r, "roc_auc_mcse") if r else np.nan)
        ax.bar(x + i * width, vals, width, yerr=errs, capsize=2, label=LABEL[m],
               edgecolor="black", linewidth=0.6, color=colors[m])
    ax.axhline(0.5, color="crimson", ls="--", lw=1.2)
    ax.text(len(scenarios) - 0.4, 0.512, "chance", color="crimson", fontsize=9)
    ax.set_xticks(x + 0.4 - width / 2)
    ax.set_xticklabels([s.replace("_", "\n") for s in scenarios], fontsize=9)
    ax.set_ylabel("ROC-AUC, parameter-level labels")
    ax.set_ylim(0.35, 0.95)
    ax.legend(fontsize=8, ncol=2, loc="upper right")
    ax.set_title("Borrowing-decision discrimination against parameter-level "
                 "exchangeability of latent rates.\nUIP-JS uses the current trial's "
                 "outcome; all other methods shown are available at design time.",
                 fontsize=10)
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)
    print("wrote", path)


def fig_external_control(ec, path):
    show = ["internal_only", "eb_only", "rule", "two_head_pro", "selective_pro",
            "robust_map_w0.5", "uip_js", "power_prior", "pooling"]
    nulls = [r for r in ec if r["world"] == "null"]
    alts = [r for r in ec if r["world"] == "alt"]
    drifts = sorted({f(r, "drift_shift") for r in nulls})
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    for ax, rows, key, title, ylab in [
            (axes[0], nulls, "reject_rate", "Type I error (no true effect)", "Rejection rate"),
            (axes[1], alts, "reject_rate", "Power (true effect 0.15)", "Rejection rate"),
            (axes[2], nulls, "control_bias", "Bias of the hybrid control rate",
             "Posterior mean $-$ true control rate")]:
        for m in show:
            v, e = _series(rows, m, "drift_shift", key, drifts)
            ax.errorbar(drifts, v, yerr=e, marker="o", ms=4, capsize=2,
                        label=LABEL[m], **style_for(m))
        ax.set_xlabel("External-control drift (logit)")
        ax.set_ylabel(ylab)
        ax.set_title(title, fontsize=10)
        ax.grid(alpha=0.25)
        ax.axvline(0.0, color="grey", lw=0.8, ls=":")
    axes[0].axhline(NOMINAL, color="crimson", ls="--", lw=1.2)
    axes[0].set_ylim(0, 0.24)
    axes[2].axhline(0.0, color="crimson", ls="--", lw=1.0)
    axes[0].legend(fontsize=7, loc="upper right", ncol=2)
    fig.suptitle("External control arm. The selective prior learns to withhold "
                 "external controls almost entirely and coincides with the EB-only "
                 "reference;\nnegative drift (borrowed controls worse than internal) "
                 "inflates type I error for methods that do borrow", y=1.06, fontsize=11)
    fig.tight_layout()
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print("wrote", path)


def main():
    design = read_csv(GS / "design_worlds.csv")
    rows = read_csv(GS / "scenario_results.csv")
    ec = read_csv(GS / "external_control.csv")
    # sanity checks against the numbers quoted in the manuscript
    def cell(w, s, m):
        r = [x for x in design if x["world"] == w and x["conflict_shift"] == s
             and x["method"] == m]
        return float(r[0]["reject_rate"])
    assert abs(cell("null", "0.0", "selective_pro") - 0.024) < 5e-4
    assert abs(cell("null", "1.5", "selective_pro") - 0.243) < 5e-4
    assert abs(cell("null", "1.0", "selective_pro_sam") - 0.070) < 5e-4
    assert abs(cell("alt", "0.0", "selective_pro") - 0.613) < 5e-4
    print("OK manuscript-value assertions")
    for d in (OUT_MAN, OUT_RES):
        d.mkdir(parents=True, exist_ok=True)
    fig_design_oc(design, OUT_MAN / "F8_design_oc.png")
    fig_ess_response(rows, OUT_MAN / "F9_ess_response.png")
    fig_discrimination(rows, OUT_MAN / "F10_discrimination.png")
    fig_external_control(ec, OUT_MAN / "F11_external_control.png")
    for n in ("F8_design_oc", "F9_ess_response", "F10_discrimination",
              "F11_external_control"):
        (OUT_RES / f"{n}.png").write_bytes((OUT_MAN / f"{n}.png").read_bytes())
    print("archived copies in", OUT_RES)


if __name__ == "__main__":
    main()
