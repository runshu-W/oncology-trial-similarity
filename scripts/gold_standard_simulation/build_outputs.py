"""Build statistics-journal style tables and figures from the simulation output.

Figures follow the reporting language of Pharmaceutical Statistics / Statistics
in Medicine: type I error and power curves against a fixed design null,
credible-interval coverage, bias, per-source effective sample size, and
diagnostic accuracy of the borrowing decision -- not retrieval leaderboards.
"""
from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

METHOD_ORDER = ["weak_only", "rule", "rule_sam", "two_head", "two_head_pro",
                "two_head_pro_sam", "two_head_pro_fixdisc", "robust_map_w0.5",
                "robust_map_w0.9", "power_prior", "uip_dirichlet", "uip_js",
                "pooling"]
LABEL = {
    "weak_only": "No borrowing", "rule": "Rule mixture", "rule_sam": "Rule + SAM",
    "two_head": "Two-head (no prospective)",
    "two_head_pro": "Two-head + prospective",
    "two_head_pro_sam": "Two-head + prospective + SAM",
    "two_head_pro_fixdisc": "Two-head + prospective, fixed discount",
    "robust_map_w0.5": "Robust-MAP (w=0.5)", "robust_map_w0.9": "Robust-MAP (w=0.9)",
    "power_prior": "Power prior (a0=0.3)", "uip_dirichlet": "UIP-Dirichlet",
    "uip_js": "UIP-JS (uses outcome)", "pooling": "Full pooling",
    "internal_only": "Internal control only",
}
HILITE = {"two_head_pro", "two_head_pro_sam"}
NOMINAL = 0.025


def read_csv(path):
    if not Path(path).exists():
        return []
    with open(path, newline="", encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def f(row, key):
    try:
        return float(row[key])
    except (TypeError, ValueError, KeyError):
        return float("nan")


def fmt(v, mcse=None, nd=3):
    if v != v:
        return "--"
    s = f"{v:.{nd}f}"
    return f"{s} ({mcse:.{nd}f})" if mcse is not None and mcse == mcse else s


# ---------------------------------------------------------------------------
# Tables
# ---------------------------------------------------------------------------
def write_scenario_table(rows, scenario, out):
    sel = [r for r in rows if r["scenario"] == scenario]
    order = {m: i for i, m in enumerate(METHOD_ORDER)}
    sel.sort(key=lambda r: order.get(r["method"], 99))
    lines = [
        f"Scenario: {scenario}",
        "Means with Monte Carlo standard errors in parentheses.",
        "",
        f"{'Method':<40}{'Bias':>16}{'RMSE':>9}{'Cov95':>15}{'ESS':>8}",
    ]
    for r in sel:
        lines.append(
            f"{LABEL.get(r['method'], r['method']):<40}"
            f"{fmt(f(r,'bias'), f(r,'bias_mcse')):>16}"
            f"{fmt(f(r,'rmse')):>9}"
            f"{fmt(f(r,'coverage95'), f(r,'coverage95_mcse')):>15}"
            f"{f(r,'prior_ess'):>8.1f}")
    lines += ["", "Borrowing-decision diagnostic accuracy against the "
                  "parameter-level gold standard",
              f"{'Method':<40}{'Sens':>14}{'Spec':>14}{'PPV':>9}{'ROC-AUC':>15}"
              f"{'Oracle rho':>12}"]
    for r in sel:
        lines.append(
            f"{LABEL.get(r['method'], r['method']):<40}"
            f"{fmt(f(r,'sensitivity'), f(r,'sensitivity_mcse')):>14}"
            f"{fmt(f(r,'specificity'), f(r,'specificity_mcse')):>14}"
            f"{fmt(f(r,'ppv')):>9}"
            f"{fmt(f(r,'roc_auc'), f(r,'roc_auc_mcse')):>15}"
            f"{fmt(f(r,'oracle_spearman')):>12}")
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return "\n".join(lines)


def write_ess_table(rows, out):
    """Learned per-source ESS by true distance -- the discount-adaptivity check."""
    lines = ["Learned per-source effective sample size, binned by the donor's true",
             "distance |theta_k - theta_0| from the query rate.",
             "A discount head that adapts to comparability should decrease across",
             "the row; a flat row means the head is inert.",
             "",
             f"{'Method':<40}{'Scenario':<18}{'d<0.05':>9}{'0.05-0.10':>11}"
             f"{'0.10-0.20':>11}{'d>0.20':>9}"]
    for s in sorted({r["scenario"] for r in rows}):
        for m in ["two_head", "two_head_pro", "two_head_pro_fixdisc"]:
            r = next((r for r in rows if r["scenario"] == s and r["method"] == m), None)
            if not r:
                continue
            lines.append(
                f"{LABEL.get(m, m):<40}{s:<18}"
                f"{f(r,'ess_d000_005'):>9.1f}{f(r,'ess_d005_010'):>11.1f}"
                f"{f(r,'ess_d010_020'):>11.1f}{f(r,'ess_d020_plus'):>9.1f}")
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return "\n".join(lines)


def write_design_table(design, out):
    nulls = [r for r in design if r["world"] == "null"]
    alts = [r for r in design if r["world"] == "alt"]
    shifts = sorted({f(r, "conflict_shift") for r in nulls})
    order = {m: i for i, m in enumerate(METHOD_ORDER)}
    methods = sorted({r["method"] for r in nulls}, key=lambda m: order.get(m, 99))
    lines = ["Design-based operating characteristics.",
             "Single-arm design testing H0: theta <= 0.20 at the 0.025 one-sided level.",
             "Type I error is the rejection rate in the null world (theta_0 = 0.20);",
             "power is the rejection rate in the alternative world (theta_0 = 0.35).",
             "",
             "TYPE I ERROR by historical-vs-current logit shift",
             f"{'Method':<40}" + "".join(f"{s:>8.2f}" for s in shifts)]
    for m in methods:
        vals = [next((f(r, "reject_rate") for r in nulls
                      if r["method"] == m and f(r, "conflict_shift") == s), float("nan"))
                for s in shifts]
        lines.append(f"{LABEL.get(m, m):<40}" + "".join(f"{v:>8.3f}" for v in vals))
    lines += ["", "POWER by historical-vs-current logit shift",
              f"{'Method':<40}" + "".join(f"{s:>8.2f}" for s in shifts)]
    for m in methods:
        vals = [next((f(r, "reject_rate") for r in alts
                      if r["method"] == m and f(r, "conflict_shift") == s), float("nan"))
                for s in shifts]
        lines.append(f"{LABEL.get(m, m):<40}" + "".join(f"{v:>8.3f}" for v in vals))
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------
def _series(rows, method, xkey, ykey, xs):
    v, e = [], []
    for x in xs:
        r = next((r for r in rows if r["method"] == method and f(r, xkey) == x), None)
        v.append(f(r, ykey) if r else np.nan)
        e.append(f(r, ykey + "_mcse") if r else np.nan)
    return v, e


def fig_design_oc(design, path):
    """Type I error and power against the fixed design null."""
    show = ["weak_only", "rule", "two_head", "two_head_pro",
            "two_head_pro_fixdisc", "robust_map_w0.5", "uip_dirichlet", "uip_js"]
    nulls = [r for r in design if r["world"] == "null"]
    alts = [r for r in design if r["world"] == "alt"]
    shifts = sorted({f(r, "conflict_shift") for r in nulls})
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.6))
    for ax, rows, title, ylab in [
        (axes[0], nulls, "Type I error (null world, $\\theta_0=0.20$)",
         "Rejection rate"),
        (axes[1], alts, "Power (alternative world, $\\theta_0=0.35$)",
         "Rejection rate")]:
        for m in show:
            v, e = _series(rows, m, "conflict_shift", "reject_rate", shifts)
            ax.errorbar(shifts, v, yerr=e, marker="o", ms=4, capsize=2,
                        lw=(2.4 if m in HILITE else 1.2), label=LABEL[m],
                        color=("#1f4e79" if m == "two_head_pro" else None),
                        zorder=(5 if m in HILITE else 2))
        ax.set_xlabel("Historical-vs-current logit shift")
        ax.set_ylabel(ylab)
        ax.set_title(title, fontsize=11)
        ax.grid(alpha=0.25)
    axes[0].axhline(NOMINAL, color="crimson", ls="--", lw=1.2)
    axes[0].annotate("nominal 0.025", (0.02, NOMINAL + 0.004), color="crimson", fontsize=8)
    # No-borrowing reference: the achievable level given binomial discreteness.
    ref = next((f(r, "reject_rate") for r in nulls
                if r["method"] == "weak_only" and f(r, "conflict_shift") == 0.0), np.nan)
    if ref == ref:
        axes[0].axhline(ref, color="grey", ls=":", lw=1.2)
        axes[0].annotate(f"no-borrowing reference {ref:.3f}", (0.02, ref + 0.004),
                         color="grey", fontsize=8)
    axes[0].set_ylim(0, 0.16)
    axes[1].legend(fontsize=7.5, loc="lower left", ncol=2)
    fig.suptitle("Operating characteristics against a fixed design null "
                 "(full pooling and power prior omitted: type I error 0.48-0.90)",
                 y=1.02, fontsize=11)
    fig.tight_layout()
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def fig_ess_response(rows, path):
    """Learned per-source ESS vs true distance: is the discount head adaptive?"""
    bins = ["ess_d000_005", "ess_d005_010", "ess_d010_020", "ess_d020_plus"]
    xlabs = ["<0.05", "0.05-0.10", "0.10-0.20", ">0.20"]
    scen = ["S1_exchangeable", "S2_trap_heavy"]
    fig, axes = plt.subplots(1, len(scen), figsize=(10, 4.1), sharey=True)
    styles = {"two_head": ("#999999", "o", "--"),
              "two_head_pro": ("#1f4e79", "s", "-"),
              "two_head_pro_fixdisc": ("#c0504d", "^", ":")}
    for ax, s in zip(axes, scen):
        for m, (col, mk, ls) in styles.items():
            r = next((r for r in rows if r["scenario"] == s and r["method"] == m), None)
            if not r:
                continue
            ax.plot(range(4), [f(r, b) for b in bins], marker=mk, ls=ls, color=col,
                    lw=2.0, ms=7, label=LABEL[m])
        ax.set_xticks(range(4))
        ax.set_xticklabels(xlabs, fontsize=9)
        ax.set_xlabel("True $|\\theta_k-\\theta_0|$")
        ax.set_title(s.replace("_", " "), fontsize=10)
        ax.grid(alpha=0.25)
    axes[0].set_ylabel("Learned per-source ESS")
    axes[0].legend(fontsize=8, loc="lower left")
    fig.suptitle("Does the discount head adapt to true comparability?\n"
                 "Flat = inert. Only the variant given the prospective set-level "
                 "conflict signal responds.", y=1.06, fontsize=11)
    fig.tight_layout()
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def fig_discrimination(rows, path):
    scenarios = sorted({r["scenario"] for r in rows})
    methods = ["rule", "two_head", "two_head_pro", "uip_dirichlet", "uip_js"]
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
               edgecolor="black", linewidth=0.6,
               color=("#1f4e79" if m == "two_head_pro" else None),
               alpha=(1.0 if m in HILITE else 0.8))
    ax.axhline(0.5, color="crimson", ls="--", lw=1.2)
    ax.text(len(scenarios) - 0.4, 0.512, "chance", color="crimson", fontsize=9)
    ax.set_xticks(x + 0.4 - width / 2)
    ax.set_xticklabels([s.replace("_", "\n") for s in scenarios], fontsize=9)
    ax.set_ylabel("ROC-AUC for identifying exchangeable donors")
    ax.set_ylim(0.35, 0.95)
    ax.legend(fontsize=8, ncol=2, loc="upper right")
    ax.set_title("Borrowing-decision discrimination. UIP-JS uses the current "
                 "trial's outcome;\nall other methods shown are available at "
                 "design time.", fontsize=10)
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def fig_external_control(ec, path):
    show = ["internal_only", "rule", "two_head_pro", "two_head_pro_fixdisc",
            "robust_map_w0.5", "uip_dirichlet", "uip_js", "power_prior", "pooling"]
    nulls = [r for r in ec if r["world"] == "null"]
    alts = [r for r in ec if r["world"] == "alt"]
    drifts = sorted({f(r, "drift_shift") for r in nulls})
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.4))
    for ax, rows, key, title, ylab in [
        (axes[0], nulls, "reject_rate", "Type I error (no true effect)", "Rejection rate"),
        (axes[1], alts, "reject_rate", "Power (true effect 0.15)", "Rejection rate"),
        (axes[2], nulls, "control_bias", "Bias of the hybrid control rate",
         "Posterior mean $-$ true control rate")]:
        for m in show:
            v, e = _series(rows, m, "drift_shift", key, drifts)
            ax.errorbar(drifts, v, yerr=e, marker="o", ms=4, capsize=2,
                        lw=(2.4 if m == "two_head_pro" else 1.1), label=LABEL[m],
                        color=("#1f4e79" if m == "two_head_pro" else
                               "black" if m == "internal_only" else None),
                        zorder=(5 if m == "two_head_pro" else 2))
        ax.set_xlabel("External-control drift (logit)")
        ax.set_ylabel(ylab)
        ax.set_title(title, fontsize=10)
        ax.grid(alpha=0.25)
        ax.axvline(0.0, color="grey", lw=0.8, ls=":")
    axes[0].axhline(NOMINAL, color="crimson", ls="--", lw=1.2)
    axes[0].set_ylim(0, 0.24)
    axes[2].axhline(0.0, color="crimson", ls="--", lw=1.0)
    axes[0].legend(fontsize=7, loc="upper right", ncol=2)
    fig.suptitle("External control arm: negative drift means the borrowed controls "
                 "are worse than the internal control,\nwhich biases the treatment "
                 "effect upward and inflates type I error", y=1.06, fontsize=11)
    fig.tight_layout()
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Lightweight results package (repo convention: see results/README.md)
# ---------------------------------------------------------------------------
def _tex_escape(s):
    return str(s).replace("_", r"\_").replace("%", r"\%").replace("&", r"\&")


def write_tex_design_oc(design, path):
    nulls = [r for r in design if r["world"] == "null"]
    alts = [r for r in design if r["world"] == "alt"]
    shifts = [0.0, 0.5, 1.0]
    order = {m: i for i, m in enumerate(METHOD_ORDER)}
    methods = sorted({r["method"] for r in nulls}, key=lambda m: order.get(m, 99))
    lines = [
        r"\begin{table}[htbp]", r"\centering", r"\small",
        r"\caption{Design-based operating characteristics from the gold-standard "
        r"simulation (2000 replicates per cell). Single-arm design testing "
        r"$H_0\!:\theta \le 0.20$ at the one-sided 0.025 level. Type I error is "
        r"the rejection rate in the null world ($\theta_0=0.20$); power is the "
        r"rejection rate in the alternative world ($\theta_0=0.35$). Columns are "
        r"the historical-versus-current logit shift.}",
        r"\label{tab:gold-standard-design-oc}",
        r"\begin{tabular}{lrrrrrr}", r"\toprule",
        r"& \multicolumn{3}{c}{Type I error} & \multicolumn{3}{c}{Power} \\",
        r"Method & " + " & ".join(f"{s:.1f}" for s in shifts) + " & "
        + " & ".join(f"{s:.1f}" for s in shifts) + r" \\", r"\midrule",
    ]
    for m in methods:
        t1 = [next((f(r, "reject_rate") for r in nulls if r["method"] == m
                    and f(r, "conflict_shift") == s), float("nan")) for s in shifts]
        pw = [next((f(r, "reject_rate") for r in alts if r["method"] == m
                    and f(r, "conflict_shift") == s), float("nan")) for s in shifts]
        lines.append(_tex_escape(LABEL.get(m, m)) + " & "
                     + " & ".join(f"{v:.3f}" for v in t1 + pw) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_tex_external_control(ec, path):
    rows = [r for r in ec if r["world"] == "null" and f(r, "drift_shift") == -1.2]
    order = {m: i for i, m in enumerate(METHOD_ORDER + ["internal_only"])}
    rows.sort(key=lambda r: order.get(r["method"], 99))
    lines = [
        r"\begin{table}[htbp]", r"\centering", r"\small",
        r"\caption{External control arm, null world (no true treatment effect), "
        r"external controls displaced downward by a logit drift of $-1.2$; 2000 "
        r"replicates. FPR is the proportion of non-comparable external controls "
        r"that were nonetheless borrowed.}",
        r"\label{tab:gold-standard-external-control}",
        r"\begin{tabular}{lrrrr}", r"\toprule",
        r"Method & Type I error & Control bias & Prior ESS & FPR \\", r"\midrule",
    ]
    for r in rows:
        lines.append(
            f"{_tex_escape(LABEL.get(r['method'], r['method']))} & "
            f"{f(r,'reject_rate'):.3f} & {f(r,'control_bias'):+.4f} & "
            f"{f(r,'prior_ess'):.1f} & {f(r,'fpr'):.2f} \\\\")
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def export_results_package(d, rows, design, ec, results_dir):
    """Write the lightweight, git-tracked evidence package.

    Large intermediate output stays in artifacts/; this mirrors only what a
    reader needs to check the numbers reported in the manuscript, following the
    convention documented in results/README.md.
    """
    tdir = results_dir / "tables"
    fdir = results_dir / "figures"
    tdir.mkdir(parents=True, exist_ok=True)
    fdir.mkdir(parents=True, exist_ok=True)

    for name, src in [("gold_standard_scenarios.csv", d / "scenario_results.csv"),
                      ("gold_standard_design_worlds.csv", d / "design_worlds.csv"),
                      ("gold_standard_external_control.csv", d / "external_control.csv"),
                      ("gold_standard_external_control_pcomp.csv",
                       d / "external_control_pcomp.csv"),
                      ("gold_standard_run_config.json", d / "run_config.json"),
                      ("gold_standard_dgm_diagnostics.json", d / "dgm_diagnostics.json")]:
        if src.exists():
            (tdir / name).write_text(src.read_text(encoding="utf-8"), encoding="utf-8")

    if design:
        write_tex_design_oc(design, tdir / "table_gold_standard_design_oc.tex")
    if ec:
        write_tex_external_control(ec, tdir / "table_gold_standard_external_control.tex")

    # Figures are re-rendered as SVG, which is what the results package uses.
    if design:
        fig_design_oc(design, fdir / "gold_standard_design_oc.svg")
    fig_ess_response(rows, fdir / "gold_standard_ess_response.svg")
    fig_discrimination(rows, fdir / "gold_standard_discrimination.svg")
    if ec:
        fig_external_control(ec, fdir / "gold_standard_external_control.svg")
    print(f"results package written to {results_dir}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", default="../../artifacts/gold_standard_simulation")
    ap.add_argument("--results", default="../../results",
                    help="lightweight, git-tracked evidence package")
    ap.add_argument("--no-results", action="store_true")
    args = ap.parse_args()
    d = Path(args.input)
    if not d.is_absolute():
        d = (Path(__file__).resolve().parent / d).resolve()

    rows = read_csv(d / "scenario_results.csv")
    design = read_csv(d / "design_worlds.csv")
    ec = read_csv(d / "external_control.csv")

    tables = d / "tables"
    tables.mkdir(exist_ok=True)
    for s in sorted({r["scenario"] for r in rows}):
        write_scenario_table(rows, s, tables / f"table_{s}.txt")
    print(write_ess_table(rows, tables / "table_ess_response.txt"))
    print()
    if design:
        print(write_design_table(design, tables / "table_design_oc.txt"))

    figs = d / "figures"
    figs.mkdir(exist_ok=True)
    if design:
        fig_design_oc(design, figs / "F1_design_operating_characteristics.png")
    fig_ess_response(rows, figs / "F2_learned_ess_response.png")
    fig_discrimination(rows, figs / "F3_borrowing_discrimination.png")
    if ec:
        fig_external_control(ec, figs / "F4_external_control.png")
    print(f"\nfigures written to {figs}")

    if not args.no_results:
        res = Path(args.results)
        if not res.is_absolute():
            res = (Path(__file__).resolve().parent / res).resolve()
        export_results_package(d, rows, design, ec, res)


if __name__ == "__main__":
    main()
