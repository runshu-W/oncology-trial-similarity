from __future__ import annotations

import csv
import html
import json
import math
import shutil
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
ARTIFACTS = REPO_ROOT / "artifacts"
RESULTS = REPO_ROOT / "results"
TABLES = RESULTS / "tables"
FIGURES = RESULTS / "figures"
DOCS = REPO_ROOT / "docs"


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames = list(rows[0].keys())
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def fnum(value: Any, digits: int = 3) -> str:
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return "NA"
    return f"{numeric:.{digits}f}" if math.isfinite(numeric) else "NA"


def latex_escape(value: Any) -> str:
    text = str(value)
    replacements = {
        "\\": r"\textbackslash{}",
        "&": r"\&",
        "%": r"\%",
        "$": r"\$",
        "#": r"\#",
        "_": r"\_",
        "{": r"\{",
        "}": r"\}",
        "~": r"\textasciitilde{}",
        "^": r"\textasciicircum{}",
    }
    return "".join(replacements.get(char, char) for char in text)


def write_latex_table(
    path: Path,
    caption: str,
    label: str,
    headers: list[str],
    rows: list[list[Any]],
    alignment: str | None = None,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    alignment = alignment or ("l" + "r" * (len(headers) - 1))
    lines = [
        r"\begin{table}[htbp]",
        r"\centering",
        r"\small",
        rf"\caption{{{latex_escape(caption)}}}",
        rf"\label{{{label}}}",
        rf"\begin{{tabular}}{{{alignment}}}",
        r"\toprule",
        " & ".join(latex_escape(header) for header in headers) + r" \\",
        r"\midrule",
    ]
    for row in rows:
        lines.append(" & ".join(latex_escape(cell) for cell in row) + r" \\")
    lines.extend([r"\bottomrule", r"\end{tabular}", r"\end{table}", ""])
    path.write_text("\n".join(lines), encoding="utf-8")


def copy_core_results() -> None:
    TABLES.mkdir(parents=True, exist_ok=True)
    FIGURES.mkdir(parents=True, exist_ok=True)
    copies = {
        ARTIFACTS / "operating_characteristics_simulation" / "simulation_operating_characteristics.csv": TABLES / "simulation_operating_characteristics.csv",
        ARTIFACTS / "paired_stage1_backend_benchmark" / "paired_backend_summary.csv": TABLES / "paired_backend_summary.csv",
        ARTIFACTS / "paired_stage1_backend_benchmark" / "paired_backend_delta_bootstrap_ci.csv": TABLES / "paired_backend_delta_bootstrap_ci.csv",
        ARTIFACTS / "borrowing_baseline_head_to_head" / "borrowing_baseline_summary.csv": TABLES / "borrowing_baseline_summary.csv",
        ARTIFACTS / "temporal_validation_true_dates" / "temporal_borrowing_nll_table.csv": TABLES / "temporal_borrowing_nll_table.csv",
        ARTIFACTS / "temporal_validation_true_dates" / "clinicaltrials_date_missingness_report.json": TABLES / "clinicaltrials_date_missingness_report.json",
        ARTIFACTS / "feature_ablation_sensitivity" / "feature_ablation_results.csv": TABLES / "feature_ablation_results.csv",
        ARTIFACTS / "feature_ablation_sensitivity" / "section_weight_sensitivity.csv": TABLES / "section_weight_sensitivity.csv",
        ARTIFACTS / "feature_ablation_sensitivity" / "feature_ablation_heatmap.svg": FIGURES / "feature_ablation_heatmap.svg",
    }
    for source, target in copies.items():
        if source.exists():
            shutil.copyfile(source, target)


def build_paired_benchmark_table() -> None:
    summary = read_csv(TABLES / "paired_backend_summary.csv")
    ci_rows = read_csv(TABLES / "paired_backend_delta_bootstrap_ci.csv")
    ci_by_key = {row["delta_key"]: row for row in ci_rows}
    rows = []
    for row in summary:
        label = row["label"]
        if label == "secret_pool":
            component_ci = ci_by_key.get("secret_pool_minus_hashing_rerank_component_ready_rate", {})
            endpoint_ci = ci_by_key.get("secret_pool_minus_hashing_rerank_mean_endpoint_match", {})
            delta_component = f"{fnum(component_ci.get('mean_delta'))} [{fnum(component_ci.get('ci_lower'))}, {fnum(component_ci.get('ci_upper'))}]"
            delta_endpoint = f"{fnum(endpoint_ci.get('mean_delta'))} [{fnum(endpoint_ci.get('ci_lower'))}, {fnum(endpoint_ci.get('ci_upper'))}]"
        else:
            delta_component = "Reference"
            delta_endpoint = "Reference"
        rows.append(
            [
                label,
                row["query_count"],
                fnum(row["mean_rerank_component_ready_rate"]),
                fnum(row["mean_rerank_mean_endpoint_match"]),
                delta_component,
                delta_endpoint,
            ]
        )
    write_latex_table(
        TABLES / "table_paired_stage1_benchmark.tex",
        "Paired Stage 1 backend benchmark on common ORR pseudo-queries.",
        "tab:paired-stage1",
        ["Backend", "Queries", "Component-ready", "Endpoint match", "Delta component-ready", "Delta endpoint match"],
        rows,
        alignment="lrrrrr",
    )


def build_baseline_table() -> None:
    rows = []
    for row in read_csv(TABLES / "borrowing_baseline_summary.csv"):
        rows.append(
            [
                row["method"],
                row["example_count"],
                fnum(row["mean_nll"], 4),
                fnum(row["mean_nll_minus_rule"], 4),
                fnum(row["mean_historical_mass"], 3),
            ]
        )
    write_latex_table(
        TABLES / "table_borrowing_baseline_head_to_head.tex",
        "Held-out beta-binomial predictive NLL for borrowing prior baselines.",
        "tab:borrowing-baselines",
        ["Method", "Examples", "Mean NLL", "Delta vs rule", "Historical mass"],
        rows,
        alignment="lrrrr",
    )


def build_temporal_tables() -> None:
    rows = read_csv(TABLES / "temporal_borrowing_nll_table.csv")
    for strategy, filename, label in (
        ("date_based", "table_true_date_temporal_nll_date_based.tex", "tab:temporal-date-based"),
        ("rolling_origin", "table_true_date_temporal_nll_rolling_origin.tex", "tab:temporal-rolling"),
    ):
        table_rows = []
        for row in rows:
            if row["split_strategy"] != strategy:
                continue
            table_rows.append(
                [
                    row["split_label"].replace("train_through_", "through "),
                    row["method"],
                    row["train_count"],
                    row["eval_count"],
                    fnum(row["mean_nll"], 4),
                    fnum(row["mean_nll_minus_rule"], 4),
                ]
            )
        write_latex_table(
            TABLES / filename,
            f"True-date temporal borrowing validation ({strategy.replace('_', ' ')} splits).",
            label,
            ["Split", "Method", "Train", "Eval", "Mean NLL", "Delta vs rule"],
            table_rows,
            alignment="llrrrr",
        )


def build_simulation_table() -> None:
    rows = []
    for row in read_csv(TABLES / "simulation_operating_characteristics.csv"):
        rows.append(
            [
                row["scenario"],
                row["method"],
                row["iterations"],
                row["example_count"],
                fnum(row["type_i_error"], 4),
                fnum(row["power"], 4),
                fnum(row["bias"], 4),
                fnum(row["mse"], 4),
                fnum(row["coverage"], 4),
                fnum(row["sam_trigger_rate"], 4),
            ]
        )
    write_latex_table(
        TABLES / "table_simulation_operating_characteristics.tex",
        "Simulation operating characteristics under exchangeability and prior-data conflict scenarios.",
        "tab:simulation-oc",
        ["Scenario", "Method", "Iterations", "Templates", "Type I error", "Power", "Bias", "MSE", "Coverage", "SAM trigger"],
        rows,
        alignment="llrrrrrrrr",
    )


def build_feature_ablation_table() -> None:
    rows = []
    for row in read_csv(TABLES / "feature_ablation_results.csv"):
        rows.append(
            [
                row["ablation"],
                row["dropped_features"] or "None",
                fnum(row["mean_nll"], 4),
                fnum(row["mean_nll_minus_full"], 4),
                row["example_count"],
            ]
        )
    write_latex_table(
        TABLES / "table_feature_ablation.tex",
        "Deterministic feature-weight proxy ablation for the nine-feature borrowability schema.",
        "tab:feature-ablation",
        ["Ablation", "Dropped features", "Mean NLL", "Delta vs full", "Examples"],
        rows,
        alignment="llrrr",
    )


def build_pipeline_svg() -> None:
    boxes = [
        ("ClinicalTrials.gov JSON", "Trial records, outcomes, eligibility, dates"),
        ("Structured summary", "Disease, regimen, endpoint, follow-up, red flags"),
        ("Stage 1 retrieval", "ClinicalBERT / Trial2Vec / SECRET-style / hashing"),
        ("SECRET pool", "Section-weighted high-recall candidate set"),
        ("Stage 2 reranker", "Explainable borrowability features"),
        ("Endpoint observations", "Extract y_i and n_i for candidate endpoints"),
        ("Beta components", "alpha_i = 1 + a_i y_i; beta_i = 1 + a_i(n_i-y_i)"),
        ("Mixture prior + SAM", "Weak prior, historical components, conflict adaptation"),
        ("Validation", "NLL, temporal validation, OC simulation, ablation"),
    ]
    width = 1200
    height = 1080
    box_w = 820
    box_h = 78
    x = 190
    y0 = 45
    gap = 38
    lines = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="1200" height="1080" fill="#ffffff"/>',
        '<style>text{font-family:Arial,Helvetica,sans-serif}.title{font-size:22px;font-weight:700;fill:#18212f}.sub{font-size:15px;fill:#4b5563}.box{fill:#f8fafc;stroke:#1f77b4;stroke-width:2}.arrow{stroke:#334155;stroke-width:2.5;marker-end:url(#arrow)}</style>',
        '<defs><marker id="arrow" markerWidth="10" markerHeight="10" refX="8" refY="3" orient="auto" markerUnits="strokeWidth"><path d="M0,0 L0,6 L9,3 z" fill="#334155"/></marker></defs>',
    ]
    for index, (title, subtitle) in enumerate(boxes):
        y = y0 + index * (box_h + gap)
        lines.append(f'<rect class="box" x="{x}" y="{y}" width="{box_w}" height="{box_h}" rx="6"/>')
        lines.append(f'<text class="title" x="{x + 28}" y="{y + 31}">{html.escape(title)}</text>')
        lines.append(f'<text class="sub" x="{x + 28}" y="{y + 56}">{html.escape(subtitle)}</text>')
        if index < len(boxes) - 1:
            y1 = y + box_h
            y2 = y + box_h + gap - 7
            lines.append(f'<line class="arrow" x1="{width / 2:.0f}" y1="{y1 + 5}" x2="{width / 2:.0f}" y2="{y2}"/>')
    lines.append("</svg>")
    (FIGURES / "pipeline_diagram.svg").write_text("\n".join(lines), encoding="utf-8")


def build_simulation_svg() -> None:
    rows = read_csv(TABLES / "simulation_operating_characteristics.csv")
    scenarios = list(dict.fromkeys(row["scenario"] for row in rows))
    methods = list(dict.fromkeys(row["method"] for row in rows))
    by_key = {(row["scenario"], row["method"]): row for row in rows}
    cell_w = 118
    cell_h = 40
    left = 250
    top = 88
    width = left + len(methods) * cell_w + 80
    height = top + len(scenarios) * cell_h + 90
    lines = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="#ffffff"/>',
        '<style>text{font-family:Arial,Helvetica,sans-serif}.title{font-size:20px;font-weight:700;fill:#111827}.label{font-size:12px;fill:#374151}.value{font-size:12px;fill:#111827}.cell{stroke:#ffffff;stroke-width:1}</style>',
        '<text class="title" x="24" y="34">Simulation operating characteristics: power by scenario and method</text>',
    ]
    for j, method in enumerate(methods):
        x = left + j * cell_w + cell_w / 2
        lines.append(f'<text class="label" text-anchor="middle" x="{x:.0f}" y="70">{html.escape(method)}</text>')
    for i, scenario in enumerate(scenarios):
        y = top + i * cell_h
        lines.append(f'<text class="label" text-anchor="end" x="{left - 12}" y="{y + 25}">{html.escape(scenario)}</text>')
        for j, method in enumerate(methods):
            row = by_key[(scenario, method)]
            power = float(row["power"])
            intensity = max(0.0, min(1.0, power / 0.35))
            blue = int(245 - 120 * intensity)
            green = int(248 - 80 * intensity)
            red = int(239 - 190 * intensity)
            x = left + j * cell_w
            fill = f"rgb({red},{green},{blue})"
            lines.append(f'<rect class="cell" x="{x}" y="{y}" width="{cell_w}" height="{cell_h}" fill="{fill}"/>')
            lines.append(f'<text class="value" text-anchor="middle" x="{x + cell_w/2:.0f}" y="{y + 25}">{power:.3f}</text>')
    lines.append('<text class="label" x="24" y="{0}">Cell values are empirical power. Type I error, coverage, bias and SAM trigger are reported in Table OC.</text>'.format(height - 24))
    lines.append("</svg>")
    (FIGURES / "simulation_oc_power_heatmap.svg").write_text("\n".join(lines), encoding="utf-8")


def write_evidence_plan() -> None:
    text = """# Manuscript Evidence Package Plan

Target journals: Pharmaceutical Statistics or Statistics in Medicine.

## Core Figures and Tables

| Item | Output | Source | Intended manuscript role |
|---|---|---|---|
| Figure 1 pipeline diagram | `results/figures/pipeline_diagram.svg`, `results/figures/pipeline_diagram.pdf` | Pipeline architecture | Methods overview |
| Simulation OC table | `results/tables/table_simulation_operating_characteristics.tex` | `artifacts/operating_characteristics_simulation/` | Simulation study |
| Simulation power heatmap | `results/figures/simulation_oc_power_heatmap.svg`, `results/figures/simulation_oc_power_heatmap.pdf` | `simulation_operating_characteristics.csv` | Visual summary of operating characteristics |
| Paired Stage 1 table | `results/tables/table_paired_stage1_benchmark.tex` | `artifacts/paired_stage1_backend_benchmark/` | Retrieval benchmark |
| Borrowing baseline table | `results/tables/table_borrowing_baseline_head_to_head.tex` | `artifacts/borrowing_baseline_head_to_head/` | Predictive calibration baselines |
| True-date temporal NLL tables | `results/tables/table_true_date_temporal_nll_*.tex` | `artifacts/temporal_validation_true_dates/` | Temporal validation |
| Feature ablation table and heatmap | `results/tables/table_feature_ablation.tex`, `results/figures/feature_ablation_heatmap.svg`, `results/figures/feature_ablation_heatmap.pdf` | `artifacts/feature_ablation_sensitivity/` | Sensitivity analysis |

## Framing

All results are retrospective predictive calibration or simulation evidence without expert borrowability labels. They support method development and internal validation, not clinical deployment or regulatory qualification.
"""
    (DOCS / "manuscript_evidence_plan.md").write_text(text, encoding="utf-8")


def build_pdf_figures_if_available() -> None:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import numpy as np
    except Exception:
        return

    _build_pipeline_pdf(plt)
    _build_simulation_pdf(plt, np)
    _build_feature_ablation_pdf(plt)


def _build_pipeline_pdf(plt: Any) -> None:
    boxes = [
        "ClinicalTrials.gov JSON",
        "Structured summary",
        "Stage 1 retrieval",
        "SECRET pool",
        "Stage 2 reranker",
        "Endpoint observations",
        "Beta components",
        "Mixture prior + SAM",
        "Validation",
    ]
    fig, ax = plt.subplots(figsize=(8.0, 10.0))
    ax.axis("off")
    y_values = list(reversed(range(len(boxes))))
    for y, label in zip(y_values, boxes):
        ax.add_patch(
            plt.Rectangle(
                (0.12, y + 0.16),
                0.76,
                0.62,
                facecolor="#f8fafc",
                edgecolor="#1f77b4",
                linewidth=1.6,
            )
        )
        ax.text(0.5, y + 0.47, label, ha="center", va="center", fontsize=11, fontweight="bold")
        if y > 0:
            ax.annotate(
                "",
                xy=(0.5, y - 0.02),
                xytext=(0.5, y + 0.14),
                arrowprops={"arrowstyle": "->", "color": "#334155", "lw": 1.4},
            )
    ax.set_xlim(0, 1)
    ax.set_ylim(0, len(boxes))
    fig.tight_layout()
    fig.savefig(FIGURES / "pipeline_diagram.pdf")
    plt.close(fig)


def _build_simulation_pdf(plt: Any, np: Any) -> None:
    rows = read_csv(TABLES / "simulation_operating_characteristics.csv")
    scenarios = list(dict.fromkeys(row["scenario"] for row in rows))
    methods = list(dict.fromkeys(row["method"] for row in rows))
    values = np.zeros((len(scenarios), len(methods)))
    for i, scenario in enumerate(scenarios):
        for j, method in enumerate(methods):
            row = next(r for r in rows if r["scenario"] == scenario and r["method"] == method)
            values[i, j] = float(row["power"])
    fig, ax = plt.subplots(figsize=(8.5, 4.8))
    image = ax.imshow(values, cmap="Blues", vmin=0, vmax=max(0.35, float(values.max())))
    ax.set_xticks(range(len(methods)), methods, rotation=30, ha="right")
    ax.set_yticks(range(len(scenarios)), scenarios)
    ax.set_title("Simulation operating characteristics: empirical power")
    for i in range(len(scenarios)):
        for j in range(len(methods)):
            ax.text(j, i, f"{values[i, j]:.3f}", ha="center", va="center", fontsize=8)
    fig.colorbar(image, ax=ax, shrink=0.75, label="Power")
    fig.tight_layout()
    fig.savefig(FIGURES / "simulation_oc_power_heatmap.pdf")
    plt.close(fig)


def _build_feature_ablation_pdf(plt: Any) -> None:
    rows = read_csv(TABLES / "feature_ablation_results.csv")
    labels = [row["ablation"].replace("drop_", "") for row in rows]
    values = [float(row["mean_nll_minus_full"]) for row in rows]
    fig, ax = plt.subplots(figsize=(8.5, 4.8))
    colors = ["#1f77b4" if value >= 0 else "#7f7f7f" for value in values]
    ax.barh(labels, values, color=colors)
    ax.axvline(0, color="#111827", linewidth=1)
    ax.set_xlabel("Mean NLL minus full feature proxy")
    ax.set_title("Feature ablation sensitivity")
    fig.tight_layout()
    fig.savefig(FIGURES / "feature_ablation_heatmap.pdf")
    plt.close(fig)


def main() -> None:
    copy_core_results()
    build_paired_benchmark_table()
    build_baseline_table()
    build_temporal_tables()
    build_simulation_table()
    build_feature_ablation_table()
    build_pipeline_svg()
    build_simulation_svg()
    build_pdf_figures_if_available()
    write_evidence_plan()


if __name__ == "__main__":
    main()
