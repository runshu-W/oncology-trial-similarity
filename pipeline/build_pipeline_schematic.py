#!/usr/bin/env python3
"""Rebuild F7 (Figure 1): pipeline schematic, selective-borrowing revision.

Replaces the pre-revision schematic, which still showed the fixed-budget
two-head core (flat Beta(1,1) anchor, no set-level weak head). The layout
mirrors the published figure: five horizontal lanes, one main flow, with the
core method highlighted and the mixture prior in a side panel.

Outputs:
  manuscript/figures/F7_pipeline_schematic.pdf
  results/figures/final_v4_selective/F7_pipeline_schematic.{pdf,tif}
"""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

ROOT = Path(__file__).resolve().parents[1]
OUT_MAN = ROOT / "manuscript" / "figures"
OUT_RES = ROOT / "results" / "figures" / "final_v4_selective"

LANES = [
    ("1  Input", "#eef2f8", "#3c5a86"),
    ("2  Retrieval", "#e8f4ef", "#1f7a5c"),
    ("3  Reranking", "#eaf3e4", "#4a7a2a"),
    ("4  Bayesian prior · CORE", "#fdf3e3", "#b97a1a"),
    ("5  Validation", "#f0f0f2", "#5a5a66"),
]
RED = "#b01c2e"
BOX_EDGE = "#2c4a6e"


def lane_box(ax, y0, y1, label, face, edge):
    ax.add_patch(FancyBboxPatch((0.02, y0), 0.96, y1 - y0,
                                boxstyle="round,pad=0.004,rounding_size=0.012",
                                facecolor=face, edgecolor=edge, linewidth=1.2,
                                zorder=1))
    ax.text(0.045, (y0 + y1) / 2, label, rotation=90, va="center", ha="center",
            fontsize=10.5, color=edge, fontweight="bold", zorder=3)


def node(ax, cx, cy, w, h, title, sub, edge=BOX_EDGE, title_color="#1a1a1a",
         face="white", lw=1.6, title_size=11, sub_size=8.8):
    ax.add_patch(FancyBboxPatch((cx - w / 2, cy - h / 2), w, h,
                                boxstyle="round,pad=0.004,rounding_size=0.014",
                                facecolor=face, edgecolor=edge, linewidth=lw,
                                zorder=4))
    if sub:
        ax.text(cx, cy + h * 0.20, title, ha="center", va="center",
                fontsize=title_size, fontweight="bold", color=title_color, zorder=5)
        ax.text(cx, cy - h * 0.22, sub, ha="center", va="center",
                fontsize=sub_size, color="#333333", zorder=5)
    else:
        ax.text(cx, cy, title, ha="center", va="center", fontsize=title_size,
                fontweight="bold", color=title_color, zorder=5)


def arrow(ax, x, y_from, y_to, label=None):
    ax.add_patch(FancyArrowPatch((x, y_from), (x, y_to),
                                 arrowstyle="-|>", mutation_scale=16,
                                 linewidth=1.6, color="#44505e", zorder=6))
    if label:
        ax.text(0.965, (y_from + y_to) / 2, label, ha="right", va="center",
                fontsize=8.2, style="italic", color="#44505e", zorder=6)


def main() -> None:
    fig, ax = plt.subplots(figsize=(11.2, 10.8))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    lanes_y = [(0.865, 0.985), (0.735, 0.855), (0.605, 0.725),
               (0.235, 0.595), (0.09, 0.225)]
    for (label, face, edge), (y0, y1) in zip(LANES, lanes_y):
        lane_box(ax, y0, y1, label, face, edge)

    cx = 0.52
    node(ax, cx, 0.925, 0.46, 0.082, "Structured trial summary",
         "ClinicalTrials.gov → disease, regimen, endpoint, units, (yᵢ, nᵢ)")
    arrow(ax, cx, 0.868, 0.848)
    node(ax, cx, 0.795, 0.46, 0.082, "Stage 1 retrieval",
         "Bio_ClinicalBERT multi-aspect + SECRET pool → top 100")
    arrow(ax, cx, 0.738, 0.718)
    node(ax, cx, 0.665, 0.46, 0.082, "Stage 2 explainable reranking",
         "9 borrowability features + conservative gate → top 10")
    arrow(ax, cx, 0.608, 0.585, label="candidates + features")

    node(ax, cx, 0.545, 0.46, 0.075, "Beta-binomial components",
         "αᵢ = 1 + aᵢyᵢ,   βᵢ = 1 + aᵢ(nᵢ−yᵢ)")
    arrow(ax, cx, 0.492, 0.472)

    # --- CORE: selective two-head model -----------------------------------
    node(ax, cx, 0.405, 0.50, 0.118, "Selective two-head DeepSets",
         "score head → mixture weight λᵢ  ·  discount head → discount aᵢ\n"
         "set-level weak head ψ → total borrowing 1−λ₀ (joint softmax)",
         edge=RED, title_color=RED, lw=2.2, face="#fdf6f4")
    ax.add_patch(FancyBboxPatch((0.185, 0.452), 0.155, 0.034,
                                boxstyle="round,pad=0.004,rounding_size=0.014",
                                facecolor=RED, edgecolor=RED, zorder=7))
    ax.text(0.2625, 0.469, "CORE METHOD", ha="center", va="center",
            fontsize=8.6, fontweight="bold", color="white", zorder=8)

    # side panel: mixture prior with EB anchor
    node(ax, 0.875, 0.405, 0.185, 0.118, "Mixture prior",
         "p(θ) = λ₀ Beta(a₀,b₀)\n"
         "      + Σᵢ λᵢ Beta(αᵢ,βᵢ)\n"
         "EB anchor (a₀,b₀); λ₀ learned\n"
         "(ablation: λ₀=0.2, Beta(1,1))",
         edge=RED, title_color=RED, lw=1.6, face="#fdf6f4",
         title_size=9.6, sub_size=7.6)
    ax.add_patch(FancyArrowPatch((cx + 0.25, 0.405), (0.782, 0.405),
                                 arrowstyle="-|>", mutation_scale=13,
                                 linewidth=1.4, color=RED, zorder=6))

    arrow(ax, cx, 0.346, 0.326)
    node(ax, cx, 0.288, 0.46, 0.068, "SAM conflict adapter",
         "under prior–data conflict, shrinks historical mass back "
         "toward the anchored weak component",
         edge=RED, lw=1.6)
    arrow(ax, cx, 0.254, 0.228, label="assembled prior p(θ)")

    node(ax, cx, 0.157, 0.50, 0.082, "EB-referenced, label-free validation",
         "held-out NLL vs intercept-only EB · disjoint forward windows ·\n"
         "calibration · known-truth simulation operating characteristics")

    ax.text(0.5, 0.038,
            "Retrievability ≠ comparability ≠ borrowing behaviour: retrieval and reranking find candidates;\n"
            "the selective two-head prior decides whether, from whom, and how much to borrow.",
            ha="center", va="center", fontsize=9.2, style="italic", color="#44505e")

    for path in (OUT_MAN / "F7_pipeline_schematic.pdf",
                 OUT_RES / "F7_pipeline_schematic.pdf"):
        path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(path, bbox_inches="tight")
        print("wrote", path)
    fig.savefig(OUT_RES / "F7_pipeline_schematic.tif", dpi=300, bbox_inches="tight")
    print("wrote", OUT_RES / "F7_pipeline_schematic.tif")


if __name__ == "__main__":
    main()
