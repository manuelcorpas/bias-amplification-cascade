#!/usr/bin/env python3
"""
Generate Publication-Ready Figures
===================================
Produces Figures 1-5 for the Communications Biology review:
  - Figure 1: Bias Amplification Cascade (schematic)
  - Figure 2: Three Invisibilities framework (three-panel)
  - Figure 3: SII vs Semantic Similarity scatter (+ partial correlation)
  - Figure 4: Disease x LLM Heatmap (ordered by SII)
  - Figure 5: Probe-level SII gradient (knowledge chain)

Author: Manuel Corpas
Date: February 2026
"""

import csv
import json
from pathlib import Path

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Patch
from matplotlib.colors import LinearSegmentedColormap
from scipy import stats
from scipy.stats import gaussian_kde

# ---------------------------------------------------------------------------
# Global style: Nature/Cell-quality defaults
# ---------------------------------------------------------------------------
matplotlib.rcParams.update({
    "figure.dpi": 150,
    "savefig.dpi": 600,
    "font.family": "sans-serif",
    "font.sans-serif": ["Helvetica Neue", "Helvetica", "Arial", "DejaVu Sans"],
    "font.size": 8,
    "axes.titlesize": 10,
    "axes.labelsize": 9,
    "xtick.labelsize": 7.5,
    "ytick.labelsize": 7.5,
    "legend.fontsize": 7,
    "axes.linewidth": 0.6,
    "xtick.major.width": 0.6,
    "ytick.major.width": 0.6,
    "xtick.major.size": 3,
    "ytick.major.size": 3,
    "xtick.minor.size": 1.5,
    "ytick.minor.size": 1.5,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "lines.linewidth": 1.2,
    "patch.linewidth": 0.5,
    "pdf.fonttype": 42,  # TrueType fonts in PDF
    "ps.fonttype": 42,
    "mathtext.default": "regular",
    "figure.facecolor": "white",
    "axes.facecolor": "white",
    "savefig.facecolor": "white",
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.1,
})

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
RESULTS_DIR = Path(__file__).parent.parent / "RESULTS"
FIGURES_DIR = Path(__file__).parent.parent / "DOCX" / "FIGURES"
SCORES_DIR = RESULTS_DIR / "SCORES"
DATA_DIR = Path(__file__).parent.parent / "DATA"

MERGED_CSV = RESULTS_DIR / "sii_correlation_results.csv"
CORR_JSON = RESULTS_DIR / "sii_correlation_results.json"
ALL_SCORES = SCORES_DIR / "all_scores.json"
GT_CSV = DATA_DIR / "ground-truth" / "disease_ground_truth.csv"

# ---------------------------------------------------------------------------
# Colour palette (Nature-style, colourblind-friendly)
# ---------------------------------------------------------------------------
CATEGORY_COLOURS = {
    "NTD": "#C44E52",
    "Infectious": "#4C72B0",
    "NCD-Cancer": "#55A868",
    "NCD-CVD": "#DD8452",
    "NCD-Other": "#8172B3",
    "Mental": "#64B5CD",
    "Injury": "#DA8BC3",
    "MNCH": "#CCB974",
    "Nutritional": "#8C8C8C",
    "Other": "#BFBFBF",
}

# Diseases to label on scatter
LABEL_DISEASES = [
    "African_trypanosomiasis",
    "Onchocerciasis",
    "Lymphatic_filariasis",
    "Ebola",
    "Malaria",
    "Tuberculosis",
    "HIV_AIDS",
    "Diabetes_mellitus",
    "Breast_cancer",
    "Ischemic_heart_disease",
    "COVID-19",
    "Tracheal",
    "Vitamin_A_deficiency",
    "Iodine_deficiency",
]

# Custom diverging colormap for heatmap
HEATMAP_CMAP = LinearSegmentedColormap.from_list(
    "equity",
    ["#C44E52", "#E8866A", "#F5C48C", "#FFFFBF", "#A8D8A8", "#55A868", "#2D7D46"],
    N=256,
)


def _panel_label(ax, label, x=-0.08, y=1.06):
    """Add bold panel label (a, b, c...) in Nature style."""
    ax.text(x, y, label, transform=ax.transAxes,
            fontsize=12, fontweight="bold", va="top", ha="left")


# ===================================================================
# FIGURE 1: The Bias Amplification Cascade
# ===================================================================
def generate_figure1():
    fig, ax = plt.subplots(figsize=(190 / 25.4, 90 / 25.4))  # 190mm x 90mm
    ax.set_xlim(-0.2, 13.5)
    ax.set_ylim(-1.8, 4.8)
    ax.axis("off")

    stages = [
        {
            "label": "Data\nCollection",
            "detail": "African health systems\nproduce fewer structured\ndigital records",
            "stat": "<7% of global\nbiomedical research",
            "colour": "#E65100",
            "bg": "#FFF3E0",
        },
        {
            "label": "Literature\nCuration",
            "detail": "93.5% biobank papers\nfrom high-income\ncountries",
            "stat": "57.8 : 1\nHIC-to-LMIC per DALY",
            "colour": "#BF360C",
            "bg": "#FBE9E7",
        },
        {
            "label": "Semantic\nEmbedding",
            "detail": "Neglected diseases\nsemantically isolated\nin PubMed",
            "stat": "NTDs 44% more\nisolated (P < 0.0001)",
            "colour": "#880E4F",
            "bg": "#FCE4EC",
        },
        {
            "label": "LLM\nTraining",
            "detail": "Corpora dominated\nby English, Global\nNorth sources",
            "stat": "Common Crawl,\nPubMed skew to HIC",
            "colour": "#6A1B9A",
            "bg": "#F3E5F5",
        },
        {
            "label": "LLM\nOutput",
            "detail": "Knowledge gaps\ncorrelate with\nstructural neglect",
            "stat": "R\u00b2 = 0.666\nn = 175 diseases",
            "colour": "#283593",
            "bg": "#E8EAF6",
        },
        {
            "label": "Clinical\nDecisions",
            "detail": "Diagnostic errors,\ntreatment gaps,\npolicy blind spots",
            "stat": "Diagnostics most\naffected (r = \u22120.313)",
            "colour": "#004D40",
            "bg": "#E0F2F1",
        },
    ]

    box_w, box_h = 1.75, 3.6
    gap = 0.45
    start_x = 0.1
    y_centre = 1.6

    for i, s in enumerate(stages):
        x = start_x + i * (box_w + gap)
        y = y_centre - box_h / 2

        # Shadow
        shadow = FancyBboxPatch(
            (x + 0.04, y - 0.04), box_w, box_h,
            boxstyle="round,pad=0.10",
            facecolor="#00000008", edgecolor="none", zorder=1,
        )
        ax.add_patch(shadow)

        # Main box
        rect = FancyBboxPatch(
            (x, y), box_w, box_h,
            boxstyle="round,pad=0.10",
            facecolor=s["bg"], edgecolor=s["colour"],
            linewidth=1.2, zorder=2,
        )
        ax.add_patch(rect)

        # Stage number badge
        badge = FancyBboxPatch(
            (x + box_w / 2 - 0.18, y + box_h - 0.42), 0.36, 0.32,
            boxstyle="round,pad=0.06",
            facecolor=s["colour"], edgecolor="none", zorder=3,
        )
        ax.add_patch(badge)
        ax.text(x + box_w / 2, y + box_h - 0.26, str(i + 1),
                ha="center", va="center", fontsize=7,
                fontweight="bold", color="white", zorder=4)

        # Title
        ax.text(x + box_w / 2, y + box_h - 0.75, s["label"],
                ha="center", va="top", fontsize=6.5,
                fontweight="bold", color=s["colour"],
                linespacing=1.15, zorder=3)

        # Separator line
        ax.plot([x + 0.2, x + box_w - 0.2],
                [y + box_h * 0.52, y + box_h * 0.52],
                color=s["colour"], alpha=0.15, linewidth=0.5, zorder=3)

        # Detail
        ax.text(x + box_w / 2, y + box_h * 0.35, s["detail"],
                ha="center", va="center", fontsize=5,
                color="#444444", linespacing=1.3, zorder=3)

        # Stat
        ax.text(x + box_w / 2, y + 0.45, s["stat"],
                ha="center", va="center", fontsize=5,
                color=s["colour"], fontstyle="italic",
                linespacing=1.25, zorder=3)

        # Chevron arrow to next
        if i < len(stages) - 1:
            intervention = i in [0, 2, 3]  # arrows 1→2, 3→4, 4→5
            ax_x = x + box_w + 0.06
            if intervention:
                # Bold green intervention arrows: thicker, brighter, larger heads
                ax.annotate(
                    "", xy=(ax_x + gap - 0.12, y_centre),
                    xytext=(ax_x, y_centre),
                    arrowprops=dict(
                        arrowstyle="-|>,head_width=0.32,head_length=0.18",
                        color="#1B8A2A", linewidth=2.6,
                        connectionstyle="arc3,rad=0",
                    ),
                    zorder=5,
                )
            else:
                # Muted grey cascade arrows: thinner, lighter
                ax.annotate(
                    "", xy=(ax_x + gap - 0.12, y_centre),
                    xytext=(ax_x, y_centre),
                    arrowprops=dict(
                        arrowstyle="-|>,head_width=0.18,head_length=0.12",
                        color="#B0BEC5", linewidth=1.0,
                        connectionstyle="arc3,rad=0",
                    ),
                    zorder=5,
                )

    # Bottom gradient bar
    for ix in range(200):
        frac = ix / 200
        c = matplotlib.colors.to_rgba("#C62828", alpha=0.05 + 0.35 * frac)
        ax.barh(-0.55, 0.065, left=0.1 + frac * 13.0, height=0.18,
                color=c, zorder=1)
    ax.text(6.7, -0.55, "BIAS AMPLIFICATION",
            ha="center", va="center", fontsize=6,
            fontweight="bold", color="#C62828", alpha=0.8, zorder=2)

    # Intervention legend (small, bottom-right)
    ax.plot([], [], marker=">", color="#1B8A2A", markersize=5,
            linestyle="-", linewidth=2.2, label="Intervention point")
    ax.legend(loc="lower right", fontsize=5, framealpha=0.9,
              edgecolor="#CCCCCC", handletextpad=0.4, borderpad=0.3)

    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIGURES_DIR / "fig1_cascade.pdf")
    fig.savefig(FIGURES_DIR / "fig1_cascade.png")
    print(f"  Saved Figure 1")
    plt.close(fig)


# ===================================================================
# FIGURE 2: The Three Invisibilities
# ===================================================================
def generate_figure2():
    fig = plt.figure(figsize=(190 / 25.4, 110 / 25.4))  # 190mm x 110mm
    gs = fig.add_gridspec(1, 3, wspace=0.10, left=0.02, right=0.98,
                          top=0.90, bottom=0.04)

    panels = [
        {
            "label": "a",
            "title": "Data Invisibility",
            "subtitle": "What LLMs Cannot See",
            "colour": "#C44E52",
            "items": [
                ("93.5%", "biobank publications from\nhigh-income countries"),
                ("57.8 : 1", "HIC-to-LMIC research\nratio per DALY"),
                ("22", "GBD categories with\ncritical research gaps"),
                ("<7%", "global biomedical research\nfrom Africa"),
            ],
        },
        {
            "label": "b",
            "title": "Epistemic Invisibility",
            "subtitle": "What LLMs Do Not Know",
            "colour": "#4C72B0",
            "items": [
                ("R\u00b2 = 0.666", "structural neglect model\nexplains 67% of LLM variance"),
                ("r = 0.662", "publication volume predicts\nLLM alignment with PubMed"),
                ("r = \u22120.313", "diagnostics most affected\nby semantic isolation"),
                ("r = 0.331", "partial correlation\ncontrolling for volume"),
            ],
        },
        {
            "label": "c",
            "title": "Infrastructural Invisibility",
            "subtitle": "Who Gets Left Behind",
            "colour": "#2D7D46",
            "items": [
                ("< 1%", "of global AI compute\nin Africa"),
                ("2,000+", "African languages with\nminimal LLM support"),
                ("0.6%", "safety-mediated query\nrefusals (Claude Opus 4.5)"),
                ("Nascent", "AI health regulation\nacross AU member states"),
            ],
        },
    ]

    for idx, panel in enumerate(panels):
        ax = fig.add_subplot(gs[0, idx])
        ax.set_xlim(0, 10)
        ax.set_ylim(0, 10)
        ax.axis("off")

        # Card background
        card = FancyBboxPatch(
            (0.15, 0.15), 9.7, 9.7,
            boxstyle="round,pad=0.25",
            facecolor="white",
            edgecolor=panel["colour"],
            linewidth=1.5, zorder=1,
        )
        ax.add_patch(card)

        # Header stripe
        header = FancyBboxPatch(
            (0.15, 8.2), 9.7, 1.65,
            boxstyle="round,pad=0.25",
            facecolor=panel["colour"],
            edgecolor="none",
            zorder=2,
        )
        ax.add_patch(header)
        # Clip bottom corners of header to look flat
        clip_rect = plt.Rectangle((0, 8.2), 10, 0.5,
                                   facecolor=panel["colour"],
                                   edgecolor="none", zorder=2)
        ax.add_patch(clip_rect)

        # Panel label
        ax.text(0.7, 9.5, panel["label"],
                fontsize=11, fontweight="bold", color="white",
                va="center", ha="left", zorder=3)

        # Title
        ax.text(5.0, 9.5, panel["title"],
                fontsize=9, fontweight="bold", color="white",
                va="center", ha="center", zorder=3)
        ax.text(5.0, 8.7, panel["subtitle"],
                fontsize=6.5, color="white", alpha=0.9,
                va="center", ha="center",
                fontstyle="italic", zorder=3)

        # Stat items
        y_pos = 7.3
        for value, desc in panel["items"]:
            # Thin separator
            ax.plot([0.8, 9.2], [y_pos + 0.7, y_pos + 0.7],
                    color=panel["colour"], alpha=0.08, linewidth=0.5, zorder=2)

            # Value (left-aligned, smaller)
            ax.text(0.9, y_pos + 0.15, value,
                    fontsize=9, fontweight="bold",
                    color=panel["colour"], ha="left", va="center", zorder=3)
            # Description (right side, clear separation)
            ax.text(5.2, y_pos + 0.1, desc,
                    fontsize=6, color="#444444",
                    ha="left", va="center",
                    linespacing=1.25, zorder=3)
            y_pos -= 1.8

    # Bidirectional arrows between panels
    for i in range(2):
        x1 = 0.345 + i * 0.328
        x2 = x1 + 0.015
        fig.patches.append(FancyArrowPatch(
            posA=(x1, 0.46), posB=(x2, 0.46),
            arrowstyle="<->,head_width=3,head_length=3",
            color="#78909C", linewidth=1.0, alpha=0.6,
            transform=fig.transFigure, figure=fig,
        ))

    fig.savefig(FIGURES_DIR / "fig2_three_invisibilities.pdf")
    fig.savefig(FIGURES_DIR / "fig2_three_invisibilities.png")
    print(f"  Saved Figure 2")
    plt.close(fig)


# ===================================================================
# Data loaders
# ===================================================================
def load_merged_data():
    data = []
    with open(MERGED_CSV, "r") as f:
        for row in csv.DictReader(f):
            data.append({
                "disease": row["disease"],
                "sii": float(row["sii"]),
                "mean_semantic_similarity": float(row["mean_semantic_similarity"]),
                "n_papers": int(row["n_papers"]),
                "log_papers": float(row["log_papers"]),
            })
    return data


def load_categories():
    cats = {}
    with open(GT_CSV, "r") as f:
        for row in csv.DictReader(f):
            cats[row["disease"]] = row.get("category", "Other")
    return cats


# ===================================================================
# FIGURE 3: SII scatter + partial correlation
# ===================================================================
def generate_figure3(data, categories, corr_results):
    fig = plt.figure(figsize=(180 / 25.4, 85 / 25.4))  # 180mm x 85mm

    # Layout: main scatter with marginals + partial correlation panel
    gs = fig.add_gridspec(
        2, 3,
        width_ratios=[3, 0.4, 1.8],
        height_ratios=[0.4, 3],
        wspace=0.05, hspace=0.05,
        left=0.08, right=0.97, top=0.94, bottom=0.12,
    )

    ax_main = fig.add_subplot(gs[1, 0])
    ax_top = fig.add_subplot(gs[0, 0], sharex=ax_main)
    ax_right = fig.add_subplot(gs[1, 1], sharey=ax_main)
    ax_partial = fig.add_subplot(gs[1, 2])
    # Shift panel b right to avoid y-label overlapping the right marginal KDE
    pos = ax_partial.get_position()
    ax_partial.set_position([pos.x0 + 0.08, pos.y0, pos.width - 0.08, pos.height])
    # Hide corner
    ax_corner = fig.add_subplot(gs[0, 1])
    ax_corner.axis("off")
    ax_corner2 = fig.add_subplot(gs[0, 2])
    ax_corner2.axis("off")

    sii = np.array([d["sii"] for d in data])
    score = np.array([d["mean_semantic_similarity"] for d in data])
    n_papers = np.array([d["n_papers"] for d in data])

    # --- Main scatter ---
    for d in data:
        cat = categories.get(d["disease"], "Other")
        colour = CATEGORY_COLOURS.get(cat, "#BFBFBF")
        size = max(12, min(120, d["n_papers"] / 1500))
        ax_main.scatter(
            d["sii"], d["mean_semantic_similarity"],
            c=colour, s=size, alpha=0.65,
            edgecolors="white", linewidth=0.3, zorder=2,
        )

    # Regression
    slope, intercept, r_val, p_val, se_slope = stats.linregress(sii, score)
    x_line = np.linspace(sii.min(), sii.max(), 200)
    y_line = slope * x_line + intercept
    ax_main.plot(x_line, y_line, color="#333333", linewidth=1.2, zorder=3)

    # 95% CI
    n = len(sii)
    residuals = score - (slope * sii + intercept)
    se = np.sqrt(np.sum(residuals ** 2) / (n - 2))
    x_mean = np.mean(sii)
    x_ss = np.sum((sii - x_mean) ** 2)
    ci = 1.96 * se * np.sqrt(1 / n + (x_line - x_mean) ** 2 / x_ss)
    ax_main.fill_between(x_line, y_line - ci, y_line + ci,
                         alpha=0.10, color="#333333", zorder=1)

    # Label diseases with manual offsets to avoid overlap
    label_offsets = {
        "COVID-19": (8, 8),
        "Malaria": (8, 6),
        "Tuberculosis": (-40, 12),
        "HIV_AIDS": (-35, -12),
        "Diabetes_mellitus": (8, -10),
        "Breast_cancer": (-50, -10),
        "Ischemic_heart_disease": (8, -8),
        "Ebola": (8, 4),
        "African_trypanosomiasis": (8, 6),
        "Onchocerciasis": (8, -8),
        "Lymphatic_filariasis": (8, -6),
        "Tracheal": (8, -8),
        "Vitamin_A_deficiency": (-55, 8),
        "Iodine_deficiency": (8, 8),
    }
    for d in data:
        if d["disease"] in LABEL_DISEASES:
            label = d["disease"].replace("_", " ")
            if label == "HIV AIDS":
                label = "HIV/AIDS"
            if len(label) > 28:
                label = label[:25] + "..."
            offset = label_offsets.get(d["disease"], (6, 4))
            ax_main.annotate(
                label,
                (d["sii"], d["mean_semantic_similarity"]),
                xytext=offset, textcoords="offset points",
                fontsize=4, color="#333333",
                arrowprops=dict(arrowstyle="-", color="#AAAAAA",
                                linewidth=0.3),
                bbox=dict(boxstyle="round,pad=0.12", facecolor="white",
                          edgecolor="none", alpha=0.8),
                zorder=4,
            )

    # Stats box
    r_p = corr_results.get("h1_pearson_r", r_val)
    p_p = corr_results.get("h1_pearson_p", p_val)
    r_s = corr_results.get("h1_spearman_rho", 0)
    stats_text = (
        f"Pearson r = {r_p:.3f}\n"
        f"Spearman \u03c1 = {r_s:.3f}\n"
        f"n = {n}"
    )
    ax_main.text(
        0.97, 0.97, stats_text, transform=ax_main.transAxes,
        fontsize=6, va="top", ha="right", linespacing=1.4,
        bbox=dict(boxstyle="round,pad=0.4", facecolor="white",
                  edgecolor="#CCCCCC", linewidth=0.5, alpha=0.9),
    )

    ax_main.set_xlabel("Semantic Isolation Index (SII)")
    ax_main.set_ylabel("Mean semantic similarity to PubMed literature")

    # Category legend
    used_cats = set(categories.get(d["disease"], "Other") for d in data)
    for cat in sorted(CATEGORY_COLOURS):
        if cat in used_cats:
            ax_main.scatter([], [], c=CATEGORY_COLOURS[cat], s=18,
                            label=cat, alpha=0.8, edgecolors="white", linewidth=0.3)
    ax_main.legend(loc="lower left", fontsize=5, ncol=2,
                   framealpha=0.9, edgecolor="#CCCCCC",
                   handletextpad=0.3, columnspacing=0.8,
                   borderpad=0.4)

    # --- Marginal: top (SII KDE) ---
    kde_x = gaussian_kde(sii, bw_method=0.3)
    x_grid = np.linspace(sii.min(), sii.max(), 200)
    ax_top.fill_between(x_grid, kde_x(x_grid), color="#78909C", alpha=0.3)
    ax_top.plot(x_grid, kde_x(x_grid), color="#78909C", linewidth=0.8)
    ax_top.set_yticks([])
    ax_top.spines["left"].set_visible(False)
    ax_top.spines["bottom"].set_visible(False)
    plt.setp(ax_top.get_xticklabels(), visible=False)

    # --- Marginal: right (score KDE) ---
    kde_y = gaussian_kde(score, bw_method=0.3)
    y_grid = np.linspace(score.min(), score.max(), 200)
    ax_right.fill_betweenx(y_grid, kde_y(y_grid), color="#78909C", alpha=0.3)
    ax_right.plot(kde_y(y_grid), y_grid, color="#78909C", linewidth=0.8)
    ax_right.set_xticks([])
    ax_right.spines["bottom"].set_visible(False)
    ax_right.spines["left"].set_visible(False)
    plt.setp(ax_right.get_yticklabels(), visible=False)

    _panel_label(ax_main, "a", x=-0.10, y=1.15)

    # --- Panel B: Partial correlation ---
    log_papers = np.log10(n_papers)

    sl_x, int_x, _, _, _ = stats.linregress(log_papers, sii)
    resid_sii = sii - (sl_x * log_papers + int_x)

    sl_y, int_y, _, _, _ = stats.linregress(log_papers, score)
    resid_score = score - (sl_y * log_papers + int_y)

    for d, rs, rsc in zip(data, resid_sii, resid_score):
        cat = categories.get(d["disease"], "Other")
        colour = CATEGORY_COLOURS.get(cat, "#BFBFBF")
        ax_partial.scatter(rs, rsc, c=colour, s=14, alpha=0.55,
                           edgecolors="white", linewidth=0.25, zorder=2)

    sl_r, int_r, r_r, p_r, _ = stats.linregress(resid_sii, resid_score)
    x_r = np.linspace(resid_sii.min(), resid_sii.max(), 200)
    ax_partial.plot(x_r, sl_r * x_r + int_r, color="#333333",
                    linewidth=1.2, zorder=3)

    partial_r = corr_results.get("h2_partial_r", r_r)
    partial_p = corr_results.get("h2_partial_p", p_r)
    ax_partial.text(
        0.97, 0.97,
        f"Partial r = {partial_r:.3f}\np = {partial_p:.1e}",
        transform=ax_partial.transAxes, fontsize=6,
        va="top", ha="right", linespacing=1.4,
        bbox=dict(boxstyle="round,pad=0.4", facecolor="white",
                  edgecolor="#CCCCCC", linewidth=0.5, alpha=0.9),
    )

    ax_partial.set_xlabel("SII residual | log(papers)")
    ax_partial.set_ylabel("Similarity residual | log(papers)", labelpad=8)
    _panel_label(ax_partial, "b", x=-0.18, y=1.15)

    fig.savefig(FIGURES_DIR / "fig3_sii_scatter.pdf")
    fig.savefig(FIGURES_DIR / "fig3_sii_scatter.png")
    print(f"  Saved Figure 3")
    plt.close(fig)


# ===================================================================
# FIGURE 4: Disease x LLM Heatmap (top/bottom 30 by SII)
# ===================================================================
def generate_figure4(all_scores_data, categories, heim_diseases):
    models = [
        "Gemini_3_Pro", "Claude_Opus_4.5", "Claude_Sonnet_4",
        "GPT-5.2", "Mistral_Large", "DeepSeek_V3",
    ]
    model_display = [
        "Gemini 3\nPro", "Claude\nOpus 4.5", "Claude\nSonnet 4",
        "GPT-5.2", "Mistral\nLarge", "DeepSeek\nV3",
    ]

    # Sort all diseases by SII descending
    all_sorted = sorted(heim_diseases, key=lambda x: float(x["sii"]),
                        reverse=True)

    # Select top 30 most isolated and bottom 30 least isolated
    N_SHOW = 30
    top_diseases = [d["disease"] for d in all_sorted[:N_SHOW]]
    bottom_diseases = [d["disease"] for d in all_sorted[-N_SHOW:]]

    # Build score matrix for all diseases (we need global range)
    score_map = {}
    for s in all_scores_data:
        if s.get("semantic_similarity") is None:
            continue
        key = (s["disease"], s["model"])
        if key not in score_map:
            score_map[key] = []
        score_map[key].append(s["semantic_similarity"])

    def build_matrix(disease_list):
        mat = np.full((len(disease_list), len(models)), np.nan)
        for i, disease in enumerate(disease_list):
            for j, model in enumerate(models):
                key = (disease, model)
                if key in score_map:
                    mat[i, j] = np.mean(score_map[key])
        return mat

    mat_top = build_matrix(top_diseases)
    mat_bottom = build_matrix(bottom_diseases)

    # Global colour range from all data
    all_disease_order = [d["disease"] for d in all_sorted]
    mat_all = build_matrix(all_disease_order)
    valid_vals = mat_all[~np.isnan(mat_all)]
    vmin = np.percentile(valid_vals, 1)
    vmax = np.percentile(valid_vals, 99)

    # --- Layout: two panels stacked ---
    fig = plt.figure(figsize=(220 / 25.4, 200 / 25.4))
    gs = fig.add_gridspec(
        2, 3,
        width_ratios=[0.025, 1, 0.04],
        height_ratios=[1, 1],
        hspace=0.28, wspace=0.01,
        left=0.42, right=0.82, top=0.93, bottom=0.06,
    )

    panels = [
        (mat_top, top_diseases, "30 most semantically isolated diseases"),
        (mat_bottom, bottom_diseases, "30 least semantically isolated diseases"),
    ]
    panel_labels = ["a", "b"]

    for pidx, (mat, diseases, title) in enumerate(panels):
        ax_bar = fig.add_subplot(gs[pidx, 0])
        ax_heat = fig.add_subplot(gs[pidx, 1])

        # Heatmap
        im = ax_heat.imshow(
            mat, aspect="auto", cmap=HEATMAP_CMAP,
            vmin=vmin, vmax=vmax, interpolation="nearest",
        )

        # Disease labels
        labels = [d.replace("_", " ") for d in diseases]
        ax_heat.set_yticks(range(len(labels)))
        ax_heat.set_yticklabels(labels, fontsize=5.5)
        ax_heat.tick_params(axis="y", length=0, pad=8)

        # Model labels on top of each panel
        ax_heat.set_xticks(range(len(model_display)))
        ax_heat.set_xticklabels(model_display, fontsize=5.5,
                                 fontweight="bold", ha="center",
                                 linespacing=0.9)
        ax_heat.xaxis.set_ticks_position("top")
        ax_heat.tick_params(axis="x", length=0, pad=4)

        # Thin gridlines between cells
        for y in range(len(diseases) + 1):
            ax_heat.axhline(y - 0.5, color="white", linewidth=0.3)
        for x in range(len(models) + 1):
            ax_heat.axvline(x - 0.5, color="white", linewidth=0.3)

        # Remove spines
        for spine in ax_heat.spines.values():
            spine.set_visible(False)

        # Panel title
        ax_heat.set_title(title, fontsize=7, fontweight="bold",
                          pad=28, loc="left", color="#333333")

        # Panel label
        _panel_label(ax_heat, panel_labels[pidx], x=-0.55, y=1.09)

        # Category sidebar
        ax_bar.set_xlim(0, 1)
        ax_bar.set_ylim(len(diseases) - 0.5, -0.5)
        for i, d in enumerate(diseases):
            cat = categories.get(d, "Other")
            colour = CATEGORY_COLOURS.get(cat, "#BFBFBF")
            ax_bar.add_patch(plt.Rectangle((0, i - 0.5), 1, 1,
                                            color=colour, linewidth=0))
        ax_bar.set_xticks([])
        ax_bar.set_yticks([])
        for spine in ax_bar.spines.values():
            spine.set_visible(False)

    # Colorbar (shared, on right side)
    cbar_ax = fig.add_subplot(gs[:, 2])
    cbar_ax.axis("off")
    cbar = fig.colorbar(im, ax=cbar_ax, shrink=0.8, aspect=30,
                        pad=0.5, location="right")
    cbar.set_label("Semantic similarity to PubMed literature", fontsize=6.5)
    cbar.ax.tick_params(labelsize=6)
    cbar.outline.set_linewidth(0.4)

    # Category legend at bottom
    used_cats = set()
    for d in top_diseases + bottom_diseases:
        used_cats.add(categories.get(d, "Other"))
    legend_handles = [Patch(facecolor=CATEGORY_COLOURS[c], label=c,
                            edgecolor="#CCCCCC", linewidth=0.3)
                      for c in sorted(CATEGORY_COLOURS) if c in used_cats]
    fig.legend(handles=legend_handles, loc="lower center",
               fontsize=5.5, ncol=5, frameon=False,
               handletextpad=0.4, columnspacing=1.0,
               bbox_to_anchor=(0.55, 0.01))

    fig.savefig(FIGURES_DIR / "fig4_heatmap.pdf")
    fig.savefig(FIGURES_DIR / "fig4_heatmap.png")
    print(f"  Saved Figure 4")
    plt.close(fig)


# ===================================================================
# FIGURE 5: Probe-level SII gradient
# ===================================================================
def generate_figure5(corr_results):
    probe_corrs = corr_results.get("probe_correlations", {})
    if not probe_corrs:
        print("  Skipping Figure 5")
        return

    ranked = sorted(probe_corrs.items(), key=lambda x: x[1]["r"])

    clean_names = []
    dims = []
    for p, v in ranked:
        name = p.split("_", 1)[1] if "_" in p else p
        name = name.replace("_", " ").title()
        clean_names.append(name)
        dims.append(v.get("dimension", ""))

    rs = [v["r"] for _, v in ranked]
    ps = [v["p"] for _, v in ranked]

    # Colour by significance
    sig_colours = {
        "***": "#C44E52",
        "**": "#DD8452",
        "*": "#4C72B0",
        "ns": "#BFBFBF",
    }
    colours = []
    sigs = []
    for p_val in ps:
        if p_val < 0.001:
            colours.append(sig_colours["***"])
            sigs.append("***")
        elif p_val < 0.01:
            colours.append(sig_colours["**"])
            sigs.append("**")
        elif p_val < 0.05:
            colours.append(sig_colours["*"])
            sigs.append("*")
        else:
            colours.append(sig_colours["ns"])
            sigs.append("ns")

    fig, ax = plt.subplots(figsize=(130 / 25.4, 70 / 25.4))  # ~130mm x 70mm

    y_pos = np.arange(len(clean_names))
    bars = ax.barh(y_pos, rs, color=colours, height=0.6,
                   edgecolor="white", linewidth=0.3)

    ax.set_yticks(y_pos)
    ax.set_yticklabels(
        [f"{n}  ({d})" for n, d in zip(clean_names, dims)],
        fontsize=6,
    )
    ax.invert_yaxis()

    ax.axvline(0, color="#333333", linewidth=0.6)
    ax.set_xlabel("Pearson r (SII vs semantic similarity)")

    # Annotations: place outside bars to avoid overlap with labels
    for i, (r, sig) in enumerate(zip(rs, sigs)):
        # Always place annotation at a fixed x position to the right
        ax.text(0.035, i, f"r = {r:.3f}  {sig}",
                va="center", ha="left", fontsize=5, color="#555555")

    # Legend - top right to avoid overlapping bars
    legend_elements = [
        Patch(facecolor=sig_colours["***"], label="p < 0.001",
              edgecolor="white", linewidth=0.3),
        Patch(facecolor=sig_colours["**"], label="p < 0.01",
              edgecolor="white", linewidth=0.3),
        Patch(facecolor=sig_colours["*"], label="p < 0.05",
              edgecolor="white", linewidth=0.3),
        Patch(facecolor=sig_colours["ns"], label="n.s.",
              edgecolor="white", linewidth=0.3),
    ]
    ax.legend(handles=legend_elements, loc="lower left", fontsize=5.5,
              framealpha=0.9, edgecolor="#CCCCCC",
              handletextpad=0.3, borderpad=0.4)

    ax.set_xlim(-0.36, 0.12)
    ax.grid(axis="x", alpha=0.12, linewidth=0.4)

    _panel_label(ax, "", x=-0.12, y=1.06)

    plt.tight_layout()
    fig.savefig(FIGURES_DIR / "fig5_probe_gradient.pdf")
    fig.savefig(FIGURES_DIR / "fig5_probe_gradient.png")
    print(f"  Saved Figure 5")
    plt.close(fig)


# ===================================================================
# Main
# ===================================================================
def main():
    print("=" * 60)
    print("Generating Publication-Ready Figures")
    print("=" * 60)

    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    print("\nFigure 1 (bias amplification cascade)...")
    generate_figure1()

    print("Figure 2 (three invisibilities)...")
    generate_figure2()

    try:
        merged_data = load_merged_data()
        categories = load_categories()
    except FileNotFoundError as e:
        print(f"ERROR: {e}")
        raise SystemExit(1)

    try:
        with open(CORR_JSON, "r") as f:
            corr_results = json.load(f)
    except FileNotFoundError:
        corr_results = {}

    heim_diseases = []
    with open(DATA_DIR / "heim_diseases.csv", "r") as f:
        heim_diseases = list(csv.DictReader(f))

    print("Figure 3 (SII scatter + partial correlation)...")
    generate_figure3(merged_data, categories, corr_results)

    try:
        with open(ALL_SCORES, "r") as f:
            all_scores_data = json.load(f)
        print("Figure 4 (disease x LLM heatmap)...")
        generate_figure4(all_scores_data, categories, heim_diseases)
    except FileNotFoundError:
        print("Skipping Figure 4 (all_scores.json not found)")

    print("Figure 5 (probe-level SII gradient)...")
    generate_figure5(corr_results)

    print(f"\n{'=' * 60}")
    print(f"COMPLETE: {FIGURES_DIR}")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
