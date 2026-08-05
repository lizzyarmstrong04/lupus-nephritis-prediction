"""
DeLong test results figure for ESRD 5-year and 10-year prediction.
Grouped bar chart + significance brackets.
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

OUT_PATH = ("/Users/elizabetharmstrong/Library/CloudStorage/"
            "OneDrive-ImperialCollegeLondon/Lupus Project/"
            "outputs/esrd/figures/esrd_delong_figure.png")

# ── Data ──────────────────────────────────────────────────────
MODELS  = ["Logistic\nRegression", "Random\nForest", "XGBoost", "LightGBM"]
AUROC_5  = [0.752, 0.795, 0.779, 0.798]
AUROC_10 = [0.778, 0.804, 0.809, 0.804]

# Significant pairs: (idx_a, idx_b, p_adj_label)  — Holm-corrected
SIG_5  = [(0, 1, "p=0.007"), (0, 3, "p=0.028")]
SIG_10 = [(0, 1, "p=0.021"), (0, 2, "p=0.017"), (0, 3, "p=0.021")]

# ── Palette (validated slots 1–4, light mode) ─────────────────
COLORS = {
    "Logistic\nRegression": "#2a78d6",   # blue
    "Random\nForest":       "#1baf7a",   # aqua
    "XGBoost":              "#eda100",   # yellow
    "LightGBM":             "#008300",   # green
}
SURFACE  = "#fcfcfb"
INK      = "#0b0b0b"
INK_MUT  = "#898781"
GRIDLINE = "#e1e0d9"

# ── Significance bracket helper ────────────────────────────────
def sig_bracket(ax, x1, x2, y_top, tick_h, label, fontsize=8):
    """Draw a bracket between bars at x1 and x2 with a label."""
    ax.plot([x1, x1, x2, x2],
            [y_top - tick_h, y_top, y_top, y_top - tick_h],
            lw=1.2, color=INK, solid_capstyle="round")
    ax.text((x1 + x2) / 2, y_top + 0.001, label,
            ha="center", va="bottom", fontsize=fontsize,
            color=INK, fontweight="bold")

# ── Figure ────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(13, 6),
                         facecolor=SURFACE, sharey=False)
fig.subplots_adjust(wspace=0.08)

x     = np.arange(len(MODELS))
width = 0.55

datasets = [
    ("5-Year ESRD", AUROC_5,  SIG_5,  0.735, 0.860),
    ("10-Year ESRD", AUROC_10, SIG_10, 0.755, 0.875),
]

for ax, (title, aurocs, sig_pairs, ylo, yhi) in zip(axes, datasets):
    ax.set_facecolor(SURFACE)

    # Bars
    bars = ax.bar(x, aurocs, width,
                  color=[COLORS[m] for m in MODELS],
                  edgecolor=SURFACE, linewidth=1.5,
                  zorder=3)

    # 4px rounded data-ends via clip_on (matplotlib bars are rectangular;
    # simulate by adding a hairline white border)
    for bar in bars:
        bar.set_linewidth(2)
        bar.set_edgecolor(SURFACE)

    # Direct-label every bar (mandatory for sub-3:1 slots)
    for bar, val in zip(bars, aurocs):
        ax.text(bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.003,
                f"{val:.3f}",
                ha="center", va="bottom",
                fontsize=9, fontweight="bold", color=INK, zorder=5)

    # Significance brackets — stack from lowest to highest pair span
    tick_h  = (yhi - ylo) * 0.018
    bracket_y = ylo + (yhi - ylo) * 0.72   # start first bracket here

    for i, (ia, ib, label) in enumerate(sig_pairs):
        y = bracket_y + i * (yhi - ylo) * 0.085
        sig_bracket(ax, x[ia], x[ib], y, tick_h, label, fontsize=8)

    # Grid + axes
    ax.set_ylim(ylo, yhi)
    ax.set_xticks(x)
    ax.set_xticklabels(MODELS, fontsize=10, color=INK)
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.2f}"))
    ax.tick_params(axis="y", labelsize=9, colors=INK_MUT)
    ax.tick_params(axis="x", length=0)
    ax.set_ylabel("CV AUROC (5×10-fold OOF)", fontsize=10, color=INK,
                  labelpad=8)
    ax.set_title(title, fontsize=13, fontweight="bold", color=INK, pad=12)

    ax.yaxis.grid(True, color=GRIDLINE, linewidth=0.7, zorder=0)
    ax.set_axisbelow(True)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color(GRIDLINE)
    ax.spines["bottom"].set_color(GRIDLINE)

# Hide right panel y-axis label (panels share implicit axis)
axes[1].set_ylabel("")

# ── Legend ───────────────────────────────────────────────────
legend_handles = [
    mpatches.Patch(color=COLORS[m], label=m.replace("\n", " "))
    for m in MODELS
]
fig.legend(handles=legend_handles, loc="lower center", ncol=4,
           fontsize=9.5, frameon=False,
           bbox_to_anchor=(0.5, -0.04),
           labelcolor=INK)

# Footnote
fig.text(0.5, -0.10,
         "Brackets show pairs with significant AUROC difference (DeLong's test, Bonferroni-Holm corrected, p < 0.05).\n"
         "No significant differences among Random Forest, XGBoost, and LightGBM in either cohort.",
         ha="center", va="top", fontsize=8, color=INK_MUT, style="italic")

plt.tight_layout()
fig.savefig(OUT_PATH, dpi=180, bbox_inches="tight", facecolor=SURFACE)
plt.close("all")
print(f"Saved: {OUT_PATH}")
