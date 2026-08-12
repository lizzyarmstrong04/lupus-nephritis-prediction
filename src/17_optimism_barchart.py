"""
Grouped bar chart of apparent vs. bias-corrected (Harrell-optimism-corrected)
AUROC, across all five cohorts and four main classifiers.

Reads already-computed bootstrap results only (no recomputation) from:
  outputs/1yr_model_results.xlsx           sheet "Bootstrap (Harrell)"
  outputs/5yr_model_results.xlsx           sheet "Bootstrap (Harrell)"
  outputs/5yr_serial_model_results.xlsx    sheet "Bootstrap (Harrell)"
  outputs/esrd/esrd_model_results.xlsx     sheets "5yr Bootstrap" / "10yr Bootstrap"

The ESRD workbook uses different column names ("Apparent AUROC" / "Optimism
AUROC" / "BC AUROC") than the flare/serial workbooks ("Apparent AUROC" /
"Optimism (bootstrap)" / "Bias-Corrected AUROC (Harrell)") - both are handled.

For each cohort panel: paired bars per model (apparent = lighter shade,
bias-corrected = darker shade of the same per-model colour), so the gap
within each pair visually represents optimism/overfitting. Uses the
project's standard model colour scheme (src/1_year/06_modelling_1yr.py etc.):
  Logistic Regression #1f77b4, Random Forest #ff7f0e, XGBoost #2ca02c,
  LightGBM #d62728.

Saves: outputs/figures/optimism_barchart.png (300 dpi)
"""
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from matplotlib.colors import to_rgb

BASE = "/Users/elizabetharmstrong/Library/CloudStorage/OneDrive-ImperialCollegeLondon/Lupus Project"
OUT = f"{BASE}/outputs"
FIG_DIR = f"{OUT}/figures"

MODEL_COLORS = {
    "Logistic Regression": "#1f77b4",
    "Random Forest":       "#ff7f0e",
    "XGBoost":             "#2ca02c",
    "LightGBM":            "#d62728",
}
MODEL_ORDER = list(MODEL_COLORS)
MODEL_ABBR = {"Logistic Regression": "LR", "Random Forest": "RF", "XGBoost": "XGB", "LightGBM": "LGBM"}
PANEL_LETTERS = ["A", "B", "C", "D", "E"]

# NB: no DeLong significance annotation on this chart. DeLong's test compares
# out-of-fold (OOF) CV predictions, a third estimate distinct from both the
# "Apparent" (full-data fit) and "Bias-corrected" (Harrell bootstrap) values
# plotted here, so a significance finding from that test doesn't correspond
# to either bar shown - see roc_all_cohorts.png (src/16_roc_calibration_plots.py)
# instead, which annotates CV AUROC, the quantity DeLong's test actually used.


def lighten(hex_color, amount=0.55):
    """Blend a colour toward white by `amount` (0 = original, 1 = white)."""
    r, g, b = to_rgb(hex_color)
    return (r + (1 - r) * amount, g + (1 - g) * amount, b + (1 - b) * amount)


def load_bootstrap(path, sheet, apparent_col, bc_col):
    df = pd.read_excel(path, sheet_name=sheet)
    return {row["Model"]: (row[apparent_col], row[bc_col]) for _, row in df.iterrows()}


cohorts = [
    {"label": "1-Year Flare",
     "data": load_bootstrap(f"{OUT}/1yr_model_results.xlsx", "Bootstrap (Harrell)",
                             "Apparent AUROC", "Bias-Corrected AUROC (Harrell)")},
    {"label": "5-Year Flare",
     "data": load_bootstrap(f"{OUT}/5yr_model_results.xlsx", "Bootstrap (Harrell)",
                             "Apparent AUROC", "Bias-Corrected AUROC (Harrell)")},
    {"label": "Serial Biopsy",
     "data": load_bootstrap(f"{OUT}/5yr_serial_model_results.xlsx", "Bootstrap (Harrell)",
                             "Apparent AUROC", "Bias-Corrected AUROC (Harrell)")},
    {"label": "ESRD 5-Year",
     "data": load_bootstrap(f"{OUT}/esrd/esrd_model_results.xlsx", "5yr Bootstrap",
                             "Apparent AUROC", "BC AUROC")},
    {"label": "ESRD 10-Year",
     "data": load_bootstrap(f"{OUT}/esrd/esrd_model_results.xlsx", "10yr Bootstrap",
                             "Apparent AUROC", "BC AUROC")},
]

for c in cohorts:
    print(f"[{c['label']}]")
    for m in MODEL_ORDER:
        app, bc = c["data"][m]
        print(f"    {m:<20} Apparent={app:.3f}  BC={bc:.3f}  Optimism={app - bc:.3f}")

plt.rcParams["font.family"] = "Times New Roman"

GRID_ROWS, GRID_COLS = 2, 3  # 5 cohort panels + 1 legend panel


def panel_position(i):
    return divmod(i, GRID_COLS)


fig, axes = plt.subplots(GRID_ROWS, GRID_COLS, figsize=(12, 8), constrained_layout=True)

x = np.arange(len(MODEL_ORDER))
bar_w = 0.35

for i, c in enumerate(cohorts):
    row, col = panel_position(i)
    ax = axes[row, col]

    for j, name in enumerate(MODEL_ORDER):
        apparent, bc = c["data"][name]
        base = MODEL_COLORS[name]
        ax.bar(x[j] - bar_w / 2, apparent, width=bar_w, color=lighten(base),
               edgecolor="0.5", linewidth=0.5, zorder=3)
        ax.bar(x[j] + bar_w / 2, bc, width=bar_w, color=base,
               edgecolor="0.5", linewidth=0.5, zorder=3)

    ax.set_xticks(x)
    ax.set_xticklabels([MODEL_ABBR[m] for m in MODEL_ORDER], rotation=0, fontsize=12)
    ax.set_title(f"{PANEL_LETTERS[i]}. {c['label']}", fontsize=15, fontweight="bold", loc="left")
    # small headroom above 1.0 so tall bars aren't flush against the frame
    # (same fix applied to the ROC figure)
    ax.set_ylim(0.4, 1.02)
    ax.set_yticks(np.arange(0.4, 1.01, 0.1))
    ax.axhline(0.5, color="grey", linestyle="--", lw=0.8, alpha=0.5, zorder=2)
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_color("black")
        spine.set_linewidth(0.8)
    ax.tick_params(labelsize=11)
    if col == 0:
        ax.set_ylabel("AUROC", fontsize=13)
    else:
        ax.set_yticklabels([])

legend_row, legend_col = panel_position(len(cohorts))
legend_ax = axes[legend_row, legend_col]
legend_ax.axis("off")
legend_handles = [
    Patch(facecolor=lighten("#808080"), edgecolor="black", linewidth=0.6, label="Apparent"),
    Patch(facecolor="#808080", edgecolor="black", linewidth=0.6, label="Bias-corrected"),
]
legend_ax.legend(handles=legend_handles, loc="center", fontsize=18, frameon=False)

fig.text(0.5, -0.02, "LR = Logistic Regression; RF = Random Forest; XGB = XGBoost; LGBM = LightGBM",
         ha="center", fontsize=11, style="italic", color="0.3")

fig.savefig(f"{FIG_DIR}/optimism_barchart.png", dpi=300, bbox_inches="tight")
plt.close(fig)
print(f"\nSaved: {FIG_DIR}/optimism_barchart.png")
