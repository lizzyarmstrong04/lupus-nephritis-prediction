"""
SHAP comparison: ESRD (5yr, 10yr) vs Flare (1yr, 5yr).
Uses Mean across models column from each saved SHAP table.
Normalises within each outcome (divide by max) so four outcomes
are on a common 0-1 relative importance scale.
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

OUT_DIR   = "/Users/elizabetharmstrong/Library/CloudStorage/OneDrive-ImperialCollegeLondon/Lupus Project/outputs"
ESRD_DIR  = f"{OUT_DIR}/esrd"

# ── Load tables ───────────────────────────────────────────────
esrd5  = pd.read_excel(f"{ESRD_DIR}/esrd_shap_table_5yr.xlsx",   index_col=0)["Mean across models"]
esrd10 = pd.read_excel(f"{ESRD_DIR}/esrd_shap_table_10yr.xlsx",  index_col=0)["Mean across models"]
fl1    = pd.read_excel(f"{OUT_DIR}/shap_importance_table.xlsx",   index_col=0)["Mean across models"]
fl5    = pd.read_excel(f"{OUT_DIR}/shap_importance_table_5yr.xlsx", index_col=0)["Mean across models"]

# ── Rename flare 5yr truncated names to match ESRD short names ─
RENAME_FL5 = {
    "% chronic gloms(%of total)":              "% chronic gloms",
    "Age at biopsy":                           "Age at biopsy",
    "Class coded 1=I 2=II 3=III 4=IV 5=V 6=III+V 7=IV+V 8=II+V 9=VI 10=other": "LN class",
    "% sclerosed gloms":                       "% sclerosed gloms",
    "% active gloms (%of those not globally sclerosed)": "% active gloms",
    "%gloms with necrosis":                    "% necrosis",
    "Prev exposure to cyclo (for Rx comparison) - related to the 'use this biopsy for this patient' biopsy": "Prev cyclophosphamide",
    "Reason for biopsy 1=new pres LN 2=relapse 3=non-response/partial response, incl on-going proteinuria 4=pre-pregnancy or Ax if drug switch/stop appropriate": "Biopsy reason",
    "CKD epi formula without ethnicity":       "eGFR (CKD-EPI)",
    "Ethnicity 1=white 2=black 3=asian (south) 4=asian (east) 5=other 6=not stated/unknown/any other mixed": "Ethnicity",
}
# Handle truncated versions (the 5yr shap script used shorten() at 35 chars)
def best_rename(idx):
    renamed = {}
    for k, v in idx.items():
        match = next((r for l, r in RENAME_FL5.items() if l.startswith(k.rstrip("…")) or k == l), None)
        renamed[k] = match if match else k
    return renamed

fl5.index = [next((v for k, v in RENAME_FL5.items()
                   if k.startswith(i.rstrip("…")) or i == k), i)
             for i in fl5.index]

# ── Union of all features ──────────────────────────────────────
all_feats = list(dict.fromkeys(
    list(esrd5.index) + list(esrd10.index) +
    list(fl1.index)   + list(fl5.index)
))

# Build aligned dataframe
df = pd.DataFrame({
    "ESRD 5-year":   [esrd5.get(f, 0)  for f in all_feats],
    "ESRD 10-year":  [esrd10.get(f, 0) for f in all_feats],
    "Flare 1-year":  [fl1.get(f, 0)    for f in all_feats],
    "Flare 5-year":  [fl5.get(f, 0)    for f in all_feats],
}, index=all_feats)

# Normalise each outcome to 0-1 (relative importance)
df_norm = df.div(df.max(axis=0), axis=1)

# Sort by mean normalised importance
df_norm["_mean"] = df_norm.mean(axis=1)
df_norm = df_norm.sort_values("_mean")
df_norm = df_norm.drop(columns="_mean")
df      = df.loc[df_norm.index]

# ── Figure ─────────────────────────────────────────────────────
COLORS = {
    "ESRD 5-year":  "#c51b7d",
    "ESRD 10-year": "#de77ae",
    "Flare 1-year": "#4d9221",
    "Flare 5-year": "#a1d76a",
}

fig, axes = plt.subplots(1, 2, figsize=(18, 8), sharey=True)

# Left: raw mean |SHAP|
ax = axes[0]
n  = len(df_norm)
y_pos  = np.arange(n)
width  = 0.18
cols   = list(COLORS.keys())

for i, col in enumerate(cols):
    ax.barh(y_pos + (i - 1.5) * width, df[col], width,
            color=COLORS[col], alpha=0.88, label=col)

ax.set_yticks(y_pos)
ax.set_yticklabels(df.index, fontsize=9)
ax.set_xlabel("Mean |SHAP value| (raw)", fontsize=10)
ax.set_title("Raw Feature Importance\nMean |SHAP| across models", fontsize=11, fontweight="bold")
ax.spines[["top", "right"]].set_visible(False)
ax.legend(fontsize=9)

# Right: normalised (0-1 per outcome)
ax = axes[1]
for i, col in enumerate(cols):
    ax.barh(y_pos + (i - 1.5) * width, df_norm[col], width,
            color=COLORS[col], alpha=0.88, label=col)

ax.set_yticks(y_pos)
ax.set_yticklabels(df_norm.index, fontsize=9)
ax.set_xlabel("Relative importance (normalised 0–1 per outcome)", fontsize=10)
ax.set_title("Relative Feature Importance\n(each outcome scaled to its own max)", fontsize=11, fontweight="bold")
ax.spines[["top", "right"]].set_visible(False)
ax.legend(fontsize=9)

fig.suptitle("SHAP Feature Importance: ESRD vs Flare Prediction",
             fontsize=14, fontweight="bold", y=1.01)
plt.tight_layout()

out_path = f"{ESRD_DIR}/figures/esrd_vs_flare_shap_comparison.png"
fig.savefig(out_path, dpi=180, bbox_inches="tight")
plt.close("all")
print(f"Saved: {out_path}")
