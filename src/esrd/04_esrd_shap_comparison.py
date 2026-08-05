"""
Side-by-side SHAP comparison: ESRD 5-year vs 10-year.
One panel per model, grouped horizontal bars.
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUT_DIR = "/Users/elizabetharmstrong/Library/CloudStorage/OneDrive-ImperialCollegeLondon/Lupus Project/outputs/esrd"

t5  = pd.read_excel(f"{OUT_DIR}/esrd_shap_table_5yr.xlsx",  index_col=0)
t10 = pd.read_excel(f"{OUT_DIR}/esrd_shap_table_10yr.xlsx", index_col=0)

MODEL_NAMES = ["Logistic Regression", "Random Forest", "XGBoost", "LightGBM"]
COLOR_5YR  = "#2166ac"
COLOR_10YR = "#d6604d"

fig, axes = plt.subplots(2, 2, figsize=(16, 14))
axes = axes.flatten()

for ax, model in zip(axes, MODEL_NAMES):
    d5  = t5[model].to_dict()
    d10 = t10[model].to_dict()

    # Union of features, sorted by average importance across both horizons
    all_feats = list(dict.fromkeys(list(d5.keys()) + list(d10.keys())))
    avg       = {f: (d5.get(f, 0) + d10.get(f, 0)) / 2 for f in all_feats}
    feats     = sorted(all_feats, key=lambda f: avg[f])

    vals5  = [d5.get(f, 0)  for f in feats]
    vals10 = [d10.get(f, 0) for f in feats]

    y_pos  = np.arange(len(feats))
    height = 0.35

    ax.barh(y_pos - height/2, vals5,  height, label="5-year",  color=COLOR_5YR,  alpha=0.85)
    ax.barh(y_pos + height/2, vals10, height, label="10-year", color=COLOR_10YR, alpha=0.85)

    ax.set_yticks(y_pos)
    ax.set_yticklabels(feats, fontsize=9)
    ax.set_xlabel("Mean |SHAP value|", fontsize=10)
    ax.set_title(model, fontsize=12, fontweight="bold")
    ax.legend(fontsize=9)
    ax.spines[["top", "right"]].set_visible(False)

fig.suptitle(
    "SHAP Feature Importance: ESRD 5-Year vs 10-Year Prediction",
    fontsize=14, fontweight="bold", y=1.01
)
plt.tight_layout()

out_path = f"{OUT_DIR}/figures/esrd_shap_5yr_vs_10yr.png"
fig.savefig(out_path, dpi=180, bbox_inches="tight")
plt.close("all")
print(f"Saved: {out_path}")
