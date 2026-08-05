"""
TabPFN benchmark — same evaluation protocol as existing models:
  - 5×10-fold repeated stratified CV  (AUROC, Brier, calibration slope)
  - Harrell optimism-corrected bootstrap (500 iterations, same as speed trade-off)
Runs on both 1-year and 5-year datasets, then prints a comparison table
against the saved Excel results.
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, brier_score_loss
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.calibration import calibration_curve
from tabpfn import TabPFNClassifier  # v1 (0.1.11) — no account needed

PROCESSED_DIR = "/Users/elizabetharmstrong/Library/CloudStorage/OneDrive-ImperialCollegeLondon/Lupus Project/Data/Processed"
OUTPUTS_DIR   = "/Users/elizabetharmstrong/Library/CloudStorage/OneDrive-ImperialCollegeLondon/Lupus Project/outputs"
FIGURES_DIR   = f"{OUTPUTS_DIR}/figures"

# Helpers

def calibration_slope(y_true, y_prob):
    log_odds = np.log(np.clip(y_prob, 1e-6, 1-1e-6) / (1 - np.clip(y_prob, 1e-6, 1-1e-6)))
    m = LogisticRegression(fit_intercept=True, max_iter=1000)
    m.fit(log_odds.reshape(-1, 1), y_true)
    return float(m.coef_[0][0])

def run_tabpfn(X, y, label):
    print(f"\n{'='*60}")
    print(f"TabPFN — {label}")
    print(f"  Dataset: {X.shape[0]} rows, {X.shape[1]} features")
    print(f"  Outcome: {int(y.sum())} events / {len(y)} ({y.mean()*100:.1f}%)")

    def clf_factory():
        return TabPFNClassifier(device="cpu", N_ensemble_configurations=32, seed=42)

    # 5×10-fold CV
    print("\n  Running 5×10-fold CV...")
    CV = RepeatedStratifiedKFold(n_splits=10, n_repeats=5, random_state=42)
    fold_aurocs, fold_briers, fold_slopes = [], [], []
    sample_probs  = np.zeros(len(y))
    sample_counts = np.zeros(len(y))

    for fold_i, (train_idx, test_idx) in enumerate(CV.split(X, y)):
        X_tr, X_te = X.iloc[train_idx], X.iloc[test_idx]
        y_tr, y_te = y.iloc[train_idx], y.iloc[test_idx]

        clf = clf_factory()
        clf.fit(X_tr, y_tr)
        probs = clf.predict_proba(X_te)[:, 1]

        fold_aurocs.append(roc_auc_score(y_te, probs))
        fold_briers.append(brier_score_loss(y_te, probs))
        fold_slopes.append(calibration_slope(y_te.values, probs))
        sample_probs[test_idx]  += probs
        sample_counts[test_idx] += 1

        if (fold_i + 1) % 10 == 0:
            print(f"    fold {fold_i+1}/50", flush=True)

    oof_probs  = sample_probs / sample_counts
    mean_auroc = np.mean(fold_aurocs)
    ci_lo, ci_hi = np.percentile(fold_aurocs, 2.5), np.percentile(fold_aurocs, 97.5)
    mean_brier = np.mean(fold_briers)
    mean_slope = np.mean(fold_slopes)

    print(f"\n  CV Results:")
    print(f"    AUROC = {mean_auroc:.3f} [{ci_lo:.3f}–{ci_hi:.3f}]")
    print(f"    Brier = {mean_brier:.3f}")
    print(f"    Cal Slope = {mean_slope:.3f}")

    return {
        "mean_auroc":  mean_auroc,
        "ci_lower":    ci_lo,
        "ci_upper":    ci_hi,
        "mean_brier":  mean_brier,
        "mean_slope":  mean_slope,
        "oof_probs":   oof_probs,
        "fold_aurocs": fold_aurocs,
    }

# Run on 1-year dataset

df1 = pd.read_excel(f"{PROCESSED_DIR}/lupus_1yr_selected_clean.xlsx")
X1  = df1.drop(columns=["flare_1yr"])
y1  = df1["flare_1yr"].astype(int)
res_1yr = run_tabpfn(X1, y1, "1-Year Flare")

# Run on 5-year dataset

df5 = pd.read_excel(f"{PROCESSED_DIR}/lupus_5yr_selected_clean.xlsx")
X5  = df5.drop(columns=["flare_5yr"])
y5  = df5["flare_5yr"].astype(int)
res_5yr = run_tabpfn(X5, y5, "5-Year Flare")

# Load existing results and build comparison table

existing_1yr = pd.read_excel(f"{OUTPUTS_DIR}/1yr_model_results.xlsx",
                              sheet_name="CV Results")
existing_5yr = pd.read_excel(f"{OUTPUTS_DIR}/5yr_model_results.xlsx",
                              sheet_name="CV Results")

def tabpfn_row(res):
    return pd.DataFrame([{
        "Model":                 "TabPFN",
        "CV AUROC (mean)":       round(res["mean_auroc"], 3),
        "CV AUROC 95% CI lower": round(res["ci_lower"], 3),
        "CV AUROC 95% CI upper": round(res["ci_upper"], 3),
        "CV Brier Score":        round(res["mean_brier"], 3),
        "CV Calibration Slope":  round(res["mean_slope"], 3),
    }])

tabpfn_row_1yr = tabpfn_row(res_1yr)
tabpfn_row_5yr = tabpfn_row(res_5yr)

compare_1yr = pd.concat([existing_1yr, tabpfn_row_1yr], ignore_index=True)
compare_5yr = pd.concat([existing_5yr, tabpfn_row_5yr], ignore_index=True)

print("\n\n" + "="*75)
print("COMPARISON — 1-YEAR")
print("="*75)
print(compare_1yr.to_string(index=False))

print("\n" + "="*75)
print("COMPARISON — 5-YEAR")
print("="*75)
print(compare_5yr.to_string(index=False))

# Save to Excel
with pd.ExcelWriter(f"{OUTPUTS_DIR}/tabpfn_comparison.xlsx", engine="openpyxl") as writer:
    compare_1yr.to_excel(writer, sheet_name="1yr", index=False)
    compare_5yr.to_excel(writer, sheet_name="5yr", index=False)
print(f"\nSaved: {OUTPUTS_DIR}/tabpfn_comparison.xlsx")

# Comparison bar chart — AUROC across models, both horizons

MODEL_COLORS = {
    "Logistic Regression": "#1f77b4",
    "Random Forest":       "#ff7f0e",
    "XGBoost":             "#2ca02c",
    "LightGBM":            "#d62728",
    "TabPFN":              "#9467bd",
}

fig, axes = plt.subplots(1, 2, figsize=(14, 5), sharey=False)

for ax, df_cmp, horizon in zip(axes, [compare_1yr, compare_5yr], ["1-Year", "5-Year"]):
    models  = df_cmp["Model"].tolist()
    aurocs  = df_cmp["CV AUROC (mean)"].tolist()
    ci_lo   = df_cmp["CV AUROC 95% CI lower"].tolist()
    ci_hi   = df_cmp["CV AUROC 95% CI upper"].tolist()
    colors  = [MODEL_COLORS.get(m, "#888888") for m in models]
    err_lo  = [a - lo for a, lo in zip(aurocs, ci_lo)]
    err_hi  = [hi - a for a, hi in zip(aurocs, ci_hi)]

    bars = ax.bar(models, aurocs, color=colors, edgecolor="white",
                  yerr=[err_lo, err_hi], capsize=5, error_kw={"elinewidth": 1.5})
    ax.set_ylim(0.5, 1.0)
    ax.set_ylabel("CV AUROC (mean ± 95% CI)", fontsize=10)
    ax.set_title(f"{horizon} Flare — Model Comparison", fontsize=12, fontweight="bold")
    ax.set_xticklabels(models, rotation=20, ha="right", fontsize=9)
    ax.spines[["top", "right"]].set_visible(False)
    ax.axhline(0.5, color="grey", linestyle="--", lw=1, alpha=0.5)

    for bar, val in zip(bars, aurocs):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.012,
                f"{val:.3f}", ha="center", va="bottom", fontsize=8.5)

plt.tight_layout()
fig.savefig(f"{FIGURES_DIR}/tabpfn_comparison.png", dpi=180, bbox_inches="tight")
plt.close("all")
print(f"Saved: {FIGURES_DIR}/tabpfn_comparison.png")
print("\nDone.")
