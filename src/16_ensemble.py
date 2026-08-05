"""
Ensemble benchmark — averages OOF predictions from TabPFN + individual models.
Runs 5x10-fold CV for all models simultaneously so OOF folds are identical,
then tests every pairwise combination + full ensemble.
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import json
import warnings
warnings.filterwarnings("ignore")

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.metrics import roc_auc_score, brier_score_loss
from sklearn.calibration import calibration_curve
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from tabpfn import TabPFNClassifier

PROCESSED_DIR = "/Users/elizabetharmstrong/Library/CloudStorage/OneDrive-ImperialCollegeLondon/Lupus Project/Data/Processed"
OUTPUTS_DIR   = "/Users/elizabetharmstrong/Library/CloudStorage/OneDrive-ImperialCollegeLondon/Lupus Project/outputs"
FIGURES_DIR   = f"{OUTPUTS_DIR}/figures"

def calibration_slope(y_true, y_prob):
    log_odds = np.log(np.clip(y_prob, 1e-6, 1-1e-6) / (1 - np.clip(y_prob, 1e-6, 1-1e-6)))
    m = LogisticRegression(fit_intercept=True, max_iter=1000)
    m.fit(log_odds.reshape(-1, 1), y_true)
    return float(m.coef_[0][0])

def run_single_cv(name, clf_factory, X, y):
    """Run CV for one model and return OOF predictions."""
    CV  = RepeatedStratifiedKFold(n_splits=10, n_repeats=5, random_state=42)
    n   = len(y)
    oof = np.zeros(n)
    cnt = np.zeros(n)
    for fold_i, (tr_idx, te_idx) in enumerate(CV.split(X, y)):
        clf = clf_factory()
        clf.fit(X.iloc[tr_idx], y.iloc[tr_idx])
        oof[te_idx] += clf.predict_proba(X.iloc[te_idx])[:, 1]
        cnt[te_idx] += 1
        if (fold_i + 1) % 10 == 0:
            print(f"    {name}: fold {fold_i+1}/50", flush=True)
    return oof / cnt

def run_cv(models_dict, X, y, label):
    print(f"\n{'='*60}")
    print(f"{label}  ({X.shape[0]} rows, {X.shape[1]} features, {int(y.sum())} events)")

    # Run each model separately to avoid memory conflicts
    oof = {}
    for name, clf_factory in models_dict.items():
        print(f"\n  Running {name}...")
        oof[name] = run_single_cv(name, clf_factory, X, y)

    # Score individual models
    results = {}
    print(f"\n  {'Model':<30} {'AUROC':>7}  {'Brier':>7}  {'CalSlope':>9}")
    print(f"  {'-'*58}")
    for name, probs in oof.items():
        auroc = roc_auc_score(y, probs)
        brier = brier_score_loss(y, probs)
        slope = calibration_slope(y.values, probs)
        results[name] = {"auroc": auroc, "brier": brier, "slope": slope, "probs": probs}
        print(f"  {name:<30} {auroc:>7.3f}  {brier:>7.3f}  {slope:>9.3f}")

    # Pairwise and full ensembles
    names = list(oof.keys())
    ensembles = {}

    # All pairings involving TabPFN
    for other in names:
        if other == "TabPFN":
            continue
        key = f"TabPFN + {other}"
        ensembles[key] = (oof["TabPFN"] + oof[other]) / 2

    # Best non-TabPFN pair (LR + best tree)
    non_tabpfn = [n for n in names if n != "TabPFN"]
    for i, a in enumerate(non_tabpfn):
        for b in non_tabpfn[i+1:]:
            key = f"{a} + {b}"
            ensembles[key] = (oof[a] + oof[b]) / 2

    # Full ensemble (all 5)
    ensembles["All models"] = np.mean([oof[n] for n in names], axis=0)

    print(f"\n  --- Ensembles ---")
    for key, probs in ensembles.items():
        auroc = roc_auc_score(y, probs)
        brier = brier_score_loss(y, probs)
        slope = calibration_slope(y.values, probs)
        results[key] = {"auroc": auroc, "brier": brier, "slope": slope, "probs": probs}
        print(f"  {key:<30} {auroc:>7.3f}  {brier:>7.3f}  {slope:>9.3f}")

    return results

# ============================================================
# Build models
# ============================================================
def get_model_factories(bp, scale_pos_weight=1.0):
    """Return dict of zero-arg factory functions, one per model."""
    return {
        "Logistic Regression": lambda: Pipeline([("s", StandardScaler()),
            ("clf", LogisticRegression(max_iter=1000, C=1.0, class_weight="balanced", random_state=42))]),
        "Random Forest": lambda: Pipeline([("s", StandardScaler()),
            ("clf", RandomForestClassifier(
                n_estimators=bp["Random Forest"].get("clf__n_estimators", 300),
                max_depth=bp["Random Forest"].get("clf__max_depth", 3),
                min_samples_leaf=bp["Random Forest"].get("clf__min_samples_leaf", 10),
                max_features=bp["Random Forest"].get("clf__max_features", "sqrt"),
                class_weight="balanced", random_state=42))]),
        "XGBoost": lambda: Pipeline([("s", StandardScaler()),
            ("clf", XGBClassifier(
                n_estimators=bp["XGBoost"].get("clf__n_estimators", 100),
                max_depth=bp["XGBoost"].get("clf__max_depth", 2),
                learning_rate=bp["XGBoost"].get("clf__learning_rate", 0.01),
                subsample=bp["XGBoost"].get("clf__subsample", 0.5),
                colsample_bytree=bp["XGBoost"].get("clf__colsample_bytree", 0.5),
                min_child_weight=bp["XGBoost"].get("clf__min_child_weight", 5),
                reg_alpha=bp["XGBoost"].get("clf__reg_alpha", 1.0),
                reg_lambda=bp["XGBoost"].get("clf__reg_lambda", 5.0),
                scale_pos_weight=scale_pos_weight,
                eval_metric="logloss", verbosity=0, random_state=42))]),
        "LightGBM": lambda: Pipeline([("s", StandardScaler()),
            ("clf", LGBMClassifier(
                n_estimators=bp["LightGBM"].get("clf__n_estimators", 100),
                max_depth=bp["LightGBM"].get("clf__max_depth", 3),
                learning_rate=bp["LightGBM"].get("clf__learning_rate", 0.01),
                subsample=bp["LightGBM"].get("clf__subsample", 0.7),
                num_leaves=bp["LightGBM"].get("clf__num_leaves", 15),
                min_child_samples=bp["LightGBM"].get("clf__min_child_samples", 10),
                class_weight="balanced", verbose=-1, random_state=42))]),
        "TabPFN": lambda: TabPFNClassifier(device="cpu", N_ensemble_configurations=32, seed=42),
    }

# ── 1-year ────────────────────────────────────────────────────
df1 = pd.read_excel(f"{PROCESSED_DIR}/lupus_1yr_selected_clean.xlsx")
X1, y1 = df1.drop(columns=["flare_1yr"]), df1["flare_1yr"].astype(int)
with open(f"{OUTPUTS_DIR}/1yr_best_params.json") as f:
    bp1 = json.load(f)
res1 = run_cv(get_model_factories(bp1, scale_pos_weight=3.34), X1, y1, "1-YEAR FLARE")

# ── 5-year ────────────────────────────────────────────────────
df5 = pd.read_excel(f"{PROCESSED_DIR}/lupus_5yr_selected_clean.xlsx")
X5, y5 = df5.drop(columns=["flare_5yr"]), df5["flare_5yr"].astype(int)
with open(f"{OUTPUTS_DIR}/5yr_best_params.json") as f:
    bp5 = json.load(f)
res5 = run_cv(get_model_factories(bp5, scale_pos_weight=round(190/166, 2)), X5, y5, "5-YEAR FLARE")

# ============================================================
# Figure — AUROC for all models + ensembles, both horizons
# ============================================================
MODEL_COLORS = {
    "Logistic Regression":         "#1f77b4",
    "Random Forest":               "#ff7f0e",
    "XGBoost":                     "#2ca02c",
    "LightGBM":                    "#d62728",
    "TabPFN":                      "#9467bd",
}
ENSEMBLE_COLOR = "#8c564b"

fig, axes = plt.subplots(1, 2, figsize=(16, 6))

for ax, res, horizon in zip(axes, [res1, res5], ["1-Year", "5-Year"]):
    names  = list(res.keys())
    aurocs = [res[n]["auroc"] for n in names]
    colors = [MODEL_COLORS.get(n, ENSEMBLE_COLOR) for n in names]

    # Sort by AUROC
    order  = np.argsort(aurocs)
    names  = [names[i] for i in order]
    aurocs = [aurocs[i] for i in order]
    colors = [colors[i] for i in order]

    bars = ax.barh(names, aurocs, color=colors, alpha=0.85, edgecolor="white")
    ax.axvline(0.5, color="grey", linestyle="--", lw=1, alpha=0.5)

    # Highlight best
    best_idx = aurocs.index(max(aurocs))
    bars[best_idx].set_edgecolor("black")
    bars[best_idx].set_linewidth(2)

    for bar, val in zip(bars, aurocs):
        ax.text(val + 0.003, bar.get_y() + bar.get_height()/2,
                f"{val:.3f}", va="center", fontsize=8)

    ax.set_xlim(0.45, max(aurocs) + 0.06)
    ax.set_xlabel("CV AUROC (OOF, 5×10-fold)", fontsize=10)
    ax.set_title(f"{horizon} Flare — Individual & Ensemble Models", fontsize=11, fontweight="bold")
    ax.spines[["top", "right"]].set_visible(False)
    ax.tick_params(axis="y", labelsize=8)

# Legend patches
from matplotlib.patches import Patch
legend_els = [Patch(color=c, label=n) for n, c in MODEL_COLORS.items()]
legend_els.append(Patch(color=ENSEMBLE_COLOR, label="Ensemble"))
fig.legend(handles=legend_els, loc="lower center", ncol=6,
           fontsize=8.5, bbox_to_anchor=(0.5, -0.04))

plt.tight_layout()
fig.savefig(f"{FIGURES_DIR}/ensemble_comparison.png", dpi=180, bbox_inches="tight")
plt.close("all")
print(f"\nSaved: {FIGURES_DIR}/ensemble_comparison.png")

# Save to Excel
rows1 = [{"Model": n, "AUROC": round(v["auroc"],3), "Brier": round(v["brier"],3),
           "Cal Slope": round(v["slope"],3)} for n, v in res1.items()]
rows5 = [{"Model": n, "AUROC": round(v["auroc"],3), "Brier": round(v["brier"],3),
           "Cal Slope": round(v["slope"],3)} for n, v in res5.items()]

with pd.ExcelWriter(f"{OUTPUTS_DIR}/ensemble_results.xlsx", engine="openpyxl") as w:
    pd.DataFrame(rows1).to_excel(w, sheet_name="1yr", index=False)
    pd.DataFrame(rows5).to_excel(w, sheet_name="5yr", index=False)
print(f"Saved: {OUTPUTS_DIR}/ensemble_results.xlsx")
print("\nDone.")
