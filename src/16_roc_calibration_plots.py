"""
ROC and calibration summary figures across all five cohorts.

For each cohort, loads the same final feature-selected dataset and tuned
best hyperparameters used by the main pipeline (no retuning — RandomizedSearchCV
is skipped, the already-selected best_params are used directly), reruns the
same outer cross-validation protocol (same n_splits/n_repeats/random_state as
the cohort's own modelling script) to reproduce identical out-of-fold (OOF)
predictions, then plots two separate figures, each a 2x3 grid (5 cohort panels
+ 1 legend panel):
  - ROC curves (model fit on full data; AUROC annotated top-left per panel)
  - Calibration curves (OOF probability decile vs. observed event rate,
    45-degree reference line)

Uses the project's standard model colour scheme (src/1_year/06_modelling_1yr.py,
src/5_year/05_modelling_5yr.py, src/5_year/11_serial_modelling_5yr.py):
  Logistic Regression #1f77b4, Random Forest #ff7f0e, XGBoost #2ca02c,
  LightGBM #d62728.

Saves: outputs/figures/roc_all_cohorts.png, outputs/figures/calibration_all_cohorts.png (300 dpi)
"""
import json
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
plt.rcParams["font.family"] = "Times New Roman"

from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.calibration import calibration_curve
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.pipeline import Pipeline
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier

BASE = "/Users/elizabetharmstrong/Library/CloudStorage/OneDrive-ImperialCollegeLondon/Lupus Project"
PROC = f"{BASE}/Data/Processed"
OUT  = f"{BASE}/outputs"
FIG_DIR = f"{OUT}/figures"

MODEL_COLORS = {
    "Logistic Regression": "#1f77b4",
    "Random Forest":       "#ff7f0e",
    "XGBoost":             "#2ca02c",
    "LightGBM":            "#d62728",
}
MODEL_ORDER = list(MODEL_COLORS)


def make_rf(p):
    return RandomForestClassifier(
        n_estimators=p.get("clf__n_estimators", 300), max_depth=p.get("clf__max_depth"),
        min_samples_leaf=p.get("clf__min_samples_leaf", 10), max_features=p.get("clf__max_features", "sqrt"),
        class_weight="balanced", random_state=42)


def make_xgb(p, spw):
    return XGBClassifier(
        n_estimators=p.get("clf__n_estimators", 100), max_depth=p.get("clf__max_depth", 2),
        learning_rate=p.get("clf__learning_rate", 0.05), subsample=p.get("clf__subsample", 0.5),
        colsample_bytree=p.get("clf__colsample_bytree", 0.5), min_child_weight=p.get("clf__min_child_weight", 10),
        reg_alpha=p.get("clf__reg_alpha", 1.0), reg_lambda=p.get("clf__reg_lambda", 5.0),
        eval_metric="logloss", verbosity=0, random_state=42, scale_pos_weight=spw)


def make_lgbm(p):
    return LGBMClassifier(
        n_estimators=p.get("clf__n_estimators", 200), max_depth=p.get("clf__max_depth", 3),
        learning_rate=p.get("clf__learning_rate", 0.05), subsample=p.get("clf__subsample", 0.8),
        num_leaves=p.get("clf__num_leaves", 31), min_child_samples=p.get("clf__min_child_samples", 20),
        verbose=-1, random_state=42, class_weight="balanced")


def build_models(best_params, spw):
    return {
        "Logistic Regression": Pipeline([("scaler", StandardScaler()),
            ("clf", LogisticRegression(max_iter=1000, random_state=42, C=1.0, class_weight="balanced"))]),
        "Random Forest": Pipeline([("scaler", StandardScaler()),
            ("clf", make_rf(best_params["Random Forest"]))]),
        "XGBoost": Pipeline([("scaler", StandardScaler()),
            ("clf", make_xgb(best_params["XGBoost"], spw))]),
        "LightGBM": Pipeline([("scaler", StandardScaler()),
            ("clf", make_lgbm(best_params["LightGBM"]))]),
    }


def get_oof_and_roc(X, y, models, n_splits, n_repeats):
    """Reruns the cohort's outer CV to get OOF probabilities (for calibration)
    and fits each model on the full data (for the ROC curve, matching the
    project's existing convention of drawing ROC on the full-data fit while
    reporting CV-mean AUROC as the annotated value)."""
    CV = RepeatedStratifiedKFold(n_splits=n_splits, n_repeats=n_repeats, random_state=42)
    results = {}
    for name, pipe in models.items():
        fold_aurocs = []
        oof_probs, oof_counts = np.zeros(len(y)), np.zeros(len(y))
        for tr, te in CV.split(X, y):
            pipe.fit(X.iloc[tr], y.iloc[tr])
            probs = pipe.predict_proba(X.iloc[te])[:, 1]
            fold_aurocs.append(roc_auc_score(y.iloc[te], probs))
            oof_probs[te] += probs
            oof_counts[te] += 1
        oof_probs = oof_probs / np.where(oof_counts > 0, oof_counts, 1)

        pipe.fit(X, y)
        fpr, tpr, _ = roc_curve(y, pipe.predict_proba(X)[:, 1])

        results[name] = {"cv_auroc": np.mean(fold_aurocs), "oof": oof_probs, "fpr": fpr, "tpr": tpr}
    return results


def load_json_params(path, key=None):
    with open(path) as f:
        d = json.load(f)
    return d[key] if key else d


INT_PARAMS = {"n_estimators", "max_depth", "min_samples_leaf", "min_child_weight",
              "num_leaves", "min_child_samples"}


def load_xlsx_params(path, sheet="Best Hyperparameters"):
    """For cohorts (serial biopsy) whose best hyperparameters are only saved
    inside the model-results workbook rather than a standalone JSON. Excel
    round-trips ints as floats, so integer-typed hyperparameters are cast
    back (sklearn/xgboost/lightgbm reject e.g. min_samples_leaf=5.0)."""
    df = pd.read_excel(path, sheet_name=sheet)
    out = {}
    for _, row in df.iterrows():
        params = {}
        for c in df.columns:
            if c == "Model" or pd.isna(row[c]):
                continue
            val = row[c]
            if c in INT_PARAMS and isinstance(val, float):
                val = int(val)
            params[f"clf__{c}"] = val
        out[row["Model"]] = params
    return out


# --- Cohort configurations ---

cohorts = []

# 1-Year flare
df = pd.read_excel(f"{PROC}/lupus_1yr_selected_clean.xlsx")
cohorts.append({
    "label": "1-Year Flare",
    "X": df.drop(columns=["flare_1yr"]), "y": df["flare_1yr"].astype(int),
    "best_params": load_json_params(f"{OUT}/1yr_best_params.json"),
    "spw": 3.34, "n_splits": 10, "n_repeats": 5, "n_bins": 10,
})

# 5-Year flare
# NB: the saved file currently still contains "dsDNA or SM or APL ever positive",
# which the Methods doc (Section 4.4) documents as manually removed post-selection
# (85.1% positive, insufficient discriminatory gradient) -> 10 final predictors, not
# 11. Dropped here so results match the published 0.673/0.679/0.673/0.678 exactly;
# Data/Processed/lupus_5yr_selected_clean.xlsx itself may need the same correction.
df = pd.read_excel(f"{PROC}/lupus_5yr_selected_clean.xlsx")
df = df.drop(columns=["dsDNA or SM or APL ever positive(1=yes 0=no)"], errors="ignore")
cohorts.append({
    "label": "5-Year Flare",
    "X": df.drop(columns=["flare_5yr"]), "y": df["flare_5yr"].astype(int),
    "best_params": load_json_params(f"{OUT}/5yr_best_params.json"),
    "spw": round(190 / 166, 2), "n_splits": 10, "n_repeats": 5, "n_bins": 10,
})

# Serial biopsy
df = pd.read_excel(f"{PROC}/lupus_5yr_serial_selected.xlsx")
cohorts.append({
    "label": "Serial Biopsy",
    "X": df.drop(columns=["flare_5yr"]), "y": df["flare_5yr"].astype(int),
    "best_params": load_xlsx_params(f"{OUT}/5yr_serial_model_results.xlsx"),
    "spw": round(36 / 34, 2), "n_splits": 5, "n_repeats": 5, "n_bins": 5,
})

# ESRD 5-Year
df = pd.read_excel(f"{PROC}/esrd_5yr_selected.xlsx")
y = df["esrd_5yr"].astype(int)
cohorts.append({
    "label": "ESRD 5-Year",
    "X": df.drop(columns=["esrd_5yr"]), "y": y,
    "best_params": load_json_params(f"{OUT}/esrd/esrd_best_params.json", key="5yr"),
    "spw": round((y == 0).sum() / (y == 1).sum(), 2), "n_splits": 10, "n_repeats": 5, "n_bins": 10,
})

# ESRD 10-Year
df = pd.read_excel(f"{PROC}/esrd_10yr_selected.xlsx")
y = df["esrd_10yr"].astype(int)
cohorts.append({
    "label": "ESRD 10-Year",
    "X": df.drop(columns=["esrd_10yr"]), "y": y,
    "best_params": load_json_params(f"{OUT}/esrd/esrd_best_params.json", key="10yr"),
    "spw": round((y == 0).sum() / (y == 1).sum(), 2), "n_splits": 10, "n_repeats": 5, "n_bins": 10,
})

# --- Compute ---

all_results = []
for c in cohorts:
    print(f"[{c['label']}] running {c['n_splits']}x{c['n_repeats']}-fold CV "
          f"({c['X'].shape[0]} rows, {c['X'].shape[1]} predictors)...")
    models = build_models(c["best_params"], c["spw"])
    res = get_oof_and_roc(c["X"], c["y"], models, c["n_splits"], c["n_repeats"])
    for name in models:
        print(f"    {name:<20} CV AUROC={res[name]['cv_auroc']:.3f}")
    all_results.append(res)

GRID_ROWS, GRID_COLS = 2, 3  # 5 cohort panels + 1 legend panel


def panel_position(i):
    return divmod(i, GRID_COLS)


# --- ROC figure ---

fig_roc, axes_roc = plt.subplots(GRID_ROWS, GRID_COLS, figsize=(12, 8), constrained_layout=True)

for i, (c, res) in enumerate(zip(cohorts, all_results)):
    row, col = panel_position(i)
    ax = axes_roc[row, col]
    ax.plot([0, 1], [0, 1], "k--", lw=1, alpha=0.5)

    annot_lines = []
    for name in MODEL_ORDER:
        r = res[name]
        ax.plot(r["fpr"], r["tpr"], color=MODEL_COLORS[name], lw=1.8)
        annot_lines.append(f"{name}: {r['cv_auroc']:.3f}")

    ax.text(0.97, 0.03, "\n".join(annot_lines), transform=ax.transAxes,
            fontsize=10.5, va="bottom", ha="right",
            bbox=dict(boxstyle="round,pad=0.3", facecolor="white", edgecolor="0.7", alpha=0.85))

    ax.set_xlim([-0.02, 1.02]); ax.set_ylim([-0.02, 1.03])
    ax.set_title(c["label"], fontsize=16, fontweight="bold")
    ax.set_xlabel("1 – Specificity", fontsize=13)
    if col == 0:
        ax.set_ylabel("Sensitivity", fontsize=13)
    else:
        ax.set_yticklabels([])
    ax.tick_params(labelsize=8)

fig_roc.get_layout_engine().set(w_pad=0.06, h_pad=0.08, wspace=0.06, hspace=0.06)

legend_row, legend_col = panel_position(len(cohorts))
legend_ax = axes_roc[legend_row, legend_col]
legend_ax.axis("off")
legend_handles = [plt.Line2D([0], [0], color=MODEL_COLORS[name], lw=2.5) for name in MODEL_ORDER]
legend_ax.legend(legend_handles, MODEL_ORDER, loc="center", fontsize=14, frameon=False)
fig_roc.savefig(f"{FIG_DIR}/roc_all_cohorts.png", dpi=300, bbox_inches="tight")
plt.close(fig_roc)
print(f"\nSaved: {FIG_DIR}/roc_all_cohorts.png")

# --- Calibration figure ---

fig_cal, axes_cal = plt.subplots(GRID_ROWS, GRID_COLS, figsize=(12, 8), constrained_layout=True)

for i, (c, res) in enumerate(zip(cohorts, all_results)):
    row, col = panel_position(i)
    ax = axes_cal[row, col]
    ax.plot([0, 1], [0, 1], "k--", lw=1, alpha=0.5)

    for name in MODEL_ORDER:
        oof = res[name]["oof"]
        try:
            frac_pos, mean_pred = calibration_curve(c["y"], oof, n_bins=c["n_bins"], strategy="quantile")
        except ValueError:
            frac_pos, mean_pred = calibration_curve(c["y"], oof, n_bins=max(3, c["n_bins"] // 2), strategy="quantile")
        ax.plot(mean_pred, frac_pos, "o-", color=MODEL_COLORS[name], lw=1.8, markersize=4)

    ax.set_xlim([0, 1]); ax.set_ylim([0, 1])
    ax.set_title(c["label"], fontsize=16, fontweight="bold")
    ax.set_xlabel("Mean Predicted Probability", fontsize=13)
    if col == 0:
        ax.set_ylabel("Observed Event Rate", fontsize=13)
    else:
        ax.set_yticklabels([])
    ax.tick_params(labelsize=8)

fig_cal.get_layout_engine().set(w_pad=0.06, h_pad=0.08, wspace=0.06, hspace=0.06)

legend_row, legend_col = panel_position(len(cohorts))
legend_ax = axes_cal[legend_row, legend_col]
legend_ax.axis("off")
legend_handles = [plt.Line2D([0], [0], color=MODEL_COLORS[name], lw=2.5, marker="o") for name in MODEL_ORDER]
legend_ax.legend(legend_handles, MODEL_ORDER, loc="center", fontsize=14, frameon=False)

fig_cal.savefig(f"{FIG_DIR}/calibration_all_cohorts.png", dpi=300, bbox_inches="tight")
plt.close(fig_cal)
print(f"Saved: {FIG_DIR}/calibration_all_cohorts.png")
