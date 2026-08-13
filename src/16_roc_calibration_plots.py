"""
ROC and calibration summary figures across all five cohorts.

For each cohort, loads the same final feature-selected dataset and tuned
best hyperparameters used by the main pipeline (no retuning — RandomizedSearchCV
is skipped, the already-selected best_params are used directly), reruns the
same outer cross-validation protocol (same n_splits/n_repeats/random_state as
the cohort's own modelling script) to reproduce identical out-of-fold (OOF)
predictions, then plots two separate figures, each a 2x3 grid (5 cohort panels
+ 1 legend panel):
  - ROC curves, built from pooled OOF predictions (not a full-data-fit) so
    the annotated AUROC always matches what's actually drawn
  - Calibration curves: equal-frequency (quantile) binned OOF probability vs.
    observed event rate, with Wilson 95% CI error bars per bin (5 bins for
    1yr/5yr/ESRD, 3 for serial biopsy - n=70 is too small for 5)

All five models (the four main classifiers + TabPFN v3, where available)
use the same OOF-pooled AUROC convention on this figure, so every printed
number matches its own curve. This differs slightly from the fold-mean "CV
AUROC" reported in the Methods/Results docs and Tables (a different,
also-valid aggregation of the same cross-validation) - the two are not
meant to be read as identical.

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
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.pipeline import Pipeline
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier


def wilson_ci(k, n, z=1.96):
    """Wilson score interval for a binomial proportion k/n."""
    if n == 0:
        return (np.nan, np.nan)
    phat = k / n
    denom = 1 + z**2 / n
    center = (phat + z**2 / (2 * n)) / denom
    margin = (z * np.sqrt(phat * (1 - phat) / n + z**2 / (4 * n**2))) / denom
    return (max(0.0, center - margin), min(1.0, center + margin))


def calibration_bins_with_ci(y_true, y_prob, n_bins):
    """Equal-frequency (quantile) binning - each bin holds a comparable
    number of patients, unlike equal-width binning. Returns per-bin mean
    predicted probability, observed event rate, and Wilson 95% CI bounds."""
    y_true = np.asarray(y_true)
    y_prob = np.asarray(y_prob)
    order = np.argsort(y_prob)
    yt_sorted, yp_sorted = y_true[order], y_prob[order]
    mean_pred, obs_rate, ci_lo, ci_hi = [], [], [], []
    for idx in np.array_split(np.arange(len(y_prob)), n_bins):
        if len(idx) == 0:
            continue
        yt, yp = yt_sorted[idx], yp_sorted[idx]
        n, k = len(idx), yt.sum()
        mean_pred.append(yp.mean())
        obs_rate.append(k / n)
        lo, hi = wilson_ci(k, n)
        ci_lo.append(lo)
        ci_hi.append(hi)
    return np.array(mean_pred), np.array(obs_rate), np.array(ci_lo), np.array(ci_hi)

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
PANEL_LETTERS = ["A", "B", "C", "D", "E"]


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
    """Reruns the cohort's outer CV to get pooled out-of-fold probabilities,
    then builds the ROC curve and AUROC directly from those OOF predictions
    (not a full-data fit) - so the plotted curve and the annotated AUROC
    are always the same underlying computation."""
    CV = RepeatedStratifiedKFold(n_splits=n_splits, n_repeats=n_repeats, random_state=42)
    results = {}
    for name, pipe in models.items():
        oof_probs, oof_counts = np.zeros(len(y)), np.zeros(len(y))
        for tr, te in CV.split(X, y):
            pipe.fit(X.iloc[tr], y.iloc[tr])
            probs = pipe.predict_proba(X.iloc[te])[:, 1]
            oof_probs[te] += probs
            oof_counts[te] += 1
        oof_probs = oof_probs / np.where(oof_counts > 0, oof_counts, 1)

        fpr, tpr, _ = roc_curve(y, oof_probs)
        auroc = roc_auc_score(y, oof_probs)

        results[name] = {"auroc": auroc, "oof": oof_probs, "fpr": fpr, "tpr": tpr}
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
    "spw": 3.34, "n_splits": 10, "n_repeats": 5, "n_bins": 5,
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
    "spw": round(190 / 166, 2), "n_splits": 10, "n_repeats": 5, "n_bins": 5,
})

# Serial biopsy
df = pd.read_excel(f"{PROC}/lupus_5yr_serial_selected.xlsx")
cohorts.append({
    "label": "Serial Biopsy",
    "X": df.drop(columns=["flare_5yr"]), "y": df["flare_5yr"].astype(int),
    "best_params": load_xlsx_params(f"{OUT}/5yr_serial_model_results.xlsx"),
    "spw": round(36 / 34, 2), "n_splits": 5, "n_repeats": 5, "n_bins": 3,
})

# ESRD 5-Year
df = pd.read_excel(f"{PROC}/esrd_5yr_selected.xlsx")
y = df["esrd_5yr"].astype(int)
cohorts.append({
    "label": "ESRD 5-Year",
    "X": df.drop(columns=["esrd_5yr"]), "y": y,
    "best_params": load_json_params(f"{OUT}/esrd/esrd_best_params.json", key="5yr"),
    "spw": round((y == 0).sum() / (y == 1).sum(), 2), "n_splits": 10, "n_repeats": 5, "n_bins": 5,
})

# ESRD 10-Year
df = pd.read_excel(f"{PROC}/esrd_10yr_selected.xlsx")
y = df["esrd_10yr"].astype(int)
cohorts.append({
    "label": "ESRD 10-Year",
    "X": df.drop(columns=["esrd_10yr"]), "y": y,
    "best_params": load_json_params(f"{OUT}/esrd/esrd_best_params.json", key="10yr"),
    "spw": round((y == 0).sum() / (y == 1).sum(), 2), "n_splits": 10, "n_repeats": 5, "n_bins": 5,
})

# --- Compute ---

all_results = []
roc_rows, cal_rows, oof_rows = [], [], []

for c in cohorts:
    print(f"[{c['label']}] running {c['n_splits']}x{c['n_repeats']}-fold CV "
          f"({c['X'].shape[0]} rows, {c['X'].shape[1]} predictors)...")
    models = build_models(c["best_params"], c["spw"])
    res = get_oof_and_roc(c["X"], c["y"], models, c["n_splits"], c["n_repeats"])
    for name in models:
        print(f"    {name:<20} OOF AUROC={res[name]['auroc']:.3f}")
    all_results.append(res)

    y_arr = c["y"].values
    for name in MODEL_ORDER:
        r = res[name]
        for fpr_v, tpr_v in zip(r["fpr"], r["tpr"]):
            roc_rows.append({"Cohort": c["label"], "Model": name, "FPR": fpr_v, "TPR": tpr_v,
                              "OOF_AUROC": round(r["auroc"], 4)})

        mean_pred, obs_rate, ci_lo, ci_hi = calibration_bins_with_ci(c["y"], r["oof"], c["n_bins"])
        for mp, orate, lo, hi in zip(mean_pred, obs_rate, ci_lo, ci_hi):
            cal_rows.append({"Cohort": c["label"], "Model": name,
                              "Mean_Predicted_Probability": mp, "Observed_Event_Rate": orate,
                              "Wilson_CI_Lower": lo, "Wilson_CI_Upper": hi})

        for idx, (true_label, prob) in enumerate(zip(y_arr, r["oof"])):
            oof_rows.append({"Cohort": c["label"], "Model": name, "Sample_Index": idx,
                              "True_Label": int(true_label), "OOF_Predicted_Probability": prob})

curve_data_path = f"{OUT}/roc_calibration_curve_data.xlsx"
with pd.ExcelWriter(curve_data_path, engine="openpyxl") as writer:
    pd.DataFrame(roc_rows).to_excel(writer, sheet_name="ROC_curves", index=False)
    pd.DataFrame(cal_rows).to_excel(writer, sheet_name="Calibration_curves", index=False)
    pd.DataFrame(oof_rows).to_excel(writer, sheet_name="OOF_predictions", index=False)
print(f"\nSaved: {curve_data_path}")
print("  ROC_curves: pooled-OOF FPR/TPR points per model per cohort (matches plotted curves)")
print("  Calibration_curves: equal-frequency binned mean predicted prob vs observed event rate, with Wilson 95% CI")
print("  OOF_predictions: raw out-of-fold probability + true label per sample, per model per cohort")
print("  (for full flexibility if you want to build your own curves/thresholds)")

# TabPFN v3: reference curves, wherever raw OOF predictions were retained
# (src/16_tabpfn_v3_benchmark.py). Excluded from DeLong's test and Harrell
# bootstrap (see Methods Section 10.1), so it is drawn distinctly (dashed,
# grey) rather than as a normal competitor. Uses the same pooled-OOF AUROC
# convention as the other four models on this figure.
TABPFN_COLOR = "0.45"
TABPFN_COHORT_KEY = {
    "1-Year Flare": "1yr_flare", "5-Year Flare": "5yr_flare", "Serial Biopsy": "serial_biopsy",
    "ESRD 5-Year": "esrd_5yr", "ESRD 10-Year": "esrd_10yr",
}
tabpfn_curves = {}
try:
    oof_tabpfn_all = pd.read_excel(f"{OUT}/tabpfn_v3_oof_predictions.xlsx")
    for label, key in TABPFN_COHORT_KEY.items():
        oof_tabpfn = oof_tabpfn_all[oof_tabpfn_all["Cohort"] == key].sort_values("Sample_Index")
        if not len(oof_tabpfn):
            continue
        y_true = oof_tabpfn["True_Label"].values
        probs = oof_tabpfn["OOF_Predicted_Probability"].values
        fpr_t, tpr_t, _ = roc_curve(y_true, probs)
        auroc_t = roc_auc_score(y_true, probs)
        n_bins = next(c["n_bins"] for c in cohorts if c["label"] == label)
        mean_pred_t, frac_pos_t, ci_lo_t, ci_hi_t = calibration_bins_with_ci(y_true, probs, n_bins)
        tabpfn_curves[label] = {"fpr": fpr_t, "tpr": tpr_t, "auroc": auroc_t,
                                 "mean_pred": mean_pred_t, "frac_pos": frac_pos_t,
                                 "ci_lo": ci_lo_t, "ci_hi": ci_hi_t}
        print(f"\n[TabPFN v3 reference] {label}: OOF AUROC={auroc_t:.3f}")
except FileNotFoundError:
    print("\n[TabPFN v3 reference] outputs/tabpfn_v3_oof_predictions.xlsx not found - skipping.")

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
        annot_lines.append(f"{name}: {r['auroc']:.3f}")

    # The only significant DeLong pairwise result across all 5 cohorts (src/13_delong_test.py):
    # Random Forest significantly outperformed LightGBM here (p=0.001 raw, p=0.006 Holm-corrected).
    if c["label"] == "1-Year Flare":
        annot_lines.append("RF > LGBM (DeLong p=0.006*)")

    tp = tabpfn_curves.get(c["label"])
    if tp is not None:
        ax.plot(tp["fpr"], tp["tpr"], color=TABPFN_COLOR, lw=1.6, linestyle="--")
        annot_lines.append(f"TabPFN v3: {tp['auroc']:.3f}")

    ax.text(0.97, 0.03, "\n".join(annot_lines), transform=ax.transAxes,
            fontsize=10.5, va="bottom", ha="right",
            bbox=dict(boxstyle="round,pad=0.3", facecolor="white", edgecolor="0.7", alpha=0.85))

    ax.set_xlim([-0.02, 1.02]); ax.set_ylim([-0.02, 1.03])
    ax.set_title(f"{PANEL_LETTERS[i]}. {c['label']}", fontsize=16, fontweight="bold", loc="left")
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
legend_labels = list(MODEL_ORDER)
if tabpfn_curves:
    legend_handles.append(plt.Line2D([0], [0], color=TABPFN_COLOR, lw=1.6, linestyle="--"))
    legend_labels.append("TabPFN v3")
legend_ax.legend(legend_handles, legend_labels, loc="center", fontsize=14, frameon=False)

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
        mean_pred, obs_rate, ci_lo, ci_hi = calibration_bins_with_ci(c["y"], oof, c["n_bins"])
        yerr = [obs_rate - ci_lo, ci_hi - obs_rate]
        ax.errorbar(mean_pred, obs_rate, yerr=yerr, fmt="o-", color=MODEL_COLORS[name],
                    lw=1.8, markersize=4, capsize=3, elinewidth=1, ecolor=MODEL_COLORS[name])

    tp = tabpfn_curves.get(c["label"])
    if tp is not None:
        yerr_t = [tp["frac_pos"] - tp["ci_lo"], tp["ci_hi"] - tp["frac_pos"]]
        ax.errorbar(tp["mean_pred"], tp["frac_pos"], yerr=yerr_t, fmt="^--", color=TABPFN_COLOR,
                    lw=1.6, markersize=5, capsize=3, elinewidth=1, ecolor=TABPFN_COLOR)
        # No text annotation here (unlike the ROC panel): AUROC is a
        # discrimination metric, not a calibration one, so it doesn't
        # belong on this plot. The curve + shared legend entry are enough,
        # consistent with how the other four models are shown here.

    ax.set_xlim([0, 1]); ax.set_ylim([0, 1])
    ax.set_title(f"{PANEL_LETTERS[i]}. {c['label']}", fontsize=16, fontweight="bold", loc="left")
    ax.set_xlabel("Mean Predicted Probability", fontsize=13)
    ax.set_ylabel("Observed Event Rate", fontsize=13)
    if col != 0:
        ax.set_yticklabels([])
    ax.tick_params(labelsize=8)

fig_cal.get_layout_engine().set(w_pad=0.06, h_pad=0.08, wspace=0.06, hspace=0.06)

legend_row, legend_col = panel_position(len(cohorts))
legend_ax = axes_cal[legend_row, legend_col]
legend_ax.axis("off")
legend_handles = [plt.Line2D([0], [0], color=MODEL_COLORS[name], lw=2.5, marker="o") for name in MODEL_ORDER]
legend_labels = list(MODEL_ORDER)
if tabpfn_curves:
    legend_handles.append(plt.Line2D([0], [0], color=TABPFN_COLOR, lw=1.6, linestyle="--", marker="^"))
    legend_labels.append("TabPFN v3")
legend_ax.legend(legend_handles, legend_labels, loc="center", fontsize=14, frameon=False)

fig_cal.savefig(f"{FIG_DIR}/calibration_all_cohorts.png", dpi=300, bbox_inches="tight")
plt.close(fig_cal)
print(f"Saved: {FIG_DIR}/calibration_all_cohorts.png")
