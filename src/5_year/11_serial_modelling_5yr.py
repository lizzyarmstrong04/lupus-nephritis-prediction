import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")

from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import (RepeatedStratifiedKFold, StratifiedKFold,
                                     RandomizedSearchCV)
from sklearn.calibration import calibration_curve
from sklearn.metrics import roc_auc_score, brier_score_loss, roc_curve
from sklearn.pipeline import Pipeline
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier

PROCESSED_DIR = "/Users/elizabetharmstrong/Library/CloudStorage/OneDrive-ImperialCollegeLondon/Lupus Project/Data/Processed"
OUTPUTS_DIR   = "/Users/elizabetharmstrong/Library/CloudStorage/OneDrive-ImperialCollegeLondon/Lupus Project/outputs"
FIGURES_DIR   = f"{OUTPUTS_DIR}/figures"
OUTCOME_COL   = "flare_5yr"

# 34 events, 36 non-events — nearly balanced; minimal weighting
SCALE_POS_WEIGHT = round(36 / 34, 2)

df = pd.read_excel(f"{PROCESSED_DIR}/lupus_5yr_serial_selected.xlsx")
X  = df.drop(columns=[OUTCOME_COL])
y  = df[OUTCOME_COL].astype(int)

print(f"Dataset: {X.shape[0]} rows, {X.shape[1]} predictors")
print(f"Outcome events: {y.sum()} / {len(y)} ({y.mean()*100:.1f}%)")
print(f"Predictors: {list(X.columns)}")
print(f"\n⚠  EPV = {int(y.sum())}/{X.shape[1]} = {y.sum()/X.shape[1]:.1f} — interpret all metrics cautiously")
print(f"   5×5-fold CV used (5×10-fold would give only 7 test patients per fold)\n")

# ============================================================
# Helpers
# ============================================================
def calibration_slope(y_true, y_prob):
    log_odds = np.log(np.clip(y_prob, 1e-6, 1-1e-6) / (1 - np.clip(y_prob, 1e-6, 1-1e-6)))
    m = LogisticRegression(fit_intercept=True, max_iter=1000)
    m.fit(log_odds.reshape(-1, 1), y_true)
    return float(m.coef_[0][0])

def harrell_bootstrap(pipeline, X, y, n_boot=1000, seed=42):
    pipeline.fit(X, y)
    p_app          = pipeline.predict_proba(X)[:, 1]
    apparent_auroc = roc_auc_score(y, p_app)
    apparent_brier = brier_score_loss(y, p_app)

    rng  = np.random.default_rng(seed)
    n    = len(y)
    X_np = X.values
    y_np = y.values
    optimisms_auroc, optimisms_brier = [], []

    for _ in range(n_boot):
        idx = rng.integers(0, n, size=n)
        X_b = pd.DataFrame(X_np[idx], columns=X.columns)
        y_b = pd.Series(y_np[idx])
        if y_b.nunique() < 2:
            continue
        pipeline.fit(X_b, y_b)
        p_boot = pipeline.predict_proba(X_b)[:, 1]
        p_orig = pipeline.predict_proba(X)[:, 1]
        optimisms_auroc.append(roc_auc_score(y_b, p_boot) - roc_auc_score(y, p_orig))
        optimisms_brier.append(brier_score_loss(y_b, p_boot) - brier_score_loss(y, p_orig))

    return {
        "apparent_auroc": round(apparent_auroc, 3),
        "optimism_auroc": round(np.mean(optimisms_auroc), 3),
        "bc_auroc":       round(apparent_auroc - np.mean(optimisms_auroc), 3),
        "apparent_brier": round(apparent_brier, 3),
        "optimism_brier": round(np.mean(optimisms_brier), 3),
        "bc_brier":       round(apparent_brier - np.mean(optimisms_brier), 3),
    }

# ============================================================
# Hyperparameter tuning (3-fold inner CV — small dataset)
# ============================================================
print("="*60)
print("STEP 1 — Hyperparameter tuning (RandomizedSearchCV, 3-fold)")
print("="*60)

inner_cv    = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
param_grids = {
    "Random Forest": {
        "clf__n_estimators":     [100, 200],
        "clf__max_depth":        [2, 3, None],
        "clf__min_samples_leaf": [5, 10, 15],
        "clf__max_features":     ["sqrt", 0.5],
    },
    "XGBoost": {
        "clf__n_estimators":     [50, 100],
        "clf__max_depth":        [2, 3],
        "clf__learning_rate":    [0.01, 0.05],
        "clf__subsample":        [0.5, 0.7],
        "clf__colsample_bytree": [0.5, 0.7],
        "clf__min_child_weight": [3, 5, 10],
        "clf__reg_alpha":        [0.5, 1.0, 2.0],
        "clf__reg_lambda":       [2.0, 5.0, 10.0],
    },
    "LightGBM": {
        "clf__n_estimators":      [50, 100],
        "clf__max_depth":         [2, 3],
        "clf__learning_rate":     [0.01, 0.05],
        "clf__subsample":         [0.7, 1.0],
        "clf__num_leaves":        [7, 15, 31],
        "clf__min_child_samples": [5, 10, 20],
    },
}

base_pipelines = {
    "Random Forest": Pipeline([
        ("scaler", StandardScaler()),
        ("clf", RandomForestClassifier(random_state=42, class_weight="balanced")),
    ]),
    "XGBoost": Pipeline([
        ("scaler", StandardScaler()),
        ("clf", XGBClassifier(random_state=42, eval_metric="logloss", verbosity=0,
                              scale_pos_weight=SCALE_POS_WEIGHT)),
    ]),
    "LightGBM": Pipeline([
        ("scaler", StandardScaler()),
        ("clf", LGBMClassifier(random_state=42, verbose=-1, class_weight="balanced")),
    ]),
}

best_params = {}
for name, pipeline in base_pipelines.items():
    print(f"\n  Tuning {name}...")
    search = RandomizedSearchCV(
        pipeline, param_distributions=param_grids[name],
        n_iter=20, scoring="roc_auc", cv=inner_cv,
        random_state=42, n_jobs=-1, refit=True,
    )
    search.fit(X, y)
    best_params[name] = search.best_params_
    print(f"  Best CV AUROC: {search.best_score_:.3f}")
    print(f"  Best params:   {search.best_params_}")

# ============================================================
# Final models
# ============================================================
def make_rf(p):
    return RandomForestClassifier(
        n_estimators=p.get("clf__n_estimators", 100),
        max_depth=p.get("clf__max_depth", None),
        min_samples_leaf=p.get("clf__min_samples_leaf", 10),
        max_features=p.get("clf__max_features", "sqrt"),
        class_weight="balanced", random_state=42)

def make_xgb(p):
    return XGBClassifier(
        n_estimators=p.get("clf__n_estimators", 100),
        max_depth=p.get("clf__max_depth", 2),
        learning_rate=p.get("clf__learning_rate", 0.05),
        subsample=p.get("clf__subsample", 0.7),
        colsample_bytree=p.get("clf__colsample_bytree", 0.7),
        min_child_weight=p.get("clf__min_child_weight", 5),
        reg_alpha=p.get("clf__reg_alpha", 1.0),
        reg_lambda=p.get("clf__reg_lambda", 5.0),
        eval_metric="logloss", verbosity=0, random_state=42,
        scale_pos_weight=SCALE_POS_WEIGHT)

def make_lgbm(p):
    return LGBMClassifier(
        n_estimators=p.get("clf__n_estimators", 100),
        max_depth=p.get("clf__max_depth", 3),
        learning_rate=p.get("clf__learning_rate", 0.05),
        subsample=p.get("clf__subsample", 0.8),
        num_leaves=p.get("clf__num_leaves", 15),
        min_child_samples=p.get("clf__min_child_samples", 10),
        verbose=-1, random_state=42, class_weight="balanced")

MODELS = {
    "Logistic Regression": Pipeline([
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(max_iter=1000, random_state=42, C=1.0,
                                   class_weight="balanced")),
    ]),
    "Random Forest": Pipeline([
        ("scaler", StandardScaler()),
        ("clf", make_rf(best_params["Random Forest"])),
    ]),
    "XGBoost": Pipeline([
        ("scaler", StandardScaler()),
        ("clf", make_xgb(best_params["XGBoost"])),
    ]),
    "LightGBM": Pipeline([
        ("scaler", StandardScaler()),
        ("clf", make_lgbm(best_params["LightGBM"])),
    ]),
}

MODEL_COLORS = {
    "Logistic Regression": "#1f77b4",
    "Random Forest":       "#ff7f0e",
    "XGBoost":             "#2ca02c",
    "LightGBM":            "#d62728",
}

# ============================================================
# 5×5-fold cross-validation (appropriate for n=70)
# ============================================================
print("\n" + "="*60)
print("STEP 2 — 5×5-fold cross-validation (n=70; 14 test pts/fold)")
print("="*60)

CV         = RepeatedStratifiedKFold(n_splits=5, n_repeats=5, random_state=42)
cv_results = {}

for name, pipeline in MODELS.items():
    print(f"  {name}...", end=" ", flush=True)
    fold_aurocs, fold_briers, fold_slopes = [], [], []
    sample_probs  = np.zeros(len(y))
    sample_counts = np.zeros(len(y))

    for train_idx, test_idx in CV.split(X, y):
        X_tr, X_te = X.iloc[train_idx], X.iloc[test_idx]
        y_tr, y_te = y.iloc[train_idx], y.iloc[test_idx]
        if y_te.nunique() < 2:
            continue
        pipeline.fit(X_tr, y_tr)
        probs = pipeline.predict_proba(X_te)[:, 1]
        fold_aurocs.append(roc_auc_score(y_te, probs))
        fold_briers.append(brier_score_loss(y_te, probs))
        fold_slopes.append(calibration_slope(y_te.values, probs))
        sample_probs[test_idx]  += probs
        sample_counts[test_idx] += 1

    oof_probs  = sample_probs / np.where(sample_counts > 0, sample_counts, 1)
    mean_auroc = np.mean(fold_aurocs)
    ci_auroc   = (np.percentile(fold_aurocs, 2.5), np.percentile(fold_aurocs, 97.5))
    mean_brier = np.mean(fold_briers)
    mean_slope = np.mean(fold_slopes)

    cv_results[name] = {
        "mean_auroc": mean_auroc, "ci_lower": ci_auroc[0], "ci_upper": ci_auroc[1],
        "mean_brier": mean_brier, "mean_cal_slope": mean_slope,
        "oof_probs": oof_probs, "fold_aurocs": fold_aurocs,
    }
    print(f"AUROC={mean_auroc:.3f} [{ci_auroc[0]:.3f}–{ci_auroc[1]:.3f}]  "
          f"Brier={mean_brier:.3f}  CalSlope={mean_slope:.3f}")

# ============================================================
# Harrell bootstrap
# ============================================================
print("\n" + "="*60)
print("STEP 3 — Harrell optimism-corrected bootstrap (1000 iterations)")
print("="*60)

boot_results = {}
for name, pipeline in MODELS.items():
    print(f"  {name}...", end=" ", flush=True)
    result = harrell_bootstrap(pipeline, X, y, n_boot=1000)
    boot_results[name] = result
    print(f"Apparent={result['apparent_auroc']:.3f}  "
          f"Optimism={result['optimism_auroc']:.3f}  "
          f"BC={result['bc_auroc']:.3f}")

# ============================================================
# Plots
# ============================================================
print("\nGenerating plots...")
fig_roc, ax_roc = plt.subplots(figsize=(7, 6))
ax_roc.plot([0, 1], [0, 1], "k--", lw=1, alpha=0.5, label="Chance")
for name, pipeline in MODELS.items():
    pipeline.fit(X, y)
    fpr, tpr, _ = roc_curve(y, pipeline.predict_proba(X)[:, 1])
    cv    = cv_results[name]
    label = (f"{name}  (CV AUROC={cv['mean_auroc']:.3f} "
             f"[{cv['ci_lower']:.3f}–{cv['ci_upper']:.3f}])")
    ax_roc.plot(fpr, tpr, color=MODEL_COLORS[name], lw=2, label=label)
ax_roc.set_xlabel("1 – Specificity"); ax_roc.set_ylabel("Sensitivity")
ax_roc.set_title("ROC — 5yr Serial Biopsy Flare\n(tuned models, n=70)")
ax_roc.legend(loc="lower right", fontsize=8)
ax_roc.set_xlim([0,1]); ax_roc.set_ylim([0,1])
plt.tight_layout()
fig_roc.savefig(f"{FIGURES_DIR}/roc_curves_5yr_serial.png", dpi=200, bbox_inches="tight")
plt.close(fig_roc)
print("  Saved: roc_curves_5yr_serial.png")

fig_cal, ax_cal = plt.subplots(figsize=(7, 6))
ax_cal.plot([0,1],[0,1],"k--",lw=1,alpha=0.5,label="Perfect calibration")
for name in MODELS:
    oof   = cv_results[name]["oof_probs"]
    slope = cv_results[name]["mean_cal_slope"]
    frac_pos, mean_pred = calibration_curve(y, oof, n_bins=5, strategy="quantile")
    ax_cal.plot(mean_pred, frac_pos, "o-", color=MODEL_COLORS[name], lw=2,
                markersize=5, label=f"{name}  (slope={slope:.2f})")
ax_cal.set_xlabel("Mean Predicted Probability"); ax_cal.set_ylabel("Observed Fraction")
ax_cal.set_title("Calibration — 5yr Serial Biopsy (OOF 5×5-fold)")
ax_cal.legend(loc="upper left", fontsize=8)
ax_cal.set_xlim([0,1]); ax_cal.set_ylim([0,1])
plt.tight_layout()
fig_cal.savefig(f"{FIGURES_DIR}/calibration_curves_5yr_serial.png", dpi=200, bbox_inches="tight")
plt.close(fig_cal)
print("  Saved: calibration_curves_5yr_serial.png")

# ============================================================
# Save results
# ============================================================
cv_rows, boot_rows, tune_rows = [], [], []
for name in MODELS:
    cv = cv_results[name]
    bt = boot_results[name]
    cv_rows.append({
        "Model": name,
        "CV AUROC (mean)":       round(cv["mean_auroc"], 3),
        "CV AUROC 95% CI lower": round(cv["ci_lower"], 3),
        "CV AUROC 95% CI upper": round(cv["ci_upper"], 3),
        "CV Brier Score":        round(cv["mean_brier"], 3),
        "CV Calibration Slope":  round(cv["mean_cal_slope"], 3),
    })
    boot_rows.append({
        "Model":                          name,
        "Apparent AUROC":                 bt["apparent_auroc"],
        "Optimism (bootstrap)":           bt["optimism_auroc"],
        "Bias-Corrected AUROC (Harrell)": bt["bc_auroc"],
        "Apparent Brier":                 bt["apparent_brier"],
        "Optimism Brier":                 bt["optimism_brier"],
        "Bias-Corrected Brier":           bt["bc_brier"],
    })
for name in ["Random Forest", "XGBoost", "LightGBM"]:
    p = best_params[name]
    tune_rows.append({"Model": name, **{k.replace("clf__", ""): v for k, v in p.items()}})

output_path = f"{OUTPUTS_DIR}/5yr_serial_model_results.xlsx"
with pd.ExcelWriter(output_path, engine="openpyxl") as writer:
    pd.DataFrame(cv_rows).to_excel(writer,   sheet_name="CV Results",           index=False)
    pd.DataFrame(boot_rows).to_excel(writer, sheet_name="Bootstrap (Harrell)",  index=False)
    pd.DataFrame(tune_rows).to_excel(writer, sheet_name="Best Hyperparameters", index=False)
print(f"  Saved: 5yr_serial_model_results.xlsx")

print(f"\n{'='*75}")
print(f"{'Model':<22} {'CV AUROC':>10}  {'95% CI':>18}  {'Brier':>7}  {'Cal Slope':>10}  {'BC AUROC':>9}")
print(f"{'='*75}")
for cv, bt in zip(cv_rows, boot_rows):
    ci = f"[{cv['CV AUROC 95% CI lower']:.3f}–{cv['CV AUROC 95% CI upper']:.3f}]"
    print(f"{cv['Model']:<22} {cv['CV AUROC (mean)']:>10.3f}  {ci:>18}  "
          f"{cv['CV Brier Score']:>7.3f}  {cv['CV Calibration Slope']:>10.3f}  "
          f"{bt['Bias-Corrected AUROC (Harrell)']:>9.3f}")
print(f"{'='*75}")
print("\n⚠  n=70, 34 events — wide CIs expected. Treat as exploratory.")
print("Done.")
