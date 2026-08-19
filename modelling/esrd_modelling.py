"""
ESRD prediction — same protocol as flare prediction:
  - Loads from feature-selected datasets (esrd_5yr_selected.xlsx / esrd_10yr_selected.xlsx)
  - StandardScaler pipeline
  - RandomizedSearchCV hyperparameter tuning (40 iter, 5-fold inner CV)
  - 5×10-fold repeated stratified CV  (AUROC, Brier, calibration slope)
  - Harrell optimism-corrected bootstrap (1,000 iterations)
  - Saves results to outputs/esrd/esrd_model_results.xlsx
"""

import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings("ignore")
import os, json

from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import (RepeatedStratifiedKFold, StratifiedKFold,
                                     RandomizedSearchCV)
from sklearn.metrics import roc_auc_score, brier_score_loss
from sklearn.pipeline import Pipeline
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier

PROCESSED_DIR = "/Users/elizabetharmstrong/Library/CloudStorage/OneDrive-ImperialCollegeLondon/Lupus Project/Data/Processed"
OUT_DIR       = "/Users/elizabetharmstrong/Library/CloudStorage/OneDrive-ImperialCollegeLondon/Lupus Project/outputs/esrd"
os.makedirs(OUT_DIR, exist_ok=True)

# Helpers

def calibration_slope(y_true, y_prob):
    log_odds = np.log(np.clip(y_prob, 1e-6, 1-1e-6) / (1 - np.clip(y_prob, 1e-6, 1-1e-6)))
    m = LogisticRegression(fit_intercept=True, max_iter=1000)
    m.fit(log_odds.reshape(-1, 1), y_true)
    return float(m.coef_[0][0])

def harrell_bootstrap(pipeline, X, y, n_boot=1000, seed=42):
    """
    Harrell optimism-corrected bootstrap.
    For each bootstrap sample b:
      1. Fit on D_b
      2. Evaluate on D_b  → C_boot
      3. Evaluate on original D → C_orig
      4. optimism_b = C_boot − C_orig
    Bias-corrected = apparent − mean(optimism)
    """
    pipeline.fit(X, y)
    p_app          = pipeline.predict_proba(X)[:, 1]
    apparent_auroc = roc_auc_score(y, p_app)
    apparent_brier = brier_score_loss(y, p_app)

    rng  = np.random.default_rng(seed)
    n    = len(y)
    X_np = X.values
    y_np = y.values
    opt_auroc, opt_brier = [], []

    for b in range(n_boot):
        idx = rng.integers(0, n, size=n)
        X_b = pd.DataFrame(X_np[idx], columns=X.columns)
        y_b = pd.Series(y_np[idx])
        if y_b.nunique() < 2:
            continue
        pipeline.fit(X_b, y_b)
        p_boot = pipeline.predict_proba(X_b)[:, 1]
        p_orig = pipeline.predict_proba(X)[:, 1]
        opt_auroc.append(roc_auc_score(y_b, p_boot) - roc_auc_score(y, p_orig))
        opt_brier.append(brier_score_loss(y_b, p_boot) - brier_score_loss(y, p_orig))
        if (b + 1) % 200 == 0:
            print(f"    bootstrap {b+1}/{n_boot}", flush=True)

    return {
        "apparent_auroc": round(apparent_auroc, 3),
        "optimism_auroc": round(np.mean(opt_auroc), 3),
        "bc_auroc":       round(apparent_auroc - np.mean(opt_auroc), 3),
        "apparent_brier": round(apparent_brier, 3),
        "optimism_brier": round(np.mean(opt_brier), 3),
        "bc_brier":       round(apparent_brier - np.mean(opt_brier), 3),
    }

# Main modelling function

def run_models(data_path, outcome_col, horizon_label):
    df = pd.read_excel(data_path)
    X  = df.drop(columns=[outcome_col])
    y  = df[outcome_col].astype(int)

    print(f"\n{'='*65}")
    print(f"ESRD — {horizon_label}")
    print(f"  n={len(y)}, events={int(y.sum())} ({y.mean()*100:.1f}%)")
    print(f"  Features ({X.shape[1]}): {list(X.columns)}")

    spw = round((y == 0).sum() / (y == 1).sum(), 2)

    # Hyperparameter tuning
    print("\n  Tuning hyperparameters (5-fold, RandomizedSearch, n_iter=40)...")
    inner_cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    param_grids = {
        "Random Forest": {
            "clf__n_estimators":     [200, 300, 500],
            "clf__max_depth":        [3, 5, 7, None],
            "clf__min_samples_leaf": [5, 10, 20],
            "clf__max_features":     ["sqrt", 0.5, 0.7],
        },
        "XGBoost": {
            "clf__n_estimators":     [100, 200],
            "clf__max_depth":        [2, 3],
            "clf__learning_rate":    [0.01, 0.05],
            "clf__subsample":        [0.5, 0.6],
            "clf__colsample_bytree": [0.5, 0.6],
            "clf__min_child_weight": [5, 10, 15],
            "clf__reg_alpha":        [0.1, 1.0, 2.0],
            "clf__reg_lambda":       [1.0, 5.0, 10.0],
        },
        "LightGBM": {
            "clf__n_estimators":      [100, 200, 300],
            "clf__max_depth":         [2, 3, 4],
            "clf__learning_rate":     [0.01, 0.05, 0.1],
            "clf__subsample":         [0.7, 0.8, 1.0],
            "clf__num_leaves":        [15, 31, 63],
            "clf__min_child_samples": [10, 20, 30],
        },
    }

    base_pipes = {
        "Random Forest": Pipeline([("s", StandardScaler()),
            ("clf", RandomForestClassifier(random_state=42, class_weight="balanced"))]),
        "XGBoost": Pipeline([("s", StandardScaler()),
            ("clf", XGBClassifier(random_state=42, eval_metric="logloss",
                                  verbosity=0, scale_pos_weight=spw))]),
        "LightGBM": Pipeline([("s", StandardScaler()),
            ("clf", LGBMClassifier(random_state=42, verbose=-1, class_weight="balanced"))]),
    }

    best_params = {}
    for name, pipe in base_pipes.items():
        search = RandomizedSearchCV(pipe, param_grids[name], n_iter=40,
                                    scoring="roc_auc", cv=inner_cv,
                                    random_state=42, n_jobs=-1, refit=True)
        search.fit(X, y)
        best_params[name] = search.best_params_
        print(f"    {name}: best CV AUROC={search.best_score_:.3f}  params={search.best_params_}")

    def make_rf(p):
        return RandomForestClassifier(
            n_estimators=p.get("clf__n_estimators", 300),
            max_depth=p.get("clf__max_depth", 3),
            min_samples_leaf=p.get("clf__min_samples_leaf", 10),
            max_features=p.get("clf__max_features", "sqrt"),
            class_weight="balanced", random_state=42)

    def make_xgb(p):
        return XGBClassifier(
            n_estimators=p.get("clf__n_estimators", 100),
            max_depth=p.get("clf__max_depth", 2),
            learning_rate=p.get("clf__learning_rate", 0.01),
            subsample=p.get("clf__subsample", 0.5),
            colsample_bytree=p.get("clf__colsample_bytree", 0.5),
            min_child_weight=p.get("clf__min_child_weight", 5),
            reg_alpha=p.get("clf__reg_alpha", 1.0),
            reg_lambda=p.get("clf__reg_lambda", 5.0),
            scale_pos_weight=spw, eval_metric="logloss", verbosity=0, random_state=42)

    def make_lgbm(p):
        return LGBMClassifier(
            n_estimators=p.get("clf__n_estimators", 100),
            max_depth=p.get("clf__max_depth", 3),
            learning_rate=p.get("clf__learning_rate", 0.01),
            subsample=p.get("clf__subsample", 0.7),
            num_leaves=p.get("clf__num_leaves", 15),
            min_child_samples=p.get("clf__min_child_samples", 10),
            class_weight="balanced", verbose=-1, random_state=42)

    MODELS = {
        "Logistic Regression": Pipeline([("s", StandardScaler()),
            ("clf", LogisticRegression(max_iter=1000, C=1.0,
                                       class_weight="balanced", random_state=42))]),
        "Random Forest": Pipeline([("s", StandardScaler()),
            ("clf", make_rf(best_params["Random Forest"]))]),
        "XGBoost":  Pipeline([("s", StandardScaler()),
            ("clf", make_xgb(best_params["XGBoost"]))]),
        "LightGBM": Pipeline([("s", StandardScaler()),
            ("clf", make_lgbm(best_params["LightGBM"]))]),
    }

    # 5×10-fold CV
    print("\n  Running 5×10-fold CV...")
    CV = RepeatedStratifiedKFold(n_splits=10, n_repeats=5, random_state=42)
    cv_results = {}

    for name, pipeline in MODELS.items():
        fold_aurocs, fold_briers, fold_slopes = [], [], []
        oof_probs  = np.zeros(len(y))
        oof_counts = np.zeros(len(y))

        for tr_idx, te_idx in CV.split(X, y):
            pipeline.fit(X.iloc[tr_idx], y.iloc[tr_idx])
            probs = pipeline.predict_proba(X.iloc[te_idx])[:, 1]
            fold_aurocs.append(roc_auc_score(y.iloc[te_idx], probs))
            fold_briers.append(brier_score_loss(y.iloc[te_idx], probs))
            fold_slopes.append(calibration_slope(y.iloc[te_idx].values, probs))
            oof_probs[te_idx]  += probs
            oof_counts[te_idx] += 1

        oof_probs /= oof_counts
        mean_auroc = np.mean(fold_aurocs)
        ci_lo = np.percentile(fold_aurocs, 2.5)
        ci_hi = np.percentile(fold_aurocs, 97.5)
        cv_results[name] = {
            "auroc": mean_auroc, "ci_lo": ci_lo, "ci_hi": ci_hi,
            "brier": np.mean(fold_briers), "slope": np.mean(fold_slopes),
            "oof": oof_probs, "fold_aurocs": fold_aurocs,
        }
        print(f"    {name:<22} AUROC={mean_auroc:.3f} [{ci_lo:.3f}–{ci_hi:.3f}]  "
              f"Brier={np.mean(fold_briers):.3f}  Slope={np.mean(fold_slopes):.3f}")

    # Harrell bootstrap (1,000 iterations)
    print("\n  Running Harrell bootstrap (1,000 iterations)...")
    boot_results = {}
    for name, pipeline in MODELS.items():
        print(f"    {name}...")
        boot_results[name] = harrell_bootstrap(pipeline, X, y, n_boot=1000)
        r = boot_results[name]
        print(f"      Apparent={r['apparent_auroc']}  Optimism={r['optimism_auroc']}  "
              f"BC={r['bc_auroc']}")

    # Collate results
    cv_rows = []
    boot_rows = []
    for name in MODELS:
        cv  = cv_results[name]
        boo = boot_results[name]
        cv_rows.append({
            "Model":                    name,
            "CV AUROC (mean)":          round(cv["auroc"], 3),
            "CV AUROC 95% CI lower":    round(cv["ci_lo"], 3),
            "CV AUROC 95% CI upper":    round(cv["ci_hi"], 3),
            "CV Brier Score":           round(cv["brier"], 3),
            "CV Calibration Slope":     round(cv["slope"], 3),
        })
        boot_rows.append({
            "Model":           name,
            "Apparent AUROC":  boo["apparent_auroc"],
            "Optimism AUROC":  boo["optimism_auroc"],
            "BC AUROC":        boo["bc_auroc"],
            "Apparent Brier":  boo["apparent_brier"],
            "Optimism Brier":  boo["optimism_brier"],
            "BC Brier":        boo["bc_brier"],
        })

    return pd.DataFrame(cv_rows), pd.DataFrame(boot_rows), cv_results, best_params

# Run for 5-year and 10-year ESRD

cv5,  boot5,  oof5,  params5  = run_models(
    f"{PROCESSED_DIR}/esrd_5yr_selected.xlsx",  "esrd_5yr",  "5-Year")
cv10, boot10, oof10, params10 = run_models(
    f"{PROCESSED_DIR}/esrd_10yr_selected.xlsx", "esrd_10yr", "10-Year")

# Save to Excel
with pd.ExcelWriter(f"{OUT_DIR}/esrd_model_results.xlsx", engine="openpyxl") as w:
    cv5.to_excel(w,   sheet_name="5yr CV Results",      index=False)
    boot5.to_excel(w, sheet_name="5yr Bootstrap",       index=False)
    cv10.to_excel(w,  sheet_name="10yr CV Results",     index=False)
    boot10.to_excel(w,sheet_name="10yr Bootstrap",      index=False)

with open(f"{OUT_DIR}/esrd_best_params.json", "w") as f:
    json.dump({"5yr": params5, "10yr": params10}, f, indent=2, default=str)

print(f"\nSaved: {OUT_DIR}/esrd_model_results.xlsx")
print(f"Saved: {OUT_DIR}/esrd_best_params.json")
print("\nDone.")
