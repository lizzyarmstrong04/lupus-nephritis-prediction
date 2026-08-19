"""
DeLong's test — pairwise comparison of model AUROCs across all three cohorts.

For each cohort, out-of-fold (OOF) predictions are generated via the same
CV scheme used in modelling.  DeLong's test is applied to every pair of models;
the p-value matrix (and adjusted p-values) are saved to Excel.

Reference: DeLong ER, DeLong DM, Clarke-Pearson DL (1988). Biometrics 44(3):837-845.
"""
import pandas as pd
import numpy as np
import json
import warnings
warnings.filterwarnings("ignore")
from itertools import combinations

from scipy import stats
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.metrics import roc_auc_score
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier

PROCESSED_DIR = "/Users/elizabetharmstrong/Library/CloudStorage/OneDrive-ImperialCollegeLondon/Lupus Project/Data/Processed"
OUTPUTS_DIR   = "/Users/elizabetharmstrong/Library/CloudStorage/OneDrive-ImperialCollegeLondon/Lupus Project/outputs"

# DeLong's test (fast implementation)

def _compute_midrank(x):
    J = np.argsort(x)
    Z = x[J]
    N = len(x)
    T = np.zeros(N, dtype=float)
    i = 0
    while i < N:
        j = i
        while j < N and Z[j] == Z[i]:
            j += 1
        T[i:j] = 0.5 * (i + j - 1)
        i = j
    T2 = np.empty(N, dtype=float)
    T2[J] = T + 1
    return T2

def _fastDeLong(predictions_sorted_transposed, label_1_count):
    m = label_1_count
    n = predictions_sorted_transposed.shape[1] - m
    positive_examples = predictions_sorted_transposed[:, :m]
    negative_examples = predictions_sorted_transposed[:, m:]
    k = predictions_sorted_transposed.shape[0]
    tx = np.empty([k, m], dtype=float)
    ty = np.empty([k, n], dtype=float)
    tz = np.empty([k, m + n], dtype=float)
    for r in range(k):
        tx[r, :] = _compute_midrank(positive_examples[r, :])
        ty[r, :] = _compute_midrank(negative_examples[r, :])
        tz[r, :] = _compute_midrank(predictions_sorted_transposed[r, :])
    aucs = (tz[:, :m].sum(axis=1) - tx.sum(axis=1)) / (m * n)
    v01  = (tz[:, :m] - tx) / n
    v10  = 1 - (tz[:, m:] - ty) / m
    sx   = np.cov(v01)
    sy   = np.cov(v10)
    delongcov = sx / m + sy / n
    return aucs, delongcov

def delong_roc_test(y_true, prob_a, prob_b):
    """Returns (auroc_a, auroc_b, p_value) using DeLong's method."""
    y_true = np.asarray(y_true)
    order  = (-y_true).argsort()
    label_1_count = int(y_true.sum())
    pst = np.vstack([prob_a[order], prob_b[order]])
    aucs, cov = _fastDeLong(pst, label_1_count)
    l = np.array([[1, -1]])
    z = np.diff(aucs) / np.sqrt(l @ cov @ l.T + 1e-12)
    p = float(2 * stats.norm.sf(abs(z)))
    return float(aucs[0]), float(aucs[1]), p

def bonferroni_holm(p_values):
    """Bonferroni-Holm correction. Returns adjusted p-values in same order."""
    n = len(p_values)
    order = np.argsort(p_values)
    p_adj = np.array(p_values, dtype=float)
    for rank, idx in enumerate(order):
        p_adj[idx] = min(1.0, p_values[idx] * (n - rank))
    # enforce monotonicity
    for i in range(1, n):
        p_adj[order[i]] = max(p_adj[order[i]], p_adj[order[i-1]])
    return p_adj.tolist()

# Model builders (mirror each modelling script exactly)

def build_models(best_params, scale_pos_weight):
    def make_rf(p):
        return RandomForestClassifier(
            n_estimators=p.get("clf__n_estimators", 200),
            max_depth=p.get("clf__max_depth", None),
            min_samples_leaf=p.get("clf__min_samples_leaf", 10),
            max_features=p.get("clf__max_features", "sqrt"),
            class_weight="balanced", random_state=42)

    def make_xgb(p):
        return XGBClassifier(
            n_estimators=p.get("clf__n_estimators", 100),
            max_depth=p.get("clf__max_depth", 2),
            learning_rate=p.get("clf__learning_rate", 0.05),
            subsample=p.get("clf__subsample", 0.5),
            colsample_bytree=p.get("clf__colsample_bytree", 0.5),
            min_child_weight=p.get("clf__min_child_weight", 5),
            reg_alpha=p.get("clf__reg_alpha", 1.0),
            reg_lambda=p.get("clf__reg_lambda", 5.0),
            eval_metric="logloss", verbosity=0, random_state=42,
            scale_pos_weight=scale_pos_weight)

    def make_lgbm(p):
        return LGBMClassifier(
            n_estimators=p.get("clf__n_estimators", 100),
            max_depth=p.get("clf__max_depth", 3),
            learning_rate=p.get("clf__learning_rate", 0.05),
            subsample=p.get("clf__subsample", 0.8),
            num_leaves=p.get("clf__num_leaves", 15),
            min_child_samples=p.get("clf__min_child_samples", 10),
            verbose=-1, random_state=42, class_weight="balanced")

    return {
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

def get_oof_probs(models, X, y, n_splits, n_repeats):
    cv = RepeatedStratifiedKFold(n_splits=n_splits, n_repeats=n_repeats, random_state=42)
    oof = {name: np.zeros(len(y)) for name in models}
    cnt = {name: np.zeros(len(y)) for name in models}
    for name, pipeline in models.items():
        print(f"    OOF CV — {name}...", end=" ", flush=True)
        for tr_idx, te_idx in cv.split(X, y):
            pipeline.fit(X.iloc[tr_idx], y.iloc[tr_idx])
            probs = pipeline.predict_proba(X.iloc[te_idx])[:, 1]
            oof[name][te_idx] += probs
            cnt[name][te_idx] += 1
        oof[name] = oof[name] / np.where(cnt[name] > 0, cnt[name], 1)
        print(f"AUROC={roc_auc_score(y, oof[name]):.3f}")
    return oof

def run_delong(cohort_name, X, y, best_params, scale_pos_weight, n_splits, n_repeats):
    print(f"\n{'='*65}")
    print(f"COHORT: {cohort_name}  (n={len(y)}, events={y.sum()}, "
          f"{n_splits}×{n_repeats}-fold CV)")
    print(f"{'='*65}")

    models = build_models(best_params, scale_pos_weight)
    oof    = get_oof_probs(models, X, y, n_splits, n_repeats)

    model_names = list(models.keys())
    pairs       = list(combinations(model_names, 2))

    rows = []
    p_values_raw = []
    for m1, m2 in pairs:
        a1, a2, p = delong_roc_test(y.values, oof[m1], oof[m2])
        rows.append({
            "Cohort":  cohort_name,
            "Model A": m1,
            "Model B": m2,
            "AUROC A": round(a1, 3),
            "AUROC B": round(a2, 3),
            "p-value (DeLong)": round(p, 4),
        })
        p_values_raw.append(p)

    p_adj = bonferroni_holm(p_values_raw)
    for row, padj in zip(rows, p_adj):
        row["p-value (Holm-corrected)"] = round(padj, 4)
        row["Significant (p<0.05)"] = "Yes" if padj < 0.05 else "No"

    print(f"\n  Pairwise DeLong's test (Bonferroni-Holm corrected):")
    print(f"  {'Model A':<22} vs  {'Model B':<22}  {'p-raw':>8}  {'p-adj':>8}  {'Sig':>5}")
    print(f"  {'-'*75}")
    for r in rows:
        print(f"  {r['Model A']:<22} vs  {r['Model B']:<22}  "
              f"{r['p-value (DeLong)']:>8.4f}  "
              f"{r['p-value (Holm-corrected)']:>8.4f}  "
              f"{r['Significant (p<0.05)']:>5}")
    return rows

# COHORT CONFIGS

cohorts = []

# 1-year
with open(f"{OUTPUTS_DIR}/1yr_best_params.json") as f:
    params_1yr = json.load(f)
df_1yr = pd.read_excel(f"{PROCESSED_DIR}/lupus_1yr_selected_clean.xlsx")
X_1yr  = df_1yr.drop(columns=["flare_1yr"])
y_1yr  = df_1yr["flare_1yr"].astype(int)
cohorts.append(("1-Year", X_1yr, y_1yr, params_1yr, round(331/99, 2), 10, 5))

# 5-year
with open(f"{OUTPUTS_DIR}/5yr_best_params.json") as f:
    params_5yr = json.load(f)
df_5yr = pd.read_excel(f"{PROCESSED_DIR}/lupus_5yr_selected_clean.xlsx")
X_5yr  = df_5yr.drop(columns=["flare_5yr"])
y_5yr  = df_5yr["flare_5yr"].astype(int)
cohorts.append(("5-Year", X_5yr, y_5yr, params_5yr, round(190/166, 2), 10, 5))

# Serial biopsy — no saved JSON, build params from the serial modelling run
# (hardcode the best params from the 11_serial_modelling_5yr.py output)
params_serial = {
    "Random Forest": {
        "clf__n_estimators": 100, "clf__min_samples_leaf": 5,
        "clf__max_features": "sqrt", "clf__max_depth": 2,
    },
    "XGBoost": {
        "clf__subsample": 0.5, "clf__reg_lambda": 5.0, "clf__reg_alpha": 2.0,
        "clf__n_estimators": 100, "clf__min_child_weight": 3,
        "clf__max_depth": 2, "clf__learning_rate": 0.01, "clf__colsample_bytree": 0.7,
    },
    "LightGBM": {
        "clf__subsample": 1.0, "clf__num_leaves": 31, "clf__n_estimators": 50,
        "clf__min_child_samples": 20, "clf__max_depth": 3, "clf__learning_rate": 0.01,
    },
}
df_serial = pd.read_excel(f"{PROCESSED_DIR}/lupus_5yr_serial_selected.xlsx")
X_serial  = df_serial.drop(columns=["flare_5yr"])
y_serial  = df_serial["flare_5yr"].astype(int)
cohorts.append(("Serial Biopsy", X_serial, y_serial, params_serial, round(36/34, 2), 5, 5))

# RUN

all_rows = []
for args in cohorts:
    all_rows.extend(run_delong(*args))

# SAVE

df_out = pd.DataFrame(all_rows)
output_path = f"{OUTPUTS_DIR}/delong_test_results.xlsx"

with pd.ExcelWriter(output_path, engine="openpyxl") as writer:
    df_out.to_excel(writer, sheet_name="All Cohorts", index=False)
    for cohort in ["1-Year", "5-Year", "Serial Biopsy"]:
        sub = df_out[df_out["Cohort"] == cohort]
        sheet = cohort.replace(" ", "_").replace("-", "")
        sub.to_excel(writer, sheet_name=sheet, index=False)

print(f"\nSaved: {output_path}")
print("Done.")
