"""
DeLong's test — pairwise AUROC comparison for ESRD models (5-year and 10-year).
Loads from feature-selected datasets; uses best params from 01_esrd_modelling.py.
OOF predictions from 5×10-fold CV; Bonferroni-Holm correction.
"""

import pandas as pd
import numpy as np
import warnings
import json
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
OUT_DIR       = "/Users/elizabetharmstrong/Library/CloudStorage/OneDrive-ImperialCollegeLondon/Lupus Project/outputs/esrd"

with open(f"{OUT_DIR}/esrd_best_params.json") as f:
    BEST_PARAMS = json.load(f)

# DeLong implementation
def _compute_midrank(x):
    J = np.argsort(x)
    Z = x[J]; N = len(x); T = np.zeros(N, dtype=float); i = 0
    while i < N:
        j = i
        while j < N and Z[j] == Z[i]: j += 1
        T[i:j] = 0.5 * (i + j - 1); i = j
    T2 = np.empty(N, dtype=float); T2[J] = T + 1
    return T2

def _fastDeLong(pst, m):
    n = pst.shape[1] - m; k = pst.shape[0]
    pos = pst[:, :m]; neg = pst[:, m:]
    tx = np.empty([k, m]); ty = np.empty([k, n]); tz = np.empty([k, m+n])
    for r in range(k):
        tx[r] = _compute_midrank(pos[r]); ty[r] = _compute_midrank(neg[r])
        tz[r] = _compute_midrank(pst[r])
    aucs = (tz[:, :m].sum(1) - tx.sum(1)) / (m * n)
    v01  = (tz[:, :m] - tx) / n; v10 = 1 - (tz[:, m:] - ty) / m
    return aucs, np.cov(v01) / m + np.cov(v10) / n

def delong_roc_test(y_true, prob_a, prob_b):
    y_true = np.asarray(y_true)
    order  = (-y_true).argsort()
    m      = int(y_true.sum())
    aucs, cov = _fastDeLong(np.vstack([prob_a[order], prob_b[order]]), m)
    l = np.array([[1, -1]])
    z = np.diff(aucs) / np.sqrt(l @ cov @ l.T + 1e-12)
    return float(aucs[0]), float(aucs[1]), float(2 * stats.norm.sf(abs(z)))

def bonferroni_holm(p_values):
    n = len(p_values); order = np.argsort(p_values); p_adj = np.array(p_values, dtype=float)
    for rank, idx in enumerate(order): p_adj[idx] = min(1.0, p_values[idx] * (n - rank))
    for i in range(1, n): p_adj[order[i]] = max(p_adj[order[i]], p_adj[order[i-1]])
    return p_adj.tolist()

# Model builder (uses saved best params)
def build_models(horizon_key, scale_pos_weight):
    p_rf   = BEST_PARAMS[horizon_key]["Random Forest"]
    p_xgb  = BEST_PARAMS[horizon_key]["XGBoost"]
    p_lgbm = BEST_PARAMS[horizon_key]["LightGBM"]

    return {
        "Logistic Regression": Pipeline([("s", StandardScaler()),
            ("clf", LogisticRegression(max_iter=1000, C=1.0,
                                       class_weight="balanced", random_state=42))]),
        "Random Forest": Pipeline([("s", StandardScaler()),
            ("clf", RandomForestClassifier(
                n_estimators=p_rf["clf__n_estimators"],
                max_depth=p_rf["clf__max_depth"],
                min_samples_leaf=p_rf["clf__min_samples_leaf"],
                max_features=p_rf["clf__max_features"],
                class_weight="balanced", random_state=42))]),
        "XGBoost": Pipeline([("s", StandardScaler()),
            ("clf", XGBClassifier(
                n_estimators=p_xgb["clf__n_estimators"],
                max_depth=p_xgb["clf__max_depth"],
                learning_rate=p_xgb["clf__learning_rate"],
                subsample=p_xgb["clf__subsample"],
                colsample_bytree=p_xgb["clf__colsample_bytree"],
                min_child_weight=p_xgb["clf__min_child_weight"],
                reg_alpha=p_xgb["clf__reg_alpha"],
                reg_lambda=p_xgb["clf__reg_lambda"],
                scale_pos_weight=scale_pos_weight,
                eval_metric="logloss", verbosity=0, random_state=42))]),
        "LightGBM": Pipeline([("s", StandardScaler()),
            ("clf", LGBMClassifier(
                n_estimators=p_lgbm["clf__n_estimators"],
                max_depth=p_lgbm["clf__max_depth"],
                learning_rate=p_lgbm["clf__learning_rate"],
                subsample=p_lgbm["clf__subsample"],
                num_leaves=p_lgbm["clf__num_leaves"],
                min_child_samples=p_lgbm["clf__min_child_samples"],
                class_weight="balanced", verbose=-1, random_state=42))]),
    }

# OOF CV + DeLong
def run_delong(data_path, outcome_col, horizon_key, cohort_name):
    df  = pd.read_excel(data_path)
    X   = df.drop(columns=[outcome_col])
    y   = df[outcome_col].astype(int)
    spw = round((y == 0).sum() / (y == 1).sum(), 2)

    print(f"\n{'='*65}")
    print(f"COHORT: {cohort_name}  (n={len(y)}, events={int(y.sum())}, 5×10-fold CV)")

    models = build_models(horizon_key, spw)
    CV     = RepeatedStratifiedKFold(n_splits=10, n_repeats=5, random_state=42)
    oof    = {name: np.zeros(len(y)) for name in models}
    cnt    = {name: np.zeros(len(y)) for name in models}

    for name, pipeline in models.items():
        print(f"  OOF CV — {name}...", end=" ", flush=True)
        for tr_idx, te_idx in CV.split(X, y):
            pipeline.fit(X.iloc[tr_idx], y.iloc[tr_idx])
            oof[name][te_idx] += pipeline.predict_proba(X.iloc[te_idx])[:, 1]
            cnt[name][te_idx] += 1
        oof[name] /= cnt[name]
        print(f"AUROC={roc_auc_score(y, oof[name]):.3f}")

    pairs  = list(combinations(models.keys(), 2))
    rows   = []
    p_raw  = []
    for m1, m2 in pairs:
        a1, a2, p = delong_roc_test(y.values, oof[m1], oof[m2])
        rows.append({"Cohort": cohort_name, "Model A": m1, "Model B": m2,
                     "AUROC A": round(a1, 3), "AUROC B": round(a2, 3),
                     "p-value (DeLong)": round(p, 4)})
        p_raw.append(p)

    p_adj = bonferroni_holm(p_raw)
    for row, pa in zip(rows, p_adj):
        row["p-value (Holm-corrected)"] = round(pa, 4)
        row["Significant (p<0.05)"]     = "Yes" if pa < 0.05 else "No"

    print(f"\n  {'Model A':<22} vs  {'Model B':<22}  {'p-raw':>8}  {'p-adj':>8}  Sig")
    print(f"  {'-'*80}")
    for r in rows:
        print(f"  {r['Model A']:<22} vs  {r['Model B']:<22}  "
              f"{r['p-value (DeLong)']:>8.4f}  {r['p-value (Holm-corrected)']:>8.4f}  "
              f"{r['Significant (p<0.05)']}")
    return rows

# Run
all_rows = []
all_rows.extend(run_delong(
    f"{PROCESSED_DIR}/esrd_5yr_selected.xlsx",  "esrd_5yr",  "5yr",  "ESRD 5-Year"))
all_rows.extend(run_delong(
    f"{PROCESSED_DIR}/esrd_10yr_selected.xlsx", "esrd_10yr", "10yr", "ESRD 10-Year"))

df_out   = pd.DataFrame(all_rows)
out_path = f"{OUT_DIR}/esrd_delong_results.xlsx"
with pd.ExcelWriter(out_path, engine="openpyxl") as w:
    df_out.to_excel(w, sheet_name="All", index=False)
    for cohort in ["ESRD 5-Year", "ESRD 10-Year"]:
        df_out[df_out["Cohort"] == cohort].to_excel(
            w, sheet_name=cohort.replace(" ", "_"), index=False)

print(f"\nSaved: {out_path}")
print("Done.")
