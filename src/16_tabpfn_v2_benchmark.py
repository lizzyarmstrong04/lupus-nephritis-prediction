"""
TabPFN v2 benchmark — same evaluation protocol as existing models:
  - 5×10-fold repeated stratified CV  (AUROC, Brier, calibration slope)
  - Harrell optimism-corrected bootstrap (500 iterations)
Compares against saved v1 results and traditional models.
"""

import os
import pandas as pd
import numpy as np
import warnings
import time
warnings.filterwarnings("ignore")

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, brier_score_loss
from sklearn.model_selection import RepeatedStratifiedKFold
import tabpfn_client
from tabpfn_client import TabPFNClassifier

tabpfn_client.set_access_token(os.environ["TABPFN_TOKEN"])  # export TABPFN_TOKEN="your_token"

PROCESSED_DIR = "/Users/elizabetharmstrong/Library/CloudStorage/OneDrive-ImperialCollegeLondon/Lupus Project/Data/Processed"
OUTPUTS_DIR   = "/Users/elizabetharmstrong/Library/CloudStorage/OneDrive-ImperialCollegeLondon/Lupus Project/outputs"

def calibration_slope(y_true, y_prob):
    log_odds = np.log(np.clip(y_prob, 1e-6, 1-1e-6) / (1 - np.clip(y_prob, 1e-6, 1-1e-6)))
    m = LogisticRegression(fit_intercept=True, max_iter=1000)
    m.fit(log_odds.reshape(-1, 1), y_true)
    return float(m.coef_[0][0])

def clf_factory():
    return TabPFNClassifier()

def run_cv(X, y, label):
    print(f"\n{'='*60}")
    print(f"TabPFN v2 — {label}")
    print(f"  n={len(y)}, events={int(y.sum())} ({y.mean()*100:.1f}%)")

    CV = RepeatedStratifiedKFold(n_splits=10, n_repeats=5, random_state=42)
    aurocs, briers, slopes = [], [], []

    for i, (tr, te) in enumerate(CV.split(X, y)):
        for attempt in range(4):
            try:
                clf = clf_factory()
                clf.fit(X.iloc[tr], y.iloc[tr])
                probs = clf.predict_proba(X.iloc[te])[:, 1]
                break
            except Exception as e:
                if attempt < 3:
                    print(f"  fold {i+1} attempt {attempt+1} failed ({e}), retrying in 10s...")
                    time.sleep(10)
                else:
                    raise
        aurocs.append(roc_auc_score(y.iloc[te], probs))
        briers.append(brier_score_loss(y.iloc[te], probs))
        slopes.append(calibration_slope(y.iloc[te].values, probs))
        time.sleep(2)  # polite pause between API calls
        if (i + 1) % 10 == 0:
            print(f"  fold {i+1}/50  running AUROC={np.mean(aurocs):.3f}", flush=True)

    return {
        "mean_auroc": np.mean(aurocs),
        "ci_lower":   np.percentile(aurocs, 2.5),
        "ci_upper":   np.percentile(aurocs, 97.5),
        "mean_brier": np.mean(briers),
        "mean_slope": np.mean(slopes),
    }

def harrell_bootstrap(X, y, n_boot=500):
    apparent_aurocs, optimisms = [], []
    clf_app = clf_factory()
    clf_app.fit(X, y)
    app_prob = clf_app.predict_proba(X)[:, 1]
    app_auroc = roc_auc_score(y, app_prob)

    rng = np.random.default_rng(42)
    for b in range(n_boot):
        idx = rng.choice(len(y), len(y), replace=True)
        X_b, y_b = X.iloc[idx], y.iloc[idx]
        if y_b.nunique() < 2:
            continue
        for attempt in range(4):
            try:
                clf_b = clf_factory()
                clf_b.fit(X_b, y_b)
                c_boot = roc_auc_score(y_b, clf_b.predict_proba(X_b)[:, 1])
                c_orig = roc_auc_score(y,   clf_b.predict_proba(X)[:, 1])
                break
            except Exception as e:
                if attempt < 3:
                    time.sleep(10)
                else:
                    raise
        apparent_aurocs.append(c_boot)
        optimisms.append(c_boot - c_orig)
        time.sleep(2)
        if (b + 1) % 100 == 0:
            print(f"  bootstrap {b+1}/{n_boot}", flush=True)

    optimism = np.mean(optimisms)
    bc_auroc = app_auroc - optimism
    print(f"  Apparent={app_auroc:.3f}  Optimism={optimism:.3f}  BC={bc_auroc:.3f}")
    return {"apparent": app_auroc, "optimism": optimism, "bc_auroc": bc_auroc}

def tabpfn_row(res):
    return pd.DataFrame([{
        "Model":                 "TabPFN v2",
        "CV AUROC (mean)":       round(res["mean_auroc"], 3),
        "CV AUROC 95% CI lower": round(res["ci_lower"],   3),
        "CV AUROC 95% CI upper": round(res["ci_upper"],   3),
        "CV Brier Score":        round(res["mean_brier"],  3),
        "CV Calibration Slope":  round(res["mean_slope"],  3),
    }])

# 1-year
df1 = pd.read_excel(f"{PROCESSED_DIR}/lupus_1yr_selected_clean.xlsx")
X1  = df1.drop(columns=["flare_1yr"])
y1  = df1["flare_1yr"].astype(int)

res1_cv   = run_cv(X1, y1, "1-Year Flare — CV")
print("\nRunning bootstrap for 1-year...")
res1_boot = harrell_bootstrap(X1, y1)

# 5-year
df5 = pd.read_excel(f"{PROCESSED_DIR}/lupus_5yr_selected_clean.xlsx")
X5  = df5.drop(columns=["flare_5yr"])
y5  = df5["flare_5yr"].astype(int)

res5_cv   = run_cv(X5, y5, "5-Year Flare — CV")
print("\nRunning bootstrap for 5-year...")
res5_boot = harrell_bootstrap(X5, y5)

# Comparison tables
existing_1yr = pd.read_excel(f"{OUTPUTS_DIR}/1yr_model_results.xlsx",  sheet_name="CV Results")
existing_5yr = pd.read_excel(f"{OUTPUTS_DIR}/5yr_model_results.xlsx",  sheet_name="CV Results")
tabpfn_v1    = pd.read_excel(f"{OUTPUTS_DIR}/tabpfn_comparison.xlsx",  sheet_name="1yr")
tabpfn_v1_5  = pd.read_excel(f"{OUTPUTS_DIR}/tabpfn_comparison.xlsx",  sheet_name="5yr")

# Add v2 row
v2_row_1 = tabpfn_row(res1_cv)
v2_row_5 = tabpfn_row(res5_cv)

compare_1 = pd.concat([existing_1yr,
                        tabpfn_v1[tabpfn_v1["Model"]=="TabPFN"].assign(Model="TabPFN v1"),
                        v2_row_1], ignore_index=True)
compare_5 = pd.concat([existing_5yr,
                        tabpfn_v1_5[tabpfn_v1_5["Model"]=="TabPFN"].assign(Model="TabPFN v1"),
                        v2_row_5], ignore_index=True)

print("\n\n" + "="*75)
print("COMPARISON — 1-YEAR FLARE")
print("="*75)
print(compare_1.to_string(index=False))

print("\n" + "="*75)
print("COMPARISON — 5-YEAR FLARE")
print("="*75)
print(compare_5.to_string(index=False))

# Save
with pd.ExcelWriter(f"{OUTPUTS_DIR}/tabpfn_v2_comparison.xlsx", engine="openpyxl") as w:
    compare_1.to_excel(w, sheet_name="1yr", index=False)
    compare_5.to_excel(w, sheet_name="5yr", index=False)
    pd.DataFrame([res1_boot]).to_excel(w, sheet_name="1yr_bootstrap", index=False)
    pd.DataFrame([res5_boot]).to_excel(w, sheet_name="5yr_bootstrap", index=False)

print(f"\nSaved: {OUTPUTS_DIR}/tabpfn_v2_comparison.xlsx")
print("Done.")
