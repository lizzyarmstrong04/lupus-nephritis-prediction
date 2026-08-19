"""
Recompute a 95% CI for the Harrell bootstrap bias-corrected (BC) AUROC, for
all 4 main models x 5 cohorts.

Why this is needed: the original per-cohort modelling scripts
(src/1_year/06_modelling_1yr.py, src/5_year/05_modelling_5yr.py,
src/5_year/11_serial_modelling_5yr.py, src/esrd/01_esrd_modelling.py) run a
1000-iteration Harrell bootstrap but only ever kept the MEAN bias-corrected
AUROC - the per-iteration values were discarded, so no CI was ever saved
for the BC AUROC (unlike CV AUROC, which does have a saved percentile CI
from fold values). Table 1 (src/25_tables_1_2.py) is being switched from
CV AUROC to BC AUROC as its primary discrimination statistic (BC AUROC is
the paper's designated primary metric per Table S5's footnote / TRIPOD
guidance), so it needs a CI to go with it.

Approach: re-run ONLY the bootstrap step (not hyperparameter tuning - the
already-tuned best_params are loaded from the saved JSON/xlsx sources, not
re-searched), this time keeping every iteration's bias-corrected AUROC
(= apparent_auroc - optimism_b for iteration b) to get a real 2.5/97.5
percentile CI. Each cohort's pipeline (scaler, fixed hyperparameters,
random_state=42, class weighting / scale_pos_weight) is reproduced exactly
from its source script so the point estimate can be cross-checked against
the already-published number before trusting the new CI.

IMPORTANT: the 5-year cohort is loaded exactly as the original script does
(11 predictors, dsDNA/SM/APL column NOT dropped) - this reproduces the
already-published BC AUROC exactly, matching the known pre-existing data
file issue (see generate_methods_doc.py's data-integrity note). Fixing
that is a separate, already-flagged issue and out of scope here.

Saves: outputs/bc_auroc_ci.xlsx (Cohort, Model, Apparent, Optimism,
BC_AUROC, BC_AUROC_CI_lower, BC_AUROC_CI_upper, Matches_Published)
"""
import json
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.metrics import roc_auc_score
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier

BASE = "/Users/elizabetharmstrong/Library/CloudStorage/OneDrive-ImperialCollegeLondon/Lupus Project"
PROC = f"{BASE}/Data/Processed"
OUT = f"{BASE}/outputs"


def harrell_bootstrap_with_ci(pipeline, X, y, n_boot=1000, seed=42):
    """Same algorithm as the original per-cohort scripts, but retains every
    iteration's bias-corrected AUROC to compute a percentile CI."""
    pipeline.fit(X, y)
    p_app = pipeline.predict_proba(X)[:, 1]
    apparent_auroc = roc_auc_score(y, p_app)

    rng = np.random.default_rng(seed)
    n = len(y)
    X_np = X.values
    y_np = y.values

    optimisms = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, size=n)
        X_b = pd.DataFrame(X_np[idx], columns=X.columns)
        y_b = pd.Series(y_np[idx])
        if y_b.nunique() < 2:
            continue
        pipeline.fit(X_b, y_b)
        p_boot = pipeline.predict_proba(X_b)[:, 1]
        c_boot = roc_auc_score(y_b, p_boot)
        p_orig = pipeline.predict_proba(X)[:, 1]
        c_orig = roc_auc_score(y, p_orig)
        optimisms.append(c_boot - c_orig)

    optimisms = np.array(optimisms)
    bc_per_iter = apparent_auroc - optimisms  # per-iteration bias-corrected AUROC
    mean_optimism = optimisms.mean()
    bc_auroc = apparent_auroc - mean_optimism

    return {
        "apparent_auroc": round(apparent_auroc, 3),
        "optimism_auroc": round(mean_optimism, 3),
        "bc_auroc": round(bc_auroc, 3),
        "bc_ci_lower": round(np.percentile(bc_per_iter, 2.5), 3),
        "bc_ci_upper": round(np.percentile(bc_per_iter, 97.5), 3),
    }


def make_rf(p, random_state=42):
    return RandomForestClassifier(
        n_estimators=int(p.get("clf__n_estimators", 300)),
        max_depth=p.get("clf__max_depth", None),
        min_samples_leaf=int(p.get("clf__min_samples_leaf", 10)),
        max_features=p.get("clf__max_features", "sqrt"),
        class_weight="balanced", random_state=random_state, n_jobs=1,
    )


def make_xgb(p, spw, random_state=42):
    return XGBClassifier(
        n_estimators=int(p.get("clf__n_estimators", 100)),
        max_depth=int(p.get("clf__max_depth", 2)),
        learning_rate=p.get("clf__learning_rate", 0.05),
        subsample=p.get("clf__subsample", 0.5),
        colsample_bytree=p.get("clf__colsample_bytree", 0.5),
        min_child_weight=p.get("clf__min_child_weight", 10),
        reg_alpha=p.get("clf__reg_alpha", 1.0),
        reg_lambda=p.get("clf__reg_lambda", 5.0),
        eval_metric="logloss", verbosity=0, random_state=random_state,
        scale_pos_weight=spw, n_jobs=1,
    )


def make_lgbm(p, random_state=42):
    return LGBMClassifier(
        n_estimators=int(p.get("clf__n_estimators", 200)),
        max_depth=int(p.get("clf__max_depth", 3)),
        learning_rate=p.get("clf__learning_rate", 0.05),
        subsample=p.get("clf__subsample", 0.8),
        num_leaves=int(p.get("clf__num_leaves", 31)),
        min_child_samples=int(p.get("clf__min_child_samples", 20)),
        verbose=-1, random_state=random_state, class_weight="balanced", n_jobs=1,
    )


def build_models(best_params, spw):
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
            ("clf", make_xgb(best_params["XGBoost"], spw)),
        ]),
        "LightGBM": Pipeline([
            ("scaler", StandardScaler()),
            ("clf", make_lgbm(best_params["LightGBM"])),
        ]),
    }


# --- Load best hyperparameters (already tuned - not re-searched) ---
with open(f"{OUT}/1yr_best_params.json") as f:
    p_1yr = json.load(f)
with open(f"{OUT}/5yr_best_params.json") as f:
    p_5yr = json.load(f)
with open(f"{OUT}/esrd/esrd_best_params.json") as f:
    p_esrd = json.load(f)

serial_df = pd.read_excel(f"{OUT}/5yr_serial_model_results.xlsx", sheet_name="Best Hyperparameters")
p_serial = {}
for _, row in serial_df.iterrows():
    p_serial[row["Model"]] = {f"clf__{c}": row[c] for c in serial_df.columns
                               if c != "Model" and pd.notna(row[c])}

# --- Load already-published point estimates for cross-check ---
def load_published(path, sheet, bc_col):
    df = pd.read_excel(path, sheet_name=sheet)
    return {row["Model"]: row[bc_col] for _, row in df.iterrows()}

published = {
    "1-Year Flare":  load_published(f"{OUT}/1yr_model_results.xlsx", "Bootstrap (Harrell)", "Bias-Corrected AUROC (Harrell)"),
    "5-Year Flare":  load_published(f"{OUT}/5yr_model_results.xlsx", "Bootstrap (Harrell)", "Bias-Corrected AUROC (Harrell)"),
    "Serial Biopsy": load_published(f"{OUT}/5yr_serial_model_results.xlsx", "Bootstrap (Harrell)", "Bias-Corrected AUROC (Harrell)"),
    "ESRD 5-Year":   load_published(f"{OUT}/esrd/esrd_model_results.xlsx", "5yr Bootstrap", "BC AUROC"),
    "ESRD 10-Year":  load_published(f"{OUT}/esrd/esrd_model_results.xlsx", "10yr Bootstrap", "BC AUROC"),
}

# --- Cohort definitions: (label, data_path, outcome_col, best_params, scale_pos_weight) ---
df1 = pd.read_excel(f"{PROC}/lupus_1yr_selected_clean.xlsx")
df5 = pd.read_excel(f"{PROC}/lupus_5yr_selected_clean.xlsx")  # NOT dropping dsDNA - matches original script exactly
dfser = pd.read_excel(f"{PROC}/lupus_5yr_serial_selected.xlsx")
dfe5 = pd.read_excel(f"{PROC}/esrd_5yr_selected.xlsx")
dfe10 = pd.read_excel(f"{PROC}/esrd_10yr_selected.xlsx")

COHORTS = [
    ("1-Year Flare", df1, "flare_1yr", p_1yr, 3.34),
    ("5-Year Flare", df5, "flare_5yr", p_5yr, round(190 / 166, 2)),
    ("Serial Biopsy", dfser, "flare_5yr", p_serial, round(36 / 34, 2)),
    ("ESRD 5-Year", dfe5, "esrd_5yr", p_esrd["5yr"], None),   # spw computed dynamically below
    ("ESRD 10-Year", dfe10, "esrd_10yr", p_esrd["10yr"], None),
]

import os
import pickle
CKPT = f"{OUT}/bc_auroc_ci_checkpoint.pkl"
checkpoint = {}
if os.path.exists(CKPT):
    with open(CKPT, "rb") as f:
        checkpoint = pickle.load(f)
    print(f"Resuming from checkpoint - {len(checkpoint)} combos already done.")

results = list(checkpoint.values())
for label, df, outcome_col, best_params, spw in COHORTS:
    X = df.drop(columns=[outcome_col])
    y = df[outcome_col].astype(int)
    if spw is None:
        spw = round((y == 0).sum() / (y == 1).sum(), 2)

    print(f"\n{'='*70}\n{label}  (n={len(y)}, events={int(y.sum())}, spw={spw})\n{'='*70}")
    models = build_models(best_params, spw)

    for name, pipeline in models.items():
        key = (label, name)
        if key in checkpoint:
            r = checkpoint[key]
            print(f"  {name:<20} [cached] BC={r['BC_AUROC']:.3f} [{r['BC_AUROC_CI_lower']:.3f}-{r['BC_AUROC_CI_upper']:.3f}]")
            continue
        r = harrell_bootstrap_with_ci(pipeline, X, y, n_boot=1000, seed=42)
        pub = published[label].get(name)
        matches = (pub is not None) and (abs(r["bc_auroc"] - pub) < 0.0005)
        print(f"  {name:<20} BC={r['bc_auroc']:.3f} [{r['bc_ci_lower']:.3f}-{r['bc_ci_upper']:.3f}]  "
              f"published={pub}  match={matches}")
        row = {
            "Cohort": label, "Model": name,
            "Apparent_AUROC": r["apparent_auroc"],
            "Optimism_AUROC": r["optimism_auroc"],
            "BC_AUROC": r["bc_auroc"],
            "BC_AUROC_CI_lower": r["bc_ci_lower"],
            "BC_AUROC_CI_upper": r["bc_ci_upper"],
            "Published_BC_AUROC": pub,
            "Matches_Published": matches,
        }
        results.append(row)
        checkpoint[key] = row
        with open(CKPT, "wb") as f:
            pickle.dump(checkpoint, f)

out_df = pd.DataFrame(results)
out_df.to_excel(f"{OUT}/bc_auroc_ci.xlsx", index=False)
print(f"\nSaved: {OUT}/bc_auroc_ci.xlsx")

n_mismatch = (~out_df["Matches_Published"]).sum()
if n_mismatch:
    print(f"\n*** WARNING: {n_mismatch} model/cohort combos did NOT reproduce the published BC AUROC exactly. ***")
    print(out_df[~out_df["Matches_Published"]].to_string(index=False))
else:
    print("\nAll 20 model/cohort combinations reproduced the published BC AUROC exactly.")
