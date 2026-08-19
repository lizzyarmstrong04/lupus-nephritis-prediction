"""
Train and save all models for the risk calculator app.
Run once: python src/app/00_save_models.py
Output: src/app/models/*.joblib
"""

import pandas as pd
import numpy as np
import joblib
import json
import os
import warnings
warnings.filterwarnings("ignore")

from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier

BASE      = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
PROC_DIR  = os.path.join(BASE, "Data", "Processed")
OUT_DIR   = os.path.join(BASE, "outputs")
MODEL_DIR = os.path.join(BASE, "src", "app", "models")
os.makedirs(MODEL_DIR, exist_ok=True)

# Load best hyperparameters
with open(os.path.join(OUT_DIR, "1yr_best_params.json"))  as f: p1  = json.load(f)
with open(os.path.join(OUT_DIR, "5yr_best_params.json"))  as f: p5  = json.load(f)
with open(os.path.join(OUT_DIR, "esrd", "esrd_best_params.json")) as f: pe = json.load(f)

ANALYSES = {
    "1yr":      {"data": os.path.join(PROC_DIR, "lupus_1yr_selected_clean.xlsx"), "outcome": "flare_1yr",  "params": p1},
    "5yr":      {"data": os.path.join(PROC_DIR, "lupus_5yr_selected_clean.xlsx"), "outcome": "flare_5yr",  "params": p5},
    "esrd_5yr": {"data": os.path.join(PROC_DIR, "esrd_5yr_selected.xlsx"),        "outcome": "esrd_5yr",   "params": pe["5yr"]},
    "esrd_10yr":{"data": os.path.join(PROC_DIR, "esrd_10yr_selected.xlsx"),       "outcome": "esrd_10yr",  "params": pe["10yr"]},
}

CLF_NAME_MAP = {"lr": "Logistic Regression", "rf": "Random Forest",
                "xgb": "XGBoost", "lgbm": "LightGBM"}

def build_pipeline(clf_key, params, spw):
    p = params.get(CLF_NAME_MAP[clf_key], {})
    g = lambda k, d: p.get(k, d)

    if clf_key == "lr":
        return Pipeline([("s", StandardScaler()),
            ("clf", LogisticRegression(max_iter=1000, C=1.0,
                                       class_weight="balanced", random_state=42))])
    elif clf_key == "rf":
        return Pipeline([("s", StandardScaler()),
            ("clf", RandomForestClassifier(
                n_estimators=g("clf__n_estimators", 300),
                max_depth=g("clf__max_depth", 3),
                min_samples_leaf=g("clf__min_samples_leaf", 10),
                max_features=g("clf__max_features", "sqrt"),
                class_weight="balanced", random_state=42))])
    elif clf_key == "xgb":
        return Pipeline([("s", StandardScaler()),
            ("clf", XGBClassifier(
                n_estimators=g("clf__n_estimators", 100),
                max_depth=g("clf__max_depth", 2),
                learning_rate=g("clf__learning_rate", 0.01),
                subsample=g("clf__subsample", 0.5),
                colsample_bytree=g("clf__colsample_bytree", 0.6),
                min_child_weight=g("clf__min_child_weight", 5),
                reg_alpha=g("clf__reg_alpha", 1.0),
                reg_lambda=g("clf__reg_lambda", 5.0),
                scale_pos_weight=spw,
                eval_metric="logloss", verbosity=0, random_state=42))])
    elif clf_key == "lgbm":
        return Pipeline([("s", StandardScaler()),
            ("clf", LGBMClassifier(
                n_estimators=g("clf__n_estimators", 100),
                max_depth=g("clf__max_depth", 3),
                learning_rate=g("clf__learning_rate", 0.01),
                subsample=g("clf__subsample", 0.7),
                num_leaves=g("clf__num_leaves", 15),
                min_child_samples=g("clf__min_child_samples", 10),
                class_weight="balanced", verbose=-1, random_state=42))])

feature_cols = {}

for key, cfg in ANALYSES.items():
    df  = pd.read_excel(cfg["data"])
    X   = df.drop(columns=[cfg["outcome"]])
    y   = df[cfg["outcome"]].astype(int)
    spw = round((y == 0).sum() / (y == 1).sum(), 2)

    feature_cols[key] = list(X.columns)
    print(f"\n{key}  (n={len(y)}, events={int(y.sum())}, spw={spw})")
    print(f"  Features: {list(X.columns)}")

    for clf_key in ["lr", "rf", "xgb", "lgbm"]:
        pipeline = build_pipeline(clf_key, cfg["params"], spw)
        pipeline.fit(X, y)
        out = os.path.join(MODEL_DIR, f"{key}_{clf_key}.joblib")
        joblib.dump(pipeline, out, compress=3)
        p = pipeline.predict_proba(X)[:, 1]
        print(f"  Saved {clf_key}: {out.split('/')[-1]}  "
              f"(apparent AUROC training check: n/a, n_pos={int(y.sum())})")

# Save feature column names for the app to use
joblib.dump(feature_cols, os.path.join(MODEL_DIR, "feature_cols.joblib"))
print(f"\nSaved feature_cols.joblib")
print("\nAll models saved.")
