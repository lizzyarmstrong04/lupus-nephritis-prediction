"""
SHAP analysis for ESRD models (5-year and 10-year).
Loads from feature-selected datasets; uses best params from 01_esrd_modelling.py.
"""

import pandas as pd
import numpy as np
import shap
import warnings
import json
warnings.filterwarnings("ignore")

from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier

PROCESSED_DIR = "/Users/elizabetharmstrong/Library/CloudStorage/OneDrive-ImperialCollegeLondon/Lupus Project/Data/Processed"
OUT_DIR       = "/Users/elizabetharmstrong/Library/CloudStorage/OneDrive-ImperialCollegeLondon/Lupus Project/outputs/esrd"

with open(f"{OUT_DIR}/esrd_best_params.json") as f:
    BEST_PARAMS = json.load(f)

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

def run_shap(data_path, outcome_col, horizon_key, horizon_label):
    df  = pd.read_excel(data_path)
    X   = df.drop(columns=[outcome_col])
    y   = df[outcome_col].astype(int)

    feat_names       = list(X.columns)
    scale_pos_weight = round((y == 0).sum() / (y == 1).sum(), 2)

    models     = build_models(horizon_key, scale_pos_weight)
    shap_table = {}

    print(f"\n{'='*60}")
    print(f"SHAP — ESRD {horizon_label}  (n={len(y)}, events={int(y.sum())})")
    print(f"  Features ({len(feat_names)}): {feat_names}")

    for name, pipeline in models.items():
        print(f"  {name}...", end=" ", flush=True)
        pipeline.fit(X, y)
        clf    = pipeline.named_steps["clf"]
        scaler = pipeline.named_steps["s"]
        X_tr   = pd.DataFrame(scaler.transform(X), columns=feat_names)

        if name == "Logistic Regression":
            explainer = shap.LinearExplainer(clf, X_tr,
                                             feature_perturbation="interventional")
            sv   = explainer(X_tr)
            vals = sv.values
        else:
            explainer = shap.TreeExplainer(clf,
                                           feature_perturbation="tree_path_dependent")
            sv   = explainer(X_tr)
            vals = sv.values
            if isinstance(vals, list):
                vals = vals[1]
            elif vals.ndim == 3:
                vals = vals[:, :, 1]
            sv = shap.Explanation(values=vals,
                                  base_values=sv.base_values,
                                  data=sv.data,
                                  feature_names=feat_names)

        mean_abs = np.abs(vals).mean(axis=0)
        shap_table[name] = dict(zip(feat_names, mean_abs.round(4)))

        print("done")

    # Cross-model summary table
    shap_df = pd.DataFrame(shap_table).round(4)
    shap_df.index.name = "Feature"
    shap_df["Mean across models"] = shap_df.mean(axis=1).round(4)
    shap_df = shap_df.sort_values("Mean across models", ascending=False)

    print(f"\n  Mean |SHAP| ({horizon_label}):")
    print(shap_df.to_string())

    shap_df.to_excel(f"{OUT_DIR}/esrd_shap_table_{horizon_key}.xlsx")
    print(f"  Saved: esrd_shap_table_{horizon_key}.xlsx")

# Run for both horizons
run_shap(f"{PROCESSED_DIR}/esrd_5yr_selected.xlsx",  "esrd_5yr",  "5yr",  "5-Year")
run_shap(f"{PROCESSED_DIR}/esrd_10yr_selected.xlsx", "esrd_10yr", "10yr", "10-Year")

print("\nDone.")
