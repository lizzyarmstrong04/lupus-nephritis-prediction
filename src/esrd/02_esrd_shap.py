"""
SHAP analysis for ESRD models (5-year and 10-year).
Loads from feature-selected datasets; uses best params from 01_esrd_modelling.py.
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import shap
import warnings
import json
import os
warnings.filterwarnings("ignore")

from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier

PROCESSED_DIR = "/Users/elizabetharmstrong/Library/CloudStorage/OneDrive-ImperialCollegeLondon/Lupus Project/Data/Processed"
OUT_DIR       = "/Users/elizabetharmstrong/Library/CloudStorage/OneDrive-ImperialCollegeLondon/Lupus Project/outputs/esrd"

MODEL_COLORS = {
    "Logistic Regression": "#2a78d6",
    "Random Forest":       "#1baf7a",
    "XGBoost":             "#eda100",
    "LightGBM":            "#008300",
}

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

    shap_dir = f"{OUT_DIR}/figures/shap/{horizon_key}"
    os.makedirs(shap_dir, exist_ok=True)

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
        safe   = name.replace(" ", "_").lower()
        color  = MODEL_COLORS[name]

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

        # Beeswarm
        fig, ax = plt.subplots(figsize=(9, max(5, len(feat_names) * 0.45)))
        shap.plots.beeswarm(sv, max_display=len(feat_names), show=False,
                            plot_size=None, color_bar=True)
        plt.title(f"SHAP Beeswarm — {name}\nESRD {horizon_label}",
                  fontsize=12, pad=10)
        plt.tight_layout()
        fig.savefig(f"{shap_dir}/{safe}_beeswarm.png", dpi=180,
                    bbox_inches="tight")
        plt.close("all")

        # Bar chart
        fig, ax = plt.subplots(figsize=(8, max(4, len(feat_names) * 0.4)))
        order = np.argsort(mean_abs)
        ax.barh([feat_names[i] for i in order], mean_abs[order],
                color=color, edgecolor="white", height=0.7)
        ax.set_xlabel("Mean |SHAP value|", fontsize=11)
        ax.set_title(f"Feature Importance — {name}\nESRD {horizon_label}",
                     fontsize=12)
        ax.spines[["top", "right"]].set_visible(False)
        plt.tight_layout()
        fig.savefig(f"{shap_dir}/{safe}_bar.png", dpi=180, bbox_inches="tight")
        plt.close("all")

        print("done")

    # Cross-model summary table
    shap_df = pd.DataFrame(shap_table).round(4)
    shap_df.index.name = "Feature"
    shap_df["Mean across models"] = shap_df.mean(axis=1).round(4)
    shap_df = shap_df.sort_values("Mean across models", ascending=False)

    print(f"\n  Mean |SHAP| ({horizon_label}):")
    print(shap_df.to_string())

    shap_df.to_excel(f"{OUT_DIR}/esrd_shap_table_{horizon_key}.xlsx")

    # All-model grouped bar
    n_feat   = len(feat_names)
    n_models = len(models)
    x        = np.arange(n_feat)
    width    = 0.18

    fig, ax = plt.subplots(figsize=(max(10, n_feat * 0.9), 6))
    for i, mname in enumerate(models):
        vals_plot = [shap_table[mname].get(f, 0) for f in shap_df.index]
        ax.bar(x + i * width, vals_plot, width, label=mname,
               color=MODEL_COLORS[mname], edgecolor="white")
    ax.set_xticks(x + width * (n_models - 1) / 2)
    ax.set_xticklabels(shap_df.index, rotation=35, ha="right", fontsize=8)
    ax.set_ylabel("Mean |SHAP value|", fontsize=11)
    ax.set_title(f"Feature Importance — All Models, ESRD {horizon_label}",
                 fontsize=12)
    ax.legend(fontsize=9)
    ax.spines[["top", "right"]].set_visible(False)
    plt.tight_layout()
    fig.savefig(f"{shap_dir}/all_models_comparison.png", dpi=180,
                bbox_inches="tight")
    plt.close("all")
    print(f"  Saved: {shap_dir}/")


# Run for both horizons
run_shap(f"{PROCESSED_DIR}/esrd_5yr_selected.xlsx",  "esrd_5yr",  "5yr",  "5-Year")
run_shap(f"{PROCESSED_DIR}/esrd_10yr_selected.xlsx", "esrd_10yr", "10yr", "10-Year")

print("\nDone.")
