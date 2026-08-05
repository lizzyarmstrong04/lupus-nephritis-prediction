import pandas as pd
import numpy as np
import json
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import shap
import warnings
warnings.filterwarnings("ignore")

from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier

PROCESSED_DIR = "/Users/elizabetharmstrong/Library/CloudStorage/OneDrive-ImperialCollegeLondon/Lupus Project/Data/Processed"
OUTPUTS_DIR   = "/Users/elizabetharmstrong/Library/CloudStorage/OneDrive-ImperialCollegeLondon/Lupus Project/outputs"
OUT_PATH      = f"{OUTPUTS_DIR}/figures/shap_1yr_vs_5yr_comparison.png"

SHORT_NAMES_1YR = {
    "% chronic gloms(%of total)":                                                         "% chronic gloms",
    "%gloms with necrosis":                                                                "% necrosis",
    "Age at biopsy":                                                                       "Age at biopsy",
    "Proteinuria at biopsy (uPCR, log)":                                                   "Proteinuria (log)",
    "Class coded 1=I 2=II 3=III 4=IV 5=V 6=III+V 7=IV+V 8=II+V 9=VI 10=other":          "LN class",
    "Ethnicity 1=white 2=black 3=asian (south) 4=asian (east) 5=other 6=not stated/unknown/any other mixed": "Ethnicity",
    "% active gloms (%of those not globally sclerosed)":                                   "% active gloms",
    "%gloms with crescents":                                                               "% crescents",
    "C4 at biopsy":                                                                        "C4 at biopsy",
}

SHORT_NAMES_5YR = {
    "% chronic gloms(%of total)":                                                                                                                                    "% chronic gloms",
    "%gloms with necrosis":                                                                                                                                          "% necrosis",
    "Class coded 1=I 2=II 3=III 4=IV 5=V 6=III+V 7=IV+V 8=II+V 9=VI 10=other":                                                                                    "LN class",
    "% active gloms (%of those not globally sclerosed)":                                                                                                             "% active gloms",
    "Prev exposure to cyclo (for Rx comparison) - related to the 'use this biopsy for this patient' biopsy":                                                         "Prev cyclophosphamide",
    "% sclerosed gloms":                                                                                                                                             "% sclerosed gloms",
    "CKD epi formula without ethnicity":                                                                                                                             "eGFR (CKD-EPI)",
    "Reason for biopsy 1=new pres LN 2=relapse 3=non-response/partial response, incl on-going proteinuria 4=pre-pregnancy or Ax if drug switch/stop appropriate":   "Biopsy reason",
    "Ethnicity 1=white 2=black 3=asian (south) 4=asian (east) 5=other 6=not stated/unknown/any other mixed":                                                         "Ethnicity",
    "Age at biopsy":                                                                                                                                                 "Age at biopsy",
}

# ============================================================
# Load data
# ============================================================
df1 = pd.read_excel(f"{PROCESSED_DIR}/lupus_1yr_selected_clean.xlsx")
X1  = df1.drop(columns=["flare_1yr"])
y1  = df1["flare_1yr"].astype(int)
X1  = X1.rename(columns=SHORT_NAMES_1YR)
feat1 = list(X1.columns)

df5 = pd.read_excel(f"{PROCESSED_DIR}/lupus_5yr_selected_clean.xlsx")
X5  = df5.drop(columns=["flare_5yr"])
y5  = df5["flare_5yr"].astype(int)
X5  = X5.rename(columns=SHORT_NAMES_5YR)
feat5 = list(X5.columns)

with open(f"{OUTPUTS_DIR}/5yr_best_params.json") as f:
    bp5 = json.load(f)

SCALE_POS_WEIGHT = round(190 / 166, 2)

# ============================================================
# Build models
# ============================================================
def models_1yr():
    return {
        "Logistic Regression": Pipeline([("scaler", StandardScaler()),
            ("clf", LogisticRegression(max_iter=1000, random_state=42, C=1.0))]),
        "Random Forest": Pipeline([("scaler", StandardScaler()),
            ("clf", RandomForestClassifier(n_estimators=300, max_depth=3,
                min_samples_leaf=10, max_features="sqrt",
                class_weight="balanced", random_state=42))]),
        "XGBoost": Pipeline([("scaler", StandardScaler()),
            ("clf", XGBClassifier(n_estimators=100, max_depth=3, learning_rate=0.01,
                subsample=0.5, colsample_bytree=0.6, min_child_weight=5,
                reg_alpha=2.0, reg_lambda=5.0,
                eval_metric="logloss", verbosity=0, random_state=42))]),
        "LightGBM": Pipeline([("scaler", StandardScaler()),
            ("clf", LGBMClassifier(n_estimators=100, max_depth=2, learning_rate=0.01,
                subsample=0.7, num_leaves=15, min_child_samples=30,
                verbose=-1, random_state=42))]),
    }

def models_5yr(p):
    return {
        "Logistic Regression": Pipeline([("scaler", StandardScaler()),
            ("clf", LogisticRegression(max_iter=1000, random_state=42, C=1.0,
                class_weight="balanced"))]),
        "Random Forest": Pipeline([("scaler", StandardScaler()),
            ("clf", RandomForestClassifier(
                n_estimators=p["Random Forest"].get("clf__n_estimators", 300),
                max_depth=p["Random Forest"].get("clf__max_depth", 3),
                min_samples_leaf=p["Random Forest"].get("clf__min_samples_leaf", 10),
                max_features=p["Random Forest"].get("clf__max_features", "sqrt"),
                class_weight="balanced", random_state=42))]),
        "XGBoost": Pipeline([("scaler", StandardScaler()),
            ("clf", XGBClassifier(
                n_estimators=p["XGBoost"].get("clf__n_estimators", 100),
                max_depth=p["XGBoost"].get("clf__max_depth", 2),
                learning_rate=p["XGBoost"].get("clf__learning_rate", 0.01),
                subsample=p["XGBoost"].get("clf__subsample", 0.5),
                colsample_bytree=p["XGBoost"].get("clf__colsample_bytree", 0.5),
                min_child_weight=p["XGBoost"].get("clf__min_child_weight", 5),
                reg_alpha=p["XGBoost"].get("clf__reg_alpha", 1.0),
                reg_lambda=p["XGBoost"].get("clf__reg_lambda", 5.0),
                eval_metric="logloss", verbosity=0, random_state=42,
                scale_pos_weight=SCALE_POS_WEIGHT))]),
        "LightGBM": Pipeline([("scaler", StandardScaler()),
            ("clf", LGBMClassifier(
                n_estimators=p["LightGBM"].get("clf__n_estimators", 100),
                max_depth=p["LightGBM"].get("clf__max_depth", 3),
                learning_rate=p["LightGBM"].get("clf__learning_rate", 0.01),
                subsample=p["LightGBM"].get("clf__subsample", 0.7),
                num_leaves=p["LightGBM"].get("clf__num_leaves", 15),
                min_child_samples=p["LightGBM"].get("clf__min_child_samples", 10),
                verbose=-1, random_state=42, class_weight="balanced"))]),
    }

# ============================================================
# Compute mean |SHAP| for all models, both horizons
# ============================================================
def get_mean_abs_shap(models_dict, X, y, features):
    results = {}
    for name, pipeline in models_dict.items():
        print(f"  {name}...", end=" ", flush=True)
        pipeline.fit(X, y)
        clf    = pipeline.named_steps["clf"]
        scaler = pipeline.named_steps["scaler"]
        X_tr   = pd.DataFrame(scaler.transform(X), columns=features)

        if name == "Logistic Regression":
            explainer = shap.LinearExplainer(clf, X_tr, feature_perturbation="interventional")
            sv = explainer(X_tr)
            vals = sv.values
        else:
            explainer = shap.TreeExplainer(clf, feature_perturbation="tree_path_dependent")
            sv = explainer(X_tr)
            vals = sv.values
            if isinstance(vals, list):
                vals = vals[1]
            elif vals.ndim == 3:
                vals = vals[:, :, 1]

        mean_abs = np.abs(vals).mean(axis=0)
        results[name] = dict(zip(features, mean_abs))
        print("done")
    return results

print("Computing 1-year SHAP values...")
shap_1yr = get_mean_abs_shap(models_1yr(), X1, y1, feat1)

print("Computing 5-year SHAP values...")
shap_5yr = get_mean_abs_shap(models_5yr(bp5), X5, y5, feat5)

# ============================================================
# Plot: 2x2 panels, one per model
# ============================================================
MODEL_NAMES = ["Logistic Regression", "Random Forest", "XGBoost", "LightGBM"]
COLOR_1YR = "#4C72B0"
COLOR_5YR = "#DD8452"

fig, axes = plt.subplots(2, 2, figsize=(16, 14))
axes = axes.flatten()

for ax, model_name in zip(axes, MODEL_NAMES):
    d1 = shap_1yr[model_name]
    d5 = shap_5yr[model_name]

    # Union of features, ordered by mean |SHAP| averaged across both horizons
    all_feats = list(dict.fromkeys(list(d1.keys()) + list(d5.keys())))
    avg_importance = {f: (d1.get(f, 0) + d5.get(f, 0)) / 2 for f in all_feats}
    all_feats_sorted = sorted(all_feats, key=lambda f: avg_importance[f])

    vals1 = [d1.get(f, 0) for f in all_feats_sorted]
    vals5 = [d5.get(f, 0) for f in all_feats_sorted]

    y_pos = np.arange(len(all_feats_sorted))
    height = 0.35

    bars1 = ax.barh(y_pos - height/2, vals1, height, label="1-year", color=COLOR_1YR, alpha=0.85)
    bars5 = ax.barh(y_pos + height/2, vals5, height, label="5-year", color=COLOR_5YR, alpha=0.85)

    ax.set_yticks(y_pos)
    ax.set_yticklabels(all_feats_sorted, fontsize=9)
    ax.set_xlabel("Mean |SHAP value|", fontsize=10)
    ax.set_title(model_name, fontsize=12, fontweight="bold")
    ax.legend(fontsize=9)
    ax.spines[["top", "right"]].set_visible(False)

    # Mark features unique to one horizon with a subtle note
    for i, feat in enumerate(all_feats_sorted):
        only_1 = feat in d1 and feat not in d5
        only_5 = feat not in d1 and feat in d5
        if only_1:
            ax.get_yticklabels()[i].set_color("#4C72B0")
        elif only_5:
            ax.get_yticklabels()[i].set_color("#DD8452")

fig.suptitle(
    "SHAP Feature Importance: 1-Year vs 5-Year Flare Prediction\n"
    "(blue tick = feature only in 1-yr model, orange tick = feature only in 5-yr model)",
    fontsize=13, y=1.01
)
plt.tight_layout()
fig.savefig(OUT_PATH, dpi=180, bbox_inches="tight")
plt.close("all")
print(f"\nSaved: {OUT_PATH}")
