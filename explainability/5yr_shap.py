import pandas as pd
import numpy as np
import json
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
OUTCOME_COL   = "flare_5yr"
SCALE_POS_WEIGHT = round(190 / 166, 2)

# 1. Load data and best hyperparameters

df = pd.read_excel(f"{PROCESSED_DIR}/lupus_5yr_selected_clean.xlsx")
X  = df.drop(columns=[OUTCOME_COL])
y  = df[OUTCOME_COL].astype(int)

with open(f"{OUTPUTS_DIR}/5yr_best_params.json") as f:
    best_params = json.load(f)

# Auto-shorten long feature names for readable plots
def shorten(name, maxlen=35):
    return name if len(name) <= maxlen else name[:maxlen].rstrip() + "…"

feature_names = list(X.columns)
short_names   = {c: shorten(c) for c in feature_names}
X_named       = X.rename(columns=short_names)
short_features = list(X_named.columns)

print(f"Dataset: {X.shape[0]} rows, {X.shape[1]} predictors")
print(f"Outcome events: {y.sum()} / {len(y)} ({y.mean()*100:.1f}%)")
print(f"Features: {short_features}")

# 2. Rebuild tuned models

def make_rf(p):
    return RandomForestClassifier(
        n_estimators=p.get("clf__n_estimators", 300),
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
        min_child_weight=p.get("clf__min_child_weight", 10),
        reg_alpha=p.get("clf__reg_alpha", 1.0),
        reg_lambda=p.get("clf__reg_lambda", 5.0),
        eval_metric="logloss", verbosity=0, random_state=42,
        scale_pos_weight=SCALE_POS_WEIGHT)

def make_lgbm(p):
    return LGBMClassifier(
        n_estimators=p.get("clf__n_estimators", 200),
        max_depth=p.get("clf__max_depth", 3),
        learning_rate=p.get("clf__learning_rate", 0.05),
        subsample=p.get("clf__subsample", 0.8),
        num_leaves=p.get("clf__num_leaves", 31),
        min_child_samples=p.get("clf__min_child_samples", 20),
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

# 3. Compute SHAP values

shap_values_dict = {}
scaler_fit = StandardScaler().fit(X_named)

print("\nComputing SHAP values...")
for name, pipeline in MODELS.items():
    print(f"  {name}...", end=" ", flush=True)
    pipeline.fit(X_named, y)
    clf   = pipeline.named_steps["clf"]
    scaler = pipeline.named_steps["scaler"]
    X_tr  = pd.DataFrame(scaler.transform(X_named), columns=short_features)

    if name == "Logistic Regression":
        explainer = shap.LinearExplainer(clf, X_tr, feature_perturbation="interventional")
        shap_vals = explainer(X_tr)
    else:
        explainer = shap.TreeExplainer(clf, feature_perturbation="tree_path_dependent")
        shap_vals = explainer(X_tr)
        vals = shap_vals.values
        base = shap_vals.base_values
        if isinstance(vals, list):
            vals = vals[1]
            base = base[1] if hasattr(base, "__len__") and len(np.shape(base)) > 0 else base
        elif vals.ndim == 3:
            vals = vals[:, :, 1]
            base = base[:, 1] if base.ndim == 2 else base
        shap_vals = shap.Explanation(
            values=vals, base_values=base,
            data=shap_vals.data, feature_names=short_features)

    shap_values_dict[name] = {"shap_vals": shap_vals, "X_tr": X_tr}
    print("done")

# 4. Mean absolute SHAP value per feature, per model

mean_shap_table = {}

for name, data in shap_values_dict.items():
    sv = data["shap_vals"]
    mean_abs = np.abs(sv.values).mean(axis=0)
    mean_shap_table[name] = dict(zip(short_features, mean_abs.round(4)))

# 5. Cross-model comparison table

print("\n" + "="*80)
print("MEAN ABSOLUTE SHAP VALUES PER FEATURE PER MODEL (5-year)")
print("="*80)

shap_df = pd.DataFrame(mean_shap_table).round(4)
shap_df.index.name = "Feature"
shap_df["Mean across models"] = shap_df.mean(axis=1).round(4)
shap_df = shap_df.sort_values("Mean across models", ascending=False)
print(shap_df.to_string())

shap_df.to_excel(f"{OUTPUTS_DIR}/shap_importance_table_5yr.xlsx")
print("\nSaved: shap_importance_table_5yr.xlsx")
print("\nDone.")
