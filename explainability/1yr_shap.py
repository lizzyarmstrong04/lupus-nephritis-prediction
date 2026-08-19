import pandas as pd
import numpy as np
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
OUTCOME_COL   = "flare_1yr"

# 1. Load data and rebuild tuned models

df = pd.read_excel(f"{PROCESSED_DIR}/lupus_1yr_selected_clean.xlsx")
X  = df.drop(columns=[OUTCOME_COL])
y  = df[OUTCOME_COL].astype(int)

# Short feature names for readable plots
SHORT_NAMES = {
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
X_named = X.rename(columns=SHORT_NAMES)
feature_names = list(X_named.columns)

# Best hyperparameters from tuning run
MODELS = {
    "Logistic Regression": Pipeline([
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(max_iter=1000, random_state=42, C=1.0)),
    ]),
    "Random Forest": Pipeline([
        ("scaler", StandardScaler()),
        ("clf", RandomForestClassifier(n_estimators=300, max_depth=3,
                                       min_samples_leaf=10, max_features="sqrt",
                                       class_weight="balanced", random_state=42)),
    ]),
    "XGBoost": Pipeline([
        ("scaler", StandardScaler()),
        ("clf", XGBClassifier(n_estimators=100, max_depth=3, learning_rate=0.01,
                              subsample=0.5, colsample_bytree=0.6,
                              min_child_weight=5, reg_alpha=2.0, reg_lambda=5.0,
                              eval_metric="logloss", verbosity=0, random_state=42)),
    ]),
    "LightGBM": Pipeline([
        ("scaler", StandardScaler()),
        ("clf", LGBMClassifier(n_estimators=100, max_depth=2, learning_rate=0.01,
                               subsample=0.7, num_leaves=15, min_child_samples=30,
                               verbose=-1, random_state=42)),
    ]),
}

# 2. Fit all models and compute SHAP values

shap_values_dict = {}
scaler_fit       = StandardScaler().fit(X_named)
X_scaled         = pd.DataFrame(scaler_fit.transform(X_named), columns=feature_names)

print("Computing SHAP values...")
for name, pipeline in MODELS.items():
    print(f"  {name}...", end=" ", flush=True)
    pipeline.fit(X_named, y)
    clf    = pipeline.named_steps["clf"]
    scaler = pipeline.named_steps["scaler"]
    X_tr   = pd.DataFrame(scaler.transform(X_named), columns=feature_names)

    if name == "Logistic Regression":
        explainer = shap.LinearExplainer(clf, X_tr, feature_perturbation="interventional")
        shap_vals = explainer(X_tr)
    else:
        explainer = shap.TreeExplainer(clf, feature_perturbation="tree_path_dependent")
        shap_vals = explainer(X_tr)

        # TreeExplainer may return 3D array (n_samples, n_features, n_classes)
        # or a list of arrays — always extract class-1 slice
        vals = shap_vals.values
        base = shap_vals.base_values

        if isinstance(vals, list):
            vals = vals[1]
            base = base[1] if hasattr(base, "__len__") and len(np.shape(base)) > 0 else base
        elif vals.ndim == 3:
            vals = vals[:, :, 1]
            base = base[:, 1] if base.ndim == 2 else base

        shap_vals = shap.Explanation(
            values=vals,
            base_values=base,
            data=shap_vals.data,
            feature_names=feature_names,
        )

    shap_values_dict[name] = {"explainer": explainer, "shap_vals": shap_vals, "X_tr": X_tr}
    print("done")

# 3. Mean absolute SHAP value per feature, per model

mean_shap_table = {}

for name, data in shap_values_dict.items():
    sv = data["shap_vals"]
    mean_abs = np.abs(sv.values).mean(axis=0)
    mean_shap_table[name] = dict(zip(feature_names, mean_abs.round(4)))

# 4. Cross-model mean |SHAP| comparison table

print("\n" + "="*80)
print("MEAN ABSOLUTE SHAP VALUES PER FEATURE PER MODEL")
print("="*80)

shap_df = pd.DataFrame(mean_shap_table).round(4)
shap_df.index.name = "Feature"
shap_df["Mean across models"] = shap_df.mean(axis=1).round(4)
shap_df = shap_df.sort_values("Mean across models", ascending=False)

print(shap_df.to_string())

# Save comparison table
shap_df.to_excel(
    "/Users/elizabetharmstrong/Library/CloudStorage/OneDrive-ImperialCollegeLondon/Lupus Project/outputs/shap_importance_table.xlsx"
)
print("\nSaved: shap_importance_table.xlsx")
print("\nDone.")
