import pandas as pd
import numpy as np
import json
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import shap
import os
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
SHAP_DIR      = f"{OUTPUTS_DIR}/figures/shap/5yr"
OUTCOME_COL   = "flare_5yr"
SCALE_POS_WEIGHT = round(190 / 166, 2)

os.makedirs(SHAP_DIR, exist_ok=True)

# ============================================================
# 1. Load data and best hyperparameters
# ============================================================
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

# ============================================================
# 2. Rebuild tuned models
# ============================================================
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

MODEL_COLORS = {
    "Logistic Regression": "#1f77b4",
    "Random Forest":       "#ff7f0e",
    "XGBoost":             "#2ca02c",
    "LightGBM":            "#d62728",
}

# ============================================================
# 3. Compute SHAP values
# ============================================================
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

# ============================================================
# 4. Plots per model
# ============================================================
def get_patients_of_interest(pipeline, X_named):
    probs      = pipeline.predict_proba(X_named)[:, 1]
    idx_high   = int(np.argmax(probs))
    idx_low    = int(np.argmin(probs))
    idx_border = int(np.argmin(np.abs(probs - 0.5)))
    return {
        "highest_risk": (idx_high,   probs[idx_high]),
        "lowest_risk":  (idx_low,    probs[idx_low]),
        "boundary":     (idx_border, probs[idx_border]),
    }

mean_shap_table = {}

for name, data in shap_values_dict.items():
    sv    = data["shap_vals"]
    X_tr  = data["X_tr"]
    safe  = name.replace(" ", "_").lower()
    color = MODEL_COLORS[name]
    print(f"\nPlotting {name}...")

    mean_abs = np.abs(sv.values).mean(axis=0)
    mean_shap_table[name] = dict(zip(short_features, mean_abs.round(4)))

    # Beeswarm
    fig, ax = plt.subplots(figsize=(9, 6))
    shap.plots.beeswarm(sv, max_display=len(short_features), show=False,
                        plot_size=None, color_bar=True)
    plt.title(f"SHAP Beeswarm — {name} (5-year)", fontsize=13, pad=12)
    plt.tight_layout()
    fig.savefig(f"{SHAP_DIR}/{safe}_beeswarm.png", dpi=180, bbox_inches="tight")
    plt.close("all")
    print(f"  Saved: {safe}_beeswarm.png")

    # Bar
    fig, ax = plt.subplots(figsize=(8, 5))
    order = np.argsort(mean_abs)
    ax.barh([short_features[i] for i in order], mean_abs[order],
            color=color, edgecolor="white", height=0.7)
    ax.set_xlabel("Mean |SHAP value|", fontsize=11)
    ax.set_title(f"Feature Importance — {name} (5-year)", fontsize=13)
    ax.spines[["top", "right"]].set_visible(False)
    plt.tight_layout()
    fig.savefig(f"{SHAP_DIR}/{safe}_bar.png", dpi=180, bbox_inches="tight")
    plt.close("all")
    print(f"  Saved: {safe}_bar.png")

    # Waterfall plots
    pipeline   = MODELS[name]
    patients   = get_patients_of_interest(pipeline, X_named)
    labels     = {
        "highest_risk": "Highest Risk Patient",
        "lowest_risk":  "Lowest Risk Patient",
        "boundary":     "Decision Boundary Patient",
    }
    for patient_key, (idx, prob) in patients.items():
        fig, ax = plt.subplots(figsize=(9, 5))
        shap.plots.waterfall(sv[idx], max_display=len(short_features), show=False)
        plt.title(
            f"{name} (5-yr) — {labels[patient_key]}\n"
            f"(Patient #{idx}, Predicted prob = {prob:.3f}, Actual = {int(y.iloc[idx])})",
            fontsize=11, pad=10)
        plt.tight_layout()
        fname = f"{safe}_waterfall_{patient_key}.png"
        fig.savefig(f"{SHAP_DIR}/{fname}", dpi=180, bbox_inches="tight")
        plt.close("all")
        print(f"  Saved: {fname}")

# ============================================================
# 5. Cross-model comparison table
# ============================================================
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

# Combined bar chart
fig, ax = plt.subplots(figsize=(11, 6))
n_feat   = len(short_features)
n_models = len(MODELS)
x        = np.arange(n_feat)
width    = 0.18

for i, mname in enumerate(MODELS):
    vals = [mean_shap_table[mname].get(f, 0) for f in shap_df.index]
    ax.bar(x + i * width, vals, width, label=mname,
           color=MODEL_COLORS[mname], edgecolor="white")

ax.set_xticks(x + width * (n_models - 1) / 2)
ax.set_xticklabels(shap_df.index, rotation=35, ha="right", fontsize=9)
ax.set_ylabel("Mean |SHAP value|", fontsize=11)
ax.set_title("Feature Importance — All Models (5-year, Mean |SHAP|)", fontsize=13)
ax.legend(fontsize=9)
ax.spines[["top", "right"]].set_visible(False)
plt.tight_layout()
fig.savefig(f"{SHAP_DIR}/all_models_shap_comparison_5yr.png", dpi=180, bbox_inches="tight")
plt.close("all")
print("Saved: all_models_shap_comparison_5yr.png")
print("\nDone.")
