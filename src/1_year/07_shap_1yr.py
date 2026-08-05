import pandas as pd
import numpy as np
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
SHAP_DIR      = "/Users/elizabetharmstrong/Library/CloudStorage/OneDrive-ImperialCollegeLondon/Lupus Project/outputs/figures/shap"
OUTCOME_COL   = "flare_1yr"

# ============================================================
# 1. Load data and rebuild tuned models
# ============================================================
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

MODEL_COLORS = {
    "Logistic Regression": "#1f77b4",
    "Random Forest":       "#ff7f0e",
    "XGBoost":             "#2ca02c",
    "LightGBM":            "#d62728",
}

# ============================================================
# 2. Fit all models and compute SHAP values
# ============================================================
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

# ============================================================
# 3. Identify 3 patients of interest per model
# ============================================================
def get_patients_of_interest(pipeline, X_named):
    probs = pipeline.predict_proba(X_named)[:, 1]
    idx_high   = int(np.argmax(probs))
    idx_low    = int(np.argmin(probs))
    idx_border = int(np.argmin(np.abs(probs - 0.5)))
    return {
        "highest_risk":  (idx_high,   probs[idx_high]),
        "lowest_risk":   (idx_low,    probs[idx_low]),
        "boundary":      (idx_border, probs[idx_border]),
    }

# ============================================================
# 4. Generate plots per model
# ============================================================
mean_shap_table = {}

for name, data in shap_values_dict.items():
    sv     = data["shap_vals"]
    X_tr   = data["X_tr"]
    print(f"\nPlotting {name}...")
    safe   = name.replace(" ", "_").lower()
    color  = MODEL_COLORS[name]

    # Mean absolute SHAP per feature
    mean_abs = np.abs(sv.values).mean(axis=0)
    mean_shap_table[name] = dict(zip(feature_names, mean_abs.round(4)))

    # --- 4a. Beeswarm (summary) plot ---
    fig, ax = plt.subplots(figsize=(9, 6))
    shap.plots.beeswarm(sv, max_display=9, show=False, plot_size=None, color_bar=True)
    plt.title(f"SHAP Beeswarm — {name}", fontsize=13, pad=12)
    plt.tight_layout()
    fig.savefig(f"{SHAP_DIR}/{safe}_beeswarm.png", dpi=180, bbox_inches="tight")
    plt.close("all")
    print(f"  Saved: {safe}_beeswarm.png")

    # --- 4b. Bar plot (mean |SHAP|) ---
    fig, ax = plt.subplots(figsize=(8, 5))
    order   = np.argsort(mean_abs)
    bars    = ax.barh(
        [feature_names[i] for i in order],
        mean_abs[order],
        color=color, edgecolor="white", height=0.7
    )
    ax.set_xlabel("Mean |SHAP value|", fontsize=11)
    ax.set_title(f"Feature Importance — {name}", fontsize=13)
    ax.spines[["top", "right"]].set_visible(False)
    plt.tight_layout()
    fig.savefig(f"{SHAP_DIR}/{safe}_bar.png", dpi=180, bbox_inches="tight")
    plt.close("all")
    print(f"  Saved: {safe}_bar.png")

    # --- 4c. Waterfall plots for 3 patients ---
    pipeline   = MODELS[name]
    patients   = get_patients_of_interest(pipeline, X_named)
    labels     = {
        "highest_risk": "Highest Risk Patient",
        "lowest_risk":  "Lowest Risk Patient",
        "boundary":     "Decision Boundary Patient",
    }

    for patient_key, (idx, prob) in patients.items():
        fig, ax = plt.subplots(figsize=(9, 5))
        shap.plots.waterfall(sv[idx], max_display=9, show=False)
        plt.title(
            f"{name} — {labels[patient_key]}\n"
            f"(Patient #{idx}, Predicted probability = {prob:.3f}, "
            f"Actual outcome = {int(y.iloc[idx])})",
            fontsize=11, pad=10
        )
        plt.tight_layout()
        fname = f"{safe}_waterfall_{patient_key}.png"
        fig.savefig(f"{SHAP_DIR}/{fname}", dpi=180, bbox_inches="tight")
        plt.close("all")
        print(f"  Saved: {fname}")

# ============================================================
# 5. Cross-model mean |SHAP| comparison table
# ============================================================
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

# ============================================================
# 6. Combined bar chart — all models side by side
# ============================================================
fig, ax = plt.subplots(figsize=(11, 6))
n_feat  = len(feature_names)
n_models = len(MODELS)
x       = np.arange(n_feat)
width   = 0.18

for i, mname in enumerate(MODELS):
    vals  = [mean_shap_table[mname].get(f, 0) for f in shap_df.index]
    ax.bar(x + i * width, vals, width, label=mname,
           color=MODEL_COLORS[mname], edgecolor="white")

ax.set_xticks(x + width * (n_models - 1) / 2)
ax.set_xticklabels(shap_df.index, rotation=35, ha="right", fontsize=9)
ax.set_ylabel("Mean |SHAP value|", fontsize=11)
ax.set_title("Feature Importance Comparison — All Models (Mean |SHAP|)", fontsize=13)
ax.legend(fontsize=9)
ax.spines[["top", "right"]].set_visible(False)
plt.tight_layout()
fig.savefig(f"{SHAP_DIR}/all_models_shap_comparison.png", dpi=180, bbox_inches="tight")
plt.close("all")
print("Saved: all_models_shap_comparison.png")
print("\nDone.")
