"""
ROC curve panel figure — publication quality.
Five panels (1yr flare, 5yr flare, serial biopsy, ESRD 5yr, ESRD 10yr),
all four classifiers overlaid with OOF cross-validated predictions.
"""

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import RepeatedStratifiedKFold, StratifiedKFold
from sklearn.metrics import roc_auc_score, roc_curve
import xgboost as xgb
import lightgbm as lgbm

PROCESSED = "/Users/elizabetharmstrong/Library/CloudStorage/OneDrive-ImperialCollegeLondon/Lupus Project/Data/Processed"
ESRD_DIR  = "/Users/elizabetharmstrong/Library/CloudStorage/OneDrive-ImperialCollegeLondon/Lupus Project/Data/Processed"
OUT_DIR   = "/Users/elizabetharmstrong/Library/CloudStorage/OneDrive-ImperialCollegeLondon/Lupus Project/outputs/figures"

# Validated palette (Wong 2011, colorblind-safe)
COLORS = {
    "Logistic\nRegression": "#0072B2",   # blue
    "Random\nForest":       "#009E73",   # teal
    "XGBoost":              "#D55E00",   # vermillion
    "LightGBM":             "#CC79A7",   # mauve (secondary encoding: line style)
}
STYLES = {
    "Logistic\nRegression": "-",
    "Random\nForest":       "--",
    "XGBoost":              (0, (3, 1.2)),   # dense dot-dash
    "LightGBM":             ":",
}
LW = 1.6   # line width

# Hyperparameters (tuned)
def make_models(params_rf, params_xgb, params_lgbm):
    rf_kw  = {k.replace("clf__",""): v for k, v in params_rf.items()}
    xgb_kw = {k.replace("clf__",""): v for k, v in params_xgb.items()}
    lgb_kw = {k.replace("clf__",""): v for k, v in params_lgbm.items()}

    return {
        "Logistic\nRegression": Pipeline([
            ("sc", StandardScaler()),
            ("clf", LogisticRegression(max_iter=1000, random_state=42)),
        ]),
        "Random\nForest": RandomForestClassifier(
            random_state=42, n_jobs=-1, class_weight="balanced", **rf_kw),
        "XGBoost": xgb.XGBClassifier(
            eval_metric="logloss", use_label_encoder=False,
            random_state=42, **xgb_kw),
        "LightGBM": lgbm.LGBMClassifier(
            verbose=-1, random_state=42, **lgb_kw),
    }

# Cross-validated OOF ROC
def safe_colnames(df):
    """Rename columns to safe identifiers (LightGBM rejects special chars)."""
    import re
    mapping = {c: re.sub(r"[^A-Za-z0-9_]", "_", c) for c in df.columns}
    return df.rename(columns=mapping)

def oof_roc(X, y, models, n_splits=10, n_repeats=5, rng=42):
    """Returns dict of model_name -> (fpr_grid, mean_tpr, std_tpr, auc)."""
    fpr_grid = np.linspace(0, 1, 200)
    results  = {}
    X_safe   = safe_colnames(X) if hasattr(X, "columns") else X

    for name, clf in models.items():
        print(f"    {name.replace(chr(10),' ')} ... ", end="", flush=True)
        cv = RepeatedStratifiedKFold(n_splits=n_splits, n_repeats=n_repeats,
                                     random_state=rng)
        tprs = []
        aucs = []
        for tr, te in cv.split(X_safe, y):
            import sklearn.base as skb
            c = skb.clone(clf)
            c.fit(X_safe.iloc[tr], y.iloc[tr])
            prob = c.predict_proba(X_safe.iloc[te])[:, 1]
            yt   = y.iloc[te]
            fpr, tpr, _ = roc_curve(yt, prob)
            tprs.append(np.interp(fpr_grid, fpr, tpr))
            aucs.append(roc_auc_score(yt, prob))

        mean_tpr = np.mean(tprs, axis=0)
        std_tpr  = np.std(tprs,  axis=0)
        mean_tpr[0] = 0.0
        mean_auc = np.mean(aucs)
        results[name] = (fpr_grid, mean_tpr, std_tpr, mean_auc)
        print(f"AUC={mean_auc:.3f}")
    return results

# Load data
import json

p1  = json.load(open(f"/Users/elizabetharmstrong/Library/CloudStorage/OneDrive-ImperialCollegeLondon/Lupus Project/outputs/1yr_best_params.json"))
p5  = json.load(open(f"/Users/elizabetharmstrong/Library/CloudStorage/OneDrive-ImperialCollegeLondon/Lupus Project/outputs/5yr_best_params.json"))
pe  = json.load(open(f"/Users/elizabetharmstrong/Library/CloudStorage/OneDrive-ImperialCollegeLondon/Lupus Project/outputs/esrd/esrd_best_params.json"))

df1    = pd.read_excel(f"{PROCESSED}/lupus_1yr_selected_clean.xlsx")
df5    = pd.read_excel(f"{PROCESSED}/lupus_5yr_selected_clean.xlsx")
df_ser = pd.read_excel(f"{PROCESSED}/lupus_5yr_serial_selected.xlsx")
df_e5  = pd.read_excel(f"{PROCESSED}/esrd_5yr_selected.xlsx")
df_e10 = pd.read_excel(f"{PROCESSED}/esrd_10yr_selected.xlsx")

cohorts = [
    {
        "label": "1-Year Flare",
        "subtitle": "(n=430, events=99)",
        "X": df1.drop(columns=["flare_1yr"]),
        "y": df1["flare_1yr"].astype(int),
        "models": make_models(p1["Random Forest"], p1["XGBoost"], p1["LightGBM"]),
        "cv": (10, 5),
    },
    {
        "label": "5-Year Flare",
        "subtitle": "(n=356, events=243)",
        "X": df5.drop(columns=["flare_5yr"]),
        "y": df5["flare_5yr"].astype(int),
        "models": make_models(p5["Random Forest"], p5["XGBoost"], p5["LightGBM"]),
        "cv": (10, 5),
    },
    {
        "label": "Serial Biopsy\n5-Year Flare",
        "subtitle": "(n=70, events=49)",
        "X": df_ser.drop(columns=["flare_5yr"]),
        "y": df_ser["flare_5yr"].astype(int),
        "models": make_models(
            {"clf__n_estimators": 100, "clf__min_samples_leaf": 5, "clf__max_features": "sqrt", "clf__max_depth": 3},
            {"clf__subsample": 0.8, "clf__reg_lambda": 1.0, "clf__reg_alpha": 1.0,
             "clf__n_estimators": 50, "clf__min_child_weight": 5, "clf__max_depth": 2,
             "clf__learning_rate": 0.05, "clf__colsample_bytree": 0.8},
            {"clf__subsample": 0.8, "clf__num_leaves": 7, "clf__n_estimators": 50,
             "clf__min_child_samples": 5, "clf__max_depth": 2, "clf__learning_rate": 0.05},
        ),
        "cv": (5, 10),   # 5-fold for small n
    },
    {
        "label": "ESRD\n5-Year",
        "subtitle": "(n=796)",
        "X": df_e5.drop(columns=["esrd_5yr"]),
        "y": df_e5["esrd_5yr"].astype(int),
        "models": make_models(pe["5yr"]["Random Forest"], pe["5yr"]["XGBoost"], pe["5yr"]["LightGBM"]),
        "cv": (10, 5),
    },
    {
        "label": "ESRD\n10-Year",
        "subtitle": "(n=796)",
        "X": df_e10.drop(columns=["esrd_10yr"]),
        "y": df_e10["esrd_10yr"].astype(int),
        "models": make_models(pe["10yr"]["Random Forest"], pe["10yr"]["XGBoost"], pe["10yr"]["LightGBM"]),
        "cv": (10, 5),
    },
]

# Run CV for all cohorts
all_results = []
for c in cohorts:
    print(f"\n{'─'*55}")
    print(f"  {c['label'].replace(chr(10),' ')}  {c['subtitle']}")
    ns, nr = c["cv"]
    res = oof_roc(c["X"], c["y"], c["models"], n_splits=ns, n_repeats=nr)
    all_results.append(res)

# Figure
matplotlib.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
    "font.size": 8,
    "axes.labelsize": 8,
    "axes.titlesize": 9,
    "xtick.labelsize": 7,
    "ytick.labelsize": 7,
    "axes.linewidth": 0.6,
    "xtick.major.width": 0.6,
    "ytick.major.width": 0.6,
    "xtick.major.size": 3,
    "ytick.major.size": 3,
    "legend.fontsize": 6.5,
    "legend.framealpha": 0.92,
    "legend.edgecolor": "#cccccc",
    "legend.borderpad": 0.5,
    "figure.dpi": 300,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
})

# 3 top + 2 bottom centred layout
fig = plt.figure(figsize=(7.2, 5.1))
# Use gridspec: 2 rows, 6 cols; top panels span cols 0-1, 2-3, 4-5
#               bottom panels span cols 1-2, 3-4 (centred)
import matplotlib.gridspec as gridspec
gs = gridspec.GridSpec(2, 6, figure=fig,
                       left=0.07, right=0.98,
                       top=0.93, bottom=0.09,
                       hspace=0.52, wspace=0.38)

axes = [
    fig.add_subplot(gs[0, 0:2]),   # 1yr
    fig.add_subplot(gs[0, 2:4]),   # 5yr
    fig.add_subplot(gs[0, 4:6]),   # serial
    fig.add_subplot(gs[1, 1:3]),   # ESRD 5yr (centred)
    fig.add_subplot(gs[1, 3:5]),   # ESRD 10yr (centred)
]

panel_letters = ["A", "B", "C", "D", "E"]

for ax, c, res, letter in zip(axes, cohorts, all_results, panel_letters):
    # diagonal
    ax.plot([0, 1], [0, 1], color="#bbbbbb", lw=0.8, ls="--", zorder=0)

    for name, (fpr, tpr, std, auc) in res.items():
        label = f"{name.replace(chr(10),' ')}  AUC {auc:.3f}"
        ax.plot(fpr, tpr,
                color=COLORS[name],
                lw=LW,
                ls=STYLES[name],
                label=label,
                zorder=3)

    ax.set_xlim(-0.01, 1.01)
    ax.set_ylim(-0.01, 1.01)
    ax.set_aspect("equal")

    title_text  = c["label"].replace("\n", " ")
    subtitle    = c["subtitle"]
    ax.set_title(f"{title_text}\n{subtitle}",
                 fontsize=8, pad=4)

    ax.set_xlabel("1 – Specificity", labelpad=3)
    ax.set_ylabel("Sensitivity",     labelpad=3)

    ax.set_xticks([0, 0.25, 0.5, 0.75, 1.0])
    ax.set_yticks([0, 0.25, 0.5, 0.75, 1.0])
    ax.set_xticklabels(["0", ".25", ".50", ".75", "1"])
    ax.set_yticklabels(["0", ".25", ".50", ".75", "1"])

    # grid — very light
    ax.set_axisbelow(True)
    ax.grid(color="#e8e8e8", linewidth=0.4)

    # legend inside, lower right
    leg = ax.legend(loc="lower right",
                    handlelength=1.8,
                    handleheight=1.1,
                    borderaxespad=0.4,
                    labelspacing=0.35)
    for lh in leg.get_lines():
        lh.set_linewidth(1.6)

    # panel letter — upper left, outside axis
    ax.text(-0.14, 1.06, letter,
            transform=ax.transAxes,
            fontsize=10, fontweight="bold", va="top", ha="left")

    # right + top spines off
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

out_path = f"{OUT_DIR}/roc_panel.pdf"
fig.savefig(out_path, format="pdf")
fig.savefig(out_path.replace(".pdf", ".png"), format="png", dpi=300)
print(f"\nSaved: {out_path}")
print(f"Saved: {out_path.replace('.pdf', '.png')}")
plt.close()
