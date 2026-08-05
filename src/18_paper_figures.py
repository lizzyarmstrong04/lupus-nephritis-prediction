"""
Three paper figures:
  Fig 2 — Calibration curve panel   (5 panels, 3+2)
  Fig 3 — Apparent vs BC AUROC      (5 panels, 3+2 dumbbell)
  Fig 4 — SHAP importance           (4 panels, 2×2)

All figures: Times New Roman, 300 dpi, PDF + PNG.
"""

import warnings
warnings.filterwarnings("ignore")

import json, os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.metrics import roc_auc_score
from sklearn.calibration import calibration_curve
import sklearn.base as skb
import xgboost as xgb
import lightgbm as lgbm
import re

# ── Paths ──────────────────────────────────────────────────────────────────
BASE     = "/Users/elizabetharmstrong/Library/CloudStorage/OneDrive-ImperialCollegeLondon/Lupus Project"
PROC     = f"{BASE}/Data/Processed"
OUT      = f"{BASE}/outputs"
FIG_DIR  = f"{OUT}/figures"
os.makedirs(FIG_DIR, exist_ok=True)

# ── Validated categorical palette (Wong 2011) ──────────────────────────────
PALETTE = {
    "Logistic\nRegression": "#0072B2",
    "Random\nForest":       "#009E73",
    "XGBoost":              "#D55E00",
    "LightGBM":             "#CC79A7",
}
LSTYLES = {
    "Logistic\nRegression": "-",
    "Random\nForest":       "--",
    "XGBoost":              (0, (3, 1.2)),
    "LightGBM":             ":",
}
MODEL_LABELS = list(PALETTE.keys())
SHORT_LABELS = ["LR", "RF", "XGBoost", "LightGBM"]

# ── Typography ─────────────────────────────────────────────────────────────
matplotlib.rcParams.update({
    "font.family":           "serif",
    "font.serif":            ["Times New Roman", "Times", "DejaVu Serif"],
    "mathtext.fontset":      "dejavuserif",
    "font.size":             8,
    "axes.labelsize":        8,
    "axes.titlesize":        9,
    "xtick.labelsize":       7,
    "ytick.labelsize":       7,
    "axes.linewidth":        0.6,
    "xtick.major.width":     0.6,
    "ytick.major.width":     0.6,
    "xtick.major.size":      3,
    "ytick.major.size":      3,
    "legend.fontsize":       6.5,
    "legend.framealpha":     0.92,
    "legend.edgecolor":      "#cccccc",
    "legend.borderpad":      0.5,
    "figure.dpi":            300,
    "savefig.dpi":           300,
    "savefig.bbox":          "tight",
})

# ── Feature name cleanup map ───────────────────────────────────────────────
FEAT_LABELS = {
    "% chronic gloms(%of total)":                                                          "Chronic glomeruli (%)",
    "%gloms with necrosis":                                                                "Glomerular necrosis (%)",
    "Age at biopsy":                                                                       "Age",
    "Proteinuria at biopsy (uPCR, log)":                                                   "Proteinuria (log uPCR)",
    "Class coded 1=I 2=II 3=III 4=IV 5=V 6=III+V 7=IV+V 8=II+V 9=VI 10=other":          "LN class",
    "Ethnicity 1=white 2=black 3=asian (south) 4=asian (east) 5=other 6=not stated/unknown/any other mixed": "Ethnicity",
    "% active gloms (%of those not globally sclerosed)":                                   "Active glomeruli (%)",
    "%gloms with crescents":                                                               "Crescents (%)",
    "C4 at biopsy":                                                                        "C4",
    "% sclerosed gloms":                                                                   "Sclerosed glomeruli (%)",
    "dsDNA or SM or APL ever positive(1=yes 0=no)":                                        "dsDNA/SM/APL positive",
    "Prev exposure to cyclo (for Rx comparison) - related to the 'use this biopsy for this patient' biopsy": "Prior cyclophosphamide",
    "CKD epi formula without ethnicity":                                                    "eGFR (CKD-EPI)",
    "Reason for biopsy 1=new pres LN 2=relapse 3=non-response/partial response, incl on-going proteinuria 4=pre-pregnancy or Ax if drug switch/stop appropriate": "Biopsy indication",
    "Creatinine at biopsy":                                                                "Creatinine",
    "%IFTA ":                                                                              "IFTA (%)",
    "Subepithelial deposit category (0=no deposits, 1=small/rare deposits, 2=large/conspicuous deposits, 3=no gloms on EM)": "Subepithelial deposits",
    "C3 at biopsy (normal range 0.7-1.7)":                                                "C3",
    "C4 low (for range 0.15-0.54)":                                                       "C4 low",
    "Crescents (Yes=1, No=0)":                                                             "Crescents",
    "TMA (Yes=1, No=0)":                                                                   "TMA",
    "Cap wall IgM":                                                                        "Cap wall IgM",
    "No. globally sclerosed gloms ":                                                       "Globally sclerosed gloms",
    "Biopsy number for patient":                                                           "Biopsy number",
    "No. gloms with crescents":                                                            "Gloms with crescents (n)",
    "Gender (1=male, 2=female)":                                                           "Sex",
    "dsDNA or SM or APL ever positive(1=yes 0=no)":                                        "dsDNA/SM/APL positive",
}

def short_feat(name):
    name = str(name).strip()
    # 1. Exact match
    if name in FEAT_LABELS:
        return FEAT_LABELS[name]
    # 2. Prefix match — covers cases where Excel stored a slightly different string
    for k, v in FEAT_LABELS.items():
        if name.startswith(k[:20]) or k.startswith(name[:20]):
            return v
    # 3. Smart fallback: strip parenthetical tail, keep meaningful words
    clean = re.sub(r'\s*[\(\[].*', '', name).strip()  # remove from first ( or [
    clean = re.sub(r'\s+', ' ', clean).strip()
    return clean if len(clean) <= 28 else clean[:27].rstrip() + "…"

# ── Safe column names (LightGBM) ───────────────────────────────────────────
def safe_cols(df):
    return df.rename(columns={c: re.sub(r"[^A-Za-z0-9_]", "_", c) for c in df.columns})

# ── Model factories ────────────────────────────────────────────────────────
def make_models(params_rf, params_xgb, params_lgbm):
    rf_kw  = {k.replace("clf__", ""): v for k, v in params_rf.items()}
    xgb_kw = {k.replace("clf__", ""): v for k, v in params_xgb.items()}
    lgb_kw = {k.replace("clf__", ""): v for k, v in params_lgbm.items()}
    return {
        "Logistic\nRegression": Pipeline([("sc", StandardScaler()),
                                          ("clf", LogisticRegression(max_iter=1000, random_state=42))]),
        "Random\nForest": RandomForestClassifier(random_state=42, n_jobs=-1,
                                                  class_weight="balanced", **rf_kw),
        "XGBoost":   xgb.XGBClassifier(eval_metric="logloss", use_label_encoder=False,
                                         random_state=42, **xgb_kw),
        "LightGBM":  lgbm.LGBMClassifier(verbose=-1, random_state=42, **lgb_kw),
    }

# ── OOF predictions ────────────────────────────────────────────────────────
def oof_probs(X, y, models, n_splits=10, n_repeats=5):
    """Returns dict model_name -> (y_true_concat, y_prob_concat)."""
    cv = RepeatedStratifiedKFold(n_splits=n_splits, n_repeats=n_repeats, random_state=42)
    X_s = safe_cols(X)
    out = {name: ([], []) for name in models}
    for tr, te in cv.split(X_s, y):
        for name, clf in models.items():
            c = skb.clone(clf)
            c.fit(X_s.iloc[tr], y.iloc[tr])
            probs = c.predict_proba(X_s.iloc[te])[:, 1]
            out[name][0].extend(y.iloc[te].tolist())
            out[name][1].extend(probs.tolist())
    return {k: (np.array(v[0]), np.array(v[1])) for k, v in out.items()}

# ── Load everything ────────────────────────────────────────────────────────
p1  = json.load(open(f"{OUT}/1yr_best_params.json"))
p5  = json.load(open(f"{OUT}/5yr_best_params.json"))
pe  = json.load(open(f"{OUT}/esrd/esrd_best_params.json"))

df1    = pd.read_excel(f"{PROC}/lupus_1yr_selected_clean.xlsx")
df5    = pd.read_excel(f"{PROC}/lupus_5yr_selected_clean.xlsx")
df_ser = pd.read_excel(f"{PROC}/lupus_5yr_serial_selected.xlsx")
df_e5  = pd.read_excel(f"{PROC}/esrd_5yr_selected.xlsx")
df_e10 = pd.read_excel(f"{PROC}/esrd_10yr_selected.xlsx")

# Bootstrap results (existing)
b1    = pd.read_excel(f"{OUT}/1yr_model_results.xlsx",        sheet_name="Bootstrap (Harrell)")
b5    = pd.read_excel(f"{OUT}/5yr_model_results.xlsx",        sheet_name="Bootstrap (Harrell)")
be5   = pd.read_excel(f"{OUT}/esrd/esrd_model_results.xlsx",  sheet_name="5yr Bootstrap")
be10  = pd.read_excel(f"{OUT}/esrd/esrd_model_results.xlsx",  sheet_name="10yr Bootstrap")

COHORTS = [
    {"label": "1-Year Flare",            "subtitle": "(n=430, events=99)",
     "X": df1.drop(columns=["flare_1yr"]),    "y": df1["flare_1yr"].astype(int),
     "models": make_models(p1["Random Forest"], p1["XGBoost"], p1["LightGBM"]),
     "cv": (10, 5)},
    {"label": "5-Year Flare",            "subtitle": "(n=356, events=243)",
     "X": df5.drop(columns=["flare_5yr"]),    "y": df5["flare_5yr"].astype(int),
     "models": make_models(p5["Random Forest"], p5["XGBoost"], p5["LightGBM"]),
     "cv": (10, 5)},
    {"label": "Serial Biopsy\n5-Year Flare", "subtitle": "(n=70, events=49)",
     "X": df_ser.drop(columns=["flare_5yr"]), "y": df_ser["flare_5yr"].astype(int),
     "models": make_models(
         {"clf__n_estimators": 100, "clf__min_samples_leaf": 5, "clf__max_features": "sqrt", "clf__max_depth": 3},
         {"clf__subsample": 0.8, "clf__reg_lambda": 1.0, "clf__reg_alpha": 1.0, "clf__n_estimators": 50,
          "clf__min_child_weight": 5, "clf__max_depth": 2, "clf__learning_rate": 0.05, "clf__colsample_bytree": 0.8},
         {"clf__subsample": 0.8, "clf__num_leaves": 7, "clf__n_estimators": 50,
          "clf__min_child_samples": 5, "clf__max_depth": 2, "clf__learning_rate": 0.05}),
     "cv": (5, 10)},
    {"label": "ESRD\n5-Year",            "subtitle": "(n=796)",
     "X": df_e5.drop(columns=["esrd_5yr"]),   "y": df_e5["esrd_5yr"].astype(int),
     "models": make_models(pe["5yr"]["Random Forest"], pe["5yr"]["XGBoost"], pe["5yr"]["LightGBM"]),
     "cv": (10, 5)},
    {"label": "ESRD\n10-Year",           "subtitle": "(n=796)",
     "X": df_e10.drop(columns=["esrd_10yr"]), "y": df_e10["esrd_10yr"].astype(int),
     "models": make_models(pe["10yr"]["Random Forest"], pe["10yr"]["XGBoost"], pe["10yr"]["LightGBM"]),
     "cv": (10, 5)},
]

# ── Bootstrap for serial (quick — n=70) ───────────────────────────────────
def serial_bootstrap(X, y, models, n_boot=500):
    X_s = safe_cols(X)
    rng = np.random.default_rng(42)
    rows = []
    for name, clf in models.items():
        clf_app = skb.clone(clf)
        clf_app.fit(X_s, y)
        app = roc_auc_score(y, clf_app.predict_proba(X_s)[:, 1])
        opts = []
        for _ in range(n_boot):
            idx = rng.choice(len(y), len(y), replace=True)
            Xb, yb = X_s.iloc[idx], y.iloc[idx]
            if yb.nunique() < 2:
                continue
            cb = skb.clone(clf)
            cb.fit(Xb, yb)
            opts.append(roc_auc_score(yb, cb.predict_proba(Xb)[:, 1]) -
                        roc_auc_score(y,  cb.predict_proba(X_s)[:, 1]))
        bc = app - np.mean(opts)
        rows.append({"Model": name.replace("\n", " "), "Apparent AUROC": app,
                     "Optimism": np.mean(opts), "BC AUROC": bc})
    return pd.DataFrame(rows)

print("Running bootstrap for serial cohort...")
b_ser = serial_bootstrap(
    df_ser.drop(columns=["flare_5yr"]),
    df_ser["flare_5yr"].astype(int),
    COHORTS[2]["models"])
print(b_ser.to_string(index=False))

# ── Run OOF for all 5 cohorts ──────────────────────────────────────────────
print("\nRunning OOF cross-validation for calibration curves...")
all_oof = []
for i, c in enumerate(COHORTS):
    print(f"  {c['label'].replace(chr(10),' ')} ...", flush=True)
    ns, nr = c["cv"]
    res = oof_probs(c["X"], c["y"], c["models"], n_splits=ns, n_repeats=nr)
    all_oof.append(res)
print("OOF done.")

# ══════════════════════════════════════════════════════════════════════
# FIGURE 2 — Calibration curves
# ══════════════════════════════════════════════════════════════════════
def make_3x2_fig():
    fig = plt.figure(figsize=(7.2, 5.1))
    gs  = gridspec.GridSpec(2, 6, figure=fig,
                            left=0.07, right=0.98, top=0.93, bottom=0.09,
                            hspace=0.52, wspace=0.38)
    return fig, [
        fig.add_subplot(gs[0, 0:2]),
        fig.add_subplot(gs[0, 2:4]),
        fig.add_subplot(gs[0, 4:6]),
        fig.add_subplot(gs[1, 1:3]),
        fig.add_subplot(gs[1, 3:5]),
    ]

fig2, axes2 = make_3x2_fig()
LETTERS = ["A", "B", "C", "D", "E"]

for ax, c, oof, letter in zip(axes2, COHORTS, all_oof, LETTERS):
    ax.plot([0, 1], [0, 1], color="#aaaaaa", lw=0.9, ls="--", zorder=0,
            label="Perfect calibration")

    for name in MODEL_LABELS:
        yt, yp = oof[name]
        # quantile strategy: each bin has ~equal number of samples
        n_bins = 8 if len(yt) < 200 else 10
        try:
            frac_pos, mean_pred = calibration_curve(yt, yp, n_bins=n_bins,
                                                     strategy="quantile")
            ax.plot(mean_pred, frac_pos,
                    color=PALETTE[name], lw=1.5, ls=LSTYLES[name],
                    marker="o", ms=3.5, zorder=3,
                    label=f"{name.replace(chr(10), ' ')}")
        except Exception:
            pass

    ax.set_xlim(-0.02, 1.02)
    ax.set_ylim(-0.02, 1.02)
    ax.set_aspect("equal")
    title_text = c["label"].replace("\n", " ")
    ax.set_title(f"{title_text}\n{c['subtitle']}", fontsize=8, pad=4)
    ax.set_xlabel("Mean predicted probability", labelpad=3)
    ax.set_ylabel("Observed event rate",        labelpad=3)
    ax.set_xticks([0, 0.25, 0.5, 0.75, 1.0])
    ax.set_yticks([0, 0.25, 0.5, 0.75, 1.0])
    ax.set_xticklabels(["0", ".25", ".50", ".75", "1"])
    ax.set_yticklabels(["0", ".25", ".50", ".75", "1"])
    ax.set_axisbelow(True)
    ax.grid(color="#e8e8e8", linewidth=0.4)
    ax.legend(loc="upper left", handlelength=1.8, borderaxespad=0.3, labelspacing=0.3)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.text(-0.14, 1.06, letter, transform=ax.transAxes,
            fontsize=10, fontweight="bold", va="top", ha="left")

fig2.savefig(f"{FIG_DIR}/calibration_panel.pdf")
fig2.savefig(f"{FIG_DIR}/calibration_panel.png", dpi=300)
plt.close(fig2)
print(f"Saved calibration_panel.pdf / .png")

# ══════════════════════════════════════════════════════════════════════
# FIGURE 3 — Apparent vs Bias-Corrected AUROC (dumbbell)
# ══════════════════════════════════════════════════════════════════════

def prep_boot(df, app_col, bc_col):
    """Standardise bootstrap table to [Model, Apparent, BC]."""
    return pd.DataFrame({
        "Model":    df["Model"].str.replace("Random Forest", "RF"),
        "Apparent": df[app_col].values,
        "BC":       df[bc_col].values,
    })

b1_std    = prep_boot(b1,   "Apparent AUROC", "Bias-Corrected AUROC (Harrell)")
b5_std    = prep_boot(b5,   "Apparent AUROC", "Bias-Corrected AUROC (Harrell)")
b_ser_std = prep_boot(b_ser,"Apparent AUROC", "BC AUROC")
be5_std   = prep_boot(be5,  "Apparent AUROC", "BC AUROC")
be10_std  = prep_boot(be10, "Apparent AUROC", "BC AUROC")

# Rename models to short labels in the std tables
MODEL_RENAME = {
    "Logistic Regression": "LR",
    "RF": "RF",
    "Random Forest": "RF",
    "XGBoost": "XGBoost",
    "LightGBM": "LightGBM",
}
for df in [b1_std, b5_std, b_ser_std, be5_std, be10_std]:
    df["Model"] = df["Model"].map(MODEL_RENAME).fillna(df["Model"])

# Dumbbell colors: BC = solid, Apparent = open circle
# Use same palette keyed by SHORT_LABELS
SHORT_PALETTE = {"LR": "#0072B2", "RF": "#009E73",
                 "XGBoost": "#D55E00", "LightGBM": "#CC79A7"}
MODEL_ORDER   = ["LR", "RF", "XGBoost", "LightGBM"]

BOOT_DATA = [b1_std, b5_std, b_ser_std, be5_std, be10_std]

fig3, axes3 = make_3x2_fig()
fig3.set_size_inches(7.2, 5.1)

for ax, c, bdf, letter in zip(axes3, COHORTS, BOOT_DATA, LETTERS):
    y_pos = np.arange(len(MODEL_ORDER))
    for yi, m in enumerate(MODEL_ORDER):
        row = bdf[bdf["Model"] == m]
        if row.empty:
            continue
        app = float(row["Apparent"].iloc[0])
        bc  = float(row["BC"].iloc[0])
        col = SHORT_PALETTE[m]
        # connector line
        ax.plot([bc, app], [yi, yi], color=col, lw=1.4, zorder=2)
        # BC dot (solid)
        ax.scatter(bc,  yi, color=col, s=28, zorder=4, marker="o", clip_on=False)
        # Apparent dot (open)
        ax.scatter(app, yi, color=col, s=28, zorder=4, marker="D",
                   facecolors="white", edgecolors=col, linewidths=1.2, clip_on=False)

    ax.set_yticks(y_pos)
    ax.set_yticklabels(MODEL_ORDER, fontsize=7)
    ax.set_xlim(0.45, 1.02)
    ax.set_xlabel("AUROC", labelpad=3)
    title_text = c["label"].replace("\n", " ")
    ax.set_title(f"{title_text}\n{c['subtitle']}", fontsize=8, pad=4)
    ax.set_axisbelow(True)
    ax.grid(axis="x", color="#e8e8e8", linewidth=0.4)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.text(-0.20, 1.06, letter, transform=ax.transAxes,
            fontsize=10, fontweight="bold", va="top", ha="left")

# Shared legend
legend_elements = [
    Line2D([0], [0], marker="o", color="#555555", ms=5,  linewidth=0,
           label="Bias-corrected (Harrell)"),
    Line2D([0], [0], marker="D", color="#555555", ms=5,  linewidth=0,
           markerfacecolor="white", markeredgewidth=1.2,
           label="Apparent (training)"),
]
fig3.legend(handles=legend_elements, loc="lower center",
            ncol=2, frameon=True, fontsize=7,
            bbox_to_anchor=(0.5, -0.01))

fig3.savefig(f"{FIG_DIR}/auroc_apparent_vs_bc.pdf")
fig3.savefig(f"{FIG_DIR}/auroc_apparent_vs_bc.png", dpi=300)
plt.close(fig3)
print(f"Saved auroc_apparent_vs_bc.pdf / .png")

# ══════════════════════════════════════════════════════════════════════
# FIGURE 4 — SHAP importance (2×2, 4 cohorts, no serial)
# ══════════════════════════════════════════════════════════════════════

s1   = pd.read_excel(f"{OUT}/shap_importance_table.xlsx")
s5   = pd.read_excel(f"{OUT}/shap_importance_table_5yr.xlsx")
se5  = pd.read_excel(f"{OUT}/esrd/esrd_shap_table_5yr.xlsx")
se10 = pd.read_excel(f"{OUT}/esrd/esrd_shap_table_10yr.xlsx")

SHAP_COHORTS = [
    {"label": "1-Year Flare",   "df": s1,   "top": 9},
    {"label": "5-Year Flare",   "df": s5,   "top": 9},
    {"label": "ESRD — 5-Year",  "df": se5,  "top": 5},
    {"label": "ESRD — 10-Year", "df": se10, "top": 10},
]

# Use a single sequential blue (magnitude encoding)
SHAP_COLOR = "#0072B2"

fig4, axes4 = plt.subplots(2, 2, figsize=(7.5, 7.8),
                            gridspec_kw={"hspace": 0.60, "wspace": 1.10})
axes4 = axes4.flatten()

SHAP_LETTERS = ["A", "B", "C", "D"]

for ax, sc, letter in zip(axes4, SHAP_COHORTS, SHAP_LETTERS):
    df = sc["df"].copy()
    df = df.rename(columns={"Feature": "feature",
                             "Mean across models": "mean_shap"})
    df = df.nlargest(sc["top"], "mean_shap").sort_values("mean_shap")
    df["feature_label"] = df["feature"].apply(short_feat)

    n_feats   = len(df)
    # Adaptive font: smaller for panels with more rows
    label_fs  = 6.0 if n_feats <= 6 else 5.5
    bar_h     = 0.55

    y_pos = np.arange(n_feats)
    ax.barh(y_pos, df["mean_shap"], color=SHAP_COLOR, height=bar_h, zorder=2)

    ax.set_yticks(y_pos)
    ax.set_yticklabels(df["feature_label"], fontsize=label_fs)
    ax.set_ylim(-0.7, n_feats - 0.3)   # give breathing room top & bottom
    ax.set_xlabel("Mean |SHAP| (ensemble average)", fontsize=7, labelpad=3)
    ax.set_title(sc["label"], fontsize=9, pad=5)
    ax.set_axisbelow(True)
    ax.grid(axis="x", color="#e8e8e8", linewidth=0.4)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.text(-0.60, 1.06, letter, transform=ax.transAxes,
            fontsize=10, fontweight="bold", va="top", ha="left")

    # Value labels on bars
    x_pad = max(df["mean_shap"]) * 0.02
    for yi, val in zip(y_pos, df["mean_shap"]):
        ax.text(val + x_pad, yi, f"{val:.3f}",
                va="center", ha="left", fontsize=5.5, color="#333333")

fig4.savefig(f"{FIG_DIR}/shap_importance.pdf")
fig4.savefig(f"{FIG_DIR}/shap_importance.png", dpi=300)
plt.close(fig4)
print(f"Saved shap_importance.pdf / .png")
print("\nAll three figures complete.")
