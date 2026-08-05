"""
Pairwise DeLong tests across all cohorts and all model pairs.
Uses OOF predictions from 5x10-fold repeated stratified CV.
Saves results to outputs/delong_pairwise.xlsx — one sheet per cohort.
"""

import warnings, json, re, itertools
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.metrics import roc_auc_score
import xgboost as xgb
import lightgbm as lgbm

BASE  = "/Users/elizabetharmstrong/Library/CloudStorage/OneDrive-ImperialCollegeLondon/Lupus Project"
PROC  = f"{BASE}/Data/Processed"
OUT   = f"{BASE}/outputs"

# ── DeLong test implementation ─────────────────────────────────────────────
def _auc_placement(y_true, y_score):
    """Compute structural components V10, V01 for DeLong variance."""
    pos = y_score[y_true == 1]
    neg = y_score[y_true == 0]
    n1, n0 = len(pos), len(neg)
    # placement values
    V10 = np.array([np.mean((p > neg) + 0.5 * (p == neg)) for p in pos])
    V01 = np.array([np.mean((n < pos) + 0.5 * (n == pos)) for n in neg])
    return V10, V01, n1, n0

def delong_test(y_true, prob_a, prob_b):
    """
    DeLong et al. (1988) two-sided test comparing two AUROCs on the same
    labelled dataset. Returns (auc_a, auc_b, z, p_value).
    """
    V10_a, V01_a, n1, n0 = _auc_placement(y_true, prob_a)
    V10_b, V01_b, _,  _  = _auc_placement(y_true, prob_b)

    auc_a = np.mean(V10_a)
    auc_b = np.mean(V10_b)

    # 2x2 covariance matrix
    S10 = np.cov(np.stack([V10_a, V10_b])) / n1
    S01 = np.cov(np.stack([V01_a, V01_b])) / n0
    S   = S10 + S01          # covariance of [auc_a, auc_b]

    # variance of (auc_a - auc_b)
    var_diff = S[0, 0] + S[1, 1] - 2 * S[0, 1]
    if var_diff <= 0:
        return auc_a, auc_b, np.nan, np.nan

    z = (auc_a - auc_b) / np.sqrt(var_diff)
    p = 2 * stats.norm.sf(abs(z))
    return auc_a, auc_b, z, p

# ── Safe column names (LightGBM restriction) ───────────────────────────────
def safe_cols(df):
    return df.rename(columns={c: re.sub(r"[^A-Za-z0-9_]", "_", c)
                               for c in df.columns})

# ── Build models from saved hyperparameters ────────────────────────────────
def build_models(params_rf, params_xgb, params_lgb):
    rf_kw  = {k.replace("clf__", ""): v for k, v in params_rf.items()}
    xgb_kw = {k.replace("clf__", ""): v for k, v in params_xgb.items()}
    lgb_kw = {k.replace("clf__", ""): v for k, v in params_lgb.items()}
    return {
        "Logistic Regression": Pipeline([
            ("sc", StandardScaler()),
            ("clf", LogisticRegression(max_iter=1000, random_state=42)),
        ]),
        "Random Forest": RandomForestClassifier(
            random_state=42, n_jobs=-1, class_weight="balanced", **rf_kw),
        "XGBoost": xgb.XGBClassifier(
            eval_metric="logloss", use_label_encoder=False,
            random_state=42, **xgb_kw),
        "LightGBM": lgbm.LGBMClassifier(verbose=-1, random_state=42, **lgb_kw),
    }

# ── Collect OOF predictions via CV ─────────────────────────────────────────
def get_oof_probs(X, y, models, n_splits=10, n_repeats=5):
    import sklearn.base as skb
    CV = RepeatedStratifiedKFold(n_splits=n_splits, n_repeats=n_repeats,
                                  random_state=42)
    X_safe = safe_cols(X)
    oof = {name: np.zeros(len(y)) for name in models}
    counts = {name: np.zeros(len(y), dtype=int) for name in models}

    for tr, te in CV.split(X_safe, y):
        for name, clf in models.items():
            c = skb.clone(clf)
            c.fit(X_safe.iloc[tr], y.iloc[tr])
            oof[name][te]    += c.predict_proba(X_safe.iloc[te])[:, 1]
            counts[name][te] += 1

    # average across repeats
    for name in oof:
        oof[name] /= counts[name]
    return oof

# ── Run pairwise DeLong for one cohort ────────────────────────────────────
def pairwise_delong(label, X, y, models, n_splits=10, n_repeats=5):
    print(f"\n{'='*60}")
    print(f"  {label}  (n={len(y)}, events={int(y.sum())})")
    print(f"  Collecting OOF predictions ({n_splits}×{n_repeats} folds)...")

    oof = get_oof_probs(X, y, models, n_splits, n_repeats)

    model_names = list(models.keys())
    rows = []
    for a, b in itertools.combinations(model_names, 2):
        auc_a, auc_b, z, p = delong_test(y.values, oof[a], oof[b])
        sig = ""
        if not np.isnan(p):
            if p < 0.001: sig = "***"
            elif p < 0.01: sig = "**"
            elif p < 0.05: sig = "*"
        print(f"    {a} vs {b}: AUROC {auc_a:.3f} vs {auc_b:.3f}  "
              f"z={z:+.2f}  p={p:.4f}  {sig}")
        rows.append({
            "Model A":    a,
            "Model B":    b,
            "AUROC A":    round(auc_a, 3),
            "AUROC B":    round(auc_b, 3),
            "Difference": round(auc_a - auc_b, 3),
            "z":          round(z, 3) if not np.isnan(z) else np.nan,
            "p-value":    round(p, 4) if not np.isnan(p) else np.nan,
            "Significant": sig if sig else "ns",
        })

    # also include individual AUROCs
    auroc_rows = []
    for name in model_names:
        auroc_rows.append({
            "Model": name,
            "OOF AUROC": round(roc_auc_score(y, oof[name]), 3),
        })

    return pd.DataFrame(rows), pd.DataFrame(auroc_rows)

# ── Load data & params ─────────────────────────────────────────────────────
p1  = json.load(open(f"{OUT}/1yr_best_params.json"))
p5  = json.load(open(f"{OUT}/5yr_best_params.json"))
pe  = json.load(open(f"{OUT}/esrd/esrd_best_params.json"))

df1   = pd.read_excel(f"{PROC}/lupus_1yr_selected_clean.xlsx")
df5   = pd.read_excel(f"{PROC}/lupus_5yr_selected_clean.xlsx")
df_e5 = pd.read_excel(f"{PROC}/esrd_5yr_selected.xlsx")
df_e10= pd.read_excel(f"{PROC}/esrd_10yr_selected.xlsx")

COHORTS = [
    ("1-Year Flare",
     df1.drop(columns=["flare_1yr"]), df1["flare_1yr"].astype(int),
     build_models(p1["Random Forest"], p1["XGBoost"], p1["LightGBM"]),
     10, 5),
    ("5-Year Flare",
     df5.drop(columns=["flare_5yr"]), df5["flare_5yr"].astype(int),
     build_models(p5["Random Forest"], p5["XGBoost"], p5["LightGBM"]),
     10, 5),
    ("ESRD 5-Year",
     df_e5.drop(columns=["esrd_5yr"]), df_e5["esrd_5yr"].astype(int),
     build_models(pe["5yr"]["Random Forest"], pe["5yr"]["XGBoost"], pe["5yr"]["LightGBM"]),
     10, 5),
    ("ESRD 10-Year",
     df_e10.drop(columns=["esrd_10yr"]), df_e10["esrd_10yr"].astype(int),
     build_models(pe["10yr"]["Random Forest"], pe["10yr"]["XGBoost"], pe["10yr"]["LightGBM"]),
     10, 5),
]

# ── Run all cohorts & save ─────────────────────────────────────────────────
out_path = f"{OUT}/delong_pairwise.xlsx"
with pd.ExcelWriter(out_path, engine="openpyxl") as writer:
    for label, X, y, models, ns, nr in COHORTS:
        df_pairs, df_auroc = pairwise_delong(label, X, y, models, ns, nr)

        sheet = label.replace("-", "").replace(" ", "_")[:31]
        df_auroc.to_excel(writer, sheet_name=sheet + "_AUROCs", index=False)
        df_pairs.to_excel(writer, sheet_name=sheet,             index=False)

print(f"\n\nSaved: {out_path}")
print("Significance codes: *** p<0.001  ** p<0.01  * p<0.05  ns = not significant")
