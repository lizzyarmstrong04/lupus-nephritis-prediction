"""
EPV-cap sensitivity analysis: 1-year flare and serial biopsy are the two
most feature-constrained cohorts (EPV_MAX derived from an EPV-10 rule).
This script relaxes the cap to an EPV-5 rule (roughly double the allowed
predictors), reruns the *same* automated feature-selection pipeline (steps
1-6, unchanged) and the *same* tuning + 5x-fold CV protocol used in the
main pipeline, and reports whether CV AUROC / Brier / calibration slope
move materially.

Notes / scope:
  - This reruns only the AUTOMATED LASSO-capped selection. It does not
    reproduce the manual post-selection clinical corrections described in
    the Methods document (e.g. Age Now -> Age at biopsy, ANA/ANCA removal)
    since those are judgement calls, not mechanically reproducible from
    the cap alone. This isolates the marginal effect of the cap itself.
  - Harrell bootstrap and plots are skipped (not needed to answer "does
    performance move materially" - CV metrics already answer that) to
    keep runtime reasonable.
  - Does NOT overwrite Data/Processed/lupus_1yr_selected_clean.xlsx or
    lupus_5yr_serial_selected.xlsx (the official pipeline outputs).

Saves: outputs/epv5_sensitivity_results.xlsx
"""
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import RepeatedStratifiedKFold, StratifiedKFold, RandomizedSearchCV
from sklearn.metrics import roc_auc_score, brier_score_loss
from sklearn.pipeline import Pipeline
from sklearn.experimental import enable_iterative_imputer  # noqa
from sklearn.impute import IterativeImputer
from statsmodels.stats.outliers_influence import variance_inflation_factor
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier

BASE = "/Users/elizabetharmstrong/Library/CloudStorage/OneDrive-ImperialCollegeLondon/Lupus Project"
PROC = f"{BASE}/Data/Processed"
OUT  = f"{BASE}/outputs"


def calibration_slope(y_true, y_prob):
    log_odds = np.log(np.clip(y_prob, 1e-6, 1 - 1e-6) / (1 - np.clip(y_prob, 1e-6, 1 - 1e-6)))
    m = LogisticRegression(fit_intercept=True, max_iter=1000)
    m.fit(log_odds.reshape(-1, 1), y_true)
    return float(m.coef_[0][0])


def select_features(X, y, epv_max, c_values):
    """Steps 3a-6 of the pipeline (dominant binary -> low variance ->
    high correlation -> VIF -> LASSO hard cap). X must already be numeric,
    leakage-free, and complete (no missing)."""
    # 3a
    binary_cols = [c for c in X.columns if X[c].dropna().isin([0, 1]).all()]
    dom = [c for c in binary_cols if X[c].value_counts(normalize=True).max() > 0.90]
    X = X.drop(columns=dom)
    # 3b
    var = X.var()
    X = X.drop(columns=var[var < 0.01].index.tolist())
    # 4
    changed = True
    while changed:
        changed = False
        corr = X.corr().abs()
        upper = corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool))
        for col in upper.columns:
            partners = upper.index[upper[col] > 0.8].tolist()
            if partners:
                X = X.drop(columns=[col]); changed = True; break
    # 5
    def vif_df(Xd):
        Xs = pd.DataFrame(StandardScaler().fit_transform(Xd), columns=Xd.columns)
        return pd.DataFrame({"Feature": Xd.columns,
                              "VIF": [variance_inflation_factor(Xs.values, i) for i in range(Xs.shape[1])]})
    while True:
        vdf = vif_df(X).sort_values("VIF", ascending=False)
        if vdf.iloc[0]["VIF"] > 10:
            X = X.drop(columns=[vdf.iloc[0]["Feature"]])
        else:
            break
    # 6 - LASSO hard cap
    Xs = StandardScaler().fit_transform(X)
    chosen_coefs, chosen_C = None, None
    if X.shape[1] > epv_max:
        for C in c_values:
            model = LogisticRegression(penalty="l1", solver="saga", C=C, max_iter=20000, random_state=42)
            model.fit(Xs, y)
            n_nonzero = int(np.sum(model.coef_[0] != 0))
            if n_nonzero <= epv_max:
                chosen_C, chosen_coefs = C, model.coef_[0]
                break
        if chosen_C is None:
            chosen_C = c_values[-1]
            model = LogisticRegression(penalty="l1", solver="saga", C=chosen_C, max_iter=20000, random_state=42)
            model.fit(Xs, y)
            chosen_coefs = model.coef_[0]
        keep = pd.Series(chosen_coefs, index=X.columns)
        X = X[keep[keep != 0].index.tolist()]
    return X


def tune_and_cv(X, y, scale_pos_weight, inner_folds, outer_splits, outer_repeats, n_iter):
    inner_cv = StratifiedKFold(n_splits=inner_folds, shuffle=True, random_state=42)
    param_grids = {
        "Random Forest": {
            "clf__n_estimators": [100, 200, 300], "clf__max_depth": [2, 3, 5, 7, None],
            "clf__min_samples_leaf": [5, 10, 15, 20], "clf__max_features": ["sqrt", 0.5, 0.7],
        },
        "XGBoost": {
            "clf__n_estimators": [50, 100, 200], "clf__max_depth": [2, 3],
            "clf__learning_rate": [0.01, 0.05], "clf__subsample": [0.5, 0.6, 0.7],
            "clf__colsample_bytree": [0.5, 0.6, 0.7], "clf__min_child_weight": [3, 5, 10, 15],
            "clf__reg_alpha": [0.1, 0.5, 1.0, 2.0], "clf__reg_lambda": [1.0, 2.0, 5.0, 10.0],
        },
        "LightGBM": {
            "clf__n_estimators": [50, 100, 200, 300], "clf__max_depth": [2, 3, 4],
            "clf__learning_rate": [0.01, 0.05, 0.1], "clf__subsample": [0.7, 0.8, 1.0],
            "clf__num_leaves": [7, 15, 31, 63], "clf__min_child_samples": [5, 10, 20, 30],
        },
    }
    base_pipelines = {
        "Random Forest": Pipeline([("scaler", StandardScaler()),
                                    ("clf", RandomForestClassifier(random_state=42, class_weight="balanced"))]),
        "XGBoost": Pipeline([("scaler", StandardScaler()),
                              ("clf", XGBClassifier(random_state=42, eval_metric="logloss", verbosity=0,
                                                     scale_pos_weight=scale_pos_weight))]),
        "LightGBM": Pipeline([("scaler", StandardScaler()),
                               ("clf", LGBMClassifier(random_state=42, verbose=-1, class_weight="balanced"))]),
    }
    best_params = {}
    for name, pipe in base_pipelines.items():
        search = RandomizedSearchCV(pipe, param_distributions=param_grids[name], n_iter=n_iter,
                                     scoring="roc_auc", cv=inner_cv, random_state=42, n_jobs=-1, refit=True)
        search.fit(X, y)
        best_params[name] = search.best_params_

    def make_rf(p): return RandomForestClassifier(
        n_estimators=p.get("clf__n_estimators", 300), max_depth=p.get("clf__max_depth"),
        min_samples_leaf=p.get("clf__min_samples_leaf", 10), max_features=p.get("clf__max_features", "sqrt"),
        class_weight="balanced", random_state=42)

    def make_xgb(p): return XGBClassifier(
        n_estimators=p.get("clf__n_estimators", 100), max_depth=p.get("clf__max_depth", 2),
        learning_rate=p.get("clf__learning_rate", 0.05), subsample=p.get("clf__subsample", 0.5),
        colsample_bytree=p.get("clf__colsample_bytree", 0.5), min_child_weight=p.get("clf__min_child_weight", 10),
        reg_alpha=p.get("clf__reg_alpha", 1.0), reg_lambda=p.get("clf__reg_lambda", 5.0),
        eval_metric="logloss", verbosity=0, random_state=42, scale_pos_weight=scale_pos_weight)

    def make_lgbm(p): return LGBMClassifier(
        n_estimators=p.get("clf__n_estimators", 200), max_depth=p.get("clf__max_depth", 3),
        learning_rate=p.get("clf__learning_rate", 0.05), subsample=p.get("clf__subsample", 0.8),
        num_leaves=p.get("clf__num_leaves", 31), min_child_samples=p.get("clf__min_child_samples", 20),
        verbose=-1, random_state=42, class_weight="balanced")

    MODELS = {
        "Logistic Regression": Pipeline([("scaler", StandardScaler()),
                                          ("clf", LogisticRegression(max_iter=1000, random_state=42, C=1.0,
                                                                      class_weight="balanced"))]),
        "Random Forest": Pipeline([("scaler", StandardScaler()), ("clf", make_rf(best_params["Random Forest"]))]),
        "XGBoost": Pipeline([("scaler", StandardScaler()), ("clf", make_xgb(best_params["XGBoost"]))]),
        "LightGBM": Pipeline([("scaler", StandardScaler()), ("clf", make_lgbm(best_params["LightGBM"]))]),
    }

    CV = RepeatedStratifiedKFold(n_splits=outer_splits, n_repeats=outer_repeats, random_state=42)
    results = {}
    for name, pipe in MODELS.items():
        aurocs, briers, slopes = [], [], []
        for tr, te in CV.split(X, y):
            pipe.fit(X.iloc[tr], y.iloc[tr])
            probs = pipe.predict_proba(X.iloc[te])[:, 1]
            aurocs.append(roc_auc_score(y.iloc[te], probs))
            briers.append(brier_score_loss(y.iloc[te], probs))
            slopes.append(calibration_slope(y.iloc[te].values, probs))
        results[name] = {"CV AUROC": round(np.mean(aurocs), 3), "CV Brier": round(np.mean(briers), 3),
                          "CV Cal Slope": round(np.mean(slopes), 3)}
    return results


def run_cohort(cohort_name, X_raw, y, epv_caps, c_values, scale_pos_weight,
                inner_folds, outer_splits, outer_repeats, n_iter):
    rows = []
    for label, epv_max in epv_caps.items():
        X_sel = select_features(X_raw.copy(), y, epv_max, c_values)
        n_features = X_sel.shape[1]
        n_events = int(y.sum())
        print(f"\n[{cohort_name} / {label}, EPV_MAX={epv_max}] "
              f"-> {n_features} features selected, EPV={n_events/n_features:.1f}")
        print(f"   Features: {list(X_sel.columns)}")
        res = tune_and_cv(X_sel, y, scale_pos_weight, inner_folds, outer_splits, outer_repeats, n_iter)
        for model, m in res.items():
            rows.append({"Cohort": cohort_name, "Cap": label, "EPV_MAX": epv_max,
                         "N features": n_features, "Model": model, **m})
            print(f"   {model:<20} AUROC={m['CV AUROC']:.3f}  Brier={m['CV Brier']:.3f}  "
                  f"CalSlope={m['CV Cal Slope']:.3f}")
    return rows


all_rows = []

# --- 1-Year flare ---
OUTCOME_COL = "flare_1yr"
ID_COL = "Lizzy Biopsy Database ID number (PLEASE KEEP THIS COLUMN FOR REFERENCE)"
df = pd.read_excel(f"{PROC}/lupus_1yr_imputed.xlsx")
y = df[OUTCOME_COL].copy()
n_events = int(y.sum())

named_leakage_kw = [
    "creatinine at one year", "creat/0.9 if male and 0.7 female.1", "minimum of creat/k and 1.1",
    "maximum if creat/k and 1.1", "ckd epi formula without ethnicity.1", "ckd epi with ethnicity.1",
    "change in proteinuria", "change in egfr", "time to pr", "non response", "cr or pr=1",
    "time from response", "***flare", "creatinine at latest follow up", "creat/0.9 if male and 0.7 female.2",
    "minimum of creat/k and 1.2", "maximum if creat/k and 1.2", "ckd epi formula without ethnicity.2",
    "ckd epi with ethnicity.2", "age at date of creatinine", "creat most recent", "rrt no=0",
    "type rrt", "death no=0", "doubling of baseline creatinine", "creatinine at date of doubling",
    "creat at ckd", "egfr<80", "egfr <80", "creatinine when egfr", "creatinine persistently",
    "egfr<70", "egfr <70", "ckd, (3a or more", "time to rrt", "time to death", "time to doubling",
    "time to egfr", "outcomes at", "long term outcome", "complete remission", "partial remission",
    "date of cr", "date of pr", "time to cr", "date lost", "date last follow", "date 5 yrs",
    "date 10 years", "date 1yr", "date 5yrs", "date 10yrs",
]
leakage_cols = [c for c in df.columns if any(kw in c.lower() for kw in named_leakage_kw)]
always_drop = [ID_COL, OUTCOME_COL]
date_cols = df.select_dtypes(include=["datetime64"]).columns.tolist()
text_cols = df.select_dtypes(include=["object"]).columns.tolist()
X = df.drop(columns=[c for c in always_drop + leakage_cols + date_cols + text_cols if c in df.columns])
X = X.select_dtypes(include="number")
X = X.drop(columns=X.columns[X.isnull().any()].tolist())
print(f"1-Year flare: {X.shape[1]} baseline numeric candidate features, {n_events} events")

C_values_1yr = [1.0, 0.5, 0.2, 0.1, 0.05, 0.02, 0.01, 0.005, 0.002, 0.001]
epv_caps_1yr = {"EPV-10 (baseline, current pipeline cap)": 12, "EPV-5 (relaxed)": 19}
all_rows += run_cohort("1-Year flare", X, y, epv_caps_1yr, C_values_1yr,
                        scale_pos_weight=3.34, inner_folds=5, outer_splits=10, outer_repeats=5, n_iter=40)

# --- Serial biopsy ---
OUTCOME_COL = "flare_5yr"
DROP_META = ["patient_id", "n_biopsies_in_cohort", "biopsy_number_recent",
             "biopsy_date_recent", "biopsy_date_previous"]
df = pd.read_excel(f"{PROC}/lupus_5yr_serial_dataset.xlsx")
y = df[OUTCOME_COL].copy()
n_events = int(y.sum())
X = df.drop(columns=[c for c in DROP_META + [OUTCOME_COL] if c in df.columns])
X = X.select_dtypes(include="number")
missing_pct = X.isnull().mean() * 100
X = X.drop(columns=missing_pct[missing_pct > 50].index.tolist())
cols_to_impute = X.columns[X.isnull().any()].tolist()
if cols_to_impute:
    imputer = IterativeImputer(random_state=42, max_iter=10, verbose=0)
    X = pd.DataFrame(imputer.fit_transform(X), columns=X.columns, index=X.index)
print(f"\nSerial biopsy: {X.shape[1]} baseline numeric candidate features, {n_events} events")

C_values_serial = [1.0, 0.5, 0.2, 0.1, 0.05, 0.02, 0.01, 0.005, 0.002, 0.001, 0.0005, 0.0001]
epv_caps_serial = {"EPV-10 (baseline, current pipeline cap)": 4, "EPV-5 (relaxed)": 7}
all_rows += run_cohort("Serial biopsy", X, y, epv_caps_serial, C_values_serial,
                        scale_pos_weight=round(36 / 34, 2), inner_folds=3, outer_splits=5, outer_repeats=5, n_iter=20)

df_out = pd.DataFrame(all_rows)
print("\n\n" + "=" * 100)
print(df_out.to_string(index=False))
out_path = f"{OUT}/epv5_sensitivity_results.xlsx"
df_out.to_excel(out_path, index=False)
print(f"\nSaved: {out_path}")
