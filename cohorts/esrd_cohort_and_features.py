"""
ESRD feature selection — same 6-step pipeline as flare prediction:
  Step 1  — leakage removal (post-biopsy columns)
  Step 3a — dominant binary removal (one class > 90%)
  Step 3b — low-variance removal (var < 0.01)
  Step 4  — high-correlation removal (r > 0.80)
  Step 5  — VIF removal (VIF > 10)
  Step 6  — LASSO with EPV-10 hard cap

Runs separately for 5-year (EPV_MAX=11) and 10-year (EPV_MAX=17).
Saves selected datasets to Data/Processed/.
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.experimental import enable_iterative_imputer  # noqa
from sklearn.impute import IterativeImputer
from statsmodels.stats.outliers_influence import variance_inflation_factor
import warnings
warnings.filterwarnings("ignore")

RAW_PATH      = "/Users/elizabetharmstrong/Library/CloudStorage/OneDrive-ImperialCollegeLondon/Lupus Project/Data/Raw/data_lupus.xlsx"
PROCESSED_DIR = "/Users/elizabetharmstrong/Library/CloudStorage/OneDrive-ImperialCollegeLondon/Lupus Project/Data/Processed"

# Raw outcome column names
OUTCOME_5YR  = (
    "Outcomes at 5yrs from first Bx in this db 1=eGFR>80 2=doubling baseline creatinine "
    "3=RRT 4=Death 5=Death on RRT 6=eGFR<80 but not doubling creatinine, RRT or death, "
    "7=eGFR<70 but not doubling creatinine, RRT or death 8=eGFR<60 but not doubling "
    "creatinine, RRT "
)
OUTCOME_10YR = (
    "Outcomes at 10yrs from first Bx in this db 1=eGFR>80 2=doubling baseline creatinine "
    "3=RRT 4=Death 5=Death on RRT 6=eGFR<80 but not doubling creatinine, RRT or death, "
    "7=eGFR<70 but not doubling creatinine, RRT or death 8=eGFR<60 but not doubling "
    "creatinine, RRT "
)
ESRD_CODES = [2, 3, 5]   # creatinine doubling, RRT, death on RRT

# Leakage list  (same as flare analysis + flare outcomes)

NAMED_LEAKAGE = [
    # 1-year creatinine and derived columns
    "Creatinine at one year in mg/dl (/88.42 conversion factor)",
    "Creat/0.9 if male and 0.7 if female.1",
    "Minimum of Creat/k and 1.1",
    "Maximum if Creat/k and 1.1",
    "CKD epi formula without ethnicity.1",
    "CKD epi with ethnicity.1",
    # Change from baseline to 1yr
    "Change in proteinuria baseline to 1yr",
    "Change in eGFR CKDEPI baseline to 1yr",
    # Treatment response outcomes (post-biopsy)
    "Time to PR (months)",
    "Non Response 1=NR 2=technically PR but NR pattern",
    "CR or PR=1, NR=0",
    "Time from response to flare (months)",
    # Flare outcome column
    "***Flare (proteinuria >50% increase and >100mg/ml for 2 consecutive visits, "
    "+/or fall >20% GFR on 2 occasions from normal renal function. N/A for NR and "
    "response N/A as you can only flare after responding to a specific treatment",
    # Latest follow-up creatinine
    "Creatinine at latest follow up in mg/dl (/88.42 conversion factor)",
    "Creat/0.9 if male and 0.7 if female.2",
    "Minimum of Creat/k and 1.2",
    "Maximum if Creat/k and 1.2",
    "CKD epi formula without ethnicity.2",
    "CKD epi with ethnicity.2",
    "CKD epi with ethnicity quoted as 90 if >/=90",
    "Age at date of creatinine for CKD EPI formula",
    "Creat most recent 06.11.2018 or if on RRT, creatinine before RRT",
    # Long-term renal/survival outcomes
    "RRT No=0, Yes=1",
    "Type RRT (1=dialysis, 2=Tx)",
    "Death No=0, Yes=1",
    "Doubling of baseline creatinine by latest follow-up",
    "Creatinine at date of doubling creatinine",
    "Creat at CKD",
    "eGFR<80 (2 consecutive readings and persistent)",
    "Creatinine when eGFR <80",
    "Creatinine persistently greater than (date and effect on eGFR taken into account)",
    "eGFR <70 (2 consecutive readings and persistent)",
    "Creatinine when eGFR <70",
    "Creatinine persistently greater than (date and effect on eGFR taken into account).1",
    "CKD, (3A or more, 2 consecutive readings </=59 and not on RRT (yet) and persistent) No=0, Yes=1.  "
    "Looked at trends, if creatinine deteriorated but then improved, did not count as CKD, took date "
    "after which eGFR consistently lower than 60, 70 or 80, even if it improved slightly, so creatinine "
    "given could be worse than 60, 70 or 80 however it will be the creat on the date listed for the "
    "eGFR<X after which it never got better than X.",
    # Time-to-event outcomes from diagnosis and from biopsy
    "NOTE: IF TIME 1, NOT HAPPENED BY ONE YEAR,0=BY 1YR, 1-4=BY 5YR, 5-9=BY 10YR. "
    "Time to RRT from diagnosis LN",
    "Time to death from diagnosis LN",
    "NOTE NOT A VALID OUTCOME UNLESS BASELINE CREAT IS SAME DATE AS DIAGNOSIS LN.  WILL NEED TO "
    "EXCLUDE FROM THIS ANALYSIS IN THESE PTS. Time to doubling creatinine from diagnosis LN",
    "Time to eGFR <80 from diagnosis LN",
    "Time to eGFR <70 from diagnosis LN",
    "Time to eGFR <60 / CKD from diagnosis LN",
    "Time to RRT from first Bx",
    "Time to death from first Bx",
    "Time to doubling creatinine from first Bx",
    "Time to eGFR <80 from first Bx (N/A could be a negative time)",
    "Time to eGFR <70 from first Bx (N/A could be a negative time)",
    "Time to eGFR <60 / CKD from first Bx (N/A could be a negative time)",
    # All outcome-at-X-years columns
    "Long term outcome @ latest fup. 1=eGFR>80 2=doubling baseline creatinine 3=RRT 4=Death alone "
    "5=Death on RRT 6=eGFR <80 but not doubling creatinine or RRT or death 7=eGFR<70 but not "
    "doubling creatinine, RRT or death 8=eGFR<60 but not doubling creatinine, RRT or death, "
    "X=not enough follow-up info",
    "NOTE: THESE CODES TAKE LAST KNOWN OUTCOME IF FOLLOW UP <1,5 OR 10YRS.  NEED TO ORDER BY "
    "FOLLOW-UP >1, 5 OR 10YRS AND AMEND.  Outcomes at 10years from LN diagnosis 1=eGFR>80 "
    "2=doubling baseline creatinine 3=RRT 4=Death 5=Death on RRT 6=eGFR<80 but not doubling "
    "creatinine, RRT or death 7=eGFR<70 but not doubling creatinine, RRT or death 8=eGFR<60 "
    "but not doubling creatinine, RRT or death",
    "Outcomes at 5 years from LN diagnosis 1=eGFR>80 2=doubling baseline creatinine 3=RRT 4=Death "
    "5=Death on RRT 6=eGFR<80 but not doubling creatinine, RRT or death 7=eGFR<70 but not doubling "
    "creatinine, RRT or death 8=eGFR<60 but not doubling creatinine, RRT or death",
    "Outcomes at 1 year from LN diagnosis 1=eGFR>80 2=doubling baseline creatinine 3=RRT 4=Death "
    "5=Death on RRT 6=eGFR<80 but not doubling creatinine, RRT or death 7=eGFR<70 but not doubling "
    "creatinine, RRT or death 8=eGFR<60 but not doubling creatinine, RRT or death",
    "Outcomes at 10yrs from first Bx in this db 1=eGFR>80 2=doubling baseline creatinine 3=RRT "
    "4=Death 5=Death on RRT 6=eGFR<80 but not doubling creatinine, RRT or death, 7=eGFR<70 but "
    "not doubling creatinine, RRT or death 8=eGFR<60 but not doubling creatinine, RRT ",
    "Outcomes at 5yrs from first Bx in this db 1=eGFR>80 2=doubling baseline creatinine 3=RRT "
    "4=Death 5=Death on RRT 6=eGFR<80 but not doubling creatinine, RRT or death, 7=eGFR<70 but "
    "not doubling creatinine, RRT or death 8=eGFR<60 but not doubling creatinine, RRT ",
    "Outcomes at 1yrs from first Bx in this db 1=eGFR>80 2=doubling baseline creatinine 3=RRT "
    "4=Death 5=Death on RRT 6=eGFR<80 but not doubling creatinine, RRT or death, 7=eGFR<70 but "
    "not doubling creatinine, RRT or death 8=eGFR<60 but not doubling creatinine, RRT ",
    # X-coded (amended) versions
    "NOTE: THESE CODES HAVE X FOR OUTCOME IF FOLLOW-UP IS LESS THAN THE NUMBER OF YEARS EXCEPT "
    "FOR DEATH PRE 1/5/10YRS..  Outcomes at 10years from LN diagnosis 1=eGFR>80 2=doubling "
    "baseline creatinine 3=RRT 4=Death 5=Death on RRT 6=eGFR<80 but not doubling creatinine, "
    "RRT or death 7=eGFR<70 but not doubling creatinine, RRT or death 8=eGFR<60 but not "
    "doubling creatinine, RRT or death",
    "Outcomes at 5 years from LN diagnosis 1=eGFR>80 2=doubling baseline creatinine 3=RRT 4=Death "
    "5=Death on RRT 6=eGFR<80 but not doubling creatinine, RRT or death 7=eGFR<70 but not doubling "
    "creatinine, RRT or death 8=eGFR<60 but not doubling creatinine, RRT or death.1",
    "Outcomes at 1 year from LN diagnosis 1=eGFR>80 2=doubling baseline creatinine 3=RRT 4=Death "
    "5=Death on RRT 6=eGFR<80 but not doubling creatinine, RRT or death 7=eGFR<70 but not doubling "
    "creatinine, RRT or death 8=eGFR<60 but not doubling creatinine, RRT or death.1",
    "Outcomes at 10yrs from first Bx in this db 1=eGFR>80 2=doubling baseline creatinine 3=RRT "
    "4=Death 5=Death on RRT 6=eGFR<80 but not doubling creatinine, RRT or death, 7=eGFR<70 but "
    "not doubling creatinine, RRT or death 8=eGFR<60 but not doubling creatinine, RRT .1",
    "Outcomes at 5yrs from first Bx in this db 1=eGFR>80 2=doubling baseline creatinine 3=RRT "
    "4=Death 5=Death on RRT 6=eGFR<80 but not doubling creatinine, RRT or death, 7=eGFR<70 but "
    "not doubling creatinine, RRT or death 8=eGFR<60 but not doubling creatinine, RRT .1",
    "Outcomes at 1yrs from first Bx in this db 1=eGFR>80 2=doubling baseline creatinine 3=RRT "
    "4=Death 5=Death on RRT 6=eGFR<80 but not doubling creatinine, RRT or death, 7=eGFR<70 but "
    "not doubling creatinine, RRT or death 8=eGFR<60 but not doubling creatinine, RRT .1",
]

LEAKAGE_KEYWORDS = [
    "complete remission", "partial remission", "date of cr", "date of pr",
    "time to cr", "time to pr", "date lost", "date last follow",
    "date 5 yrs", "date 10 years", "date 1yr", "date 5yrs", "date 10yrs",
    "flare by 1 year", "flare_1yr", "flare_5yr",
    "1 year proteinuria", "1yr proteinuria", "steroid dose at 1",
    # 1-year follow-up measurements (post-baseline — leakage for ESRD)
    "creatinine 1 year",   # "Creatinine 1 year"
    "proteinuria 1year",   # "Proteinuria 1year"
    "egfr at 1 year",      # "eGFR at 1 year from records"
    "at 1 year from",      # catch-all for "X at 1 year from ..."
    "1year",               # generic catch (e.g. "Proteinuria 1year")
    "cr=1,pr=2",           # treatment response coded column ("CR=1,PR=2,NR=3,N/A=4")
    "response n/a",        # treatment response N/A column
    # Latest follow-up creatinine (different column name variant not in named list)
    "at last follow",      # "Creatinine at last follow up"
    "creat at last",
    # Post-biopsy treatment decisions (not valid baseline predictors)
    "post bx",             # "Oral steroids post Bx", "Rx upgraded post-biopsy"
    "post biopsy",
    "post-biopsy",
    # Technical staining method — not a clinical predictor
    "immunostaining",
    "if or immuno",
]

# ID column — must always be excluded (not a predictor)
ID_COL = "Lizzy Biopsy Database ID number (PLEASE KEEP THIS COLUMN FOR REFERENCE)"

# Helper functions

def calculate_vif(X_df):
    scaler = StandardScaler()
    Xs = pd.DataFrame(scaler.fit_transform(X_df), columns=X_df.columns)
    vifs = [variance_inflation_factor(Xs.values, i) for i in range(Xs.shape[1])]
    return pd.DataFrame({"Feature": X_df.columns, "VIF": vifs}).sort_values("VIF", ascending=False)

def run_feature_selection(df_cohort, y, outcome_label, epv_max, outcome_raw_col):
    """
    Apply 6-step feature selection to df_cohort.
    y     : binary outcome series (aligned with df_cohort index)
    Returns: selected feature DataFrame
    """
    print(f"\n{'='*70}")
    print(f"ESRD FEATURE SELECTION — {outcome_label}")
    print(f"  n={len(y)}  events={int(y.sum())} ({y.mean()*100:.1f}%)  EPV_MAX={epv_max}")
    print(f"{'='*70}")

    # Step 1: Remove leakage columns
    # Also drop both outcome raw columns (the current and the other horizon)
    both_outcomes = [OUTCOME_5YR, OUTCOME_10YR]
    always_drop   = [ID_COL]
    named_to_drop = [c for c in NAMED_LEAKAGE + both_outcomes + always_drop
                     if c in df_cohort.columns]

    keyword_to_drop = [
        c for c in df_cohort.columns
        if any(kw in c.lower() for kw in LEAKAGE_KEYWORDS)
        and c not in named_to_drop
    ]

    all_leakage = named_to_drop + keyword_to_drop

    date_cols = df_cohort.select_dtypes(include=["datetime64"]).columns.tolist()
    # Keep object columns that can be coerced to numeric; drop true text/date columns
    # (same approach as the ESRD modelling script)
    drop_non_numeric = list(set(all_leakage + date_cols))
    candidate_cols = [c for c in df_cohort.columns if c not in drop_non_numeric]

    # Coerce all candidate columns to numeric (non-convertible → NaN)
    X = pd.DataFrame(index=df_cohort.index)
    non_numeric_cols = []
    for col in candidate_cols:
        series = pd.to_numeric(df_cohort[col], errors="coerce")
        if series.notna().sum() >= 10:   # at least 10 non-missing values
            X[col] = series
        else:
            non_numeric_cols.append(col)

    print(f"\n--- STEP 1: Leakage removal ---")
    print(f"  Named leakage + ID found in data: {len(named_to_drop)}")
    print(f"  Keyword leakage:                  {len(keyword_to_drop)}")
    print(f"  Date columns:                     {len(date_cols)}")
    print(f"  Non-numeric / all-NaN:            {len(non_numeric_cols)}")
    print(f"  → {X.shape[1]} numeric columns remain")

    # MICE imputation (before feature selection, on remaining cols)
    pct_missing = X.isnull().mean()
    high_missing = pct_missing[pct_missing > 0.50].index.tolist()
    X = X.drop(columns=high_missing)
    if high_missing:
        print(f"\n  Dropped {len(high_missing)} columns with >50% missing before MICE:")
        for c in high_missing:
            print(f"    {c[:80]}  ({pct_missing[c]*100:.0f}% missing)")

    print(f"\n  Running MICE imputation on {X.shape[1]} columns...")
    imputer = IterativeImputer(random_state=42, max_iter=10)
    X_imp = pd.DataFrame(imputer.fit_transform(X), columns=X.columns, index=X.index)
    print(f"  Missing after MICE: {X_imp.isnull().sum().sum()}")

    X = X_imp.copy()
    removal_log = {}

    # Step 3a: Dominant binary removal
    binary_cols = [c for c in X.columns if set(X[c].round(0).unique()).issubset({0.0, 1.0, 0, 1})]
    dominant_binary = []
    for c in binary_cols:
        max_freq = X[c].value_counts(normalize=True).max()
        if max_freq > 0.90:
            dominant_binary.append((c, round(max_freq * 100, 1)))
            removal_log[c] = f"Step 3a — Dominant binary ({max_freq*100:.1f}%)"

    X = X.drop(columns=[c for c, _ in dominant_binary])
    print(f"\n--- STEP 3a: Dominant binary removal (one class > 90%) ---")
    print(f"Removed {len(dominant_binary)}  →  {X.shape[1]} remaining")
    for c, pct in dominant_binary:
        print(f"  [-] {c[:80]}  ({pct}%)")

    # Step 3b: Low-variance removal
    variances = X.var()
    low_var = variances[variances < 0.01].index.tolist()
    for c in low_var:
        removal_log[c] = f"Step 3b — Low variance (var={variances[c]:.5f})"
    X = X.drop(columns=low_var)
    print(f"\n--- STEP 3b: Low-variance removal (var < 0.01) ---")
    print(f"Removed {len(low_var)}  →  {X.shape[1]} remaining")
    for c in low_var:
        print(f"  [-] {c[:80]}  (var={variances[c]:.5f})")

    # Step 4: High-correlation removal (r > 0.80)
    print(f"\n--- STEP 4: High correlation removal (r > 0.80) ---")
    corr_removed = []
    changed = True
    while changed:
        changed = False
        corr_matrix = X.corr().abs()
        upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
        for col in upper.columns:
            partners = upper.index[upper[col] > 0.8].tolist()
            if partners:
                r = round(corr_matrix.loc[col, partners[0]], 3)
                corr_removed.append((col, partners[0], r))
                removal_log[col] = f"Step 4 — High corr (r={r} with '{partners[0]}')"
                X = X.drop(columns=[col])
                changed = True
                break

    print(f"Removed {len(corr_removed)}  →  {X.shape[1]} remaining")
    for removed, kept, r in corr_removed:
        print(f"  [-] {removed[:70]}  (r={r} with '{kept[:50]}')")

    # Step 5: VIF removal (VIF > 10)
    print(f"\n--- STEP 5: High VIF removal (VIF > 10) ---")
    vif_removed = []
    while True:
        vif_df = calculate_vif(X)
        worst = vif_df.iloc[0]
        if worst["VIF"] > 10:
            col = worst["Feature"]
            removal_log[col] = f"Step 5 — High VIF ({worst['VIF']:.2f})"
            vif_removed.append((col, round(worst["VIF"], 2)))
            X = X.drop(columns=[col])
        else:
            break

    print(f"Removed {len(vif_removed)}  →  {X.shape[1]} remaining")
    for col, vif in vif_removed:
        print(f"  [-] {col[:80]}  (VIF={vif})")

    vif_final = calculate_vif(X)
    print(f"\nFinal VIF values after step 5:")
    print(vif_final.to_string(index=False))

    # Step 6: LASSO with EPV cap
    n_events = int(y.sum())
    print(f"\n--- STEP 6: LASSO (hard cap EPV_MAX={epv_max}) ---")
    print(f"Features entering LASSO: {X.shape[1]}")

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    lasso_removed = []
    final_features = list(X.columns)

    if X.shape[1] > epv_max:
        C_values = [1.0, 0.5, 0.2, 0.1, 0.05, 0.02, 0.01, 0.005, 0.002, 0.001]
        chosen_C = None
        chosen_coefs = None

        print(f"  Searching C values:")
        for C in C_values:
            model = LogisticRegression(
                penalty="l1", solver="saga", C=C,
                max_iter=10000, random_state=42,
                class_weight="balanced"
            )
            model.fit(X_scaled, y)
            n_nonzero = np.sum(model.coef_[0] != 0)
            print(f"    C={C}  →  {n_nonzero} non-zero features")
            if n_nonzero <= epv_max and chosen_C is None:
                chosen_C = C
                chosen_coefs = model.coef_[0]

        if chosen_C is None:
            chosen_C = C_values[-1]
            model = LogisticRegression(
                penalty="l1", solver="saga", C=chosen_C,
                max_iter=10000, random_state=42,
                class_weight="balanced"
            )
            model.fit(X_scaled, y)
            chosen_coefs = model.coef_[0]

        print(f"\n  Selected C={chosen_C}")
        coef_df = pd.DataFrame({
            "Feature": X.columns,
            "Coefficient": chosen_coefs
        }).sort_values("Coefficient", key=abs, ascending=False)

        zeroed = coef_df[coef_df["Coefficient"] == 0]["Feature"].tolist()
        for c in zeroed:
            removal_log[c] = f"Step 6 — LASSO zeroed (C={chosen_C})"
            lasso_removed.append(c)

        final_features = coef_df[coef_df["Coefficient"] != 0]["Feature"].tolist()
        X = X[final_features]

        print(f"Removed {len(lasso_removed)}  →  {X.shape[1]} remaining")
        print(f"\nLASSO retained features (sorted by |coefficient|):")
        print(coef_df[coef_df["Coefficient"] != 0].to_string(index=False))
    else:
        print(f"{X.shape[1]} ≤ {epv_max}  →  LASSO not needed")

    # Manual correction: Age (Now) → Age at biopsy
    # "Age (Now)" = age at database compilation (~13.5 yrs post-biopsy on average).
    # Not a valid baseline predictor. Replace with "Age at biopsy" — same correction
    # applied in the 1yr and 5yr flare feature selection pipelines.
    AGE_NOW_COL = "Age (Now) "
    AGE_BX_COL  = "Age at biopsy"
    if AGE_NOW_COL in X.columns and AGE_BX_COL in df_cohort.columns:
        X = X.drop(columns=[AGE_NOW_COL])
        X.insert(0, AGE_BX_COL, df_cohort[AGE_BX_COL].values)
        removal_log[AGE_NOW_COL] = "Manual correction — not a baseline predictor"
        print(f"\n  Manual correction: '{AGE_NOW_COL}' → '{AGE_BX_COL}'")

    # Final summary
    print(f"\n{'='*70}")
    print(f"FINAL FEATURES — {outcome_label}  ({X.shape[1]} predictors / {n_events} events)")
    print(f"  EPV = {n_events} / {X.shape[1]} = {n_events / X.shape[1]:.1f}")
    print(f"{'='*70}")
    for col in X.columns:
        print(f"  {col}")

    print(f"\n--- Removal log ---")
    for col, reason in removal_log.items():
        print(f"  {reason[:35]}  |  {col[:70]}")

    return X, removal_log

# 1. Load raw data

print("Loading raw data...")
df_raw = pd.read_excel(RAW_PATH)
print(f"Raw data: {df_raw.shape[0]} rows × {df_raw.shape[1]} columns")

# 2. Run for 5-year ESRD (EPV_MAX = floor(112/10) = 11)

mask5 = df_raw[OUTCOME_5YR].notna() & ~df_raw[OUTCOME_5YR].isin(["X"])
df5   = df_raw[mask5].copy().reset_index(drop=True)
y5    = df5[OUTCOME_5YR].isin(ESRD_CODES).astype(int)
print(f"\n5-year cohort: {len(df5)} rows, {int(y5.sum())} events ({y5.mean()*100:.1f}%)")
EPV_MAX_5YR = int(y5.sum()) // 10
print(f"EPV_MAX (5yr) = floor({int(y5.sum())} / 10) = {EPV_MAX_5YR}")

X5_selected, log5 = run_feature_selection(df5, y5, "5-Year ESRD", EPV_MAX_5YR, OUTCOME_5YR)

out5 = pd.concat([y5.rename("esrd_5yr"), X5_selected], axis=1)
out5.to_excel(f"{PROCESSED_DIR}/esrd_5yr_selected.xlsx", index=False)
print(f"\nSaved: {PROCESSED_DIR}/esrd_5yr_selected.xlsx  ({out5.shape})")

# 3. Run for 10-year ESRD (EPV_MAX = floor(175/10) = 17)

mask10 = df_raw[OUTCOME_10YR].notna() & ~df_raw[OUTCOME_10YR].isin(["X"])
df10   = df_raw[mask10].copy().reset_index(drop=True)
y10    = df10[OUTCOME_10YR].isin(ESRD_CODES).astype(int)
print(f"\n10-year cohort: {len(df10)} rows, {int(y10.sum())} events ({y10.mean()*100:.1f}%)")
EPV_MAX_10YR = int(y10.sum()) // 10
print(f"EPV_MAX (10yr) = floor({int(y10.sum())} / 10) = {EPV_MAX_10YR}")

X10_selected, log10 = run_feature_selection(df10, y10, "10-Year ESRD", EPV_MAX_10YR, OUTCOME_10YR)

out10 = pd.concat([y10.rename("esrd_10yr"), X10_selected], axis=1)
out10.to_excel(f"{PROCESSED_DIR}/esrd_10yr_selected.xlsx", index=False)
print(f"\nSaved: {PROCESSED_DIR}/esrd_10yr_selected.xlsx  ({out10.shape})")

# 4. Side-by-side feature comparison

feats5  = set(X5_selected.columns)
feats10 = set(X10_selected.columns)
print(f"\n{'='*70}")
print("FEATURE COMPARISON: 5yr vs 10yr")
print(f"{'='*70}")
print(f"  5yr selected ({len(feats5)}):  {sorted(feats5)}")
print(f"  10yr selected ({len(feats10)}): {sorted(feats10)}")
print(f"  In both:  {sorted(feats5 & feats10)}")
print(f"  5yr only: {sorted(feats5 - feats10)}")
print(f"  10yr only: {sorted(feats10 - feats5)}")

print("\nDone.")
