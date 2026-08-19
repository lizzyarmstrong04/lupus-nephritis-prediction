import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from statsmodels.stats.outliers_influence import variance_inflation_factor
import warnings
warnings.filterwarnings("ignore")

PROCESSED_DIR = "/Users/elizabetharmstrong/Library/CloudStorage/OneDrive-ImperialCollegeLondon/Lupus Project/Data/Processed"
OUTCOME_COL   = "flare_5yr"
ID_COL        = "Lizzy Biopsy Database ID number (PLEASE KEEP THIS COLUMN FOR REFERENCE)"
EPV_MAX       = 15   # 166 events / 10 = 16.6; cap at 15

df = pd.read_excel(f"{PROCESSED_DIR}/lupus_5yr_imputed.xlsx")
print(f"Loaded: {df.shape[0]} rows, {df.shape[1]} columns")

y        = df[OUTCOME_COL].copy()
n_events = int(y.sum())
print(f"Outcome events: {n_events}  →  EPV hard cap: {EPV_MAX} predictors")

# 1. Leakage removal

always_drop = [ID_COL, OUTCOME_COL]

named_leakage = [
    # 1-year creatinine and derived
    "Creatinine at one year in mg/dl (/88.42 conversion factor)",
    "Creat/0.9 if male and 0.7 if female.1",
    "Minimum of Creat/k and 1.1",
    "Maximum if Creat/k and 1.1",
    "CKD epi formula without ethnicity.1",
    "CKD epi with ethnicity.1",
    # Change from baseline to 1yr
    "Change in proteinuria baseline to 1yr",
    "Change in eGFR CKDEPI baseline to 1yr",
    # Treatment response outcomes
    "Time to PR (months)",
    "Non Response 1=NR 2=technically PR but NR pattern",
    "CR or PR=1, NR=0",
    "Time from response to flare (months)",
    # Flare definition column
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
    # Time-to-event from diagnosis
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
    # Outcomes at 1/5/10 years
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
    # 1-year flare outcome (post-biopsy; leakage for 5-year prediction)
    "flare_1yr",
]

LEAKAGE_KEYWORDS = [
    "complete remission", "partial remission", "date of cr", "date of pr",
    "time to cr", "time to pr", "date lost", "date last follow",
    "date 5 yrs", "date 10 years", "date 1yr", "date 5yrs", "date 10yrs",
    "flare by 1 year", "proteinuria 1year", "on pred one year",
    "on pred six months", "dose at one year", "dose at six months",
]

keyword_leakage = [
    c for c in df.columns
    if any(kw in c.lower() for kw in LEAKAGE_KEYWORDS)
    and c not in named_leakage + always_drop
]

all_leakage = [c for c in named_leakage if c in df.columns] + keyword_leakage
print(f"\n--- STEP 1: Leakage columns dropped ({len(all_leakage)}) ---")
for c in all_leakage:
    print(f"  [LEAKAGE] {c[:100]}")

drop_all  = always_drop + all_leakage
date_cols = df.select_dtypes(include=["datetime64"]).columns.tolist()
text_cols = df.select_dtypes(include=["object"]).columns.tolist()

X = df.drop(columns=[c for c in drop_all + date_cols + text_cols if c in df.columns])
X = X.select_dtypes(include="number")

incomplete = X.columns[X.isnull().any()].tolist()
X = X.drop(columns=incomplete)
print(f"\nAlso dropped {len(incomplete)} columns with remaining missing values (>50%)")
print(f"  → {X.shape[1]} baseline numeric features to evaluate")

removal_log = {}

# 3a. Dominant binary removal (one class > 90%)

binary_cols     = [c for c in X.columns if X[c].dropna().isin([0, 1]).all()]
dominant_binary = []
for c in binary_cols:
    max_freq = X[c].value_counts(normalize=True).max()
    if max_freq > 0.90:
        dominant_binary.append((c, round(max_freq * 100, 1)))
        removal_log[c] = f"Step 3a — Dominant binary ({max_freq*100:.1f}%)"

X = X.drop(columns=[c for c, _ in dominant_binary])
print(f"\n--- STEP 3a: Dominant binary removal (one class > 90%) ---")
print(f"Removed {len(dominant_binary)} features  →  {X.shape[1]} remaining")
for c, pct in dominant_binary:
    print(f"  [-] {c}  ({pct}%)")

# 3b. Low variance (< 0.01)

variances = X.var()
low_var   = variances[variances < 0.01].index.tolist()
for c in low_var:
    removal_log[c] = f"Step 3b — Low variance (var={variances[c]:.5f})"
X = X.drop(columns=low_var)
print(f"\n--- STEP 3b: Low-variance removal (var < 0.01) ---")
print(f"Removed {len(low_var)} features  →  {X.shape[1]} remaining")
for c in low_var:
    print(f"  [-] {c}  (var={variances[c]:.5f})")

# 4. High correlation (> 0.8)

print(f"\n--- STEP 4: High correlation removal (r > 0.8) ---")
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
            removal_log[col] = f"Step 4 — High correlation (r={r} with '{partners[0]}')"
            X = X.drop(columns=[col])
            changed = True
            break

print(f"Removed {len(corr_removed)} features  →  {X.shape[1]} remaining")
for removed, kept, r in corr_removed:
    print(f"  [-] {removed[:80]}  (r={r} with '{kept[:60]}')")

# 5. High VIF (> 10)

def calculate_vif(X_df):
    scaler = StandardScaler()
    Xs = pd.DataFrame(scaler.fit_transform(X_df), columns=X_df.columns)
    vifs = [variance_inflation_factor(Xs.values, i) for i in range(Xs.shape[1])]
    return pd.DataFrame({"Feature": X_df.columns, "VIF": vifs}).sort_values("VIF", ascending=False)

print(f"\n--- STEP 5: High VIF removal (VIF > 10) ---")
vif_removed = []
while True:
    vif_df = calculate_vif(X)
    worst  = vif_df.iloc[0]
    if worst["VIF"] > 10:
        col = worst["Feature"]
        removal_log[col] = f"Step 5 — High VIF (VIF={worst['VIF']:.2f})"
        vif_removed.append((col, round(worst["VIF"], 2)))
        X = X.drop(columns=[col])
    else:
        break

print(f"Removed {len(vif_removed)} features  →  {X.shape[1]} remaining")
for col, vif in vif_removed:
    print(f"  [-] {col[:80]}  (VIF={vif})")

vif_final = calculate_vif(X)
print(f"\nFinal VIF values after step 5:")
print(vif_final.to_string(index=False))

# 6. LASSO with hard cap at EPV_MAX

print(f"\n--- STEP 6: LASSO (hard cap {EPV_MAX} predictors) ---")
print(f"Features entering LASSO: {X.shape[1]}")

scaler   = StandardScaler()
X_scaled = scaler.fit_transform(X)

lasso_removed  = []
final_features = list(X.columns)

if X.shape[1] > EPV_MAX:
    C_values = [1.0, 0.5, 0.2, 0.1, 0.05, 0.02, 0.01, 0.005, 0.002, 0.001]
    chosen_C, chosen_coefs = None, None

    for C in C_values:
        model = LogisticRegression(penalty="l1", solver="saga", C=C,
                                   max_iter=10000, random_state=42)
        model.fit(X_scaled, y)
        if np.sum(model.coef_[0] != 0) <= EPV_MAX:
            chosen_C    = C
            chosen_coefs = model.coef_[0]
            break

    if chosen_C is None:
        chosen_C = C_values[-1]
        model = LogisticRegression(penalty="l1", solver="saga", C=chosen_C,
                                   max_iter=10000, random_state=42)
        model.fit(X_scaled, y)
        chosen_coefs = model.coef_[0]

    print(f"LASSO regularisation: C={chosen_C}")
    coef_df = pd.DataFrame({"Feature": X.columns, "Coefficient": chosen_coefs})
    coef_df = coef_df.sort_values("Coefficient", key=abs, ascending=False)

    zeroed = coef_df[coef_df["Coefficient"] == 0]["Feature"].tolist()
    for c in zeroed:
        removal_log[c] = f"Step 6 — LASSO zeroed (C={chosen_C})"
        lasso_removed.append(c)

    final_features = coef_df[coef_df["Coefficient"] != 0]["Feature"].tolist()
    X = X[final_features]

    print(f"Removed {len(lasso_removed)} features  →  {X.shape[1]} remaining")
    for c in lasso_removed:
        print(f"  [-] {c[:80]}")
    print(f"\nLASSO retained features (sorted by |coefficient|):")
    print(coef_df[coef_df["Coefficient"] != 0].to_string(index=False))
else:
    print(f"{X.shape[1]} features ≤ {EPV_MAX}  →  LASSO not needed")

# 7. Summary

print(f"\n{'='*65}")
print(f"FINAL SELECTED FEATURES  ({X.shape[1]} predictors / {n_events} events)")
print(f"{'='*65}")
for col in X.columns:
    print(f"  {col}")

print(f"\n--- Removal log (all steps) ---")
for col, reason in removal_log.items():
    print(f"  {reason[:30]}  |  {col[:80]}")

df_out = pd.concat([y.reset_index(drop=True), X.reset_index(drop=True)], axis=1)
output_path = f"{PROCESSED_DIR}/lupus_5yr_selected_clean.xlsx"
df_out.to_excel(output_path, index=False)
print(f"\nSaved: {output_path}")
print(f"Final shape: {df_out.shape[0]} rows × {df_out.shape[1]} columns")
