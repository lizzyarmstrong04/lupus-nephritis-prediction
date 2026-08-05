import pandas as pd
import numpy as np
from sklearn.experimental import enable_iterative_imputer  # noqa
from sklearn.impute import IterativeImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from statsmodels.stats.outliers_influence import variance_inflation_factor
import warnings
warnings.filterwarnings("ignore")

PROCESSED_DIR = "/Users/elizabetharmstrong/Library/CloudStorage/OneDrive-ImperialCollegeLondon/Lupus Project/Data/Processed"
OUTCOME_COL   = "flare_5yr"
EPV_MAX       = 4    # 34 events / 10 = 3.4 → cap at 4

# Columns that are identifiers/dates, not predictors
DROP_META = ["patient_id", "n_biopsies_in_cohort", "biopsy_number_recent",
             "biopsy_date_recent", "biopsy_date_previous"]

df = pd.read_excel(f"{PROCESSED_DIR}/lupus_5yr_serial_dataset.xlsx")
print(f"Loaded: {df.shape[0]} rows, {df.shape[1]} columns")

y = df[OUTCOME_COL].copy()
n_events = int(y.sum())
print(f"Outcome events: {n_events} / {len(y)} ({y.mean()*100:.1f}%)")
print(f"EPV rule: {n_events} events → hard cap = {EPV_MAX} predictors\n")

X = df.drop(columns=[c for c in DROP_META + [OUTCOME_COL] if c in df.columns])
X = X.select_dtypes(include="number")
print(f"Features after dropping metadata: {X.shape[1]}")

# ============================================================
# Missingness analysis
# ============================================================
missing_pct  = X.isnull().mean() * 100
print(f"\n--- MISSINGNESS SUMMARY ---")
print(f"  No missing:      {(missing_pct == 0).sum()} columns")
print(f"  <5% missing:     {((missing_pct > 0) & (missing_pct < 5)).sum()} columns")
print(f"  5-50% missing:   {((missing_pct >= 5) & (missing_pct <= 50)).sum()} columns")
print(f"  >50% missing:    {(missing_pct > 50).sum()} columns")
cols_with_missing = missing_pct[missing_pct > 0].sort_values(ascending=False)
if len(cols_with_missing):
    print(f"\n  Columns with missing values:")
    for c, pct in cols_with_missing.items():
        print(f"    {pct:.1f}%  {c[:80]}")

# Drop >50% missing (unusable even after imputation)
cols_over_50 = missing_pct[missing_pct > 50].index.tolist()
if cols_over_50:
    X = X.drop(columns=cols_over_50)
    print(f"\nDropped {len(cols_over_50)} columns >50% missing")

# ============================================================
# MICE imputation (any column with >0% missing)
# ============================================================
cols_to_impute = missing_pct[(missing_pct > 0) & (missing_pct <= 50)].index.tolist()
cols_complete  = missing_pct[missing_pct == 0].index.tolist()

if cols_to_impute:
    print(f"\n--- MICE IMPUTATION ({len(cols_to_impute)} columns) ---")
    imputer = IterativeImputer(random_state=42, max_iter=10, verbose=0)
    imputed = imputer.fit_transform(X)
    X       = pd.DataFrame(imputed, columns=X.columns, index=X.index)
    print(f"  After imputation: {X.isnull().sum().sum()} missing values remaining")
else:
    print("\nNo imputation needed — all columns complete.")

removal_log = {}

# ============================================================
# Step 3a: Dominant binary removal (>90%)
# ============================================================
binary_cols     = [c for c in X.columns if X[c].dropna().isin([0, 1]).all()]
dominant_binary = []
for c in binary_cols:
    max_freq = X[c].value_counts(normalize=True).max()
    if max_freq > 0.90:
        dominant_binary.append((c, round(max_freq * 100, 1)))
        removal_log[c] = f"Step 3a — Dominant binary ({max_freq*100:.1f}%)"
X = X.drop(columns=[c for c, _ in dominant_binary])
print(f"\n--- STEP 3a: Dominant binary (>90%) — removed {len(dominant_binary)} ---")
for c, pct in dominant_binary:
    print(f"  [-] {c[:80]}  ({pct}%)")

# ============================================================
# Step 3b: Low variance (<0.01)
# ============================================================
variances = X.var()
low_var   = variances[variances < 0.01].index.tolist()
for c in low_var:
    removal_log[c] = f"Step 3b — Low variance (var={variances[c]:.5f})"
X = X.drop(columns=low_var)
print(f"\n--- STEP 3b: Low variance (<0.01) — removed {len(low_var)} ---")
for c in low_var:
    print(f"  [-] {c[:80]}  (var={variances[c]:.5f})")

# ============================================================
# Step 4: High correlation (r > 0.8)
# ============================================================
print(f"\n--- STEP 4: High correlation (r > 0.8) ---")
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
            removal_log[col] = f"Step 4 — High corr (r={r} with '{partners[0][:40]}')"
            X = X.drop(columns=[col])
            changed = True
            break
print(f"Removed {len(corr_removed)} features  →  {X.shape[1]} remaining")
for removed, kept, r in corr_removed:
    print(f"  [-] {removed[:70]}  (r={r} with '{kept[:50]}')")

# ============================================================
# Step 5: High VIF (>10)
# ============================================================
def calculate_vif(X_df):
    scaler = StandardScaler()
    Xs = pd.DataFrame(scaler.fit_transform(X_df), columns=X_df.columns)
    vifs = [variance_inflation_factor(Xs.values, i) for i in range(Xs.shape[1])]
    return pd.DataFrame({"Feature": X_df.columns, "VIF": vifs}).sort_values("VIF", ascending=False)

print(f"\n--- STEP 5: High VIF (>10) ---")
vif_removed = []
while True:
    vif_df = calculate_vif(X)
    worst  = vif_df.iloc[0]
    if worst["VIF"] > 10:
        col = worst["Feature"]
        removal_log[col] = f"Step 5 — High VIF ({worst['VIF']:.2f})"
        vif_removed.append((col, round(worst["VIF"], 2)))
        X = X.drop(columns=[col])
    else:
        break
print(f"Removed {len(vif_removed)} features  →  {X.shape[1]} remaining")
for col, vif in vif_removed:
    print(f"  [-] {col[:80]}  (VIF={vif})")
vif_final = calculate_vif(X)
print(f"\nMax VIF after step 5: {vif_final['VIF'].max():.2f}")

# ============================================================
# Step 6: LASSO — hard cap EPV_MAX=4
# ============================================================
print(f"\n--- STEP 6: LASSO (hard cap ≤{EPV_MAX} predictors) ---")
print(f"  Features entering LASSO: {X.shape[1]}")
print(f"  ⚠  EPV constraint: {n_events} events → max {EPV_MAX} predictors (EPV-10 rule)")

scaler   = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Search over decreasing C until ≤ EPV_MAX features remain
C_values = [1.0, 0.5, 0.2, 0.1, 0.05, 0.02, 0.01, 0.005, 0.002, 0.001, 0.0005, 0.0001]
chosen_C, chosen_coefs = None, None

print(f"\n  C search:")
for C in C_values:
    model = LogisticRegression(penalty="l1", solver="saga", C=C,
                               max_iter=20000, random_state=42)
    model.fit(X_scaled, y)
    n_nonzero = int(np.sum(model.coef_[0] != 0))
    print(f"    C={C:<8}  non-zero features={n_nonzero}")
    if n_nonzero <= EPV_MAX and chosen_C is None:
        chosen_C    = C
        chosen_coefs = model.coef_[0]

if chosen_C is None:
    chosen_C = C_values[-1]
    model = LogisticRegression(penalty="l1", solver="saga", C=chosen_C,
                               max_iter=20000, random_state=42)
    model.fit(X_scaled, y)
    chosen_coefs = model.coef_[0]

coef_df = pd.DataFrame({"Feature": X.columns, "Coefficient": chosen_coefs})
coef_df = coef_df.sort_values("Coefficient", key=abs, ascending=False)

zeroed = coef_df[coef_df["Coefficient"] == 0]["Feature"].tolist()
for c in zeroed:
    removal_log[c] = f"Step 6 — LASSO zeroed (C={chosen_C})"

final_features = coef_df[coef_df["Coefficient"] != 0]["Feature"].tolist()
X = X[final_features]

print(f"\n  Selected C={chosen_C}  →  {len(final_features)} features retained")
print(f"\n  LASSO retained features (sorted by |coefficient|):")
print(coef_df[coef_df["Coefficient"] != 0].to_string(index=False))

# ============================================================
# Final summary
# ============================================================
print(f"\n{'='*65}")
print(f"FINAL SELECTED FEATURES  ({X.shape[1]} predictors / {n_events} events)")
print(f"EPV = {n_events}/{X.shape[1]} = {n_events/X.shape[1]:.1f}")
print(f"{'='*65}")
for col in X.columns:
    print(f"  {col}")

df_out = pd.concat([y.reset_index(drop=True), X.reset_index(drop=True)], axis=1)
output_path = f"{PROCESSED_DIR}/lupus_5yr_serial_selected.xlsx"
df_out.to_excel(output_path, index=False)
print(f"\nSaved: {output_path}")
print(f"Shape: {df_out.shape[0]} rows × {df_out.shape[1]} columns")
