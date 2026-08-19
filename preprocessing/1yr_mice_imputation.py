import pandas as pd
import numpy as np
from sklearn.experimental import enable_iterative_imputer  # noqa
from sklearn.impute import IterativeImputer

DATA_DIR = "/Users/elizabetharmstrong/Library/CloudStorage/OneDrive-ImperialCollegeLondon/Lupus Project/Data/Processed"
PROCESSED_DIR = "/Users/elizabetharmstrong/Library/CloudStorage/OneDrive-ImperialCollegeLondon/Lupus Project/Data/Processed"

# 1. Load dataset

df = pd.read_excel(f"{DATA_DIR}/lupus_1yr_flare_dataset.xlsx")
print(f"Loaded dataset: {df.shape[0]} rows, {df.shape[1]} columns")

# 2. Define columns to never impute

ID_COL = "Lizzy Biopsy Database ID number (PLEASE KEEP THIS COLUMN FOR REFERENCE)"
OUTCOME_COL = "flare_1yr"

# Date columns (auto-detected by dtype + name pattern)
date_cols = df.select_dtypes(include=["datetime64"]).columns.tolist()
date_cols += [c for c in df.columns if "date" in c.lower() and c not in date_cols]

# Text/free-text columns
text_cols = [c for c in df.columns if df[c].dtype == object and c not in date_cols]

# Columns to always hold out (never impute)
holdout_cols = [ID_COL, OUTCOME_COL] + date_cols + text_cols
holdout_cols = [c for c in holdout_cols if c in df.columns]

# 3. Split out holdout columns

df_holdout = df[holdout_cols].copy()
df_numeric = df.drop(columns=holdout_cols)

# 4. Categorise numeric columns by missingness

missing_pct = df_numeric.isnull().mean() * 100

cols_under_5   = missing_pct[missing_pct < 5].index.tolist()
cols_5_to_50   = missing_pct[(missing_pct >= 5) & (missing_pct <= 50)].index.tolist()
cols_over_50   = missing_pct[missing_pct > 50].index.tolist()

print(f"\nNumeric columns with <5% missing (left as-is):   {len(cols_under_5)}")
print(f"Numeric columns with 5-50% missing (MICE):       {len(cols_5_to_50)}")
print(f"Numeric columns with >50% missing (excluded):    {len(cols_over_50)}")

# 5. Before missingness summary

print("\n--- BEFORE IMPUTATION (5-50% missing columns) ---")
before = missing_pct[cols_5_to_50].sort_values(ascending=False)
for col, pct in before.items():
    print(f"  {pct:.1f}%  {col}")

# 6. Apply MICE to 5-50% missing columns only

# Fit imputer on all numeric data (so relationships are captured),
# but only replace values in the 5-50% columns
cols_to_impute = cols_under_5 + cols_5_to_50  # exclude >50% from imputation input
df_for_imputation = df_numeric[cols_to_impute].copy()

imputer = IterativeImputer(random_state=42, max_iter=10, verbose=0)
imputed_array = imputer.fit_transform(df_for_imputation)
df_imputed = pd.DataFrame(imputed_array, columns=cols_to_impute, index=df.index)

# 7. Reassemble full dataset

df_out = pd.concat([
    df_holdout.reset_index(drop=True),
    df_imputed.reset_index(drop=True),
    df_numeric[cols_over_50].reset_index(drop=True)
], axis=1)

# Reorder to match original column order
original_kept = [c for c in df.columns if c in df_out.columns]
df_out = df_out[original_kept]

# 8. After missingness summary

print("\n--- AFTER IMPUTATION (same columns) ---")
after_missing = df_out[cols_5_to_50].isnull().mean() * 100
for col in before.index:
    after_pct = after_missing.get(col, 0.0)
    print(f"  {before[col]:.1f}% → {after_pct:.1f}%  {col}")

# 9. Save

output_path = f"{PROCESSED_DIR}/lupus_1yr_imputed.xlsx"
df_out.to_excel(output_path, index=False)
print(f"\nImputed dataset saved to:\n{output_path}")
print(f"Final shape: {df_out.shape[0]} rows, {df_out.shape[1]} columns")
