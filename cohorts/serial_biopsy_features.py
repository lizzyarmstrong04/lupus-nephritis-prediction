import pandas as pd
import numpy as np

PROCESSED_DIR = "/Users/elizabetharmstrong/Library/CloudStorage/OneDrive-ImperialCollegeLondon/Lupus Project/Data/Processed"

df = pd.read_excel(f"{PROCESSED_DIR}/lupus_5yr_serial_dataset.xlsx")
print(f"Loaded: {df.shape[0]} rows, {df.shape[1]} columns")

# Drop prev exposure to cyclo (69/70 missing — unusable)
cyclo_col = [c for c in df.columns if "cyclo" in c.lower()]
df = df.drop(columns=cyclo_col)
print(f"Dropped {len(cyclo_col)} cyclo column(s): {cyclo_col}")

# Add time-normalised delta features
# Each raw delta divided by time_between_bx_months
# Interpretation: change per month between biopsies
raw_delta_cols = [c for c in df.columns if c.startswith("delta_")]
time_col       = "time_between_bx_months"

for col in raw_delta_cols:
    normed_name = col + "_per_month"
    df[normed_name] = df[col] / df[time_col]
    df[normed_name] = df[normed_name].replace([np.inf, -np.inf], np.nan)

print(f"\nAdded {len(raw_delta_cols)} time-normalised delta features:")
for col in raw_delta_cols:
    print(f"  {col}_per_month")

# Final summary
feature_cols = [c for c in df.columns
                if c not in ["patient_id", "n_biopsies_in_cohort",
                             "biopsy_number_recent", "biopsy_date_recent",
                             "biopsy_date_previous", "flare_5yr"]]

print(f"\nFinal dataset: {df.shape[0]} rows × {df.shape[1]} columns")
print(f"Predictor columns ({len(feature_cols)}):")
for c in feature_cols:
    miss = df[c].isna().sum()
    print(f"  {c[:70]:<70s}  missing={miss}")

df.to_excel(f"{PROCESSED_DIR}/lupus_5yr_serial_dataset.xlsx", index=False)
print(f"\nSaved: lupus_5yr_serial_dataset.xlsx")
