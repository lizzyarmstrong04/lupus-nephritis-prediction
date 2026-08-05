import pandas as pd

DATA_DIR = "/Users/elizabetharmstrong/Library/CloudStorage/OneDrive-ImperialCollegeLondon/Lupus Project/Data/Processed"

df = pd.read_excel(f"{DATA_DIR}/lupus_1yr_flare_dataset.xlsx")

print(f"Dataset shape: {df.shape[0]} rows, {df.shape[1]} columns\n")

# Overall missingness
missing = df.isnull().mean() * 100
missing_df = missing.reset_index()
missing_df.columns = ["Variable", "Missing (%)"]
missing_df = missing_df.sort_values("Missing (%)", ascending=False)

# Summary counts
no_missing = (missing_df["Missing (%)"] == 0).sum()
low_missing = ((missing_df["Missing (%)"] > 0) & (missing_df["Missing (%)"] <= 10)).sum()
high_missing = (missing_df["Missing (%)"] > 10).sum()

print("--- MISSINGNESS SUMMARY ---")
print(f"Variables with no missing data:      {no_missing}")
print(f"Variables with 1-10% missing:        {low_missing}")
print(f"Variables with >10% missing:         {high_missing}")
print()

print("--- ALL VARIABLES (sorted by missingness) ---")
print(missing_df.to_string(index=False))

# Save to Excel
output_path = f"{DATA_DIR}/missingness_1yr_dataset.xlsx"
missing_df.to_excel(output_path, index=False)
print(f"\nSaved to: {output_path}")
