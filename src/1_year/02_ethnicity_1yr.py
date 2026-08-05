import pandas as pd

PROCESSED_DIR = "/Users/elizabetharmstrong/Library/CloudStorage/OneDrive-ImperialCollegeLondon/Lupus Project/Data/Processed"

df = pd.read_excel(f"{PROCESSED_DIR}/lupus_1yr_flare_dataset.xlsx")

eth_col = [c for c in df.columns if "Ethnicity" in c][0]
print("Using column:", eth_col)

ethnicity_map = {
    1: "White",
    2: "Black",
    3: "Asian (South)",
    4: "Asian (East)",
    5: "Other",
    6: "Unknown"
}

df["Ethnicity_clean"] = df[eth_col].map(ethnicity_map)

eth_counts = df["Ethnicity_clean"].value_counts(dropna=False)
eth_percent = df["Ethnicity_clean"].value_counts(normalize=True, dropna=False) * 100

ethnicity_summary = pd.DataFrame({
    "Count": eth_counts,
    "Percentage (%)": eth_percent.round(1)
})

print("\nEthnicity Summary:")
print(ethnicity_summary)

ethnicity_summary.to_excel(f"{PROCESSED_DIR}/ethnicity_summary.xlsx")
print("\nSaved to ethnicity_summary.xlsx")
