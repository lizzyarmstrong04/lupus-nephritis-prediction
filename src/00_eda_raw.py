import pandas as pd

RAW_DIR = "/Users/elizabetharmstrong/Library/CloudStorage/OneDrive-ImperialCollegeLondon/Lupus Project/Data/Raw"
PROCESSED_DIR = "/Users/elizabetharmstrong/Library/CloudStorage/OneDrive-ImperialCollegeLondon/Lupus Project/Data/Processed"

df = pd.read_excel(f"{RAW_DIR}/data_lupus.xlsx")

print(df.head())
print(df.shape)
print(df.columns.tolist())
print(df.info())

missing = df.isnull().mean() * 100
print(missing.sort_values(ascending=False))

missing_df = (df.isnull().mean() * 100).sort_values(ascending=False).reset_index()
missing_df.columns = ["Variable", "Missing (%)"]

print(missing_df.head(30))
missing_df.to_excel(f"{PROCESSED_DIR}/missingness_summary.xlsx", index=False)

key_vars = [
    "Age at biopsy",
    "Gender (1=male, 2=female)",
    "Ethnicity 1=white 2=black 3=asian (south) 4=asian (east) 5=other 6=not stated/unknown/any other mixed",
    "C3 at biopsy (normal range 0.7-1.7)",
    "C4 at biopsy (normal range 0.16-0.54)",
    "dsDNA 0-30 (X=no result, <10 listed as 0)",
    "eGFR as quoted at biopsy",
    "Proteinuria (uPCR) at biopsy (highest value within 4/52 pre or post Bx)",
    "Class coded 1=I 2=II 3=III 4=IV 5=V 6=III+V 7=IV+V 8=II+V 9=VI 10=other",
    "Mesangial deposits",
    "HCQ ",
    "ACE/ARB Y=1 N=0",
    "Immunosuppression protocol post biopsy",
    "Flare by 1 year"
]

key_missing = df[key_vars].isnull().mean() * 100
print(key_missing.sort_values(ascending=False))
