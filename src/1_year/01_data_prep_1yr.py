import pandas as pd

RAW_DIR = "/Users/elizabetharmstrong/Library/CloudStorage/OneDrive-ImperialCollegeLondon/Lupus Project/Data/Raw"
PROCESSED_DIR = "/Users/elizabetharmstrong/Library/CloudStorage/OneDrive-ImperialCollegeLondon/Lupus Project/Data/Processed"

# ---------------------------
# 1. Load full dataset
# ---------------------------
df = pd.read_excel(f"{RAW_DIR}/data_lupus.xlsx")

# ---------------------------
# 2. Helper to find columns by keyword
# ---------------------------
def find_column(df, keyword):
    matches = [col for col in df.columns if keyword.lower() in col.lower()]
    if not matches:
        raise ValueError(f"No column found containing: {keyword}")
    if len(matches) > 1:
        print(f"Multiple matches for '{keyword}':")
        for m in matches:
            print("  -", m)
        print(f"Using first match: {matches[0]}")
    return matches[0]

# ---------------------------
# 3. Find key columns
# ---------------------------
flare_col = find_column(df, "Flare by 1 year")
nr_col = find_column(df, "Non Response")

print("\nUsing columns:")
print("Flare outcome column:", flare_col)
print("Non-response column:", nr_col)

# ---------------------------
# 4. Convert to numeric where possible
# ---------------------------
df[flare_col] = pd.to_numeric(df[flare_col], errors="coerce")
df[nr_col] = pd.to_numeric(df[nr_col], errors="coerce")

# ---------------------------
# 5. Define exclusions
# ---------------------------
non_responders = df[nr_col].isin([1, 2])
analysable = df[flare_col].isin([0, 1]) & (~non_responders)
inadequate_followup = df[flare_col].isna() & (~non_responders)

# ---------------------------
# 6. Create final 1-year flare dataset
# ---------------------------
df_1yr = df.loc[analysable].copy()
df_1yr = df_1yr.rename(columns={flare_col: "flare_1yr"})

# ---------------------------
# 7. Summary counts
# ---------------------------
total_episodes = len(df)
excluded_nonresponders = int(non_responders.sum())
excluded_inadequate_followup = int(inadequate_followup.sum())
analysable_n = len(df_1yr)
flare_events = int((df_1yr["flare_1yr"] == 1).sum())
non_flare_controls = int((df_1yr["flare_1yr"] == 0).sum())
event_rate = flare_events / analysable_n * 100 if analysable_n > 0 else 0

print("\n--- SUMMARY ---")
print(f"Total episodes: {total_episodes}")
print(f"Excluded non-responders: {excluded_nonresponders}")
print(f"Excluded inadequate follow-up / unusable 1-year outcome: {excluded_inadequate_followup}")
print(f"Analysable cohort: {analysable_n}")
print(f"Flare events: {flare_events}")
print(f"Non-flare controls: {non_flare_controls}")
print(f"Event rate: {event_rate:.1f}%")

# ---------------------------
# 8. Save filtered dataset
# ---------------------------
output_path = f"{PROCESSED_DIR}/lupus_1yr_flare_dataset.xlsx"
df_1yr.to_excel(output_path, index=False)
print(f"\nFiltered dataset saved to:\n{output_path}")
