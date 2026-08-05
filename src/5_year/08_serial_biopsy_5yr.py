import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings("ignore")

PROCESSED_DIR = "/Users/elizabetharmstrong/Library/CloudStorage/OneDrive-ImperialCollegeLondon/Lupus Project/Data/Processed"
OUTCOME_COL   = "flare_5yr"

# Column definitions

DELTA_COLS = {
    "delta_chronic_gloms":    "% chronic gloms(%of total)",
    "delta_active_gloms":     "% active gloms (%of those not globally sclerosed)",
    "delta_necrosis":         "%gloms with necrosis",
    "delta_sclerosed_gloms":  "% sclerosed gloms",
    "delta_ln_class":         "Class coded 1=I 2=II 3=III 4=IV 5=V 6=III+V 7=IV+V 8=II+V 9=VI 10=other",
    "delta_egfr":             "CKD epi formula without ethnicity",
}
PROT_COL = "Proteinuria (uPCR) at biopsy (highest value within 4/52 pre or post Bx)"
DATE_COL = "Biopsy date"
BX_NUM   = "Biopsy number for patient"

# The 10 single-biopsy predictors from the final 5yr feature set
SINGLE_BX_COLS = [
    "% chronic gloms(%of total)",
    "%gloms with necrosis",
    "Class coded 1=I 2=II 3=III 4=IV 5=V 6=III+V 7=IV+V 8=II+V 9=VI 10=other",
    "% active gloms (%of those not globally sclerosed)",
    "Prev exposure to cyclo (for Rx comparison) - related to the 'use this biopsy for this patient' biopsy",
    "% sclerosed gloms",
    "CKD epi formula without ethnicity",
    "Reason for biopsy 1=new pres LN 2=relapse 3=non-response/partial response, incl on-going proteinuria 4=pre-pregnancy or Ax if drug switch/stop appropriate",
    "Ethnicity 1=white 2=black 3=asian (south) 4=asian (east) 5=other 6=not stated/unknown/any other mixed",
    "Age at biopsy",
]

# 1. Load

df = pd.read_excel(f"{PROCESSED_DIR}/lupus_5yr_flare_dataset.xlsx")
print(f"Loaded: {df.shape[0]} rows, {df.shape[1]} columns")

# 2. Create proxy patient ID (DOB + gender)
#    DOB is a true per-patient fixed value; combined with gender
#    it gives 634 unique patient groups (max 8 biopsies/group)

df[DATE_COL] = pd.to_datetime(df[DATE_COL])
df["patient_id"] = df["DOB"].astype(str) + "|" + df["Gender (1=male, 2=female)"].astype(str)

n_null_dob = df["DOB"].isna().sum()
if n_null_dob > 0:
    print(f"  Warning: {n_null_dob} rows with missing DOB — assigned to group 'NaT|<gender>'")

# Log-transform proteinuria (clip to avoid log(0))
df["log_proteinuria"] = np.log(pd.to_numeric(df[PROT_COL], errors="coerce").clip(lower=0.01))

# Sort: patient → biopsy date → biopsy number (tie-break)
df = df.sort_values(["patient_id", DATE_COL, BX_NUM]).reset_index(drop=True)

# 3. Identify serial patients (≥2 biopsies in 5yr cohort)

group_sizes = df.groupby("patient_id").size()
serial_ids  = group_sizes[group_sizes >= 2].index

print(f"\nSerial biopsy summary (within 5yr cohort):")
print(f"  Total patients (unique DOB+gender):        {group_sizes.shape[0]}")
print(f"  Patients with ≥2 biopsies:                 {len(serial_ids)}")
print(f"  Biopsy episodes from serial patients:      {group_sizes[serial_ids].sum()}")
print(f"\n  Group size distribution:")
for size, count in group_sizes.value_counts().sort_index().items():
    print(f"    {size} biopsy/biopsies: {count} patient(s)")

# 4. Build serial dataset
#    For each serial patient: most recent biopsy = index biopsy
#    Deltas = most_recent − immediately_preceding

records = []

for pid in serial_ids:
    group = df[df["patient_id"] == pid].reset_index(drop=True)
    # Already sorted; most recent = last row, previous = second-to-last
    recent = group.iloc[-1]
    prev   = group.iloc[-2]

    row = {}
    row["patient_id"]          = pid
    row["n_biopsies_in_cohort"] = len(group)
    row["biopsy_number_recent"] = recent[BX_NUM]
    row["biopsy_date_recent"]   = recent[DATE_COL]
    row["biopsy_date_previous"] = prev[DATE_COL]

    # Time between biopsies (months)
    days_diff = (recent[DATE_COL] - prev[DATE_COL]).days
    row["time_between_bx_months"] = round(days_diff / 30.44, 1)

    # Delta features (recent − previous; positive = worsening/increasing)
    for delta_name, col in DELTA_COLS.items():
        if col in df.columns:
            r_val = pd.to_numeric(recent[col], errors="coerce")
            p_val = pd.to_numeric(prev[col],   errors="coerce")
            row[delta_name] = round(float(r_val - p_val), 4) if pd.notna(r_val) and pd.notna(p_val) else np.nan

    # Delta proteinuria (log scale)
    row["delta_log_proteinuria"] = round(
        float(recent["log_proteinuria"] - prev["log_proteinuria"]), 4
    ) if pd.notna(recent["log_proteinuria"]) and pd.notna(prev["log_proteinuria"]) else np.nan

    # Carry forward 10 single-biopsy predictors from most recent biopsy
    for col in SINGLE_BX_COLS:
        if col in df.columns:
            row[col] = recent[col]

    # Outcome from most recent biopsy
    row[OUTCOME_COL] = recent[OUTCOME_COL]

    records.append(row)

result = pd.DataFrame(records)

# 5. Summary

n_serial   = len(result)
n_events   = int(result[OUTCOME_COL].sum())
event_rate = n_events / n_serial * 100

print(f"\n{'='*65}")
print(f"SERIAL BIOPSY DATASET SUMMARY")
print(f"{'='*65}")
print(f"  Patients (rows):        {n_serial}")
print(f"  Flare events:           {n_events}  ({event_rate:.1f}%)")
print(f"  Non-events:             {n_serial - n_events}")
print(f"  Total features:         {result.shape[1] - 2}  (incl. patient_id + outcome)")

print(f"\n--- Delta features (missingness) ---")
delta_cols = [c for c in result.columns if c.startswith("delta_") or c == "time_between_bx_months"]
for c in delta_cols:
    miss = result[c].isna().sum()
    print(f"  {c:<35s}  missing={miss}/{n_serial}  "
          f"mean={result[c].mean():.3f}  std={result[c].std():.3f}")

print(f"\n--- Single-biopsy features (from most recent biopsy) ---")
for col in SINGLE_BX_COLS:
    if col in result.columns:
        miss = result[col].isna().sum()
        short = col[:60]
        print(f"  {short:<60s}  missing={miss}/{n_serial}")

print(f"\n--- Time between biopsies (months) ---")
print(f"  Mean:    {result['time_between_bx_months'].mean():.1f}")
print(f"  Median:  {result['time_between_bx_months'].median():.1f}")
print(f"  Min:     {result['time_between_bx_months'].min():.1f}")
print(f"  Max:     {result['time_between_bx_months'].max():.1f}")

# 6. Save

output_path = f"{PROCESSED_DIR}/lupus_5yr_serial_dataset.xlsx"
result.to_excel(output_path, index=False)
print(f"\nSaved: {output_path}")
print(f"Shape: {result.shape[0]} rows × {result.shape[1]} columns")
print(f"\nColumns saved:")
for c in result.columns:
    print(f"  {c}")
