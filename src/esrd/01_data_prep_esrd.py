import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings("ignore")

RAW_PATH      = "/Users/elizabetharmstrong/Library/CloudStorage/OneDrive-ImperialCollegeLondon/Lupus Project/Data/Raw/data_lupus.xlsx"
PROCESSED_DIR = "/Users/elizabetharmstrong/Library/CloudStorage/OneDrive-ImperialCollegeLondon/Lupus Project/Data/Processed"

# Column names
BX_NUM   = "Biopsy number for patient"
OUTCOME_COL_STRICT = (
    "Outcomes at 5yrs from first Bx in this db 1=eGFR>80 2=doubling baseline creatinine "
    "3=RRT 4=Death 5=Death on RRT 6=eGFR<80 but not doubling creatinine, RRT or death, "
    "7=eGFR<70 but not doubling creatinine, RRT or death 8=eGFR<60 but not doubling "
    "creatinine, RRT .1"
)   # .1 suffix = stricter version (X for insufficient follow-up)

OUTCOME_COL_LAX = (
    "Outcomes at 5yrs from first Bx in this db 1=eGFR>80 2=doubling baseline creatinine "
    "3=RRT 4=Death 5=Death on RRT 6=eGFR<80 but not doubling creatinine, RRT or death, "
    "7=eGFR<70 but not doubling creatinine, RRT or death 8=eGFR<60 but not doubling "
    "creatinine, RRT "
)   # no suffix = uses last known outcome even if <5yr follow-up

# 1. Load & restrict to index biopsies (Bx number = 1)

df = pd.read_excel(RAW_PATH)
print(f"Raw data: {df.shape[0]} rows × {df.shape[1]} columns")

df_idx = df[df[BX_NUM] == 1].copy().reset_index(drop=True)
print(f"Index biopsies (Bx=1): {len(df_idx)}")

# 2. Outcome coding
#    Binary: 1 = codes 2, 3, 5 (renal progression)
#            0 = code 1 only  (stable eGFR>80 at 5yr)
#    Excluded: code 4 (death alone, competing event)
#              codes 6, 7, 8 (eGFR decline not meeting composite)
#              X  (insufficient follow-up)
#              NaN (missing)

EVENT_CODES   = {2, 3, 5}
CONTROL_CODES = {1}
EXCLUDE_CODES = {4, 6, 7, 8}

def apply_outcome(series, col_label):
    """Convert raw coded column to binary outcome; return df with counts."""
    s = series.copy()

    # Coerce to comparable type — values may be int/float or string
    def normalise(v):
        if pd.isna(v):
            return np.nan
        if str(v).strip() == "X":
            return "X"
        try:
            return int(float(v))
        except (ValueError, TypeError):
            return np.nan

    s = s.map(normalise)
    counts = s.value_counts(dropna=False)

    binary = s.map(lambda v: 1 if v in EVENT_CODES else
                              0 if v in CONTROL_CODES else
                              np.nan)
    return binary, counts

print(f"\n{'='*65}")
print("OUTCOME COLUMN: 5yrs from first Bx (X-coded, strict follow-up)")
print(f"{'='*65}")

outcome_strict, vc_strict = apply_outcome(df_idx[OUTCOME_COL_STRICT], "strict")

print(f"\nRaw value counts in outcome column:")
print(vc_strict.to_string())

# Exclusion breakdown
n_total_idx   = len(df_idx)
n_event       = int((outcome_strict == 1).sum())
n_nonevent    = int((outcome_strict == 0).sum())
n_x           = int((df_idx[OUTCOME_COL_STRICT].map(lambda v: str(v).strip() == "X")).sum())
n_nan         = int(df_idx[OUTCOME_COL_STRICT].isna().sum())
n_code4       = int((df_idx[OUTCOME_COL_STRICT].map(
                     lambda v: normalise(v) == 4 if callable(normalise) else False
                 )).sum())

# Re-do cleanly
def code_val(v):
    if pd.isna(v): return "NaN"
    s = str(v).strip()
    if s == "X": return "X"
    try: return int(float(s))
    except: return "other"

coded = df_idx[OUTCOME_COL_STRICT].map(code_val)
n_nan    = (coded == "NaN").sum()
n_x      = (coded == "X").sum()
n_code4  = (coded == 4).sum()
n_code6  = (coded == 6).sum()
n_code7  = (coded == 7).sum()
n_code8  = (coded == 8).sum()
n_excl_interim = n_nan + n_x + n_code4 + n_code6 + n_code7 + n_code8

print(f"\n{'='*65}")
print("COHORT SUMMARY")
print(f"{'='*65}")
print(f"  Total index biopsies in raw data:          {n_total_idx}")
print(f"")
print(f"  Excluded:")
print(f"    Missing (NaN) — no outcome data:         {n_nan}")
print(f"    X — insufficient follow-up (<5yr):        {n_x}")
print(f"    Code 4 — death alone (competing event):   {n_code4}")
print(f"    Code 6 — eGFR<80 (sub-threshold):         {n_code6}")
print(f"    Code 7 — eGFR<70 (sub-threshold):         {n_code7}")
print(f"    Code 8 — eGFR<60 (sub-threshold):         {n_code8}")
print(f"    Total excluded:                           {n_excl_interim}")
print(f"")
print(f"  Final analytic cohort:                     {n_event + n_nonevent}")
print(f"    Events (renal progression):               {n_event}  "
      f"[codes 2=Cr doubling, 3=RRT, 5=death on RRT]")
print(f"    Non-events (stable, eGFR>80):             {n_nonevent}  [code 1]")
print(f"    Event rate:                               {n_event/(n_event+n_nonevent)*100:.1f}%")

print(f"\n  EPV check (EPV-10 rule):")
print(f"    {n_event} events → max {n_event // 10} predictors")

# Also show the lax version for comparison
print(f"\n{'='*65}")
print("COMPARISON: Lax version (last-known-outcome, no X-coding)")
print(f"{'='*65}")
outcome_lax, vc_lax = apply_outcome(df_idx[OUTCOME_COL_LAX], "lax")
coded_lax = df_idx[OUTCOME_COL_LAX].map(code_val)
n_event_lax    = int((outcome_lax == 1).sum())
n_nonevent_lax = int((outcome_lax == 0).sum())
n_nan_lax      = (coded_lax == "NaN").sum()
n_x_lax        = (coded_lax == "X").sum()
print(f"  Analytic cohort:  {n_event_lax + n_nonevent_lax}  "
      f"(events={n_event_lax}, non-events={n_nonevent_lax}, "
      f"event rate={n_event_lax/(n_event_lax+n_nonevent_lax)*100:.1f}%)")
print(f"  NaN: {n_nan_lax},  X: {n_x_lax}")
print(f"  ⚠  Non-events here include patients with <5yr follow-up coded as stable")
print(f"     → inflates non-events artificially; strict version preferred")

# Event breakdown by code
print(f"\n{'='*65}")
print("EVENT BREAKDOWN (strict version)")
print(f"{'='*65}")
for c, label in [(2, "Creatinine doubling"), (3, "RRT"), (5, "Death on RRT")]:
    n = (coded == c).sum()
    print(f"  Code {c} — {label}: {n}")

# 3. Build and save the dataset

df_out = df_idx[outcome_strict.notna()].copy()
df_out["esrd_5yr"] = outcome_strict[outcome_strict.notna()].values

print(f"\n{'='*65}")
print(f"Saving dataset: {len(df_out)} rows × {df_out.shape[1]} columns")
output_path = f"{PROCESSED_DIR}/lupus_esrd_dataset.xlsx"
df_out.to_excel(output_path, index=False)
print(f"Saved: {output_path}")
