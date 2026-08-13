"""
TabPFN v3 (client API) benchmark — all five cohorts:
  1-year flare, 5-year flare, ESRD 5-year, ESRD 10-year, serial biopsy

Protocol: RepeatedStratifiedKFold CV, matching each cohort's own modelling
          script - 5x10-fold (50 folds) for 1yr/5yr/ESRD, 5x5-fold (25
          folds) for serial biopsy (n=70, too small for 10-fold).
Metrics:  AUROC, Brier score, calibration slope
Saves:    outputs/tabpfn_v3_all_cohorts.xlsx  (one sheet per cohort)
          + incremental .pkl checkpoint after every fold so a dropped
          connection doesn't lose completed work

Robustness: 5 retry attempts per fold, 15s sleep on failure, 3s pause
            between every fold.

Checkpoint resume: cohorts already marked "done" in the checkpoint are
skipped entirely (no re-billed API calls) - so a rerun only spends usage
on cohorts not yet completed with the current script version.

OOF predictions: retained (per-sample, mapped back to original row order)
so ROC/calibration curves can be built for TabPFN v3, not just aggregate
metrics (see outputs/tabpfn_v3_oof_predictions.xlsx). All 5 cohorts now
have this - it was added after the first 4 cohorts' initial run (which
only kept aggregate fold metrics), then backfilled by clearing those 4
checkpoint entries and rerunning with OOF retention.
"""

import time, warnings, os, pickle
warnings.filterwarnings("ignore")

# Set TABPFN_TOKEN in your environment before running:
#   export TABPFN_TOKEN="your_token_here"

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, brier_score_loss
from sklearn.model_selection import RepeatedStratifiedKFold
import tabpfn_client
from tabpfn_client import TabPFNClassifier

# Auth
tabpfn_client.set_access_token(os.environ["TABPFN_TOKEN"])

BASE  = "/Users/elizabetharmstrong/Library/CloudStorage/OneDrive-ImperialCollegeLondon/Lupus Project"
PROC  = f"{BASE}/Data/Processed"
OUT   = f"{BASE}/outputs"
CKPT  = f"{OUT}/tabpfn_v3_checkpoint.pkl"   # incremental save

# Calibration slope
def calibration_slope(y_true, y_prob):
    lo = np.log(np.clip(y_prob, 1e-6, 1-1e-6) / (1 - np.clip(y_prob, 1e-6, 1-1e-6)))
    m  = LogisticRegression(fit_intercept=True, max_iter=1000)
    m.fit(lo.reshape(-1, 1), y_true)
    return float(m.coef_[0][0])

# Load checkpoint (resume if a previous run was interrupted)
if os.path.exists(CKPT):
    with open(CKPT, "rb") as f:
        checkpoint = pickle.load(f)
    print(f"Resuming from checkpoint — completed cohorts: "
          f"{[k for k,v in checkpoint.items() if v.get('done')]}")
else:
    checkpoint = {}

def save_ckpt():
    with open(CKPT, "wb") as f:
        pickle.dump(checkpoint, f)

# CV with retry + incremental save
def run_cv(X, y, label, n_splits=10, n_repeats=5,
           max_retries=5, retry_sleep=15, fold_pause=3):

    if label in checkpoint and checkpoint[label].get("done"):
        print(f"\n[{label}] already complete — loading from checkpoint.")
        return checkpoint[label]["result"]

    print(f"\n{'='*60}")
    print(f"TabPFN v3 — {label}")
    print(f"  n={len(y)}, events={int(y.sum())} ({y.mean()*100:.1f}%)")

    n_folds = n_splits * n_repeats

    # Resume partially-completed folds if checkpoint exists
    completed = checkpoint.get(label, {}).get("folds", {})
    aurocs = list(completed.get("aurocs", []))
    briers = list(completed.get("briers", []))
    slopes = list(completed.get("slopes", []))
    oof_sum   = np.array(completed.get("oof_sum", np.zeros(len(y))), dtype=float)
    oof_count = np.array(completed.get("oof_count", np.zeros(len(y))), dtype=float)
    start_fold = len(aurocs)

    if start_fold > 0:
        print(f"  Resuming from fold {start_fold+1}/{n_folds}")

    CV = RepeatedStratifiedKFold(n_splits=n_splits, n_repeats=n_repeats,
                                  random_state=42)
    splits = list(CV.split(X, y))

    for i, (tr, te) in enumerate(splits):
        if i < start_fold:
            continue   # already done

        success = False
        for attempt in range(max_retries):
            try:
                clf = TabPFNClassifier()
                clf.fit(X.iloc[tr], y.iloc[tr])
                probs = clf.predict_proba(X.iloc[te])[:, 1]
                success = True
                break
            except Exception as e:
                wait = retry_sleep * (attempt + 1)   # back-off
                print(f"  fold {i+1} attempt {attempt+1} failed "
                      f"({type(e).__name__}: {str(e)[:60]}), "
                      f"retry in {wait}s...")
                if attempt < max_retries - 1:
                    time.sleep(wait)
                else:
                    print(f"  fold {i+1} FAILED after {max_retries} attempts — skipping.")

        if not success:
            continue

        aurocs.append(roc_auc_score(y.iloc[te], probs))
        briers.append(brier_score_loss(y.iloc[te], probs))
        slopes.append(calibration_slope(y.iloc[te].values, probs))
        oof_sum[te]   += probs
        oof_count[te] += 1

        # Save after every fold
        checkpoint[label] = {
            "done":   False,
            "folds":  {"aurocs": aurocs, "briers": briers, "slopes": slopes,
                       "oof_sum": oof_sum.tolist(), "oof_count": oof_count.tolist()},
            "result": None,
        }
        save_ckpt()

        time.sleep(fold_pause)

        if (i + 1) % 10 == 0:
            print(f"  fold {i+1}/{n_folds}  running AUROC={np.mean(aurocs):.3f}", flush=True)

    oof_probs = oof_sum / np.where(oof_count > 0, oof_count, 1)

    result = {
        "Model":                  "TabPFN v3",
        "CV AUROC (mean)":        round(np.mean(aurocs), 3),
        "CV AUROC 95% CI lower":  round(np.percentile(aurocs, 2.5), 3),
        "CV AUROC 95% CI upper":  round(np.percentile(aurocs, 97.5), 3),
        "CV Brier Score":         round(np.mean(briers), 3),
        "CV Calibration Slope":   round(np.mean(slopes), 3),
        "N folds completed":      len(aurocs),
        "oof_probs":              oof_probs.tolist(),
    }

    checkpoint[label] = {"done": True, "folds": None, "result": result}
    save_ckpt()

    print(f"\n  ✓ {label} complete — AUROC={result['CV AUROC (mean)']:.3f}  "
          f"[{result['CV AUROC 95% CI lower']:.3f}–{result['CV AUROC 95% CI upper']:.3f}]  "
          f"Brier={result['CV Brier Score']:.3f}  Slope={result['CV Calibration Slope']:.3f}")
    return result

# Load datasets
df1     = pd.read_excel(f"{PROC}/lupus_1yr_selected_clean.xlsx")
df5     = pd.read_excel(f"{PROC}/lupus_5yr_selected_clean.xlsx")
df_e5   = pd.read_excel(f"{PROC}/esrd_5yr_selected.xlsx")
df_e10  = pd.read_excel(f"{PROC}/esrd_10yr_selected.xlsx")
df_ser  = pd.read_excel(f"{PROC}/lupus_5yr_serial_selected.xlsx")

# (label, X, y, n_splits, n_repeats) - serial biopsy uses 5x5-fold (n=70,
# too small for 10-fold), matching src/5_year/11_serial_modelling_5yr.py
COHORTS = [
    ("1yr_flare",     df1.drop(columns=["flare_1yr"]),    df1["flare_1yr"].astype(int),    10, 5),
    ("5yr_flare",     df5.drop(columns=["flare_5yr"]),    df5["flare_5yr"].astype(int),    10, 5),
    ("esrd_5yr",      df_e5.drop(columns=["esrd_5yr"]),   df_e5["esrd_5yr"].astype(int),   10, 5),
    ("esrd_10yr",     df_e10.drop(columns=["esrd_10yr"]), df_e10["esrd_10yr"].astype(int), 10, 5),
    ("serial_biopsy", df_ser.drop(columns=["flare_5yr"]), df_ser["flare_5yr"].astype(int),  5, 5),
]

# Run all cohorts (already-completed cohorts are skipped via checkpoint resume)
results = {}
for label, X, y, n_splits, n_repeats in COHORTS:
    results[label] = run_cv(X, y, label, n_splits=n_splits, n_repeats=n_repeats)

# Load existing model results to build comparison tables
cv1   = pd.read_excel(f"{OUT}/1yr_model_results.xlsx",              sheet_name="CV Results")
cv5   = pd.read_excel(f"{OUT}/5yr_model_results.xlsx",              sheet_name="CV Results")
cve5  = pd.read_excel(f"{OUT}/esrd/esrd_model_results.xlsx",        sheet_name="5yr CV Results")
cve10 = pd.read_excel(f"{OUT}/esrd/esrd_model_results.xlsx",        sheet_name="10yr CV Results")
cvser = pd.read_excel(f"{OUT}/5yr_serial_model_results.xlsx",       sheet_name="CV Results")

def append_tabpfn(existing_df, res):
    row = pd.DataFrame([{c: res.get(c, np.nan) for c in existing_df.columns if c != "oof_probs"}])
    return pd.concat([existing_df, row], ignore_index=True)

compare = {
    "1yr_flare":     append_tabpfn(cv1,   results["1yr_flare"]),
    "5yr_flare":     append_tabpfn(cv5,   results["5yr_flare"]),
    "esrd_5yr":      append_tabpfn(cve5,  results["esrd_5yr"]),
    "esrd_10yr":     append_tabpfn(cve10, results["esrd_10yr"]),
    "serial_biopsy": append_tabpfn(cvser, results["serial_biopsy"]),
}

# Save OOF predictions where available (only newly-run cohorts retain this -
# see module docstring), for building ROC/calibration curves
oof_export_rows = []
for label, X, y, *_ in COHORTS:
    oof = results[label].get("oof_probs")
    if oof is None:
        continue
    for idx, (true_label, prob) in enumerate(zip(y.values, oof)):
        oof_export_rows.append({"Cohort": label, "Sample_Index": idx,
                                 "True_Label": int(true_label), "OOF_Predicted_Probability": prob})
if oof_export_rows:
    oof_path = f"{OUT}/tabpfn_v3_oof_predictions.xlsx"
    pd.DataFrame(oof_export_rows).to_excel(oof_path, index=False)
    print(f"Saved: {oof_path}")

# Print summary
for label, df in compare.items():
    print(f"\n{'='*70}")
    print(f"COMPARISON — {label.upper()}")
    print("="*70)
    print(df[["Model","CV AUROC (mean)","CV AUROC 95% CI lower",
              "CV AUROC 95% CI upper","CV Brier Score","CV Calibration Slope"]].to_string(index=False))

# Save
out_path = f"{OUT}/tabpfn_v3_all_cohorts.xlsx"
with pd.ExcelWriter(out_path, engine="openpyxl") as w:
    for label, df in compare.items():
        df.to_excel(w, sheet_name=label, index=False)

print(f"\nSaved: {out_path}")
print("Checkpoint file (can delete when done):", CKPT)
print("\nDone.")
