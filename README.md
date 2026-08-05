# Lupus Nephritis ML Prediction Models

Machine learning models for predicting clinical outcomes in lupus nephritis (LN), developed at Imperial College London.

## Outcomes predicted

| Model | Cohort | n |
|-------|--------|---|
| 1-year flare | LN patients after biopsy | 430 |
| 5-year flare | LN patients after biopsy | 356 |
| Serial biopsy 5-year flare | Patients with repeat biopsy | 70 |
| ESRD 5-year | LN patients | 796 |
| ESRD 10-year | LN patients | 796 |

## Algorithms

Logistic Regression, Random Forest, XGBoost, LightGBM — with hyperparameter tuning via 5×10-fold repeated stratified cross-validation. Benchmarked against TabPFN v3 (Prior Labs).

## Repository structure

```
src/
├── 1_year/          # 1-year flare: data prep → imputation → feature selection → modelling → SHAP
├── 5_year/          # 5-year flare + serial biopsy pipeline
├── esrd/            # ESRD 5-year and 10-year pipeline + DeLong + SHAP
├── app/             # Streamlit risk calculator + saved models
├── 00_eda_raw.py
├── 13_delong_test.py
├── 14_shap_1yr_vs_5yr.py
├── 15_shap_mean_rank_panel.py
├── 16b_tabpfn_v2_all_cohorts.py   # TabPFN v3 benchmark (all cohorts)
├── 17_roc_panel_figure.py          # Publication ROC panel
├── 18_paper_figures.py             # Calibration, dumbbell, SHAP figures
├── 19_delong_pairwise.py           # Pairwise DeLong tests
├── 20_multitask_esrd.py            # Multi-task neural network (ESRD)
└── 21_km_analysis.py               # Kaplan-Meier analysis
outputs/figures/                    # Publication-quality figures (PDF + PNG)
```

## Evaluation

- **Internal validation**: 5×10-fold repeated stratified CV (OOF predictions)
- **Calibration**: Harrell optimism-corrected bootstrap (500 iterations)
- **Pairwise comparisons**: DeLong test on pooled OOF predictions
- **SHAP**: Linear SHAP (exact, coef × scaled value) for interpretability

## Requirements

```bash
pip install -r requirements_app.txt
```

## Running the risk calculator app

```bash
streamlit run src/app/app.py
```

## TabPFN benchmark

Set your API token before running:

```bash
export TABPFN_TOKEN="your_token_here"
python src/16b_tabpfn_v2_all_cohorts.py
```

## Notes

- Patient data is not included in this repository
- All hardcoded file paths reference `Data/Processed/` — update to your local data directory
- Models saved as `.joblib` in `src/app/models/` contain fitted coefficients only, no patient data
