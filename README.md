# Lupus Nephritis ML Prediction Models

Machine learning models predicting clinical outcomes in lupus nephritis (LN), developed at Imperial College London.

This repository contains the code used for the analysis in the dissertation.

## Outcomes predicted

| Cohort | n |
|---|---|
| 1 year flare | 430 |
| 5 year flare | 356 |
| Serial biopsy sub cohort (5 year flare, 2+ biopsies) | 70 |
| ESRD 5 year | 796 |
| ESRD 10 year | 796 |

## Models

Logistic Regression, Random Forest, XGBoost, LightGBM. The tree based models were tuned; logistic regression was left at default settings. All four were evaluated with repeated stratified cross validation (5 repeats of 10 folds; 5 repeats of 5 folds for the smaller serial biopsy cohort) and benchmarked against TabPFN v3 (Prior Labs).

## Repository structure

```
cohorts/            Cohort derivation, exclusion logic, composite ESRD outcome, serial biopsy sub cohort
preprocessing/       Missingness assessment, MICE imputation
feature_selection/   Six step selection pipeline (leakage, correlation, VIF, EPV capped LASSO)
modelling/           Model training, tuning, cross validation, bootstrap; TabPFN v3 benchmark
evaluation/          DeLong pairwise tests, bias corrected AUROC + CI, pmsampsize, EPV sensitivity
explainability/      SHAP computation + cross model rank averaging
risk_calculator/     Streamlit risk calculator (scoring/banding logic) + saved models
```

Each `cohorts/`, `preprocessing/`, and `feature_selection/` script is specific to one cohort (filename prefix indicates which). ESRD's cohort derivation and feature selection are combined in `cohorts/esrd_cohort_and_features.py`, matching how that step is implemented.

## Evaluation

- **Internal validation**: repeated stratified cross validation, out of fold predictions
- **Optimism correction**: Harrell bootstrap (1,000 iterations); bias corrected AUROC is the primary reported discrimination metric, not raw CV AUROC
- **Pairwise comparisons**: DeLong's test on pooled out of fold predictions, Bonferroni Holm corrected
- **Sample size adequacy**: post hoc calculation following Riley et al. 2020 (BMJ 368:m441)
- **SHAP**: linear explainer (logistic regression), tree explainer (ensemble models); cross model mean SHAP rank averaging

## Requirements

```bash
pip install -r requirements_app.txt
```

## Running the risk calculator app

```bash
streamlit run risk_calculator/app.py
```

## TabPFN benchmark

Set your own API token before running:

```bash
export TABPFN_TOKEN="your_token_here"
python modelling/tabpfn_v3_benchmark.py
```

## Notes

- Patient data is not included in this repository
- File paths point to `Data/Processed/` on the original development machine; update these to your own local data directory before running
- Models saved as `.joblib` in `risk_calculator/models/` contain fitted coefficients only, no patient data
