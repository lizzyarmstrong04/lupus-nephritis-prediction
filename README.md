# Lupus Nephritis ML Prediction Models

Machine learning models for predicting clinical outcomes in lupus nephritis (LN), developed at Imperial College London.

This repository contains the code implementing the analysis described in the dissertation's Methods, Supplementary Methods, and Results sections.

## Outcomes predicted

| Cohort | n |
|---|---|
| 1-year flare | 430 |
| 5-year flare | 356 |
| Serial biopsy sub-cohort (5-year flare, ≥2 biopsies) | 70 |
| ESRD 5-year | 796 |
| ESRD 10-year | 796 |

## Algorithms

Logistic Regression, Random Forest, XGBoost, LightGBM with hyperparameter tuning (RandomizedSearchCV) for the tree-based models, evaluated via 5×10-fold repeated stratified cross-validation (5×5-fold for the serial-biopsy cohort, n=70). Benchmarked against TabPFN v3 (Prior Labs).

## Repository structure

```
cohorts/            # Cohort derivation, exclusion logic, composite ESRD outcome, serial-biopsy sub-cohort
preprocessing/       # Missingness assessment, MICE imputation
feature_selection/   # Six-step selection pipeline (leakage/correlation/VIF/EPV-capped LASSO)
modelling/           # Model training, tuning, CV, Harrell bootstrap; TabPFN v3 benchmark
evaluation/          # DeLong pairwise tests, bias-corrected AUROC + CI, pmsampsize, EPV sensitivity
explainability/      # SHAP computation + cross-model rank-averaging
risk_calculator/     # Streamlit risk calculator (scoring/banding logic) + saved models
```

Each `cohorts/`, `preprocessing/`, and `feature_selection/` script is specific to one cohort (filename prefix indicates which). ESRD's cohort derivation and feature selection are combined in `cohorts/esrd_cohort_and_features.py`, matching how that step is implemented.

## Evaluation

- **Internal validation**: 5×10-fold (5×5-fold for serial biopsy) repeated stratified CV, out-of-fold predictions
- **Optimism correction**: Harrell bootstrap (1,000 iterations), bias-corrected AUROC is the primary reported discrimination metric, not raw CV AUROC
- **Pairwise comparisons**: DeLong's test on pooled OOF predictions, Bonferroni-Holm corrected
- **Sample size adequacy**: post-hoc pmsampsize-style calculation (Riley et al. 2020, BMJ 368:m441)
- **SHAP**: Linear explainer (logistic regression), tree explainer (ensemble models); cross-model mean |SHAP| rank-averaging

## Requirements

```bash
pip install -r requirements_app.txt
```

## Running the risk calculator app

```bash
streamlit run risk_calculator/app.py
```

## TabPFN benchmark

Set your API token before running:

```bash
export TABPFN_TOKEN="your_token_here"
python modelling/tabpfn_v3_benchmark.py
```

## Notes

- Patient data is not included in this repository
- All hardcoded file paths reference `Data/Processed/` — update to your local data directory
- Models saved as `.joblib` in `risk_calculator/models/` contain fitted coefficients only, no patient data
