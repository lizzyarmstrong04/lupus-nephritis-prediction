"""
Formal minimum sample-size justification (Riley et al. 2020, BMJ 368:m441) for
each of the five cohorts, using the binary-outcome closed-form criteria:
  1) shrinkage        - limits overfitting of predictor effects to <=10%
  2) small R2 shrinkage - <=0.05 absolute difference between apparent and
                          adjusted Nagelkerke R2
  3) precise intercept  - margin of error <=0.05 in the estimated event risk
Final required n = max(n1, n2, n3).

Formulas re-implemented directly from the pmsampsize R package / its official
Python port (Whittle & Ensor, https://pypi.org/project/pmsampsize/) - not
pip-installable here as it requires Python >=3.12, so the binary-outcome path
is reproduced standalone below (cstat2rsq + pmsampsize_bin), dependency-light
(numpy, scipy, statsmodels only - already project dependencies).

Inputs per cohort:
  - parameters  = final predictor count actually used in the model
  - prevalence  = observed event rate in the analytic cohort
  - cstatistic  = the cohort's own optimism-adjusted (BC) AUROC for Logistic
                  Regression (the paper's designated primary model) - per
                  Riley's guidance to use an optimism-adjusted C-statistic.
  - shrinkage   = 0.9 (standard default)

Saves: outputs/pmsampsize_results.xlsx
"""
import math
import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf
from scipy.stats import norm

OUT = "/Users/elizabetharmstrong/Library/CloudStorage/OneDrive-ImperialCollegeLondon/Lupus Project/outputs"


def cstat2rsq(cstatistic, prevalence, seed=123456):
    """Approximate Cox-Snell R2 from a target C-statistic and prevalence
    (Riley et al. 2020 simulation approach)."""
    n = 1_000_000
    mu = math.sqrt(2) * norm.ppf(cstatistic)
    rng = np.random.default_rng(seed)
    n_event = int(prevalence * n)
    n_nonevent = n - n_event
    LP = np.concatenate([
        rng.normal(loc=0, scale=1, size=n_nonevent),
        rng.normal(loc=mu, scale=1, size=n_event),
    ])
    y = np.concatenate([np.zeros(n_nonevent), np.ones(n_event)])
    data = pd.DataFrame({"y": y, "LP": LP})
    model = smf.glm(formula="y ~ LP", data=data, family=sm.families.Binomial()).fit()
    r2_coxsnell = 1 - np.exp(-(model.null_deviance - model.deviance) / n)
    return r2_coxsnell


def pmsampsize_bin(rsquared, parameters, prevalence, shrinkage=0.9):
    r2a = rsquared

    # Criterion 1 - shrinkage
    n1 = math.ceil(parameters / ((shrinkage - 1) * math.log(1 - (r2a / shrinkage))))

    # Criterion 2 - small absolute difference in R2 adjusted
    E1 = n1 * prevalence
    lnLnull = (E1 * math.log(E1 / n1)) + ((n1 - E1) * math.log(1 - (E1 / n1)))
    max_r2a = 1 - math.exp((2 * lnLnull) / n1)
    s_small_diff = r2a / (r2a + (0.05 * max_r2a))
    n2 = math.ceil(parameters / ((s_small_diff - 1) * math.log(1 - (r2a / s_small_diff))))

    # Criterion 3 - precise estimation of the intercept (margin of error 0.05)
    n3 = math.ceil(((1.96 / 0.05) ** 2) * (prevalence * (1 - prevalence)))

    n_final = max(n1, n2, n3)
    E_final = n_final * prevalence
    epp_final = E_final / parameters

    return {
        "n1_shrinkage": n1, "n2_r2_diff": n2, "n3_intercept": n3,
        "n_required": n_final, "events_required": E_final,
        "EPP_required": epp_final, "max_nagelkerke_r2": max_r2a,
    }


def pmsampsize_binary(parameters, prevalence, cstatistic, shrinkage=0.9, seed=123456):
    r2 = cstat2rsq(cstatistic, prevalence, seed=seed)
    out = pmsampsize_bin(r2, parameters, prevalence, shrinkage=shrinkage)
    out["cox_snell_r2"] = r2
    return out


if __name__ == "__main__":
    cohorts = [
        {"name": "1-Year flare",  "n_actual": 430, "events_actual": 99,  "parameters": 9,  "prevalence": 0.230, "cstatistic": 0.709},
        {"name": "5-Year flare",  "n_actual": 356, "events_actual": 166, "parameters": 10, "prevalence": 0.466, "cstatistic": 0.670},
        {"name": "Serial biopsy", "n_actual": 70,  "events_actual": 34,  "parameters": 2,  "prevalence": 0.486, "cstatistic": 0.660},
        {"name": "ESRD 5-Year",   "n_actual": 796, "events_actual": 112, "parameters": 5,  "prevalence": 0.141, "cstatistic": 0.797},
        {"name": "ESRD 10-Year",  "n_actual": 796, "events_actual": 175, "parameters": 17, "prevalence": 0.220, "cstatistic": 0.817},
    ]

    rows = []
    for c in cohorts:
        r = pmsampsize_binary(c["parameters"], c["prevalence"], c["cstatistic"])
        epv_actual = c["events_actual"] / c["parameters"]
        adequate = c["n_actual"] >= r["n_required"]
        rows.append({
            "Cohort": c["name"],
            "Parameters (p)": c["parameters"],
            "Prevalence": c["prevalence"],
            "Anticipated C-statistic (LR, BC AUROC)": c["cstatistic"],
            "Derived Cox-Snell R2": round(r["cox_snell_r2"], 4),
            "n required (pmsampsize)": r["n_required"],
            "Events required": round(r["events_required"], 1),
            "EPP required": round(r["EPP_required"], 2),
            "n actual": c["n_actual"],
            "Events actual": c["events_actual"],
            "EPV actual": round(epv_actual, 2),
            "Adequate per pmsampsize?": "Yes" if adequate else "No",
        })

    df = pd.DataFrame(rows)
    print(df.to_string(index=False))

    out_path = f"{OUT}/pmsampsize_results.xlsx"
    df.to_excel(out_path, index=False)
    print(f"\nSaved: {out_path}")
