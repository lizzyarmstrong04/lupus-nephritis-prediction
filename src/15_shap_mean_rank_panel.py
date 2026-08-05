"""
Combined 2x2 panel figure: SHAP mean-rank across models for all four cohorts
(1-year flare, 5-year flare, ESRD 5-year, ESRD 10-year).

For each cohort, each model's SHAP importance column is converted to a rank
(1 = most important feature for that model), then averaged across the four
models to give "mean rank across models" (lower = more important).
"""

import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

BASE    = "/Users/elizabetharmstrong/Library/CloudStorage/OneDrive-ImperialCollegeLondon/Lupus Project"
OUT_DIR = f"{BASE}/outputs/figures"

MODEL_COLS = ["Logistic Regression", "Random Forest", "XGBoost", "LightGBM"]
BAR_COLOR  = "#4C72B0"

SHORT_NAMES_1YR = {
    "% chronic gloms(%of total)":                                                         "% chronic gloms",
    "%gloms with necrosis":                                                                "% necrosis",
    "Age at biopsy":                                                                       "Age at biopsy",
    "Proteinuria at biopsy (uPCR, log)":                                                   "Proteinuria (log)",
    "Class coded 1=I 2=II 3=III 4=IV 5=V 6=III+V 7=IV+V 8=II+V 9=VI 10=other":          "LN class",
    "Ethnicity 1=white 2=black 3=asian (south) 4=asian (east) 5=other 6=not stated/unknown/any other mixed": "Ethnicity",
    "% active gloms (%of those not globally sclerosed)":                                   "% active gloms",
    "%gloms with crescents":                                                               "% crescents",
    "C4 at biopsy":                                                                        "C4 at biopsy",
}

SHORT_NAMES_5YR = {
    "% chronic gloms(%of total)":                                                                                                                                    "% chronic gloms",
    "%gloms with necrosis":                                                                                                                                          "% necrosis",
    "Class coded 1=I 2=II 3=III 4=IV 5=V 6=III+V 7=IV+V 8=II+V 9=VI 10=other":                                                                                    "LN class",
    "% active gloms (%of those not globally sclerosed)":                                                                                                             "% active gloms",
    "Prev exposure to cyclo (for Rx comparison) - related to the 'use this biopsy for this patient' biopsy":                                                         "Prev cyclophosphamide",
    "% sclerosed gloms":                                                                                                                                             "% sclerosed gloms",
    "CKD epi formula without ethnicity":                                                                                                                             "eGFR (CKD-EPI)",
    "Reason for biopsy 1=new pres LN 2=relapse 3=non-response/partial response, incl on-going proteinuria 4=pre-pregnancy or Ax if drug switch/stop appropriate":   "Biopsy reason",
    "Ethnicity 1=white 2=black 3=asian (south) 4=asian (east) 5=other 6=not stated/unknown/any other mixed":                                                         "Ethnicity",
    "Age at biopsy":                                                                                                                                                 "Age at biopsy",
}

SHORT_NAMES_ESRD_5YR = {
    "Creatinine at biopsy":              "Creatinine at biopsy",
    "Subepithelial deposit category (0=no deposits, 1=small/rare deposits, 2=large/conspicuous deposits, 3=no gloms on EM)": "Subepithelial deposits",
    "CKD epi formula without ethnicity": "eGFR (CKD-EPI)",
    "% chronic gloms(%of total)":        "% chronic gloms",
    "%IFTA ":                            "% IFTA",
}

SHORT_NAMES_ESRD_10YR = {
    "Creatinine at biopsy":              "Creatinine at biopsy",
    "%IFTA ":                            "% IFTA",
    "Age at biopsy":                     "Age at biopsy",
    "% chronic gloms(%of total)":        "% chronic gloms",
    "CKD epi formula without ethnicity": "eGFR (CKD-EPI)",
    "C3 at biopsy (normal range 0.7-1.7)": "C3 at biopsy",
    "C4 low (for range 0.15-0.54)":      "C4 low",
    "Crescents (Yes=1, No=0)":           "Crescents (present)",
    "Prev exposure to cyclo (for Rx comparison) - related to the 'use this biopsy for this patient' biopsy": "Prev cyclophosphamide",
    "Cap wall IgM":                      "Capillary wall IgM",
    "Biopsy number for patient":         "Biopsy number",
    "TMA (Yes=1, No=0)":                 "TMA (present)",
    "Subepithelial deposit category (0=no deposits, 1=small/rare deposits, 2=large/conspicuous deposits, 3=no gloms on EM)": "Subepithelial deposits",
    "Class coded 1=I 2=II 3=III 4=IV 5=V 6=III+V 7=IV+V 8=II+V 9=VI 10=other": "LN class",
    "No. globally sclerosed gloms ":     "No. globally sclerosed gloms",
    "Gender (1=male, 2=female)":         "Gender",
    "No. gloms with crescents":          "No. gloms with crescents",
}

COHORTS = [
    ("A", "1-Year Flare",  f"{BASE}/outputs/shap_importance_table.xlsx",          SHORT_NAMES_1YR),
    ("B", "5-Year Flare",  f"{BASE}/outputs/shap_importance_table_5yr.xlsx",      SHORT_NAMES_5YR),
    ("C", "ESRD 5-Year",   f"{BASE}/outputs/esrd/esrd_shap_table_5yr.xlsx",       SHORT_NAMES_ESRD_5YR),
    ("D", "ESRD 10-Year",  f"{BASE}/outputs/esrd/esrd_shap_table_10yr.xlsx",      SHORT_NAMES_ESRD_10YR),
]


def resolve_short_name(raw, short_names):
    if raw in short_names:
        return short_names[raw]
    stripped = raw.rstrip("…").strip()
    for full, short in short_names.items():
        if full.startswith(stripped):
            return short
    return raw


def mean_rank_table(path, short_names):
    df = pd.read_excel(path)
    df["Feature"] = df["Feature"].apply(lambda f: resolve_short_name(f, short_names))
    ranks = df[MODEL_COLS].rank(ascending=False, method="average")
    df["Mean rank across models"] = ranks.mean(axis=1)
    return df[["Feature", "Mean rank across models"]].sort_values(
        "Mean rank across models", ascending=False
    )


fig, axes = plt.subplots(2, 2, figsize=(12, 10))
axes = axes.flatten()

for ax, (letter, title, path, short_names) in zip(axes, COHORTS):
    d = mean_rank_table(path, short_names)

    ax.barh(d["Feature"], d["Mean rank across models"], color=BAR_COLOR, alpha=0.85)
    ax.set_xlabel("Mean rank across models (lower = more important)", fontsize=10)
    ax.tick_params(axis="both", labelsize=10)
    ax.set_title(f"{letter}. {title}", fontsize=12, fontweight="bold", loc="left")
    ax.text(-0.1, 1.05, letter, transform=ax.transAxes,
            fontweight="bold", fontsize=14)
    ax.spines[["top", "right"]].set_visible(False)

plt.tight_layout()

png_path = f"{OUT_DIR}/shap_mean_rank_panel.png"
svg_path = f"{OUT_DIR}/shap_mean_rank_panel.svg"
tiff_path = f"{OUT_DIR}/shap_mean_rank_panel.tiff"

fig.savefig(png_path, dpi=300, bbox_inches="tight")
fig.savefig(svg_path, dpi=300, bbox_inches="tight")
fig.savefig(tiff_path, dpi=300, bbox_inches="tight")
plt.close("all")

print(f"Saved: {png_path}")
print(f"Saved: {svg_path}")
print(f"Saved: {tiff_path}")
