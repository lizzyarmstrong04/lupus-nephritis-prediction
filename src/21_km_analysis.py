"""
Kaplan-Meier analysis: survival curves stratified by model-predicted risk tier.
Panels:
  A — 1-year flare model → time-to-flare (n=430)
  B — 5-year flare model → time-to-flare (n=356)
  C — 1-year flare model → time-to-RRT / ESRD (n=430)
Log-rank p-values shown. Times in years from biopsy.
"""

import warnings, re
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import joblib
from lifelines import KaplanMeierFitter
from lifelines.statistics import multivariate_logrank_test, logrank_test

BASE   = "/Users/elizabetharmstrong/Library/CloudStorage/OneDrive-ImperialCollegeLondon/Lupus Project"
MODELS = f"{BASE}/src/app/models"
OUT    = f"{BASE}/outputs/figures"

# ── Youden thresholds (same as app) ───────────────────────────────────────
THRESH = {"1yr": 0.410, "5yr": 0.509, "esrd_5yr": 0.482, "esrd_10yr": 0.470}

def risk_tier(p, key):
    t = THRESH[key]
    if p < t * 0.60: return "Low"
    if p < t:        return "Moderate"
    return                   "High"

# ── Tier colours ──────────────────────────────────────────────────────────
TIER_COLORS = {"Low": "#007F3B", "Moderate": "#ED8B00", "High": "#DA291C"}
TIER_ORDER  = ["Low", "Moderate", "High"]

# ── Time-to-event from dates ───────────────────────────────────────────────
def days_between(d1, d2):
    return (pd.to_datetime(d2) - pd.to_datetime(d1)).dt.days / 365.25

# ── Load data ──────────────────────────────────────────────────────────────
df1 = pd.read_excel(f"{BASE}/Data/Processed/lupus_1yr_imputed.xlsx")
df5 = pd.read_excel(f"{BASE}/Data/Processed/lupus_5yr_imputed.xlsx")

# ── Feature columns (match what each model was trained on) ────────────────
feat1 = joblib.load(f"{MODELS}/feature_cols.joblib")["1yr"]
feat5 = joblib.load(f"{MODELS}/feature_cols.joblib")["5yr"]

lr1 = joblib.load(f"{MODELS}/1yr_lr.joblib")
lr5 = joblib.load(f"{MODELS}/5yr_lr.joblib")

# ── Compute predicted probabilities ───────────────────────────────────────
prob1 = lr1.predict_proba(df1[feat1])[:, 1]
prob5 = lr5.predict_proba(df5[feat5])[:, 1]

tier1 = [risk_tier(p, "1yr") for p in prob1]
tier5 = [risk_tier(p, "5yr") for p in prob5]

# ── Build survival datasets ────────────────────────────────────────────────
# 1yr cohort — time to flare
flare_col = [c for c in df1.columns if "***Flare" in c][0]
last_fu    = "Date last follow-up, note 06/11/2018 taken as last follow-up if not lost to follow-up"

t_flare1 = np.where(
    df1[flare_col] == 1,
    days_between(df1["Biopsy date"], df1["Date of flare"]),
    days_between(df1["Biopsy date"], df1[last_fu])
)
e_flare1 = (df1[flare_col] == 1).astype(int).values

# Clip at 15 years and remove negative/zero times
valid1 = (t_flare1 > 0) & (t_flare1 <= 15)
t_flare1_c = np.clip(t_flare1[valid1], 0.01, 15)
e_flare1_c = e_flare1[valid1]
tier1_c    = np.array(tier1)[valid1]

# 5yr cohort — time to flare
t_flare5 = np.where(
    df5[flare_col] == 1,
    days_between(df5["Biopsy date"], df5["Date of flare"]),
    days_between(df5["Biopsy date"], df5[last_fu])
)
e_flare5 = (df5[flare_col] == 1).astype(int).values

valid5 = (t_flare5 > 0) & (t_flare5 <= 15)
t_flare5_c = np.clip(t_flare5[valid5], 0.01, 15)
e_flare5_c = e_flare5[valid5]
tier5_c    = np.array(tier5)[valid5]

# 1yr cohort — time to RRT (ESRD)
rrt_col = "RRT No=0, Yes=1"
t_rrt = np.where(
    df1[rrt_col] == 1,
    days_between(df1["Biopsy date"], df1["Date RRT"]),
    days_between(df1["Biopsy date"], df1[last_fu])
)
e_rrt = df1[rrt_col].values

valid_rrt = (t_rrt > 0) & (t_rrt <= 20)
t_rrt_c   = np.clip(t_rrt[valid_rrt], 0.01, 20)
e_rrt_c   = e_rrt[valid_rrt]
tier_rrt  = np.array(tier1)[valid_rrt]

print(f"1yr flare cohort: n={valid1.sum()}, events={e_flare1_c.sum()}")
for t in TIER_ORDER:
    mask = tier1_c == t
    print(f"  {t}: n={mask.sum()}, events={e_flare1_c[mask].sum()}")

print(f"\n5yr flare cohort: n={valid5.sum()}, events={e_flare5_c.sum()}")
for t in TIER_ORDER:
    mask = tier5_c == t
    print(f"  {t}: n={mask.sum()}, events={e_flare5_c[mask].sum()}")

print(f"\n1yr → RRT cohort: n={valid_rrt.sum()}, events={e_rrt_c.sum()}")
for t in TIER_ORDER:
    mask = tier_rrt == t
    print(f"  {t}: n={mask.sum()}, events={e_rrt_c[mask].sum()}")

# ── KM plot helper ─────────────────────────────────────────────────────────
def km_panel(ax, times, events, tiers, title, xlabel, max_t, letter):
    present = [t for t in TIER_ORDER if t in tiers]
    kmfs = {}
    for tier in present:
        mask = tiers == tier
        if mask.sum() < 3:
            continue
        kmf = KaplanMeierFitter()
        kmf.fit(times[mask], events[mask], label=tier)
        n      = mask.sum()
        n_ev   = events[mask].sum()
        label  = f"{tier}  (n={n}, events={n_ev})"
        kmf.plot_survival_function(
            ax=ax,
            color=TIER_COLORS[tier],
            ci_show=True,
            ci_alpha=0.10,
            label=label,
            linewidth=1.6,
        )
        kmfs[tier] = (times[mask], events[mask])

    # Log-rank p-value (multivariate across all tiers)
    if len(kmfs) >= 2:
        result = multivariate_logrank_test(times, events, tiers)
        p = result.p_value
        p_str = f"p < 0.001" if p < 0.001 else f"p = {p:.3f}"
        ax.text(0.97, 0.96, f"Log-rank {p_str}",
                transform=ax.transAxes, ha="right", va="top",
                fontsize=7, style="italic", color="#333333")

    ax.set_xlim(0, max_t)
    ax.set_ylim(-0.02, 1.05)
    ax.set_xlabel(xlabel, fontsize=8, labelpad=3)
    ax.set_ylabel("Survival probability", fontsize=8, labelpad=3)
    ax.set_title(title, fontsize=9, pad=5)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(axis="y", color="#e8e8e8", linewidth=0.4, zorder=0)
    ax.set_axisbelow(True)
    ax.tick_params(labelsize=7)

    leg = ax.legend(loc="lower left", fontsize=6.5, framealpha=0.9,
                    edgecolor="#cccccc", borderpad=0.5)

    ax.text(-0.14, 1.06, letter,
            transform=ax.transAxes,
            fontsize=10, fontweight="bold", va="top", ha="left")

# ── Figure ─────────────────────────────────────────────────────────────────
matplotlib.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
    "font.size": 8,
    "figure.dpi": 300,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "axes.linewidth": 0.6,
    "xtick.major.width": 0.6,
    "ytick.major.width": 0.6,
    "xtick.major.size": 3,
    "ytick.major.size": 3,
})

fig = plt.figure(figsize=(10.5, 3.8))
gs  = gridspec.GridSpec(1, 3, figure=fig,
                        left=0.07, right=0.98,
                        top=0.91, bottom=0.14,
                        wspace=0.40)

ax1 = fig.add_subplot(gs[0, 0])
ax2 = fig.add_subplot(gs[0, 1])
ax3 = fig.add_subplot(gs[0, 2])

km_panel(ax1, t_flare1_c, e_flare1_c, tier1_c,
         "1-Year Flare Model\nTime to Any Flare  (n=430)",
         "Years from biopsy", 15, "A")

km_panel(ax2, t_flare5_c, e_flare5_c, tier5_c,
         "5-Year Flare Model\nTime to Any Flare  (n=356)",
         "Years from biopsy", 15, "B")

km_panel(ax3, t_rrt_c, e_rrt_c, tier_rrt,
         "1-Year Flare Model\nTime to Renal Replacement Therapy  (n=430)",
         "Years from biopsy", 20, "C")

out_pdf = f"{OUT}/km_analysis.pdf"
out_png = f"{OUT}/km_analysis.png"
fig.savefig(out_pdf, format="pdf")
fig.savefig(out_png, format="png", dpi=300)
plt.close()

print(f"\nSaved: {out_pdf}")
print(f"Saved: {out_png}")
