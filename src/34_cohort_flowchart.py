"""
Data-cleaning / cohort-construction flow diagram: raw dataset -> 5 final
analytic cohorts (1 Year Flare, 5 Year Flare, Serial Biopsy, ESRD 5 Year,
ESRD 10 Year).

Compact version: exclusion counts are small borderless italic text labels
directly on each column's arrow (not full boxes) - far less vertical space
per step than the original boxed-exclusion layout, so the whole diagram
fits in well under half an A4 page height at full page width. No hyphens
anywhere in the rendered text (per request); the raw-data box no longer
names the source Excel file, just the row/column counts.

Layout: each of the 4 leaf columns (1 Year, 5 Year, ESRD 5yr, ESRD 10yr)
runs its own strictly vertical, inline sequence (split point -> exclusion
label -> next box -> ...) - no horizontal collisions between neighbouring
columns regardless of branch depth. Splits use T-junction connectors
(vertical stub -> horizontal bus -> vertical drops), no diagonal lines.
Same visual style as the Introduction pathway figure
(src/32_intro_pathway_figure.py): pastel category fills + matching darker
border/heading text, thin uniform box borders (rounding_size=0.55, same
as that figure), Times New Roman.

All counts were read from re-running the actual cohort-construction
scripts against the raw data (not retyped from memory/estimated):
  src/1_year/01_data_prep_1yr.py     -> 1 Year Flare exclusions/counts
  src/5_year/01_data_prep_5yr.py     -> 5 Year Flare exclusions/counts
  src/5_year/08_serial_biopsy_5yr.py -> Serial Biopsy exclusions/counts
  src/esrd/00_esrd_feature_selection.py -> ESRD 5/10 Year counts (this is
    the actually-used ESRD pipeline; src/esrd/01_data_prep_esrd.py is an
    earlier/superseded exploratory script with a stricter, different
    outcome definition and is NOT the source of the published n=796
    cohorts - confirmed by running both and comparing to the published
    Table S1/S3 figures)

Saves: outputs/figures/cohort_flowchart.png (300 dpi)
       outputs/figures/cohort_flowchart.pdf (vector)
"""
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

BASE = "/Users/elizabetharmstrong/Library/CloudStorage/OneDrive-ImperialCollegeLondon/Lupus Project"
FIG_DIR = f"{BASE}/outputs/figures"

plt.rcParams["font.family"] = "Times New Roman"

NEUTRAL = {"fill": "#f2f2f0", "edge": "#6b6b66", "text": "#2b2b28"}
BLUE    = {"fill": "#e4eefa", "edge": "#2a5faa", "text": "#173e73"}
VIOLET  = {"fill": "#ece8f8", "edge": "#5c4aa8", "text": "#3a2d70"}
RED     = {"fill": "#fbe9e8", "edge": "#b33b38", "text": "#7d2926"}
ARROW_COL = "#4a4a46"
LABEL_COL = "#55554f"

FIG_W, FIG_H = 10.2, 7.6   # short/wide - fits well under half an A4 page at full width

fig, ax = plt.subplots(figsize=(FIG_W, FIG_H))
ax.axis("off")
fig.patch.set_facecolor("white")
ax.set_facecolor("white")

MAIN_W, MAIN_H = 25, 11.5
GAP = 3           # arrow segment length


def box(cx, cy, w, h, title, subtitle, style, title_size=12.5, sub_size=10.5):
    x, y = cx - w / 2, cy - h / 2
    patch = FancyBboxPatch(
        (x, y), w, h, boxstyle="round,pad=0,rounding_size=0.55",  # matches src/32_intro_pathway_figure.py
        linewidth=1.3, edgecolor=style["edge"], facecolor=style["fill"], zorder=3,
    )
    ax.add_patch(patch)
    lines = subtitle.split("\n") if subtitle else []
    if lines:
        n_rows = len(lines) + 1
        usable_h = h * 0.78
        row_h = usable_h / n_rows
        top_y = cy + usable_h / 2 - row_h / 2
        ax.text(cx, top_y, title, ha="center", va="center", fontsize=title_size,
                 fontweight="bold", color=style["text"], zorder=4)
        for i, line in enumerate(lines):
            ax.text(cx, top_y - (i + 1) * row_h, line, ha="center", va="center",
                     fontsize=sub_size, color=style["text"], zorder=4)
    else:
        ax.text(cx, cy, title, ha="center", va="center", fontsize=title_size,
                 fontweight="bold", color=style["text"], zorder=4)
    return y, y + h  # (bottom, top)


def vline(x, y1, y2, arrow_head=True, lw=1.2):
    if arrow_head:
        arr = FancyArrowPatch((x, y1), (x, y2), arrowstyle="-|>", mutation_scale=12,
                               linewidth=lw, color=ARROW_COL, zorder=2)
        ax.add_patch(arr)
    else:
        ax.plot([x, x], [y1, y2], color=ARROW_COL, linewidth=lw, zorder=2, solid_capstyle="butt")


def hline(x1, x2, y, lw=1.2):
    ax.plot([x1, x2], [y, y], color=ARROW_COL, linewidth=lw, zorder=2, solid_capstyle="butt")


def t_split(parent_x, parent_bottom, child_xs, child_top, stub=2.2):
    bus_y = parent_bottom - stub
    vline(parent_x, parent_bottom, bus_y, arrow_head=False)
    hline(min(child_xs), max(child_xs), bus_y)
    for cx in child_xs:
        vline(cx, bus_y, child_top)


def excl_step(cx, top, text, gap_above=4.5, label_h=8.5, gap_below=4.5):
    """Compact exclusion step: short arrow down to the TOP of a small
    italic text label (no box/border), text block, then a second short
    arrow down from the label's BOTTOM into the next box. The connector
    lines only ever span the gaps outside the text block's own [top,
    bottom] span - never through it - since text is centred on the same
    x as the line (any y-overlap would draw the line through the glyphs,
    caught visually before this fix). Returns the y of the next box's top."""
    block_top = top - gap_above
    vline(cx, top, block_top, arrow_head=False)

    lines = text.split("\n")
    n = max(len(lines), 1)
    step = label_h / n
    for i, line in enumerate(lines):
        line_y = block_top - step / 2 - i * step
        ax.text(cx, line_y, line, ha="center", va="center",
                 fontsize=8.7, style="italic", color=LABEL_COL, zorder=4)
    block_bottom = block_top - label_h

    next_top = block_bottom - gap_below
    vline(cx, block_bottom, next_top)
    return next_top


# ---------------------------------------------------------------------------
X_1YR, X_5YR, X_ESRD5, X_ESRD10 = 9, 36, 63, 90
X_RAW = (X_5YR + X_ESRD5) / 2
X_ELIG = (X_1YR + X_5YR) / 2

y = 72

# --- Raw dataset (no filename - just the counts) ---
bot, top = box(X_RAW, y - MAIN_H / 2, 54, MAIN_H, "Raw Dataset",
                "n = 1,070 biopsy episodes, 207 variables", NEUTRAL,
                title_size=13.5, sub_size=11)

t_split(X_RAW, bot, [X_ELIG, X_ESRD5, X_ESRD10], bot - 4)
branch_top = bot - 4

# ============================== ESRD branch ==============================
esrd5_next_top = excl_step(X_ESRD5, branch_top, "n = 274 excluded:\nmissing 5 year follow up")
esrd10_next_top = excl_step(X_ESRD10, branch_top, "n = 274 excluded:\nmissing 10 year follow up")
esrd_fin_cy = esrd5_next_top - MAIN_H / 2
box(X_ESRD5, esrd_fin_cy, MAIN_W, MAIN_H, "ESRD 5 Year", "n = 796\nEvents = 112 (14.1%)", RED)
box(X_ESRD10, esrd_fin_cy, MAIN_W, MAIN_H, "ESRD 10 Year", "n = 796\nEvents = 175 (22.0%)", RED)

# ============================== Flare branch ==============================
elig_bot, elig_top = box(X_ELIG, branch_top - MAIN_H / 2, 38, MAIN_H,
                          "Eligible for Flare Analysis", "n = 893", NEUTRAL,
                          title_size=11.5, sub_size=11)

t_split(X_ELIG, elig_bot, [X_1YR, X_5YR], elig_bot - 4)
flare_split_top = elig_bot - 4

yr1_next_top = excl_step(X_1YR, flare_split_top, "n = 463 excluded:\ninadequate outcome data")
yr5_next_top = excl_step(X_5YR, flare_split_top, "n = 537 excluded:\ninadequate outcome data")
flare_fin_cy = yr1_next_top - MAIN_H / 2
_, _ = box(X_1YR, flare_fin_cy, MAIN_W, MAIN_H, "1 Year Flare", "n = 430\nEvents = 99 (23.0%)", BLUE)
yr5_bot, _ = box(X_5YR, flare_fin_cy, MAIN_W, MAIN_H, "5 Year Flare", "n = 356\nEvents = 166 (46.6%)", BLUE)

# --- Serial Biopsy sub-branch (from 5 Year Flare) ---
ser_next_top = excl_step(X_5YR, yr5_bot, "189 of 259 patients excluded:\nonly 1 biopsy available")
ser_fin_cy = ser_next_top - MAIN_H / 2
_, _ = box(X_5YR, ser_fin_cy, 30, MAIN_H, "Serial Biopsy",
           "n = 70 patients (≥ 2 biopsies)\nEvents = 34 (48.6%)", VIOLET)
serial_bottom = ser_fin_cy - MAIN_H / 2

# --- Title & footnote ---
fig.text(0.5, 0.975, "Cohort Construction: Raw Data to Final Analytic Cohorts",
         ha="center", fontsize=16.5, fontweight="bold", fontfamily="Times New Roman")

footnote_y = serial_bottom - 3
ax.text(
    X_RAW, footnote_y,
    "All five cohorts derive from the same raw dataset. ESRD outcomes assessed independently at 5 and 10\n"
    "years (composite: creatinine doubling, RRT, or death on RRT vs. eGFR>80 stable; death alone and\n"
    "sub threshold eGFR decline treated as competing/ambiguous events, excluded). Serial Biopsy is a\n"
    "subcohort of the 5 Year Flare cohort (patients with ≥ 2 biopsies).",
    ha="center", va="top", fontsize=9, color="#3a3a36", linespacing=1.55,
)

ax.set_xlim(-3, 107)
ax.set_ylim(footnote_y - 8, 78)
fig.subplots_adjust(left=0.01, right=0.99, top=0.94, bottom=0.01)

fig.savefig(f"{FIG_DIR}/cohort_flowchart.png", dpi=300, bbox_inches="tight", facecolor="white")
fig.savefig(f"{FIG_DIR}/cohort_flowchart.pdf", bbox_inches="tight", facecolor="white")
print(f"Saved: {FIG_DIR}/cohort_flowchart.png")
print(f"Saved: {FIG_DIR}/cohort_flowchart.pdf")
