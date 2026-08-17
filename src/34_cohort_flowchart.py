"""
Data-cleaning / cohort-construction flow diagram: raw dataset -> 5 final
analytic cohorts (1-Year Flare, 5-Year Flare, Serial Biopsy, ESRD 5-Year,
ESRD 10-Year).

Layout: each of the 4 leaf columns (1-Year, 5-Year, ESRD-5Yr, ESRD-10Yr)
runs its own strictly vertical, inline sequence of boxes (split point ->
exclusion box -> next box -> ...) - guarantees no horizontal collisions
between neighbouring columns regardless of how many exclusion steps a
given branch needs, since box widths are chosen safely within each
column's own lane. Splits use T-junction connectors (vertical stub ->
horizontal bus -> vertical drops), so no diagonal lines cross. Same visual
style as the Introduction pathway figure (src/32_intro_pathway_figure.py):
pastel category fills + matching darker border/heading text, thin uniform
borders, Times New Roman. Exclusion-step boxes are visually distinguished
(dashed border, off-white fill, italic) from the main pipeline/final
cohort boxes. A4-portrait-proportioned, scaled for readable (not tiny)
text across the whole page.

All counts below were read from re-running the actual cohort-construction
scripts against the raw data (not retyped from memory/estimated):
  src/1_year/01_data_prep_1yr.py     -> 1-Year Flare exclusions/counts
  src/5_year/01_data_prep_5yr.py     -> 5-Year Flare exclusions/counts
  src/5_year/08_serial_biopsy_5yr.py -> Serial Biopsy exclusions/counts
  src/esrd/00_esrd_feature_selection.py -> ESRD 5-/10-Year counts (this is
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
EXCL    = {"fill": "#fcfcfb", "edge": "#a8a8a2", "text": "#4a4a46"}
ARROW_COL = "#4a4a46"

FIG_W, FIG_H = 9.2, 14.0

fig, ax = plt.subplots(figsize=(FIG_W, FIG_H))
ax.axis("off")
fig.patch.set_facecolor("white")
ax.set_facecolor("white")

MAIN_W, MAIN_H = 25, 13
EXCL_W, EXCL_H = 23, 13
GAP = 8          # arrow length between two stacked boxes in the same column


def box(cx, cy, w, h, title, subtitle, style, dashed=False, italic=False,
        title_size=12, sub_size=10.3):
    x, y = cx - w / 2, cy - h / 2
    patch = FancyBboxPatch(
        (x, y), w, h, boxstyle="round,pad=0,rounding_size=0.5",
        linewidth=1.1 if dashed else 1.3, edgecolor=style["edge"], facecolor=style["fill"],
        linestyle=(0, (4, 2)) if dashed else "solid", zorder=3,
    )
    ax.add_patch(patch)
    lines = subtitle.split("\n") if subtitle else []
    if lines:
        # Adaptive centering: total text block (title + N subtitle lines)
        # is spaced to fit within a fixed fraction of the box height
        # regardless of N, so 3-line subtitles can't overflow below the
        # box the way a fixed-step layout would (caught visually - the
        # last line of every 3-line exclusion box was rendering outside
        # the box before this fix).
        n_rows = len(lines) + 1
        usable_h = h * 0.80
        row_h = usable_h / n_rows
        top_y = cy + usable_h / 2 - row_h / 2
        ax.text(cx, top_y, title, ha="center", va="center",
                 fontsize=title_size, fontweight="bold", fontstyle="italic" if italic else "normal",
                 color=style["text"], zorder=4)
        for i, line in enumerate(lines):
            ax.text(cx, top_y - (i + 1) * row_h, line, ha="center", va="center",
                     fontsize=sub_size, fontstyle="italic" if italic else "normal",
                     color=style["text"], zorder=4)
    else:
        ax.text(cx, cy, title, ha="center", va="center", fontsize=title_size,
                 fontweight="bold", fontstyle="italic" if italic else "normal",
                 color=style["text"], zorder=4)
    return y, y + h  # (bottom, top)


def vline(x, y1, y2, arrow_head=True, lw=1.3):
    if arrow_head:
        arr = FancyArrowPatch((x, y1), (x, y2), arrowstyle="-|>", mutation_scale=14,
                               linewidth=lw, color=ARROW_COL, zorder=2)
        ax.add_patch(arr)
    else:
        ax.plot([x, x], [y1, y2], color=ARROW_COL, linewidth=lw, zorder=2, solid_capstyle="butt")


def hline(x1, x2, y, lw=1.3):
    ax.plot([x1, x2], [y, y], color=ARROW_COL, linewidth=lw, zorder=2, solid_capstyle="butt")


def t_split(parent_x, parent_bottom, child_xs, child_top, stub=5):
    """Vertical stub down from parent, horizontal bus, vertical arrow into
    each child column - classic CONSORT T-junction, no diagonal lines."""
    bus_y = parent_bottom - stub
    vline(parent_x, parent_bottom, bus_y, arrow_head=False)
    hline(min(child_xs), max(child_xs), bus_y)
    for cx in child_xs:
        vline(cx, bus_y, child_top)


def excl(cx, cy, text):
    return box(cx, cy, EXCL_W, EXCL_H, "Excluded", text, EXCL, dashed=True, italic=True,
               title_size=10, sub_size=9)


def stack_down(prev_bottom):
    """Return (top, cy) for the next box, `GAP` below `prev_bottom`."""
    top = prev_bottom - GAP
    return top, top - MAIN_H / 2


# ---------------------------------------------------------------------------
X_1YR, X_5YR, X_ESRD5, X_ESRD10 = 10, 38, 66, 94
X_RAW = (X_5YR + X_ESRD5) / 2   # 52, centred over the two middle columns
X_ELIG = (X_1YR + X_5YR) / 2    # 24, centred over the two flare columns

y = 128  # running cursor, top of current box

# --- Raw dataset ---
top, bot = y, y - MAIN_H
box(X_RAW, y - MAIN_H / 2, 56, MAIN_H, "Raw Dataset",
    "data_lupus.xlsx   —   n = 1,070 biopsy episodes, 207 variables", NEUTRAL,
    title_size=13.5, sub_size=11)

t_split(X_RAW, bot, [X_ELIG, X_ESRD5, X_ESRD10], bot - 5 - 4)

# ============================== ESRD branch ==============================
esrd_excl_top = bot - 5 - 4
_, ecy = esrd_excl_top, esrd_excl_top - EXCL_H / 2
esrd5_excl_bot, _ = excl(X_ESRD5, ecy, "n = 274:\nmissing/insufficient\n5-year follow-up")
esrd10_excl_bot, _ = excl(X_ESRD10, ecy, "n = 274:\nmissing/insufficient\n10-year follow-up")

vline(X_ESRD5, esrd5_excl_bot, esrd5_excl_bot - GAP)
vline(X_ESRD10, esrd10_excl_bot, esrd10_excl_bot - GAP)
fin_cy = esrd5_excl_bot - GAP - MAIN_H / 2
box(X_ESRD5, fin_cy, MAIN_W, MAIN_H, "ESRD 5-Year", "n = 796\nEvents = 112 (14.1%)", RED)
box(X_ESRD10, fin_cy, MAIN_W, MAIN_H, "ESRD 10-Year", "n = 796\nEvents = 175 (22.0%)", RED)

# ============================== Flare branch ==============================
elig_top = esrd_excl_top
elig_cy = elig_top - MAIN_H / 2
elig_bot = elig_top - MAIN_H
box(X_ELIG, elig_cy, 40, MAIN_H, "Eligible for Flare-Outcome Analysis", "n = 893", NEUTRAL,
    title_size=10.8, sub_size=10.3)
excl_note_top, excl_note_cy = elig_top + GAP - GAP, None  # unused placeholder (kept for clarity)

t_split(X_ELIG, elig_bot, [X_1YR, X_5YR], elig_bot - 5 - 4)

split_excl_top = elig_bot - 5 - 4
excl_cy = split_excl_top - EXCL_H / 2
yr1_excl_bot, _ = excl(X_1YR, excl_cy, "n = 463:\ninadequate/unusable\n1-year outcome data")
yr5_excl_bot, _ = excl(X_5YR, excl_cy, "n = 537:\ninadequate/unusable\n5-year outcome data")

vline(X_1YR, yr1_excl_bot, yr1_excl_bot - GAP)
vline(X_5YR, yr5_excl_bot, yr5_excl_bot - GAP)
flare_fin_cy = yr1_excl_bot - GAP - MAIN_H / 2
yr1_bot, _ = box(X_1YR, flare_fin_cy, MAIN_W, MAIN_H, "1-Year Flare", "n = 430\nEvents = 99 (23.0%)", BLUE)
yr5_bot, _ = box(X_5YR, flare_fin_cy, MAIN_W, MAIN_H, "5-Year Flare", "n = 356\nEvents = 166 (46.6%)", BLUE)
yr5_bot = flare_fin_cy - MAIN_H / 2

# --- Serial Biopsy sub-branch (from 5-Year Flare) ---
vline(X_5YR, yr5_bot, yr5_bot - GAP)
ser_excl_top = yr5_bot - GAP
ser_excl_cy = ser_excl_top - EXCL_H / 2
ser_excl_bot, _ = excl(X_5YR, ser_excl_cy,
                       "259 unique patients ->\nn = 189: only 1 biopsy\navailable")
vline(X_5YR, ser_excl_bot, ser_excl_bot - GAP)
ser_fin_cy = ser_excl_bot - GAP - MAIN_H / 2
box(X_5YR, ser_fin_cy, 30, MAIN_H, "Serial Biopsy",
    "n = 70 patients (≥ 2 biopsies)\nEvents = 34 (48.6%)", VIOLET, title_size=12, sub_size=10.3)
serial_bottom = ser_fin_cy - MAIN_H / 2

# --- Title & footnote ---
fig.text(0.5, 0.985, "Cohort Construction: Raw Data to Final Analytic Cohorts",
         ha="center", fontsize=16, fontweight="bold", fontfamily="Times New Roman")

footnote_y = serial_bottom - 8
ax.text(
    X_RAW, footnote_y,
    "All five cohorts derive from the same raw dataset (data_lupus.xlsx). ESRD outcomes were assessed\n"
    "independently at 5 and 10 years (composite: creatinine doubling, RRT, or death on RRT vs. eGFR>80\n"
    "stable; death alone and sub-threshold eGFR decline treated as competing/ambiguous events, excluded).\n"
    "Serial Biopsy is a sub-cohort of the 5-Year Flare cohort (patients with ≥ 2 biopsies).",
    ha="center", va="top", fontsize=10.3, color="#3a3a36", linespacing=1.7,
)

ax.set_xlim(-3, 107)
ax.set_ylim(footnote_y - 12, 129.5)
fig.subplots_adjust(left=0.01, right=0.99, top=0.965, bottom=0.01)

fig.savefig(f"{FIG_DIR}/cohort_flowchart.png", dpi=300, bbox_inches="tight", facecolor="white")
fig.savefig(f"{FIG_DIR}/cohort_flowchart.pdf", bbox_inches="tight", facecolor="white")
print(f"Saved: {FIG_DIR}/cohort_flowchart.png")
print(f"Saved: {FIG_DIR}/cohort_flowchart.pdf")
