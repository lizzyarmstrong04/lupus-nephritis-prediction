"""
Introduction figure: SLE -> Lupus Nephritis -> Renal Biopsy -> Induction ->
Response/Flare -> Sustained Remission/ESRD pathway diagram.

Redrawn (not a literal copy) from the user's own mockup, in a cleaner
editorial/"Nature style": generous whitespace, thin uniform-weight box
borders, restrained pastel fills (light tint of the category hue + a
darker same-hue border/heading, dark charcoal body text - not pure black,
matching Nature Portfolio's editorial convention), consistent arrowheads,
Times New Roman throughout per request.

Category colours loosely follow the project's validated categorical
palette (dataviz skill reference) but lightened to pastel fills for a
diagram/schematic rather than a data-encoded chart - every node carries
its own text label directly on it, so CVD-safe hue *separation* matters
less here than it would for an unlabelled chart; muted, print-safe tones
were chosen for a professional look rather than run through the chart
validator.

Saves: outputs/figures/intro_pathway_diagram.png (300 dpi)
       outputs/figures/intro_pathway_diagram.pdf (vector, for crisp Word insert)
"""
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
from matplotlib.path import Path

BASE = "/Users/elizabetharmstrong/Library/CloudStorage/OneDrive-ImperialCollegeLondon/Lupus Project"
FIG_DIR = f"{BASE}/outputs/figures"

plt.rcParams["font.family"] = "Times New Roman"

# --- Palette: light pastel fill + darker same-hue border/heading text ---
NEUTRAL   = {"fill": "#f2f2f0", "edge": "#6b6b66", "text": "#2b2b28"}
BLUE      = {"fill": "#e4eefa", "edge": "#2a5faa", "text": "#173e73"}
VIOLET    = {"fill": "#ece8f8", "edge": "#5c4aa8", "text": "#3a2d70"}
GREEN     = {"fill": "#e5f4e7", "edge": "#2e7d3c", "text": "#1e5628"}
RED       = {"fill": "#fbe9e8", "edge": "#b33b38", "text": "#7d2926"}
ARROW_COL = "#4a4a46"
DASH_COL  = "#8a8a85"

FIG_W, FIG_H = 8.6, 8.6   # inches - portrait, fits a Word column/page cleanly

fig, ax = plt.subplots(figsize=(FIG_W, FIG_H))
ax.set_xlim(0, 100)
ax.set_ylim(2, 110)
ax.axis("off")
fig.patch.set_facecolor("white")
ax.set_facecolor("white")


def box(cx, cy, w, h, title, subtitle, style, title_size=11.5, sub_size=9.5):
    """Rounded-rect node, bold title line + optional lighter subtitle line(s)."""
    x, y = cx - w / 2, cy - h / 2
    patch = FancyBboxPatch(
        (x, y), w, h,
        boxstyle="round,pad=0,rounding_size=0.55",
        linewidth=1.3,
        edgecolor=style["edge"],
        facecolor=style["fill"],
        zorder=2,
    )
    ax.add_patch(patch)

    if subtitle:
        ax.text(cx, cy + h * 0.16, title, ha="center", va="center",
                 fontsize=title_size, fontweight="bold", color=style["text"], zorder=3)
        ax.text(cx, cy - h * 0.24, subtitle, ha="center", va="center",
                 fontsize=sub_size, color=style["text"], zorder=3,
                 linespacing=1.35)
    else:
        ax.text(cx, cy, title, ha="center", va="center",
                 fontsize=title_size, fontweight="bold", color=style["text"], zorder=3)
    return x, y, w, h


def solid_arrow(x1, y1, x2, y2, connectionstyle="arc3,rad=0"):
    arr = FancyArrowPatch(
        (x1, y1), (x2, y2),
        connectionstyle=connectionstyle,
        arrowstyle="-|>", mutation_scale=16,
        linewidth=1.4, color=ARROW_COL, zorder=1,
    )
    ax.add_patch(arr)


def dashed_arrow(x1, y1, x2, y2, connectionstyle, label=None, label_pos=(0, 0)):
    arr = FancyArrowPatch(
        (x1, y1), (x2, y2),
        connectionstyle=connectionstyle,
        arrowstyle="-|>", mutation_scale=14,
        linewidth=1.2, color=DASH_COL, linestyle=(0, (5, 3)), zorder=1,
    )
    ax.add_patch(arr)
    if label:
        ax.text(label_pos[0], label_pos[1], label, ha="center", va="center",
                 fontsize=9, color=DASH_COL, style="italic", zorder=3,
                 bbox=dict(boxstyle="round,pad=0.15", facecolor="white",
                           edgecolor="none"))


# --- Nodes (top to bottom) ---
W_MAIN, H_SM, H_MD = 52, 9, 13.5

box(50, 104, 34, H_SM, "SLE", None, NEUTRAL)
box(50, 90, W_MAIN, H_MD, "Lupus Nephritis",
    "Immune complex deposition → glomerular injury", BLUE)
box(50, 74, W_MAIN, H_MD, "Renal Biopsy",
    "Histopathology: NIH activity & chronicity indices", VIOLET)
box(50, 58, 40, H_SM, "Induction Treatment", None, NEUTRAL)

box(24, 40, 40, H_MD, "Response / Remission",
    "Initial treatment response", GREEN)
box(76, 40, 40, H_MD, "Flare",
    "Up to 66% after induction", RED)

box(24, 20, 40, H_MD, "Sustained Remission",
    "~95% 10-year survival\n(vs. ~46% without)", GREEN)
box(76, 20, 40, H_MD, "ESRD",
    "15–30% of patients\nwithin 15 years", RED)

# --- Solid pathway arrows ---
solid_arrow(50, 99.5, 50, 97)     # SLE -> LN
solid_arrow(50, 83.5, 50, 81)     # LN -> Biopsy
solid_arrow(50, 67, 50, 62.5)     # Biopsy -> Induction

# Induction -> Response / Flare (branch)
solid_arrow(50, 53.5, 24, 45)
solid_arrow(50, 53.5, 76, 45)

solid_arrow(24, 33.5, 24, 27)     # Response -> Sustained remission
solid_arrow(76, 33.5, 76, 27)     # Flare -> ESRD

# --- Dashed arrows: relapse (Response -> Flare) and repeat biopsy (Flare -> Biopsy) ---
dashed_arrow(44, 40, 56, 40, "arc3,rad=-0.25", label="relapse", label_pos=(50, 45.5))
dashed_arrow(90, 44, 68, 74, "arc3,rad=0.3", label="repeat\nbiopsy", label_pos=(94, 58))

# --- Footnote ---
ax.text(
    50, 10.5,
    "Each flare independently accelerates nephron loss and progression to ESRD.\n"
    "Dashed arrows: relapse risk after remission, and repeat biopsy motivating the serial-biopsy analysis.",
    ha="center", va="top", fontsize=9, color="#3a3a36", linespacing=1.6, zorder=3,
)

fig.tight_layout()
fig.savefig(f"{FIG_DIR}/intro_pathway_diagram.png", dpi=300, bbox_inches="tight", facecolor="white")
fig.savefig(f"{FIG_DIR}/intro_pathway_diagram.pdf", bbox_inches="tight", facecolor="white")
print(f"Saved: {FIG_DIR}/intro_pathway_diagram.png")
print(f"Saved: {FIG_DIR}/intro_pathway_diagram.pdf")
