"""
Table 1 (Discrimination / AUROC) and Table 2 (Calibration / accuracy),
publication style, in one document: Times New Roman, three-line table format
(top rule, header rule, bottom rule only - no internal vertical lines, no
cell shading), landscape orientation (needed to fit 5 cohort columns at
readable single-line width). Both share the same row layout: Model rows
(Logistic Regression, Random Forest, XGBoost, LightGBM, then TabPFN v3 in
italics) x columns grouped under 'Flare Cohorts' (1-Year Flare, 5-Year Flare,
Serial Biopsy) and 'Kidney Failure (ESRD) Cohorts' (ESRD 5-Year, ESRD 10-Year).

Table 1: AUROC (bold if highest in column among the four main classifiers,
ties bolded together) on line 1, 95% CI on line 2 (smaller, non-bold).
No significance-testing footnote (explicitly skipped per instruction).

Table 2: 'Brier X.XXX' (bold if lowest in column) on line 1, 'Cal X.XXX'
(bold if closest to 1.0 in column - independent criterion from Brier) on
line 2 (smaller text).

TabPFN v3 row: italic throughout, excluded from all bold comparisons in
both tables (not run through DeLong's test or Harrell bootstrap). Serial
Biopsy's TabPFN v3 AUROC uses the fold-mean 'CV AUROC' convention (0.626),
matching every other cell in these tables - NOT the 0.588 pooled-OOF value
used for the ROC/calibration figures (see src/16_roc_calibration_plots.py),
which is a different aggregation of the same underlying predictions.

Footnotes: TabPFN v3 exclusion note under both tables; an additional note
under Table 2 that all five models' serial-biopsy calibration is degenerate
(n=70) and should not be read against the other cohorts' values at face
value (e.g. XGBoost's slope of 0.053 reflects near-constant predictions,
not a typo).

Saves: outputs/Tables_1_2_Publication.docx
"""
from docx import Document
from docx.shared import Pt, Cm
from docx.oxml.ns import qn
from docx.oxml import OxmlElement
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.enum.section import WD_ORIENT
from docx.enum.table import WD_TABLE_ALIGNMENT, WD_ALIGN_VERTICAL

OUTPUT = "/Users/elizabetharmstrong/Library/CloudStorage/OneDrive-ImperialCollegeLondon/Lupus Project/outputs/Tables_1_2_Publication.docx"
FONT = "Times New Roman"

COHORTS = ["1-Year Flare", "5-Year Flare", "Serial Biopsy", "ESRD 5-Year", "ESRD 10-Year"]
FLARE_GROUP_SIZE = 3   # 1-Year, 5-Year, Serial Biopsy
ESRD_GROUP_SIZE  = 2   # ESRD 5-Year, ESRD 10-Year
MODELS = ["Logistic Regression", "Random Forest", "XGBoost", "LightGBM"]
N = len(COHORTS)

# --- Table 1 data: (auroc, ci_lower, ci_upper) ---
AUROC_DATA = {
    "Logistic Regression": [(0.708, 0.553, 0.858), (0.673, 0.536, 0.788), (0.676, 0.434, 0.873), (0.797, 0.669, 0.904), (0.811, 0.656, 0.903)],
    "Random Forest":       [(0.690, 0.549, 0.834), (0.679, 0.505, 0.809), (0.588, 0.318, 0.824), (0.787, 0.623, 0.898), (0.817, 0.710, 0.931)],
    "XGBoost":              [(0.674, 0.544, 0.837), (0.673, 0.479, 0.811), (0.648, 0.453, 0.818), (0.792, 0.631, 0.901), (0.821, 0.696, 0.933)],
    "LightGBM":             [(0.659, 0.532, 0.818), (0.678, 0.518, 0.792), (0.631, 0.443, 0.855), (0.797, 0.653, 0.923), (0.809, 0.701, 0.926)],
}
AUROC_TABPFN = [(0.684, 0.567, 0.834), (0.671, 0.508, 0.815), (0.626, 0.352, 0.841), (0.796, 0.621, 0.898), (0.817, 0.710, 0.926)]

# --- Table 2 data: (brier, cal_slope) ---
CAL_DATA = {
    "Logistic Regression": [(0.220, 0.719), (0.232, 0.669), (0.235, 0.570), (0.178, 0.873), (0.167, 0.874)],
    "Random Forest":       [(0.211, 0.849), (0.227, 0.827), (0.251, 0.320), (0.164, 1.013), (0.139, 1.152)],
    "XGBoost":              [(0.224, 0.835), (0.239, 0.705), (0.249, 0.053), (0.178, 1.288), (0.155, 0.963)],
    "LightGBM":             [(0.224, 0.686), (0.229, 0.878), (0.240, 0.346), (0.171, 1.053), (0.140, 0.714)],
}
CAL_TABPFN = [(0.166, 0.747), (0.230, 0.734), (0.251, 0.190), (0.100, 0.891), (0.122, 0.904)]

# --- Bold winners ---
best_auroc = {}  # col -> set of models tied for highest AUROC
for col in range(N):
    vals = {m: AUROC_DATA[m][col][0] for m in MODELS}
    top = max(vals.values())
    best_auroc[col] = {m for m in MODELS if vals[m] == top}

best_brier, best_cal = {}, {}
for col in range(N):
    briers = {m: CAL_DATA[m][col][0] for m in MODELS}
    cals   = {m: CAL_DATA[m][col][1] for m in MODELS}
    best_brier[col] = min(briers, key=briers.get)
    best_cal[col]   = min(cals, key=lambda m: abs(cals[m] - 1.0))

print("Table 1 - bold AUROC (highest, ties included) per column:",
      {COHORTS[c]: sorted(best_auroc[c]) for c in range(N)})
print("Table 2 - bold Brier (lowest) per column:", {COHORTS[c]: best_brier[c] for c in range(N)})
print("Table 2 - bold Cal slope (closest to 1.0) per column:", {COHORTS[c]: best_cal[c] for c in range(N)})

# --- Border / font helpers (matching Table 2's established style) ---

def set_cell_borders(cell, top=None, bottom=None, left=None, right=None, insideH=None, insideV=None):
    tcPr = cell._tc.get_or_add_tcPr()
    tcBorders = tcPr.find(qn('w:tcBorders'))
    if tcBorders is None:
        tcBorders = OxmlElement('w:tcBorders')
        tcPr.append(tcBorders)
    specs = {"top": top, "bottom": bottom, "left": left, "right": right,
             "insideH": insideH, "insideV": insideV}
    for edge, spec in specs.items():
        el = tcBorders.find(qn(f'w:{edge}'))
        if el is None:
            el = OxmlElement(f'w:{edge}')
            tcBorders.append(el)
        if spec is None:
            el.set(qn('w:val'), 'nil')
        else:
            el.set(qn('w:val'), 'single')
            el.set(qn('w:sz'), str(spec))
            el.set(qn('w:color'), '000000')
            el.set(qn('w:space'), '0')


def clear_all_borders(cell):
    set_cell_borders(cell)


def set_run_font(run, size=10.5, bold=False, italic=False):
    run.font.name = FONT
    run.font.size = Pt(size)
    run.font.bold = bold
    run.font.italic = italic
    rPr = run._element.get_or_add_rPr()
    rFonts = rPr.find(qn('w:rFonts'))
    if rFonts is None:
        rFonts = OxmlElement('w:rFonts')
        rPr.append(rFonts)
    for attr in ('w:ascii', 'w:hAnsi', 'w:cs', 'w:eastAsia'):
        rFonts.set(qn(attr), FONT)


CELL_SPACE_AFTER = Pt(3)  # a touch of breathing room within each (now single-line) row


def set_cell_lines(cell, lines, align=WD_ALIGN_PARAGRAPH.CENTER):
    """lines: list of (text, size, bold, italic) tuples, one per paragraph line."""
    cell.text = ""
    for i, (text, size, bold, italic) in enumerate(lines):
        p = cell.paragraphs[0] if i == 0 else cell.add_paragraph()
        p.alignment = align
        p.paragraph_format.space_after = CELL_SPACE_AFTER
        p.paragraph_format.space_before = Pt(0)
        run = p.add_run(text)
        set_run_font(run, size=size, bold=bold, italic=italic)
    cell.vertical_alignment = WD_ALIGN_VERTICAL.CENTER


def set_cell_run_line(cell, runs, align=WD_ALIGN_PARAGRAPH.CENTER):
    """runs: list of (text, size, bold, italic) tuples, all on ONE line (one
    paragraph, multiple runs) - lets part of a single line be bold/smaller
    than the rest, e.g. '0.708' bold + ' (0.55–13)' smaller/non-bold."""
    cell.text = ""
    p = cell.paragraphs[0]
    p.alignment = align
    p.paragraph_format.space_after = CELL_SPACE_AFTER
    p.paragraph_format.space_before = Pt(0)
    for text, size, bold, italic in runs:
        run = p.add_run(text)
        set_run_font(run, size=size, bold=bold, italic=italic)
    cell.vertical_alignment = WD_ALIGN_VERTICAL.CENTER


def set_table_cell_margins(table, top_pt=3, bottom_pt=3, left_pt=5, right_pt=5):
    """Table-level cell margins (Word: Table Layout > Cell Margins), in
    twips (1pt = 20 twips) - adds real padding inside every cell, not just
    space between lines."""
    tbl = table._tbl
    tblPr = tbl.tblPr
    cellMar = OxmlElement('w:tblCellMar')
    for edge, pt_val in (('top', top_pt), ('bottom', bottom_pt), ('left', left_pt), ('right', right_pt)):
        node = OxmlElement(f'w:{edge}')
        node.set(qn('w:w'), str(int(pt_val * 20)))
        node.set(qn('w:type'), 'dxa')
        cellMar.append(node)
    tblPr.append(cellMar)


def build_three_line_table(doc, title_text, space_before=Pt(0)):
    """Creates the (2 header rows + 4 model rows + 1 TabPFN row) x
    (1 Model col + N cohort cols) skeleton with three-line borders, returns
    the table object."""
    title = doc.add_paragraph()
    run = title.add_run(title_text)
    set_run_font(run, size=11, bold=True)
    title.paragraph_format.space_before = space_before
    title.paragraph_format.space_after = Pt(8)

    n_rows = 2 + len(MODELS) + 1
    n_cols = 1 + N
    table = doc.add_table(rows=n_rows, cols=n_cols)
    table.style = "Table Grid"
    table.alignment = WD_TABLE_ALIGNMENT.CENTER
    table.autofit = False
    set_table_cell_margins(table, top_pt=3, bottom_pt=3, left_pt=5, right_pt=5)

    col_widths = [4.4] + [3.6] * N
    for row in table.rows:
        for cell, w in zip(row.cells, col_widths):
            cell.width = Cm(w)

    for row in table.rows:
        for cell in row.cells:
            clear_all_borders(cell)

    model_hdr = table.cell(0, 0).merge(table.cell(1, 0))
    set_cell_lines(model_hdr, [("Model", 11, True, False)], align=WD_ALIGN_PARAGRAPH.LEFT)

    flare_hdr = table.cell(0, 1)
    for c in range(2, 1 + FLARE_GROUP_SIZE):
        flare_hdr = flare_hdr.merge(table.cell(0, c))
    set_cell_lines(flare_hdr, [("Flare Cohorts", 11, True, False)])

    esrd_start = 1 + FLARE_GROUP_SIZE
    esrd_hdr = table.cell(0, esrd_start)
    for c in range(esrd_start + 1, esrd_start + ESRD_GROUP_SIZE):
        esrd_hdr = esrd_hdr.merge(table.cell(0, c))
    set_cell_lines(esrd_hdr, [("Kidney Failure (ESRD) Cohorts", 11, True, False)])

    for c, name in enumerate(COHORTS, start=1):
        set_cell_lines(table.cell(1, c), [(name, 11, True, False)])

    return table, n_rows


def apply_three_line_borders(table, tabpfn_row, n_cols):
    THICK, THIN = 18, 8
    for c in range(n_cols):
        set_cell_borders(table.cell(0, c), top=THICK)
        set_cell_borders(table.cell(1, c), bottom=THIN)
    for c in range(n_cols):
        set_cell_borders(table.cell(tabpfn_row, c), bottom=THICK)


def add_footnote(doc, text, space_after=Pt(18)):
    p = doc.add_paragraph()
    run = p.add_run(text)
    set_run_font(run, size=9, italic=True)
    p.paragraph_format.space_before = Pt(6)
    p.paragraph_format.space_after = space_after
    return p


TABPFN_FOOTNOTE = ("TabPFN v3 shown in italics for reference; not included in formal pairwise "
                    "significance testing (DeLong's test) or bootstrap correction.")
SERIAL_CI_FOOTNOTE = ("Serial Biopsy confidence intervals are wide (n=70, 34 events) — treat this "
                       "column as exploratory; see Section 12 of the Methods document for the formal "
                       "sample-size justification.")
SERIAL_CAL_FOOTNOTE = ("Serial Biopsy calibration is degenerate for all five models at this sample "
                        "size (e.g. XGBoost slope=0.053 reflects near-constant predictions, not a "
                        "data error) — these values should not be compared against the other cohorts "
                        "at face value.")

# --- Build document ---

doc = Document()
section = doc.sections[0]
section.orientation = WD_ORIENT.LANDSCAPE
section.page_width, section.page_height = section.page_height, section.page_width
section.top_margin = section.bottom_margin = Cm(2.5)
section.left_margin = section.right_margin = Cm(2.5)

n_cols = 1 + N

# Table 1 — AUROC
table1, n_rows = build_three_line_table(doc, "Table 1. Discrimination (AUROC, 95% CI) by model and cohort")
for r, model in enumerate(MODELS, start=2):
    set_cell_lines(table1.cell(r, 0), [(model, 11, False, False)], align=WD_ALIGN_PARAGRAPH.LEFT)
    for c in range(N):
        auroc, lo, hi = AUROC_DATA[model][c]
        bold = model in best_auroc[c]
        set_cell_run_line(table1.cell(r, c + 1), [
            (f"{auroc:.3f}", 11, bold, False),
            (f" ({lo:.2f}–{hi:.2f})", 9, False, False),
        ])
tabpfn_row1 = 2 + len(MODELS)
set_cell_lines(table1.cell(tabpfn_row1, 0), [("TabPFN v3 (Prior Labs)", 10.5, False, True)], align=WD_ALIGN_PARAGRAPH.LEFT)
for c in range(N):
    auroc, lo, hi = AUROC_TABPFN[c]
    set_cell_run_line(table1.cell(tabpfn_row1, c + 1), [
        (f"{auroc:.3f}", 11, False, True),
        (f" ({lo:.2f}–{hi:.2f})", 9, False, True),
    ])
apply_three_line_borders(table1, tabpfn_row1, n_cols)
add_footnote(doc, TABPFN_FOOTNOTE)
add_footnote(doc, SERIAL_CI_FOOTNOTE, space_after=Pt(36))  # extra gap before Table 2 starts

# Table 2 — Calibration
table2, _ = build_three_line_table(doc, "Table 2. Calibration performance (Brier score, calibration slope) by model and cohort",
                                    space_before=Pt(12))
for r, model in enumerate(MODELS, start=2):
    set_cell_lines(table2.cell(r, 0), [(model, 11, False, False)], align=WD_ALIGN_PARAGRAPH.LEFT)
    for c in range(N):
        brier, cal = CAL_DATA[model][c]
        brier_bold = (best_brier[c] == model)
        cal_bold = (best_cal[c] == model)
        set_cell_run_line(table2.cell(r, c + 1), [
            (f"Brier {brier:.3f}", 11, brier_bold, False),
            (" (Cal ", 9, False, False),
            (f"{cal:.3f}", 9, cal_bold, False),
            (")", 9, False, False),
        ])
tabpfn_row2 = 2 + len(MODELS)
set_cell_lines(table2.cell(tabpfn_row2, 0), [("TabPFN v3 (Prior Labs)", 10.5, False, True)], align=WD_ALIGN_PARAGRAPH.LEFT)
for c in range(N):
    brier, cal = CAL_TABPFN[c]
    set_cell_run_line(table2.cell(tabpfn_row2, c + 1), [
        (f"Brier {brier:.3f}", 11, False, True),
        (f" (Cal {cal:.3f})", 9, False, True),
    ])
apply_three_line_borders(table2, tabpfn_row2, n_cols)
add_footnote(doc, "Bold Brier score = lowest (best) in that column among the four main classifiers. "
                  "Bold calibration slope = closest to the ideal value of 1.0 in that column among the "
                  "four main classifiers (independent criterion from Brier score). " + TABPFN_FOOTNOTE)
add_footnote(doc, SERIAL_CAL_FOOTNOTE)

doc.save(OUTPUT)
print(f"\nSaved: {OUTPUT}")
