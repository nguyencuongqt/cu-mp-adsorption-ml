"""Materials for the expert raters, version 2 (15 October 2026).

Same 15 cases and 60 explanations as version 1 (same item_id, same order, KEY file unchanged); changes:
- explanations are shown as a reader would see them: paragraphs and bullet lists kept (version 1 joined list items into
  run-on text), the HTML <title> is not shown (it is not displayed by a browser either), and the heading
  "Rule-based local interpretation" is removed because it reveals how the text was produced; the words are otherwise unchanged;
- three helper columns restate, from the facts, what has to be checked (the three numbers and the warnings that apply);
- scores are restricted to whole numbers 1-5, the header and the first columns are frozen;
- the rubric keeps the anchors 1, 3 and 5 given to the LLM judges word for word and adds decision rules for 2 and 4;
- a worked-examples file (two cases that are not in the scoring set) replaces the practice items of an earlier model version.

Outputs (Revision/04_LLM_evaluation/): expert_scoring_sheet_final_15cases.xlsx, Worked_examples_for_raters.xlsx,
Briefing_for_expert_raters.docx
"""

from __future__ import annotations

import re
from html.parser import HTMLParser
from pathlib import Path

import pandas as pd
from docx import Document
from docx.enum.table import WD_TABLE_ALIGNMENT
from docx.shared import Cm, Pt, RGBColor
from openpyxl import Workbook
from openpyxl.styles import Alignment, Border, Font, PatternFill, Side
from openpyxl.worksheet.datavalidation import DataValidation

ROOT = Path(__file__).resolve().parents[1]
EVAL = ROOT / "Manuscript" / "Submission_Env pollution" / "Revision" / "04_LLM_evaluation"
REPORTS = ROOT / "outputs" / "reports"
DIMENSIONS = ["accuracy", "mechanistic_depth", "literature_grounding", "condition_specificity", "readability",
              "limits_communicated"]
LABELS = {"accuracy": "Accuracy", "mechanistic_depth": "Mechanistic depth", "literature_grounding": "Literature grounding",
          "condition_specificity": "Condition specificity", "readability": "Readability",
          "limits_communicated": "Limits communicated"}
WORKED_CASES = ["case_03", "case_17"]          # not among the 15 cases of the scoring sheet

# ---------------------------------------------------------------------------------------------------- rubric
# "anchors" = the text given to the LLM judges (unchanged); "rules" = how to decide, added for the human raters
RUBRIC = [
    ("accuracy", "Are the three predicted numbers of the facts stated correctly?",
     "predicted values absent or wrong",
     "the predicted qe is stated correctly but the interval or the field-scale value is missing or wrong",
     "predicted qe, prediction interval and field-scale value are all stated and agree with the facts",
     "Check only the three numbers in the column 'numbers_to_check' (rounding to 2 significant figures is fine; the interval may "
     "be given as a range or as 'multiplied and divided by ...'). 4 = all three present, one slightly misstated. 2 = qe correct, "
     "but interval and field-scale value both missing or wrong; or qe wrong but the other two correct. Other numbers (the input multipliers) and statements about "
     "mechanisms are NOT judged here."),
    ("mechanistic_depth", "Does the text explain WHY Cu(II) sorbs on this particle?",
     "no mechanism",
     "generic statement (e.g. \"adsorption occurs\")",
     "specific named mechanisms (electrostatic attraction, surface complexation, ion exchange, ...) tied to this polymer and "
     "ageing status",
     "2 = the only mechanisms named do not apply to Cu(II) or to this polymer. 4 = plausible named mechanisms linked to the "
     "ageing status, but listed rather than explained, or not linked to the chemistry of this polymer. 5 = explains how this "
     "polymer and its ageing produce the sites that bind Cu(II) (e.g. oxidation forming C=O/COOH groups that complex Cu(II)). "
     "If plausible mechanisms are mixed with one that is wrong for Cu(II) or for this polymer, give at most 3 and name it in the "
     "comment."),
    ("literature_grounding", "Are statements supported by specific sources?",
     "no reference to sources",
     "vague mention of \"literature\" or \"the dataset\"",
     "at least two specific source ids cited with content relevant to the case",
     "Source ids (e.g. Huang_2024) are articles of the study corpus; you do not need to look them up. 2 = a source is named but "
     "is clearly unrelated to the statement. 4 = one source id with relevant content, or two or more ids cited together behind "
     "one general sentence without saying what each found. 5 = two or more ids, each linked to what that source reported. If "
     "you know that a statement attributed to a source is wrong, lower the score and comment."),
    ("condition_specificity", "Is the text about THIS case?",
     "statements that would apply to any case",
     "one input discussed specifically",
     "concentration level, polymer, ageing status and adsorption system all addressed with reasoning specific to this case",
     "Count how many of the four inputs (Ce level, polymer, aging status, adsorption system) are discussed for this case; "
     "stating the direction and size of an input's effect counts. 1 = none (numbers only); 2 = inputs named but not "
     "discussed; 3 = one; 4 = two or three; 5 = all four. If an effect is explained the wrong "
     "way round (e.g. a multiplier below 1 for Ce read as 'higher concentrations lower sorption' when Ce is simply below the "
     "average of the data), take one point off and comment."),
    ("readability", "Could an environmental scientist read and use it?",
     "raw numbers or incomprehensible",
     "understandable but disorganised",
     "clear, well structured and usable by an environmental scientist",
     "A line of numbers with codes (e.g. 'lab_qe=...') = 1. A clear bullet list is as good as prose. 4 = clear, with some "
     "repetition or awkward order. Do not judge English style beyond clarity."),
    ("limits_communicated", "Does the reader learn how far the number can be trusted?",
     "no statement about uncertainty or validity",
     "uncertainty mentioned but the applicable warnings are missing or wrong",
     "the prediction interval and every warning listed in the facts are conveyed correctly, and no warning is raised that does "
     "not apply",
     "Use the column 'warnings_to_convey' (three possible warnings: outside the laboratory range; below 0.1 mg/L; polymer "
     "from one study). 2 = only a vague 'the result is uncertain'. 3 = the interval is given as bare numbers "
     "without saying what it means, OR a required warning is missing. 4 = interval explained and warnings given, but one is "
     "vague or incomplete, or a warning is raised that does not apply. 5 = interval explained and every required warning "
     "given (when the facts say that no warning applies, the explained interval is enough)."),
]

# ---------------------------------------------------------------------------------------------------- worked examples
# Reference scores prepared by the authors with the rules above; order of the criteria as in DIMENSIONS.
REFERENCE = {          # explanations of the rerun of 18 October 2026 (final model, 961 observations; corpus of 41 articles)
    "item_104": ([5, 1, 1, 1, 1, 3], [
        "Accuracy 5: qe, interval and field-scale value all correct.",
        "Mechanistic depth 1 / Literature 1: none.",
        "Condition specificity 1: numbers only, no input discussed.",
        "Readability 1: a line of numbers with codes.",
        "Limits 3: the interval is given as bare numbers, and the extrapolation warning required by the facts is missing."]),
    "item_048": ([5, 1, 1, 5, 5, 5], [
        "Accuracy 5: all three numbers correct.",
        "Mechanistic depth 1 / Literature 1: no mechanism, no source.",
        "Condition specificity 5: all four inputs with the direction and size of their effect, and the concentration placed "
        "below the laboratory range.",
        "Readability 5: short, clear and structured.",
        "Limits 5: interval explained (x/÷29, narrowing to x/÷2.6 with one measurement) and the extrapolation warning given."]),
    "item_072": ([5, 3, 4, 5, 5, 5], [
        "Accuracy 5: all three numbers correct.",
        "Mechanistic depth 3: ageing is said to increase the surface area and the oxygen-containing groups of PLA, but no "
        "binding mechanism for Cu(II) (e.g. complexation with carboxyl groups formed by ester hydrolysis) is named.",
        "Literature 4: two source ids (Hu_2024, Dai_2025) cited together behind one sentence; what each found is not said.",
        "Condition specificity 5: all four inputs, and the concentration is placed outside the laboratory range.",
        "Readability 5: clear.",
        "Limits 5: interval explained, extrapolation warning given, effect of one local measurement mentioned."]),
    "item_103": ([5, 3, 4, 5, 4, 5], [
        "Accuracy 5: all three numbers correct.",
        "Mechanistic depth 3: the same surface properties as A4 (surface area, oxygen-containing groups) are linked to ageing, "
        "but no binding mechanism is named or explained for PLA.",
        "Literature 4: one source id (Huang_2024) with content relevant to the case.",
        "Condition specificity 5: all four inputs discussed.",
        "Readability 4: clear, but the title is repeated and garbled ('Cu(II Sorption').",
        "Limits 5: interval, extrapolation warning and the effect of one measurement are all given correctly."]),
    "item_083": ([5, 1, 1, 1, 1, 3], [
        "Accuracy 5: all three numbers correct.",
        "Mechanistic depth 1 / Literature 1 / Condition specificity 1: numbers only.",
        "Readability 1: a line of numbers with codes.",
        "Limits 3: no warning applies here, but the interval is only given as bare numbers, without saying what it means."]),
    "item_120": ([5, 1, 1, 5, 5, 5], [
        "Accuracy 5: all three numbers correct.",
        "Mechanistic depth 1 / Literature 1: none.",
        "Condition specificity 5: all four inputs with direction and size of their effect.",
        "Readability 5: clear list.",
        "Limits 5: interval explained; correctly states that no validity warning applies."]),
    "item_056": ([5, 3, 4, 5, 4, 5], [
        "Accuracy 5: all three numbers correct.",
        "Mechanistic depth 3 (at most 3): electrostatic interactions and surface complexation are plausible, but hydrogen "
        "bonding and π–π interactions are mechanisms for organic sorbates, not for Cu(II). Comment: 'π–π interactions and "
        "hydrogen bonding do not apply to Cu(II)'.",
        "Literature 4: one source id (Zhou_2022b) with content relevant to the case.",
        "Condition specificity 5: all four inputs with the direction and size of their effect.",
        "Readability 4: clear, but the title is repeated and garbled ('Cu(II Sorption').",
        "Limits 5: interval explained; correctly says that no warning applies."]),
    "item_079": ([5, 4, 4, 5, 5, 5], [
        "Accuracy 5: all three numbers correct.",
        "Mechanistic depth 4: electrostatic attraction, surface complexation and oxygen-containing groups enhanced by ageing "
        "are plausible and linked to ageing, but nothing is specific to PS (e.g. oxidation of the aromatic ring).",
        "Literature 4: one source id (Huang_2024) with content relevant to the case.",
        "Condition specificity 5: all four inputs discussed.",
        "Readability 5: clear.",
        "Limits 5: interval explained; correctly says that no warning applies."]),
}
EXAMPLE_ORDER = {"case_03": ["item_048", "item_104", "item_103", "item_072"],     # mixed order, labels A1-A4
                 "case_17": ["item_056", "item_083", "item_120", "item_079"]}     # labels B1-B4


class _Text(HTMLParser):
    """HTML -> plain text with paragraphs and bullets kept; drops <head>/<title> and the rule-based heading."""

    def __init__(self) -> None:
        super().__init__()
        self.out: list[str] = []
        self.skip = 0
        self.heading = False

    def handle_starttag(self, tag, attrs):
        if tag in ("head", "title"):
            self.skip += 1
        elif tag in ("p", "div", "section", "ul", "h1", "h2", "h3", "br"):
            self.out.append("\n")
            self.heading = tag in ("h1", "h2", "h3")
        elif tag == "li":
            self.out.append("\n• ")

    def handle_endtag(self, tag):
        if tag in ("head", "title"):
            self.skip -= 1
        elif tag in ("p", "h1", "h2", "h3", "ul"):
            self.out.append("\n")
            self.heading = False

    def handle_data(self, data):
        if self.skip:
            return
        if self.heading and data.strip() == "Rule-based local interpretation":
            return
        self.out.append(re.sub(r"\s+", " ", data))


def plain(html: str) -> str:
    if "<" not in html:
        return html.strip()
    p = _Text()
    p.feed(html)
    text = "".join(p.out)
    lines = [ln.strip() for ln in text.split("\n")]
    text = "\n".join(lines)
    return re.sub(r"\n{3,}", "\n\n", text).strip()


def facts_parts(facts: str) -> tuple[str, str, str]:
    lines = facts.split("\n")
    inputs = lines[0].replace("Inputs: ", "")
    qe = re.search(r"qe: ([\d.eE-]+ mg/g)", facts).group(1)
    pi = re.search(r"training data: ([\d.eE-]+ to [\d.eE-]+ mg/g)", facts).group(1)
    fv = re.search(r"that ratio\): ([\d.eE-]+ mg/g)", facts).group(1)
    numbers = f"Laboratory qe: {qe}\n90% interval: {pi}\nField-scale value: {fv}"
    warns = [ln.replace("WARNING: ", "• ") for ln in lines if ln.startswith("WARNING")]
    warnings = "\n".join(warns) if warns else "None: the text must not raise a warning (the interval must still be conveyed)."
    return inputs, numbers, warnings


# ---------------------------------------------------------------------------------------------------- Excel helpers
HEAD = PatternFill("solid", fgColor="1B4F72")
HELP = PatternFill("solid", fgColor="EAF2F8")
SCORE = PatternFill("solid", fgColor="FEF9E7")
THIN = Side(style="thin", color="B0B7BF")
BOX = Border(left=THIN, right=THIN, top=THIN, bottom=THIN)


def header(ws, names, widths):
    for j, (n, w) in enumerate(zip(names, widths), start=1):
        c = ws.cell(row=1, column=j, value=n)
        c.font, c.fill = Font(bold=True, color="FFFFFF"), HEAD
        c.alignment = Alignment(wrap_text=True, vertical="center", horizontal="center")
        c.border = BOX
        ws.column_dimensions[c.column_letter].width = w
    ws.row_dimensions[1].height = 32


def rubric_sheet(wb):
    ws = wb.create_sheet("rubric")
    header(ws, ["Criterion", "Question", "1", "3", "5", "How to decide (and when to give 2 or 4)"], [20, 26, 22, 30, 36, 70])
    for i, (key, q, a1, a3, a5, rule) in enumerate(RUBRIC, start=2):
        for j, v in enumerate([LABELS[key], q, a1, a3, a5, rule], start=1):
            c = ws.cell(row=i, column=j, value=v)
            c.alignment, c.border = Alignment(wrap_text=True, vertical="top"), BOX
            if j == 1:
                c.font = Font(bold=True)
    ws.freeze_panes = "B2"
    return ws


def score_cells(ws, first_row, last_row, first_col):
    dv = DataValidation(type="whole", operator="between", formula1="1", formula2="5", allow_blank=True,
                        showErrorMessage=True, errorTitle="Score", error="Please enter a whole number from 1 to 5.")
    ws.add_data_validation(dv)
    for r in range(first_row, last_row + 1):
        for k in range(len(DIMENSIONS)):
            c = ws.cell(row=r, column=first_col + k)
            c.fill, c.border = SCORE, BOX
            c.alignment = Alignment(horizontal="center", vertical="top")
            dv.add(c)


# ---------------------------------------------------------------------------------------------------- outputs
def scoring_sheet(items: pd.DataFrame) -> None:
    wb = Workbook()
    ws = wb.active
    ws.title = "instructions"
    lines = [
        "HOW TO SCORE (about 2–3 minutes per row; 60 rows; you may stop and continue later)",
        "1. Read the explanation (column E).",
        "2. Accuracy: compare it with column C (three numbers).",
        "3. Limits communicated: compare it with column D (warnings that must be given, or 'None').",
        "4. Score the other four criteria with the sheet 'rubric' (decision rules in its last column).",
        "5. Enter whole numbers 1–5 in the yellow cells (G–L). Add a comment only if something is wrong or unclear.",
        "",
        "The facts (column N; summarised in the blue columns B–D) are the model output and the REFERENCE: treat them as correct.",
        "Each explanation is written from these facts and is expected to differ from them; you judge how well it conveys and interprets them.",
        "Accuracy and Limits are scored against the facts; the other four criteria with your own expertise and the rubric.",
        "Rows are in random order; four rows belong to each case. Score every row on its own; do not try to guess how a text was made.",
        "Source ids such as Huang_2024 are articles of the study corpus (first author and year); you do not need to look them up.",
        "Warnings (column D) are of three kinds: outside the laboratory range; below 0.1 mg/L; polymer from one study.",
        "Before you start: read the briefing and the worked examples (file Worked_examples_for_raters.xlsx, about 10 minutes).",
    ]
    for i, t in enumerate(lines, start=1):
        c = ws.cell(row=i, column=1, value=t)
        c.font = Font(bold=(i == 1), size=12 if i == 1 else 11)
    ws.column_dimensions["A"].width = 125

    ws = wb.create_sheet("score_here")
    names = ["item_id", "inputs", "numbers_to_check", "warnings_to_convey", "explanation", "",
             *[LABELS[d] for d in DIMENSIONS], "comment", "facts (full model output)"]
    header(ws, names, [9, 24, 26, 30, 90, 1, 11, 11, 11, 11, 11, 11, 30, 60])
    for i, r in enumerate(items.itertuples(), start=2):
        inputs, numbers, warnings = facts_parts(r.facts)
        values = [r.item_id, inputs, numbers, warnings, plain(r.explanation), None, *[None] * 6, None, r.facts]
        for j, v in enumerate(values, start=1):
            c = ws.cell(row=i, column=j, value=v)
            c.alignment, c.border = Alignment(wrap_text=True, vertical="top"), BOX
            if j in (2, 3, 4):
                c.fill = HELP
    score_cells(ws, 2, len(items) + 1, 7)
    ws.freeze_panes = "B2"
    rubric_sheet(wb)
    wb.active = 1
    wb.save(EVAL / "expert_scoring_sheet_final_15cases.xlsx")


def worked_examples(items: pd.DataFrame) -> None:
    wb = Workbook()
    ws = wb.active
    ws.title = "read_me"
    lines = [
        "WORKED EXAMPLES (not part of the evaluation; about 10 minutes)",
        "Two cases from the same evaluation that are NOT in your scoring sheet, four explanations each (A1–A4 and B1–B4).",
        "",
        "FACTS VERSUS EXPLANATION",
        "• The facts (column N; summarised in the blue columns B–D) are the model output for the case. There is one set of facts per",
        "  case, and it is the REFERENCE: treat it as correct.",
        "• Each explanation (column E) is one of four texts written from the same facts. It is EXPECTED to differ from the facts: it",
        "  may omit, restate, misstate or add to them. The differences are what you score.",
        "• Accuracy and Limits communicated: compare the explanation with the facts (columns C and D).",
        "• Mechanistic depth, Literature grounding, Condition specificity, Readability: the facts contain no mechanisms or sources;",
        "  judge these with your own expertise and the rubric.",
        "",
        "Case A: concentration below the laboratory range, so the extrapolation warning is required.",
        "Case B: concentration inside the laboratory range, so no warning applies.",
        "For each explanation the sheet 'examples' gives reference scores and the reason for each score.",
        "Suggested use: cover columns G–M, score A1–A4 yourself, then compare; read B1–B4 with the reasons.",
        "The reference scores show how the rules in the sheet 'rubric' are applied; differences of one point are normal.",
    ]
    for i, t in enumerate(lines, start=1):
        ws.cell(row=i, column=1, value=t).font = Font(bold=(i == 1), size=12 if i == 1 else 11)
    ws.column_dimensions["A"].width = 120

    ws = wb.create_sheet("examples")
    names = ["example", "inputs", "numbers_to_check", "warnings_to_convey", "explanation", "",
             *[LABELS[d] for d in DIMENSIONS], "why these scores", "facts (full model output)"]
    header(ws, names, [9, 24, 26, 30, 80, 1, 11, 11, 11, 11, 11, 11, 70, 60])
    row = 2
    for prefix, case in zip("AB", WORKED_CASES):
        for k, item in enumerate(EXAMPLE_ORDER[case], start=1):
            r = items[items["item_id"] == item].iloc[0]
            inputs, numbers, warnings = facts_parts(r["facts"])
            scores, why = REFERENCE[item]
            values = [f"{prefix}{k}", inputs, numbers, warnings, plain(r["explanation"]), None, *scores, "\n".join(why),
                      r["facts"]]
            for j, v in enumerate(values, start=1):
                c = ws.cell(row=row, column=j, value=v)
                c.alignment, c.border = Alignment(wrap_text=True, vertical="top"), BOX
                if j in (2, 3, 4):
                    c.fill = HELP
                if 7 <= j <= 12:
                    c.fill = SCORE
                    c.alignment = Alignment(horizontal="center", vertical="top")
                    c.font = Font(bold=True)
            row += 1
        row += 1                                               # blank row between the two cases
    ws.freeze_panes = "B2"
    rubric_sheet(wb)
    wb.active = 1
    wb.save(EVAL / "Worked_examples_for_raters.xlsx")


def briefing(items: pd.DataFrame) -> None:
    doc = Document()
    st = doc.styles["Normal"]
    st.font.name, st.font.size = "Calibri", Pt(11)
    for s in doc.sections:
        s.left_margin = s.right_margin = Cm(2.2)
        s.top_margin = s.bottom_margin = Cm(2.0)

    def h(text, level=1):
        p = doc.add_heading(text, level=level)
        for r in p.runs:
            r.font.color.rgb = RGBColor(0x1B, 0x4F, 0x72)

    def para(text, bold_lead=None, italic=False):
        p = doc.add_paragraph()
        if bold_lead:
            p.add_run(bold_lead).bold = True
        r = p.add_run(text)
        r.italic = italic
        p.paragraph_format.space_after = Pt(4)
        return p

    def bullets(rows, style="List Bullet"):
        for t in rows:
            p = doc.add_paragraph(style=style)
            if isinstance(t, tuple):
                p.add_run(t[0]).bold = True
                p.add_run(t[1])
            else:
                p.add_run(t)
            p.paragraph_format.space_after = Pt(2)

    def table(rows, widths, header=True):
        t = doc.add_table(rows=0, cols=len(rows[0]))
        t.style = "Light Grid Accent 1"
        t.alignment = WD_TABLE_ALIGNMENT.CENTER
        for i, row in enumerate(rows):
            cells = t.add_row().cells
            for j, v in enumerate(row):
                cells[j].text = v
                for p in cells[j].paragraphs:
                    for r in p.runs:
                        r.font.size = Pt(10)
                        r.bold = header and i == 0
        for j, w in enumerate(widths):
            for c in t.columns[j].cells:
                c.width = Cm(w)
        doc.add_paragraph().paragraph_format.space_after = Pt(2)

    doc.add_heading("Briefing for expert raters", 0)
    para("Thank you for helping. You do not need to know the manuscript: everything you need is on these pages. "
         "Reading them takes about 10 minutes.")

    h("1. The task in one minute")
    bullets([
        "A statistical model predicts how much Cu(II) is sorbed on a microplastic (qe, mg/g) for a given polymer, aging status "
        "(virgin or aged), adsorption system (Cu alone, or Cu with other solutes = 'mixed') and dissolved Cu concentration at "
        "equilibrium (Ce, mg/L).",
        "For each prediction, a short text explains the result to a user. You score 60 such texts (15 cases × 4 texts) on six "
        "criteria, each from 1 (poor) to 5 (very good).",
        "You judge the texts, not the model.",
        "Time: about 10 minutes for the worked examples, then 2–3 minutes per text (2–3 hours in total; two sessions are fine).",
    ])

    h("2. The key idea: facts and explanation")
    table([["", "Facts", "Explanation"],
           ["What it is", "the model output for one case: predicted numbers, their uncertainty and warnings",
            "a text written from the facts for a user"],
           ["How many", "one set per case", "four different texts per case"],
           ["Your role", "the REFERENCE: take it as correct", "what you SCORE"]], [3.0, 6.5, 6.5])
    para("The four texts of a case are written from the same facts but differ: some only list numbers, some explain them; "
         "some leave things out, restate them, get them wrong or add knowledge from the literature. These differences are "
         "what you score.")
    bullets([
        ("Accuracy and Limits communicated: ", "compare the text with the facts (the facts are the answer key)."),
        ("Mechanistic depth, Literature grounding, Condition specificity, Readability: ", "the facts contain no mechanisms "
         "or sources; judge these with your own expertise and the rubric."),
    ])

    h("3. How to read the facts (one example)")
    para("Each facts block has the same lines. This one is from the worked examples (case A):", italic=True)
    ex = items[items["case_id"] == WORKED_CASES[0]]["facts"].iloc[0].split("\n")
    notes = ["the case: the four inputs",
             "Number 1 to check: the predicted laboratory qe",
             "Number 2 to check: the 90% prediction interval. For a material never measured, the true value is expected in this "
             "range 9 times out of 10; it spans more than an order of magnitude either way",
             "how much one measurement on the material would narrow the interval",
             "an empirical ratio between Cu on microplastics collected in rivers and the laboratory prediction",
             "Number 3 to check: the field-scale value = laboratory qe × that ratio",
             "how each input moves qe compared with an average observation of the laboratory data: a multiplier below 1 means "
             "lower qe than average, above 1 higher. Example: 0.02 for Ce means that this very low Ce gives a qe about 50 times "
             "lower than at a typical laboratory Ce",
             "what the model is and how well this polymer is represented",
             "WARNING lines: the explanation must pass these on (or the line 'No validity warning applies')"]
    table([["Line of the facts", "What it means"]] + [[a, b] for a, b in zip(ex, notes)], [8.0, 8.0])
    para("Only three kinds of warning occur:", bold_lead=None)
    bullets([
        ("Outside the laboratory range – ", "Ce is below (or above) all laboratory data; the value is an extrapolation."),
        ("Below 0.1 mg/L – ", "Ce is inside the range but in its sparsely covered low part (5% of the data)."),
        ("Polymer from one study – ", "the polymer was measured by a single laboratory, so its effect cannot be separated "
                                      "from that laboratory."),
    ])

    h("4. Your files and the columns")
    bullets([
        ("Worked_examples_for_raters.xlsx – ", "8 texts already scored, with the reason for every score. Read these first."),
        ("expert_scoring_sheet_final_15cases.xlsx – ", "sheet 'score_here' (the 60 texts) and sheet 'rubric' (the scale)."),
    ])
    table([["Column", "Content"],
           ["B  inputs (blue)", "polymer, aging status, adsorption system, Ce"],
           ["C  numbers_to_check (blue)", "the three numbers the text must state (from the facts)"],
           ["D  warnings_to_convey (blue)", "the warnings the text must give, or 'None'"],
           ["E  explanation", "the text you score"],
           ["G–L  yellow cells", "your six scores, whole numbers 1–5"],
           ["M  comment", "only if something is wrong or unclear"],
           ["N  facts", "the full model output (the blue columns summarise it)"]], [5.5, 10.5])

    h("5. Scoring one text, step by step")
    bullets([
        "Read the text (column E).",
        ("Accuracy – ", "are the three numbers of column C stated correctly? Nothing else counts here."),
        ("Limits communicated – ", "is the 90% interval explained, and is every warning of column D passed on (and no "
                                   "warning given that does not apply)?"),
        ("Mechanistic depth – ", "does it explain why Cu(II) binds to this polymer in this aging state?"),
        ("Literature grounding – ", "are statements linked to specific sources, and is it said what each source found?"),
        ("Condition specificity – ", "how many of the four inputs are discussed for this case?"),
        ("Readability – ", "could an environmental scientist read and use it?"),
        "Enter the six scores in the yellow cells; add a comment only when something is wrong.",
    ], style="List Number")
    para("The sheet 'rubric' describes scores 1, 3 and 5 for each criterion, and its last column says when to give 2 or 4. "
         "Use the whole scale: give 5 when a criterion is fully met and 1 when the element is absent.")

    h("6. Situations that come up often")
    bullets([
        ("A text with only numbers (e.g. 'lab_qe=...'): ", "score it like any other – Accuracy 5 if the numbers are right, "
         "Readability 1, Condition specificity 1, Limits at most 3."),
        ("A clear bullet list: ", "as readable as prose; do not penalise the format."),
        ("A mechanism that does not apply to Cu(II) ", "(e.g. π–π interactions, which concern aromatic organic molecules): "
         "Mechanistic depth at most 3; name it in the comment."),
        ("An input effect explained the wrong way round: ", "one point off Condition specificity; comment."),
        ("Source names such as Huang_2024: ", "articles of the study corpus (first author, year). You do not need to look them "
         "up; if you know a statement attributed to a source is wrong, lower Literature grounding and comment."),
        ("Unsure between two scores: ", "give the higher score whose rule in the rubric is fully met."),
        ("A criterion cannot be judged: ", "give your best score and explain in the comment. Leave no yellow cell empty."),
    ])

    h("7. Please do not")
    bullets(["discuss the texts with other raters before all files are returned;",
             "try to work out how a text was produced (the four texts of a case are mixed among the 60 rows on purpose);",
             "edit the facts or the texts."])

    h("8. Returning the file")
    para("Save the scoring file with your initials added to its name (e.g. expert_scoring_sheet_final_15cases_NXC.xlsx) "
         "and e-mail it to the corresponding author. If something is unclear, please ask before you start. Scores are "
         "reported in aggregate; with your agreement you will be thanked by name in the Acknowledgements.")
    doc.save(EVAL / "Briefing_for_expert_raters.docx")


def main() -> None:
    items = pd.read_csv(REPORTS / "llm_eval_v2_explanations_final.csv")
    key = pd.read_csv(EVAL / "KEY_do_not_share_with_raters_final_15cases.csv")
    chosen = items.set_index("item_id").loc[key["item_id"]].reset_index()      # same rows and order as version 1
    assert len(chosen) == 60 and chosen["case_id"].nunique() == 15
    assert not set(WORKED_CASES) & set(chosen["case_id"]), "worked examples must not be scored items"
    for case, ids in EXAMPLE_ORDER.items():
        assert set(items[items.case_id == case]["item_id"]) == set(ids), case
    scoring_sheet(chosen)
    worked_examples(items)
    briefing(items)
    leak = sum("Rule-based local interpretation" in plain(t) for t in chosen["explanation"])
    print(f"scoring sheet: {len(chosen)} rows, {chosen.case_id.nunique()} cases; label left in {leak} rows")
    print("worked examples:", ", ".join(f"{c}" for c in WORKED_CASES))


if __name__ == "__main__":
    main()
