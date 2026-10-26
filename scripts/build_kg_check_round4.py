"""Round 4 of the hand check of the knowledge-graph relations (17 October 2026).

The corpus was extended with the four articles of the field-study search of 16 October 2026: Turner and Holmes (2015),
Ta and Babel (2020b), Ta and Babel (2023b; second report of the Ta_Babel_2023 programme) and Ling et al. (2026). The
extraction rule is unchanged. The relations of the rebuilt corpus are compared with those of the previous corpus
(outputs/models/_corpus_37_articles_2026-10-16/chunks.jsonl); every relation that is new is listed (no sampling), as in
round 3. A relation is one (polymer, mechanism, sentence) triple, as in src/rag/llm_interpretation_v2.py (_relations).

Outputs (Revision/04_LLM_evaluation/knowledge_graph_check/round 4/):
  To_send_HBT_VKHB/KG_relation_check_sheet_round4_HBT.xlsx, ..._VKHB.xlsx   (identical sheets, one per checker)
  To_send_HBT_VKHB/Instructions_KG_check_round4.docx
  To_send_HBT_VKHB/00_Email_text_do_not_attach.md
  round4_relation_counts.csv  (relations before and after, by study and mechanism; relations no longer extracted)
Usage: python scripts/build_kg_check_round4.py
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

import pandas as pd
from docx import Document
from docx.shared import Pt
from openpyxl.styles import Alignment, Font
from openpyxl.worksheet.datavalidation import DataValidation

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src" / "rag"))
import llm_interpretation_v2 as interp  # noqa: E402

OLD_CHUNKS = ROOT / "outputs" / "models" / "_corpus_37_articles_2026-10-16" / "chunks.jsonl"
NEW_CHUNKS = ROOT / "data" / "chunks.jsonl"
OUT = ROOT / "Manuscript" / "Submission_Env pollution" / "Revision" / "04_LLM_evaluation" / "knowledge_graph_check" / "round 4"
SEND = OUT / "To_send_HBT_VKHB"
CHECKERS = ["HBT", "VKHB"]
ARTICLES = ("Turner and Holmes (2015), Ta and Babel (2020b), Ta and Babel (2023b) and Ling et al. (2026)")
SHEET_INSTRUCTIONS = [
    f"Each row is one relation of the knowledge graph that is new after four articles were added to the corpus ({ARTICLES}): "
    "a polymer, a mechanism (or surface property) and the sentence it was taken from. All new relations are listed (no sampling).",
    "polymer_correct: Y if the sentence is about the named polymer (the property or mechanism is attributed to it), otherwise N.",
    "mechanism_correct: Y if the sentence states this mechanism or surface property in the sense of the label, otherwise N "
    "(for example: 'pore' in 'porous media', 'diffusion' in a kinetic model, complexation in solution rather than on the particle).",
    "A relation counts as correct only if both answers are Y.",
    "Judge the sentence as printed; open the article (column paper) only if the sentence cannot be understood alone, and say so in the comment.",
    "Work alone, without discussing with the other checker and without AI tools.",
]
LABEL_RULES = {
    "electrostatic attraction": "electrostatic attraction or interaction between the particle surface and metal ions, including its "
                                "dependence on surface charge, as part of the uptake of a metal",
    "surface complexation": "complexation or chelation of metal ions by groups on the particle surface (not complexation in solution, "
                            "for example with dissolved organic matter)",
    "ion exchange": "exchange of metal ions with ions on the particle surface, as part of the uptake of a metal",
    "surface precipitation": "precipitation of a metal on the particle surface, as part of its uptake",
    "oxygen-containing functional groups": "presence, formation or increase of O-containing groups (carbonyl, carboxyl, hydroxyl, "
                                           "C=O, C–O) on the polymer, or their role in sorption",
    "surface area and pore structure": "specific surface area, pores, roughness or cracks of the polymer, or their role in sorption",
}


def triples(chunks_file: Path) -> pd.DataFrame:
    interp.CHUNKS_FILE = chunks_file
    interp._paper_chunks.cache_clear()
    interp._relations.cache_clear()
    rows = [{"paper": r["paper"], "polymer": p, "mechanism": m,
             "sentence": re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f]", " ", r["text"])}
            for r in interp._relations() for p in r["polymers"] for m in r["mechanisms"]]
    return pd.DataFrame(rows)


def write_sheet(path: Path, rel: pd.DataFrame, counts: pd.DataFrame) -> None:
    with pd.ExcelWriter(path, engine="openpyxl") as writer:
        pd.DataFrame({"instructions": SHEET_INSTRUCTIONS}).to_excel(writer, sheet_name="Instructions", index=False)
        rel.to_excel(writer, sheet_name="Relations", index=False)
        counts.to_excel(writer, sheet_name="Counts", index=False)
        ws = writer.sheets["Relations"]
        for col, width in {"A": 6, "B": 16, "C": 9, "D": 30, "E": 110, "F": 14, "G": 14, "H": 30}.items():
            ws.column_dimensions[col].width = width
        for row in ws.iter_rows(min_row=2):
            for cell in row:
                cell.alignment = Alignment(wrap_text=True, vertical="top")
        for cell in ws[1]:
            cell.font = Font(bold=True)
        ws.freeze_panes = "B2"
        dv = DataValidation(type="list", formula1='"Y,N"', allow_blank=True)
        ws.add_data_validation(dv)
        dv.add(f"F2:G{len(rel) + 1}")
        writer.sheets["Instructions"].column_dimensions["A"].width = 140


def write_instructions(path: Path, rel: pd.DataFrame, n_before: int, n_after: int, lost: int) -> None:
    doc = Document()
    doc.styles["Normal"].font.name, doc.styles["Normal"].font.size = "Calibri", Pt(11)
    doc.add_heading(f"Knowledge-graph relation check – Round 4 ({len(rel)} new relations)", level=1)
    doc.add_paragraph("Manuscript ENVPOL-D-26-10175 (revision). Checkers: HBT and VKHB. Time needed: about "
                      f"{max(10, round(len(rel) * 0.6 / 5) * 5)} minutes per person.")
    doc.add_heading("1. What this round is", level=2)
    by_paper = rel["paper"].value_counts()
    doc.add_paragraph(
        f"You did rounds 1–3. After a search for field studies, four articles were added to the corpus: {ARTICLES}. Turner and "
        "Holmes (2015) is a laboratory study made in river water; the other three are field studies. Ta and Babel (2023b) is the "
        "second report of the sampling programme already in the corpus as Ta_Babel_2023, so its relations carry that code. The "
        f"extraction rule did not change. The graph had {n_before} relations and now has {n_after}; {len(rel)} relations are new "
        f"and only these have to be checked ({', '.join(f'{n} from {p}' for p, n in by_paper.items())})."
        + (f" {lost} relations of the previous graph are no longer extracted (they are not to be checked)." if lost else ""))
    doc.add_paragraph("No sampling: all new relations are in the sheet. Same task and same rules as rounds 1–3: two Y/N answers per "
                      "row. Do not re-check the earlier rounds.")
    doc.add_heading("2. Your file", level=2)
    doc.add_paragraph(f"Open your own file, sheet \"Relations\" ({len(rel)} rows). Fill in only column F (polymer_correct: Y/N), "
                      "column G (mechanism_correct: Y/N) and, if needed, column H (comment). Do not change columns A–E, the order of "
                      "the rows or the item numbers.")
    doc.add_heading("3. Work independently", level=2)
    doc.add_paragraph("Each checker works alone. Do not discuss the items and do not look at the other file until both are returned. "
                      "Do not use an AI tool: the result is reported as a check by the authors. Judge the sentence as printed; open "
                      "the article only if the sentence cannot be understood alone, and then write \"article opened\" in the comment.")
    doc.add_heading("4. The two questions for each row", level=2)
    doc.add_paragraph("Question 1 – polymer (column F). Is the statement in the sentence about the polymer named in the row? Y: the "
                      "property or mechanism is attributed to that polymer. N: the polymer is only mentioned, and the property or "
                      "mechanism belongs to another polymer, to another material (for example sediment, biofilm or dissolved organic "
                      "matter) or to no polymer.")
    doc.add_paragraph("Question 2 – mechanism (column G). Does the sentence report the mechanism or surface property named in the "
                      "row, in the sense below?")
    labels = [m for m in LABEL_RULES if m in set(rel["mechanism"])]
    table = doc.add_table(rows=1, cols=2)
    table.style = "Table Grid"
    table.rows[0].cells[0].text, table.rows[0].cells[1].text = "Label in the sheet", "Counts as correct (Y) when the sentence reports"
    for m in labels:
        cells = table.add_row().cells
        cells[0].text, cells[1].text = m, LABEL_RULES[m]
    doc.add_paragraph("A relation is correct only if both answers are Y.")
    doc.add_heading("5. Rules for cases that are not obvious (same as rounds 1–3)", level=2)
    for rule in [
        "One sentence that names several polymers appears in several rows, one per polymer. Answer each row for its own polymer.",
        "Surface properties (oxygen-containing groups; surface area and pore structure): a sentence that only describes the "
        "property for the polymer counts as Y; no statement on sorption is needed.",
        "Process mechanisms (electrostatic attraction, surface complexation, ion exchange, surface precipitation) must take part "
        "in the uptake of a metal. If stated only for an organic compound, or negated, answer N for the mechanism.",
        "New in this round, because three articles are field studies: a property of field-collected particles counts for the "
        "polymer only if the sentence attributes it to particles of that polymer. Statements about sediments, biofilms, "
        "suspended matter or the water are N for the polymer.",
        "Keyword used in another sense – N for surface area and pore structure: diffusion in kinetic models, \"porous media\", "
        "filter or membrane pore sizes, \"pore water\".",
        "Changes caused by aging or weathering (UV, heat, acid, beaching) still count for the polymer.",
        "Statements quoted from other studies are judged in the same way.",
        "Fragments (table rows, figure captions without a statement, broken text): N for the mechanism, comment \"fragment\".",
        "Still unclear after opening the article: N, comment \"unclear\". Do not guess.",
    ]:
        doc.add_paragraph(rule, style="List Bullet")
    doc.add_heading("6. When you have finished", level=2)
    doc.add_paragraph(f"Check that all {len(rel)} rows have Y or N in both column F and column G. Save the file with its name "
                      "unchanged and send it to the corresponding author (Xuan Cuong Nguyen) only, not to the other checker. The "
                      "two files are then compared (precision for each checker, agreement and Cohen's kappa, rows on which you "
                      "disagree); you then discuss only the disagreeing rows. The result is added to Section 2.4 and Text S5, next "
                      "to rounds 1–3, and reported as found, whatever its value.")
    table = doc.add_table(rows=1, cols=2)
    table.style = "Table Grid"
    table.rows[0].cells[0].text, table.rows[0].cells[1].text = "Checker", "Fill in and return this file (name unchanged)"
    for c in CHECKERS:
        cells = table.add_row().cells
        cells[0].text, cells[1].text = c, f"KG_relation_check_sheet_round4_{c}.xlsx"
    doc.save(path)


def write_email(path: Path, n: int) -> None:
    minutes = max(10, round(n * 0.6 / 5) * 5)
    path.write_text(f"""# E-mail to HBT and VKHB (one e-mail to each person, each with their own sheet)

## Tiếng Việt

**Tiêu đề:** [ENVPOL revision] Kiểm tra bổ sung quan hệ knowledge graph – vòng 4 ({n} dòng, ~{minutes} phút)

Chào anh/chị,

Sau đợt tìm kiếm các nghiên cứu hiện trường, corpus có thêm 4 bài (Turner & Holmes 2015; Ta & Babel 2020b; Ta & Babel 2023b; Ling et al. 2026) và knowledge graph có thêm {n} quan hệ mới. Em/anh nhờ anh/chị kiểm tra như các vòng trước, lần này chỉ {n} dòng mới.

- File của anh/chị: **KG_relation_check_sheet_round4_<tên viết tắt>.xlsx** (đính kèm), sheet **Relations**.
- Chỉ điền cột F (polymer_correct: Y/N), cột G (mechanism_correct: Y/N) và cột H (ghi chú, nếu cần).
- Hướng dẫn chi tiết: **Instructions_KG_check_round4.docx** (đính kèm). Lưu ý quy tắc mới cho bài hiện trường (mục 5).
- Làm độc lập, không trao đổi với người kia và không dùng công cụ AI.
- Xong thì giữ nguyên tên file và gửi lại cho em/anh trước ngày [ngày].

Cảm ơn anh/chị.

## English

**Subject:** [ENVPOL revision] Knowledge-graph relation check – round 4 ({n} rows, ~{minutes} min)

Dear [name],

After the search for field studies, four articles were added to the corpus (Turner & Holmes 2015; Ta & Babel 2020b; Ta & Babel 2023b; Ling et al. 2026), and the knowledge graph gained {n} new relations. Could you check them as in the earlier rounds? Only these {n} rows need checking.

- Your file: **KG_relation_check_sheet_round4_<initials>.xlsx** (attached), sheet **Relations**.
- Fill in only column F (polymer_correct: Y/N), column G (mechanism_correct: Y/N) and, if needed, column H (comment).
- Detailed instructions: **Instructions_KG_check_round4.docx** (attached); note the new rule for field studies (section 5).
- Please work independently, without discussing with the other checker and without AI tools.
- Return the file with its name unchanged by [date].

Thank you.

## Attachments
- To HBT: Instructions_KG_check_round4.docx + KG_relation_check_sheet_round4_HBT.xlsx
- To VKHB: Instructions_KG_check_round4.docx + KG_relation_check_sheet_round4_VKHB.xlsx
""", encoding="utf-8")


def main() -> None:
    old, new = triples(OLD_CHUNKS), triples(NEW_CHUNKS)
    key = ["paper", "polymer", "mechanism", "sentence"]
    merged = new.merge(old.assign(_old=True), on=key, how="left")
    fresh = merged[merged["_old"].isna()].drop(columns="_old").reset_index(drop=True)
    lost = old.merge(new.assign(_new=True), on=key, how="left")
    lost = lost[lost["_new"].isna()]
    rel = fresh.sort_values(["paper", "mechanism", "polymer"]).reset_index(drop=True)
    rel.insert(0, "item", range(1, len(rel) + 1))
    rel["polymer_correct (Y/N)"], rel["mechanism_correct (Y/N)"], rel["comment"] = "", "", ""
    counts = rel.groupby(["paper", "mechanism"]).size().rename("relations").reset_index()
    OUT.mkdir(parents=True, exist_ok=True)
    if rel.empty:   # nothing to check: no sentence of the added articles names a polymer together with a mechanism
        sent = [r for r in interp._relations() if r["paper"] in {"Turner_2015", "Ta_Babel_2020b", "Ling_2026", "Ta_Babel_2023"}]
        pd.DataFrame(sent).to_csv(OUT / "round4_sentences_without_polymer.csv", index=False)
        (OUT / "README_round4.md").write_text(
            f"Round 4 not needed: the graph has {len(new)} relations before and after the four articles were added "
            f"({len(lost)} no longer extracted). Their mechanism sentences name no specific polymer (see "
            "round4_sentences_without_polymer.csv), so they add no polymer-mechanism relation.\n", encoding="utf-8")
        print(f"relations before {len(old)}, after {len(new)}; new 0; no longer extracted {len(lost)}: no check sheets written")
        return
    SEND.mkdir(parents=True, exist_ok=True)
    for c in CHECKERS:
        write_sheet(SEND / f"KG_relation_check_sheet_round4_{c}.xlsx", rel, counts)
    write_instructions(SEND / "Instructions_KG_check_round4.docx", rel, len(old), len(new), len(lost))
    write_email(SEND / "00_Email_text_do_not_attach.md", len(rel))
    summary = pd.concat([
        old.groupby("paper").size().rename("relations_before"), new.groupby("paper").size().rename("relations_after"),
        lost.groupby("paper").size().rename("relations_no_longer_extracted")], axis=1).fillna(0).astype(int)
    summary.loc["TOTAL"] = summary.sum()
    summary.to_csv(OUT / "round4_relation_counts.csv")
    print(f"relations before {len(old)} ({old['paper'].nunique()} studies), after {len(new)} ({new['paper'].nunique()} studies); "
          f"new {len(rel)}; no longer extracted {len(lost)}")
    print(counts.to_string(index=False))
    if len(lost):
        print("WARNING: some relations of the previous graph are no longer extracted; see round4_relation_counts.csv")


if __name__ == "__main__":
    main()
