"""Sample of knowledge-graph relations for a hand check (precision of the keyword-derived relations).

A relation is one (polymer, mechanism, supporting sentence) triple of the sentence-level relations that the interpretation
layer uses as graph evidence (src/rag/llm_interpretation_v2.py, _relations). The sample is stratified by mechanism in
proportion to the number of relations, with every mechanism represented; the seed is fixed so the sheet can be rebuilt.

Output: Revision/04_LLM_evaluation/knowledge_graph_check/KG_relation_check_sheet.xlsx (to be filled in by the authors)
"""

from __future__ import annotations

import random
import re
import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.rag import llm_interpretation_v2 as interp  # noqa: E402

N_SAMPLE = 100
SEED = 20261010                      # round 2 (round 1: 20261009, before the rule for process mechanisms)
OUT_DIR = PROJECT_ROOT / "Manuscript" / "Submission_Env pollution" / "Revision" / "04_LLM_evaluation" / "knowledge_graph_check"
ROUND_1_SHEET = OUT_DIR / "round_1" / "KG_relation_check_sheet.xlsx"     # its relations were used to revise the rules
OUT_NAME = "KG_relation_check_sheet_round2.xlsx"

INSTRUCTIONS = [
    "Each row is one relation of the knowledge graph: a polymer, a mechanism (or surface property) and the sentence it was taken from.",
    "polymer_correct: Y if the sentence is about the named polymer (the property or mechanism is attributed to it), otherwise N.",
    "mechanism_correct: Y if the sentence states this mechanism or surface property in the sense of the label, otherwise N "
    "(for example: 'pore' in 'porous media', 'diffusion' in a kinetic model, complexation in solution rather than on the particle).",
    "A relation counts as correct only if both answers are Y. Precision = correct relations / relations checked.",
    "Judge the sentence as printed; open the article (column paper) only if the sentence cannot be understood alone, and say so in the comment.",
    "Two authors should fill in separate copies without seeing each other's answers; disagreements are then resolved by discussion.",
]


def main() -> None:
    relations = interp._relations()
    triples = [{"paper": r["paper"], "polymer": p, "mechanism": m, "sentence": re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f]", " ", r["text"]), "mentions_metal": r["copper"]}
               for r in relations for p in r["polymers"] for m in r["mechanisms"]]
    frame = pd.DataFrame(triples)
    n_all = len(frame)
    seen = pd.read_excel(ROUND_1_SHEET, sheet_name="Relations")[["polymer", "mechanism", "sentence"]].assign(seen=True)
    frame = frame.merge(seen, on=["polymer", "mechanism", "sentence"], how="left")
    frame = frame[frame["seen"].isna()].drop(columns="seen").reset_index(drop=True)      # only relations not checked in round 1
    rng = random.Random(SEED)
    counts = frame["mechanism"].value_counts()
    quota = (counts / counts.sum() * N_SAMPLE).round().astype(int).clip(lower=1).clip(upper=counts)
    while quota.sum() != N_SAMPLE:                       # adjust the largest stratum so that the total is exact
        quota[quota.idxmax()] += 1 if quota.sum() < N_SAMPLE else -1
    picks = []
    for mechanism, n in quota.items():
        idx = list(frame.index[frame["mechanism"] == mechanism])
        picks += rng.sample(idx, int(n))
    rng.shuffle(picks)
    sheet = frame.loc[picks].reset_index(drop=True)
    sheet.insert(0, "item", range(1, len(sheet) + 1))
    for col in ["polymer_correct (Y/N)", "mechanism_correct (Y/N)", "comment"]:
        sheet[col] = ""

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    population = pd.DataFrame({"relations in the graph": counts, "relations in the sample": quota})
    population.loc["total"] = population.sum()
    with pd.ExcelWriter(OUT_DIR / OUT_NAME) as writer:
        pd.DataFrame({"instructions": INSTRUCTIONS}).to_excel(writer, sheet_name="Instructions", index=False)
        sheet.to_excel(writer, sheet_name="Relations", index=False)
        population.to_excel(writer, sheet_name="Sampling")
        ws = writer.sheets["Relations"]
        for letter, width in {"A": 6, "B": 18, "C": 9, "D": 32, "E": 110, "F": 14, "G": 20, "H": 22, "I": 40}.items():
            ws.column_dimensions[letter].width = width
        for row in ws.iter_rows(min_row=2, min_col=5, max_col=5):
            row[0].alignment = row[0].alignment.copy(wrap_text=True, vertical="top")
        writer.sheets["Instructions"].column_dimensions["A"].width = 150
    print(f"{len(relations)} sentences, {n_all} relations in the graph, {len(frame)} not checked in round 1 "
          f"({frame['paper'].nunique()} papers); sample of {len(sheet)}")
    print(population.to_string())


if __name__ == "__main__":
    main()
