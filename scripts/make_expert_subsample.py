"""Blinded expert scoring sheet for a stratified subset of the final LLM evaluation (15 of 30 cases, 60 explanations)."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "src" / "rag"))
from config import RANDOM_SEED, REPORTS_DIR  # noqa: E402
from llm_eval_v2 import DIMENSIONS, EXPERT_DIR, RUBRIC  # noqa: E402
from llm_interpretation_v2 import plain_text  # noqa: E402

PER_STRATUM = {"river_level": 4, "low_laboratory": 3, "in_range": 7, "above_range": 1}
INSTRUCTIONS = [
    "Each row is one explanation of one model prediction. The column 'facts' gives the model output for that case; treat it as correct.",
    "Score the explanation in each of the six columns from 1 to 5, using the sheet 'rubric'. Use whole numbers.",
    "Rows are in random order. Four explanations belong to each case; please score every row on its own merits.",
    "Do not try to guess how an explanation was produced. Add a comment only if something is wrong or unclear.",
    "Source ids such as Wang_2021 refer to articles in the study corpus (first author and year).",
]


def main() -> None:
    rng = np.random.default_rng(RANDOM_SEED)
    items = pd.read_csv(REPORTS_DIR / "llm_eval_v2_explanations_final.csv")
    cases = items.drop_duplicates("case_id")[["case_id", "stratum", "sparse_warning_needed"]]
    chosen = []
    for stratum, k in PER_STRATUM.items():
        pool = cases[cases["stratum"] == stratum]
        sparse = pool[pool["sparse_warning_needed"]]["case_id"].tolist()
        keep = list(rng.choice(sparse, size=1)) if sparse and k > 1 else []      # keep sparsely represented polymers in the subset
        rest = [c for c in pool["case_id"] if c not in keep]
        chosen += keep + list(rng.choice(rest, size=k - len(keep), replace=False))
    sub = items[items["case_id"].isin(chosen)].sort_values("item_id")
    sheet = sub[["item_id", "facts", "explanation"]].copy()
    sheet["explanation"] = sheet["explanation"].map(plain_text)
    for d in DIMENSIONS:
        sheet[d] = ""
    sheet["comment"] = ""
    out = EXPERT_DIR / "expert_scoring_sheet_final_15cases.xlsx"
    with pd.ExcelWriter(out) as writer:
        pd.DataFrame({"instructions": INSTRUCTIONS}).to_excel(writer, sheet_name="instructions", index=False)
        sheet.to_excel(writer, sheet_name="score_here", index=False)
        pd.DataFrame({"rubric": RUBRIC.split("\n")}).to_excel(writer, sheet_name="rubric", index=False)
        for name, widths in {"score_here": {"A": 10, "B": 70, "C": 100}, "instructions": {"A": 130}, "rubric": {"A": 130}}.items():
            ws = writer.sheets[name]
            for col, w in widths.items():
                ws.column_dimensions[col].width = w
            for row in ws.iter_rows(min_row=2):
                for cell in row:
                    cell.alignment = cell.alignment.copy(wrap_text=True, vertical="top")
    sub[["item_id", "case_id", "stratum", "condition"]].to_csv(EXPERT_DIR / "KEY_do_not_share_with_raters_final_15cases.csv", index=False)

    scores = pd.read_csv(REPORTS_DIR / "llm_eval_v2_scores_final.csv")
    print(f"{out.name}: {sub['case_id'].nunique()} cases, {len(sub)} explanations; "
          f"sparse-polymer cases: {int(cases[cases.case_id.isin(chosen)]['sparse_warning_needed'].sum())}")
    print(cases[cases.case_id.isin(chosen)].groupby("stratum").size().to_string())
    print("\nLLM-judge overall score, subset versus all 30 cases")
    print(pd.concat({"subset": scores[scores.case_id.isin(chosen)].groupby(["judge", "condition"])["overall_weighted"].mean().unstack(),
                     "all": scores.groupby(["judge", "condition"])["overall_weighted"].mean().unstack()}).round(2).to_string())


if __name__ == "__main__":
    main()
