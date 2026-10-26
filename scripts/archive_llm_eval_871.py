"""Copy the results of the LLM evaluation run with the 871-observation model (before the two Web of Science studies were added)
to dated folders, before the evaluation is run again with the final model (951 observations, 31 studies). Nothing is deleted.

Usage: python scripts/archive_llm_eval_871.py
"""

from __future__ import annotations

import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
REPORTS = ROOT / "outputs" / "reports"
EXPERT = ROOT / "Manuscript" / "Submission_Env pollution" / "Revision" / "04_LLM_evaluation"
TARGETS = {
    REPORTS / "_llm_eval_871_2026-10-16": sorted(REPORTS.glob("llm_eval_v2_*_final.csv")),
    ROOT / "outputs" / "models" / "_corpus_34_articles_2026-10-16": [ROOT / "outputs" / "models" / n for n in
                                                                    ("corpus_v2_papers.csv", "faiss_index.bin", "chunks_metadata.pkl", "knowledge_graph.gpickle")]
                                                                   + [ROOT / "data" / "chunks.jsonl"],
    EXPERT / "_previous_2026-10-16_llm_eval_871": [f for f in EXPERT.iterdir() if f.is_file()],
}

for folder, files in TARGETS.items():
    folder.mkdir(parents=True, exist_ok=True)
    for f in files:
        if f.exists():
            shutil.copy2(f, folder / f.name)
            print(f"copied {f.relative_to(ROOT)} -> {folder.relative_to(ROOT)}")
        else:
            print(f"not found (skipped): {f.relative_to(ROOT)}")
cases = REPORTS / "_llm_eval_871_2026-10-16" / "llm_eval_v2_cases_final.csv"
assert cases.exists(), "the cases of the first run are needed for the rerun"
print("done")
