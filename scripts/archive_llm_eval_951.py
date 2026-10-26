"""Copy the results of the LLM evaluation run with the 951-observation model and the 37-article corpus (16 October 2026) to
dated folders, before the corpus is extended with the four articles of the field-study search (Turner and Holmes, 2015;
Ta and Babel, 2020b; 2023b; Ling et al., 2026) and the evaluation is run again with the 961-observation model. Nothing is deleted.

Usage: python scripts/archive_llm_eval_951.py
"""

from __future__ import annotations

import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
REPORTS = ROOT / "outputs" / "reports"
EXPERT = ROOT / "Manuscript" / "Submission_Env pollution" / "Revision" / "04_LLM_evaluation"
RUN = REPORTS / "_llm_eval_951_2026-10-16"
CORPUS = ROOT / "outputs" / "models" / "_corpus_37_articles_2026-10-16"
TARGETS = {
    RUN: sorted(REPORTS.glob("llm_eval_v2_*_final.csv")),
    CORPUS: [ROOT / "outputs" / "models" / n for n in ("corpus_v2_papers.csv", "faiss_index.bin", "chunks_metadata.pkl",
                                                       "knowledge_graph.gpickle")] + [ROOT / "data" / "chunks.jsonl"],
    EXPERT / "_previous_2026-10-16_llm_eval_951": [f for f in EXPERT.iterdir() if f.is_file()],
}

for folder, files in TARGETS.items():
    if folder.exists() and any(folder.iterdir()):
        print(f"{folder.relative_to(ROOT)} exists already; not overwritten")
        continue
    folder.mkdir(parents=True, exist_ok=True)
    for f in files:
        if f.exists():
            shutil.copy2(f, folder / f.name)
            print(f"copied {f.relative_to(ROOT)} -> {folder.relative_to(ROOT)}")
        else:
            print(f"not found (skipped): {f.relative_to(ROOT)}")
assert (RUN / "llm_eval_v2_cases_final.csv").exists(), "the cases of the previous run are needed for the rerun"
assert (CORPUS / "chunks.jsonl").exists(), "the previous corpus is needed for the round-4 check of the knowledge graph"
print("done")
