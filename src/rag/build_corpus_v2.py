"""Rebuild the retrieval corpus, the vector index and the knowledge graph for the revised dataset.

The corpus of the submitted version held 24 papers, including studies that are no longer in the
dataset (seawater, nano-sized particles). The revised corpus holds the laboratory studies of the
current dataset and the field studies used for the field comparison. Papers are identified by their
study key (e.g. ``Wang_2021``), so citations in the explanations can be traced to the corpus.

The previous artefacts are copied to outputs/models/backup_24paper_corpus/ before being replaced.
"""

from __future__ import annotations

import json
import re
import shutil
import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
MODULE_DIR = Path(__file__).resolve().parent
for path in (PROJECT_ROOT, MODULE_DIR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from config import CHUNKS_FILE, CHUNKS_METADATA_FILE, FAISS_INDEX_FILE, KNOWLEDGE_GRAPH_FILE, MODELS_DIR  # noqa: E402
from document_ingestion import _chunk_text, _dataset_summary_chunks, extract_metadata  # noqa: E402

INDEX_FILE = PROJECT_ROOT / "Manuscript" / "Submission_Env pollution" / "Revision" / "05_Literature_search" / "literature_index.csv"
CORPUS_LIST = MODELS_DIR / "corpus_v2_papers.csv"
PAPERS = "Manuscript/Submission_Env pollution/Revision/07_Papers/additional paper/new wos/"
# Laboratory studies added after the Web of Science search (14 October 2026). A study may have more than one article:
# LiuZhou_2025 is Liu et al. (2025) and Zhou et al. (2025), which share their pristine isotherms; both articles are in the corpus.
EXTRA_PAPERS = {
    "Yan_2023": [("10.1007/s11356-023-27906-1", PAPERS + "s11356-023-27906-1.pdf")],
    "LiuZhou_2025": [("10.1002/ldr.5392", PAPERS + "Land Degrad Dev - 2024 - Liu - The Effect of Different Aging Methods on the Heavy "
                      "Metal Adsorption Capacity o.pdf"),
                     ("10.1016/j.jece.2025.120355", PAPERS + "1-s2.0-S2213343725050523-main.pdf")],
}
# Added after the field-study search of 16 October 2026 (corpus round 4, 17 October 2026): one laboratory study (isotherms in
# river water) and the three field studies with paired Cu on particles and in water. Ta & Babel (2023b, STOTEN) is the second
# report of the Ta_Babel_2023 sampling programme and is added to that study.
FIELD_SEARCH = "Manuscript/Submission_Env pollution/Revision/07_Papers/field search 2026-10/"
EXTRA_PAPERS_KIND = {"Turner_2015": "laboratory", "Ta_Babel_2020b": "field", "Ta_Babel_2023": "field", "Ling_2026": "field"}
EXTRA_PAPERS.update({
    "Turner_2015": [("10.1071/EN14143", FIELD_SEARCH + "turner2015.pdf")],
    "Ta_Babel_2020b": [("10.1016/j.chemosphere.2020.127234", FIELD_SEARCH + "1-s2.0-S0045653520314272-main.pdf")],
    "Ta_Babel_2023": [("10.1016/j.scitotenv.2023.165405", FIELD_SEARCH + "1-s2.0-S0048969723040287-main.pdf")],
    "Ling_2026": [("10.1016/j.jece.2026.122083", FIELD_SEARCH + "1-s2.0-S2213343726010572-main.pdf")],
})


def _parse_pdf(path: Path) -> str:
    """Text in reading order (text blocks, so that the two columns of a page are not interleaved), without the
    reference list, e-mail addresses and line-break hyphenation."""
    import fitz

    parts = []
    with fitz.open(path) as doc:
        for page in doc:
            for block in page.get_text("blocks"):
                if block[6] == 0 and block[4].strip():
                    parts.append(block[4].strip())
    text = "\n".join(parts)
    cut = [m.start() for m in re.finditer(r"\n\s*(?:References|REFERENCES|Literature cited)\s*\n", text)]
    if cut and cut[-1] > 0.5 * len(text):
        text = text[:cut[-1]]
    text = re.sub(r"\S+@\S+", " ", text)
    text = re.sub(r"(?<=[a-z])-\n(?=[a-z])", "", text)
    return re.sub(r"\s+", " ", text)


def corpus_papers() -> pd.DataFrame:
    idx = pd.read_csv(INDEX_FILE)
    lab = idx[idx["role"] == "laboratory dataset"]
    field = idx[(idx["role"] == "field evidence") & idx["status"].astype(str).str.contains("tier 1")]
    papers = pd.concat([lab.assign(kind="laboratory"), field.assign(kind="field")])
    papers = papers[papers["pdf_path"].notna()].copy()
    papers["pdf"] = papers["pdf_path"].str.split("; ").str[0]
    papers = papers[["key", "doi", "kind", "pdf"]]
    extra = [{"key": k, "doi": d, "kind": EXTRA_PAPERS_KIND.get(k, "laboratory"), "pdf": f} for k, docs in EXTRA_PAPERS.items() for d, f in docs
             if not ((papers["key"] == k) & (papers["doi"] == d)).any()]
    papers = pd.concat([papers, pd.DataFrame(extra)], ignore_index=True)
    missing = [p for p in papers["pdf"] if not (PROJECT_ROOT / p).exists()]
    if missing:
        raise FileNotFoundError(f"PDF not found: {missing}")
    return papers.reset_index(drop=True)


def main() -> None:
    backup = MODELS_DIR / "backup_37article_corpus_2026-10-17"
    backup.mkdir(parents=True, exist_ok=True)
    for f in (CHUNKS_FILE, CHUNKS_METADATA_FILE, FAISS_INDEX_FILE, KNOWLEDGE_GRAPH_FILE):
        if f.exists() and not (backup / f.name).exists():
            shutil.copy2(f, backup / f.name)

    papers = corpus_papers()
    chunks, counts, done = [], [], {}
    for row in papers.itertuples():
        text = _parse_pdf(PROJECT_ROOT / row.pdf)
        parts = _chunk_text(text)
        counts.append(len(parts))
        start = done.get(row.key, 0)                 # chunk numbers continue when a study has two articles
        for i, part in enumerate(parts, start=start):
            chunks.append({"chunk_id": f"{row.key}::chunk_{i:04d}", "paper_id": row.key, "source_type": f"{row.kind}_paper",
                           "text": part, "metadata": extract_metadata(part)})
        done[row.key] = start + len(parts)
        print(f"{row.key}: {len(parts)} chunks", flush=True)
    papers["n_chunks"] = counts
    chunks += _dataset_summary_chunks()
    with CHUNKS_FILE.open("w", encoding="utf-8") as handle:
        for chunk in chunks:
            handle.write(json.dumps(chunk, ensure_ascii=False) + "\n")
    papers.to_csv(CORPUS_LIST, index=False)
    for f in (FAISS_INDEX_FILE, CHUNKS_METADATA_FILE, KNOWLEDGE_GRAPH_FILE):
        if f.exists():
            f.unlink()

    from knowledge_graph import build_graph
    from vector_store import build_index

    build_index()
    graph = build_graph()
    print(f"corpus: {papers['key'].nunique()} studies, {len(papers)} articles ({(papers['kind'] == 'laboratory').sum()} laboratory, {(papers['kind'] == 'field').sum()} field), "
          f"{len(chunks)} chunks; graph: {graph.number_of_nodes()} nodes, {graph.number_of_edges()} edges")
    empty = papers.loc[papers["n_chunks"] == 0, "key"].tolist()
    if empty:
        print("WARNING: no text extracted from", empty)


if __name__ == "__main__":
    main()
