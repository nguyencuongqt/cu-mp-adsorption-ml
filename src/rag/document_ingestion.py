"""Document ingestion for the LLM Interpretation System.

Parses PDFs from ``data/papers/`` with pdfplumber, chunks text into overlapping
300-token windows, extracts lightweight metadata with regex, and adds
dataset-derived group summaries from ``data/dataset.csv``. Output is written as
JSONL to ``data/chunks.jsonl``.
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path
from typing import Any

import pandas as pd
import pdfplumber
from tqdm.auto import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from config import (  # noqa: E402
    CAT_COLS,
    CHUNK_OVERLAP,
    CHUNK_TOKENS,
    CHUNKS_FILE,
    DATASET_CSV,
    NUM_COLS,
    PAPERS_DIR,
    TARGET,
)


POLYMERS = ["CPE", "HPE", "LPE", "PA", "PBAT", "PE", "PET", "PLA", "PMMA", "PP", "PS", "PVC", "TPU"]
MECHANISMS = [
    "electrostatic",
    "complexation",
    "surface complexation",
    "ion exchange",
    "precipitation",
    "diffusion",
    "adsorption",
    "partition",
    "hydrophobic",
    "chelation",
]
ISOTHERMS = ["Langmuir", "Freundlich", "Temkin", "Sips", "Dubinin", "Redlich"]


def _tokens(text: str) -> list[str]:
    return re.findall(r"\S+", text)


def _chunk_text(text: str, chunk_tokens: int = CHUNK_TOKENS, overlap: int = CHUNK_OVERLAP) -> list[str]:
    words = _tokens(text)
    if not words:
        return []
    chunks: list[str] = []
    step = max(chunk_tokens - overlap, 1)
    for start in range(0, len(words), step):
        chunk = " ".join(words[start : start + chunk_tokens])
        if chunk:
            chunks.append(chunk)
        if start + chunk_tokens >= len(words):
            break
    return chunks


def _range(pattern: str, text: str) -> str | None:
    matches = re.findall(pattern, text, flags=re.I)
    if not matches:
        return None
    values: list[float] = []
    for match in matches:
        if isinstance(match, tuple):
            for part in match:
                try:
                    values.append(float(part))
                except ValueError:
                    pass
        else:
            try:
                values.append(float(match))
            except ValueError:
                pass
    if not values:
        return None
    return f"{min(values):g}-{max(values):g}"


def extract_metadata(text: str) -> dict[str, Any]:
    polymers = sorted({poly for poly in POLYMERS if re.search(rf"\b{re.escape(poly)}\b", text, flags=re.I)})
    mechanisms = sorted({m for m in MECHANISMS if re.search(re.escape(m), text, flags=re.I)})
    isotherms = sorted({iso for iso in ISOTHERMS if re.search(rf"\b{iso}\b", text, flags=re.I)})
    return {
        "polymers": polymers,
        "pH_range": _range(r"pH\s*(?:=|:|of|at|from)?\s*(\d+(?:\.\d+)?)", text),
        "Ce_range": _range(r"(?:Ce|C0|initial concentration)[^\d]{0,20}(\d+(?:\.\d+)?)", text),
        "mechanisms": mechanisms,
        "isotherm_type": isotherms,
        "qe_values": re.findall(r"q[eE](?:max)?[^\d]{0,20}(\d+(?:\.\d+)?)", text, flags=re.I)[:10],
    }


def _parse_pdf(path: Path) -> str:
    pages: list[str] = []
    with pdfplumber.open(path) as pdf:
        for page in pdf.pages:
            pages.append(page.extract_text() or "")
    return "\n".join(pages)


def _paper_chunks() -> list[dict[str, Any]]:
    chunks: list[dict[str, Any]] = []
    pdfs = sorted(PAPERS_DIR.glob("*.pdf"))
    if not pdfs:
        return chunks
    if len(pdfs) != 24:
        print(f"Warning: expected 24 PDFs in {PAPERS_DIR}, found {len(pdfs)}.")

    for pdf_path in tqdm(pdfs, desc="Parsing PDFs"):
        text = _parse_pdf(pdf_path)
        paper_id = pdf_path.stem
        for idx, chunk_text in enumerate(_chunk_text(text)):
            chunks.append(
                {
                    "chunk_id": f"{paper_id}::chunk_{idx:04d}",
                    "paper_id": paper_id,
                    "source_type": "paper",
                    "text": chunk_text,
                    "metadata": extract_metadata(chunk_text),
                }
            )
    return chunks


def _dataset_summary_chunks() -> list[dict[str, Any]]:
    if not DATASET_CSV.exists():
        return []
    df = pd.read_csv(DATASET_CSV)
    required = set(CAT_COLS + NUM_COLS + [TARGET])
    missing = sorted(required - set(df.columns))
    if missing:
        print(f"Warning: dataset.csv missing columns for summaries: {missing}")
        return []

    for col in NUM_COLS + [TARGET]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    chunks: list[dict[str, Any]] = []
    for (ret, ags), group in df.groupby(["ReT", "AgS"], dropna=False):
        qe_mean = group[TARGET].mean()
        qe_sd = group[TARGET].std()
        ce_min, ce_max = group["Ce"].min(), group["Ce"].max()
        ph_min, ph_max = group["pH"].min(), group["pH"].max()
        text = (
            f"From {len(group)} cases with {ret} ({ags}): mean qe={qe_mean:.4g}±{qe_sd:.4g}; "
            f"Ce {ce_min:.4g}–{ce_max:.4g}; pH {ph_min:.4g}–{ph_max:.4g}."
        )
        chunks.append(
            {
                "chunk_id": f"dataset::{ret}::{ags}",
                "paper_id": "dataset",
                "source_type": "dataset",
                "text": text,
                "metadata": {
                    "polymers": [str(ret)],
                    "pH_range": f"{ph_min:g}-{ph_max:g}",
                    "Ce_range": f"{ce_min:g}-{ce_max:g}",
                    "mechanisms": [],
                    "isotherm_type": [],
                    "qe_values": [float(qe_mean)] if pd.notna(qe_mean) else [],
                    "ReT": ret,
                    "AgS": ags,
                },
            }
        )
    return chunks


def ingest_all() -> list[dict[str, Any]]:
    """Parse all corpus inputs and save chunks to ``data/chunks.jsonl``."""
    CHUNKS_FILE.parent.mkdir(parents=True, exist_ok=True)
    chunks = _paper_chunks() + _dataset_summary_chunks()
    with CHUNKS_FILE.open("w", encoding="utf-8") as handle:
        for chunk in chunks:
            handle.write(json.dumps(chunk, ensure_ascii=False) + "\n")
    print(f"Saved {len(chunks)} chunks to {CHUNKS_FILE}")
    return chunks


if __name__ == "__main__":
    ingest_all()
