"""FAISS vector store for RAG retrieval."""

from __future__ import annotations

import json
import pickle
import sys
from pathlib import Path
from typing import Any

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
from tqdm.auto import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from config import (  # noqa: E402
    CHUNKS_FILE,
    CHUNKS_METADATA_FILE,
    EMBEDDING_MODEL,
    FAISS_INDEX_FILE,
    RAG_BATCH_SIZE,
)


_model: SentenceTransformer | None = None
_index: faiss.Index | None = None
_metadata: list[dict[str, Any]] | None = None
_polymer_vector_cache: dict[str, tuple[list[int], np.ndarray]] = {}


def _load_chunks() -> list[dict[str, Any]]:
    if not CHUNKS_FILE.exists():
        raise FileNotFoundError(f"Missing chunks file: {CHUNKS_FILE}. Run document_ingestion.ingest_all().")
    chunks: list[dict[str, Any]] = []
    with CHUNKS_FILE.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                chunks.append(json.loads(line))
    return chunks


def _embedding_model() -> SentenceTransformer:
    global _model
    if _model is None:
        _model = SentenceTransformer(EMBEDDING_MODEL)
    return _model


def _embed_texts(texts: list[str]) -> np.ndarray:
    model = _embedding_model()
    vectors: list[np.ndarray] = []
    for start in tqdm(range(0, len(texts), RAG_BATCH_SIZE), desc="Embedding chunks"):
        batch = texts[start : start + RAG_BATCH_SIZE]
        vectors.append(model.encode(batch, convert_to_numpy=True, normalize_embeddings=True))
    return np.vstack(vectors).astype("float32")


def build_index() -> tuple[faiss.Index, list[dict[str, Any]]]:
    """Build and save a cosine FAISS IndexFlatIP over chunk text."""
    chunks = _load_chunks()
    if not chunks:
        raise ValueError("No chunks found. Add PDFs/data and run document_ingestion.ingest_all().")
    texts = [chunk["text"] for chunk in chunks]
    embeddings = _embed_texts(texts)
    index = faiss.IndexFlatIP(embeddings.shape[1])
    index.add(embeddings)
    FAISS_INDEX_FILE.parent.mkdir(parents=True, exist_ok=True)
    faiss.write_index(index, str(FAISS_INDEX_FILE))
    with CHUNKS_METADATA_FILE.open("wb") as handle:
        pickle.dump(chunks, handle)
    print(f"Saved FAISS index to {FAISS_INDEX_FILE}")
    return index, chunks


def _load_index() -> tuple[faiss.Index, list[dict[str, Any]]]:
    global _index, _metadata
    if _index is None or _metadata is None:
        if not FAISS_INDEX_FILE.exists() or not CHUNKS_METADATA_FILE.exists():
            build_index()
        _index = faiss.read_index(str(FAISS_INDEX_FILE))
        with CHUNKS_METADATA_FILE.open("rb") as handle:
            _metadata = pickle.load(handle)
    return _index, _metadata


def _reconstruct_vectors(index: faiss.Index, indices: list[int]) -> np.ndarray:
    vectors = [index.reconstruct(int(idx)) for idx in indices]
    return np.vstack(vectors).astype("float32")


def retrieve(query: str, top_k: int = 5) -> list[dict[str, Any]]:
    """Return top-k chunks ranked by cosine similarity."""
    index, chunks = _load_index()
    query_vec = _embedding_model().encode([query], convert_to_numpy=True, normalize_embeddings=True).astype("float32")
    scores, indices = index.search(query_vec, min(top_k, len(chunks)))
    results: list[dict[str, Any]] = []
    for score, idx in zip(scores[0], indices[0]):
        if idx < 0:
            continue
        item = dict(chunks[int(idx)])
        item["score"] = float(score)
        results.append(item)
    return results


def retrieve_by_polymer(polymer: str, top_k: int = 8) -> list[dict[str, Any]]:
    """Pre-filter chunks by polymer keyword, then rank within that subset."""
    index, chunks = _load_index()
    polymer_upper = str(polymer).upper()
    if polymer_upper not in _polymer_vector_cache:
        filtered_indices = [
            idx for idx, chunk in enumerate(chunks)
            if polymer_upper in [str(p).upper() for p in chunk.get("metadata", {}).get("polymers", [])]
            or polymer_upper in chunk.get("text", "").upper()
        ]
        if filtered_indices:
            _polymer_vector_cache[polymer_upper] = (
                filtered_indices,
                _reconstruct_vectors(index, filtered_indices),
            )

    if polymer_upper not in _polymer_vector_cache:
        return retrieve(f"Cu adsorption on {polymer}", top_k=top_k)

    filtered_indices, chunk_vecs = _polymer_vector_cache[polymer_upper]
    query_vec = _embedding_model().encode([f"Cu adsorption on {polymer}"], convert_to_numpy=True, normalize_embeddings=True).astype("float32")
    scores = (chunk_vecs @ query_vec[0]).astype(float)
    order = np.argsort(scores)[::-1][:top_k]
    results: list[dict[str, Any]] = []
    for idx in order:
        item = dict(chunks[filtered_indices[int(idx)]])
        item["score"] = float(scores[int(idx)])
        results.append(item)
    return results


if __name__ == "__main__":
    build_index()
