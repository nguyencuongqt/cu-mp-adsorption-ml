"""Hybrid vector + graph retrieval for the LLM Interpretation System."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
MODULE_DIR = Path(__file__).resolve().parent
for path in (PROJECT_ROOT, MODULE_DIR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from config import DATASET_CSV  # noqa: E402
from knowledge_graph import get_polymer_profile, graph_retrieve  # noqa: E402
from vector_store import retrieve, retrieve_by_polymer  # noqa: E402


def classify_Ce(Ce: float) -> str:
    """Classify Ce as low, mid, or high."""
    value = float(Ce)
    if value < 1:
        return "low"
    if value <= 50:
        return "mid"
    return "high"


def _graph_weight(text: str) -> float:
    marker = "weight="
    if marker not in text:
        return 1.0
    try:
        return float(text.split(marker, 1)[1].split("]", 1)[0])
    except ValueError:
        return 1.0


def _graph_score_for_chunk(chunk: dict[str, Any], graph_lines: list[str]) -> float:
    text = (chunk.get("text", "") + " " + str(chunk.get("metadata", {}))).upper()
    score = 0.0
    for line in graph_lines:
        parts = line.upper().replace("POLYMER:", " ").replace("MECHANISM:", " ").split()
        if any(part.strip(":;,") in text for part in parts):
            score += _graph_weight(line)
    return score


def hybrid_retrieve(
    input_dict: dict[str, Any],
    pred_dict: dict[str, Any],
    vector_k: int = 8,
    hops: int = 2,
) -> tuple[str, list[dict[str, Any]]]:
    """Retrieve and rerank context from FAISS plus graph evidence."""
    query = (
        f"Cu2+ on {input_dict.get('ReT')} ({input_dict.get('AgS')}), "
        f"pH={input_dict.get('pH')}, Ce={input_dict.get('Ce')}, "
        f"lab_qe={pred_dict.get('lab_qe', 0):.4f}, "
        f"river_qe={pred_dict.get('river_qe', 0):.6f}"
    )
    vector_hits = retrieve(query, top_k=vector_k)
    graph_lines = graph_retrieve(
        {
            "polymer": input_dict.get("ReT"),
            "AgS": input_dict.get("AgS"),
            "Ce_bin": classify_Ce(float(input_dict.get("Ce", 0))),
        },
        hops=hops,
    )

    max_graph = max([_graph_weight(line) for line in graph_lines] + [1.0])
    reranked: list[dict[str, Any]] = []
    seen: set[str] = set()
    for hit in vector_hits:
        key = hit.get("chunk_id") or hit.get("text", "")[:80]
        if key in seen:
            continue
        seen.add(str(key))
        graph_score = _graph_score_for_chunk(hit, graph_lines) / max_graph
        score = 0.6 * float(hit.get("score", 0.0)) + 0.4 * graph_score
        item = dict(hit)
        item["hybrid_score"] = float(score)
        item["graph_weight"] = float(graph_score)
        reranked.append(item)

    reranked = sorted(reranked, key=lambda item: item["hybrid_score"], reverse=True)[:6]
    context_parts: list[str] = []
    token_budget = 1500
    tokens_used = 0
    for item in reranked:
        text = f"[Source: {item.get('paper_id', 'unknown')}] {item.get('text', '')}"
        n_tokens = len(text.split())
        if tokens_used + n_tokens > token_budget:
            remaining = max(token_budget - tokens_used, 0)
            if remaining <= 0:
                break
            text = " ".join(text.split()[:remaining])
            n_tokens = remaining
        context_parts.append(text)
        tokens_used += n_tokens
    return "\n\n".join(context_parts), reranked


def format_dataset_stats(input_dict: dict[str, Any]) -> str:
    """Return one-line dataset summary filtered by ReT and AgS."""
    if not DATASET_CSV.exists():
        return "Dataset summary unavailable: data/dataset.csv not found."
    df = pd.read_csv(DATASET_CSV)
    required = {"ReT", "AgS", "qe", "Ce", "pH"}
    if not required.issubset(df.columns):
        return "Dataset summary unavailable: dataset.csv is missing required columns."

    mask = (
        df["ReT"].astype(str).str.upper().eq(str(input_dict.get("ReT", "")).upper())
        & df["AgS"].astype(str).str.lower().eq(str(input_dict.get("AgS", "")).lower())
    )
    group = df.loc[mask].copy()
    if group.empty:
        return f"No matching dataset cases for {input_dict.get('ReT')} ({input_dict.get('AgS')})."
    for col in ["qe", "Ce", "pH"]:
        group[col] = pd.to_numeric(group[col], errors="coerce")
    return (
        f"Dataset stats for {input_dict.get('ReT')} ({input_dict.get('AgS')}): "
        f"n={len(group)}, qe={group['qe'].mean():.4g}±{group['qe'].std():.4g}, "
        f"Ce={group['Ce'].min():.4g}–{group['Ce'].max():.4g}, "
        f"pH={group['pH'].min():.4g}–{group['pH'].max():.4g}."
    )


def polymer_context(polymer: str, top_k: int = 5) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Return KG profile plus polymer-specific vector chunks."""
    return get_polymer_profile(polymer), retrieve_by_polymer(polymer, top_k=top_k)
