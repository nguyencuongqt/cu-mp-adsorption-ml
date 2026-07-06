"""Keyword-derived knowledge graph for Cu adsorption interpretation."""

from __future__ import annotations

import json
import pickle
import sys
from collections import Counter, defaultdict, deque
from pathlib import Path
from typing import Any

import networkx as nx
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from config import CHUNKS_FILE, DATASET_CSV, KNOWLEDGE_GRAPH_FILE, TARGET  # noqa: E402


def _load_chunks() -> list[dict[str, Any]]:
    if not CHUNKS_FILE.exists():
        return []
    with CHUNKS_FILE.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def _ce_bin(value: float) -> str:
    if value < 1:
        return "low"
    if value <= 50:
        return "mid"
    return "high"


def _ph_bin(value: float) -> str:
    if value < 6:
        return "acidic"
    if value <= 8:
        return "neutral"
    return "alkaline"


def _add_node(graph: nx.MultiDiGraph, node: str, kind: str, **attrs: Any) -> None:
    graph.add_node(node, kind=kind, **attrs)


def build_graph() -> nx.MultiDiGraph:
    """Build and save the keyword-derived knowledge graph."""
    graph = nx.MultiDiGraph()
    co_counts: Counter[tuple[str, str]] = Counter()

    for chunk in _load_chunks():
        meta = chunk.get("metadata", {})
        paper_id = str(chunk.get("paper_id", "unknown"))
        paper_node = f"Paper:{paper_id}"
        finding_node = f"Finding:{chunk.get('chunk_id', paper_id)}"
        _add_node(graph, paper_node, "Paper", paper_id=paper_id)
        _add_node(graph, finding_node, "Finding", text=chunk.get("text", "")[:500])
        graph.add_edge(paper_node, finding_node, label="studied_in", weight=1.0)

        polymers = [str(p).upper() for p in meta.get("polymers", [])]
        for polymer in polymers:
            p_node = f"Polymer:{polymer}"
            _add_node(graph, p_node, "Polymer", polymer=polymer)
            graph.add_edge(paper_node, p_node, label="studied_in", weight=1.0)
            graph.add_edge(p_node, finding_node, label="studied_in", weight=1.0)

        for mech in meta.get("mechanisms", []):
            m_node = f"Mechanism:{mech}"
            _add_node(graph, m_node, "Mechanism", mechanism=mech)
            graph.add_edge(m_node, finding_node, label="explains", weight=1.0)

        for left in polymers:
            for right in polymers:
                if left < right:
                    co_counts[(left, right)] += 1

    for (left, right), count in co_counts.items():
        graph.add_edge(f"Polymer:{left}", f"Polymer:{right}", label="co_studied_with", weight=float(count))
        graph.add_edge(f"Polymer:{right}", f"Polymer:{left}", label="co_studied_with", weight=float(count))

    if DATASET_CSV.exists():
        df = pd.read_csv(DATASET_CSV)
        needed = {"ReT", "AgS", "Ce", "pH", TARGET}
        if needed.issubset(df.columns):
            for col in ["Ce", "pH", TARGET]:
                df[col] = pd.to_numeric(df[col], errors="coerce")
            for (ret, ags, ce_b, ph_b), group in df.assign(
                Ce_bin=df["Ce"].map(lambda x: _ce_bin(x) if pd.notna(x) else "unknown"),
                pH_bin=df["pH"].map(lambda x: _ph_bin(x) if pd.notna(x) else "unknown"),
            ).groupby(["ReT", "AgS", "Ce_bin", "pH_bin"], dropna=False):
                finding = f"Finding:dataset:{ret}:{ags}:{ce_b}:{ph_b}"
                condition = f"Condition:{ags}:{ce_b}:{ph_b}"
                polymer = f"Polymer:{str(ret).upper()}"
                _add_node(graph, condition, "Condition", AgS=ags, Ce_bin=ce_b, pH_bin=ph_b)
                _add_node(graph, finding, "Finding", mean_qe=float(group[TARGET].mean()), n=int(len(group)))
                _add_node(graph, polymer, "Polymer", polymer=str(ret).upper())
                label = "positively_affects" if group[TARGET].mean() >= df[TARGET].mean() else "negatively_affects"
                graph.add_edge(condition, finding, label=label, weight=float(len(group)))
                graph.add_edge(polymer, finding, label="studied_in", weight=float(len(group)))

    KNOWLEDGE_GRAPH_FILE.parent.mkdir(parents=True, exist_ok=True)
    with KNOWLEDGE_GRAPH_FILE.open("wb") as handle:
        pickle.dump(graph, handle)
    print(f"Saved knowledge graph to {KNOWLEDGE_GRAPH_FILE}")
    return graph


def _load_graph() -> nx.MultiDiGraph:
    if not KNOWLEDGE_GRAPH_FILE.exists():
        return build_graph()
    with KNOWLEDGE_GRAPH_FILE.open("rb") as handle:
        return pickle.load(handle)


def graph_retrieve(query_dict: dict[str, Any], hops: int = 2) -> list[str]:
    """Return graph context strings sorted by edge weight."""
    graph = _load_graph()
    starts: list[str] = []
    polymer = query_dict.get("polymer") or query_dict.get("ReT")
    if polymer:
        starts.append(f"Polymer:{str(polymer).upper()}")
    ags = query_dict.get("AgS")
    ce_bin = query_dict.get("Ce_bin")
    ph_bin = query_dict.get("pH_bin")
    if ags and ce_bin and ph_bin:
        starts.append(f"Condition:{ags}:{ce_bin}:{ph_bin}")

    seen: set[str] = set()
    records: list[tuple[float, str]] = []
    queue = deque((node, 0) for node in starts if node in graph)
    while queue:
        node, depth = queue.popleft()
        if (node, depth) in seen or depth > hops:
            continue
        seen.add((node, depth))
        for _, nbr, data in graph.out_edges(node, data=True):
            weight = float(data.get("weight", 1.0))
            label = data.get("label", "related_to")
            records.append((weight, f"{node} -[{label}; weight={weight:g}]-> {nbr}"))
            if depth + 1 <= hops:
                queue.append((nbr, depth + 1))
    return [text for _, text in sorted(records, key=lambda item: item[0], reverse=True)]


def get_polymer_profile(polymer: str) -> dict[str, Any]:
    """Summarize paper, mechanism, condition, and co-study evidence for a polymer."""
    graph = _load_graph()
    node = f"Polymer:{str(polymer).upper()}"
    if node not in graph:
        return {"n_papers": 0, "mechanisms": [], "mean_qe_by_condition": {}, "co_studied_with": {}}

    papers: set[str] = set()
    mechanisms: set[str] = set()
    condition_qe: dict[str, list[float]] = defaultdict(list)
    co: Counter[str] = Counter()

    for pred in graph.predecessors(node):
        if graph.nodes[pred].get("kind") == "Paper":
            papers.add(graph.nodes[pred].get("paper_id", pred))
    for _, nbr, data in graph.out_edges(node, data=True):
        if data.get("label") == "co_studied_with":
            co[nbr.replace("Polymer:", "")] += int(data.get("weight", 1))
        if graph.nodes[nbr].get("kind") == "Finding":
            mean_qe = graph.nodes[nbr].get("mean_qe")
            if mean_qe is not None:
                condition_qe[nbr].append(float(mean_qe))
    for pred, _, data in graph.in_edges(node, data=True):
        if graph.nodes[pred].get("kind") == "Mechanism":
            mechanisms.add(graph.nodes[pred].get("mechanism", pred))

    return {
        "n_papers": len(papers),
        "mechanisms": sorted(mechanisms),
        "mean_qe_by_condition": {k: sum(v) / len(v) for k, v in condition_qe.items()},
        "co_studied_with": dict(co.most_common(10)),
    }


if __name__ == "__main__":
    build_graph()
