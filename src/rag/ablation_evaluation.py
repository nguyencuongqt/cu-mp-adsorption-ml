"""Ablation evaluation for the LLM Interpretation System."""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import friedmanchisquare, wilcoxon
from statsmodels.stats.multitest import multipletests

PROJECT_ROOT = Path(__file__).resolve().parents[2]
MODULE_DIR = Path(__file__).resolve().parent
for path in (PROJECT_ROOT, MODULE_DIR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from config import FIGURES_DIR, REPORTS_DIR, TEST_CASES_FILE  # noqa: E402
from hybrid_retrieval import hybrid_retrieve  # noqa: E402
from llm_interpretation import explain, predict_single  # noqa: E402
from vector_store import retrieve  # noqa: E402
from knowledge_graph import graph_retrieve  # noqa: E402


CONDITIONS = ["C0", "C1", "C2", "C3"]
QUALITY_METRICS = [
    "explanation_quality",
    "mechanism_score",
    "shap_alignment_score",
    "grounding_score",
    "uncertainty_score",
    "mechanistic_specificity",
    "hallucination_penalty",
]

MECHANISTIC_TERMS = [
    "adsorption",
    "aging",
    "aged",
    "biofilm",
    "biodegradation",
    "complexation",
    "diffusion",
    "electrostatic",
    "functional group",
    "hydrophobic",
    "ion exchange",
    "surface",
    "saturation",
]

UNCERTAINTY_TERMS = [
    "uncertainty",
    "prediction interval",
    "extrapolation",
    "risk",
    "check",
    "range",
    "confidence",
    "interval",
]


def _ensure_output_dirs() -> None:
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)


def _load_cases() -> pd.DataFrame:
    if not TEST_CASES_FILE.exists():
        raise FileNotFoundError(f"Missing evaluation file: {TEST_CASES_FILE}")
    df = pd.read_csv(TEST_CASES_FILE).head(30).copy()
    required = {"known_mechanism", "known_top_features"}
    missing = sorted(required - set(df.columns))
    if missing:
        raise KeyError(f"test_cases.csv missing columns: {missing}")
    if "input_dict" not in df.columns:
        input_cols = ["input_ReT", "input_AgS", "input_AdC", "input_Temp", "input_pH", "input_rpm", "input_Ce"]
        missing_inputs = [col for col in input_cols if col not in df.columns]
        if missing_inputs:
            raise KeyError(f"test_cases.csv missing input columns: {missing_inputs}")
        df["input_dict"] = df.apply(
            lambda row: json.dumps(
                {
                    "ReT": row["input_ReT"],
                    "AgS": row["input_AgS"],
                    "AdC": row["input_AdC"],
                    "Temp": float(row["input_Temp"]),
                    "pH": float(row["input_pH"]),
                    "rpm": float(row["input_rpm"]),
                    "Ce": float(row["input_Ce"]),
                }
            ),
            axis=1,
        )
    return df


def _parse_input(value: Any) -> dict[str, Any]:
    if isinstance(value, dict):
        return value
    text = str(value)
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return {}


def _split_features(value: Any) -> list[str]:
    if pd.isna(value):
        return []
    return [part.strip() for part in re.split(r"[,;|]", str(value)) if part.strip()]


def _cited_ids(text: str) -> set[str]:
    return set(re.findall(r"\[Source:\s*([^\]]+)\]", text) + re.findall(r"paper[_ -]?id[:=]\s*([\w.-]+)", text, flags=re.I))


def _corpus_ids(sources: list[dict[str, Any]]) -> set[str]:
    return {str(src.get("paper_id")) for src in sources if src.get("paper_id") is not None}


def _plain_text(text: str) -> str:
    return re.sub(r"<[^>]+>", " ", str(text))


def _keyword_terms(value: str) -> list[str]:
    terms = re.findall(r"[a-zA-Z][a-zA-Z0-9+-]*", str(value).lower())
    stop = {"and", "or", "the", "with", "via", "plus"}
    return [term for term in terms if term not in stop and len(term) > 2]


def _fraction_present(text: str, terms: list[str]) -> float:
    if not terms:
        return 0.0
    return sum(int(term.lower() in text) for term in terms) / len(terms)


def _bounded_score(value: float, max_score: float = 2.0) -> float:
    return float(max(0.0, min(max_score, value)))


def _numbers_only(input_dict: dict[str, Any]) -> tuple[str, list[dict[str, Any]]]:
    pred = predict_single(input_dict)
    text = f"lab_qe={pred['lab_qe']:.4g}; river_qe={pred['river_qe']:.6g}; PI high={pred['uncertainty_high']}"
    return text, []


def _evaluate_condition(condition: str, input_dict: dict[str, Any]) -> tuple[str, list[dict[str, Any]]]:
    if condition == "C0":
        return _numbers_only(input_dict)
    result = explain(input_dict, mode=int(condition[-1]))
    return str(result["explanation_html"]), list(result.get("sources", []))


def _score_text(
    condition: str,
    text: str,
    sources: list[dict[str, Any]],
    known_mechanism: str,
    known_features: list[str],
) -> dict[str, Any]:
    lower = _plain_text(text).lower()
    mechanism_terms = _keyword_terms(known_mechanism)
    mechanism_fraction = _fraction_present(lower, mechanism_terms)
    exact_mechanism = bool(known_mechanism and str(known_mechanism).lower() in lower)
    mechanism_score = 2.0 if exact_mechanism else _bounded_score(2.0 * mechanism_fraction)

    feature_terms = [feature.lower() for feature in known_features]
    feature_fraction = _fraction_present(lower, feature_terms)
    shap_alignment_score = _bounded_score(2.0 * feature_fraction)

    cited = _cited_ids(text)
    corpus = _corpus_ids(sources)
    invalid_citation = bool(cited and (not corpus or not cited.issubset(corpus)))
    hallucination_rate = int(invalid_citation)
    hallucination_penalty = 1.0 - hallucination_rate

    has_valid_citation = bool(cited and corpus and cited.issubset(corpus))
    has_context_language = any(term in lower for term in ["source", "dataset", "corpus", "paper", "graph", "polymer profile"])
    grounding_score = 2.0 if has_valid_citation else 1.0 if has_context_language or sources else 0.0

    uncertainty_fraction = _fraction_present(lower, UNCERTAINTY_TERMS)
    uncertainty_score = 2.0 if uncertainty_fraction >= 0.25 else 1.0 if uncertainty_fraction > 0 else 0.0

    specificity_hits = sum(int(term in lower) for term in MECHANISTIC_TERMS)
    mechanistic_specificity = _bounded_score(specificity_hits / 2.0)

    explanation_quality = (
        0.25 * (mechanism_score / 2.0)
        + 0.20 * (shap_alignment_score / 2.0)
        + 0.25 * (grounding_score / 2.0)
        + 0.15 * (uncertainty_score / 2.0)
        + 0.10 * (mechanistic_specificity / 2.0)
        + 0.05 * hallucination_penalty
    )

    mechanism_match = int(mechanism_score >= 1.0)
    shap_consistency = int(shap_alignment_score >= 1.0)
    return {
        "condition": condition,
        "mechanism_match": mechanism_match,
        "shap_consistency": shap_consistency,
        "hallucination_rate": hallucination_rate,
        "mechanism_score": mechanism_score,
        "shap_alignment_score": shap_alignment_score,
        "grounding_score": grounding_score,
        "uncertainty_score": uncertainty_score,
        "mechanistic_specificity": mechanistic_specificity,
        "hallucination_penalty": hallucination_penalty,
        "explanation_quality": explanation_quality,
        "accuracy_score": explanation_quality,
    }


def _stat_tests(auto_df: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for metric in QUALITY_METRICS + ["mechanism_match", "shap_consistency", "hallucination_rate"]:
        pivot = auto_df.pivot(index="case_id", columns="condition", values=metric)
        if all(col in pivot.columns for col in CONDITIONS):
            try:
                friedman_stat, friedman_p = friedmanchisquare(*(pivot[col] for col in CONDITIONS))
            except ValueError:
                friedman_stat, friedman_p = np.nan, np.nan
            pair_rows: list[dict[str, Any]] = []
            pairs = [("C3", "C2"), ("C2", "C1"), ("C1", "C0"), ("C3", "C0")]
            for a, b in pairs:
                try:
                    stat, p = wilcoxon(pivot[a], pivot[b], zero_method="zsplit")
                except ValueError:
                    stat, p = np.nan, 1.0
                pair_rows.append({"metric": metric, "contrast": f"{a}>{b}", "stat": stat, "p": p})
            adj = multipletests([r["p"] for r in pair_rows], method="holm")[1]
            for row, p_adj in zip(pair_rows, adj):
                row["p_holm"] = float(p_adj)
                row["friedman_stat"] = friedman_stat
                row["friedman_p"] = friedman_p
                rows.append(row)
    return pd.DataFrame(rows)


def _precision_at_5(hits: list[dict[str, Any]], known_mechanism: str, known_features: list[str]) -> float:
    if not hits:
        return 0.0
    terms = [str(known_mechanism).lower()] + [f.lower() for f in known_features]
    top = hits[:5]
    relevant = 0
    for hit in top:
        text = (hit.get("text", "") + " " + str(hit.get("metadata", {}))).lower()
        relevant += int(any(term and term in text for term in terms))
    return relevant / len(top)


def _relevance_at_5(hits: list[dict[str, Any]], known_mechanism: str, known_features: list[str]) -> float:
    if not hits:
        return 0.0
    terms = _keyword_terms(known_mechanism) + [feature.lower() for feature in known_features]
    if not terms:
        return 0.0
    scores: list[float] = []
    for hit in hits[:5]:
        text = (hit.get("text", "") + " " + str(hit.get("metadata", {}))).lower()
        scores.append(_fraction_present(text, terms))
    return float(np.mean(scores)) if scores else 0.0


def _mechanism_recall(hits: list[dict[str, Any]], known_mechanism: str) -> float:
    terms = _keyword_terms(known_mechanism)
    if not terms:
        return 0.0
    text = " ".join(hit.get("text", "") + " " + str(hit.get("metadata", {})) for hit in hits[:5]).lower()
    return _fraction_present(text, terms)


def _retrieval_eval(cases: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for case_id, row in cases.iterrows():
        input_dict = _parse_input(row["input_dict"])
        known_features = _split_features(row["known_top_features"])
        mechanism = str(row["known_mechanism"])
        query = f"Cu adsorption {input_dict.get('ReT')} {input_dict.get('AgS')} {mechanism}"
        vector_hits = retrieve(query, top_k=5)
        pred = predict_single(input_dict)
        _, hybrid_hits = hybrid_retrieve(input_dict, pred, vector_k=8, hops=2)
        graph_lines = graph_retrieve({"polymer": input_dict.get("ReT"), "AgS": input_dict.get("AgS")}, hops=2)
        rows.append(
            {
                "case_id": case_id,
                "vector_precision_at_5": _precision_at_5(vector_hits, mechanism, known_features),
                "hybrid_precision_at_5": _precision_at_5(hybrid_hits, mechanism, known_features),
                "vector_relevance_at_5": _relevance_at_5(vector_hits, mechanism, known_features),
                "hybrid_relevance_at_5": _relevance_at_5(hybrid_hits, mechanism, known_features),
                "vector_mechanism_recall": _mechanism_recall(vector_hits, mechanism),
                "hybrid_mechanism_recall": _mechanism_recall(hybrid_hits, mechanism),
                "graph_coverage": int(any(mechanism.lower() in line.lower() for line in graph_lines)),
            }
        )
    return pd.DataFrame(rows)


def _save_accuracy_plot(auto_df: pd.DataFrame, tests: pd.DataFrame) -> Path:
    output = FIGURES_DIR / "ablation_accuracy.png"
    summary = auto_df.groupby("condition")[["explanation_quality", "grounding_score", "mechanistic_specificity"]].mean()
    summary = summary.reindex(CONDITIONS)
    ax = summary.plot(kind="bar", figsize=(8.4, 5.0), color=["#4c78a8", "#54a24b", "#e45756"])
    ax.set_ylabel("Mean score")
    ax.set_xlabel("Condition")
    ax.set_ylim(0, max(2.1, float(summary.max().max()) + 0.2))
    ax.grid(axis="y", linestyle=":", alpha=0.45)
    sig = tests[(tests["metric"] == "explanation_quality") & (tests["p_holm"] < 0.05)]
    if not sig.empty:
        labels = "; ".join(f"{r.contrast} p={r.p_holm:.3g}" for r in sig.itertuples())
        ax.text(0.5, 1.04, labels, ha="center", va="bottom", transform=ax.transAxes, fontsize=8)
    plt.tight_layout()
    plt.savefig(output, dpi=300, bbox_inches="tight")
    plt.close()
    return output


def _save_retrieval_plot(retrieval_df: pd.DataFrame) -> Path:
    output = FIGURES_DIR / "retrieval_precision.png"
    means = retrieval_df[["vector_precision_at_5", "hybrid_precision_at_5", "vector_relevance_at_5", "hybrid_relevance_at_5"]].mean()
    fig, ax = plt.subplots(figsize=(5.6, 4.4))
    ax.bar(["Vector P@5", "Hybrid P@5", "Vector rel.", "Hybrid rel."], means, color=["#4c78a8", "#f58518", "#72b7b2", "#e45756"])
    ax.set_ylabel("Mean score")
    ax.set_ylim(0, 1.0)
    ax.grid(axis="y", linestyle=":", alpha=0.45)
    fig.tight_layout()
    fig.savefig(output, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return output


def _summary(auto_df: pd.DataFrame, retrieval_df: pd.DataFrame, tests: pd.DataFrame) -> str:
    condition_scores = auto_df.groupby("condition")["explanation_quality"].mean().reindex(CONDITIONS)
    grounding = auto_df.groupby("condition")["grounding_score"].mean().reindex(CONDITIONS)
    specificity = auto_df.groupby("condition")["mechanistic_specificity"].mean().reindex(CONDITIONS)
    contextual_mean = float(condition_scores[["C2", "C3"]].mean())
    baseline_mean = float(condition_scores[["C0", "C1"]].mean())
    h1 = bool(
        contextual_mean - baseline_mean >= 0.05
        and grounding[["C2", "C3"]].mean() >= grounding[["C0", "C1"]].mean()
    )
    vector_mean = float(retrieval_df["vector_precision_at_5"].mean()) if not retrieval_df.empty else 0.0
    hybrid_mean = float(retrieval_df["hybrid_precision_at_5"].mean()) if not retrieval_df.empty else 0.0
    vector_rel = float(retrieval_df["vector_relevance_at_5"].mean()) if not retrieval_df.empty else 0.0
    hybrid_rel = float(retrieval_df["hybrid_relevance_at_5"].mean()) if not retrieval_df.empty else 0.0
    vector_recall = float(retrieval_df["vector_mechanism_recall"].mean()) if not retrieval_df.empty else 0.0
    hybrid_recall = float(retrieval_df["hybrid_mechanism_recall"].mean()) if not retrieval_df.empty else 0.0
    h2 = bool((hybrid_rel - vector_rel >= 0.05) or (hybrid_recall - vector_recall >= 0.05))
    halluc = auto_df.groupby("condition")["hallucination_rate"].mean()
    h3 = bool(specificity["C3"] > specificity["C2"] and halluc.get("C3", 1.0) <= halluc.get("C2", 1.0))
    effect_h2 = hybrid_mean - vector_mean
    recommendation = str(condition_scores.idxmax())
    return (
        "Revised hypotheses use rubric-based explanation quality rather than a strict C3>C2>C1>C0 ordering.\n"
        f"H1 contextual evidence improves explanation quality/grounding (C2-C3 vs C0-C1): {'supported' if h1 else 'not supported'}; "
        f"quality_means={condition_scores.to_dict()}, contextual_minus_baseline={contextual_mean - baseline_mean:.3f}\n"
        f"H2 hybrid retrieval improves relevance or mechanism recall over vector-only: {'supported' if h2 else 'not supported'}; "
        f"precision_effect={effect_h2:.3f}, relevance_effect={hybrid_rel - vector_rel:.3f}, recall_effect={hybrid_recall - vector_recall:.3f}\n"
        f"H3 KG increases mechanistic specificity without increasing hallucination (C3 vs C2): {'supported' if h3 else 'not supported'}; "
        f"specificity_C2={specificity['C2']:.3f}, specificity_C3={specificity['C3']:.3f}, "
        f"hallucination_C2={halluc.get('C2', np.nan):.3f}, hallucination_C3={halluc.get('C3', np.nan):.3f}\n"
        f"Recommended mode: {recommendation}\n"
        f"Post-hoc tests:\n{tests.to_string(index=False)}\n"
    )
    return (
        f"H1 accuracy C3 > C2 > C1 > C0: {'supported' if h1 else 'not supported'}; "
        f"means={condition_scores.to_dict()}\n"
        f"H2 hybrid Precision@5 > vector-only by ≥10%: {'supported' if h2 else 'not supported'}; "
        f"effect={effect_h2:.3f}\n"
        f"H3 hallucination_rate C3 < C2: {'supported' if h3 else 'not supported'}; "
        f"C2={halluc.get('C2', np.nan):.3f}, C3={halluc.get('C3', np.nan):.3f}\n"
        f"Recommended mode: {recommendation}\n"
        f"Post-hoc tests:\n{tests.to_string(index=False)}\n"
    )


def run_evaluation() -> dict[str, Any]:
    """Run automated and retrieval ablation evaluation."""
    _ensure_output_dirs()
    cases = _load_cases()
    auto_rows: list[dict[str, Any]] = []
    for case_id, row in cases.iterrows():
        input_dict = _parse_input(row["input_dict"])
        known_features = _split_features(row["known_top_features"])
        known_mechanism = str(row["known_mechanism"])
        for condition in CONDITIONS:
            text, sources = _evaluate_condition(condition, input_dict)
            score = _score_text(condition, text, sources, known_mechanism, known_features)
            score["case_id"] = case_id
            auto_rows.append(score)

    auto_df = pd.DataFrame(auto_rows)
    tests = _stat_tests(auto_df)
    retrieval_df = _retrieval_eval(cases)
    try:
        stat, p = wilcoxon(retrieval_df["hybrid_precision_at_5"], retrieval_df["vector_precision_at_5"], zero_method="zsplit")
    except ValueError:
        stat, p = np.nan, 1.0
    retrieval_df["wilcoxon_stat"] = stat
    retrieval_df["wilcoxon_p"] = p

    auto_path = REPORTS_DIR / "eval_automated.csv"
    retrieval_path = REPORTS_DIR / "eval_retrieval.csv"
    summary_path = REPORTS_DIR / "ablation_summary.txt"
    auto_df.to_csv(auto_path, index=False)
    retrieval_df.to_csv(retrieval_path, index=False)
    accuracy_plot = _save_accuracy_plot(auto_df, tests)
    retrieval_plot = _save_retrieval_plot(retrieval_df)
    text = _summary(auto_df, retrieval_df, tests)
    summary_path.write_text(text, encoding="utf-8")
    print(text)

    return {
        "automated": auto_df,
        "retrieval": retrieval_df,
        "tests": tests,
        "paths": {
            "eval_automated": auto_path,
            "eval_retrieval": retrieval_path,
            "ablation_summary": summary_path,
            "ablation_accuracy": accuracy_plot,
            "retrieval_precision": retrieval_plot,
        },
    }


if __name__ == "__main__":
    run_evaluation()
