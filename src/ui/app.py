"""Streamlit UI for the Cu2+-MP Predictor.

Run with:
    streamlit run src/ui/app.py
"""

from __future__ import annotations

import importlib.util
import io
import os
import sys
import hashlib
import html
import re
from pathlib import Path
from typing import Any

import joblib
import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt

import networkx as nx
import numpy as np
import pandas as pd
import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parents[2]
RAG_DIR = PROJECT_ROOT / "src" / "rag"
for path in (PROJECT_ROOT, RAG_DIR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import config  # noqa: E402
from llm_provider import resolve_provider  # noqa: E402


st.set_page_config(
    page_title="Cu²⁺-MP Predictor",
    layout="wide",
    page_icon="\U0001f9ea",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    /* Metric cards */
    [data-testid="metric-container"] {
        background: #f8fafc;
        border: 1px solid #e2e8f0;
        border-radius: 8px;
        padding: 12px 16px;
    }
    /* Tab styling */
    [data-testid="stTab"] { font-weight: 600; }
    /* Section header accent */
    .section-header {
        border-left: 4px solid #4c78a8;
        padding-left: 10px;
        margin-bottom: 8px;
    }
    /* Badge row in sidebar */
    .badge-ok   { color: #16803c; font-weight: 700; }
    .badge-fail { color: #b42318; font-weight: 700; }
    /* Input summary card */
    .input-card {
        background: #f0f7ff;
        border-radius: 8px;
        padding: 14px 18px;
        font-size: 0.92rem;
        line-height: 1.7;
    }
</style>
""", unsafe_allow_html=True)


POLYMER_OPTIONS = ["PP", "PE", "PET", "PS", "PVC", "PLA", "PBAT", "TPU", "PA6", "HDPE", "LDPE"]
CE50_MG_L = 29.5

# Human-readable labels for parameters
PARAM_LABELS = {
    "ReT": "Resin Type (Polymer)",
    "AgS": "Ageing Status",
    "AdC": "Adsorption Condition",
    "Temp": "Temperature (°C)",
    "pH": "pH",
    "rpm": "Mixing Speed (rpm)",
    "Ce": "Equilibrium Cu²⁺ conc. Ce (mg/L)",
}

PARAM_HELP = {
    "ReT": "Microplastic polymer type used in the adsorption experiment.",
    "AgS": "Whether the MP is virgin (as-produced) or aged (environmentally weathered).",
    "AdC": "Single-ion: only Cu²⁺ present. Mixed: multiple metal ions compete.",
    "Temp": "Experimental temperature in °C. Training range: 5–60 °C.",
    "pH": "Solution pH. Warning: 46% of training data had missing pH (imputed).",
    "rpm": "Agitation speed in revolutions per minute. Training range: 50–250 rpm.",
    "Ce": "Equilibrium dissolved Cu²⁺ concentration. CE₅₀ reference = 29.5 mg/L.",
}


def _load_module(name: str, path: Path) -> Any:
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load {name} from {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@st.cache_resource
def get_interpretation_module() -> Any:
    return _load_module("llm_interpretation_ui", RAG_DIR / "llm_interpretation.py")


@st.cache_resource
def get_hybrid_module() -> Any:
    return _load_module("hybrid_retrieval_ui", RAG_DIR / "hybrid_retrieval.py")


@st.cache_resource
def get_kg_module() -> Any:
    return _load_module("knowledge_graph_ui", RAG_DIR / "knowledge_graph.py")


@st.cache_resource
def load_qrf_model() -> Any | None:
    if config.QRF_MODEL_FILE.exists():
        return joblib.load(config.QRF_MODEL_FILE)
    return None


@st.cache_resource
def load_rf_model() -> Any | None:
    for name in ["RandomForest_tuned.joblib", "rf_tuned.joblib", "best_rf_model.joblib"]:
        path = config.MODEL_DIR / name
        if path.exists():
            return joblib.load(path)
    return None


@st.cache_resource
def load_kg() -> nx.MultiDiGraph | None:
    if not config.KNOWLEDGE_GRAPH_FILE.exists():
        return None
    return get_kg_module()._load_graph()


@st.cache_resource
def load_eval_data() -> tuple[pd.DataFrame, pd.DataFrame]:
    auto_path = config.REPORTS_DIR / "eval_automated.csv"
    retrieval_path = config.REPORTS_DIR / "eval_retrieval.csv"
    auto = pd.read_csv(auto_path) if auto_path.exists() else pd.DataFrame()
    retrieval = pd.read_csv(retrieval_path) if retrieval_path.exists() else pd.DataFrame()
    return auto, retrieval


def load_judge_data() -> pd.DataFrame:
    path = config.REPORTS_DIR / "llm_judge_scores.csv"
    return pd.read_csv(path) if path.exists() else pd.DataFrame()


@st.cache_data(show_spinner=False)
def load_paper_metadata() -> dict[str, dict[str, str]]:
    path = config.PAPER_METADATA_FILE
    if not path.exists():
        return {}
    df = pd.read_csv(path).fillna("")
    metadata: dict[str, dict[str, str]] = {}
    for row in df.to_dict(orient="records"):
        paper_id = str(row.get("paper_id", "")).strip()
        if not paper_id:
            continue
        metadata[paper_id] = {key: str(value).strip() for key, value in row.items() if str(value).strip()}
    return metadata


def _strip_code_fences(text: str) -> str:
    value = str(text).strip()
    if value.startswith("```"):
        value = re.sub(r"^```[a-zA-Z0-9_-]*\s*", "", value)
        value = re.sub(r"\s*```$", "", value)
    return value.strip()


def _render_explanation_text(explanation_html: str) -> str:
    text = _strip_code_fences(explanation_html)
    text = re.sub(r"(?is)<\s*br\s*/?\s*>", "\n", text)
    text = re.sub(r"(?is)</\s*p\s*>", "\n\n", text)
    text = re.sub(r"(?is)<\s*p[^>]*>", "", text)
    text = re.sub(r"(?is)</\s*li\s*>", "\n", text)
    text = re.sub(r"(?is)<\s*li[^>]*>", "- ", text)
    text = re.sub(r"(?is)</\s*(ul|ol|section|div|article)\s*>", "\n", text)
    text = re.sub(r"(?is)<\s*(ul|ol|section|div|article)[^>]*>", "\n", text)
    text = re.sub(r"(?is)</\s*h[1-6]\s*>", "\n\n", text)
    text = re.sub(r"(?is)<\s*h[1-6][^>]*>", "**", text)
    text = re.sub(r"(?is)</\s*(strong|b)\s*>", "**", text)
    text = re.sub(r"(?is)<\s*(strong|b)[^>]*>", "**", text)
    text = re.sub(r"(?is)</\s*(em|i)\s*>", "*", text)
    text = re.sub(r"(?is)<\s*(em|i)[^>]*>", "*", text)
    text = re.sub(r"(?is)<[^>]+>", "", text)
    text = html.unescape(text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def _citation_from_paper_id(paper_id: str) -> str:
    match = re.match(r"([A-Za-z]+)_?(\d{4})", str(paper_id))
    if match:
        surname, year = match.groups()
        return f"{surname} et al. ({year})"
    return str(paper_id).replace("_", " ").strip()


def _doi_from_source(src: dict[str, Any]) -> str | None:
    paper_id = str(src.get("paper_id", "")).strip()
    if paper_id:
        metadata = load_paper_metadata().get(paper_id, {})
        doi = metadata.get("doi")
        if doi:
            return doi
    for candidate in (src.get("doi"), src.get("DOI")):
        if candidate:
            return str(candidate).strip()
    text = str(src.get("text", ""))
    match = re.search(r"(10\.\d{4,9}/[-._;()/:A-Z0-9]+)", text, flags=re.I)
    if match:
        return match.group(1).rstrip(".,;)")
    return None


def _format_source_label(src: dict[str, Any]) -> str:
    citation = _citation_from_paper_id(str(src.get("paper_id", "unknown")))
    doi = _doi_from_source(src)
    return f"{citation} | DOI: {doi}" if doi else citation


def _unique_source_labels(sources: list[dict[str, Any]]) -> list[str]:
    labels: list[str] = []
    seen: set[str] = set()
    for src in sources:
        label = _format_source_label(src)
        if label in seen:
            continue
        seen.add(label)
        labels.append(label)
    return labels


def openai_api_available() -> bool:
    provider, api_key = resolve_provider()
    return provider == "openai" and bool(api_key)


def openai_api_fingerprint() -> str:
    _, api_key = resolve_provider()
    if not api_key:
        return "missing"
    return hashlib.sha256(api_key.encode("utf-8")).hexdigest()[:12]


def status_badge(label: str, ok: bool) -> None:
    icon = "✓" if ok else "✗"
    color_class = "badge-ok" if ok else "badge-fail"
    st.sidebar.markdown(
        f"<span class='{color_class}'>{icon}</span>&nbsp;{label}",
        unsafe_allow_html=True,
    )


def normalize_input(raw: dict[str, Any]) -> dict[str, Any]:
    ret_map = {"PA6": "PA", "HDPE": "HPE", "LDPE": "LPE"}
    ags_map = {"virgin": "Virgin", "aged": "Aged"}
    adc_map = {"single": "Single-ion", "mixed": "Mixed"}
    values = dict(raw)
    values["ReT"] = ret_map.get(str(values["ReT"]), str(values["ReT"]))
    values["AgS"] = ags_map.get(str(values["AgS"]).lower(), values["AgS"])
    values["AdC"] = adc_map.get(str(values["AdC"]).lower(), values["AdC"])
    return values


def domain_warnings(input_dict: dict[str, Any]) -> list[str]:
    warnings: list[str] = []
    values = normalize_input(input_dict)
    for col in config.NUM_COLS:
        low, high = config.DOMAIN[col]
        if not (low <= float(values[col]) <= high):
            warnings.append(f"{col} = {values[col]} is outside the training domain [{low}, {high}].")
    for col in config.CAT_COLS:
        if values[col] not in config.DOMAIN[col]:
            warnings.append(f"{col} = {values[col]} is outside the training domain.")
    return warnings


def predict_with_model(input_dict: dict[str, Any], faf: float) -> dict[str, Any]:
    qrf = load_qrf_model()
    values = normalize_input(input_dict)
    if qrf is None:
        raise FileNotFoundError("QRF model missing. Run src/06_uncertainty_qrf.py first.")
    x = pd.DataFrame([{col: values[col] for col in config.CAT_COLS + config.NUM_COLS}])
    pred = np.asarray(qrf.predict(x, quantiles=[0.05, 0.50, 0.95]))
    if pred.ndim == 2:
        lab_low, lab_med, lab_high = [float(v) for v in pred[0]]
    else:
        lab_low = lab_med = lab_high = float(pred[0])
    return {
        "lab_qe": lab_med,
        "lab_qe_lower": lab_low,
        "lab_qe_upper": lab_high,
        "river_qe": lab_med * faf,
        "river_qe_lower": lab_low * faf,
        "river_qe_upper": lab_high * faf,
        "shap_top3": [],
        "uncertainty_high": bool((lab_high - lab_low) > 2),
    }


def run_predict_single(input_dict: dict[str, Any], faf: float) -> dict[str, Any]:
    try:
        mod = get_interpretation_module()
        pred = mod.predict_single(normalize_input(input_dict))
        pred["river_qe"] = pred["lab_qe"] * faf
        pred["river_qe_lower"] = pred["lab_qe_lower"] * faf
        pred["river_qe_upper"] = pred["lab_qe_upper"] * faf
        return pred
    except Exception:
        return predict_with_model(input_dict, faf)


def _input_cache_key(input_dict: dict[str, Any]) -> tuple[tuple[str, str], ...]:
    values = normalize_input(input_dict)
    keys = config.CAT_COLS + config.NUM_COLS
    return tuple((col, str(values[col])) for col in keys)


@st.cache_data(show_spinner=False)
def cached_explanation(input_key: tuple[tuple[str, str], ...], mode: int, api_fingerprint: str) -> dict[str, Any]:
    values: dict[str, Any] = dict(input_key)
    for col in config.NUM_COLS:
        values[col] = float(values[col])
    return get_interpretation_module().explain(values, mode=mode)


def prediction_bar(pred: dict[str, Any]) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(6.2, 3.8))
    labels = ["Lab qe", "River qe"]
    mids = [pred["lab_qe"], pred["river_qe"]]
    lows = [pred["lab_qe_lower"], pred["river_qe_lower"]]
    highs = [pred["lab_qe_upper"], pred["river_qe_upper"]]
    yerr = np.vstack([np.array(mids) - np.array(lows), np.array(highs) - np.array(mids)])
    ax.bar(labels, mids, yerr=yerr, capsize=8, color=["#4c78a8", "#54a24b"], edgecolor="black", linewidth=0.6)
    ax.axhline(CE50_MG_L, color="#e45756", linestyle="--", linewidth=1.0, label="CE₅₀ reference")
    ax.set_ylabel("qe (mg/g)")
    ax.legend(frameon=False)
    ax.grid(axis="y", linestyle=":", alpha=0.45)
    fig.tight_layout()
    return fig


def shap_waterfall(pred: dict[str, Any]) -> tuple[plt.Figure, pd.DataFrame]:
    top = pred.get("shap_top3", []) or []
    if not top:
        table = pd.DataFrame(columns=["feature", "value", "SHAP", "direction"])
        fig, ax = plt.subplots(figsize=(6.0, 2.8))
        ax.text(0.5, 0.5, "SHAP unavailable until compatible RF/QRF artifact is loaded.", ha="center", va="center")
        ax.axis("off")
        return fig, table
    df = pd.DataFrame(top, columns=["feature", "SHAP", "direction"])
    df["value"] = ""
    colors = ["#e45756" if v > 0 else "#4c78a8" for v in df["SHAP"]]
    fig, ax = plt.subplots(figsize=(7.0, 3.8))
    ax.barh(df["feature"], df["SHAP"], color=colors)
    ax.axvline(0, color="black", linewidth=0.8)
    ax.set_xlabel("Local SHAP contribution")
    ax.invert_yaxis()
    fig.tight_layout()
    return fig, df[["feature", "value", "SHAP", "direction"]]


def ce_position_plot(input_dict: dict[str, Any]) -> plt.Figure:
    ce = float(input_dict["Ce"])
    fig, ax = plt.subplots(figsize=(6.2, 3.2))
    x = np.linspace(0.001, 200, 200)
    y = 1 / (1 + np.exp(-(np.log(x) - np.log(CE50_MG_L))))
    ax.plot(x, y, color="#4c78a8")
    ax.axvline(CE50_MG_L, color="#e45756", linestyle="--", label="CE₅₀=29.5 mg/L")
    ax.axvline(ce, color="black", linewidth=1.2, label="Input Ce")
    ax.set_xscale("log")
    ax.set_xlabel("Ce (mg/L)")
    ax.set_ylabel("Relative positive SHAP tendency")
    ax.legend(frameon=False)
    fig.tight_layout()
    return fig


def fig_to_png_bytes(fig: plt.Figure) -> bytes:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=300, bbox_inches="tight")
    return buf.getvalue()


def render_graph(polymer: str) -> tuple[plt.Figure, list[str]]:
    kg = load_kg()
    if kg is None:
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.text(0.5, 0.5, "Knowledge graph missing. Run knowledge_graph.py first.", ha="center", va="center")
        ax.axis("off")
        return fig, []
    center = f"Polymer:{polymer}"
    if center not in kg:
        center = f"Polymer:{normalize_input({'ReT': polymer, 'AgS': 'aged', 'AdC': 'mixed', 'Temp': 25, 'pH': 7, 'rpm': 160, 'Ce': 0.01})['ReT']}"
    if center not in kg:
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.text(0.5, 0.5, f"No KG node for {polymer}.", ha="center", va="center")
        ax.axis("off")
        return fig, []
    nodes = {center}
    for node in list(nodes):
        nodes.update(nx.single_source_shortest_path_length(kg, node, cutoff=2).keys())
    sub = kg.subgraph(nodes).copy()
    kind_colors = {"Polymer": "#4c78a8", "Condition": "#f58518", "Mechanism": "#54a24b", "Paper": "#b279a2", "Finding": "#bab0ac"}
    colors = [kind_colors.get(sub.nodes[n].get("kind"), "#999999") for n in sub.nodes]
    fig, ax = plt.subplots(figsize=(8, 5.5))
    pos = nx.spring_layout(sub, seed=42, k=0.8)
    nx.draw_networkx_nodes(sub, pos, node_color=colors, node_size=520, ax=ax, alpha=0.9)
    nx.draw_networkx_labels(sub, pos, labels={n: n.split(":", 1)[-1][:18] for n in sub.nodes}, font_size=7, ax=ax)
    nx.draw_networkx_edges(sub, pos, arrows=True, alpha=0.35, ax=ax)
    edge_labels = {(u, v): d.get("label", "") for u, v, d in sub.edges(data=True)}
    nx.draw_networkx_edge_labels(sub, pos, edge_labels=edge_labels, font_size=6, ax=ax)
    ax.axis("off")
    fig.tight_layout()
    lines = get_kg_module().graph_retrieve({"polymer": polymer}, hops=2)
    return fig, lines


# ── Sidebar ───────────────────────────────────────────────────────────────────

def render_sidebar() -> tuple[dict[str, Any], float, str, bool, Any]:
    st.sidebar.title("\U0001f9ea Cu²⁺-MP Predictor")
    st.sidebar.caption("Quantile Random Forest + RAG interpretation system")
    st.sidebar.divider()

    # ── Component status ──────────────────────────────────────────────────────
    rf = load_rf_model()
    qrf = load_qrf_model()
    kg = load_kg()
    api_ok = openai_api_available()
    preprocessor_ok = bool(qrf is not None and hasattr(qrf, "named_steps") and "preprocess" in qrf.named_steps)

    with st.sidebar.expander("\U0001f4cb Model & component status", expanded=False):
        status_badge("Preprocessor", preprocessor_ok)
        status_badge("Random Forest (RF)", rf is not None)
        status_badge("Quantile RF (QRF)", qrf is not None)
        status_badge("FAISS index", config.FAISS_INDEX_FILE.exists() and config.CHUNKS_METADATA_FILE.exists())
        status_badge("Knowledge Graph", kg is not None)
        status_badge("API key", api_ok)

    st.sidebar.subheader("Experimental inputs")

    # ── Categorical inputs ────────────────────────────────────────────────────
    ret_val = st.sidebar.selectbox(
        PARAM_LABELS["ReT"],
        POLYMER_OPTIONS,
        index=0,
        help=PARAM_HELP["ReT"],
    )
    ags_val = st.sidebar.radio(
        PARAM_LABELS["AgS"],
        ["virgin", "aged"],
        horizontal=True,
        help=PARAM_HELP["AgS"],
    )
    adc_val = st.sidebar.radio(
        PARAM_LABELS["AdC"],
        ["single", "mixed"],
        horizontal=True,
        help=PARAM_HELP["AdC"],
    )

    st.sidebar.subheader("Reaction conditions")

    # ── Numeric inputs ────────────────────────────────────────────────────────
    temp_val = st.sidebar.slider(
        PARAM_LABELS["Temp"],
        5, 60, 25,
        help=PARAM_HELP["Temp"],
    )
    ph_val = st.sidebar.slider(
        PARAM_LABELS["pH"],
        3.0, 11.0, 7.0, 0.1,
        help=PARAM_HELP["pH"],
    )
    st.sidebar.caption(
        "⚠️ pH training coverage: 46% missing (imputed). "
        "Predictions in pH 3–5 and pH 9–11 carry higher uncertainty."
    )
    rpm_val = st.sidebar.slider(
        PARAM_LABELS["rpm"],
        50, 250, 160,
        help=PARAM_HELP["rpm"],
    )
    ce_val = st.sidebar.number_input(
        PARAM_LABELS["Ce"],
        min_value=0.001,
        max_value=500.0,
        value=0.00877,
        format="%.5f",
        help=PARAM_HELP["Ce"],
    )

    ce_badge = "LOW" if ce_val < CE50_MG_L else "MID" if ce_val <= 200 else "HIGH"
    ce_color = {"LOW": "#16803c", "MID": "#b45309", "HIGH": "#b42318"}[ce_badge]
    st.sidebar.markdown(
        f"Ce vs CE₅₀ (29.5 mg/L): "
        f"<span style='color:{ce_color};font-weight:700'>{ce_badge}</span>",
        unsafe_allow_html=True,
    )

    st.sidebar.divider()
    st.sidebar.subheader("Field adjustment")
    faf = st.sidebar.slider(
        "Field Adjustment Factor (FAF)",
        0.003, 0.048, 0.012, 0.001,
        format="%.3f",
        help=(
            "Scales lab qe to environmental (river) conditions. "
            "Central estimate = 0.012; plausible range 0.003–0.048."
        ),
    )
    st.sidebar.caption(f"Selected: FAF = **{faf:.3f}**  →  river qe = lab qe × {faf:.3f}")

    st.sidebar.divider()
    batch = st.sidebar.toggle("Batch mode (CSV/XLSX upload)", value=False)

    input_dict = {
        "ReT": ret_val,
        "AgS": ags_val,
        "AdC": adc_val,
        "Temp": temp_val,
        "pH": ph_val,
        "rpm": rpm_val,
        "Ce": ce_val,
    }
    return input_dict, faf, ce_badge, batch, qrf


# ── Panel A – input summary ───────────────────────────────────────────────────

def panel_input_summary(input_dict: dict[str, Any], faf: float, batch: bool = False) -> None:
    st.markdown("<div class='section-header'><h3>Panel A — Input Summary</h3></div>", unsafe_allow_html=True)
    batch_out: pd.DataFrame | None = st.session_state.get("batch_results")

    if batch and batch_out is not None and not batch_out.empty:
        saved_faf = st.session_state.get("batch_faf", faf)
        st.success(f"Batch mode — {len(batch_out)} rows processed · FAF = {saved_faf:.3f}")
        col1, col2 = st.columns(2)
        with col1:
            if "ReT" in batch_out.columns:
                st.markdown("**Polymer distribution**")
                poly_counts = (
                    batch_out["ReT"].value_counts().rename_axis("Polymer").reset_index(name="Count")
                )
                st.dataframe(poly_counts, hide_index=True, use_container_width=True)
            if "AgS" in batch_out.columns:
                st.markdown("**Ageing status**")
                st.dataframe(
                    batch_out["AgS"].value_counts().rename_axis("AgS").reset_index(name="Count"),
                    hide_index=True, use_container_width=True,
                )
        with col2:
            st.markdown("**Numeric ranges in batch**")
            range_rows = []
            for col in config.NUM_COLS:
                if col in batch_out.columns:
                    s = pd.to_numeric(batch_out[col], errors="coerce")
                    range_rows.append({
                        "Parameter": PARAM_LABELS.get(col, col),
                        "Min": f"{s.min():.4g}",
                        "Mean": f"{s.mean():.4g}",
                        "Max": f"{s.max():.4g}",
                    })
            if range_rows:
                st.dataframe(pd.DataFrame(range_rows), hide_index=True, use_container_width=True)
        if "lab_qe" in batch_out.columns:
            st.markdown("**Prediction summary (lab qe mg/g)**")
            qe = pd.to_numeric(batch_out["lab_qe"], errors="coerce").dropna()
            if not qe.empty:
                m1, m2, m3, m4 = st.columns(4)
                m1.metric("Min", f"{qe.min():.4g}")
                m2.metric("Mean", f"{qe.mean():.4g}")
                m3.metric("Median", f"{qe.median():.4g}")
                m4.metric("Max", f"{qe.max():.4g}")
    elif batch:
        st.info("Batch mode active — upload a file in Panel B and run prediction to see a summary here.")
        st.caption(f"Current sidebar FAF = {faf:.3f}")
    else:
        normed = normalize_input(input_dict)
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("**Categorical parameters**")
            cat_df = pd.DataFrame(
                [(PARAM_LABELS.get(k, k), normed[k]) for k in config.CAT_COLS if k in normed],
                columns=["Parameter", "Value"],
            )
            st.dataframe(cat_df, hide_index=True, use_container_width=True)
        with col2:
            st.markdown("**Numeric parameters**")
            num_df = pd.DataFrame(
                [(PARAM_LABELS.get(k, k), f"{normed[k]:.4g}") for k in config.NUM_COLS if k in normed],
                columns=["Parameter", "Value"],
            )
            st.dataframe(num_df, hide_index=True, use_container_width=True)
        st.caption(f"FAF = {faf:.3f} (field adjustment factor applied to all river-scale outputs)")


# ── Panel B – prediction ──────────────────────────────────────────────────────

def panel_prediction(input_dict: dict[str, Any], faf: float, batch: bool) -> dict[str, Any] | None:
    # ── Batch mode ────────────────────────────────────────────────────────────
    # Domain warnings are NOT applied here — each row in the file is validated
    # individually, so sidebar values do not block the uploader.
    if batch:
        upload = st.file_uploader(
            "Upload CSV/XLSX — columns must include: "
            + ", ".join(config.CAT_COLS + config.NUM_COLS),
            type=["csv", "xlsx"],
        )

        if upload is not None:
            if st.button("\U0001f9ee Run Batch Prediction", type="primary"):
                try:
                    df = (
                        pd.read_csv(upload)
                        if upload.name.lower().endswith(".csv")
                        else pd.read_excel(upload)
                    )
                except Exception as exc:
                    st.error(f"Could not read file: {exc}")
                    return None

                rows = []
                progress = st.progress(0, text="Starting…")
                for i, (_, row) in enumerate(df.iterrows()):
                    # Fall back to current sidebar values for missing columns
                    raw = {
                        key: row[key] if key in row.index and pd.notna(row[key]) else input_dict.get(key)
                        for key in config.CAT_COLS + config.NUM_COLS
                    }
                    try:
                        pred = run_predict_single(raw, faf)
                        rows.append({
                            **raw,
                            "lab_qe": pred["lab_qe"],
                            "lab_qe_lower": pred["lab_qe_lower"],
                            "lab_qe_upper": pred["lab_qe_upper"],
                            "river_qe": pred["river_qe"],
                            "river_qe_lower": pred["river_qe_lower"],
                            "river_qe_upper": pred["river_qe_upper"],
                        })
                    except Exception as exc:
                        rows.append({**raw, "error": str(exc)})
                    progress.progress((i + 1) / len(df), text=f"Row {i+1}/{len(df)}")

                # Second pass: batch SHAP (single TreeExplainer for all valid rows)
                valid_idx = [i for i, r in enumerate(rows) if "error" not in r]
                if valid_idx:
                    progress.progress(1.0, text=f"Computing SHAP for {len(valid_idx)} rows…")
                    normed_inputs = [
                        normalize_input({col: rows[i][col] for col in config.CAT_COLS + config.NUM_COLS})
                        for i in valid_idx
                    ]
                    shap_results = _shap_batch(normed_inputs)
                    for orig_i, shap_vals in zip(valid_idx, shap_results):
                        for feat, val in shap_vals.items():
                            rows[orig_i][f"shap_{feat}"] = val
                progress.empty()

                st.session_state["batch_results"] = pd.DataFrame(rows)
                st.session_state["batch_faf"] = faf

        # Show results persistently (survives Streamlit re-runs after button click)
        batch_out: pd.DataFrame | None = st.session_state.get("batch_results")
        if batch_out is not None and not batch_out.empty:
            saved_faf = st.session_state.get("batch_faf", faf)
            st.caption(f"Showing {len(batch_out)} rows — FAF = {saved_faf:.3f} used at run time")
            st.dataframe(batch_out, use_container_width=True)
            st.download_button(
                "\U0001f4be Download batch results CSV",
                batch_out.to_csv(index=False),
                "batch_predictions.csv",
                "text/csv",
            )

        return None

    # ── Single prediction mode ─────────────────────────────────────────────────
    warnings = domain_warnings(input_dict)
    for warning in warnings:
        st.warning(f"⚠️ Out-of-domain: {warning}")
    if warnings:
        st.stop()

    if st.button("\U0001f9ee Run Prediction", type="primary"):
        with st.spinner("Running QRF prediction…"):
            st.session_state["input_dict"] = normalize_input(input_dict)
            st.session_state["prediction"] = run_predict_single(input_dict, faf)

    pred = st.session_state.get("prediction")
    if not pred:
        st.info("ℹ️ Configure inputs in the sidebar and click **Run Prediction**.")
        return None

    # ── Metrics row ───────────────────────────────────────────────────────────
    left, right = st.columns(2)
    left.metric(
        "Lab qe (mg/g)",
        f"{pred['lab_qe']:.4g}",
        f"90% PI: {pred['lab_qe_lower']:.4g} – {pred['lab_qe_upper']:.4g}",
    )
    right.metric(
        "River-adjusted qe (mg/g)",
        f"{pred['river_qe']:.6g}",
        f"90% PI: {pred['river_qe_lower']:.6g} – {pred['river_qe_upper']:.6g}",
    )

    # ── Uncertainty gauge ─────────────────────────────────────────────────────
    pi_width = pred["lab_qe_upper"] - pred["lab_qe_lower"]
    gauge = min(pi_width / 3, 1.0)
    st.progress(gauge, text=f"Uncertainty width (PIₐₐ): {pi_width:.3g} mg/g")
    if pi_width < 1:
        st.success("\U0001f7e2 Uncertainty: green zone — PI width < 1 mg/g")
    elif pi_width <= 2:
        st.warning("\U0001f7e1 Uncertainty: amber zone — PI width 1–2 mg/g")
    else:
        st.error("\U0001f534 Uncertainty: red zone — PI width > 2 mg/g")

    fig = prediction_bar(pred)
    st.pyplot(fig, use_container_width=True)
    st.download_button(
        "\U0001f4be Download prediction figure",
        fig_to_png_bytes(fig),
        "prediction_uncertainty.png",
        "image/png",
    )
    return pred


# ── Panel C – SHAP ────────────────────────────────────────────────────────────

def _group_shap(shap_row: pd.Series) -> dict[str, float]:
    grouped: dict[str, float] = {}
    for name, value in shap_row.items():
        feature = name
        for cat in config.CAT_COLS:
            if name == cat or name.startswith(f"{cat}_"):
                feature = cat
        grouped[feature] = grouped.get(feature, 0.0) + float(value)
    return grouped


def _shap_batch(input_dicts: list[dict[str, Any]]) -> list[dict[str, float]]:
    """Compute grouped SHAP for multiple rows — TreeExplainer created once."""
    rf = load_rf_model()
    if rf is None or not hasattr(rf, "named_steps") or "preprocess" not in rf.named_steps:
        return [{} for _ in input_dicts]
    try:
        import shap as shap_lib
        pre = rf.named_steps["preprocess"]
        estimator = rf.named_steps["model"]
        x = pd.DataFrame([{col: d[col] for col in config.CAT_COLS + config.NUM_COLS} for d in input_dicts])
        x_proc = pre.transform(x)
        names = [n.replace("num__", "").replace("cat__", "") for n in pre.get_feature_names_out()]
        x_named = pd.DataFrame(x_proc, columns=names)
        explainer = shap_lib.TreeExplainer(estimator)
        values = np.asarray(explainer.shap_values(x_named, check_additivity=False))
        if values.ndim == 3:
            values = values[:, :, 0]
        return [_group_shap(pd.Series(row_vals, index=names)) for row_vals in values]
    except Exception:
        return [{} for _ in input_dicts]


def _try_shap_from_rf(input_dict: dict[str, Any]) -> list:
    """SHAP top-3 for a single row via standalone RF model."""
    grouped = _shap_batch([input_dict])
    g = grouped[0] if grouped else {}
    if not g:
        return []
    order = sorted(g.items(), key=lambda item: abs(item[1]), reverse=True)[:3]
    return [(f, v, "increases" if v > 0 else "decreases") for f, v in order]


def _shap_batch_panel(batch_out: pd.DataFrame) -> None:
    """Render mean-|SHAP| bar, heatmap, and per-row drilldown for batch results."""
    shap_cols = sorted(c for c in batch_out.columns if c.startswith("shap_"))
    if not shap_cols:
        st.info(
            "ℹ️ SHAP values were not computed for this batch — "
            "re-run the batch prediction in Panel B to include SHAP automatically."
        )
        return

    features = [c[len("shap_"):] for c in shap_cols]
    shap_df = batch_out[shap_cols].copy().rename(columns=dict(zip(shap_cols, features)))
    n_rows = len(shap_df)

    # ── Mean |SHAP| bar chart ─────────────────────────────────────────────────
    st.markdown("**Feature importance — mean |SHAP| across all batch rows**")
    mean_abs = shap_df.abs().mean().sort_values(ascending=True)
    bar_colors = ["#e45756" if shap_df[f].mean() >= 0 else "#4c78a8" for f in mean_abs.index]
    fig_bar, ax_bar = plt.subplots(figsize=(7, max(2.5, len(features) * 0.45)))
    ax_bar.barh(mean_abs.index, mean_abs.values, color=bar_colors)
    ax_bar.set_xlabel("Mean |SHAP| (mg/g)")
    ax_bar.set_title(f"n = {n_rows} rows", fontsize=9)
    ax_bar.grid(axis="x", linestyle=":", alpha=0.45)
    ax_bar.spines["top"].set_visible(False)
    ax_bar.spines["right"].set_visible(False)
    fig_bar.tight_layout()
    st.pyplot(fig_bar, use_container_width=True)
    st.download_button(
        "\U0001f4be Bar chart PNG", fig_to_png_bytes(fig_bar),
        "shap_importance_batch.png", "image/png",
    )

    # ── SHAP heatmap ──────────────────────────────────────────────────────────
    st.divider()
    st.markdown("**SHAP heatmap — every row × every feature**")
    feat_order = mean_abs.index[::-1].tolist()   # most important left
    heat_data = shap_df[feat_order].values
    vmax = float(np.abs(heat_data).max()) or 1.0
    fig_heat, ax_heat = plt.subplots(figsize=(9, max(3.5, n_rows * 0.38)))
    im = ax_heat.imshow(heat_data, cmap="RdBu_r", aspect="auto", vmin=-vmax, vmax=vmax)
    ax_heat.set_xticks(range(len(feat_order)))
    ax_heat.set_xticklabels(feat_order, rotation=35, ha="right", fontsize=9)
    row_tick_labels = [
        f"R{i+1} ({batch_out.iloc[i]['ReT']})" if "ReT" in batch_out.columns else f"Row {i+1}"
        for i in range(n_rows)
    ]
    ax_heat.set_yticks(range(n_rows))
    ax_heat.set_yticklabels(row_tick_labels, fontsize=8)
    plt.colorbar(im, ax=ax_heat, label="SHAP (mg/g)", fraction=0.025)
    ax_heat.set_title("Red = increases qe · Blue = decreases qe", fontsize=9)
    fig_heat.tight_layout()
    st.pyplot(fig_heat, use_container_width=True)
    dl_c1, dl_c2 = st.columns(2)
    dl_c1.download_button(
        "\U0001f4be Heatmap PNG", fig_to_png_bytes(fig_heat),
        "shap_heatmap_batch.png", "image/png",
    )
    shap_export = pd.concat(
        [batch_out[[c for c in config.CAT_COLS + config.NUM_COLS if c in batch_out.columns]], shap_df],
        axis=1,
    )
    dl_c2.download_button(
        "\U0001f4be SHAP table CSV", shap_export.to_csv(index=False),
        "shap_batch.csv", "text/csv",
    )

    # ── Drilldown: individual row waterfall ───────────────────────────────────
    st.divider()
    st.markdown("**Drilldown — waterfall for a single row**")
    row_labels = [
        f"Row {i+1}  |  {batch_out.iloc[i].get('ReT', '?')}  |  "
        f"pH={batch_out.iloc[i].get('pH', '?')}  |  Ce={batch_out.iloc[i].get('Ce', '?')}"
        for i in range(n_rows)
    ]
    row_idx = st.selectbox(
        "Select row", range(n_rows),
        format_func=lambda i: row_labels[i],
        key="shap_drilldown_row",
    )
    selected = batch_out.iloc[row_idx]
    row_shap = {feat: float(selected[f"shap_{feat}"]) for feat in features if f"shap_{feat}" in selected.index}
    top3 = sorted(row_shap.items(), key=lambda kv: abs(kv[1]), reverse=True)[:3]
    row_pred_wf = {
        "lab_qe": float(selected.get("lab_qe", 0)),
        "lab_qe_lower": float(selected.get("lab_qe_lower", 0)),
        "lab_qe_upper": float(selected.get("lab_qe_upper", 0)),
        "shap_top3": [(f, v, "increases" if v > 0 else "decreases") for f, v in top3],
    }
    fig_wf, tbl_wf = shap_waterfall(row_pred_wf)
    col_wf, col_ce = st.columns([3, 2])
    with col_wf:
        st.pyplot(fig_wf, use_container_width=True)
        st.dataframe(tbl_wf, use_container_width=True)
        st.download_button(
            "\U0001f4be Waterfall PNG", fig_to_png_bytes(fig_wf),
            f"shap_waterfall_row{row_idx+1}.png", "image/png",
        )
    with col_ce:
        row_input = {col: selected[col] for col in config.CAT_COLS + config.NUM_COLS if col in selected.index}
        ce_fig = ce_position_plot(row_input)
        st.pyplot(ce_fig, use_container_width=True)


def panel_shap(input_dict: dict[str, Any], pred: dict[str, Any] | None) -> None:
    # ── Batch mode ────────────────────────────────────────────────────────────
    if not pred:
        batch_out: pd.DataFrame | None = st.session_state.get("batch_results")
        if batch_out is not None and not batch_out.empty:
            _shap_batch_panel(batch_out)
        else:
            st.info("ℹ️ Run a prediction first (Panel B).")
        return

    shap_top3 = pred.get("shap_top3") or []

    # If QRF-based SHAP failed (empty), try the standalone RF model
    if not shap_top3:
        with st.spinner("Computing SHAP values via RF model…"):
            shap_top3 = _try_shap_from_rf(input_dict)
        if shap_top3:
            pred = {**pred, "shap_top3": shap_top3}

    # ── SHAP waterfall ────────────────────────────────────────────────────────
    fig, feature_table = shap_waterfall(pred)
    shap_available = not feature_table.empty

    if shap_available:
        st.pyplot(fig, use_container_width=True)
        st.dataframe(feature_table, use_container_width=True)
        dl1, dl2 = st.columns(2)
        dl1.download_button(
            "\U0001f4be SHAP figure PNG", fig_to_png_bytes(fig), "local_shap.png", "image/png"
        )
        dl2.download_button(
            "\U0001f4be SHAP table CSV", feature_table.to_csv(index=False), "local_shap.csv", "text/csv"
        )
    else:
        st.warning(
            "⚠️ SHAP values unavailable for this prediction.\n\n"
            "**Likely cause:** The QRF (QuantileRandomForest) pipeline does not expose "
            "a standard sklearn `RandomForestRegressor` compatible with SHAP's TreeExplainer, "
            "and no standalone RF artifact was found at `outputs/models/RandomForest_tuned.joblib`.\n\n"
            "**Fix:** Re-run the pipeline to generate a separate RF artifact, or ensure "
            "`named_steps['model']` is a standard sklearn RF."
        )

    # ── Ce vs CE₅₀ position ───────────────────────────────────────────────────
    st.divider()
    st.markdown("**Ce vs CE₅₀ position**")
    st.caption(
        f"Input Ce = {input_dict.get('Ce', '?')} mg/L  ·  "
        f"Reference CE₅₀ = {CE50_MG_L} mg/L"
    )
    ce_fig = ce_position_plot(input_dict)
    st.pyplot(ce_fig, use_container_width=True)
    st.download_button(
        "\U0001f4be Ce position PNG", fig_to_png_bytes(ce_fig), "ce_position.png", "image/png"
    )

    # ── What-if sensitivity (only when uncertainty is high) ───────────────────
    if pred.get("uncertainty_high"):
        st.divider()
        try:
            what_if = get_interpretation_module().what_if_sensitivity(input_dict)
            st.warning("⚠️ High uncertainty detected — what-if sensitivity analysis:")
            st.dataframe(what_if, use_container_width=True)
            st.download_button(
                "\U0001f4be What-if CSV", what_if.to_csv(index=False), "what_if_sensitivity.csv", "text/csv"
            )
        except Exception as exc:
            st.info(f"What-if sensitivity unavailable: {exc}")


# ── Panel D – LLM ────────────────────────────────────────────────────────────

def _build_combined_html(batch_exps: list[dict], mode: int) -> str:
    import html as _html
    rows_html = ""
    for e in batch_exps:
        body = (
            e["html"] if e.get("status") == "OK"
            else f"<p style='color:#b42318'>Error: {_html.escape(str(e.get('status', '')))}</p>"
        )
        rows_html += (
            f"<section style='margin-bottom:2em;border-top:2px solid #4c78a8;padding-top:1em'>"
            f"<h2>Row {e['row']} — {_html.escape(str(e.get('ReT','?')))} | "
            f"pH={_html.escape(str(e.get('pH','?')))} | Ce={_html.escape(str(e.get('Ce','?')))} | "
            f"lab qe={_html.escape(str(e.get('lab_qe','?')))} mg/g</h2>"
            f"{body}</section>"
        )
    return (
        "<!DOCTYPE html><html lang='en'><head><meta charset='utf-8'>"
        "<title>Batch LLM Explanations</title>"
        "<style>body{font-family:sans-serif;max-width:900px;margin:auto;padding:2em}"
        "h1,h2{color:#4c78a8}</style></head><body>"
        f"<h1>Cu²⁺-MP Batch LLM Explanations (C{mode})</h1>"
        f"{rows_html}</body></html>"
    )


def _panel_llm_batch_all(valid: pd.DataFrame, api_ok: bool) -> None:
    """Batch section: generate LLM explanations for all rows at once."""
    n = len(valid)
    st.markdown("**Generate explanations for all rows**")

    mode_options = ["C1 — Rule-based (offline)", "C2 — RAG-augmented", "C3 — Graph-RAG-augmented"]
    all_mode_label = st.radio("Mode for all rows", mode_options, horizontal=True, key="llm_all_mode")
    all_mode = int(all_mode_label[1])

    if all_mode in (2, 3) and not api_ok:
        st.warning("⚠️ C2/C3 require an API key. Will fall back to C1.")
    elif all_mode in (2, 3):
        st.caption(f"⚠️ Will make ~{n} API calls — estimated {n * 8}–{n * 20} s. Check usage limits.")

    if st.button(f"\U0001f504 Generate all {n} explanations", key="llm_gen_all", type="secondary"):
        safe_mode = all_mode if api_ok or all_mode == 1 else 1
        results: list[dict] = []
        prog = st.progress(0, text="Starting…")
        shap_features = [c[len("shap_"):] for c in valid.columns if c.startswith("shap_")]
        for i, (_, row) in enumerate(valid.iterrows()):
            row_input = normalize_input(
                {col: row[col] for col in config.CAT_COLS + config.NUM_COLS if col in row.index}
            )
            try:
                exp = cached_explanation(_input_cache_key(row_input), safe_mode, openai_api_fingerprint())
                results.append({
                    "row": i + 1,
                    "ReT": row.get("ReT", "?"),
                    "pH": row.get("pH", "?"),
                    "Ce": row.get("Ce", "?"),
                    "lab_qe": round(float(row.get("lab_qe", 0)), 4),
                    "river_qe": round(float(row.get("river_qe", 0)), 6),
                    "mode": safe_mode,
                    "html": exp["explanation_html"],
                    "status": "OK",
                })
            except Exception as exc:
                results.append({
                    "row": i + 1,
                    "ReT": row.get("ReT", "?"),
                    "pH": row.get("pH", "?"),
                    "Ce": row.get("Ce", "?"),
                    "status": f"Error: {exc}",
                })
            prog.progress((i + 1) / n, text=f"Row {i+1}/{n}…")
        prog.empty()
        st.session_state["batch_explanations"] = results
        st.session_state["batch_exp_mode"] = safe_mode

    batch_exps: list[dict] | None = st.session_state.get("batch_explanations")
    if not batch_exps:
        return

    exp_mode = st.session_state.get("batch_exp_mode", 1)
    n_ok = sum(1 for e in batch_exps if e.get("status") == "OK")
    st.success(f"✓ {n_ok}/{len(batch_exps)} explanations generated (C{exp_mode})")

    # ── Summary table ─────────────────────────────────────────────────────────
    summary_df = pd.DataFrame([
        {
            "Row": e["row"],
            "Polymer": e.get("ReT", "?"),
            "pH": e.get("pH", "?"),
            "Ce": e.get("Ce", "?"),
            "lab_qe (mg/g)": e.get("lab_qe", ""),
            "river_qe (mg/g)": e.get("river_qe", ""),
            "Status": e.get("status", "?"),
        }
        for e in batch_exps
    ])
    st.dataframe(summary_df, hide_index=True, use_container_width=True)

    # ── Per-row expandable explanations ───────────────────────────────────────
    st.markdown("**Per-row explanations (click to expand)**")
    for e in batch_exps:
        label = (
            f"Row {e['row']}  |  {e.get('ReT','?')}  |  "
            f"pH={e.get('pH','?')}  |  Ce={e.get('Ce','?')}  |  "
            f"lab qe={e.get('lab_qe','?')} mg/g"
        )
        with st.expander(label):
            if e.get("status") == "OK":
                st.markdown(e["html"], unsafe_allow_html=True)
            else:
                st.error(e["status"])

    # ── Downloads ─────────────────────────────────────────────────────────────
    dl1, dl2 = st.columns(2)
    dl1.download_button(
        "\U0001f4be All explanations HTML",
        _build_combined_html(batch_exps, exp_mode),
        "batch_explanations.html",
        "text/html",
    )
    dl2.download_button(
        "\U0001f4be Summary CSV",
        summary_df.to_csv(index=False),
        "batch_explanations_summary.csv",
        "text/csv",
    )


def panel_llm(input_dict: dict[str, Any], pred: dict[str, Any] | None) -> None:
    _batch_valid: pd.DataFrame | None = None

    # ── Batch mode: pick a row to explain ─────────────────────────────────────
    if not pred:
        batch_out: pd.DataFrame | None = st.session_state.get("batch_results")
        if batch_out is not None and not batch_out.empty:
            valid = (
                batch_out[batch_out["lab_qe"].notna()].reset_index(drop=True)
                if "lab_qe" in batch_out.columns
                else batch_out.reset_index(drop=True)
            )
            if valid.empty:
                st.warning("All batch rows failed prediction — cannot generate explanation.")
                return
            _batch_valid = valid
            n_valid = len(valid)
            row_labels = [
                f"Row {i+1}  |  {valid.iloc[i].get('ReT', '?')}  |  "
                f"pH={valid.iloc[i].get('pH', '?')}  |  Ce={valid.iloc[i].get('Ce', '?')}"
                for i in range(n_valid)
            ]
            st.markdown("**Per-sample explanation**")
            row_idx = st.selectbox(
                "Select row",
                range(n_valid),
                format_func=lambda i: row_labels[i],
                key="llm_batch_row",
            )
            sel = valid.iloc[row_idx]
            input_dict = normalize_input(
                {col: sel[col] for col in config.CAT_COLS + config.NUM_COLS if col in sel.index}
            )
            shap_features = [c[len("shap_"):] for c in valid.columns if c.startswith("shap_")]
            top3_raw = sorted(
                [(f, float(sel[f"shap_{f}"])) for f in shap_features if f"shap_{f}" in sel.index],
                key=lambda kv: abs(kv[1]), reverse=True,
            )[:3]
            pred = {
                "lab_qe": float(sel.get("lab_qe", 0)),
                "lab_qe_lower": float(sel.get("lab_qe_lower", 0)),
                "lab_qe_upper": float(sel.get("lab_qe_upper", 0)),
                "river_qe": float(sel.get("river_qe", 0)),
                "river_qe_lower": float(sel.get("river_qe_lower", 0)),
                "river_qe_upper": float(sel.get("river_qe_upper", 0)),
                "shap_top3": [(f, v, "increases" if v > 0 else "decreases") for f, v in top3_raw],
                "uncertainty_high": bool(
                    float(sel.get("lab_qe_upper", 0)) - float(sel.get("lab_qe_lower", 0)) > 2.0
                ),
            }
        else:
            st.info("ℹ️ Run a prediction first (Panel B).")
            return

    api_ok = openai_api_available()

    mode_options = ["C1 — Rule-based (offline)", "C2 — RAG-augmented", "C3 — Graph-RAG-augmented"]
    mode_label = st.radio("Interpretation mode", mode_options, horizontal=True, key="llm_single_mode")
    mode = int(mode_label[1])

    if mode in (2, 3) and not api_ok:
        st.warning("⚠️ C2/C3 require `OPENAI_API_KEY`. Falling back to C1.")

    if st.button("\U0001f4ac Generate Explanation", key="llm_gen_single"):
        safe_mode = mode if api_ok or mode == 1 else 1
        with st.spinner("Generating LLM explanation…"):
            result = cached_explanation(_input_cache_key(input_dict), safe_mode, openai_api_fingerprint())
        st.session_state["explanation"] = result

    result = st.session_state.get("explanation")
    if result:
        st.markdown(_render_explanation_text(result["explanation_html"]))
        sources = result.get("sources", [])
        if result["mode"] >= 2:
            source_labels = _unique_source_labels(sources)
            with st.expander(f"\U0001f4da Retrieved sources ({len(source_labels)} papers)"):
                st.caption("Key supporting papers used for retrieval.")
                for label in source_labels:
                    st.markdown(f"- {label}")
        if result["mode"] == 3:
            fig, lines = render_graph(normalize_input(input_dict)["ReT"])
            with st.expander("\U0001f578️ Knowledge Graph context"):
                st.pyplot(fig, use_container_width=True)
            with st.expander("\U0001f9f5 Graph path summary"):
                st.text("\n".join(lines[:30]) if lines else "No graph paths available.")
        html_text = result["explanation_html"]
        dl_col1, dl_col2 = st.columns(2)
        dl_col1.download_button("\U0001f4be Explanation HTML", html_text, "llm_interpretation.html", "text/html")
        try:
            path = get_interpretation_module().save_report(result, "streamlit_explanation")
            with path.open("rb") as handle:
                dl_col2.download_button("\U0001f4be Full report", handle.read(), path.name, "text/html")
        except Exception:
            pass

    # ── All-rows batch section (batch mode only) ──────────────────────────────
    if _batch_valid is not None:
        st.divider()
        _panel_llm_batch_all(_batch_valid, api_ok)

    st.divider()
    st.caption(
        "⚠️ **Limitations:** Grounded in a 24-paper corpus only. "
        "Ce > 200 mg/L extrapolation risk. "
        "KG built via keyword extraction. "
        "Cross-check against SHAP outputs."
    )


# ── Panel F – ablation ────────────────────────────────────────────────────────

_COND_COLORS = {"C0": "#bab0ac", "C1": "#4c78a8", "C2": "#f58518", "C3": "#54a24b"}
_JUDGE_DIMS = ["accuracy", "mechanistic_depth", "literature_grounding", "condition_specificity", "readability"]


def _judge_bar_figure(judge_df: pd.DataFrame) -> plt.Figure:
    CONDITIONS = ["C0", "C1", "C2", "C3"]
    dims = [d for d in _JUDGE_DIMS if d in judge_df.columns]
    summary = judge_df.groupby("condition")[dims + ["overall"]].agg(["mean", "sem"]).reindex(CONDITIONS)
    all_cols = dims + ["overall"]
    x = np.arange(len(all_cols))
    width = 0.18
    offsets = np.linspace(-1.5, 1.5, 4) * width
    fig, ax = plt.subplots(figsize=(11, 5))
    for i, cond in enumerate(CONDITIONS):
        if cond not in summary.index:
            continue
        means = [summary.loc[cond, (col, "mean")] for col in all_cols]
        sems = [summary.loc[cond, (col, "sem")] for col in all_cols]
        ax.bar(x + offsets[i], means, width, label=cond,
               color=_COND_COLORS[cond], yerr=sems, capsize=3,
               edgecolor="white", linewidth=0.5, error_kw={"elinewidth": 0.8})
    ax.set_xticks(x)
    ax.set_xticklabels([c.replace("_", "\n") for c in all_cols], fontsize=9)
    ax.set_ylim(0, 5.5)
    ax.set_ylabel("Mean score (1–5 Likert)", fontsize=10)
    ax.set_title("LLM-as-Judge scores by condition and dimension", fontsize=10)
    ax.legend(title="Condition", frameon=False, fontsize=9)
    ax.grid(axis="y", linestyle=":", alpha=0.45)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()
    return fig


def _judge_radar_figure(judge_df: pd.DataFrame) -> plt.Figure:
    CONDITIONS = ["C0", "C1", "C2", "C3"]
    dims = [d for d in _JUDGE_DIMS if d in judge_df.columns]
    summary = judge_df.groupby("condition")[dims].mean().reindex(CONDITIONS)
    N = len(dims)
    angles = np.linspace(0, 2 * np.pi, N, endpoint=False).tolist()
    angles += angles[:1]
    fig, ax = plt.subplots(figsize=(6, 6), subplot_kw={"polar": True})
    ax.set_theta_offset(np.pi / 2)
    ax.set_theta_direction(-1)
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels([d.replace("_", "\n") for d in dims], fontsize=9)
    ax.set_ylim(0, 5)
    ax.set_yticks([1, 2, 3, 4, 5])
    ax.set_yticklabels(["1", "2", "3", "4", "5"], fontsize=7)
    for cond in CONDITIONS:
        if cond not in summary.index:
            continue
        vals = summary.loc[cond, dims].tolist() + [summary.loc[cond, dims[0]]]
        ax.plot(angles, vals, linewidth=1.8, label=cond, color=_COND_COLORS[cond])
        ax.fill(angles, vals, alpha=0.10, color=_COND_COLORS[cond])
    ax.legend(loc="upper right", bbox_to_anchor=(1.35, 1.15), frameon=False, fontsize=9)
    ax.set_title("LLM-as-Judge: mean score per dimension", fontsize=10, pad=18)
    fig.tight_layout()
    return fig


def _panel_ablation_automated(auto: pd.DataFrame, retrieval: pd.DataFrame) -> None:
    """Keyword-based automated metrics (legacy, ceiling-affected)."""
    with st.expander("Automated keyword metrics (ceiling-affected — see note below)", expanded=False):
        st.caption(
            "⚠️ **Known limitation:** C1, C2, C3 score identically (0.55) on this rubric "
            "because the C1 rule-based template already contains all scored keywords. "
            "Use LLM-as-Judge results below for manuscript-quality discrimination."
        )
        rubric_cols = ["explanation_quality", "grounding_score", "mechanistic_specificity", "hallucination_rate"]
        legacy_cols = ["mechanism_match", "shap_consistency", "hallucination_rate"]
        metric_cols = rubric_cols if set(rubric_cols).issubset(auto.columns) else legacy_cols
        summary = auto.groupby("condition")[metric_cols].mean().reset_index()
        st.dataframe(summary, hide_index=True, use_container_width=True)

        retrieval_cols = (
            ["vector_relevance_at_5", "hybrid_relevance_at_5"]
            if {"vector_relevance_at_5", "hybrid_relevance_at_5"}.issubset(retrieval.columns)
            else ["vector_precision_at_5", "hybrid_precision_at_5"]
        )
        if set(retrieval_cols).issubset(retrieval.columns):
            means = retrieval[retrieval_cols].mean()
            gain = (means.iloc[1] - means.iloc[0]) * 100
            fig, ax = plt.subplots(figsize=(5.0, 3.5))
            ax.bar(["Vector-only", "Hybrid"], means, color=["#4c78a8", "#f58518"])
            ax.set_ylabel("Relevance@5")
            ax.set_ylim(0, 1)
            ax.set_title(f"Hybrid retrieval gain: {gain:+.1f}%")
            fig.tight_layout()
            st.pyplot(fig, use_container_width=True)
            st.download_button(
                "\U0001f4be Retrieval plot PNG", fig_to_png_bytes(fig),
                "retrieval_precision_ui.png", "image/png",
            )


def _panel_ablation_judge(judge_df: pd.DataFrame) -> None:
    """LLM-as-Judge results section."""
    CONDITIONS = ["C0", "C1", "C2", "C3"]
    st.markdown("**LLM-as-Judge scores** (5-dimension Likert rubric, n≈10 cases)")

    # ── Summary table ─────────────────────────────────────────────────────────
    dims = [d for d in _JUDGE_DIMS if d in judge_df.columns]
    all_metrics = dims + ["overall"]
    summary = judge_df.groupby("condition")[all_metrics].mean().reindex(CONDITIONS).round(2)
    highlight = [c for c in all_metrics if c != "overall"]
    styled = (
        summary.style
        .highlight_max(subset=highlight, color="#d1fadf")
        .highlight_max(subset=["overall"], color="#fef9c3")
        .format("{:.2f}")
    )
    st.dataframe(styled, use_container_width=True)
    st.download_button(
        "\U0001f4be Judge scores CSV",
        judge_df.to_csv(index=False),
        "llm_judge_scores_ui.csv",
        "text/csv",
    )

    # ── Figures ───────────────────────────────────────────────────────────────
    col_bar, col_radar = st.columns([3, 2])
    with col_bar:
        bar_fig = _judge_bar_figure(judge_df)
        st.pyplot(bar_fig, use_container_width=True)
        st.download_button(
            "\U0001f4be Bar chart PNG", fig_to_png_bytes(bar_fig),
            "llm_judge_bars_ui.png", "image/png",
        )
    with col_radar:
        radar_fig = _judge_radar_figure(judge_df)
        st.pyplot(radar_fig, use_container_width=True)
        st.download_button(
            "\U0001f4be Radar chart PNG", fig_to_png_bytes(radar_fig),
            "llm_judge_radar_ui.png", "image/png",
        )

    # ── Hypothesis metrics ────────────────────────────────────────────────────
    st.divider()
    st.markdown("**Hypothesis evaluation (LLM-as-Judge)**")
    means = judge_df.groupby("condition")["overall"].mean()
    dim_means = judge_df.groupby("condition")[dims].mean() if dims else pd.DataFrame()

    h1 = (
        means.get("C2", 0) > means.get("C1", 0)
        or means.get("C3", 0) > means.get("C1", 0)
    )
    h3_depth = (
        not dim_means.empty
        and "mechanistic_depth" in dim_means.columns
        and dim_means.loc["C3", "mechanistic_depth"] > dim_means.loc["C2", "mechanistic_depth"]
        if "C3" in dim_means.index and "C2" in dim_means.index else False
    )
    h3_grd = (
        not dim_means.empty
        and "literature_grounding" in dim_means.columns
        and dim_means.loc["C3", "literature_grounding"] >= dim_means.loc["C2", "literature_grounding"]
        if "C3" in dim_means.index and "C2" in dim_means.index else False
    )

    mc1, mc2, mc3 = st.columns(3)
    mc1.metric(
        "H1: C2/C3 > C1 overall",
        "✓ Confirmed" if h1 else "✗ Not confirmed",
        f"C3={means.get('C3', float('nan')):.2f}, C1={means.get('C1', float('nan')):.2f}",
    )
    mc2.metric(
        "H3a: C3 > C2 mechanistic depth",
        "✓ Confirmed" if h3_depth else "✗ Not confirmed",
        (
            f"C3={dim_means.loc['C3','mechanistic_depth']:.2f}, "
            f"C2={dim_means.loc['C2','mechanistic_depth']:.2f}"
        ) if not dim_means.empty and "mechanistic_depth" in dim_means.columns and "C3" in dim_means.index else "—",
    )
    mc3.metric(
        "H3b: C3 ≥ C2 literature grounding",
        "✓ Confirmed" if h3_grd else "✗ Not confirmed",
        (
            f"C3={dim_means.loc['C3','literature_grounding']:.2f}, "
            f"C2={dim_means.loc['C2','literature_grounding']:.2f}"
        ) if not dim_means.empty and "literature_grounding" in dim_means.columns and "C3" in dim_means.index else "—",
    )

    # ── Per-case rationale ────────────────────────────────────────────────────
    if "rationale" in judge_df.columns:
        with st.expander("\U0001f4cb Per-case judge rationales (raw)"):
            display_cols = ["case_id", "condition"] + [d for d in dims[:3] if d in judge_df.columns] + ["overall", "rationale"]
            st.dataframe(
                judge_df[[c for c in display_cols if c in judge_df.columns]],
                hide_index=True,
                use_container_width=True,
            )

    # ── Bias warning ─────────────────────────────────────────────────────────
    with st.expander("\U0001f6a7 Self-preference bias caveat"):
        st.markdown(
            "When the same LLM provider generates C2/C3 explanations **and** acts as judge, "
            "self-preference bias may inflate C2/C3 scores relative to C1 (rule-based, no LLM). "
            "Mitigations in this study: (1) C1 is generated without any LLM call; "
            "(2) the rubric scores observable criteria (citations present, mechanism named, "
            "values consistent); (3) raw judge responses are saved in `llm_judge_raw.csv` "
            "for manual inspection. This limitation must be stated in the manuscript."
        )


def panel_ablation() -> None:
    auto, retrieval = load_eval_data()
    judge_df = load_judge_data()

    # ── LLM-as-Judge (primary, manuscript-quality) ────────────────────────────
    judge_path = config.REPORTS_DIR / "llm_judge_scores.csv"
    if judge_df.empty:
        st.info(
            "ℹ️ LLM-as-Judge evaluation not yet run. "
            "Click the button below to start (requires API key, ~10 min for 10 cases)."
        )
        api_ok = openai_api_available()
        if not api_ok:
            st.warning("⚠️ No API key detected. Set `OPENAI_API_KEY` first.")
        n_cases = st.number_input("Number of cases to evaluate", min_value=3, max_value=30, value=10)
        if st.button("\U0001f916 Run LLM-as-Judge evaluation", disabled=not api_ok, type="primary"):
            with st.spinner(f"Running LLM-as-Judge on {n_cases} cases × 4 conditions × 2 API passes… (~{n_cases*4*2} calls)"):
                try:
                    judge_mod = _load_module("llm_judge_ui", RAG_DIR / "llm_judge.py")
                    judge_mod.run_llm_judge(n_cases=int(n_cases))
                    st.success("Done! Reload this tab to see results.")
                    st.cache_resource.clear()
                except Exception as exc:
                    st.error(f"Judge run failed: {exc}")
    else:
        n_cases_done = judge_df["case_id"].nunique() if "case_id" in judge_df.columns else "?"
        st.success(f"✓ LLM-as-Judge results available ({n_cases_done} cases × 4 conditions)")
        # Summary file
        summary_path = config.REPORTS_DIR / "llm_judge_summary.txt"
        if summary_path.exists():
            with st.expander("\U0001f4c4 Full statistical summary"):
                st.text(summary_path.read_text(encoding="utf-8"))
        _panel_ablation_judge(judge_df)
        col_rerun, _ = st.columns([1, 3])
        api_ok = openai_api_available()
        n_cases = col_rerun.number_input("Cases for re-run", min_value=3, max_value=30, value=10)
        if col_rerun.button("\U0001f504 Re-run judge", disabled=not api_ok):
            with st.spinner("Re-running LLM-as-Judge…"):
                try:
                    judge_mod = _load_module("llm_judge_ui", RAG_DIR / "llm_judge.py")
                    judge_mod.run_llm_judge(n_cases=int(n_cases))
                    st.success("Done! Reload to see updated results.")
                    st.cache_resource.clear()
                except Exception as exc:
                    st.error(f"Re-run failed: {exc}")

    # ── Automated metrics (legacy, collapsed by default) ──────────────────────
    st.divider()
    if not auto.empty and not retrieval.empty:
        _panel_ablation_automated(auto, retrieval)


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    st.title("Cu²⁺ – Microplastic Adsorption Predictor")
    st.caption(
        "Quantile Random Forest uncertainty · SHAP explanations · "
        "River-scale estimation · LLM interpretation (C1/C2/C3)"
    )

    input_dict, faf, _ce_badge, batch, _qrf = render_sidebar()

    # ── Panel A (always visible above tabs) ───────────────────────────────────
    with st.expander("\U0001f4cb Panel A — Input summary", expanded=True):
        panel_input_summary(input_dict, faf, batch=batch)

    st.divider()

    # ── Tabbed panels B–D ─────────────────────────────────────────────────────
    tab_b, tab_c, tab_d = st.tabs([
        "\U0001f4ca B — Prediction",
        "\U0001f9e0 C — SHAP",
        "\U0001f4ac D — LLM",
    ])

    with tab_b:
        pred = panel_prediction(input_dict, faf, batch)

    with tab_c:
        # Use the input_dict frozen at prediction time, not current sidebar values.
        # This ensures Ce position plot matches the prediction that was run.
        saved_input = st.session_state.get("input_dict") or normalize_input(input_dict)
        panel_shap(saved_input, st.session_state.get("prediction"))

    with tab_d:
        panel_llm(normalize_input(input_dict), st.session_state.get("prediction"))
    st.caption(
        "Explanation-module validation and ablation results are reported in the manuscript and SI; "
        "the app focuses on prediction, SHAP, and user-facing interpretation."
    )


if __name__ == "__main__":
    main()
