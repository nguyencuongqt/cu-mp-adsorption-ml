"""Main LLM Interpretation System with three context-enrichment modes."""

from __future__ import annotations

import html
import re
import sys
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
import shap

PROJECT_ROOT = Path(__file__).resolve().parents[2]
MODULE_DIR = Path(__file__).resolve().parent
for path in (PROJECT_ROOT, MODULE_DIR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from config import (  # noqa: E402
    CAT_COLS,
    DOMAIN,
    FAF_CENTRAL,
    FAF_RESULTS_FILE,
    MODEL_DIR,
    NUM_COLS,
    QRF_MODEL_FILE,
    REPORTS_DIR,
)
from hybrid_retrieval import classify_Ce, format_dataset_stats, hybrid_retrieve, polymer_context  # noqa: E402
from knowledge_graph import get_polymer_profile, graph_retrieve  # noqa: E402
from llm_provider import resolve_provider  # noqa: E402


_qrf_model: Any | None = None
_faf: tuple[float, float, float] | None = None


def _load_qrf() -> Any:
    global _qrf_model
    if _qrf_model is None:
        path = QRF_MODEL_FILE if QRF_MODEL_FILE.exists() else MODEL_DIR / "qrf_model.joblib"
        if not path.exists():
            raise FileNotFoundError(f"Missing QRF model: {path}. Run src/06_uncertainty_qrf.py first.")
        _qrf_model = joblib.load(path)
    return _qrf_model


def _load_faf() -> tuple[float, float, float]:
    global _faf
    if _faf is not None:
        return _faf
    if FAF_RESULTS_FILE.exists():
        df = pd.read_csv(FAF_RESULTS_FILE)
        cols = {"faf_central", "faf_ci_low", "faf_ci_high"}
        if cols.issubset(df.columns):
            row = df.iloc[0]
            _faf = (float(row["faf_central"]), float(row["faf_ci_low"]), float(row["faf_ci_high"]))
            return _faf
    _faf = (float(FAF_CENTRAL), float(FAF_CENTRAL), float(FAF_CENTRAL))
    return _faf


def _canonical_input(input_dict: dict[str, Any]) -> dict[str, Any]:
    values = dict(input_dict)
    for col in NUM_COLS:
        if col not in values:
            raise ValueError(f"Missing numeric input: {col}")
        low, high = DOMAIN[col]
        value = float(values[col])
        if not (low <= value <= high):
            raise ValueError(f"{col}={value} is outside domain [{low}, {high}].")
        values[col] = value
    for col in CAT_COLS:
        if col not in values:
            raise ValueError(f"Missing categorical input: {col}")
        allowed = DOMAIN[col]
        if str(values[col]) not in allowed:
            upper_allowed = {str(v).upper(): v for v in allowed}
            key = str(values[col]).upper()
            if key in upper_allowed:
                values[col] = upper_allowed[key]
            else:
                raise ValueError(f"{col}={values[col]} is outside domain {allowed}.")
    return values


def _input_frame(input_dict: dict[str, Any]) -> pd.DataFrame:
    return pd.DataFrame([{col: input_dict[col] for col in CAT_COLS + NUM_COLS}])


def _predict_quantiles(model: Any, input_dict: dict[str, Any]) -> tuple[float, float, float]:
    preds = np.asarray(model.predict(_input_frame(input_dict), quantiles=[0.05, 0.50, 0.95]))
    if preds.ndim == 2:
        return float(preds[0, 0]), float(preds[0, 1]), float(preds[0, 2])
    return float(preds[0]), float(preds[0]), float(preds[0])


def _shap_top3(model: Any, input_dict: dict[str, Any]) -> list[tuple[str, float, str]]:
    if not hasattr(model, "named_steps") or "preprocess" not in model.named_steps:
        return []
    try:
        pre = model.named_steps["preprocess"]
        rf = model.named_steps["model"]
        x = _input_frame(input_dict)
        x_proc = pre.transform(x)
        names = [name.replace("num__", "").replace("cat__", "") for name in pre.get_feature_names_out()]
        x_named = pd.DataFrame(x_proc, columns=names)
        # No background → TreeExplainer uses tree mean as baseline.
        # Using background=x_named (single point) causes all SHAP values = 0.
        explainer = shap.TreeExplainer(rf)
        values = np.asarray(explainer.shap_values(x_named, check_additivity=False))
        if values.ndim == 3:
            values = values[:, :, 0]
        shap_row = pd.Series(values[0], index=names)
    except Exception:
        return []

    grouped: dict[str, float] = {}
    for name, value in shap_row.items():
        feature = name
        for cat in CAT_COLS:
            if name == cat or name.startswith(f"{cat}_"):
                feature = cat
        grouped[feature] = grouped.get(feature, 0.0) + float(value)
    order = sorted(grouped.items(), key=lambda item: abs(item[1]), reverse=True)[:3]
    return [(feature, value, "increases" if value > 0 else "decreases") for feature, value in order]


def predict_single(input_dict: dict[str, Any]) -> dict[str, Any]:
    """Validate inputs, predict lab/river qe, compute top SHAP drivers."""
    values = _canonical_input(input_dict)
    model = _load_qrf()
    lab_low, lab_med, lab_high = _predict_quantiles(model, values)
    faf, faf_low, faf_high = _load_faf()
    result = {
        "lab_qe": lab_med,
        "lab_qe_lower": lab_low,
        "lab_qe_upper": lab_high,
        "river_qe": lab_med * faf,
        "river_qe_lower": lab_low * faf_low,
        "river_qe_upper": lab_high * faf_high,
        "shap_top3": _shap_top3(model, values),
        "uncertainty_high": bool((lab_high - lab_low) > 2.0),
    }
    return result


def _rule_based_html(input_dict: dict[str, Any], pred: dict[str, Any]) -> str:
    profile = get_polymer_profile(str(input_dict["ReT"]))
    stats = format_dataset_stats(input_dict)
    shap_items = "".join(
        f"<li>{html.escape(feature)}: {direction} qe ({value:.4g})</li>"
        for feature, value, direction in pred.get("shap_top3", [])
    )
    mechanisms = ", ".join(profile.get("mechanisms", [])) or "not resolved in the current graph"
    return (
        "<section>"
        f"<h3>Rule-based local interpretation</h3>"
        f"<p>Predicted lab qe is <b>{pred['lab_qe']:.4g} mg/g</b>; FAF-adjusted river qe is "
        f"<b>{pred['river_qe']:.6g} mg/g</b>.</p>"
        f"<p>Top local drivers from SHAP:</p><ul>{shap_items}</ul>"
        f"<p>Polymer profile for {html.escape(str(input_dict['ReT']))}: "
        f"{profile.get('n_papers', 0)} linked papers; mechanisms: {html.escape(mechanisms)}.</p>"
        f"<p>{html.escape(stats)}</p>"
        "<p>Check Ce extrapolation and prediction interval width before using this explanation for decisions.</p>"
        "</section>"
    )


def _call_llm(system: str, user: str) -> str:
    provider, api_key = resolve_provider()
    if provider == "openai" and api_key:
        from openai import OpenAI

        client = OpenAI(api_key=api_key)
        resp = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "system", "content": system}, {"role": "user", "content": user}],
            temperature=0.2,
        )
        return resp.choices[0].message.content or ""
    raise RuntimeError("No API key found. Set OPENAI_API_KEY or add it to .env.")


def _short_citation_id(paper_id: str) -> str:
    match = re.match(r"([A-Za-z]+)_(\d{4})", str(paper_id).strip())
    if match:
        return f"{match.group(1)}_{match.group(2)}"
    return str(paper_id).strip()


def _normalize_citation_ids(text: str, sources: list[dict[str, Any]]) -> str:
    normalized = str(text)
    paper_ids = sorted(
        {str(src.get("paper_id", "")).strip() for src in sources if str(src.get("paper_id", "")).strip()},
        key=len,
        reverse=True,
    )
    for paper_id in paper_ids:
        short_id = _short_citation_id(paper_id)
        normalized = re.sub(
            rf"(?<![A-Za-z0-9_]){re.escape(paper_id)}(?![A-Za-z0-9_])",
            short_id,
            normalized,
        )
    return normalized


def _llm_html(input_dict: dict[str, Any], pred: dict[str, Any], mode: int) -> tuple[str, list[dict[str, Any]]]:
    context, sources = hybrid_retrieve(input_dict, pred)
    stats = format_dataset_stats(input_dict)
    extra = ""
    if mode == 3:
        profile, polymer_chunks = polymer_context(str(input_dict["ReT"]), top_k=5)
        graph_lines = graph_retrieve(
            {"polymer": input_dict["ReT"], "AgS": input_dict["AgS"], "Ce_bin": classify_Ce(input_dict["Ce"])},
            hops=2,
        )
        chunk_text = "\n".join(f"[Source: {c.get('paper_id')}] {c.get('text')}" for c in polymer_chunks)
        extra = f"\nPolymer profile: {profile}\nPolymer chunks:\n{chunk_text}\nGraph context:\n" + "\n".join(graph_lines[:12])

    system = "Answer from context only. Ce>200 = extrapolation zone."
    task = (
        "Write 150-200 words covering: why this qe | literature support | river implication | key uncertainty. "
        "Format as HTML."
    )
    if mode == 3:
        task += (
            " Explain via polymer physicochemical properties from KG; cite compact source ids in the form "
            "Surname_Year (for example, Muthuraja_2024), not full paper_ids; add graph-based mechanistic insight."
        )
    user = (
        f"Inputs: {input_dict}\nOutputs: {pred}\nDataset stats: {stats}\nContext:\n{context}\n{extra}\n"
        f"Task: {task}\n"
        "Append exactly: Grounded in 24-paper corpus only. Ce>200 extrapolation risk. "
        "KG via keyword extraction. Cross-check SHAP outputs."
    )
    html_text = _call_llm(system, user)
    return _normalize_citation_ids(html_text, sources), sources


def what_if_sensitivity(input_dict: dict[str, Any]) -> pd.DataFrame:
    """Perturb inputs when uncertainty is high and return qe changes."""
    base = _canonical_input(input_dict)
    base_pred = predict_single(base)
    perturbations = [
        ("pH -1", "pH", base["pH"] - 1),
        ("pH +1", "pH", base["pH"] + 1),
        ("Temp -5", "Temp", base["Temp"] - 5),
        ("Temp +5", "Temp", base["Temp"] + 5),
        ("Ce x0.5", "Ce", base["Ce"] * 0.5),
        ("Ce x2", "Ce", base["Ce"] * 2),
    ]
    rows: list[dict[str, Any]] = []
    for label, key, value in perturbations:
        low, high = DOMAIN[key]
        scenario = dict(base)
        scenario[key] = min(max(float(value), low), high)
        pred = predict_single(scenario)
        d_lab = pred["lab_qe"] - base_pred["lab_qe"]
        d_river = pred["river_qe"] - base_pred["river_qe"]
        rows.append(
            {
                "perturbation": label,
                "Δlab_qe": d_lab,
                "Δriver_qe": d_river,
                "direction": "increase" if d_lab > 0 else "decrease" if d_lab < 0 else "no change",
            }
        )
    return pd.DataFrame(rows)


def explain(input_dict: dict[str, Any], mode: int = 1) -> dict[str, Any]:
    """Return explanation output for mode 1, 2, or 3."""
    values = _canonical_input(input_dict)
    pred = predict_single(values)
    sources: list[dict[str, Any]] = []
    if mode == 1:
        explanation_html = _rule_based_html(values, pred)
    else:
        try:
            explanation_html, sources = _llm_html(values, pred, mode=mode)
        except Exception as exc:
            explanation_html = _rule_based_html(values, pred) + f"<p>LLM fallback reason: {html.escape(str(exc))}</p>"
    what_if_df = what_if_sensitivity(values) if pred["uncertainty_high"] else pd.DataFrame()
    return {
        "mode": mode,
        "explanation_html": explanation_html,
        "sources": sources,
        "what_if_df": what_if_df,
        "lab_qe": pred["lab_qe"],
        "river_qe": pred["river_qe"],
    }


def save_report(result: dict[str, Any], filename: str) -> Path:
    """Save an explanation result to outputs/reports/{filename}.html."""
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    safe_name = filename if filename.endswith(".html") else f"{filename}.html"
    output = REPORTS_DIR / safe_name
    output.write_text(result["explanation_html"], encoding="utf-8")
    return output


if __name__ == "__main__":
    demo = {"ReT": "PP", "AgS": "aged", "AdC": "mixed", "Temp": 25, "pH": 7, "rpm": 160, "Ce": 0.00877}
    for demo_mode in (1, 2, 3):
        print(f"\nMODE {demo_mode}")
        print(explain(demo, mode=demo_mode)["explanation_html"])
