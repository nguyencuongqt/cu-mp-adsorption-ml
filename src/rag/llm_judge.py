"""LLM-as-Judge evaluation for the LLM Interpretation System ablation study.

An LLM judge scores C0–C3 explanations on a structured 5-dimension rubric,
providing a discrimination-sensitive alternative to the keyword-based automated
metrics (which suffer from a ceiling effect at C1=C2=C3).

Method
------
1. Sample up to N_CASES diverse test cases from test_cases.csv.
2. For each case × condition, generate the explanation (C0=numbers only,
   C1=rule-based, C2=RAG, C3=Graph-RAG).
3. Call an LLM judge with a structured rubric to score the plain-text
   explanation on 5 dimensions (1–5 Likert each).
4. Run Friedman + Wilcoxon–Holm statistical tests across conditions.
5. Save scores, raw responses, summary text, and two manuscript-ready figures.

Limitation (must be reported in the manuscript)
-------------------------------------------------
When the same OpenAI model family is used for both generation and judging, the
judge evaluates related C2/C3 outputs. This introduces potential self-preference
bias. Mitigations applied here: (a) a blinded rubric that scores observable
criteria (citations present, mechanism named, values consistent); (b) C1 is
generated without the LLM, providing an independent reference; (c) raw judge
responses are saved for manual inspection.

Run with:
    python src/rag/llm_judge.py

Outputs:
    outputs/reports/llm_judge_scores.csv    — per-case per-condition scores
    outputs/reports/llm_judge_raw.csv       — raw LLM judge responses
    outputs/reports/llm_judge_summary.txt   — stats + hypothesis narrative
    outputs/figures/llm_judge_bars.png      — grouped bar chart (manuscript)
    outputs/figures/llm_judge_radar.png     — radar chart per dimension
"""

from __future__ import annotations

import json
import re
import sys
import time
from decimal import Decimal, ROUND_HALF_UP
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
from llm_interpretation import explain, predict_single  # noqa: E402
from llm_provider import mask_key, resolve_provider  # noqa: E402

# ── Constants ──────────────────────────────────────────────────────────────────

CONDITIONS = ["C0", "C1", "C2", "C3"]
N_CASES = 10  # number of diverse cases to evaluate

DIMENSIONS = [
    "accuracy",
    "mechanistic_depth",
    "literature_grounding",
    "condition_specificity",
    "readability",
]

# Dimension weights for computing a weighted overall score
DIM_WEIGHTS = {
    "accuracy": 0.25,
    "mechanistic_depth": 0.30,
    "literature_grounding": 0.20,
    "condition_specificity": 0.15,
    "readability": 0.10,
}

# Rubric text shown verbatim to the judge
RUBRIC = """\
Score each dimension 1–5 using these criteria:

accuracy (1–5)
  1 = qe values absent or wrong; 5 = qe, PI width, and FAF scaling all stated
  and internally consistent with the given prediction.

mechanistic_depth (1–5)
  1 = no mechanism mentioned; 3 = generic phrase (e.g. "adsorption occurs");
  5 = specific named mechanism (electrostatic attraction, surface complexation,
  ion exchange, hydrophobic interaction, etc.) tied to this polymer/condition.

literature_grounding (1–5)
  1 = no citation or corpus reference; 3 = vague mention of "literature" or
  "dataset"; 5 = ≥2 specific paper IDs cited with relevant mechanistic content.

condition_specificity (1–5)
  1 = generic statements applicable to any case; 3 = at least one condition
  (pH, Ce, or polymer) discussed specifically; 5 = all key conditions (pH,
  Ce level vs CE50, polymer type, ageing) addressed with tailored reasoning.

readability (1–5)
  1 = incomprehensible or just raw numbers; 3 = understandable but disorganised;
  5 = clear, well-structured, interpretable by a domain scientist."""

JUDGE_SYSTEM = (
    "You are an expert scientific reviewer evaluating AI-generated explanations "
    "of Cu²⁺ adsorption onto microplastics. "
    "Score each explanation objectively using the rubric. "
    "Be consistent: identical quality must receive identical scores. "
    "Output ONLY a valid JSON object — no markdown fences, no extra text."
)

API_DELAY_S = 0.8  # seconds between API calls to respect rate limits


# ── Utilities ─────────────────────────────────────────────────────────────────

def _ensure_dirs() -> None:
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)


def _strip_html(text: str) -> str:
    plain = re.sub(r"<[^>]+>", " ", str(text))
    return re.sub(r"\s+", " ", plain).strip()


def _parse_json_response(text: str) -> dict[str, Any]:
    """Extract JSON from LLM response, with regex fallback."""
    # Try direct parse first
    try:
        return json.loads(text.strip())
    except json.JSONDecodeError:
        pass
    # Try extracting first {...} block
    m = re.search(r"\{[^{}]*\}", text, re.DOTALL)
    if m:
        try:
            return json.loads(m.group())
        except json.JSONDecodeError:
            pass
    # Fallback: extract individual scores with regex
    scores: dict[str, Any] = {}
    for dim in DIMENSIONS + ["overall"]:
        m2 = re.search(rf'"{dim}"\s*:\s*([1-5])', text)
        if m2:
            scores[dim] = int(m2.group(1))
    m3 = re.search(r'"rationale"\s*:\s*"([^"]{10,})"', text)
    scores["rationale"] = m3.group(1) if m3 else "parse error"
    return scores


def _is_auth_error(message: str) -> bool:
    text = str(message).lower()
    return "401" in text or "invalid_api_key" in text or "incorrect api key" in text


def _call_llm_judge(system: str, user: str) -> str:
    """Call the OpenAI API for the judge role."""
    provider, api_key = resolve_provider()
    if provider == "openai" and api_key:
        from openai import OpenAI
        client = OpenAI(api_key=api_key)
        resp = client.chat.completions.create(
            model="gpt-4o-mini",
            temperature=0.0,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
        )
        return resp.choices[0].message.content or ""
    raise RuntimeError("No API key found. Set OPENAI_API_KEY or add it to .env.")


def _judge_explanation(
    input_dict: dict[str, Any],
    pred: dict[str, Any],
    explanation_html: str,
    condition: str,
) -> dict[str, Any]:
    """Call the LLM judge and return a dict of dimension scores."""
    plain = _strip_html(explanation_html)[:2500]  # cap to ~600 tokens
    user = (
        "=== Experimental conditions ===\n"
        f"Polymer (ReT): {input_dict['ReT']}\n"
        f"Ageing status (AgS): {input_dict['AgS']}\n"
        f"Adsorption type (AdC): {input_dict['AdC']}\n"
        f"Temperature: {input_dict['Temp']} °C\n"
        f"pH: {input_dict['pH']}\n"
        f"Mixing speed: {input_dict['rpm']} rpm\n"
        f"Equilibrium Cu²⁺ concentration (Ce): {input_dict['Ce']} mg/L\n\n"
        "=== Model prediction ===\n"
        f"Lab qe: {pred['lab_qe']:.4g} mg/g  "
        f"(90% PI: {pred['lab_qe_lower']:.4g}–{pred['lab_qe_upper']:.4g})\n"
        f"River-adjusted qe: {pred['river_qe']:.6g} mg/g\n"
        f"High uncertainty flag: {pred['uncertainty_high']}\n\n"
        "=== Explanation to evaluate ===\n"
        f"{plain}\n\n"
        "=== Scoring rubric ===\n"
        f"{RUBRIC}\n\n"
        "Output a JSON object with exactly these keys:\n"
        '{"accuracy": <1-5>, "mechanistic_depth": <1-5>, '
        '"literature_grounding": <1-5>, "condition_specificity": <1-5>, '
        '"readability": <1-5>, "overall": <1-5>, '
        '"rationale": "<50-word summary of your scoring decision>"}'
    )
    raw = _call_llm_judge(JUDGE_SYSTEM, user)
    parsed = _parse_json_response(raw)
    parsed["condition"] = condition
    parsed["raw_response"] = raw
    return parsed


# ── Sampling ──────────────────────────────────────────────────────────────────

def _sample_diverse_cases(df: pd.DataFrame, n: int) -> pd.DataFrame:
    """Select up to n cases maximising diversity across polymer type and Ce."""
    if len(df) <= n:
        return df.copy()

    # Prefer rows where Ce spans the full range (low / mid / high vs CE50=29.5)
    df = df.copy()
    ce_col = "input_Ce" if "input_Ce" in df.columns else None
    ret_col = "input_ReT" if "input_ReT" in df.columns else None

    if ret_col and df[ret_col].nunique() > 1:
        # Stratify by polymer type first
        selected_indices: list[int] = []
        per_polymer = max(1, n // df[ret_col].nunique())
        for _, grp in df.groupby(ret_col):
            selected_indices.extend(
                grp.sample(min(per_polymer, len(grp)), random_state=42).index.tolist()
            )
        # Top up to n using Ce quantiles on remaining rows
        remaining = df.drop(index=selected_indices)
    else:
        selected_indices = []
        remaining = df.copy()

    if ce_col and not remaining.empty:
        remaining = remaining.copy()
        remaining["_ce_bin"] = pd.qcut(
            pd.to_numeric(remaining[ce_col], errors="coerce").fillna(0),
            q=min(n - len(selected_indices), len(remaining)),
            labels=False,
            duplicates="drop",
        )
        for _, grp in remaining.groupby("_ce_bin"):
            if len(selected_indices) >= n:
                break
            selected_indices.append(grp.index[0])
    else:
        step = max(1, len(remaining) // max(1, n - len(selected_indices)))
        for i in range(0, len(remaining), step):
            if len(selected_indices) >= n:
                break
            selected_indices.append(remaining.index[i])

    return df.loc[sorted(set(selected_indices))[:n]].copy()


# ── C0 explanation (numbers only, no LLM) ────────────────────────────────────

def _c0_explanation(pred: dict[str, Any]) -> str:
    return (
        f"lab_qe={pred['lab_qe']:.4g} mg/g; "
        f"river_qe={pred['river_qe']:.6g} mg/g; "
        f"90% PI: {pred['lab_qe_lower']:.4g}–{pred['lab_qe_upper']:.4g}; "
        f"uncertainty_high={pred['uncertainty_high']}"
    )


# ── Generate explanations ─────────────────────────────────────────────────────

def _generate_explanation(input_dict: dict[str, Any], condition: str) -> tuple[str, list]:
    """Return (html_text, sources) for the given condition."""
    if condition == "C0":
        pred = predict_single(input_dict)
        return _c0_explanation(pred), []
    mode = int(condition[-1])
    result = explain(input_dict, mode=mode)
    return result["explanation_html"], result.get("sources", [])


# ── Statistical tests ─────────────────────────────────────────────────────────

def _stat_tests(scores_df: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    metrics = DIMENSIONS + ["overall"]
    pairs = [("C3", "C2"), ("C2", "C1"), ("C1", "C0"), ("C3", "C0")]

    for metric in metrics:
        pivot = scores_df.pivot(index="case_id", columns="condition", values=metric).reindex(
            columns=CONDITIONS
        )
        pivot = pivot.dropna(how="any")
        if len(pivot) < 3 or pivot.shape[1] < 4:
            continue
        try:
            friedman_stat, friedman_p = friedmanchisquare(*(pivot[c] for c in CONDITIONS))
        except ValueError:
            friedman_stat, friedman_p = np.nan, np.nan

        pair_rows: list[dict[str, Any]] = []
        for a, b in pairs:
            try:
                stat, p = wilcoxon(pivot[a], pivot[b], zero_method="zsplit")
            except ValueError:
                stat, p = np.nan, 1.0
            pair_rows.append({"metric": metric, "contrast": f"{a}>{b}", "stat": stat, "p": p})

        adj = multipletests([r["p"] for r in pair_rows], method="holm")[1]
        for row, p_adj in zip(pair_rows, adj):
            row.update({"p_holm": float(p_adj), "friedman_stat": friedman_stat, "friedman_p": friedman_p})
            rows.append(row)

    return pd.DataFrame(rows)


# ── Figures ───────────────────────────────────────────────────────────────────

_CONDITION_COLORS = {
    "C0": "#bab0ac",
    "C1": "#4c78a8",
    "C2": "#f58518",
    "C3": "#54a24b",
}


def _format_score(value: float) -> str:
    """Format reported scores with conventional half-up rounding."""
    return format(
        Decimal(str(float(value))).quantize(Decimal("0.01"), rounding=ROUND_HALF_UP),
        ".2f",
    )


def _bar_chart(scores_df: pd.DataFrame, tests: pd.DataFrame) -> Path:
    """Grouped bar chart: one group per dimension, bars per condition."""
    summary = scores_df.groupby("condition")[DIMENSIONS + ["overall"]].agg(["mean", "sem"])
    summary = summary.reindex(CONDITIONS)

    dims_to_plot = DIMENSIONS + ["overall"]
    x = np.arange(len(dims_to_plot))
    n_conditions = len(CONDITIONS)
    width = 0.14
    offsets = np.linspace(-(n_conditions - 1) / 2, (n_conditions - 1) / 2, n_conditions) * width

    fig, ax = plt.subplots(figsize=(16.5, 8.8))
    sig = tests[(tests["contrast"] == "C3>C0") & (tests["p_holm"] < 0.05)]
    for i, cond in enumerate(CONDITIONS):
        means = [summary.loc[cond, (dim, "mean")] for dim in dims_to_plot]
        sems = [summary.loc[cond, (dim, "sem")] for dim in dims_to_plot]
        ax.bar(
            x + offsets[i], means, width,
            label=cond,
            color=_CONDITION_COLORS[cond],
            yerr=sems,
            capsize=4,
            edgecolor="white",
            linewidth=0.6,
            error_kw={"elinewidth": 1.0},
        )
    # Annotate significant contrasts (p_holm < 0.05 for C3>C0 on each dim)
    for j, dim in enumerate(dims_to_plot):
        if not sig[sig["metric"] == dim].empty:
            ax.text(x[j], 5.23, "*", ha="center", va="bottom", fontsize=28, color="#e45756", weight="bold")

    ax.set_xticks(x)
    ax.set_xticklabels(
        [d.replace("_", "\n") for d in dims_to_plot],
        fontsize=20,
    )
    ax.set_ylim(0, 5.5)
    ax.set_ylabel("Mean score (1-5 Likert)", fontsize=22, labelpad=12)
    ax.set_title(
        "LLM-as-Judge scores by condition and dimension\n(* = C3>C0, Holm-adjusted p<0.05)",
        fontsize=26,
        pad=22,
    )
    ax.legend(
        title="Condition",
        frameon=False,
        fontsize=19,
        title_fontsize=20,
        ncol=4,
        loc="upper center",
        bbox_to_anchor=(0.5, -0.12),
    )
    ax.grid(axis="y", linestyle=":", alpha=0.45)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(axis="y", labelsize=19, pad=8)
    fig.subplots_adjust(top=0.83, bottom=0.25, left=0.08, right=0.98)

    output = FIGURES_DIR / "llm_judge_bars.png"
    fig.savefig(output, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return output


def _radar_chart(scores_df: pd.DataFrame) -> Path:
    """Radar / spider chart: one polygon per condition, axes = 5 dimensions."""
    summary = scores_df.groupby("condition")[DIMENSIONS].mean().reindex(CONDITIONS)
    categories = [d.replace("_", "\n") for d in DIMENSIONS]
    N = len(categories)
    angles = np.linspace(0, 2 * np.pi, N, endpoint=False).tolist()
    angles += angles[:1]  # close the polygon

    fig, ax = plt.subplots(figsize=(10.8, 9.4), subplot_kw={"polar": True})
    ax.set_theta_offset(np.pi / 2)
    ax.set_theta_direction(-1)
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(categories, fontsize=22)
    ax.tick_params(axis="x", pad=22)
    ax.set_ylim(0, 5)
    ax.set_yticks([1, 2, 3, 4, 5])
    ax.set_yticklabels(["1", "2", "3", "4", "5"], fontsize=17)

    for cond in CONDITIONS:
        if cond not in summary.index:
            continue
        values = summary.loc[cond, DIMENSIONS].tolist()
        values += values[:1]
        ax.plot(angles, values, linewidth=2.6, label=cond, color=_CONDITION_COLORS[cond])
        ax.fill(angles, values, alpha=0.11, color=_CONDITION_COLORS[cond])

    ax.legend(loc="upper right", bbox_to_anchor=(1.36, 1.18), frameon=False, fontsize=19)
    ax.set_title("LLM-as-Judge: mean score per dimension", fontsize=26, pad=40)
    fig.subplots_adjust(top=0.85, bottom=0.06, left=0.02, right=0.80)

    output = FIGURES_DIR / "llm_judge_radar.png"
    fig.savefig(output, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return output


# ── Narrative summary ─────────────────────────────────────────────────────────

def _build_summary(scores_df: pd.DataFrame, tests: pd.DataFrame, n_cases: int) -> str:
    provider, api_key = resolve_provider()
    provider_label = {
        "openai": "GPT-4o-mini",
        None: "None",
    }[provider]
    means = scores_df.groupby("condition")["overall"].mean().reindex(CONDITIONS)
    dims_present = [d for d in DIMENSIONS if d in scores_df.columns]
    dim_means = (
        scores_df.groupby("condition")[dims_present].mean().reindex(CONDITIONS)
        if dims_present else pd.DataFrame(index=CONDITIONS)
    )

    if not tests.empty and "contrast" in tests.columns and "p_holm" in tests.columns:
        sig_c3c0 = tests[(tests["contrast"] == "C3>C0") & (tests["p_holm"] < 0.05)]["metric"].tolist()
        sig_c3c1 = tests[(tests["contrast"] == "C3>C2") & (tests["p_holm"] < 0.05)]["metric"].tolist()
    else:
        sig_c3c0, sig_c3c1 = [], []

    def _safe_mean(cond: str) -> float:
        v = means.get(cond, float("nan"))
        return float(v) if not pd.isna(v) else float("nan")

    def _safe_dim(cond: str, dim: str) -> float:
        if dim_means.empty or dim not in dim_means.columns or cond not in dim_means.index:
            return float("nan")
        v = dim_means.loc[cond, dim]
        return float(v) if not pd.isna(v) else float("nan")

    c1m, c2m, c3m = _safe_mean("C1"), _safe_mean("C2"), _safe_mean("C3")
    h1 = (not np.isnan(c2m) and c2m > c1m) or (not np.isnan(c3m) and c3m > c1m)
    md_c2, md_c3 = _safe_dim("C2", "mechanistic_depth"), _safe_dim("C3", "mechanistic_depth")
    lg_c2, lg_c3 = _safe_dim("C2", "literature_grounding"), _safe_dim("C3", "literature_grounding")
    h3_spec = (not np.isnan(md_c3)) and (not np.isnan(md_c2)) and md_c3 > md_c2
    h3_grd = (not np.isnan(lg_c3)) and (not np.isnan(lg_c2)) and lg_c3 >= lg_c2

    lines = [
        "LLM-as-Judge Ablation Evaluation",
        "=" * 60,
        f"N cases evaluated: {n_cases}",
        f"Judge model: {provider_label}",
        "Judge key: provided locally, not recorded",
        "",
        "IMPORTANT: When the same API provider generates C2/C3 explanations",
        "and acts as judge, self-preference bias may exist. C1 (rule-based,",
        "no LLM) serves as an unbiased reference for generation. Raw judge",
        "responses saved in llm_judge_raw.csv for manual inspection.",
        "",
        "── Overall scores (mean ± SE) ─────────────────────────────────────",
    ]
    for cond in CONDITIONS:
        subset = scores_df[scores_df["condition"] == cond]["overall"]
        lines.append(
            f"  {cond}: {_format_score(subset.mean())} ± {_format_score(subset.sem())}  "
            f"(n={len(subset)})"
        )

    lines += [
        "",
        "── Dimension means ────────────────────────────────────────────────",
        f"  {'':28s}" + "  ".join(f"{c:>5}" for c in CONDITIONS),
    ]
    for dim in DIMENSIONS:
        row_vals = "  ".join(f"{dim_means.loc[c, dim]:5.2f}" for c in CONDITIONS)
        lines.append(f"  {dim:<28s}{row_vals}")

    lines += [
        "",
        "── Statistical tests (Friedman + Wilcoxon–Holm post-hoc) ─────────",
        tests.to_string(index=False),
        "",
        "── Hypothesis evaluation ──────────────────────────────────────────",
        f"  H1 (contextual modes C2/C3 > C1): {'supported' if h1 else 'not supported'}",
        f"  H3 (C3 > C2 on mechanistic_depth): {'supported' if h3_spec else 'not supported'}"
        f" | C3={md_c3:.2f}, C2={md_c2:.2f}",
        f"  H3 (C3 ≥ C2 on literature_grounding): {'supported' if h3_grd else 'not supported'}"
        f" | C3={lg_c3:.2f}, C2={lg_c2:.2f}",
        f"  Dims with significant C3>C0 (p_holm<0.05): {sig_c3c0 or 'none'}",
        f"  Dims with significant C3>C2 (p_holm<0.05): {sig_c3c1 or 'none'}",
    ]
    return "\n".join(lines)


# ── Main evaluation ───────────────────────────────────────────────────────────

def run_llm_judge(n_cases: int = N_CASES) -> dict[str, Any]:
    """Run LLM-as-Judge evaluation and return result dict.

    Parameters
    ----------
    n_cases:
        Number of diverse cases to evaluate (default 10).

    Returns
    -------
    dict with keys: scores, raw_responses, tests, paths
    """
    _ensure_dirs()

    # ── Load & sample cases ───────────────────────────────────────────────────
    if not TEST_CASES_FILE.exists():
        raise FileNotFoundError(f"Missing: {TEST_CASES_FILE}")
    all_cases = pd.read_csv(TEST_CASES_FILE)
    cases = _sample_diverse_cases(all_cases, n_cases).reset_index(drop=True)
    n_actual = len(cases)
    print(f"Selected {n_actual} diverse cases for LLM-as-Judge evaluation.")
    provider, api_key = resolve_provider()
    provider_label = {
        "openai": "GPT-4o-mini",
        None: "None",
    }[provider]
    print(f"Judge provider: {provider_label} | key={mask_key(api_key)}")

    # Build input dicts from columns or pre-built 'input_dict' column
    def _row_to_input(row: pd.Series) -> dict[str, Any]:
        if "input_dict" in row and pd.notna(row["input_dict"]):
            try:
                return json.loads(str(row["input_dict"]))
            except json.JSONDecodeError:
                pass
        return {
            "ReT": str(row["input_ReT"]),
            "AgS": str(row["input_AgS"]),
            "AdC": str(row["input_AdC"]),
            "Temp": float(row["input_Temp"]),
            "pH": float(row["input_pH"]),
            "rpm": float(row["input_rpm"]),
            "Ce": float(row["input_Ce"]),
        }

    # ── Phase 1: generate explanations ───────────────────────────────────────
    print("\nPhase 1: Generating explanations for C0–C3 …")
    generated: list[dict[str, Any]] = []
    for case_idx, (_, row) in enumerate(cases.iterrows()):
        input_dict = _row_to_input(row)
        pred = predict_single(input_dict)
        for condition in CONDITIONS:
            print(f"  Case {case_idx+1}/{n_actual} | {condition} … ", end="", flush=True)
            try:
                html_text, sources = _generate_explanation(input_dict, condition)
                status = "ok"
            except Exception as exc:
                html_text = f"Generation error: {exc}"
                sources = []
                status = f"error: {exc}"
            generated.append({
                "case_id": case_idx,
                "condition": condition,
                "input_dict": json.dumps(input_dict),
                "html_text": html_text,
                "n_sources": len(sources),
                "status": status,
            })
            print(status)
            if condition != "C0":  # C0 has no LLM call
                time.sleep(API_DELAY_S)

    gen_df = pd.DataFrame(generated)

    # ── Phase 2: judge explanations ───────────────────────────────────────────
    print("\nPhase 2: LLM judge scoring …")
    score_rows: list[dict[str, Any]] = []
    raw_rows: list[dict[str, Any]] = []

    for _, gen_row in gen_df.iterrows():
        case_idx = int(gen_row["case_id"])
        condition = str(gen_row["condition"])
        input_dict = json.loads(str(gen_row["input_dict"]))
        pred = predict_single(input_dict)
        print(f"  Case {case_idx+1}/{n_actual} | {condition} judging … ", end="", flush=True)

        try:
            scored = _judge_explanation(input_dict, pred, str(gen_row["html_text"]), condition)
            raw_rows.append({
                "case_id": case_idx,
                "condition": condition,
                "raw_response": scored.pop("raw_response", ""),
            })
            # Clip scores to [1, 5] and compute weighted overall
            clean: dict[str, Any] = {"case_id": case_idx, "condition": condition}
            for dim in DIMENSIONS:
                val = scored.get(dim)
                clean[dim] = int(np.clip(int(val), 1, 5)) if val is not None else np.nan
            # LLM-reported overall (capped)
            llm_overall = scored.get("overall")
            clean["llm_overall"] = int(np.clip(int(llm_overall), 1, 5)) if llm_overall is not None else np.nan
            # Weighted overall from rubric weights
            if all(clean[d] is not np.nan for d in DIMENSIONS):
                clean["overall"] = sum(DIM_WEIGHTS[d] * clean[d] for d in DIMENSIONS)
            else:
                clean["overall"] = np.nan
            clean["rationale"] = str(scored.get("rationale", ""))
            score_rows.append(clean)
            print("ok")
        except Exception as exc:
            print(f"error: {exc}")
            if _is_auth_error(str(exc)):
                provider, api_key = resolve_provider()
                provider_label = {
                    "openai": "GPT-4o-mini",
                    None: "None",
                }[provider]
                raise RuntimeError(
                    "LLM judge authentication failed. "
                    f"Provider={provider_label}, key={mask_key(api_key)}. "
                    "Update OPENAI_API_KEY or project .env, then rerun."
                ) from exc
            raw_rows.append({"case_id": case_idx, "condition": condition, "raw_response": str(exc)})
            score_rows.append({"case_id": case_idx, "condition": condition,
                                **{d: np.nan for d in DIMENSIONS},
                                "overall": np.nan, "llm_overall": np.nan, "rationale": str(exc)})

        time.sleep(API_DELAY_S)

    scores_df = pd.DataFrame(score_rows)
    raw_df = pd.DataFrame(raw_rows)

    # ── Phase 3: statistics & outputs ─────────────────────────────────────────
    print("\nPhase 3: Statistical tests …")
    tests = _stat_tests(scores_df)

    scores_path = REPORTS_DIR / "llm_judge_scores.csv"
    raw_path = REPORTS_DIR / "llm_judge_raw.csv"
    summary_path = REPORTS_DIR / "llm_judge_summary.txt"
    scores_df.to_csv(scores_path, index=False)
    raw_df.to_csv(raw_path, index=False)

    summary_text = _build_summary(scores_df, tests, n_actual)
    summary_path.write_text(summary_text, encoding="utf-8")
    print(summary_text)

    bar_path = _bar_chart(scores_df, tests)
    radar_path = _radar_chart(scores_df)
    print(f"\nSaved: {scores_path}\n       {raw_path}\n       {summary_path}")
    print(f"       {bar_path}\n       {radar_path}")

    return {
        "scores": scores_df,
        "raw_responses": raw_df,
        "tests": tests,
        "paths": {
            "scores": scores_path,
            "raw": raw_path,
            "summary": summary_path,
            "bar_chart": bar_path,
            "radar_chart": radar_path,
        },
    }


if __name__ == "__main__":
    run_llm_judge()
