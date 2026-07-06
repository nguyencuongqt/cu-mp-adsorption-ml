"""Estimate riverine MP-bound Cu2+ mass globally and for top polluted rivers.

This step combines river microplastic abundance, dissolved Cu concentration,
QRF-predicted adsorption capacity, and the Field Adjustment Factor (FAF) to
estimate annual Cu2+ mass adsorbed onto riverine microplastics. It produces a
Table 2-style river output and a horizontal bar plot for the 20 most
MP-polluted rivers.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import joblib
import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import FuncFormatter
import numpy as np
import pandas as pd

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from config import (  # noqa: E402
    CAT_COLS,
    DATA_DIR,
    FAF_CENTRAL,
    FIGURES_DIR,
    GLOBAL_CU_MG_L,
    GLOBAL_DISCHARGE_M3_YR,
    GLOBAL_MP_MASS_MG_M3,
    MODEL_DIR,
    NUM_COLS,
    PH_SCENARIOS,
    REPORTS_DIR,
    RIVER_DEFAULTS,
)


RIVER_DATA_FILE = DATA_DIR / "river_data.csv"
QRF_MODEL_FILE = MODEL_DIR / "qrf_model.joblib"
FAF_RESULTS_FILE = REPORTS_DIR / "faf_results.csv"
REQUIRED_RIVER_COLS = ["river", "country", "MP_type", "MP_items_m3", "Ce_mg_L", "Q_m3_yr"]
PARTICLE_MASS_G = 0.125e-4
DISSOLVED_CU_FLUX_T_YR = 51_000.0
DISSOLVED_CU_FLUX_KG_YR = DISSOLVED_CU_FLUX_T_YR * 1000.0
DISSOLVED_CU_FLUX_MOL_YR = 8.04e8


def _ensure_output_dirs() -> None:
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)


def _load_river_data(river_file: Path = RIVER_DATA_FILE) -> pd.DataFrame:
    if not river_file.exists():
        raise FileNotFoundError(
            f"River data file not found: {river_file}. Expected columns: {REQUIRED_RIVER_COLS}."
        )

    river_df = pd.read_csv(river_file)
    missing = [col for col in REQUIRED_RIVER_COLS if col not in river_df.columns]
    if missing:
        raise KeyError(f"river_data.csv is missing required columns: {missing}")

    return river_df


def _load_qrf_model(model_file: Path = QRF_MODEL_FILE) -> Any:
    if not model_file.exists():
        raise FileNotFoundError(
            f"QRF model not found: {model_file}. Run src/06_uncertainty_qrf.py first."
        )
    return joblib.load(model_file)


def _load_faf() -> tuple[float, float, float]:
    if not FAF_RESULTS_FILE.exists():
        print(
            f"Warning: {FAF_RESULTS_FILE} not found. Using config.FAF_CENTRAL for "
            "central/low/high FAF; run src/07_faf_calibration.py for calibrated CI."
        )
        return float(FAF_CENTRAL), float(FAF_CENTRAL), float(FAF_CENTRAL)

    faf_df = pd.read_csv(FAF_RESULTS_FILE)
    required = ["faf_central", "faf_ci_low", "faf_ci_high"]
    if all(col in faf_df.columns for col in required):
        row = faf_df.iloc[0]
        return (
            float(row["faf_central"]),
            float(row["faf_ci_low"]),
            float(row["faf_ci_high"]),
        )

    if "faf" in faf_df.columns:
        faf_values = pd.to_numeric(faf_df["faf"], errors="coerce").dropna()
        if len(faf_values) > 0:
            central = float(np.median(faf_values))
            low, high = np.percentile(faf_values, [2.5, 97.5])
            return central, float(low), float(high)

    print("Warning: FAF results did not contain CI columns. Falling back to config.FAF_CENTRAL.")
    return float(FAF_CENTRAL), float(FAF_CENTRAL), float(FAF_CENTRAL)


def _normalize_mp_type(value: Any) -> list[str]:
    if pd.isna(value):
        return ["PP", "PE"]

    text = str(value).strip()
    if not text or text.lower() in {"unknown", "mixed", "na", "nan"}:
        return ["PP", "PE"]

    upper = text.upper()
    for sep in ["/", ",", ";", "+", "|"]:
        upper = upper.replace(sep, " ")
    types = [token.strip() for token in upper.split() if token.strip()]
    return types if types else ["PP", "PE"]


def _prepare_river_inputs(river_df: pd.DataFrame) -> pd.DataFrame:
    df = river_df.copy()
    df["Ce_mg_L"] = pd.to_numeric(df["Ce_mg_L"], errors="coerce").fillna(GLOBAL_CU_MG_L)
    df["MP_items_m3"] = pd.to_numeric(df["MP_items_m3"], errors="coerce").fillna(480.0)
    df["Q_m3_yr"] = pd.to_numeric(df["Q_m3_yr"], errors="coerce")
    df = df.dropna(subset=["Q_m3_yr"]).reset_index(drop=True)

    df["MP_mass_mg_m3"] = df["MP_items_m3"] * PARTICLE_MASS_G * 1000.0

    defaults = dict(RIVER_DEFAULTS)
    for col in ["pH", "Temp", "rpm"]:
        source_col = col if col in df.columns else None
        if source_col:
            df[col] = pd.to_numeric(df[col], errors="coerce").fillna(defaults[col])
        else:
            df[col] = defaults[col]

    for col in ["AgS", "AdC"]:
        if col in df.columns:
            df[col] = df[col].fillna(defaults[col])
        else:
            df[col] = defaults[col]

    return df


def _build_prediction_rows(river_df: pd.DataFrame) -> tuple[pd.DataFrame, np.ndarray]:
    rows: list[dict[str, Any]] = []
    weights: list[float] = []

    for idx, row in river_df.iterrows():
        mp_types = _normalize_mp_type(row["MP_type"])
        weight = 1.0 / len(mp_types)
        for mp_type in mp_types:
            rows.append(
                {
                    "river_index": idx,
                    "AgS": row["AgS"],
                    "ReT": mp_type,
                    "Temp": row["Temp"],
                    "pH": row["pH"],
                    "rpm": row["rpm"],
                    "AdC": row["AdC"],
                    "Ce": row["Ce_mg_L"],
                }
            )
            weights.append(weight)

    pred_df = pd.DataFrame(rows)
    return pred_df[["AgS", "ReT", "Temp", "pH", "rpm", "AdC", "Ce", "river_index"]], np.asarray(weights)


def _predict_median(qrf_model: Any, x: pd.DataFrame) -> np.ndarray:
    preds = np.asarray(qrf_model.predict(x, quantiles=[0.50]))
    if preds.ndim == 2:
        return preds[:, 0].astype(float)
    return preds.astype(float)


def _aggregate_river_qe(
    pred_df: pd.DataFrame,
    weights: np.ndarray,
    qe_lab: np.ndarray,
    n_rivers: int,
) -> np.ndarray:
    qe = np.zeros(n_rivers, dtype=float)
    for river_idx, weight, value in zip(pred_df["river_index"], weights, qe_lab):
        qe[int(river_idx)] += weight * value
    return qe


def _save_river_plot(river_df: pd.DataFrame) -> Path:
    output = FIGURES_DIR / "river_barplot.png"
    plot_df = river_df.sort_values("mass_kg_yr", ascending=False).head(20)
    plot_df = plot_df.sort_values("mass_kg_yr", ascending=True)

    values = pd.to_numeric(plot_df["mass_kg_yr"], errors="coerce").clip(lower=0)
    lows = pd.to_numeric(plot_df.get("mass_low_kg_yr", values), errors="coerce").fillna(values).clip(lower=0)
    highs = pd.to_numeric(plot_df.get("mass_high_kg_yr", values), errors="coerce").fillna(values).clip(lower=0)
    positive = pd.concat([values[values > 0], lows[lows > 0], highs[highs > 0]])
    xmin = float(positive.min() * 0.55) if not positive.empty else 1e-4
    xmax = float(max(values.max(), highs.max()) * 1.8) if values.max() > 0 else 1.0
    plot_values = values.mask(values <= 0, xmin)
    plot_lows = lows.mask(lows <= 0, xmin).clip(upper=plot_values)
    y_pos = np.arange(len(plot_df))
    point_color = "#2f6f9f"
    ci_color = "#c43c39"
    guide_color = "#c8d3df"

    fig, ax = plt.subplots(figsize=(9.2, 7.0))
    labels = plot_df["river"].astype(str)
    ax.hlines(y_pos, xmin, plot_values, color=guide_color, linewidth=2.4, zorder=1)
    xerr = np.vstack([(plot_values - plot_lows).clip(lower=0), (highs - plot_values).clip(lower=0)])
    ax.errorbar(plot_values, y_pos, xerr=xerr, fmt="none", ecolor=ci_color, elinewidth=1.2, capsize=3, zorder=2)
    ax.scatter(plot_values, y_pos, s=46, color=point_color, edgecolor="white", linewidth=0.7, zorder=3)
    for y, value, label_value in zip(y_pos, plot_values, values):
        ax.text(value * 1.08, y, f"{label_value:.3g}", va="center", ha="left", fontsize=8, color=point_color)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(labels)
    ax.set_xscale("log")
    ax.set_xlim(xmin, xmax)
    ax.xaxis.set_major_formatter(FuncFormatter(lambda x, _: f"{x:g}"))
    ax.set_xlabel("Cu2+ adsorbed onto riverine MPs (kg/yr)")
    ax.set_ylabel("River")
    ax.grid(axis="x", which="both", linestyle=":", alpha=0.45)
    legend_handles = [
        Line2D([0], [0], marker="o", color="none", markerfacecolor=point_color, markeredgecolor="white", markersize=8, label="Median estimate"),
        Line2D([0], [0], color=ci_color, marker="|", markersize=9, linewidth=1.4, label="95% CI"),
    ]
    ax.legend(handles=legend_handles, loc="lower right", frameon=True, framealpha=0.95, fontsize=9)
    fig.tight_layout()
    fig.savefig(output, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return output


def _print_benchmarks(global_mass_kg: float, top20_mass_kg: float) -> None:
    pct_dissolved = global_mass_kg / DISSOLVED_CU_FLUX_KG_YR * 100
    top20_pct_global = top20_mass_kg / global_mass_kg * 100 if global_mass_kg > 0 else float("nan")
    global_mass_t = global_mass_kg / 1000.0

    print("Global riverine MP-bound Cu2+ estimate")
    print("=" * 56)
    print(f"Global mass: {global_mass_kg:.6g} kg/yr ({global_mass_t:.6g} t/yr)")
    print(
        f"Percent of total dissolved Cu global flux "
        f"({DISSOLVED_CU_FLUX_T_YR:,.0f} t/yr = {DISSOLVED_CU_FLUX_MOL_YR:.2e} mol/yr): "
        f"{pct_dissolved:.6g}%"
    )
    print(f"Top-20 river contribution vs global: {top20_pct_global:.6g}%")


def run(
    qrf_model: Any | None = None,
    river_df: pd.DataFrame | None = None,
    faf_central: float | None = None,
    faf_ci: tuple[float, float] | None = None,
) -> tuple[float, pd.DataFrame]:
    """Estimate global and top-river Cu2+ mass adsorbed onto MPs.

    Parameters
    ----------
    qrf_model:
        Optional saved QRF pipeline. If omitted, ``MODEL_DIR/qrf_model.joblib``
        is loaded.
    river_df:
        Optional river data frame. If omitted, ``data/river_data.csv`` is read.
    faf_central, faf_ci:
        Optional FAF central estimate and CI. If omitted, the script loads
        ``outputs/reports/faf_results.csv`` or falls back to config.FAF_CENTRAL.

    Returns
    -------
    tuple
        ``(global_mass_kg, river_df)`` with river-level Table 2-equivalent
        results.
    """
    _ensure_output_dirs()
    qrf_model = qrf_model if qrf_model is not None else _load_qrf_model()
    raw_rivers = river_df.copy() if river_df is not None else _load_river_data()

    if faf_central is None or faf_ci is None:
        loaded_central, loaded_low, loaded_high = _load_faf()
        faf_central = loaded_central if faf_central is None else faf_central
        faf_ci = (loaded_low, loaded_high) if faf_ci is None else faf_ci
    faf_low, faf_high = faf_ci

    rivers = _prepare_river_inputs(raw_rivers)
    pred_df, weights = _build_prediction_rows(rivers)
    qrf_x = pred_df[CAT_COLS + NUM_COLS]
    lab_qe_rows = _predict_median(qrf_model, qrf_x)
    lab_qe = _aggregate_river_qe(pred_df, weights, lab_qe_rows, len(rivers))

    rivers["lab_qe_mg_g"] = lab_qe
    rivers["river_qe_mg_g"] = rivers["lab_qe_mg_g"] * faf_central
    rivers["river_qe_low_mg_g"] = rivers["lab_qe_mg_g"] * faf_low
    rivers["river_qe_high_mg_g"] = rivers["lab_qe_mg_g"] * faf_high

    rivers["mass_kg_yr"] = (
        rivers["MP_mass_mg_m3"] * rivers["river_qe_mg_g"] * rivers["Q_m3_yr"] / 1e9
    )
    rivers["mass_low_kg_yr"] = (
        rivers["MP_mass_mg_m3"] * rivers["river_qe_low_mg_g"] * rivers["Q_m3_yr"] / 1e9
    )
    rivers["mass_high_kg_yr"] = (
        rivers["MP_mass_mg_m3"] * rivers["river_qe_high_mg_g"] * rivers["Q_m3_yr"] / 1e9
    )

    global_x = pd.DataFrame(
        [
            {
                "AgS": RIVER_DEFAULTS["AgS"],
                "ReT": "PP",
                "Temp": RIVER_DEFAULTS["Temp"],
                "pH": RIVER_DEFAULTS.get("pH", PH_SCENARIOS[1]),
                "rpm": RIVER_DEFAULTS["rpm"],
                "AdC": RIVER_DEFAULTS["AdC"],
                "Ce": GLOBAL_CU_MG_L,
            },
            {
                "AgS": RIVER_DEFAULTS["AgS"],
                "ReT": "PE",
                "Temp": RIVER_DEFAULTS["Temp"],
                "pH": RIVER_DEFAULTS.get("pH", PH_SCENARIOS[1]),
                "rpm": RIVER_DEFAULTS["rpm"],
                "AdC": RIVER_DEFAULTS["AdC"],
                "Ce": GLOBAL_CU_MG_L,
            },
        ]
    )
    global_lab_qe = float(np.mean(_predict_median(qrf_model, global_x[CAT_COLS + NUM_COLS])))
    global_qe = global_lab_qe * faf_central
    global_mass_kg = GLOBAL_DISCHARGE_M3_YR * GLOBAL_MP_MASS_MG_M3 * global_qe / 1e9
    top20_mass_kg = float(rivers.sort_values("mass_kg_yr", ascending=False).head(20)["mass_kg_yr"].sum())

    rivers["faf_central"] = faf_central
    rivers["faf_ci_low"] = faf_low
    rivers["faf_ci_high"] = faf_high
    rivers["global_qe_mg_g"] = global_qe
    rivers["global_mass_kg_yr"] = global_mass_kg

    output_csv = REPORTS_DIR / "river_results.csv"
    rivers.to_csv(output_csv, index=False)
    figure_path = _save_river_plot(rivers)

    _print_benchmarks(global_mass_kg, top20_mass_kg)
    print(f"Saved river results: {output_csv}")
    print(f"Saved river bar plot: {figure_path}")

    return global_mass_kg, rivers


if __name__ == "__main__":
    run()
