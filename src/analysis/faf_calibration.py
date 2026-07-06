"""Calibrate the Field Adjustment Factor (FAF) for river conditions.

The FAF converts laboratory-scale QRF adsorption predictions into field-scale
river estimates. It is derived from 14 calibration rows manually transcribed
from Table S1 into ``outputs/reports/field_data.csv``. For each row, the saved
QRF model predicts laboratory-style qe, and the observed field qe is divided by
that modeled value. The median ratio is used as the central FAF, with a
bootstrap confidence interval propagated to river-adjusted QRF predictions.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import joblib
import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from config import (  # noqa: E402
    CAT_COLS,
    DATA_FILE,
    FAF_BOOTSTRAP_ITERS,
    FIGURES_DIR,
    MODEL_DIR,
    NUM_COLS,
    QRF_LOWER,
    QRF_UPPER,
    RANDOM_SEED,
    REPORTS_DIR,
    TARGET,
    TEST_SIZE,
)
from src.analysis.data_cleaning import clean_model_data  # noqa: E402


FIELD_FILE = REPORTS_DIR / "field_data.csv"
QRF_MODEL_FILE = MODEL_DIR / "qrf_model.joblib"
QRF_QUANTILES = [QRF_LOWER, 0.50, QRF_UPPER]
FIELD_REQUIRED_COLS = CAT_COLS + NUM_COLS + ["real_qe"]


def _ensure_output_dirs() -> None:
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)


def _load_raw_dataset() -> pd.DataFrame:
    data_file = Path(DATA_FILE)
    if not data_file.exists():
        raise FileNotFoundError(f"DATA_FILE does not exist: {data_file}")

    if data_file.suffix.lower() in {".xlsx", ".xls"}:
        return pd.read_excel(data_file)
    if data_file.suffix.lower() == ".csv":
        return pd.read_csv(data_file)

    raise ValueError(f"Unsupported DATA_FILE extension: {data_file.suffix}")


def _load_field_data(field_file: Path = FIELD_FILE) -> pd.DataFrame:
    if not field_file.exists():
        raise FileNotFoundError(
            f"Field calibration file not found: {field_file}. "
            "Create outputs/reports/field_data.csv from Table S1 with columns "
            f"{FIELD_REQUIRED_COLS}."
        )

    field_df = pd.read_csv(field_file)
    missing = [col for col in FIELD_REQUIRED_COLS if col not in field_df.columns]
    if missing:
        raise KeyError(f"field_data.csv is missing required columns: {missing}")

    if len(field_df) != 14:
        print(f"Warning: expected 14 field calibration rows, found {len(field_df)}.")

    for col in NUM_COLS + ["real_qe"]:
        field_df[col] = pd.to_numeric(field_df[col], errors="coerce")

    field_df = field_df.dropna(subset=["real_qe"]).reset_index(drop=True)
    if field_df.empty:
        raise ValueError("No usable field rows remain after dropping missing real_qe.")

    return field_df


def _load_qrf_model(model_file: Path = QRF_MODEL_FILE) -> Any:
    if not model_file.exists():
        raise FileNotFoundError(
            f"QRF model not found: {model_file}. Run src/06_uncertainty_qrf.py first."
        )
    return joblib.load(model_file)


def _predict_quantiles(qrf_model: Any, x: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    preds = np.asarray(qrf_model.predict(x, quantiles=QRF_QUANTILES))
    if preds.ndim == 1:
        raise ValueError("QRF returned one prediction column; expected q05, q50, q95.")
    lower = preds[:, 0].astype(float)
    median = preds[:, 1].astype(float)
    upper = preds[:, 2].astype(float)
    return lower, median, upper


def _bootstrap_faf(
    faf_values: np.ndarray,
    n_bootstrap: int = FAF_BOOTSTRAP_ITERS,
) -> tuple[np.ndarray, float, float]:
    rng = np.random.default_rng(RANDOM_SEED)
    boot = np.empty(n_bootstrap, dtype=float)
    for i in range(n_bootstrap):
        sample = rng.choice(faf_values, size=len(faf_values), replace=True)
        boot[i] = np.median(sample)
    ci_low, ci_high = np.percentile(boot, [2.5, 97.5])
    return boot, float(ci_low), float(ci_high)


def _save_bootstrap_plot(
    boot: np.ndarray,
    faf_central: float,
    ci_low: float,
    ci_high: float,
) -> Path:
    output = FIGURES_DIR / "faf_bootstrap.png"
    fig, ax = plt.subplots(figsize=(7.2, 4.8))
    ax.hist(boot, bins=35, color="#4c78a8", edgecolor="white", alpha=0.86)
    ax.axvline(faf_central, color="black", linewidth=1.5, label="Median FAF")
    ax.axvline(ci_low, color="#f58518", linestyle="--", linewidth=1.2, label="95% CI")
    ax.axvline(ci_high, color="#f58518", linestyle="--", linewidth=1.2)
    ax.set_xlabel("Bootstrap median FAF")
    ax.set_ylabel("Frequency")
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(output, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return output


def _save_results(
    field_df: pd.DataFrame,
    qe_modeled: np.ndarray,
    faf_values: np.ndarray,
    faf_central: float,
    ci_low: float,
    ci_high: float,
) -> Path:
    results = field_df.copy()
    results["qe_modeled"] = qe_modeled
    results["faf"] = faf_values
    results["faf_central"] = faf_central
    results["faf_ci_low"] = ci_low
    results["faf_ci_high"] = ci_high
    output = REPORTS_DIR / "faf_results.csv"
    results.to_csv(output, index=False)
    return output


def _stratified_bins(y: pd.Series, n_splits: int = 5) -> pd.Series:
    max_bins = min(10, y.nunique(), len(y) // n_splits)
    for n_bins in range(max_bins, 1, -1):
        bins = pd.qcut(y, q=n_bins, labels=False, duplicates="drop")
        counts = pd.Series(bins).value_counts(dropna=False)
        if counts.min() >= n_splits:
            return bins.astype(int)
    return pd.Series(np.zeros(len(y), dtype=int))


def run(
    X_test: pd.DataFrame | None = None,
    y_pred_median: np.ndarray | None = None,
    y_pred_lower: np.ndarray | None = None,
    y_pred_upper: np.ndarray | None = None,
    qrf_model: Any | None = None,
    field_df: pd.DataFrame | None = None,
) -> tuple[float, tuple[float, float], dict[str, np.ndarray]]:
    """Derive FAF and apply it to QRF test predictions.

    Parameters
    ----------
    X_test:
        Raw held-out test predictors. Required when QRF prediction arrays are
        not supplied.
    y_pred_median, y_pred_lower, y_pred_upper:
        Optional QRF predictions from ``06_uncertainty_qrf.run``. If omitted,
        they are generated from ``qrf_model.predict(X_test)``.
    qrf_model:
        Optional loaded QRF pipeline. If omitted, ``MODEL_DIR/qrf_model.joblib``
        is loaded.
    field_df:
        Optional field calibration data. If omitted,
        ``outputs/reports/field_data.csv`` is loaded.

    Returns
    -------
    tuple
        ``(faf_central, (faf_ci_low, faf_ci_high), river_qe_predictions)`` where
        ``river_qe_predictions`` contains ``river_qe``, ``river_qe_lower``, and
        ``river_qe_upper`` arrays.
    """
    _ensure_output_dirs()
    qrf_model = qrf_model if qrf_model is not None else _load_qrf_model()
    field_df = field_df.copy() if field_df is not None else _load_field_data()

    field_x = field_df[CAT_COLS + NUM_COLS]
    _, qe_modeled, _ = _predict_quantiles(qrf_model, field_x)
    valid = np.isfinite(qe_modeled) & (qe_modeled > 0) & field_df["real_qe"].notna().to_numpy()
    if not valid.any():
        raise ValueError("No valid field rows with positive modeled qe for FAF calculation.")

    field_valid = field_df.loc[valid].reset_index(drop=True)
    qe_modeled_valid = qe_modeled[valid]
    faf_values = field_valid["real_qe"].to_numpy(dtype=float) / qe_modeled_valid
    faf_values = faf_values[np.isfinite(faf_values)]
    if len(faf_values) == 0:
        raise ValueError("FAF calculation produced no finite values.")

    faf_central = float(np.median(faf_values))
    boot, ci_low, ci_high = _bootstrap_faf(faf_values)

    if y_pred_median is None or y_pred_lower is None or y_pred_upper is None:
        if X_test is None:
            raise ValueError("Provide X_test or all QRF prediction arrays to apply FAF.")
        y_pred_lower, y_pred_median, y_pred_upper = _predict_quantiles(qrf_model, pd.DataFrame(X_test))
    else:
        y_pred_median = np.asarray(y_pred_median, dtype=float)
        y_pred_lower = np.asarray(y_pred_lower, dtype=float)
        y_pred_upper = np.asarray(y_pred_upper, dtype=float)

    river_predictions = {
        "river_qe": y_pred_median * faf_central,
        "river_qe_lower": y_pred_lower * ci_low,
        "river_qe_upper": y_pred_upper * ci_high,
    }

    field_pct = faf_central * 100
    print(f"FAF = {faf_central:.6g} (95% CI: {ci_low:.6g}–{ci_high:.6g}). Field qe ≈ {field_pct:.2f}% of lab prediction.")

    results_path = _save_results(
        field_valid,
        qe_modeled_valid,
        faf_values,
        faf_central,
        ci_low,
        ci_high,
    )
    figure_path = _save_bootstrap_plot(boot, faf_central, ci_low, ci_high)
    print(f"Saved FAF results: {results_path}")
    print(f"Saved FAF bootstrap plot: {figure_path}")

    return faf_central, (ci_low, ci_high), river_predictions


if __name__ == "__main__":
    raw_df = clean_model_data(_load_raw_dataset())
    feature_cols = [col for col in CAT_COLS + NUM_COLS if col in raw_df.columns]
    y = raw_df[TARGET]
    bins = _stratified_bins(y, n_splits=5)
    _, x_test_main, _, _ = train_test_split(
        raw_df[feature_cols],
        y,
        test_size=TEST_SIZE,
        random_state=RANDOM_SEED,
        stratify=bins,
    )
    run(X_test=x_test_main)
