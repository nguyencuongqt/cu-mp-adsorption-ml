"""Train a Quantile Random Forest and evaluate prediction intervals.

The ``run`` function accepts raw train/test predictors, fits a
``RandomForestQuantileRegressor`` from the ``quantile-forest`` package, and
evaluates both point prediction quality and 90% prediction interval quality.
The fitted object saved to disk is a full sklearn Pipeline, so the same raw
project columns can be supplied when the model is reused.
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
from quantile_forest import RandomForestQuantileRegressor
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

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
    FIGURES_DIR,
    MODEL_DIR,
    NUM_COLS,
    QRF_LOWER,
    QRF_UPPER,
    RANDOM_SEED,
    RF_PARAMS,
    TARGET,
    TEST_SIZE,
)
from src.analysis.data_cleaning import clean_model_data  # noqa: E402


QRF_QUANTILES = [QRF_LOWER, 0.50, QRF_UPPER]
PAPER_TARGETS = {
    "mae": 0.319,
    "r2": 0.807,
    "coverage": 0.894,
    "niw": 0.220,
}


def _ensure_output_dirs() -> None:
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    MODEL_DIR.mkdir(parents=True, exist_ok=True)


def _load_raw_dataset() -> pd.DataFrame:
    data_file = Path(DATA_FILE)
    if not data_file.exists():
        raise FileNotFoundError(f"DATA_FILE does not exist: {data_file}")

    if data_file.suffix.lower() in {".xlsx", ".xls"}:
        return pd.read_excel(data_file)
    if data_file.suffix.lower() == ".csv":
        return pd.read_csv(data_file)

    raise ValueError(f"Unsupported DATA_FILE extension: {data_file.suffix}")


def _make_one_hot_encoder() -> OneHotEncoder:
    try:
        return OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    except TypeError:
        return OneHotEncoder(handle_unknown="ignore", sparse=False)


def _present_columns(x: pd.DataFrame, requested: list[str]) -> list[str]:
    return [col for col in requested if col in x.columns]


def _make_preprocessor(x: pd.DataFrame) -> ColumnTransformer:
    cat_cols = _present_columns(x, CAT_COLS)
    num_cols = _present_columns(x, NUM_COLS)
    extra_numeric = [
        col
        for col in x.columns
        if col not in cat_cols + num_cols and pd.api.types.is_numeric_dtype(x[col])
    ]
    num_cols = num_cols + extra_numeric

    transformers: list[tuple[str, Any, Any]] = []
    if num_cols:
        transformers.append(("num", SimpleImputer(strategy="median"), num_cols))
    if cat_cols:
        cat_pipe = Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="most_frequent")),
                ("onehot", _make_one_hot_encoder()),
            ]
        )
        transformers.append(("cat", cat_pipe, cat_cols))

    if not transformers:
        raise ValueError("No usable predictor columns found in X_train.")

    return ColumnTransformer(
        transformers=transformers,
        remainder="drop",
        verbose_feature_names_out=False,
    )


def _make_qrf_pipeline(x_train: pd.DataFrame) -> Pipeline:
    params = dict(RF_PARAMS)
    params.update(
        {
            "default_quantiles": QRF_QUANTILES,
            "random_state": RANDOM_SEED,
            "n_jobs": -1,
        }
    )
    return Pipeline(
        steps=[
            ("preprocess", _make_preprocessor(x_train)),
            ("model", RandomForestQuantileRegressor(**params)),
        ]
    )


def _prepare_inputs(
    x_train: pd.DataFrame,
    y_train: pd.Series | np.ndarray,
    x_test: pd.DataFrame,
    y_test: pd.Series | np.ndarray,
) -> tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series]:
    x_train_df = pd.DataFrame(x_train).reset_index(drop=True)
    x_test_df = pd.DataFrame(x_test).reset_index(drop=True)
    y_train_s = pd.Series(y_train, name=TARGET).reset_index(drop=True)
    y_test_s = pd.Series(y_test, name=TARGET).reset_index(drop=True)

    y_train_s = pd.to_numeric(y_train_s, errors="coerce")
    y_test_s = pd.to_numeric(y_test_s, errors="coerce")
    train_keep = y_train_s.notna()
    test_keep = y_test_s.notna()

    return (
        x_train_df.loc[train_keep].reset_index(drop=True),
        y_train_s.loc[train_keep].reset_index(drop=True),
        x_test_df.loc[test_keep].reset_index(drop=True),
        y_test_s.loc[test_keep].reset_index(drop=True),
    )


def _predict_quantiles(qrf_model: Pipeline, x_test: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    preds = np.asarray(qrf_model.predict(x_test, quantiles=QRF_QUANTILES))
    if preds.ndim == 1:
        raise ValueError("QRF returned one prediction column; expected q05, q50, q95.")
    lower = preds[:, 0].astype(float)
    median = preds[:, 1].astype(float)
    upper = preds[:, 2].astype(float)
    return lower, median, upper


def _interval_score(
    y_true: np.ndarray,
    lower: np.ndarray,
    upper: np.ndarray,
    alpha: float = 0.10,
) -> float:
    width = upper - lower
    lower_penalty = (2 / alpha) * (lower - y_true) * (y_true < lower)
    upper_penalty = (2 / alpha) * (y_true - upper) * (y_true > upper)
    return float(np.mean(width + lower_penalty + upper_penalty))


def _compute_metrics(
    y_train: pd.Series,
    y_test: pd.Series,
    pred_median: np.ndarray,
    pred_lower: np.ndarray,
    pred_upper: np.ndarray,
) -> dict[str, float]:
    y_test_arr = y_test.to_numpy(dtype=float)
    y_range = float(y_train.max() - y_train.min())
    if y_range <= 0:
        y_range = float(y_test.max() - y_test.min())

    mae = float(mean_absolute_error(y_test_arr, pred_median))
    metrics = {
        "mae": mae,
        "nmae": float(mae / y_range) if y_range > 0 else float("nan"),
        "r2": float(r2_score(y_test_arr, pred_median)),
        "coverage": float(np.mean((y_test_arr >= pred_lower) & (y_test_arr <= pred_upper))),
        "niw": float(np.mean(pred_upper - pred_lower) / y_range) if y_range > 0 else float("nan"),
        "interval_score": _interval_score(y_test_arr, pred_lower, pred_upper),
        "y_range": y_range,
    }
    return metrics


def _save_prediction_plot(
    y_test: pd.Series,
    pred_median: np.ndarray,
    pred_lower: np.ndarray,
    pred_upper: np.ndarray,
    metrics: dict[str, float] | None = None,
) -> Path:
    output = FIGURES_DIR / "qrf_predictions.png"
    observed = y_test.to_numpy(dtype=float)
    residual = pred_median - observed

    yerr = np.vstack(
        [
            np.maximum(pred_median - pred_lower, 0),
            np.maximum(pred_upper - pred_median, 0),
        ]
    )

    fig, ax = plt.subplots(figsize=(9.0, 7.8))
    ax.errorbar(
        observed,
        pred_median,
        yerr=yerr,
        fmt="none",
        ecolor="0.70",
        elinewidth=0.8,
        alpha=0.55,
        zorder=1,
    )
    scatter = ax.scatter(
        observed,
        pred_median,
        c=residual,
        cmap="bwr_r",
        s=38,
        edgecolor="black",
        linewidth=0.25,
        alpha=0.92,
        zorder=2,
    )

    min_axis = float(min(observed.min(), pred_lower.min(), pred_median.min()))
    max_axis = float(max(observed.max(), pred_upper.max(), pred_median.max()))
    pad = (max_axis - min_axis) * 0.05 if max_axis > min_axis else 0.1
    ax.plot([min_axis - pad, max_axis + pad], [min_axis - pad, max_axis + pad], color="black", linewidth=1.0)

    ax.set_xlim(min_axis - pad, max_axis + pad)
    ax.set_ylim(min_axis - pad, max_axis + pad)
    ax.set_xlabel("Observed qe (mg/g)", fontsize=20)
    ax.set_ylabel("Predicted qe, QRF median (mg/g)", fontsize=20)
    ax.tick_params(axis="both", labelsize=16)

    if metrics is not None:
        stats_text = (
            f"QRF median MAE = {metrics['mae']:.3f} mg/g\n"
            f"90% PI coverage = {metrics['coverage'] * 100:.1f}%\n"
            f"NIW = {metrics['niw']:.3f}"
        )
        ax.text(
            0.04,
            0.96,
            stats_text,
            transform=ax.transAxes,
            ha="left",
            va="top",
            fontsize=16,
            bbox={"boxstyle": "round,pad=0.35", "facecolor": "white", "edgecolor": "0.35", "alpha": 0.92},
        )

    colorbar = fig.colorbar(scatter, ax=ax)
    colorbar.set_label("Residual: predicted - observed\nblue=overestimate, red=underestimate", fontsize=16)
    colorbar.ax.tick_params(labelsize=15)
    fig.tight_layout()
    fig.savefig(output, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return output


def _print_metrics(metrics: dict[str, float]) -> None:
    print("Quantile Random Forest metrics")
    print("=" * 48)
    print(f"MAE: {metrics['mae']:.4f} | paper target ≈ {PAPER_TARGETS['mae']:.3f}")
    print(f"nMAE: {metrics['nmae']:.4f}")
    print(f"R²: {metrics['r2']:.4f} | paper target ≈ {PAPER_TARGETS['r2']:.3f}")
    print(
        f"Coverage: {metrics['coverage'] * 100:.2f}% "
        f"| paper target ≈ {PAPER_TARGETS['coverage'] * 100:.1f}%"
    )
    print(f"NIW: {metrics['niw']:.4f} | paper target ≈ {PAPER_TARGETS['niw']:.3f}")
    print(f"Interval Score: {metrics['interval_score']:.4f}")
    print("")
    print("Difference from paper targets")
    print(f"ΔMAE = {metrics['mae'] - PAPER_TARGETS['mae']:+.4f}")
    print(f"ΔR² = {metrics['r2'] - PAPER_TARGETS['r2']:+.4f}")
    print(f"ΔCoverage = {(metrics['coverage'] - PAPER_TARGETS['coverage']) * 100:+.2f} percentage points")
    print(f"ΔNIW = {metrics['niw'] - PAPER_TARGETS['niw']:+.4f}")


def _stratified_bins(y: pd.Series, n_splits: int = 5) -> pd.Series:
    max_bins = min(10, y.nunique(), len(y) // n_splits)
    for n_bins in range(max_bins, 1, -1):
        bins = pd.qcut(y, q=n_bins, labels=False, duplicates="drop")
        counts = pd.Series(bins).value_counts(dropna=False)
        if counts.min() >= n_splits:
            return bins.astype(int)
    return pd.Series(np.zeros(len(y), dtype=int))


def run(
    X_train: pd.DataFrame,
    y_train: pd.Series | np.ndarray,
    X_test: pd.DataFrame,
    y_test: pd.Series | np.ndarray,
) -> tuple[Pipeline, np.ndarray, np.ndarray, np.ndarray, dict[str, float]]:
    """Train QRF and evaluate point and interval predictions.

    Parameters
    ----------
    X_train, y_train, X_test, y_test:
        Raw training and held-out test data. Predictors may contain missing
        numeric values and categorical columns; preprocessing is fit on the
        training set only inside the saved model pipeline.

    Returns
    -------
    tuple
        ``(qrf_model, y_pred_median, y_pred_lower, y_pred_upper, metrics_dict)``.
    """
    _ensure_output_dirs()
    x_train, y_train_s, x_test, y_test_s = _prepare_inputs(X_train, y_train, X_test, y_test)

    qrf_model = _make_qrf_pipeline(x_train)
    print("Training Quantile Random Forest...")
    qrf_model.fit(x_train, y_train_s)

    y_pred_lower, y_pred_median, y_pred_upper = _predict_quantiles(qrf_model, x_test)
    metrics = _compute_metrics(y_train_s, y_test_s, y_pred_median, y_pred_lower, y_pred_upper)

    _print_metrics(metrics)
    figure_path = _save_prediction_plot(y_test_s, y_pred_median, y_pred_lower, y_pred_upper, metrics=metrics)
    model_path = MODEL_DIR / "qrf_model.joblib"
    joblib.dump(qrf_model, model_path)

    print(f"Saved QRF prediction plot: {figure_path}")
    print(f"Saved QRF model: {model_path}")

    return qrf_model, y_pred_median, y_pred_lower, y_pred_upper, metrics


if __name__ == "__main__":
    raw_df = clean_model_data(_load_raw_dataset())
    feature_cols = [col for col in CAT_COLS + NUM_COLS if col in raw_df.columns]
    y = raw_df[TARGET]
    bins = _stratified_bins(y, n_splits=5)
    X_train_main, X_test_main, y_train_main, y_test_main = train_test_split(
        raw_df[feature_cols],
        y,
        test_size=TEST_SIZE,
        random_state=RANDOM_SEED,
        stratify=bins,
    )
    run(X_train_main, y_train_main, X_test_main, y_test_main)
