"""Explain the best Random Forest qe model with SHAP.

This module produces the SHAP interpretation figures and summary table for the
Cu2+ adsorption capacity model. It accepts a fitted Random Forest model or a
sklearn Pipeline containing preprocessing plus a RandomForestRegressor. When
the model contains one-hot encoded categorical predictors, dummy-level SHAP
values are collapsed back to the original project variables ``ReT``, ``AgS``,
and ``AdC`` so global importance is reported at the scientific-variable level.

The module also estimates the Ce50 threshold: the Ce concentration where the
SHAP contribution of Ce crosses from non-positive to positive. This mirrors the
interpretive threshold analysis in the manuscript workflow and should be run on
the held-out test set.
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
import shap
from scipy.stats import ranksums
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

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
    RANDOM_SEED,
    REPORTS_DIR,
    TARGET,
    TEST_SIZE,
)
from src.analysis.data_cleaning import clean_model_data  # noqa: E402


CE_COL = "Ce"
BOOTSTRAP_ITERS = 1000


def _ensure_output_dirs() -> None:
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
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


def _load_rf_model() -> Any:
    """Load the trained Random Forest model from MODEL_DIR."""
    candidates = [
        MODEL_DIR / "RandomForest_tuned.joblib",
        MODEL_DIR / "RandomForest.joblib",
        MODEL_DIR / "rf_tuned.joblib",
        MODEL_DIR / "best_rf_model.joblib",
    ]
    for path in candidates:
        if path.exists():
            return joblib.load(path)

    raise FileNotFoundError(
        "No trained RF model found in MODEL_DIR. Expected one of: "
        + ", ".join(str(path.name) for path in candidates)
    )


def _as_dataframe(x: pd.DataFrame | np.ndarray, feature_names: list[str] | None) -> pd.DataFrame:
    if isinstance(x, pd.DataFrame):
        return x.reset_index(drop=True).copy()

    if feature_names is None:
        feature_names = [f"x{i}" for i in range(np.asarray(x).shape[1])]
    return pd.DataFrame(x, columns=feature_names).reset_index(drop=True)


def _pipeline_to_rf_inputs(
    model: Any,
    x_train: pd.DataFrame,
    x_test: pd.DataFrame,
    feature_names: list[str] | None,
) -> tuple[Any, pd.DataFrame, pd.DataFrame, list[str]]:
    """Return RF estimator and transformed train/test matrices for SHAP."""
    if isinstance(model, Pipeline):
        if "model" in model.named_steps:
            rf_model = model.named_steps["model"]
            model_step_index = list(model.named_steps).index("model")
            transform_pipeline = Pipeline(model.steps[:model_step_index])
        else:
            rf_model = model.steps[-1][1]
            transform_pipeline = Pipeline(model.steps[:-1])

        if transform_pipeline.steps:
            x_train_arr = transform_pipeline.transform(x_train)
            x_test_arr = transform_pipeline.transform(x_test)
            first_step = transform_pipeline.steps[0][1]
            if hasattr(first_step, "get_feature_names_out"):
                transformed_names = list(first_step.get_feature_names_out())
            elif feature_names is not None:
                transformed_names = feature_names
            else:
                transformed_names = [f"x{i}" for i in range(x_train_arr.shape[1])]
        else:
            x_train_arr = x_train.to_numpy()
            x_test_arr = x_test.to_numpy()
            transformed_names = feature_names or list(x_train.columns)
    else:
        rf_model = model
        x_train_arr = x_train.to_numpy()
        x_test_arr = x_test.to_numpy()
        transformed_names = feature_names or list(x_train.columns)

    transformed_names = [name.replace("num__", "").replace("cat__", "") for name in transformed_names]
    x_train_proc = pd.DataFrame(x_train_arr, columns=transformed_names)
    x_test_proc = pd.DataFrame(x_test_arr, columns=transformed_names)
    return rf_model, x_train_proc, x_test_proc, transformed_names


def _original_feature_name(encoded_name: str) -> str:
    for cat in CAT_COLS:
        if encoded_name == cat or encoded_name.startswith(f"{cat}_") or encoded_name.startswith(f"{cat}="):
            return cat
    return encoded_name


def _collapse_shap(
    shap_values: np.ndarray,
    transformed_names: list[str],
    x_test_raw: pd.DataFrame,
    x_test_proc: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Collapse one-hot SHAP columns back to original variables."""
    shap_df = pd.DataFrame(shap_values, columns=transformed_names)
    groups: dict[str, list[str]] = {}
    for name in transformed_names:
        groups.setdefault(_original_feature_name(name), []).append(name)

    collapsed_values: dict[str, pd.Series] = {}
    collapsed_abs: dict[str, pd.Series] = {}
    collapsed_data: dict[str, pd.Series] = {}

    for feature, cols in groups.items():
        collapsed_values[feature] = shap_df[cols].sum(axis=1)
        collapsed_abs[feature] = shap_df[cols].abs().sum(axis=1)

        if feature in x_test_raw.columns:
            raw_col = x_test_raw[feature].reset_index(drop=True)
            if pd.api.types.is_numeric_dtype(raw_col):
                collapsed_data[feature] = pd.to_numeric(raw_col, errors="coerce")
            else:
                collapsed_data[feature] = pd.Series(pd.Categorical(raw_col).codes, name=feature)
        elif feature in x_test_proc.columns:
            collapsed_data[feature] = x_test_proc[feature].reset_index(drop=True)
        else:
            collapsed_data[feature] = pd.Series(np.nan, index=x_test_raw.index, name=feature)

    collapsed_shap = pd.DataFrame(collapsed_values)
    collapsed_abs_shap = pd.DataFrame(collapsed_abs)
    collapsed_feature_data = pd.DataFrame(collapsed_data)
    return collapsed_shap, collapsed_abs_shap, collapsed_feature_data


def _save_global_importance(summary: pd.DataFrame) -> Path:
    output = FIGURES_DIR / "shap_global.png"
    plot_df = summary.sort_values("mean_abs_shap", ascending=True)

    fig, ax = plt.subplots(figsize=(7.2, 6.2))
    ax.barh(plot_df["feature"], plot_df["pct"], color="#4c78a8")
    ax.set_xlabel("Contribution to total mean |SHAP| (%)", fontsize=20)
    ax.set_ylabel("Feature", fontsize=20)
    ax.tick_params(axis="both", labelsize=18)
    ax.grid(axis="x", linestyle=":", alpha=0.45)
    fig.tight_layout()
    fig.savefig(output, dpi=600, bbox_inches="tight")
    plt.close(fig)
    return output


def _save_beeswarm(
    collapsed_shap: pd.DataFrame,
    collapsed_feature_data: pd.DataFrame,
    summary: pd.DataFrame,
) -> Path:
    output = FIGURES_DIR / "shap_beeswarm.png"
    ordered_features = summary["feature"].tolist()

    explanation = shap.Explanation(
        values=collapsed_shap[ordered_features].to_numpy(),
        data=collapsed_feature_data[ordered_features].to_numpy(),
        feature_names=ordered_features,
    )
    with plt.rc_context({"font.size": 18, "axes.labelsize": 20, "xtick.labelsize": 18, "ytick.labelsize": 18}):
        shap.plots.beeswarm(explanation, max_display=len(ordered_features), show=False)
        plt.gcf().set_size_inches(8.2, 6.4)
        ax = plt.gca()
        ax.tick_params(axis="both", labelsize=18)
        ax.xaxis.label.set_size(20)
        plt.tight_layout()
        plt.savefig(output, dpi=600, bbox_inches="tight")
        plt.close()
    return output


def _fit_ce50(log_ce: np.ndarray, shap_pos: np.ndarray) -> float:
    """Fit logistic regression and return Ce50 in original mg/L units."""
    if len(np.unique(shap_pos)) < 2:
        return float("nan")

    clf = LogisticRegression(random_state=RANDOM_SEED)
    clf.fit(log_ce.reshape(-1, 1), shap_pos)
    intercept = float(clf.intercept_[0])
    coef = float(clf.coef_[0][0])
    if np.isclose(coef, 0):
        return float("nan")
    log_ce50 = -intercept / coef
    if log_ce50 > np.log(np.finfo(float).max) or log_ce50 < np.log(np.finfo(float).tiny):
        return float("nan")
    return float(np.exp(log_ce50))


def _bootstrap_ce50(
    ce: pd.Series,
    ce_shap: pd.Series,
    n_bootstrap: int = BOOTSTRAP_ITERS,
) -> tuple[float, float, float]:
    valid = ce.notna() & ce_shap.notna() & (ce > 0)
    ce_values = ce.loc[valid].to_numpy(dtype=float)
    shap_values = ce_shap.loc[valid].to_numpy(dtype=float)
    shap_pos = (shap_values > 0).astype(int)

    if len(ce_values) < 10 or len(np.unique(shap_pos)) < 2:
        return float("nan"), float("nan"), float("nan")

    log_ce = np.log(ce_values)
    ce50 = _fit_ce50(log_ce, shap_pos)

    rng = np.random.default_rng(RANDOM_SEED)
    boot_values: list[float] = []
    for _ in range(n_bootstrap):
        idx = rng.integers(0, len(ce_values), len(ce_values))
        if len(np.unique(shap_pos[idx])) < 2:
            continue
        boot_ce50 = _fit_ce50(log_ce[idx], shap_pos[idx])
        if np.isfinite(boot_ce50):
            boot_values.append(boot_ce50)

    if not boot_values:
        return ce50, float("nan"), float("nan")

    ci_low, ci_high = np.percentile(boot_values, [2.5, 97.5])
    return ce50, float(ci_low), float(ci_high)


def _wilcoxon_ce(ce: pd.Series, ce_shap: pd.Series) -> tuple[float, float]:
    valid = ce.notna() & ce_shap.notna()
    ce_clean = ce.loc[valid]
    shap_clean = ce_shap.loc[valid]
    ce_positive = ce_clean.loc[shap_clean > 0]
    ce_nonpositive = ce_clean.loc[shap_clean <= 0]

    if len(ce_positive) == 0 or len(ce_nonpositive) == 0:
        return float("nan"), float("nan")

    try:
        stat, p_value = ranksums(ce_positive.to_numpy(), ce_nonpositive.to_numpy())
    except ValueError:
        return float("nan"), float("nan")
    return float(stat), float(p_value)


def run(
    model: Any,
    X_train: pd.DataFrame | np.ndarray,
    X_test: pd.DataFrame | np.ndarray,
    y_test: pd.Series | np.ndarray | None = None,
    feature_names: list[str] | None = None,
) -> pd.DataFrame:
    """Compute SHAP explanations and Ce50 threshold for a fitted RF model.

    Parameters
    ----------
    model:
        A fitted RandomForestRegressor or a fitted sklearn Pipeline whose final
        estimator is the RF model and whose earlier steps transform raw inputs.
    X_train, X_test:
        Training background data and held-out test data. They may be raw data
        frames or already-transformed matrices. If matrices are supplied,
        ``feature_names`` should describe their columns.
    y_test:
        Accepted for pipeline compatibility and future diagnostics; not used by
        the SHAP calculations.
    feature_names:
        Optional feature names for matrix inputs.

    Returns
    -------
    pandas.DataFrame
        Collapsed global SHAP summary with columns ``feature``,
        ``mean_abs_shap``, and ``pct``.
    """
    _ensure_output_dirs()
    x_train_raw = _as_dataframe(X_train, feature_names)
    x_test_raw = _as_dataframe(X_test, feature_names)

    rf_model, x_train_proc, x_test_proc, transformed_names = _pipeline_to_rf_inputs(
        model,
        x_train_raw,
        x_test_raw,
        feature_names,
    )

    explainer = shap.TreeExplainer(rf_model, x_train_proc)
    shap_values = explainer.shap_values(x_test_proc, check_additivity=False)
    shap_array = np.asarray(shap_values)
    if shap_array.ndim == 3:
        shap_array = shap_array[:, :, 0]

    collapsed_shap, collapsed_abs_shap, collapsed_feature_data = _collapse_shap(
        shap_array,
        transformed_names,
        x_test_raw,
        x_test_proc,
    )

    mean_abs = collapsed_abs_shap.mean(axis=0).sort_values(ascending=False)
    total = float(mean_abs.sum())
    pct = mean_abs / total * 100 if total > 0 else mean_abs * np.nan
    summary = pd.DataFrame(
        {
            "feature": mean_abs.index,
            "mean_abs_shap": mean_abs.to_numpy(),
            "pct": pct.to_numpy(),
        }
    )

    summary_path = REPORTS_DIR / "shap_summary.csv"
    summary.to_csv(summary_path, index=False)
    global_path = _save_global_importance(summary)
    beeswarm_path = _save_beeswarm(collapsed_shap, collapsed_feature_data, summary)

    if CE_COL not in collapsed_shap.columns:
        print("Ce50 analysis skipped: Ce SHAP column not found.")
    elif CE_COL not in x_test_raw.columns:
        print("Ce50 analysis skipped: Ce values not found in X_test.")
    else:
        ce50, ci_low, ci_high = _bootstrap_ce50(
            pd.to_numeric(x_test_raw[CE_COL], errors="coerce"),
            collapsed_shap[CE_COL],
        )
        print(f"Ce₅₀ = {ce50:.6g} mg/L (95% CI: {ci_low:.6g}–{ci_high:.6g})")

        w_stat, p_value = _wilcoxon_ce(
            pd.to_numeric(x_test_raw[CE_COL], errors="coerce"),
            collapsed_shap[CE_COL],
        )
        print(f"Wilcoxon test, Ce when SHAP>0 vs SHAP≤0: W = {w_stat:.6g}, p = {p_value:.6g}")

    print(f"Saved SHAP global importance: {global_path}")
    print(f"Saved SHAP beeswarm: {beeswarm_path}")
    print(f"Saved SHAP summary: {summary_path}")
    return summary


if __name__ == "__main__":
    raw_df = clean_model_data(_load_raw_dataset())
    feature_cols = [col for col in CAT_COLS + NUM_COLS if col in raw_df.columns]
    y = raw_df[TARGET]
    bins = pd.qcut(y, q=min(10, y.nunique()), labels=False, duplicates="drop")
    x_train, x_test, _, y_test = train_test_split(
        raw_df[feature_cols],
        y,
        test_size=TEST_SIZE,
        random_state=RANDOM_SEED,
        stratify=bins,
    )
    rf = _load_rf_model()
    run(rf, x_train, x_test, y_test, feature_names=feature_cols)
