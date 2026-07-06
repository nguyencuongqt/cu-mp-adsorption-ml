"""Test robustness of RF results to pH missingness assumptions.

pH has substantial missingness in the Cu adsorption dataset, so this script
compares the original median-imputed pH pipeline with three fixed-pH scenarios
from config.PH_SCENARIOS. Each scenario uses the same train/test split, the
same Random Forest hyperparameters, and the same preprocessing structure. The
goal is to check whether predictive performance and pH SHAP importance are
stable when pH is forced to plausible river-relevant constants.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
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
    NUM_COLS,
    PH_SCENARIOS,
    RANDOM_SEED,
    REPORTS_DIR,
    RF_PARAMS,
    TARGET,
    TEST_SIZE,
)
from src.analysis.data_cleaning import clean_model_data  # noqa: E402


PH_COL = "pH"


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


def _make_one_hot_encoder() -> OneHotEncoder:
    try:
        return OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    except TypeError:
        return OneHotEncoder(handle_unknown="ignore", sparse=False)


def _make_pipeline() -> Pipeline:
    cat_pipe = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", _make_one_hot_encoder()),
        ]
    )
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", SimpleImputer(strategy="median"), NUM_COLS),
            ("cat", cat_pipe, CAT_COLS),
        ],
        remainder="drop",
        verbose_feature_names_out=False,
    )

    params = dict(RF_PARAMS)
    params.update({"random_state": RANDOM_SEED, "n_jobs": -1})
    return Pipeline(
        steps=[
            ("preprocess", preprocessor),
            ("model", RandomForestRegressor(**params)),
        ]
    )


def _stratified_bins(y: pd.Series, n_splits: int = 5) -> pd.Series:
    max_bins = min(10, y.nunique(), len(y) // n_splits)
    for n_bins in range(max_bins, 1, -1):
        bins = pd.qcut(y, q=n_bins, labels=False, duplicates="drop")
        counts = pd.Series(bins).value_counts(dropna=False)
        if counts.min() >= n_splits:
            return bins.astype(int)
    return pd.Series(np.zeros(len(y), dtype=int))


def _prepare_xy(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    required = CAT_COLS + NUM_COLS + [TARGET]
    missing = [col for col in required if col not in df.columns]
    if missing:
        raise KeyError(f"Missing required columns: {missing}")

    clean = df[required].copy()
    clean[TARGET] = pd.to_numeric(clean[TARGET], errors="coerce")
    for col in NUM_COLS:
        clean[col] = pd.to_numeric(clean[col], errors="coerce")
    clean = clean_model_data(clean)
    return clean[CAT_COLS + NUM_COLS], clean[TARGET]


def _force_ph(x_train: pd.DataFrame, x_test: pd.DataFrame, ph_value: float | None) -> tuple[pd.DataFrame, pd.DataFrame]:
    train = x_train.copy()
    test = x_test.copy()
    if ph_value is not None:
        train[PH_COL] = ph_value
        test[PH_COL] = ph_value
    return train, test


def _original_feature_name(encoded_name: str) -> str:
    name = encoded_name.replace("num__", "").replace("cat__", "")
    for cat in CAT_COLS:
        if name == cat or name.startswith(f"{cat}_") or name.startswith(f"{cat}="):
            return cat
    return name


def _collapsed_shap_importance(model: Pipeline, x_train: pd.DataFrame, x_test: pd.DataFrame) -> pd.DataFrame:
    preprocessor = model.named_steps["preprocess"]
    rf = model.named_steps["model"]
    x_train_proc = preprocessor.transform(x_train)
    x_test_proc = preprocessor.transform(x_test)
    feature_names = [
        name.replace("num__", "").replace("cat__", "")
        for name in preprocessor.get_feature_names_out()
    ]

    explainer = shap.TreeExplainer(rf, pd.DataFrame(x_train_proc, columns=feature_names))
    shap_values = explainer.shap_values(
        pd.DataFrame(x_test_proc, columns=feature_names),
        check_additivity=False,
    )
    shap_array = np.asarray(shap_values)
    if shap_array.ndim == 3:
        shap_array = shap_array[:, :, 0]

    shap_df = pd.DataFrame(shap_array, columns=feature_names)
    grouped: dict[str, list[str]] = {}
    for name in feature_names:
        grouped.setdefault(_original_feature_name(name), []).append(name)

    mean_abs = {
        feature: float(shap_df[cols].abs().sum(axis=1).mean())
        for feature, cols in grouped.items()
    }
    summary = pd.DataFrame(
        {
            "feature": list(mean_abs.keys()),
            "mean_abs_shap": list(mean_abs.values()),
        }
    )
    total = float(summary["mean_abs_shap"].sum())
    summary["pct"] = summary["mean_abs_shap"] / total * 100 if total > 0 else np.nan
    summary = summary.sort_values("mean_abs_shap", ascending=False).reset_index(drop=True)
    summary["rank"] = np.arange(1, len(summary) + 1)
    return summary


def _evaluate_scenario(
    scenario: str,
    ph_value: float | None,
    x_train: pd.DataFrame,
    y_train: pd.Series,
    x_test: pd.DataFrame,
    y_test: pd.Series,
) -> dict[str, Any]:
    scenario_x_train, scenario_x_test = _force_ph(x_train, x_test, ph_value)
    model = _make_pipeline()
    model.fit(scenario_x_train, y_train)

    pred = model.predict(scenario_x_test)
    y_range = float(y_train.max() - y_train.min())
    mae = float(mean_absolute_error(y_test, pred))
    nmae = float(mae / y_range) if y_range > 0 else float("nan")
    r2 = float(r2_score(y_test, pred))

    shap_summary = _collapsed_shap_importance(model, scenario_x_train, scenario_x_test)
    ph_row = shap_summary.loc[shap_summary["feature"] == PH_COL]
    if ph_row.empty:
        shap_pct = float("nan")
        shap_rank = float("nan")
    else:
        shap_pct = float(ph_row.iloc[0]["pct"])
        shap_rank = float(ph_row.iloc[0]["rank"])

    return {
        "scenario": scenario,
        "MAE": mae,
        "nMAE": nmae,
        "R2": r2,
        "SHAP_pH_pct": shap_pct,
        "SHAP_pH_rank": shap_rank,
    }


def _save_plot(sensitivity_df: pd.DataFrame) -> Path:
    output = FIGURES_DIR / "ph_sensitivity.png"
    plot_df = sensitivity_df.copy()
    plot_df["scenario_label"] = pd.Categorical(
        plot_df["scenario"],
        categories=["imputed"] + [f"pH={value:g}" for value in PH_SCENARIOS],
        ordered=True,
    )
    plot_df = plot_df.sort_values("scenario_label")

    fig, axes = plt.subplots(1, 2, figsize=(10.0, 4.4))
    x = np.arange(len(plot_df))

    axes[0].plot(x, plot_df["MAE"], marker="o", color="#4c78a8", linewidth=1.8)
    axes[0].set_ylabel("Test MAE (mg/g)")
    axes[0].set_xlabel("Scenario")
    axes[0].grid(axis="y", linestyle=":", alpha=0.45)

    axes[1].plot(x, plot_df["SHAP_pH_pct"], marker="o", color="#f58518", linewidth=1.8)
    axes[1].set_ylabel("pH contribution (% mean |SHAP|)")
    axes[1].set_xlabel("Scenario")
    axes[1].grid(axis="y", linestyle=":", alpha=0.45)

    for ax in axes:
        ax.set_xticks(x)
        ax.set_xticklabels(plot_df["scenario"], rotation=20, ha="right")

    fig.tight_layout()
    fig.savefig(output, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return output


def _print_conclusion(sensitivity_df: pd.DataFrame) -> bool:
    baseline_mae = float(sensitivity_df.loc[sensitivity_df["scenario"] == "imputed", "MAE"].iloc[0])
    max_delta_mae = float((sensitivity_df["MAE"] - baseline_mae).abs().max())
    rank_shift = float(
        sensitivity_df["SHAP_pH_rank"].max(skipna=True) - sensitivity_df["SHAP_pH_rank"].min(skipna=True)
    )

    mae_robust = max_delta_mae < 0.05
    shap_sensitive = rank_shift > 2
    is_robust = bool(mae_robust and not shap_sensitive)

    if is_robust:
        conclusion = (
            "Robustness conclusion: results are robust to fixed-pH scenarios "
            f"(max ΔMAE = {max_delta_mae:.4f} mg/g; pH SHAP rank shift = {rank_shift:.1f})."
        )
    elif shap_sensitive:
        conclusion = (
            "Robustness conclusion: predictive error is "
            f"{'robust' if mae_robust else 'not robust'} "
            f"(max ΔMAE = {max_delta_mae:.4f} mg/g), but interpretation is sensitive "
            f"because pH SHAP rank shifts by {rank_shift:.1f} positions."
        )
    else:
        conclusion = (
            "Robustness conclusion: results are not robust to fixed-pH scenarios "
            f"because max ΔMAE = {max_delta_mae:.4f} mg/g, exceeding 0.05 mg/g."
        )

    print(conclusion)
    return is_robust


def run(df: pd.DataFrame | None = None) -> tuple[pd.DataFrame, bool]:
    """Run baseline and fixed-pH sensitivity scenarios.

    Parameters
    ----------
    df:
        Optional raw dataset. If omitted, ``config.DATA_FILE`` is loaded.

    Returns
    -------
    tuple[pandas.DataFrame, bool]
        The scenario comparison table and a robustness flag. Robustness is true
        only when max ΔMAE is below 0.05 mg/g and pH SHAP rank does not shift by
        more than two positions.
    """
    _ensure_output_dirs()
    raw_df = clean_model_data(df.copy() if df is not None else _load_raw_dataset())
    x, y = _prepare_xy(raw_df)

    bins = _stratified_bins(y, n_splits=5)
    x_train, x_test, y_train, y_test = train_test_split(
        x,
        y,
        test_size=TEST_SIZE,
        random_state=RANDOM_SEED,
        stratify=bins,
    )

    rows: list[dict[str, Any]] = []
    scenarios: list[tuple[str, float | None]] = [("imputed", None)]
    scenarios.extend((f"pH={value:g}", float(value)) for value in PH_SCENARIOS)

    for scenario, ph_value in scenarios:
        print(f"Running pH sensitivity scenario: {scenario}")
        rows.append(_evaluate_scenario(scenario, ph_value, x_train, y_train, x_test, y_test))

    sensitivity_df = pd.DataFrame(rows)
    report_path = REPORTS_DIR / "ph_sensitivity.csv"
    sensitivity_df.to_csv(report_path, index=False)
    figure_path = _save_plot(sensitivity_df)

    print("")
    print("pH sensitivity comparison")
    print("=" * 72)
    print(sensitivity_df.to_string(index=False, float_format=lambda value: f"{value:.4f}"))
    print("")
    is_robust = _print_conclusion(sensitivity_df)
    print(f"Saved pH sensitivity report: {report_path}")
    print(f"Saved pH sensitivity plot: {figure_path}")

    return sensitivity_df, is_robust


if __name__ == "__main__":
    run()
