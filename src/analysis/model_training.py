"""Train and compare ML models for Cu2+ adsorption capacity prediction.

The public entry point is ``run(X_train, y_train, X_test, y_test)``. It tunes
RandomForest, XGBoost, LightGBM, and ElasticNet with 10-fold stratified
cross-validation on binned qe values, computes out-of-fold metrics, evaluates
the tuned models on a held-out test set, performs pairwise Wilcoxon tests with
Holm correction, saves model/report/figure artifacts, and returns the best
model by OOF MAE.

The preprocessing pipeline is fit inside each CV fold and inside each
RandomizedSearchCV split. Numeric predictors are median-imputed, categorical
predictors are most-frequent imputed and one-hot encoded, and ElasticNet also
uses standard scaling after preprocessing. This keeps tuning and OOF estimates
free from validation/test leakage while accepting raw project data frames.
"""

from __future__ import annotations

import itertools
import sys
from pathlib import Path
from typing import Any

import joblib
import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor
from scipy.stats import loguniform, randint, uniform, wilcoxon
from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import ElasticNet
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from statsmodels.stats.multitest import multipletests
from tqdm.auto import tqdm
from xgboost import XGBRegressor

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
    ENET_PARAMS,
    FIGURES_DIR,
    LGBM_PARAMS,
    MODELS_DIR,
    N_FOLDS,
    NUM_COLS,
    RANDOM_SEED,
    REPORTS_DIR,
    RF_PARAMS,
    TARGET,
    TEST_SIZE,
    XGB_PARAMS,
)
from src.analysis.data_cleaning import clean_model_data  # noqa: E402


MODEL_ORDER = ["RandomForest", "XGBoost", "LightGBM", "ElasticNet"]
PLOT_MODEL_ORDER = ["RandomForest", "LightGBM", "XGBoost", "ElasticNet"]
MODEL_LABELS = {
    "RandomForest": "RF",
    "LightGBM": "LGBM",
    "XGBoost": "XGB",
    "ElasticNet": "ElasticNet",
}


def _ensure_output_dirs() -> None:
    """Create output directories needed by this training step."""
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)


def _load_raw_dataset() -> pd.DataFrame:
    """Load config.DATA_FILE for the optional command-line execution path."""
    data_file = Path(DATA_FILE)
    if not data_file.exists():
        raise FileNotFoundError(f"DATA_FILE does not exist: {data_file}")

    if data_file.suffix.lower() in {".xlsx", ".xls"}:
        return pd.read_excel(data_file)
    if data_file.suffix.lower() == ".csv":
        return pd.read_csv(data_file)

    raise ValueError(f"Unsupported DATA_FILE extension: {data_file.suffix}")


def _make_one_hot_encoder() -> OneHotEncoder:
    """Return a dense OneHotEncoder compatible with sklearn 1.1+ APIs."""
    try:
        return OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    except TypeError:
        return OneHotEncoder(handle_unknown="ignore", sparse=False)


def _present_columns(x: pd.DataFrame, requested: list[str]) -> list[str]:
    return [col for col in requested if col in x.columns]


def _make_preprocessor(x: pd.DataFrame) -> ColumnTransformer:
    """Build preprocessing for raw project predictors.

    Numeric project columns are median-imputed; categorical project columns are
    most-frequent imputed and one-hot encoded. If callers pass an already
    numeric matrix, all non-config columns are treated as numeric predictors.
    """
    cat_cols = _present_columns(x, CAT_COLS)
    num_cols = _present_columns(x, NUM_COLS)
    extra_numeric = [
        col
        for col in x.columns
        if col not in cat_cols + num_cols and pd.api.types.is_numeric_dtype(x[col])
    ]
    num_cols = num_cols + extra_numeric

    transformers: list[tuple[str, Any, list[str]]] = []
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


def _stratified_bins(y: pd.Series, n_splits: int) -> pd.Series:
    """Create qe bins with enough samples for StratifiedKFold."""
    y = pd.Series(y).reset_index(drop=True)
    max_bins = min(10, y.nunique(), len(y) // n_splits)
    for n_bins in range(max_bins, 1, -1):
        bins = pd.qcut(y, q=n_bins, labels=False, duplicates="drop")
        counts = pd.Series(bins).value_counts(dropna=False)
        if counts.min() >= n_splits:
            return bins.astype(int)

    return pd.Series(np.zeros(len(y), dtype=int))


def _nmae(y_true: pd.Series | np.ndarray, y_pred: np.ndarray, y_range: float) -> float:
    """Compute normalized MAE using a supplied target range."""
    if y_range <= 0:
        return float("nan")
    return float(mean_absolute_error(y_true, y_pred) / y_range)


def _base_models(x_train: pd.DataFrame) -> dict[str, Pipeline]:
    """Construct model pipelines with config starting parameters."""
    preprocessor = _make_preprocessor(x_train)

    rf_params = dict(RF_PARAMS)
    rf_params.update({"random_state": RANDOM_SEED, "n_jobs": -1})

    xgb_params = dict(XGB_PARAMS)
    xgb_params.update(
        {
            "objective": "reg:squarederror",
            "eval_metric": "mae",
            "random_state": RANDOM_SEED,
            "n_jobs": -1,
            "tree_method": "hist",
        }
    )

    lgbm_params = dict(LGBM_PARAMS)
    lgbm_params.update({"random_state": RANDOM_SEED, "n_jobs": -1, "verbose": -1})

    enet_params = dict(ENET_PARAMS)
    enet_params.update({"random_state": RANDOM_SEED})

    return {
        "RandomForest": Pipeline(
            steps=[
                ("preprocess", preprocessor),
                ("model", RandomForestRegressor(**rf_params)),
            ]
        ),
        "XGBoost": Pipeline(
            steps=[
                ("preprocess", clone(preprocessor)),
                ("model", XGBRegressor(**xgb_params)),
            ]
        ),
        "LightGBM": Pipeline(
            steps=[
                ("preprocess", clone(preprocessor)),
                ("model", LGBMRegressor(**lgbm_params)),
            ]
        ),
        "ElasticNet": Pipeline(
            steps=[
                ("preprocess", clone(preprocessor)),
                ("scale", StandardScaler()),
                ("model", ElasticNet(**enet_params)),
            ]
        ),
    }


def _search_spaces() -> dict[str, dict[str, Any]]:
    """Define RandomizedSearchCV spaces around config starting parameters."""
    return {
        "RandomForest": {
            "model__n_estimators": randint(500, 2501),
            "model__max_features": ["sqrt", "log2", 0.4, 0.6, 0.8, 1.0],
            "model__min_samples_leaf": randint(1, 9),
            "model__min_samples_split": randint(2, 13),
            "model__bootstrap": [True],
        },
        "XGBoost": {
            "model__n_estimators": randint(300, 1801),
            "model__max_depth": randint(2, 9),
            "model__learning_rate": loguniform(0.005, 0.2),
            "model__subsample": uniform(0.55, 0.45),
            "model__colsample_bytree": uniform(0.55, 0.45),
            "model__min_child_weight": loguniform(0.5, 10.0),
            "model__reg_alpha": loguniform(1e-4, 10.0),
            "model__reg_lambda": loguniform(0.1, 20.0),
        },
        "LightGBM": {
            "model__n_estimators": randint(300, 1801),
            "model__num_leaves": randint(7, 64),
            "model__learning_rate": loguniform(0.005, 0.2),
            "model__subsample": uniform(0.55, 0.45),
            "model__colsample_bytree": uniform(0.55, 0.45),
            "model__min_child_samples": randint(5, 60),
            "model__reg_alpha": loguniform(1e-4, 10.0),
            "model__reg_lambda": loguniform(0.1, 20.0),
        },
        "ElasticNet": {
            "model__alpha": loguniform(1e-4, 10.0),
            "model__l1_ratio": uniform(0.0, 1.0),
            "model__max_iter": [10000, 20000],
        },
    }


def _tune_model(
    name: str,
    estimator: Pipeline,
    param_distributions: dict[str, Any],
    x_train: pd.DataFrame,
    y_train: pd.Series,
    cv_splits: list[tuple[np.ndarray, np.ndarray]],
    n_iter: int,
) -> RandomizedSearchCV:
    """Run RandomizedSearchCV for one model using MAE as the objective."""
    search = RandomizedSearchCV(
        estimator=estimator,
        param_distributions=param_distributions,
        n_iter=n_iter,
        scoring="neg_mean_absolute_error",
        cv=cv_splits,
        random_state=RANDOM_SEED,
        n_jobs=-1,
        refit=True,
        verbose=0,
        error_score="raise",
    )
    print(f"Tuning {name} with {n_iter} randomized settings...")
    search.fit(x_train, y_train)
    return search


def _oof_evaluate(
    estimator: Pipeline,
    x_train: pd.DataFrame,
    y_train: pd.Series,
    cv_splits: list[tuple[np.ndarray, np.ndarray]],
) -> dict[str, Any]:
    """Compute OOF predictions and per-fold MAE for a tuned estimator."""
    oof_pred = np.full(len(y_train), np.nan, dtype=float)
    fold_mae: list[float] = []
    fold_nmae: list[float] = []
    fold_r2: list[float] = []
    train_range = float(y_train.max() - y_train.min())

    for fold, (train_idx, valid_idx) in enumerate(cv_splits, start=1):
        fold_estimator = clone(estimator)
        x_fold_train = x_train.iloc[train_idx]
        y_fold_train = y_train.iloc[train_idx]
        x_fold_valid = x_train.iloc[valid_idx]
        y_fold_valid = y_train.iloc[valid_idx]

        fold_estimator.fit(x_fold_train, y_fold_train)
        pred = fold_estimator.predict(x_fold_valid)
        oof_pred[valid_idx] = pred

        fold_mae.append(float(mean_absolute_error(y_fold_valid, pred)))
        fold_nmae.append(_nmae(y_fold_valid, pred, train_range))
        fold_r2.append(float(r2_score(y_fold_valid, pred)))

    return {
        "oof_pred": oof_pred,
        "fold_mae": fold_mae,
        "fold_nmae": fold_nmae,
        "fold_r2": fold_r2,
        "oof_mae": float(mean_absolute_error(y_train, oof_pred)),
        "oof_nmae": _nmae(y_train, oof_pred, train_range),
        "oof_r2": float(r2_score(y_train, oof_pred)),
        "fold_mae_mean": float(np.mean(fold_mae)),
        "fold_mae_std": float(np.std(fold_mae, ddof=1)),
    }


def _wilcoxon_holm(per_fold_mae: dict[str, list[float]]) -> pd.DataFrame:
    """Pairwise Wilcoxon signed-rank tests with Holm-adjusted p-values."""
    rows: list[dict[str, Any]] = []
    pairs = list(itertools.combinations(MODEL_ORDER, 2))

    for model_a, model_b in pairs:
        a = np.asarray(per_fold_mae[model_a], dtype=float)
        b = np.asarray(per_fold_mae[model_b], dtype=float)
        try:
            stat, p_value = wilcoxon(a, b, zero_method="wilcox", alternative="two-sided")
        except ValueError:
            stat, p_value = np.nan, 1.0
        rows.append(
            {
                "model_a": model_a,
                "model_b": model_b,
                "wilcoxon_stat": float(stat) if np.isfinite(stat) else np.nan,
                "p_value": float(p_value),
            }
        )

    p_values = [row["p_value"] for row in rows]
    reject, p_adjusted, _, _ = multipletests(p_values, alpha=0.05, method="holm")
    for row, adj_p, is_reject in zip(rows, p_adjusted, reject):
        row["p_adjusted_holm"] = float(adj_p)
        row["significant_0.05"] = bool(is_reject)

    return pd.DataFrame(rows)


def _save_model_comparison(
    metrics_rows: list[dict[str, Any]],
    significance: pd.DataFrame,
    per_fold_mae: dict[str, list[float]],
) -> Path:
    """Save Table 1-style model comparison plus Holm p-values."""
    metrics = pd.DataFrame(metrics_rows)
    metrics = metrics.sort_values("oof_mae").reset_index(drop=True)
    metrics_path = REPORTS_DIR / "model_comparison.csv"
    metrics.to_csv(metrics_path, index=False)

    sig_path = REPORTS_DIR / "model_comparison_wilcoxon_holm.csv"
    significance.to_csv(sig_path, index=False)

    fold_mae_rows = []
    for model_name, values in per_fold_mae.items():
        for fold, mae in enumerate(values, start=1):
            fold_mae_rows.append(
                {
                    "model": model_name,
                    "model_label": MODEL_LABELS.get(model_name, model_name),
                    "fold": fold,
                    "mae": mae,
                }
            )
    pd.DataFrame(fold_mae_rows).to_csv(REPORTS_DIR / "model_fold_mae.csv", index=False)
    return metrics_path


def _significance_label(p_adjusted: float) -> str:
    if p_adjusted < 0.05:
        return "p_adj < 0.05"
    return "n.s."


def _add_significance_brackets(
    ax: plt.Axes,
    per_fold_mae: dict[str, list[float]],
    significance: pd.DataFrame,
) -> None:
    """Draw Holm significance brackets above the MAE box plot."""
    y_max = max(max(values) for values in per_fold_mae.values())
    y_min = min(min(values) for values in per_fold_mae.values())
    step = max((y_max - y_min) * 0.08, 0.01)
    current_y = y_max + step

    significant = significance[significance["significant_0.05"]].copy()
    significant = significant.sort_values(["p_adjusted_holm", "model_b", "model_a"]).head(6)

    if significant.empty:
        ax.text(
            0.5,
            0.98,
            "No Holm-significant pairwise differences",
            transform=ax.transAxes,
            ha="center",
            va="top",
            fontsize=13,
        )
        return

    positions = {name: idx + 1 for idx, name in enumerate(PLOT_MODEL_ORDER)}
    for _, row in significant.iterrows():
        x1 = positions[row["model_a"]]
        x2 = positions[row["model_b"]]
        if x1 > x2:
            x1, x2 = x2, x1

        ax.plot([x1, x1, x2, x2], [current_y, current_y + step / 3, current_y + step / 3, current_y], color="black", linewidth=0.8)
        ax.text(
            (x1 + x2) / 2,
            current_y + step / 2.5,
            _significance_label(row["p_adjusted_holm"]),
            ha="center",
            va="bottom",
            fontsize=12,
        )
        current_y += step

    ax.set_ylim(top=current_y + step)


def _save_mae_boxplot(
    per_fold_mae: dict[str, list[float]],
    significance: pd.DataFrame,
) -> Path:
    """Save Fig. 2a-style box plot of per-fold MAE with Holm brackets."""
    output = FIGURES_DIR / "model_mae_comparison.png"
    data = [per_fold_mae[name] for name in PLOT_MODEL_ORDER]
    labels = [MODEL_LABELS[name] for name in PLOT_MODEL_ORDER]

    fig, ax = plt.subplots(figsize=(7.2, 5.6))
    box = ax.boxplot(data, labels=labels, patch_artist=True, showmeans=True, widths=0.45)
    colors = ["#4c78a8", "#54a24b", "#f58518", "#b279a2"]
    for patch, color in zip(box["boxes"], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.72)

    ax.set_ylabel("Per-fold MAE", fontsize=17)
    ax.set_xlabel("Model", fontsize=17)
    ax.tick_params(axis="both", labelsize=15)
    ax.grid(axis="y", linestyle=":", alpha=0.45)
    _add_significance_brackets(ax, per_fold_mae, significance)
    fig.tight_layout()
    fig.savefig(output, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return output


def _prepare_inputs(
    x_train: pd.DataFrame,
    y_train: pd.Series | np.ndarray,
    x_test: pd.DataFrame,
    y_test: pd.Series | np.ndarray,
) -> tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series]:
    """Validate and align external train/test inputs."""
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


def run(
    X_train: pd.DataFrame,
    y_train: pd.Series | np.ndarray,
    X_test: pd.DataFrame,
    y_test: pd.Series | np.ndarray,
    n_iter: int = 50,
) -> tuple[Pipeline, str, dict[str, Any]]:
    """Train, tune, statistically compare, and save four qe prediction models.

    Parameters
    ----------
    X_train, y_train, X_test, y_test:
        Training and held-out test data. Predictors may be raw project columns
        with missing values and categorical variables; preprocessing is handled
        inside model pipelines.
    n_iter:
        Number of RandomizedSearchCV settings. Defaults to 50, matching the
        project requirement.

    Returns
    -------
    tuple
        ``(best_model, best_model_name, all_cv_results)`` where ``best_model``
        is refit on the full training set and ``all_cv_results`` contains OOF
        metrics, per-fold MAE vectors, test metrics, best parameters, and Holm
        Wilcoxon results.
    """
    _ensure_output_dirs()
    x_train, y_train_s, x_test, y_test_s = _prepare_inputs(X_train, y_train, X_test, y_test)

    bins = _stratified_bins(y_train_s, N_FOLDS)
    cv = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=RANDOM_SEED)
    cv_splits = list(cv.split(x_train, bins))
    base_models = _base_models(x_train)
    search_spaces = _search_spaces()
    train_range = float(y_train_s.max() - y_train_s.min())
    test_range = float(y_test_s.max() - y_test_s.min())

    metrics_rows: list[dict[str, Any]] = []
    per_fold_mae: dict[str, list[float]] = {}
    tuned_models: dict[str, Pipeline] = {}
    all_cv_results: dict[str, Any] = {"models": {}}

    for name in tqdm(MODEL_ORDER, desc="Model comparison"):
        search = _tune_model(
            name,
            base_models[name],
            search_spaces[name],
            x_train,
            y_train_s,
            cv_splits,
            n_iter=n_iter,
        )

        best_estimator = search.best_estimator_
        oof = _oof_evaluate(best_estimator, x_train, y_train_s, cv_splits)

        best_estimator.fit(x_train, y_train_s)
        test_pred = best_estimator.predict(x_test)
        test_mae = float(mean_absolute_error(y_test_s, test_pred))
        test_nmae = _nmae(y_test_s, test_pred, test_range)
        test_r2 = float(r2_score(y_test_s, test_pred))

        model_path = MODELS_DIR / f"{name}_tuned.joblib"
        joblib.dump(best_estimator, model_path)

        metrics = {
            "model": name,
            "oof_mae": oof["oof_mae"],
            "oof_nmae": oof["oof_nmae"],
            "oof_r2": oof["oof_r2"],
            "fold_mae_mean": oof["fold_mae_mean"],
            "fold_mae_std": oof["fold_mae_std"],
            "test_mae": test_mae,
            "test_nmae": test_nmae,
            "test_r2": test_r2,
            "best_cv_mae_from_search": float(-search.best_score_),
            "model_path": str(model_path),
        }
        metrics_rows.append(metrics)
        per_fold_mae[name] = oof["fold_mae"]
        tuned_models[name] = best_estimator
        all_cv_results["models"][name] = {
            "metrics": metrics,
            "best_params": search.best_params_,
            "fold_mae": oof["fold_mae"],
            "fold_nmae": oof["fold_nmae"],
            "fold_r2": oof["fold_r2"],
            "oof_pred": oof["oof_pred"],
            "test_pred": test_pred,
        }

        print(
            f"{name}: OOF MAE={oof['oof_mae']:.4f}, "
            f"OOF R²={oof['oof_r2']:.4f}, "
            f"test MAE={test_mae:.4f}, test R²={test_r2:.4f}"
        )

    significance = _wilcoxon_holm(per_fold_mae)
    all_cv_results["wilcoxon_holm"] = significance
    all_cv_results["per_fold_mae"] = per_fold_mae

    comparison_path = _save_model_comparison(metrics_rows, significance, per_fold_mae)
    figure_path = _save_mae_boxplot(per_fold_mae, significance)

    comparison = pd.DataFrame(metrics_rows).sort_values("oof_mae").reset_index(drop=True)
    best_model_name = str(comparison.iloc[0]["model"])
    best_model = tuned_models[best_model_name]
    best_model.fit(x_train, y_train_s)
    joblib.dump(best_model, MODELS_DIR / f"{best_model_name}_tuned.joblib")

    print("")
    print("Model comparison")
    print("=" * 72)
    print(
        comparison[
            ["model", "oof_mae", "oof_nmae", "oof_r2", "test_mae", "test_nmae", "test_r2"]
        ].to_string(index=False, float_format=lambda value: f"{value:.4f}")
    )
    print("")
    print("Pairwise Wilcoxon signed-rank tests with Holm correction")
    print(significance.to_string(index=False, float_format=lambda value: f"{value:.6g}"))
    print("")
    print(f"Best model by lowest OOF MAE: {best_model_name}")
    print(f"Saved model comparison: {comparison_path}")
    print(f"Saved MAE box plot: {figure_path}")

    return best_model, best_model_name, all_cv_results


if __name__ == "__main__":
    raw_df = _load_raw_dataset()
    feature_cols = [col for col in CAT_COLS + NUM_COLS if col in raw_df.columns]
    clean_df = clean_model_data(raw_df)
    y = clean_df[TARGET]
    bins = _stratified_bins(y, n_splits=5)
    X_train_main, X_test_main, y_train_main, y_test_main = train_test_split(
        clean_df[feature_cols],
        y,
        test_size=TEST_SIZE,
        random_state=RANDOM_SEED,
        stratify=bins,
    )
    run(X_train_main, y_train_main, X_test_main, y_test_main)
