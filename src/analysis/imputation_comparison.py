"""Compare pH imputation strategies with leakage-safe cross-validation.

This script evaluates how the pH missing-data strategy changes predictive
performance and interpretation. pH is important in the Cu adsorption workflow:
it drives OHE rank #4 with about 12% SHAP contribution, so the choice between
median imputation, MICE, and complete-case analysis can alter both model error
and the apparent role of pH in the final Random Forest.

Each imputation strategy is fit inside the training fold only. Validation-fold
rows are transformed with objects learned from the corresponding training fold,
so no validation outcome or validation distribution is used to fit imputers,
encoders, or Random Forest models. Complete-case analysis drops rows with
missing pH; remaining missing numeric predictors are median-imputed within fold
so the RandomForestRegressor can be trained.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import miceforest as mf
import numpy as np
import pandas as pd
import shap
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import StratifiedKFold
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
    N_FOLDS,
    NUM_COLS,
    RANDOM_SEED,
    REPORTS_DIR,
    RF_PARAMS,
    TARGET,
)
from src.analysis.data_cleaning import clean_model_data  # noqa: E402


PH_COL = "pH"
MICE_ITERATIONS = 20


class ImputationStrategy(Protocol):
    """Interface used by cross-validation for all pH strategies."""

    name: str

    def fit(self, x_train: pd.DataFrame) -> "ImputationStrategy":
        """Fit imputation parameters using only the current training fold."""

    def transform(self, x: pd.DataFrame) -> pd.DataFrame:
        """Transform a training or validation fold without using its target."""


@dataclass
class MedianPHImputer:
    """Median-impute numeric predictors, including pH, as the R baseline."""

    name: str = "Median"
    numeric_imputer: SimpleImputer | None = None

    def fit(self, x_train: pd.DataFrame) -> "MedianPHImputer":
        self.numeric_imputer = SimpleImputer(strategy="median")
        self.numeric_imputer.fit(x_train[NUM_COLS])
        return self

    def transform(self, x: pd.DataFrame) -> pd.DataFrame:
        if self.numeric_imputer is None:
            raise RuntimeError("MedianPHImputer must be fit before transform.")

        out = x.copy()
        out.loc[:, NUM_COLS] = self.numeric_imputer.transform(out[NUM_COLS])
        return out


@dataclass
class MicePHImputer:
    """MICE-impute numeric predictors with miceforest for pH-focused analysis."""

    name: str = "MICE"
    iterations: int = MICE_ITERATIONS
    random_state: int = RANDOM_SEED
    kernel: mf.ImputationKernel | None = None

    def fit(self, x_train: pd.DataFrame) -> "MicePHImputer":
        numeric = x_train[NUM_COLS].reset_index(drop=True).copy()
        self.kernel = mf.ImputationKernel(
            numeric,
            num_datasets=1,
            mean_match_candidates=0,
            random_state=self.random_state,
        )
        self.kernel.mice(self.iterations, verbose=False)
        return self

    def transform(self, x: pd.DataFrame) -> pd.DataFrame:
        if self.kernel is None:
            raise RuntimeError("MicePHImputer must be fit before transform.")

        out = x.copy()
        numeric = out[NUM_COLS].reset_index(drop=True).copy()

        if numeric.isna().any().any():
            imputed = self.kernel.impute_new_data(
                numeric,
                iterations=self.iterations,
                random_state=self.random_state,
                verbose=False,
            )
            out.loc[:, NUM_COLS] = imputed.complete_data(dataset=0).to_numpy()
        else:
            out.loc[:, NUM_COLS] = numeric.to_numpy()

        return out


@dataclass
class CompleteCasePHImputer:
    """Drop missing-pH rows, then median-impute other numeric missingness."""

    name: str = "CCA"
    numeric_imputer: SimpleImputer | None = None

    def fit(self, x_train: pd.DataFrame) -> "CompleteCasePHImputer":
        self.numeric_imputer = SimpleImputer(strategy="median")
        self.numeric_imputer.fit(x_train[NUM_COLS])
        return self

    def transform(self, x: pd.DataFrame) -> pd.DataFrame:
        if self.numeric_imputer is None:
            raise RuntimeError("CompleteCasePHImputer must be fit before transform.")

        out = x.copy()
        out.loc[:, NUM_COLS] = self.numeric_imputer.transform(out[NUM_COLS])
        return out


def _ensure_output_dirs() -> None:
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)


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


def _make_preprocessor() -> ColumnTransformer:
    categorical_pipe = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", _make_one_hot_encoder()),
        ]
    )
    return ColumnTransformer(
        transformers=[
            ("num", "passthrough", NUM_COLS),
            ("cat", categorical_pipe, CAT_COLS),
        ],
        remainder="drop",
        verbose_feature_names_out=False,
    )


def _feature_names(preprocessor: ColumnTransformer) -> list[str]:
    names = list(preprocessor.get_feature_names_out())
    return [name.replace("num__", "").replace("cat__", "") for name in names]


def _make_rf(fold: int) -> RandomForestRegressor:
    params = dict(RF_PARAMS)
    params["random_state"] = RANDOM_SEED + fold
    params["n_jobs"] = -1
    return RandomForestRegressor(**params)


def _stratified_bins(y: pd.Series, n_splits: int) -> pd.Series:
    """Create qe bins with enough members for StratifiedKFold."""
    y = pd.Series(y).reset_index(drop=True)
    max_bins = min(10, y.nunique(), len(y) // n_splits)
    for n_bins in range(max_bins, 1, -1):
        bins = pd.qcut(y, q=n_bins, labels=False, duplicates="drop")
        counts = pd.Series(bins).value_counts(dropna=False)
        if counts.min() >= n_splits:
            return bins.astype(int)

    return pd.Series(np.zeros(len(y), dtype=int))


def _clean_training_frame(train_df: pd.DataFrame) -> pd.DataFrame:
    required = CAT_COLS + NUM_COLS + [TARGET]
    missing = [col for col in required if col not in train_df.columns]
    if missing:
        raise KeyError(f"Missing required columns: {missing}")

    clean = train_df[required].copy()
    clean[TARGET] = pd.to_numeric(clean[TARGET], errors="coerce")
    for col in NUM_COLS:
        clean[col] = pd.to_numeric(clean[col], errors="coerce")

    clean = clean_model_data(clean)
    if len(clean) < N_FOLDS:
        raise ValueError(f"Need at least {N_FOLDS} rows after dropping missing target.")

    return clean


def _fold_shap_ph_importance(
    model: RandomForestRegressor,
    x_valid_model: np.ndarray,
    feature_names: list[str],
) -> tuple[float, int]:
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(x_valid_model, check_additivity=False)
    shap_array = np.asarray(shap_values)

    if shap_array.ndim == 3:
        shap_array = shap_array[:, :, 0]

    mean_abs = np.abs(shap_array).mean(axis=0)
    ph_index = feature_names.index(PH_COL)
    ph_mean_abs = float(mean_abs[ph_index])
    order = np.argsort(mean_abs)[::-1]
    ph_rank = int(np.where(order == ph_index)[0][0] + 1)
    return ph_mean_abs, ph_rank


def _evaluate_strategy(
    df: pd.DataFrame,
    strategy: ImputationStrategy,
) -> tuple[dict[str, float | str], ImputationStrategy]:
    if strategy.name == "CCA":
        eval_df = df.dropna(subset=[PH_COL]).reset_index(drop=True)
    else:
        eval_df = df.reset_index(drop=True)

    x_all = eval_df[CAT_COLS + NUM_COLS]
    y_all = eval_df[TARGET]
    bins = _stratified_bins(y_all, N_FOLDS)
    splitter = StratifiedKFold(
        n_splits=N_FOLDS,
        shuffle=True,
        random_state=RANDOM_SEED,
    )

    fold_mae: list[float] = []
    fold_r2: list[float] = []
    fold_shap: list[float] = []
    fold_rank: list[int] = []

    for fold, (train_idx, valid_idx) in enumerate(splitter.split(x_all, bins), start=1):
        x_train = x_all.iloc[train_idx].copy()
        x_valid = x_all.iloc[valid_idx].copy()
        y_train = y_all.iloc[train_idx].copy()
        y_valid = y_all.iloc[valid_idx].copy()

        fitted_imputer = type(strategy)()
        fitted_imputer.fit(x_train)
        x_train_imp = fitted_imputer.transform(x_train)
        x_valid_imp = fitted_imputer.transform(x_valid)

        preprocessor = _make_preprocessor()
        x_train_model = preprocessor.fit_transform(x_train_imp)
        x_valid_model = preprocessor.transform(x_valid_imp)
        feature_names = _feature_names(preprocessor)

        model = _make_rf(fold)
        model.fit(x_train_model, y_train)
        pred = model.predict(x_valid_model)

        fold_mae.append(float(mean_absolute_error(y_valid, pred)))
        fold_r2.append(float(r2_score(y_valid, pred)))

        ph_shap, ph_rank = _fold_shap_ph_importance(
            model,
            x_valid_model,
            feature_names,
        )
        fold_shap.append(ph_shap)
        fold_rank.append(ph_rank)

    final_imputer = type(strategy)()
    final_imputer.fit(x_all)

    metrics: dict[str, float | str] = {
        "strategy": strategy.name,
        "n_rows_evaluated": float(len(eval_df)),
        "cv_mae_mean": float(np.mean(fold_mae)),
        "cv_mae_std": float(np.std(fold_mae, ddof=1)),
        "cv_r2_mean": float(np.mean(fold_r2)),
        "cv_r2_std": float(np.std(fold_r2, ddof=1)),
        "mean_shap_ph": float(np.mean(fold_shap)),
        "std_shap_ph": float(np.std(fold_shap, ddof=1)),
        "mean_shap_rank_ph": float(np.mean(fold_rank)),
        "median_shap_rank_ph": float(np.median(fold_rank)),
    }
    return metrics, final_imputer


def _format_comparison(results: pd.DataFrame) -> pd.DataFrame:
    table = results.copy()
    table["CV-MAE ± std"] = table.apply(
        lambda row: f"{row['cv_mae_mean']:.4f} ± {row['cv_mae_std']:.4f}",
        axis=1,
    )
    table["CV-R2"] = table["cv_r2_mean"].map("{:.4f}".format)
    table["mean SHAP(pH)"] = table["mean_shap_ph"].map("{:.4f}".format)
    table["mean pH rank"] = table["mean_shap_rank_ph"].map("{:.2f}".format)
    return table[
        [
            "strategy",
            "n_rows_evaluated",
            "CV-MAE ± std",
            "CV-R2",
            "mean SHAP(pH)",
            "mean pH rank",
        ]
    ]


def _save_mae_plot(results: pd.DataFrame) -> Path:
    output = FIGURES_DIR / "imputation_comparison.png"

    colors = ["#4c78a8", "#f58518", "#54a24b"]
    plt.figure(figsize=(7.5, 4.8))
    plt.bar(
        results["strategy"],
        results["cv_mae_mean"],
        yerr=results["cv_mae_std"],
        capsize=6,
        color=colors[: len(results)],
        edgecolor="black",
        linewidth=0.6,
    )
    plt.ylabel("10-fold CV MAE")
    plt.xlabel("pH missing-data strategy")
    plt.tight_layout()
    plt.savefig(output, dpi=300, bbox_inches="tight")
    plt.close()
    return output


def _recommend_strategy(results: pd.DataFrame) -> tuple[str, str]:
    best = results.sort_values(["cv_mae_mean", "cv_mae_std"]).iloc[0]
    best_name = str(best["strategy"])
    median_row = results.loc[results["strategy"] == "Median"].iloc[0]
    mice_row = results.loc[results["strategy"] == "MICE"].iloc[0]

    if best_name == "CCA":
        if median_row["cv_mae_mean"] <= mice_row["cv_mae_mean"]:
            return (
                "Median",
                "Recommendation: keep median imputation as the primary strategy. "
                "CCA has the lowest CV-MAE, but it discards rows with missing pH "
                "and can change the study population after MAR evidence; MICE "
                "does not improve CV-MAE over the median baseline here. Report "
                "CCA and MICE as sensitivity analyses.",
            )

        return (
            "MICE",
            "Recommendation: use MICE going forward. CCA has the lowest CV-MAE, "
            "but it discards rows with missing pH and can change the study "
            "population after MAR evidence; MICE retains all rows and performs "
            "better than the median baseline.",
        )

    if best_name == "MICE":
        return (
            "MICE",
            "Recommendation: use MICE going forward because it gives the lowest "
            "CV-MAE while retaining all rows and respecting the MAR evidence from "
            "the missingness diagnosis.",
        )

    if best["cv_mae_mean"] <= median_row["cv_mae_mean"]:
        return (
            "Median",
            "Recommendation: keep median imputation as the primary strategy "
            "because it is simplest, reproducible, and performs best or tied-best "
            "in CV-MAE; report MICE/CCA as sensitivity checks.",
        )

    return (
        best_name,
        f"Recommendation: use {best_name} going forward because it has the "
        "lowest CV-MAE; retain the other strategies as sensitivity analyses.",
    )


def run(train_df: pd.DataFrame) -> tuple[str, ImputationStrategy]:
    """Run 10-fold imputation comparison and return the recommended strategy.

    Parameters
    ----------
    train_df:
        Raw training data before imputation. It must contain config.CAT_COLS,
        config.NUM_COLS, and config.TARGET. The function does not create a
        holdout split; callers should pass the training portion only if a
        separate test set exists.

    Returns
    -------
    tuple[str, ImputationStrategy]
        The recommended strategy name and the corresponding imputer fitted on
        all rows used for that strategy.
    """
    _ensure_output_dirs()
    df = _clean_training_frame(train_df)

    strategies: list[ImputationStrategy] = [
        MedianPHImputer(),
        MicePHImputer(),
        CompleteCasePHImputer(),
    ]

    rows: list[dict[str, float | str]] = []
    final_imputers: dict[str, ImputationStrategy] = {}

    for strategy in strategies:
        print(f"Evaluating {strategy.name}...")
        metrics, fitted_imputer = _evaluate_strategy(df, strategy)
        rows.append(metrics)
        final_imputers[str(metrics["strategy"])] = fitted_imputer

    results = pd.DataFrame(rows)
    results = results.sort_values("cv_mae_mean").reset_index(drop=True)

    report_path = REPORTS_DIR / "imputation_comparison.csv"
    results.to_csv(report_path, index=False)
    figure_path = _save_mae_plot(results)

    print("")
    print("Imputation comparison")
    print("=" * 60)
    print(_format_comparison(results).to_string(index=False))
    print("")
    print(f"Saved metrics: {report_path}")
    print(f"Saved figure: {figure_path}")
    print("")

    best_name, recommendation = _recommend_strategy(results)
    print(recommendation)

    return best_name, final_imputers[best_name]


if __name__ == "__main__":
    raw_train_df = clean_model_data(_load_raw_dataset())
    run(raw_train_df)
