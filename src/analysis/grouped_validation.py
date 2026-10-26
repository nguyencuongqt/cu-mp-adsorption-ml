"""Study-grouped validation of the Cu2+-MP adsorption models.

Observations from one study share polymers, apparatus and conditions, so a
random split places points of the same isotherm in both training and test
folds. This module quantifies that optimism by evaluating every model under
three resampling schemes on identical data:

- ``random``: repeated qe-stratified K-fold (the original design);
- ``group``: repeated GroupKFold with whole studies held out;
- ``loso``: leave-one-study-out.

It also reports (i) raw-scale and log-scale targets, (ii) predictor-set
ablations (without rpm; polymer/ageing/condition/Ce only), (iii) the effect of
excluding seawater studies, (iv) QRF prediction-interval coverage under each
scheme, and (v) out-of-study performance by polymer and ageing status.

Hyperparameters default to the tuned values in
``outputs/reports/tuned_hyperparameters.json``. ``--nested`` re-tunes inside
every outer training fold with an inner GroupKFold, so no held-out study
informs model selection.

Usage
-----
    python scripts/build_grouped_dataset.py
    python src/analysis/grouped_validation.py
    python src/analysis/grouped_validation.py --nested --n-iter 20
"""

from __future__ import annotations

import argparse
import json
import sys
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator

import joblib
import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor
from quantile_forest import RandomForestQuantileRegressor
from sklearn.base import clone
from sklearn.compose import ColumnTransformer, TransformedTargetRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import ElasticNet
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import (
    GroupKFold,
    LeaveOneGroupOut,
    RandomizedSearchCV,
    StratifiedKFold,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, StandardScaler
from xgboost import XGBRegressor

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from config import (  # noqa: E402
    CAT_COLS,
    DATA_GROUPED_FILE,
    ENET_PARAMS,
    FIGURES_DIR,
    GROUP_COL,
    LGBM_PARAMS,
    MODELS_DIR,
    N_FOLDS,
    NUM_COLS,
    QRF_LOWER,
    QRF_UPPER,
    RANDOM_SEED,
    REPORTS_DIR,
    RF_PARAMS,
    TARGET,
    TUNED_PARAMS_FILE,
    XGB_PARAMS,
)

warnings.filterwarnings("ignore", category=UserWarning)

MODEL_ORDER = ["RandomForest", "XGBoost", "LightGBM", "ElasticNet"]
FEATURE_SETS = {
    "full": CAT_COLS + NUM_COLS,
    "no_rpm": CAT_COLS + [c for c in NUM_COLS if c != "rpm"],
    "core": CAT_COLS + ["Ce"],
}
CE_FLOOR = 1e-7  # mg/L; keeps log10(Ce) finite for the single Ce = 0 record
QE_FLOOR = 1e-6  # mg/g; floor applied to predictions before log-scale metrics
QRF_QUANTILES = [QRF_LOWER, 0.50, QRF_UPPER]


@dataclass(frozen=True)
class Experiment:
    model: str
    target: str  # "raw" or "log"
    features: str
    subset: str  # "all" or "freshwater"
    schemes: tuple[str, ...]


# --------------------------------------------------------------------------
# Data and hyperparameters
# --------------------------------------------------------------------------
def load_data(subset: str = "all") -> pd.DataFrame:
    if not DATA_GROUPED_FILE.exists():
        raise FileNotFoundError(
            f"{DATA_GROUPED_FILE} not found. Run scripts/build_grouped_dataset.py first."
        )
    df = pd.read_csv(DATA_GROUPED_FILE)
    if subset == "freshwater":
        df = df[df["matrix"] == "freshwater"]
    return df.reset_index(drop=True)


def export_tuned_params() -> dict[str, dict[str, Any]]:
    """Read tuned hyperparameters, extracting them from saved models once."""
    if TUNED_PARAMS_FILE.exists():
        return json.loads(TUNED_PARAMS_FILE.read_text(encoding="utf-8"))

    keep = {
        "RandomForest": ["n_estimators", "max_features", "min_samples_leaf", "min_samples_split"],
        "XGBoost": ["n_estimators", "max_depth", "learning_rate", "subsample",
                    "colsample_bytree", "min_child_weight", "reg_alpha", "reg_lambda"],
        "LightGBM": ["n_estimators", "num_leaves", "learning_rate", "subsample",
                     "colsample_bytree", "min_child_samples", "reg_alpha", "reg_lambda"],
        "ElasticNet": ["alpha", "l1_ratio", "max_iter"],
    }
    fallback = {"RandomForest": RF_PARAMS, "XGBoost": XGB_PARAMS,
                "LightGBM": LGBM_PARAMS, "ElasticNet": ENET_PARAMS}
    params: dict[str, dict[str, Any]] = {}
    for name in MODEL_ORDER:
        path = MODELS_DIR / f"{name}_tuned.joblib"
        try:
            model = joblib.load(path).named_steps["model"]
            fitted = model.get_params()
            params[name] = {k: fitted[k] for k in keep[name] if fitted.get(k) is not None}
        except Exception as exc:  # missing or version-incompatible artifact
            print(f"Tuned {name} not readable ({exc}); using config defaults.")
            params[name] = dict(fallback[name])
            if name == "RandomForest":
                params[name]["max_features"] = 0.8
    params = json.loads(json.dumps(params, default=float))
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    TUNED_PARAMS_FILE.write_text(json.dumps(params, indent=2), encoding="utf-8")
    return params


# --------------------------------------------------------------------------
# Pipelines
# --------------------------------------------------------------------------
def _log10_ce(x: Any) -> np.ndarray:
    return np.log10(np.clip(np.asarray(x, dtype=float), CE_FLOOR, None))


def _log10(y: Any) -> np.ndarray:
    return np.log10(np.clip(np.asarray(y, dtype=float), QE_FLOOR, None))


def _pow10(y: Any) -> np.ndarray:
    return np.power(10.0, np.asarray(y, dtype=float))


def make_preprocessor(features: list[str], log_ce: bool) -> ColumnTransformer:
    cat = [c for c in features if c in CAT_COLS]
    num = [c for c in features if c in NUM_COLS and not (log_ce and c == "Ce")]
    transformers: list[tuple[str, Any, list[str]]] = []
    if num:
        transformers.append(("num", SimpleImputer(strategy="median"), num))
    if log_ce and "Ce" in features:
        transformers.append(("log_ce", FunctionTransformer(_log10_ce), ["Ce"]))
    if cat:
        cat_pipe = Pipeline([
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ])
        transformers.append(("cat", cat_pipe, cat))
    return ColumnTransformer(transformers, remainder="drop", verbose_feature_names_out=False)


def make_regressor(name: str, params: dict[str, Any]) -> Any:
    if name == "RandomForest":
        return RandomForestRegressor(**params, random_state=RANDOM_SEED, n_jobs=-1)
    if name == "XGBoost":
        return XGBRegressor(**params, objective="reg:squarederror", tree_method="hist",
                            random_state=RANDOM_SEED, n_jobs=-1)
    if name == "LightGBM":
        return LGBMRegressor(**params, random_state=RANDOM_SEED, n_jobs=-1, verbose=-1)
    if name == "ElasticNet":
        return ElasticNet(**params, random_state=RANDOM_SEED)
    if name == "QRF":
        return RandomForestQuantileRegressor(**params, default_quantiles=QRF_QUANTILES,
                                             random_state=RANDOM_SEED, n_jobs=-1)
    raise ValueError(name)


def make_pipeline(name: str, params: dict[str, Any], features: list[str], target: str) -> Pipeline:
    """Model pipeline; ``target='log'`` fits log10(qe) with log10(Ce) as input."""
    log = target == "log"
    steps: list[tuple[str, Any]] = [("preprocess", make_preprocessor(features, log_ce=log))]
    if name == "ElasticNet":
        steps.append(("scale", StandardScaler()))
    regressor = make_regressor(name, params)
    if log and name != "QRF":
        regressor = TransformedTargetRegressor(regressor=regressor, func=_log10, inverse_func=_pow10,
                                               check_inverse=False)
    steps.append(("model", regressor))
    return Pipeline(steps)


# --------------------------------------------------------------------------
# Resampling schemes
# --------------------------------------------------------------------------
def _qe_bins(y: pd.Series, n_splits: int) -> np.ndarray:
    return pd.qcut(y, q=min(10, len(y) // n_splits), labels=False, duplicates="drop").to_numpy()


def iter_splits(
    scheme: str, y: pd.Series, groups: pd.Series, repeats: int
) -> Iterator[tuple[int, int, np.ndarray, np.ndarray]]:
    """Yield ``(repeat, fold, train_idx, test_idx)`` for a resampling scheme."""
    index = np.arange(len(y))
    if scheme == "random":
        bins = _qe_bins(y, N_FOLDS)
        for rep in range(repeats):
            cv = StratifiedKFold(N_FOLDS, shuffle=True, random_state=RANDOM_SEED + rep)
            for fold, (tr, te) in enumerate(cv.split(index, bins)):
                yield rep, fold, tr, te
    elif scheme == "group":
        for rep in range(repeats):
            cv = GroupKFold(N_FOLDS, shuffle=True, random_state=RANDOM_SEED + rep)
            for fold, (tr, te) in enumerate(cv.split(index, y, groups)):
                yield rep, fold, tr, te
    elif scheme == "loso":
        for fold, (tr, te) in enumerate(LeaveOneGroupOut().split(index, y, groups)):
            yield 0, fold, tr, te
    else:
        raise ValueError(scheme)


# --------------------------------------------------------------------------
# Metrics
# --------------------------------------------------------------------------
def point_metrics(y: np.ndarray, pred: np.ndarray) -> dict[str, float]:
    ly, lp = _log10(y), _log10(pred)
    return {
        "mae": float(mean_absolute_error(y, pred)),
        "r2": float(r2_score(y, pred)) if len(y) > 1 else np.nan,
        "mae_log10": float(mean_absolute_error(ly, lp)),
        "r2_log10": float(r2_score(ly, lp)) if len(y) > 1 else np.nan,
        "median_fold_error": float(np.power(10.0, np.median(np.abs(ly - lp)))),
    }


def interval_metrics(y: np.ndarray, lo: np.ndarray, hi: np.ndarray, y_range: float) -> dict[str, float]:
    alpha = 1.0 - (QRF_UPPER - QRF_LOWER)
    score = (hi - lo) + (2 / alpha) * (lo - y) * (y < lo) + (2 / alpha) * (y - hi) * (y > hi)
    return {
        "coverage": float(np.mean((y >= lo) & (y <= hi))),
        "niw": float(np.mean(hi - lo) / y_range),
        "interval_score": float(np.mean(score)),
    }


def summarise(oof: pd.DataFrame, by: str) -> dict[str, float]:
    """Mean and SD of pooled metrics across ``by`` (repeats or studies)."""
    rows = [point_metrics(g["y"].to_numpy(), g["pred"].to_numpy()) for _, g in oof.groupby(by)]
    table = pd.DataFrame(rows)
    out: dict[str, float] = {}
    for col in table.columns:
        out[f"{col}_mean"] = float(table[col].mean())
        out[f"{col}_sd"] = float(table[col].std(ddof=1)) if len(table) > 1 else np.nan
    return out


# --------------------------------------------------------------------------
# Evaluation
# --------------------------------------------------------------------------
def _nested_search(pipe: Pipeline, name: str, target: str, n_iter: int) -> RandomizedSearchCV:
    from src.analysis.model_training import _search_spaces  # reuse the tuning spaces

    space = _search_spaces()[name]
    if target == "log":
        space = {k.replace("model__", "model__regressor__"): v for k, v in space.items()}
    return RandomizedSearchCV(pipe, space, n_iter=n_iter, cv=GroupKFold(5),
                              scoring="neg_mean_absolute_error", random_state=RANDOM_SEED,
                              n_jobs=1, refit=True)


def run_experiment(
    exp: Experiment, params: dict[str, dict[str, Any]], repeats: int,
    nested: bool = False, n_iter: int = 20,
) -> tuple[list[dict[str, Any]], pd.DataFrame]:
    df = load_data(exp.subset)
    features = FEATURE_SETS[exp.features]
    x, y, groups = df[features], df[TARGET], df[GROUP_COL]
    pipe = make_pipeline(exp.model, params[exp.model], features, exp.target)

    summaries: list[dict[str, Any]] = []
    oof_all: list[pd.DataFrame] = []
    for scheme in exp.schemes:
        parts = []
        for rep, fold, tr, te in iter_splits(scheme, y, groups, repeats):
            if nested:
                est = _nested_search(clone(pipe), exp.model, exp.target, n_iter)
                est.fit(x.iloc[tr], y.iloc[tr], groups=groups.iloc[tr])
            else:
                est = clone(pipe).fit(x.iloc[tr], y.iloc[tr])
            parts.append(pd.DataFrame({
                "repeat": rep, "fold": fold, "row": te, GROUP_COL: groups.iloc[te].to_numpy(),
                "ReT": df["ReT"].iloc[te].to_numpy(), "AgS": df["AgS"].iloc[te].to_numpy(),
                "Ce": df["Ce"].iloc[te].to_numpy(),
                "y": y.iloc[te].to_numpy(), "pred": est.predict(x.iloc[te]),
            }))
        oof = pd.concat(parts, ignore_index=True)
        # Repeats are the replication unit for random/group; studies for LOSO.
        stats = summarise(oof, by=GROUP_COL if scheme == "loso" else "repeat")
        pooled = point_metrics(oof["y"].to_numpy(), oof["pred"].to_numpy())
        label = dict(model=exp.model, target=exp.target, features=exp.features,
                     subset=exp.subset, scheme=scheme, nested=nested)
        summaries.append({**label, "n_obs": len(df), "n_studies": groups.nunique(),
                          **{f"pooled_{k}": v for k, v in pooled.items()}, **stats})
        oof_all.append(oof.assign(**label))
        print(f"{exp.model:<12} {exp.target:<3} {exp.features:<6} {exp.subset:<10} {scheme:<6} "
              f"MAE={pooled['mae']:.3f} R2={pooled['r2']:.3f} "
              f"R2log={pooled['r2_log10']:.3f} foldErr={pooled['median_fold_error']:.2f}")
    return summaries, pd.concat(oof_all, ignore_index=True)


def run_qrf(params: dict[str, dict[str, Any]], repeats: int) -> pd.DataFrame:
    """QRF interval coverage under each scheme, target scale and data subset."""
    rows = []
    for subset in ["all", "freshwater"]:
        df = load_data(subset)
        features = FEATURE_SETS["full"]
        x, y, groups = df[features], df[TARGET], df[GROUP_COL]
        y_range = float(y.max() - y.min())
        for target in ["raw", "log"]:
            pipe = make_pipeline("QRF", params["RandomForest"], features, target)
            y_fit = pd.Series(_log10(y)) if target == "log" else y
            for scheme in ["random", "group", "loso"]:
                parts = []
                for rep, fold, tr, te in iter_splits(scheme, y, groups, repeats):
                    est = clone(pipe).fit(x.iloc[tr], y_fit.iloc[tr])
                    q = np.asarray(est.predict(x.iloc[te], quantiles=QRF_QUANTILES))
                    q = _pow10(q) if target == "log" else q
                    parts.append(pd.DataFrame({"repeat": rep, GROUP_COL: groups.iloc[te].to_numpy(),
                                               "y": y.iloc[te].to_numpy(),
                                               "lo": q[:, 0], "pred": q[:, 1], "hi": q[:, 2]}))
                oof = pd.concat(parts, ignore_index=True)
                by = GROUP_COL if scheme == "loso" else "repeat"
                per = pd.DataFrame([
                    interval_metrics(g["y"].to_numpy(), g["lo"].to_numpy(), g["hi"].to_numpy(), y_range)
                    for _, g in oof.groupby(by)
                ])
                pooled = interval_metrics(oof["y"].to_numpy(), oof["lo"].to_numpy(),
                                          oof["hi"].to_numpy(), y_range)
                point = point_metrics(oof["y"].to_numpy(), oof["pred"].to_numpy())
                rows.append({"subset": subset, "target": target, "scheme": scheme,
                             **{f"pooled_{k}": v for k, v in {**pooled, **point}.items()},
                             "coverage_sd": float(per["coverage"].std(ddof=1)),
                             "niw_sd": float(per["niw"].std(ddof=1))})
                print(f"QRF {subset:<10} {target:<3} {scheme:<6} coverage={pooled['coverage']*100:.1f}% "
                      f"NIW={pooled['niw']:.3f} MAE={point['mae']:.3f}")
    return pd.DataFrame(rows)


def subgroup_table(oof: pd.DataFrame) -> pd.DataFrame:
    """Out-of-study (LOSO) performance by polymer and by ageing status."""
    rows = []
    for column in ["ReT", "AgS"]:
        for level, g in oof.groupby(column):
            rows.append({"variable": column, "level": level, "n": len(g),
                         "n_studies": g[GROUP_COL].nunique(),
                         **point_metrics(g["y"].to_numpy(), g["pred"].to_numpy())})
    return pd.DataFrame(rows)


def save_comparison_plot(summary: pd.DataFrame) -> Path:
    """Bar chart of random vs study-grouped performance for the full models."""
    sel = summary[(summary["features"] == "full") & (summary["subset"] == "all")]
    schemes = ["random", "group", "loso"]
    labels = {"random": "Random 10-fold", "group": "Study-grouped 10-fold", "loso": "Leave-one-study-out"}
    colors = {"random": "#9aa5b1", "group": "#2f6f9f", "loso": "#c43c39"}
    fig, axes = plt.subplots(2, 2, figsize=(11, 7.5), sharex=True)
    panels = [("raw", "pooled_r2", "R² (qe)"), ("raw", "pooled_mae", "MAE (mg/g)"),
              ("log", "pooled_r2_log10", "R² (log10 qe)"), ("log", "pooled_median_fold_error", "Median fold error")]
    width = 0.26
    for ax, (target, metric, title) in zip(axes.ravel(), panels):
        part = sel[sel["target"] == target]
        for i, scheme in enumerate(schemes):
            vals = [part[(part["model"] == m) & (part["scheme"] == scheme)][metric].mean() for m in MODEL_ORDER]
            ax.bar(np.arange(len(MODEL_ORDER)) + (i - 1) * width, vals, width,
                   color=colors[scheme], label=labels[scheme])
        ax.axhline(0, color="black", linewidth=0.6)
        ax.set_title(f"{title} — {target}-scale model", fontsize=11)
        ax.set_xticks(np.arange(len(MODEL_ORDER)))
        ax.set_xticklabels(["RF", "XGB", "LightGBM", "ElasticNet"])
        ax.grid(axis="y", linestyle=":", alpha=0.5)
    axes[0, 0].legend(frameon=False, fontsize=9)
    fig.tight_layout()
    output = FIGURES_DIR / "grouped_cv_comparison.png"
    fig.savefig(output, dpi=300)
    plt.close(fig)
    return output


def build_experiments() -> list[Experiment]:
    all_schemes = ("random", "group", "loso")
    grouped = ("group", "loso")
    exps = [Experiment(m, t, "full", "all", all_schemes) for m in MODEL_ORDER for t in ("raw", "log")]
    for m in ("RandomForest", "XGBoost"):
        for t in ("raw", "log"):
            exps += [Experiment(m, t, f, "all", grouped) for f in ("no_rpm", "core")]
            exps.append(Experiment(m, t, "full", "freshwater", grouped))
    return exps


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--repeats", type=int, default=5, help="Repeats for random/group K-fold.")
    parser.add_argument("--nested", action="store_true", help="Re-tune inside each outer fold.")
    parser.add_argument("--n-iter", type=int, default=20, help="Random-search settings when nested.")
    parser.add_argument("--models", nargs="+", default=MODEL_ORDER, help="Subset of models to run.")
    parser.add_argument("--skip-qrf", action="store_true")
    args = parser.parse_args()

    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    params = export_tuned_params()
    suffix = "_nested" if args.nested else ""

    experiments = [e for e in build_experiments() if e.model in args.models]
    if args.nested:  # nested tuning is reported for the full predictor set only
        experiments = [Experiment(e.model, e.target, e.features, e.subset, ("group", "loso"))
                       for e in experiments if e.features == "full" and e.subset == "all"]
    repeats = 1 if args.nested else args.repeats

    summaries: list[dict[str, Any]] = []
    oofs: list[pd.DataFrame] = []
    for exp in experiments:
        rows, oof = run_experiment(exp, params, repeats, nested=args.nested, n_iter=args.n_iter)
        summaries += rows
        oofs.append(oof)

    summary = pd.DataFrame(summaries)
    summary.to_csv(REPORTS_DIR / f"grouped_cv_summary{suffix}.csv", index=False)
    oof_all = pd.concat(oofs, ignore_index=True)
    loso = oof_all[(oof_all["scheme"] == "loso") & (oof_all["features"] == "full") & (oof_all["subset"] == "all")]
    loso.to_csv(REPORTS_DIR / f"grouped_cv_loso_predictions{suffix}.csv", index=False)

    if not args.nested:
        primary = loso[(loso["model"] == "RandomForest") & (loso["target"] == "log")]
        subgroup_table(primary).to_csv(REPORTS_DIR / "grouped_cv_subgroups.csv", index=False)
        per_study = pd.DataFrame([
            {GROUP_COL: s, "n": len(g), **point_metrics(g["y"].to_numpy(), g["pred"].to_numpy())}
            for s, g in primary.groupby(GROUP_COL)
        ])
        per_study.to_csv(REPORTS_DIR / "grouped_cv_per_study.csv", index=False)
        print(f"Saved figure: {save_comparison_plot(summary)}")
        if not args.skip_qrf:
            run_qrf(params, args.repeats).to_csv(REPORTS_DIR / "grouped_qrf_coverage.csv", index=False)
    print(f"Saved reports to {REPORTS_DIR}")


if __name__ == "__main__":
    main()
