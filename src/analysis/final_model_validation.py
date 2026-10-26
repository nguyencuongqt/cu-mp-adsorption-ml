"""Final validation numbers for the reduced-predictor models.

Part A - nested leave-one-study-out (LOSO) for the LightGBM-based models.
    Hyperparameters are re-selected inside every outer fold (random search scored by
    grouped 5-fold CV on the training studies only), so the held-out study never
    influences tuning. The pooled Freundlich model has no hyperparameters; its LOSO
    result is reported alongside, together with LightGBM at fixed (globally tuned)
    hyperparameters to show the size of the tuning leakage.

Part B - study-level prediction intervals for the main model (isotherm-constrained
    LightGBM, ``lgbm_splice``) and for the pooled Freundlich reference.
    Split-conformal intervals on log10(qe) with studies as the exchangeable unit:
    the interval is prediction x/÷ 10**q, where q is the study-balanced quantile of
    absolute LOSO residuals. Coverage is checked with a second, outer LOSO loop so
    that the held-out study is used neither for fitting nor for calibrating q.
    Two situations are covered: a new laboratory without any local measurement, and
    a new laboratory where one point per isotherm is known (anchored prediction).

Outputs (outputs/reports): final_nested_loso.csv, final_nested_loso_predictions.csv,
final_prediction_intervals.csv, final_prediction_intervals_by_study.csv.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor
from sklearn.model_selection import GroupKFold

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from config import CURVE_COL, GROUP_COL, RANDOM_SEED, REPORTS_DIR, TARGET  # noqa: E402
from src.analysis.grouped_validation import export_tuned_params, load_data, make_preprocessor  # noqa: E402
from src.analysis.transfer_analysis import FEATURES, _lgbm_splice, fit_predict, log10  # noqa: E402

N_CANDIDATES, INNER_FOLDS = 12, 5
SEARCH_SPACE = {
    "n_estimators": [100, 200, 400],
    "num_leaves": [4, 8, 15, 31],
    "learning_rate": [0.02, 0.05, 0.1],
    "min_child_samples": [5, 10, 20, 40],
    "subsample": [0.7, 0.85, 1.0],
    "colsample_bytree": [0.7, 1.0],
    "reg_lambda": [0.0, 1.0, 5.0],
}
COVERAGE_LEVELS = [0.80, 0.90]
INTERVAL_MODELS = ["xgb_blend", "freundlich_pooled"]  # main model and simple reference


def scores(err: np.ndarray) -> dict[str, float]:
    """err = log10(prediction) - log10(observation)."""
    return {"median_fold_error": float(10 ** np.median(np.abs(err))),
            "p90_fold_error": float(10 ** np.quantile(np.abs(err), 0.9)),
            "mae_log10": float(np.mean(np.abs(err))), "bias_log10": float(np.mean(err))}


def r2_log(obs_log: np.ndarray, err: np.ndarray) -> float:
    return float(1 - np.sum(err ** 2) / np.sum((obs_log - obs_log.mean()) ** 2))


# ---------------------------------------------------------------- Part A
def lgbm_fit_predict(train: pd.DataFrame, test: pd.DataFrame, params: dict) -> np.ndarray:
    pre = make_preprocessor(FEATURES, log_ce=True)  # first column is log10(Ce)
    x_train, x_test = pre.fit_transform(train[FEATURES]), pre.transform(test[FEATURES])
    reg = LGBMRegressor(**params, subsample_freq=1, monotone_constraints=[1] + [0] * (x_train.shape[1] - 1),
                        random_state=RANDOM_SEED, n_jobs=2, verbose=-1)
    reg.fit(x_train, log10(train[TARGET]))
    return reg.predict(x_test)


def candidates() -> list[dict]:
    rng = np.random.default_rng(RANDOM_SEED)
    return [{k: v[rng.integers(len(v))] for k, v in SEARCH_SPACE.items()} for _ in range(N_CANDIDATES)]


def select_params(train: pd.DataFrame, grid: list[dict]) -> dict:
    """Random search scored by grouped CV on the training studies."""
    folds = list(GroupKFold(INNER_FOLDS).split(train, groups=train[GROUP_COL]))
    best, best_score = grid[0], np.inf
    for params in grid:
        err = np.concatenate([lgbm_fit_predict(train.iloc[tr], train.iloc[va], params) - log10(train.iloc[va][TARGET])
                              for tr, va in folds])
        score = float(np.mean(np.abs(err)))
        if score < best_score:
            best, best_score = params, score
    return best


def nested_loso(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    grid, rf_params = candidates(), export_tuned_params()["RandomForest"]
    parts = []
    for k, study in enumerate(sorted(df[GROUP_COL].unique()), 1):
        test, train = df[df[GROUP_COL] == study], df[df[GROUP_COL] != study]
        chosen = select_params(train, grid)
        out = test[[GROUP_COL, CURVE_COL, "Ce", TARGET]].copy()
        out["log_obs"] = log10(test[TARGET])
        out["lgbm_monotone_nested"] = lgbm_fit_predict(train, test, chosen)
        out["lgbm_monotone_fixed"] = np.log10(fit_predict("lgbm_monotone", train, test, rf_params))
        out["lgbm_splice_fixed"] = np.log10(fit_predict("lgbm_splice", train, test, rf_params))
        # Main model with hyperparameters selected inside the fold
        out["lgbm_splice_nested"] = np.log10(_lgbm_splice(
            train, test, rf_params, ml_log=lambda tr, rows, p=chosen: lgbm_fit_predict(tr, rows, p)))
        out["freundlich_pooled"] = np.log10(fit_predict("freundlich_pooled", train, test, rf_params))
        out["chosen_params"] = str(chosen)
        parts.append(out)
        print(f"nested LOSO {k}/{df[GROUP_COL].nunique()}: {study}", flush=True)
    pred = pd.concat(parts)
    rows = []
    for model in ["lgbm_splice_nested", "lgbm_splice_fixed", "freundlich_pooled", "lgbm_monotone_fixed",
                  "lgbm_monotone_nested"]:
        err = (pred[model] - pred["log_obs"]).to_numpy()
        by_study = pd.Series(np.abs(err), index=pred[GROUP_COL].to_numpy()).groupby(level=0).median()
        rows.append({"model": model, "n_points": len(err), "n_studies": by_study.size, **scores(err),
                     "r2_log10": r2_log(pred["log_obs"].to_numpy(), err),
                     "median_of_study_median_fold_error": float(10 ** by_study.median()),
                     "worst_study_median_fold_error": float(10 ** by_study.max())})
    return pd.DataFrame(rows), pred


# ---------------------------------------------------------------- Part B
def model_err(train: pd.DataFrame, test: pd.DataFrame, model: str) -> np.ndarray:
    return np.log10(fit_predict(model, train, test, {})) - log10(test[TARGET])


def anchored_err(test: pd.DataFrame, err: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Residuals after shifting each isotherm to pass through one known point.

    The anchor is the point whose Ce is closest to the geometric mean Ce of the isotherm; it is
    removed from the evaluation. Returns the residuals and the positions kept.
    """
    err = pd.Series(err, index=test.index)
    keep, out = [], []
    for _, g in test.groupby(CURVE_COL):
        if len(g) < 2:
            continue
        log_ce = np.log10(g["Ce"])
        anchor = (log_ce - log_ce.mean()).abs().idxmin()
        rest = g.index.drop(anchor)
        keep.extend(rest)
        out.extend((err.loc[rest] - err.loc[anchor]).to_numpy())
    return np.asarray(out), test.index.get_indexer(keep)


def balanced_quantile(abs_err: np.ndarray, groups: np.ndarray, level: float) -> float:
    """Quantile of |residual| with every study given the same total weight."""
    sizes = pd.Series(groups).map(pd.Series(groups).value_counts()).to_numpy()
    k = len(np.unique(groups))
    target = min(1.0, level * (1 + 1 / k))  # finite-sample correction with k calibration studies
    order = np.argsort(abs_err)
    cum = np.cumsum((1 / sizes)[order]) / k
    return float(abs_err[order][min(np.searchsorted(cum, target), len(order) - 1)])


def conformal(df: pd.DataFrame, model: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    studies = sorted(df[GROUP_COL].unique())
    by_study, final_cal = [], {"new_laboratory": ([], []), "one_anchor_per_isotherm": ([], [])}
    for s in studies:
        test, rest = df[df[GROUP_COL] == s], df[df[GROUP_COL] != s]
        cal = {"new_laboratory": ([], []), "one_anchor_per_isotherm": ([], [])}
        for t in [x for x in studies if x != s]:
            held, train = rest[rest[GROUP_COL] == t], rest[rest[GROUP_COL] != t]
            e = model_err(train, held, model)
            cal["new_laboratory"][0].append(np.abs(e)); cal["new_laboratory"][1].append(np.full(len(e), t))
            ea, _ = anchored_err(held, e)
            cal["one_anchor_per_isotherm"][0].append(np.abs(ea)); cal["one_anchor_per_isotherm"][1].append(np.full(len(ea), t))
        e_test = model_err(rest, test, model)
        test_err = {"new_laboratory": e_test, "one_anchor_per_isotherm": anchored_err(test, e_test)[0]}
        for case, e in test_err.items():
            final_cal[case][0].append(np.abs(e)); final_cal[case][1].append(np.full(len(e), s))
            a, g = np.concatenate(cal[case][0]), np.concatenate(cal[case][1])
            for level in COVERAGE_LEVELS:
                q = balanced_quantile(a, g, level)
                by_study.append({"case": case, "nominal": level, GROUP_COL: s, "n_points": len(e),
                                 "interval_factor": 10 ** q, "coverage": float(np.mean(np.abs(e) <= q))})
        print(f"conformal {model}: {s}", flush=True)
    by_study = pd.DataFrame(by_study)
    by_study.insert(0, "model", model)
    rows = []
    for (case, level), g in by_study.groupby(["case", "nominal"]):
        a, grp = np.concatenate(final_cal[case][0]), np.concatenate(final_cal[case][1])
        rows.append({"model": model, "case": case, "nominal": level,
                     "interval_factor_final": 10 ** balanced_quantile(a, grp, level),
                     "coverage_points": float(np.average(g["coverage"], weights=g["n_points"])),
                     "coverage_mean_over_studies": float(g["coverage"].mean()),
                     "studies_with_coverage_below_50pct": int((g["coverage"] < 0.5).sum()),
                     "n_studies": len(g), "median_fold_error": float(10 ** np.median(a))})
    return pd.DataFrame(rows), by_study


def main() -> None:
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    df = load_data("all")
    results = [conformal(df, m) for m in INTERVAL_MODELS]
    intervals = pd.concat([r[0] for r in results], ignore_index=True)
    by_study = pd.concat([r[1] for r in results], ignore_index=True)
    intervals.to_csv(REPORTS_DIR / "final_prediction_intervals.csv", index=False)
    by_study.to_csv(REPORTS_DIR / "final_prediction_intervals_by_study.csv", index=False)
    nested, pred = nested_loso(df)
    nested.to_csv(REPORTS_DIR / "final_nested_loso.csv", index=False)
    pred.to_csv(REPORTS_DIR / "final_nested_loso_predictions.csv", index=False)

    fmt = lambda v: f"{v:.3g}"  # noqa: E731
    pd.set_option("display.width", 220)
    print("\nNested leave-one-study-out")
    print(nested.to_string(index=False, float_format=fmt))
    print("\nStudy-level prediction intervals (prediction x/÷ factor)")
    print(intervals.to_string(index=False, float_format=fmt))
    low = by_study[(by_study["nominal"] == 0.9) & (by_study["coverage"] < 0.5)]
    print("\nStudies with coverage below 50% at the 90% level")
    print(low.to_string(index=False, float_format=fmt) if len(low) else "none")


if __name__ == "__main__":
    main()
