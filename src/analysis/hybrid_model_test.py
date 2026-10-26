"""Test of an isotherm-informed hybrid model: Freundlich backbone + ML on the residuals.

    log10(qe) = Freundlich_pooled(log10 Ce, polymer, aging, system) + f_ML(same predictors)

f_ML is a tree ensemble trained on the residuals of the Freundlich fit. Inside the
laboratory domain it can correct curvature and interactions; outside the Ce range of
the data a tree ensemble returns a constant, so the hybrid keeps the Freundlich slope.

Compared with pooled Freundlich, monotone LightGBM and the unconstrained Random Forest on
(1) random CV, leave-one-isotherm-out and leave-one-study-out, (2) hold-out of the low-Ce
points, (3) predictions at river-level Ce and (4) monotonicity along Ce.

Outputs (outputs/reports): hybrid_test_validation.csv, hybrid_test_low_ce.csv,
hybrid_test_river_ce.csv.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from config import GROUP_COL, RANDOM_SEED, REPORTS_DIR, TARGET  # noqa: E402
from src.analysis.grouped_validation import export_tuned_params, load_data, make_preprocessor  # noqa: E402
from src.analysis.transfer_analysis import FEATURES, TIERS, fit_predict, iter_splits, log10  # noqa: E402

HYBRIDS = ["hybrid_lgbm", "hybrid_rf"]
MODELS = ["freundlich_pooled", *HYBRIDS, "lgbm_monotone", "rf_raw"]
LOW_CE_THRESHOLD = 0.1
RIVER_CE = [0.0003, 0.001, 0.003, 0.01, 0.1, 1.0]
SCENARIO = {"AgS": "Aged", "AdC": "Mixed"}
SCENARIO_POLYMERS = ["PE", "PP", "PS", "PET"]
PARAMS = export_tuned_params()


def predict_log(model: str, train: pd.DataFrame, test: pd.DataFrame) -> np.ndarray:
    if model not in HYBRIDS:
        return np.log10(fit_predict(model, train, test, PARAMS["RandomForest"]))
    pre = make_preprocessor(FEATURES, log_ce=True)
    x_train, x_test = pre.fit_transform(train[FEATURES]), pre.transform(test[FEATURES])
    backbone = LinearRegression().fit(x_train, log10(train[TARGET]))
    resid = log10(train[TARGET]) - backbone.predict(x_train)
    if model == "hybrid_lgbm":
        ml = LGBMRegressor(**PARAMS["LightGBM"], random_state=RANDOM_SEED, n_jobs=2, verbose=-1)
    else:
        ml = RandomForestRegressor(**PARAMS["RandomForest"], random_state=RANDOM_SEED, n_jobs=-1)
    ml.fit(x_train, resid)
    return backbone.predict(x_test) + ml.predict(x_test)


def summary(err: np.ndarray, obs_log: np.ndarray) -> dict[str, float]:
    return {"median_fold_error": float(10 ** np.median(np.abs(err))),
            "p90_fold_error": float(10 ** np.quantile(np.abs(err), 0.9)),
            "r2_log10": float(1 - np.sum(err ** 2) / np.sum((obs_log - obs_log.mean()) ** 2)),
            "bias_as_factor": float(10 ** np.mean(err))}


def validation(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for tier in TIERS:
        splits = list(iter_splits(tier, df))
        for model in MODELS:
            err, obs = [], []
            for _, tr, te in splits:
                train, test = df.iloc[tr], df.iloc[te]
                err.append(predict_log(model, train, test) - log10(test[TARGET]))
                obs.append(log10(test[TARGET]))
            rows.append({"tier": tier, "model": model, **summary(np.concatenate(err), np.concatenate(obs))})
            print(f"{tier}: {model} done", flush=True)
    return pd.DataFrame(rows)


def low_ce(df: pd.DataFrame) -> pd.DataFrame:
    low, high = df[df["Ce"] <= LOW_CE_THRESHOLD], df[df["Ce"] > LOW_CE_THRESHOLD]
    rows = []
    for model in MODELS:
        e_same = predict_log(model, high, low) - log10(low[TARGET])
        e_new = np.concatenate([predict_log(model, high[high[GROUP_COL] != s], g) - log10(g[TARGET])
                                for s, g in low.groupby(GROUP_COL)])
        for design, e in [("same_lab", e_same), ("new_lab", e_new)]:
            rows.append({"design": design, "model": model, "n_test": len(e),
                         "median_fold_error": float(10 ** np.median(np.abs(e))), "bias_as_factor": float(10 ** np.mean(e))})
    return pd.DataFrame(rows)


def river_predictions(df: pd.DataFrame) -> pd.DataFrame:
    grid = np.logspace(-4, 2, 121)
    ce = np.unique(np.concatenate([grid, RIVER_CE]))
    frame = pd.DataFrame([{"ReT": p, "Ce": c, **SCENARIO} for p in SCENARIO_POLYMERS for c in ce])
    rows = []
    for model in MODELS:
        pred = predict_log(model, df, frame).reshape(len(SCENARIO_POLYMERS), len(ce))
        mean = pred.mean(axis=0)
        slope = np.diff(mean) / np.diff(np.log10(ce))
        mid = np.sqrt(ce[1:] * ce[:-1])
        row = {"model": model}
        row.update({f"qe_ug_g_at_{c:g}_mg_L": float(10 ** mean[np.argmin(np.abs(ce - c))] * 1e3) for c in RIVER_CE})
        row["slope_Ce_below_0.01"] = float(slope[mid < 0.01].mean())
        row["slope_Ce_0.1_to_10"] = float(slope[(mid >= 0.1) & (mid < 10)].mean())
        row["share_of_grid_steps_decreasing"] = float(np.mean(np.diff(pred, axis=1) < -1e-9))
        rows.append(row)
    return pd.DataFrame(rows)


def main() -> None:
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    df = load_data("all")
    val = validation(df)
    val.to_csv(REPORTS_DIR / "hybrid_test_validation.csv", index=False)
    low = low_ce(df)
    low.to_csv(REPORTS_DIR / "hybrid_test_low_ce.csv", index=False)
    river = river_predictions(df)
    river.to_csv(REPORTS_DIR / "hybrid_test_river_ce.csv", index=False)

    pd.set_option("display.width", 240)
    fmt = lambda v: f"{v:.3g}"  # noqa: E731
    print("\nMedian fold error by validation tier")
    print(val.pivot(index="model", columns="tier", values="median_fold_error")[TIERS].loc[MODELS].to_string(float_format=fmt))
    print("\nR2 on log10(qe) by validation tier")
    print(val.pivot(index="model", columns="tier", values="r2_log10")[TIERS].loc[MODELS].to_string(float_format=fmt))
    print(f"\nHold-out of Ce <= {LOW_CE_THRESHOLD} mg/L (trained on higher Ce only)")
    print(low.to_string(index=False, float_format=fmt))
    print("\nPredictions at river-level Ce (geometric mean of PE, PP, PS, PET; aged, mixed)")
    print(river.to_string(index=False, float_format=fmt))


if __name__ == "__main__":
    main()
