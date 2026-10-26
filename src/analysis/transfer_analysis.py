"""What the laboratory model transfers to an unseen study, and what it does not.

Three analyses on the study-grouped dataset, all with the reduced predictor set
(AgS, ReT, AdC, Ce; temperature, pH and shaking speed are left out):

1. Three validation tiers: random K-fold (interpolation), leave-one-isotherm-out
   (new series of a laboratory already seen) and leave-one-study-out (new laboratory).
2. Reference models under the same tiers: global geometric mean, polymer
   geometric mean and a pooled Freundlich regression.
3. Shape versus level under leave-one-study-out: the held-out study (or each of its
   isotherms) contributes k anchor points that fix a multiplicative offset; the
   remaining points are scored. "Anchor only" predicts the remaining points with the
   geometric mean of the anchors and shows what the anchors achieve without a model.

Outputs (outputs/reports):
    transfer_tiers.csv, transfer_per_study.csv, transfer_anchored.csv,
    transfer_within_study_rank.csv, transfer_predictions.csv
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm
from lightgbm import LGBMRegressor
from scipy.stats import spearmanr
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import LeaveOneGroupOut, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from xgboost import XGBRegressor

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from config import CURVE_COL, GROUP_COL, RANDOM_SEED, REPORTS_DIR, TARGET  # noqa: E402
from src.analysis.grouped_validation import (  # noqa: E402
    FEATURE_SETS,
    QE_FLOOR,
    export_tuned_params,
    load_data,
    make_pipeline,
    make_preprocessor,
    point_metrics,
)

FEATURES = FEATURE_SETS["core"]
MODELS = ["rf_raw", "rf_log", "freundlich_pooled", "polymer_geomean", "global_geomean"]
# Second round: isotherm-constrained alternatives
EXTRA_MODELS = ["freundlich_mixed", "xgb_log", "xgb_monotone", "lgbm_monotone"]
REFERENCE_MODELS = {"polymer_geomean", "global_geomean"}
CATEGORICAL = [c for c in FEATURES if c != "Ce"]
TIERS = ["random", "leave_one_isotherm_out", "leave_one_study_out"]
N_FOLDS, N_REPEATS_RANDOM = 5, 3
ANCHOR_SIZES = [1, 2, 3]
N_DRAWS = 200


def log10(y) -> np.ndarray:
    return np.log10(np.clip(np.asarray(y, dtype=float), QE_FLOOR, None))


def _freundlich_mixed(train: pd.DataFrame, test: pd.DataFrame) -> np.ndarray:
    """Freundlich with study-level random intercept and slope; new studies get the fixed effects."""
    encoder = OneHotEncoder(drop="first", handle_unknown="ignore", sparse_output=False)

    def design(frame: pd.DataFrame, fit: bool) -> np.ndarray:
        cats = encoder.fit_transform(frame[CATEGORICAL]) if fit else encoder.transform(frame[CATEGORICAL])
        log_ce = np.log10(frame["Ce"].to_numpy(dtype=float))
        return np.column_stack([np.ones(len(frame)), log_ce, cats])

    x_train = design(train, fit=True)
    y_train, groups = log10(train[TARGET]), train[GROUP_COL].to_numpy()
    try:
        fit = sm.MixedLM(y_train, x_train, groups=groups, exog_re=x_train[:, :2]).fit(reml=True, method="lbfgs")
    except Exception:  # fall back to a random intercept
        fit = sm.MixedLM(y_train, x_train, groups=groups).fit(reml=True)
    return np.power(10.0, design(test, fit=False) @ np.asarray(fit.fe_params))


def _boosted_log(model: str, train: pd.DataFrame, test: pd.DataFrame, params: dict) -> np.ndarray:
    """Gradient boosting on log10(qe) against log10(Ce); optionally non-decreasing in Ce."""
    pre = make_preprocessor(FEATURES, log_ce=True)  # first column is log10(Ce)
    x_train, x_test = pre.fit_transform(train[FEATURES]), pre.transform(test[FEATURES])
    constraints = [1] + [0] * (x_train.shape[1] - 1)
    if model == "lgbm_monotone":
        reg = LGBMRegressor(**params["LightGBM"], monotone_constraints=constraints, random_state=RANDOM_SEED,
                            n_jobs=-1, verbose=-1)
    else:
        extra = {"monotone_constraints": tuple(constraints)} if model == "xgb_monotone" else {}
        reg = XGBRegressor(**params["XGBoost"], objective="reg:squarederror", tree_method="hist",
                           random_state=RANDOM_SEED, n_jobs=-1, **extra)
    reg.fit(x_train, log10(train[TARGET]))
    return np.power(10.0, reg.predict(x_test))


SPLICE_PERCENTILE = 5  # Ce0 of the spliced model: this percentile of the training Ce values


SPLICE_WINDOW = 3.0     # the tail level is anchored on the ML predictions over [Ce0, SPLICE_WINDOW * Ce0]
SPLICE_WINDOW_POINTS = 15


def _lgbm_splice(train: pd.DataFrame, test: pd.DataFrame, rf_params: dict, ml_log=None) -> np.ndarray:
    """Monotone LightGBM for Ce >= Ce0; below Ce0 a tail with the pooled Freundlich exponent n.

    The level L of the tail is the median, over a window above Ce0, of the ML prediction brought
    back to Ce0 along the Freundlich slope, so it does not depend on where a single tree step lies:

        log L = median_c { log qe_ML(c) - n * (log c - log Ce0) },  c in [Ce0, w * Ce0]
        qe(Ce) = L * (Ce / Ce0) ** n                                for Ce < Ce0
        qe(Ce) = max(qe_ML(Ce), L * (min(Ce, w * Ce0) / Ce0) ** n)  for Ce >= Ce0

    The ML fit is kept where laboratory data are dense, and the prediction is non-decreasing in Ce.
    ``ml_log(train, rows)`` may supply the log10 predictions of another monotone ML body
    (used for nested tuning); the default is ``lgbm_monotone`` with the stored hyperparameters.
    """
    ce0 = float(np.percentile(train["Ce"], SPLICE_PERCENTILE))
    ref = train.iloc[[0, 0]].assign(Ce=[1.0, 10.0])
    n = float(np.diff(np.log10(fit_predict("freundlich_pooled", train, ref, rf_params)))[0])
    ce = test["Ce"].to_numpy(dtype=float)
    combos = test[CATEGORICAL].drop_duplicates().reset_index(drop=True)
    window = np.logspace(np.log10(ce0), np.log10(ce0 * SPLICE_WINDOW), SPLICE_WINDOW_POINTS)
    grid = combos.loc[np.repeat(combos.index, len(window))].assign(Ce=np.tile(window, len(combos)))
    rows = pd.concat([test[CATEGORICAL].assign(Ce=np.maximum(ce, ce0)), grid], ignore_index=True)
    ml = ml_log(train, rows) if ml_log else np.log10(fit_predict("lgbm_monotone", train, rows, rf_params))
    body, anchor = ml[:len(test)], ml[len(test):].reshape(len(combos), len(window))
    level = np.median(anchor - n * (np.log10(window) - np.log10(ce0)), axis=1)
    key = test[CATEGORICAL].merge(combos.assign(_i=combos.index), on=CATEGORICAL, how="left")["_i"].to_numpy()
    line = level[key] + n * (np.log10(np.minimum(ce, ce0 * SPLICE_WINDOW)) - np.log10(ce0))
    return np.power(10.0, np.where(ce < ce0, line, np.maximum(body, line)))


BLEND_LOW, BLEND_HIGH, BLEND_GRID = 5, 25, 60   # blending zone: percentiles of the training Ce; grid points across the zone


def _isotherm_blend(train: pd.DataFrame, test: pd.DataFrame, rf_params: dict, body: str) -> np.ndarray:
    """Isotherm-anchored ML: the ML model where laboratory data are dense, the pooled Freundlich isotherm where they are
    not, with a smooth transition.

        log qe = F + w(Ce) * (ML - F),   w = 0 below Ca, 1 above Cb (smoothstep in log Ce)

    Ca and Cb are the 5th and 25th percentiles of the training Ce, F the pooled Freundlich prediction and ML a monotone
    boosted model (``body``). Inside the zone the blend is evaluated on a fixed grid, made non-decreasing by a running
    maximum and interpolated, so the prediction is non-decreasing in Ce everywhere: the isotherm below Ca, the larger of
    the ML prediction and the value at Cb above Cb.
    """
    a, b = np.log10(np.percentile(train["Ce"], [BLEND_LOW, BLEND_HIGH]))
    lce = np.log10(test["Ce"].to_numpy(dtype=float))
    combos = test[CATEGORICAL].drop_duplicates().reset_index(drop=True)
    key = test[CATEGORICAL].merge(combos.assign(_i=combos.index), on=CATEGORICAL, how="left")["_i"].to_numpy()
    grid = np.linspace(a, b, BLEND_GRID)
    zone = combos.iloc[np.repeat(np.arange(len(combos)), len(grid))].reset_index(drop=True).assign(Ce=np.tile(10.0 ** grid, len(combos)))
    own = test[CATEGORICAL].reset_index(drop=True).assign(Ce=10.0 ** lce)
    f = np.log10(fit_predict("freundlich_pooled", train, pd.concat([zone, own], ignore_index=True), rf_params))
    ml_rows = pd.concat([zone, own.assign(Ce=10.0 ** np.maximum(lce, b))], ignore_index=True)
    ml = np.log10(fit_predict(body, train, ml_rows, rf_params))
    t = (grid - a) / (b - a)
    w = np.tile(t * t * (3.0 - 2.0 * t), len(combos))
    n_zone = len(zone)
    curve = np.maximum.accumulate((f[:n_zone] + w * (ml[:n_zone] - f[:n_zone])).reshape(len(combos), len(grid)), axis=1)
    out = f[n_zone:].copy()                                                    # below the zone: the isotherm
    inside = (lce > a) & (lce <= b)
    for i in np.unique(key[inside]):
        sel = inside & (key == i)
        out[sel] = np.interp(lce[sel], grid, curve[i])
    above = lce > b
    out[above] = np.maximum(ml[n_zone:][above], curve[key[above], -1])
    return np.power(10.0, out)


def fit_predict(model: str, train: pd.DataFrame, test: pd.DataFrame, rf_params: dict) -> np.ndarray:
    if model == "lgbm_splice":
        return _lgbm_splice(train, test, rf_params)
    if model == "xgb_splice":
        return _lgbm_splice(train, test, rf_params, ml_log=lambda tr, rows: np.log10(fit_predict("xgb_monotone", tr, rows, rf_params)))
    if model in ("xgb_blend", "lgbm_blend"):
        return _isotherm_blend(train, test, rf_params, body=model.split("_")[0] + "_monotone")
    if model == "freundlich_mixed":
        return _freundlich_mixed(train, test)
    if model in ("xgb_log", "xgb_monotone", "lgbm_monotone"):
        return _boosted_log(model, train, test, export_tuned_params())
    if model in ("rf_raw", "rf_log"):
        pipe = make_pipeline("RandomForest", rf_params, FEATURES, model.split("_")[1])
        pipe.fit(train[FEATURES], train[TARGET])
        return np.asarray(pipe.predict(test[FEATURES]), dtype=float)
    if model == "freundlich_pooled":
        pipe = Pipeline([("preprocess", make_preprocessor(FEATURES, log_ce=True)), ("model", LinearRegression())])
        pipe.fit(train[FEATURES], log10(train[TARGET]))
        return np.power(10.0, pipe.predict(test[FEATURES]))
    logs = pd.Series(log10(train[TARGET]), index=train.index)
    overall = float(logs.mean())
    if model == "global_geomean":
        return np.full(len(test), 10.0 ** overall)
    if model == "polymer_geomean":
        by_polymer = logs.groupby(train["ReT"]).mean()
        return np.power(10.0, test["ReT"].map(by_polymer).fillna(overall).to_numpy())
    raise ValueError(model)


def iter_splits(tier: str, df: pd.DataFrame):
    index = np.arange(len(df))
    if tier == "random":
        bins = pd.qcut(df[TARGET], q=10, labels=False, duplicates="drop").to_numpy()
        for rep in range(N_REPEATS_RANDOM):
            cv = StratifiedKFold(N_FOLDS, shuffle=True, random_state=RANDOM_SEED + rep)
            for tr, te in cv.split(index, bins):
                yield rep, tr, te
    else:
        groups = df[CURVE_COL] if tier == "leave_one_isotherm_out" else df[GROUP_COL]
        for tr, te in LeaveOneGroupOut().split(index, groups=groups):
            yield 0, tr, te


def out_of_fold(df: pd.DataFrame, rf_params: dict, models: list[str]) -> pd.DataFrame:
    parts = []
    for tier in TIERS:
        for model in models:
            for rep, tr, te in iter_splits(tier, df):
                pred = fit_predict(model, df.iloc[tr], df.iloc[te], rf_params)
                parts.append(pd.DataFrame({
                    "tier": tier, "model": model, "repeat": rep, GROUP_COL: df[GROUP_COL].to_numpy()[te],
                    CURVE_COL: df[CURVE_COL].to_numpy()[te], "ReT": df["ReT"].to_numpy()[te],
                    "Ce": df["Ce"].to_numpy()[te], "y": df[TARGET].to_numpy()[te], "pred": pred}))
            print(f"done {tier:24s} {model}", flush=True)
    return pd.concat(parts, ignore_index=True)


def tier_table(oof: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (tier, model), g in oof.groupby(["tier", "model"], sort=False):
        per_repeat = pd.DataFrame([point_metrics(r["y"].to_numpy(), r["pred"].to_numpy()) for _, r in g.groupby("repeat")])
        row = {"tier": tier, "model": model, "n_obs": int(len(g) / g["repeat"].nunique())}
        row.update(per_repeat.mean().to_dict())
        rows.append(row)
    return pd.DataFrame(rows)


def per_study_table(oof: pd.DataFrame) -> pd.DataFrame:
    loso = oof[oof["tier"] == "leave_one_study_out"]
    rows = []
    for (model, study), g in loso.groupby(["model", GROUP_COL]):
        err = log10(g["pred"]) - log10(g["y"])
        rows.append({"model": model, GROUP_COL: study, "n": len(g), "mean_log10_error": float(err.mean()),
                     "sd_log10_error_within": float(err.std(ddof=1)),
                     "median_fold_error": float(10 ** np.median(np.abs(err)))})
    return pd.DataFrame(rows)


def rank_table(oof: pd.DataFrame) -> pd.DataFrame:
    loso = oof[oof["tier"] == "leave_one_study_out"]
    rows = []
    for (model, study), g in loso.groupby(["model", GROUP_COL]):
        rho = spearmanr(g["y"], g["pred"]).statistic if g["pred"].nunique() > 1 else np.nan
        rows.append({"model": model, GROUP_COL: study, "n": len(g), "spearman_within_study": rho})
    return pd.DataFrame(rows)


def anchored_table(oof: pd.DataFrame) -> pd.DataFrame:
    """Score held-out points after fixing the level with k anchor points per study or per isotherm."""
    rng = np.random.default_rng(RANDOM_SEED)
    loso = oof[oof["tier"] == "leave_one_study_out"]
    rows = []
    for model in [m for m in pd.unique(loso["model"]) if m not in REFERENCE_MODELS]:
        data = loso[loso["model"] == model]
        ly, lp = log10(data["y"]), log10(data["pred"])
        for unit_name, unit in (("study", data[GROUP_COL].to_numpy()), ("isotherm", data[CURVE_COL].to_numpy())):
            groups = [np.nonzero(unit == u)[0] for u in pd.unique(unit)]
            for k in ANCHOR_SIZES:
                usable = [g for g in groups if len(g) >= k + 2]
                collected = {"unanchored": [], "model_anchored": [], "anchor_only": []}
                for _ in range(N_DRAWS):
                    errs = {name: [] for name in collected}
                    observed = []
                    for g in usable:
                        anchors = rng.choice(g, size=k, replace=False)
                        rest = np.setdiff1d(g, anchors)
                        offset = float(np.mean(ly[anchors] - lp[anchors]))
                        observed.append(ly[rest])
                        errs["unanchored"].append(lp[rest] - ly[rest])
                        errs["model_anchored"].append(lp[rest] + offset - ly[rest])
                        errs["anchor_only"].append(float(np.mean(ly[anchors])) - ly[rest])
                    obs = np.concatenate(observed)
                    total = float(np.sum((obs - obs.mean()) ** 2))
                    for name in collected:
                        e = np.concatenate(errs[name])
                        collected[name].append((10 ** np.median(np.abs(e)), np.mean(np.abs(e)),
                                                1 - float(np.sum(e ** 2)) / total))
                for name, vals in collected.items():
                    arr = np.array(vals)
                    rows.append({"model": model, "anchor_unit": unit_name, "k_anchor_points": k, "prediction": name,
                                 "n_units": len(usable), "median_fold_error": float(arr[:, 0].mean()),
                                 "mae_log10": float(arr[:, 1].mean()), "r2_log10": float(arr[:, 2].mean())})
        print(f"anchored {model}", flush=True)
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--models", nargs="+", default=MODELS, choices=MODELS + EXTRA_MODELS)
    parser.add_argument("--tag", default="", help="Suffix for the output files of this run.")
    args = parser.parse_args()
    suffix = f"_{args.tag}" if args.tag else ""

    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    df = load_data("all")
    rf_params = export_tuned_params()["RandomForest"]
    print(f"{len(df)} rows, {df[GROUP_COL].nunique()} studies, {df[CURVE_COL].nunique()} isotherms; features = {FEATURES}")

    oof = out_of_fold(df, rf_params, args.models)
    oof.to_csv(REPORTS_DIR / f"transfer_predictions{suffix}.csv", index=False)
    tiers = tier_table(oof)
    tiers.to_csv(REPORTS_DIR / f"transfer_tiers{suffix}.csv", index=False)
    per_study = per_study_table(oof)
    per_study.to_csv(REPORTS_DIR / f"transfer_per_study{suffix}.csv", index=False)
    ranks = rank_table(oof)
    ranks.to_csv(REPORTS_DIR / f"transfer_within_study_rank{suffix}.csv", index=False)
    anchored = anchored_table(oof)
    anchored.to_csv(REPORTS_DIR / f"transfer_anchored{suffix}.csv", index=False)

    pd.set_option("display.width", 200)
    print("\nValidation tiers")
    print(tiers.round(3).to_string(index=False))
    print("\nBetween- and within-study error on the log10 scale (leave-one-study-out)")
    print(per_study.groupby("model").agg(sd_between_studies=("mean_log10_error", lambda s: s.std(ddof=1)),
                                         median_sd_within_study=("sd_log10_error_within", "median")).round(3).to_string())
    print("\nMedian within-study Spearman correlation (leave-one-study-out)")
    print(ranks.groupby("model")["spearman_within_study"].median().round(3).to_string())
    print("\nAnchored transfer")
    print(anchored.round(3).to_string(index=False))


if __name__ == "__main__":
    main()
