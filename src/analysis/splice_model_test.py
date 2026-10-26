"""Test of ML models with an isotherm-constrained low-concentration tail.

"splice": monotone LightGBM is used where laboratory data are dense (Ce >= Ce0). Below
Ce0 the prediction continues from the ML value at Ce0 with the pooled Freundlich exponent n:

    qe(Ce) = qe_ML(Ce0; polymer, aging, system) * (Ce / Ce0) ** n        for Ce < Ce0

Ce0 is a low percentile of the training Ce values, so it is set from the data and moves
with the training set. The result is monotone in Ce by construction, keeps the ML fit
inside the laboratory domain, and has a physically based slope outside it.

"ensemble": geometric mean of monotone LightGBM and pooled Freundlich predictions.
"ensemble_splice": the ensemble with the same Freundlich tail below Ce0.

Evaluated like hybrid_model_test.py. Outputs (outputs/reports): splice_test_validation.csv,
splice_test_low_ce.csv, splice_test_river_ce.csv.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from config import GROUP_COL, REPORTS_DIR, TARGET  # noqa: E402
from src.analysis.grouped_validation import export_tuned_params, load_data  # noqa: E402
from src.analysis.transfer_analysis import TIERS, fit_predict, iter_splits, log10  # noqa: E402

SPLICE_PERCENTILES = {"splice_p05": 5, "splice_p10": 10, "splice_p20": 20}
MODELS = ["freundlich_pooled", "lgbm_monotone", *SPLICE_PERCENTILES, "ensemble", "ensemble_splice_p10", "rf_raw"]
LOW_CE_THRESHOLDS = [0.1, 0.3]
RIVER_CE = [0.0003, 0.001, 0.003, 0.01, 0.1, 1.0]
SCENARIO = {"AgS": "Aged", "AdC": "Mixed"}
SCENARIO_POLYMERS = ["PE", "PP", "PS", "PET"]
RF_PARAMS = export_tuned_params()["RandomForest"]


def freundlich_exponent(train: pd.DataFrame) -> float:
    ref = train.iloc[[0, 0]].copy()
    ref["Ce"] = [1.0, 10.0]
    p = np.log10(fit_predict("freundlich_pooled", train, ref, RF_PARAMS))
    return float(p[1] - p[0])


def base_log(model: str, train: pd.DataFrame, test: pd.DataFrame) -> np.ndarray:
    return np.log10(fit_predict(model, train, test, RF_PARAMS))


def predict_log(model: str, train: pd.DataFrame, test: pd.DataFrame) -> np.ndarray:
    if model in ("freundlich_pooled", "lgbm_monotone", "rf_raw"):
        return base_log(model, train, test)
    if model == "ensemble":
        return 0.5 * (base_log("lgbm_monotone", train, test) + base_log("freundlich_pooled", train, test))
    percentile = SPLICE_PERCENTILES.get(model.replace("ensemble_", ""))
    ce0 = float(np.percentile(train["Ce"], percentile))
    clipped = test.copy()
    clipped["Ce"] = np.maximum(test["Ce"].to_numpy(dtype=float), ce0)
    body = base_log("lgbm_monotone", train, clipped)
    if model.startswith("ensemble"):
        body = 0.5 * (body + base_log("freundlich_pooled", train, clipped))
    tail = np.minimum(np.log10(test["Ce"].to_numpy(dtype=float)) - np.log10(ce0), 0.0)
    return body + freundlich_exponent(train) * tail


def summary(err: np.ndarray, obs_log: np.ndarray) -> dict[str, float]:
    return {"median_fold_error": float(10 ** np.median(np.abs(err))),
            "r2_log10": float(1 - np.sum(err ** 2) / np.sum((obs_log - obs_log.mean()) ** 2))}


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
    rows = []
    for threshold in LOW_CE_THRESHOLDS:
        low, high = df[df["Ce"] <= threshold], df[df["Ce"] > threshold]
        for model in MODELS:
            e_same = predict_log(model, high, low) - log10(low[TARGET])
            e_new = np.concatenate([predict_log(model, high[high[GROUP_COL] != s], g) - log10(g[TARGET])
                                    for s, g in low.groupby(GROUP_COL)])
            for design, e in [("same_lab", e_same), ("new_lab", e_new)]:
                rows.append({"threshold_mg_L": threshold, "design": design, "model": model, "n_test": len(e),
                             "median_fold_error": float(10 ** np.median(np.abs(e))),
                             "bias_as_factor": float(10 ** np.mean(e))})
    return pd.DataFrame(rows)


def river_predictions(df: pd.DataFrame) -> pd.DataFrame:
    ce = np.unique(np.concatenate([np.logspace(-4, 2, 121), RIVER_CE]))
    frame = pd.DataFrame([{"ReT": p, "Ce": c, **SCENARIO} for p in SCENARIO_POLYMERS for c in ce])
    rows = []
    for model in MODELS:
        pred = predict_log(model, df, frame).reshape(len(SCENARIO_POLYMERS), len(ce))
        mean = pred.mean(axis=0)
        slope, mid = np.diff(mean) / np.diff(np.log10(ce)), np.sqrt(ce[1:] * ce[:-1])
        row = {"model": model}
        row.update({f"qe_ug_g_at_{c:g}": float(10 ** mean[np.argmin(np.abs(ce - c))] * 1e3) for c in RIVER_CE})
        row["slope_below_0.01"] = float(slope[mid < 0.01].mean())
        row["share_steps_decreasing"] = float(np.mean(np.diff(pred, axis=1) < -1e-9))
        rows.append(row)
    return pd.DataFrame(rows)


def main() -> None:
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    df = load_data("all")
    print("Ce0 on the full data (mg/L): " + ", ".join(f"{k} = {np.percentile(df['Ce'], v):.3g}"
                                                      for k, v in SPLICE_PERCENTILES.items()))
    print(f"Freundlich exponent: {freundlich_exponent(df):.3f}")
    low = low_ce(df)
    low.to_csv(REPORTS_DIR / "splice_test_low_ce.csv", index=False)
    river = river_predictions(df)
    river.to_csv(REPORTS_DIR / "splice_test_river_ce.csv", index=False)
    val = validation(df)
    val.to_csv(REPORTS_DIR / "splice_test_validation.csv", index=False)

    pd.set_option("display.width", 240)
    fmt = lambda v: f"{v:.3g}"  # noqa: E731
    print("\nMedian fold error by validation tier")
    print(val.pivot(index="model", columns="tier", values="median_fold_error")[TIERS].loc[MODELS].to_string(float_format=fmt))
    print("\nR2 on log10(qe)")
    print(val.pivot(index="model", columns="tier", values="r2_log10")[TIERS].loc[MODELS].to_string(float_format=fmt))
    print("\nHold-out of low-Ce points (trained on higher Ce only): bias as factor / median fold error")
    wide = low.pivot_table(index="model", columns=["threshold_mg_L", "design"], values=["bias_as_factor", "median_fold_error"])
    print(wide.loc[MODELS].to_string(float_format=fmt))
    print("\nPredictions at river-level Ce (ug/g; geometric mean of PE, PP, PS, PET; aged, mixed)")
    print(river.to_string(index=False, float_format=fmt))


if __name__ == "__main__":
    main()
