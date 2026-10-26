"""Can the laboratory model be extended to river-level Cu concentrations?

River water holds roughly 0.001-0.01 mg/L of dissolved Cu, below nearly all of the
laboratory isotherm data. Four checks, all with the reduced predictor set:

1. Data support: points, studies and isotherms below each Ce threshold.
2. Hold-out by concentration: train on Ce > T, score on Ce <= T, for several T.
   "same_lab" keeps the high-Ce points of the scored study in the training set;
   "new_lab" also leaves the whole scored study out.
3. Observed low-Ce slope: d log10(qe) / d log10(Ce) of each isotherm, fitted on its
   points at Ce <= 1 mg/L. A slope near zero means qe has levelled off; a slope
   between 0 and 1 means qe keeps falling as Ce falls.
4. Predicted qe at river-level Ce from models fitted on all data.

Outputs (outputs/reports): low_ce_support.csv, low_ce_holdout.csv,
low_ce_observed_slopes.csv, low_ce_predicted_curves.csv; figure low_ce_extrapolation.png.
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from config import CURVE_COL, FIGURES_DIR, GROUP_COL, REPORTS_DIR, TARGET  # noqa: E402
from src.analysis.grouped_validation import export_tuned_params, load_data  # noqa: E402
from src.analysis.transfer_analysis import fit_predict, log10  # noqa: E402

MODELS = ["lgbm_monotone", "xgb_monotone", "freundlich_pooled", "rf_raw", "rf_log"]
THRESHOLDS_MG_L = [0.1, 0.3, 1.0]
SUPPORT_LEVELS_MG_L = [0.001, 0.003, 0.01, 0.03, 0.1, 0.3, 1.0]
SLOPE_MAX_CE, SLOPE_MIN_POINTS = 1.0, 3
RIVER_CE_MG_L = [0.001, 0.003, 0.01, 0.03, 0.1, 1.0]
SCENARIO = {"AgS": "Aged", "AdC": "Mixed"}
SCENARIO_POLYMERS = ["PE", "PP", "PS", "PET"]


def support_table(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for level in SUPPORT_LEVELS_MG_L:
        part = df[df["Ce"] <= level]
        rows.append({"Ce_at_or_below_mg_L": level, "points": len(part), "studies": part[GROUP_COL].nunique(),
                     "isotherms": part[CURVE_COL].nunique()})
    return pd.DataFrame(rows)


def score(y: np.ndarray, pred: np.ndarray) -> dict[str, float]:
    err = log10(pred) - log10(y)
    return {"median_fold_error": float(10 ** np.median(np.abs(err))), "mae_log10": float(np.mean(np.abs(err))),
            "mean_bias_log10": float(np.mean(err)), "bias_as_factor": float(10 ** np.mean(err))}


def holdout_table(df: pd.DataFrame, rf_params: dict) -> pd.DataFrame:
    rows = []
    for threshold in THRESHOLDS_MG_L:
        low, high = df[df["Ce"] <= threshold], df[df["Ce"] > threshold]
        for model in MODELS:
            pred = fit_predict(model, high, low, rf_params)
            rows.append({"threshold_mg_L": threshold, "design": "same_lab", "model": model, "n_test": len(low),
                         "test_studies": low[GROUP_COL].nunique(), **score(low[TARGET].to_numpy(), pred)})
            obs, est = [], []
            for study, test in low.groupby(GROUP_COL):
                train = high[high[GROUP_COL] != study]
                obs.append(test[TARGET].to_numpy())
                est.append(fit_predict(model, train, test, rf_params))
            rows.append({"threshold_mg_L": threshold, "design": "new_lab", "model": model, "n_test": len(low),
                         "test_studies": low[GROUP_COL].nunique(), **score(np.concatenate(obs), np.concatenate(est))})
        print(f"hold-out at {threshold} mg/L done", flush=True)
    return pd.DataFrame(rows)


def observed_slopes(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (study, series), g in df[df["Ce"] <= SLOPE_MAX_CE].groupby([GROUP_COL, CURVE_COL]):
        if len(g) < SLOPE_MIN_POINTS or g["Ce"].nunique() < SLOPE_MIN_POINTS:
            continue
        slope = np.polyfit(np.log10(g["Ce"]), log10(g[TARGET]), 1)[0]
        rows.append({GROUP_COL: study, CURVE_COL: series, "n_points": len(g), "Ce_min_mg_L": g["Ce"].min(),
                     "Ce_max_mg_L": g["Ce"].max(), "slope_log_log": float(slope)})
    return pd.DataFrame(rows)


def predicted_curves(df: pd.DataFrame, rf_params: dict) -> pd.DataFrame:
    grid = np.logspace(-4, 1, 101)
    ce = np.unique(np.concatenate([grid, RIVER_CE_MG_L]))
    frame = pd.DataFrame([{"ReT": p, "Ce": c, **SCENARIO} for p in SCENARIO_POLYMERS for c in ce])
    out = []
    for model in MODELS:
        part = frame.copy()
        part["model"] = model
        part["qe_pred_mg_g"] = fit_predict(model, df, frame, rf_params)
        out.append(part)
    return pd.concat(out, ignore_index=True)


def save_figure(df: pd.DataFrame, curves: pd.DataFrame) -> Path:
    mean = curves.groupby(["model", "Ce"])["qe_pred_mg_g"].apply(lambda s: 10 ** np.mean(np.log10(s))).reset_index()
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.scatter(df["Ce"], df[TARGET], s=6, color="0.75", label="laboratory data", zorder=1)
    for model, g in mean.groupby("model"):
        ax.plot(g["Ce"], g["qe_pred_mg_g"], label=model, zorder=2)
    ax.axvspan(0.001, 0.01, color="tab:blue", alpha=0.12, label="typical river Ce")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Ce (mg/L)")
    ax.set_ylabel("qe (mg/g)")
    ax.legend(fontsize=8)
    fig.tight_layout()
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    path = FIGURES_DIR / "low_ce_extrapolation.png"
    fig.savefig(path, dpi=300)
    plt.close(fig)
    return path


def main() -> None:
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    df = load_data("all")
    rf_params = export_tuned_params()["RandomForest"]

    support = support_table(df)
    support.to_csv(REPORTS_DIR / "low_ce_support.csv", index=False)
    holdout = holdout_table(df, rf_params)
    holdout.to_csv(REPORTS_DIR / "low_ce_holdout.csv", index=False)
    slopes = observed_slopes(df)
    slopes.to_csv(REPORTS_DIR / "low_ce_observed_slopes.csv", index=False)
    curves = predicted_curves(df, rf_params)
    curves.to_csv(REPORTS_DIR / "low_ce_predicted_curves.csv", index=False)
    figure = save_figure(df, curves)

    pd.set_option("display.width", 200)
    print("\nData support"); print(support.to_string(index=False))
    print("\nHold-out by concentration"); print(holdout.round(3).to_string(index=False))
    low_part = slopes[slopes["Ce_min_mg_L"] <= 0.1]
    print(f"\nObserved log-log slope at Ce <= {SLOPE_MAX_CE} mg/L: {len(slopes)} isotherms from "
          f"{slopes[GROUP_COL].nunique()} studies")
    print(slopes["slope_log_log"].describe(percentiles=[0.1, 0.25, 0.5, 0.75, 0.9]).round(3).to_string())
    print(f"Isotherms reaching Ce <= 0.1 mg/L ({len(low_part)}): median slope {low_part['slope_log_log'].median():.3f}")
    table = curves[curves["Ce"].isin(RIVER_CE_MG_L)].groupby(["model", "Ce"])["qe_pred_mg_g"].apply(
        lambda s: 10 ** np.mean(np.log10(s))).unstack("Ce")
    print("\nPredicted qe (mg/g), geometric mean over PE, PP, PS, PET; aged, mixed system")
    print(table.round(5).to_string())
    print("\nRatio of qe at 1 mg/L to qe at 0.001 mg/L:")
    print((table[1.0] / table[0.001]).round(1).to_string())
    print(f"\nFigure: {figure}")


if __name__ == "__main__":
    main()
