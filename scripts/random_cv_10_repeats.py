"""Random cross-validation with 10 repeats (R3-6, R4-6): per-repeat metrics, their mean and SD, and the ratio of each
model's median fold error to that of the pooled Freundlich model with a 95% interval from a bootstrap over studies.

Replaces the random-CV rows of outputs/reports/table1_model_performance.csv (3 repeats, metrics of the predictions
averaged over repeats) by the mean over 10 repeats, and writes outputs/reports/random_cv_by_repeat.csv.
The leave-one-isotherm-out and leave-one-study-out rows are not changed (single partition).
Usage: python scripts/random_cv_10_repeats.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from config import GROUP_COL, RANDOM_SEED, REPORTS_DIR, TARGET  # noqa: E402
from src.analysis import transfer_analysis as ta  # noqa: E402
from src.analysis.grouped_validation import export_tuned_params, load_data  # noqa: E402

N_REPEATS = 10
MODELS = ["freundlich_pooled", "xgb_blend", "xgb_monotone", "lgbm_monotone", "rf_raw"]
REFERENCE = "freundlich_pooled"
N_BOOT = 2000


def main() -> None:
    ta.N_REPEATS_RANDOM = N_REPEATS
    df = load_data("all")
    rf_params = export_tuned_params()["RandomForest"]
    y = np.asarray(ta.log10(df[TARGET]))
    pred = {(m, r): np.full(len(df), np.nan) for m in MODELS for r in range(N_REPEATS)}
    for rep, tr, te in ta.iter_splits("random", df):
        for m in MODELS:
            pred[(m, rep)][te] = np.log10(ta.fit_predict(m, df.iloc[tr], df.iloc[te], rf_params))
        print(f"repeat {rep}: fold done", flush=True)
    rows = []
    for (m, r), p in pred.items():
        rows.append({"model": m, "repeat": r, "median_fold_error": 10 ** np.median(np.abs(p - y)),
                     "r2_log10": 1 - ((p - y) ** 2).sum() / ((y - y.mean()) ** 2).sum(), "bias_as_factor": 10 ** float(np.mean(p - y))})
    by_rep = pd.DataFrame(rows)
    agg = by_rep.groupby("model").agg(mfe_mean=("median_fold_error", "mean"), mfe_sd=("median_fold_error", "std"),
                                      mfe_min=("median_fold_error", "min"), mfe_max=("median_fold_error", "max"),
                                      r2_mean=("r2_log10", "mean"), r2_sd=("r2_log10", "std"),
                                      bias_mean=("bias_as_factor", "mean")).reset_index()
    by_rep.merge(agg, on="model").to_csv(REPORTS_DIR / "random_cv_by_repeat.csv", index=False)

    # ratio to the linear model: per repeat, then averaged; 95% interval from a bootstrap over studies
    err = {k: np.abs(v - y) for k, v in pred.items()}
    studies = df[GROUP_COL].to_numpy()
    uniq = pd.unique(studies)
    idx = {s: np.nonzero(studies == s)[0] for s in uniq}
    rng = np.random.default_rng(RANDOM_SEED)
    boot = {m: [] for m in MODELS}
    for _ in range(N_BOOT):
        rows_b = np.concatenate([idx[s] for s in rng.choice(uniq, len(uniq))])
        ref = np.array([np.median(err[(REFERENCE, r)][rows_b]) for r in range(N_REPEATS)])
        for m in MODELS:
            mm = np.array([np.median(err[(m, r)][rows_b]) for r in range(N_REPEATS)])
            boot[m].append(10 ** np.mean(mm - ref))
    ratio = {m: 10 ** np.mean([np.median(err[(m, r)]) - np.median(err[(REFERENCE, r)]) for r in range(N_REPEATS)]) for m in MODELS}

    t1 = pd.read_csv(REPORTS_DIR / "table1_model_performance.csv")
    for m in MODELS:
        a = agg.set_index("model").loc[m]
        sel = (t1["tier"] == "random") & (t1["model"] == m)
        t1.loc[sel, ["median_fold_error", "r2_log10", "bias_as_factor", "ratio_to_freundlich", "ratio_ci95_low", "ratio_ci95_high"]] = [
            a["mfe_mean"], a["r2_mean"], a["bias_mean"], ratio[m], np.percentile(boot[m], 2.5), np.percentile(boot[m], 97.5)]
    t1.to_csv(REPORTS_DIR / "table1_model_performance.csv", index=False)
    pd.set_option("display.width", 200)
    print(agg.round(4).to_string(index=False))
    print(t1[t1["tier"] == "random"].round(3).to_string(index=False))


if __name__ == "__main__":
    main()
