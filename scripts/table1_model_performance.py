"""Table 1 and the low-concentration hold-out of the revised manuscript: the five models in the three validation designs (871-point dataset).

Median fold error, R2 and bias on the log10 scale for random cross-validation, leave-one-isotherm-out and
leave-one-study-out, with the ratio of each model's median fold error to that of the pooled Freundlich
model and its 95% interval from a bootstrap over studies.

Output: outputs/reports/table1_model_performance.csv
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from config import GROUP_COL, RANDOM_SEED, REPORTS_DIR, TARGET  # noqa: E402
from src.analysis.grouped_validation import export_tuned_params, load_data  # noqa: E402
from src.analysis.transfer_analysis import TIERS, fit_predict, iter_splits, log10  # noqa: E402

MODELS = ["freundlich_pooled", "xgb_blend", "xgb_monotone", "lgbm_monotone", "rf_raw"]
REFERENCE = "freundlich_pooled"
N_BOOT = 2000


def main() -> None:
    df = load_data("all")
    rf_params = export_tuned_params()["RandomForest"]
    rng = np.random.default_rng(RANDOM_SEED)
    y = np.asarray(log10(df[TARGET]))
    rows = []
    for tier in TIERS:
        pred = {m: np.zeros(len(df)) for m in MODELS}
        count = np.zeros(len(df))
        for _, tr, te in iter_splits(tier, df):
            train, test = df.iloc[tr], df.iloc[te]
            for m in MODELS:
                pred[m][te] += np.log10(fit_predict(m, train, test, rf_params))
            count[te] += 1
        err = pd.DataFrame({m: np.abs(v / count - y) for m, v in pred.items()})
        err[GROUP_COL] = df[GROUP_COL].to_numpy()
        parts = {s: g for s, g in err.groupby(GROUP_COL)}
        studies = list(parts)
        boot = [pd.concat([parts[s] for s in rng.choice(studies, len(studies))])[MODELS].median() for _ in range(N_BOOT)]
        boot = pd.DataFrame(boot)
        for m in MODELS:
            p = pred[m] / count
            diff = boot[m] - boot[REFERENCE]
            rows.append({"tier": tier, "model": m, "median_fold_error": 10 ** err[m].median(),
                         "r2_log10": 1 - ((p - y) ** 2).sum() / ((y - y.mean()) ** 2).sum(),
                         "bias_as_factor": 10 ** float(np.mean(p - y)),
                         "ratio_to_freundlich": 10 ** (err[m].median() - err[REFERENCE].median()),
                         "ratio_ci95_low": 10 ** np.percentile(diff, 2.5), "ratio_ci95_high": 10 ** np.percentile(diff, 97.5)})
        print(f"{tier} done", flush=True)
    out = pd.DataFrame(rows)
    out.to_csv(REPORTS_DIR / "table1_model_performance.csv", index=False)

    # low-concentration hold-out: models trained above a threshold and scored below it
    low_rows = []
    for threshold in (0.1, 0.3):
        low, high = df[df["Ce"] <= threshold], df[df["Ce"] > threshold]
        for m in MODELS:
            same = np.log10(fit_predict(m, high, low, rf_params)) - np.asarray(log10(low[TARGET]))
            new = np.concatenate([np.log10(fit_predict(m, high[high[GROUP_COL] != s], g, rf_params)) - np.asarray(log10(g[TARGET]))
                                  for s, g in low.groupby(GROUP_COL)])
            for design, e in (("same_lab", same), ("new_lab", new)):
                low_rows.append({"threshold_mg_L": threshold, "design": design, "model": m, "n_test": len(e),
                                 "bias_as_factor": 10 ** float(np.mean(e)), "median_fold_error": 10 ** float(np.median(np.abs(e)))})
        print(f"low-concentration hold-out {threshold} done", flush=True)
    low_table = pd.DataFrame(low_rows)
    low_table.to_csv(REPORTS_DIR / "low_ce_holdout_final.csv", index=False)

    # monotonicity of each model along Ce (share of steps on a fine grid that decrease)
    grid = np.logspace(-4, 2.3, 253)
    combos = df[["ReT", "AgS", "AdC"]].drop_duplicates().reset_index(drop=True)
    frame = combos.iloc[np.repeat(np.arange(len(combos)), len(grid))].reset_index(drop=True).assign(Ce=np.tile(grid, len(combos)))
    for m in MODELS:
        p = np.log10(fit_predict(m, df, frame, rf_params)).reshape(len(combos), len(grid))
        print(f"{m}: share of decreasing steps {np.mean(np.diff(p, axis=1) < -1e-9):.4f}")
    print(low_table.round(3).to_string(index=False))
    pd.set_option("display.width", 200)
    print(out.round(3).to_string(index=False))


if __name__ == "__main__":
    main()
