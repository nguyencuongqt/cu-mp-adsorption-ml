"""Test of a window-anchored Freundlich tail for the spliced model.

Point anchor (current main model): the tail starts from the ML prediction exactly at Ce0.
Because a tree ensemble is a step function, that value depends on where the nearest step lies.

Window anchor: the level of the tail is the median, over a window [Ce0, w*Ce0], of the ML
prediction brought back to Ce0 along the Freundlich slope:

    log L = median_c { log qe_ML(c) - n * (log c - log Ce0) },   c in [Ce0, w*Ce0]
    qe(Ce) = L * (Ce / Ce0) ** n                                 for Ce < Ce0
    qe(Ce) = max( qe_ML(Ce), L * (min(Ce, w*Ce0) / Ce0) ** n )   for Ce >= Ce0

The second line keeps the prediction non-decreasing in Ce.

Outputs (outputs/reports): splice_anchor_test.csv, splice_anchor_river_ce.csv.
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
from src.analysis.transfer_analysis import fit_predict, iter_splits, log10  # noqa: E402

RF_PARAMS = export_tuned_params()["RandomForest"]
CATS = ["ReT", "AgS", "AdC"]
N_WINDOW = 15
# name: (percentile for Ce0, window factor w; w = 1 is the point anchor)
VARIANTS = {"point_p05": (5, 1), "window3_p05": (5, 3), "window10_p05": (5, 10),
            "point_p10": (10, 1), "window3_p10": (10, 3), "window10_p10": (10, 10),
            "point_p20": (20, 1), "window3_p20": (20, 3), "window10_p20": (20, 10)}
REFERENCES = ["freundlich_pooled", "lgbm_monotone"]
LOW_CE_THRESHOLDS = [0.1, 0.3]
RIVER_CE = [0.0003, 0.001, 0.003, 0.01, 0.1, 1.0]
SCENARIO = {"AgS": "Aged", "AdC": "Mixed"}
SCENARIO_POLYMERS = ["PE", "PP", "PS", "PET"]


def exponent(train: pd.DataFrame) -> float:
    ref = train.iloc[[0, 0]].assign(Ce=[1.0, 10.0])
    return float(np.diff(np.log10(fit_predict("freundlich_pooled", train, ref, RF_PARAMS)))[0])


def predict_log(model: str, train: pd.DataFrame, test: pd.DataFrame) -> np.ndarray:
    if model in REFERENCES:
        return np.log10(fit_predict(model, train, test, RF_PARAMS))
    percentile, w = VARIANTS[model]
    ce0, n = float(np.percentile(train["Ce"], percentile)), exponent(train)
    combos = test[CATS].drop_duplicates().reset_index(drop=True)
    window = np.logspace(np.log10(ce0), np.log10(ce0 * w), N_WINDOW) if w > 1 else np.array([ce0])
    ce = test["Ce"].to_numpy(dtype=float)
    # one ML call: the test rows (Ce clipped at Ce0) followed by the anchor window of every combination
    grid = combos.loc[np.repeat(combos.index, len(window))].assign(Ce=np.tile(window, len(combos)))
    rows = pd.concat([test[CATS].assign(Ce=np.maximum(ce, ce0)), grid], ignore_index=True)
    ml = np.log10(fit_predict("lgbm_monotone", train, rows, RF_PARAMS))
    body, anchor = ml[:len(test)], ml[len(test):].reshape(len(combos), len(window))
    level = np.median(anchor - n * (np.log10(window) - np.log10(ce0)), axis=1)
    key = test[CATS].merge(combos.assign(_i=combos.index), on=CATS, how="left")["_i"].to_numpy()
    line = level[key] + n * (np.log10(np.minimum(ce, ce0 * w)) - np.log10(ce0))
    return np.where(ce < ce0, line, np.maximum(body, line))


def evaluate(df: pd.DataFrame) -> pd.DataFrame:
    models = REFERENCES + list(VARIANTS)
    out = {m: {} for m in models}
    for tier in ["random", "leave_one_study_out"]:
        splits = list(iter_splits(tier, df))
        for m in models:
            err = np.concatenate([predict_log(m, df.iloc[tr], df.iloc[te]) - log10(df.iloc[te][TARGET]) for _, tr, te in splits])
            out[m][f"fold_error_{tier}"] = float(10 ** np.median(np.abs(err)))
        print(f"{tier} done", flush=True)
    for t in LOW_CE_THRESHOLDS:
        low, high = df[df["Ce"] <= t], df[df["Ce"] > t]
        for m in models:
            same = predict_log(m, high, low) - log10(low[TARGET])
            new = np.concatenate([predict_log(m, high[high[GROUP_COL] != s], g) - log10(g[TARGET])
                                  for s, g in low.groupby(GROUP_COL)])
            out[m][f"bias_same_lab_{t}"] = float(10 ** same.mean())
            out[m][f"bias_new_lab_{t}"] = float(10 ** new.mean())
            out[m][f"fold_error_same_lab_{t}"] = float(10 ** np.median(np.abs(same)))
            out[m][f"fold_error_new_lab_{t}"] = float(10 ** np.median(np.abs(new)))
        print(f"hold-out {t} done", flush=True)
    return pd.DataFrame(out).T.rename_axis("model").reset_index()


def river(df: pd.DataFrame) -> pd.DataFrame:
    ce = np.unique(np.concatenate([np.logspace(-4, 2, 241), RIVER_CE]))
    frame = pd.DataFrame([{"ReT": p, "Ce": c, **SCENARIO} for p in SCENARIO_POLYMERS for c in ce])
    rows = []
    for m in REFERENCES + list(VARIANTS):
        pred = predict_log(m, df, frame).reshape(len(SCENARIO_POLYMERS), len(ce))
        mean = pred.mean(axis=0)
        row = {"model": m, **{f"qe_ug_g_at_{c:g}": float(10 ** mean[np.argmin(np.abs(ce - c))] * 1e3) for c in RIVER_CE}}
        row["largest_upward_step_factor"] = float(10 ** np.diff(pred, axis=1).max())
        row["share_steps_decreasing"] = float(np.mean(np.diff(pred, axis=1) < -1e-9))
        rows.append(row)
    return pd.DataFrame(rows)


def main() -> None:
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    df = load_data("all")
    riv = river(df)
    riv.to_csv(REPORTS_DIR / "splice_anchor_river_ce.csv", index=False)
    res = evaluate(df)
    res.to_csv(REPORTS_DIR / "splice_anchor_test.csv", index=False)
    pd.set_option("display.width", 260)
    fmt = lambda v: f"{v:.3g}"  # noqa: E731
    print("\nValidation and low-Ce hold-out")
    print(res.to_string(index=False, float_format=fmt))
    print("\nPredictions at river-level Ce (ug/g) and smoothness")
    print(riv.to_string(index=False, float_format=fmt))
    for w in ["point", "window3", "window10"]:
        part = res[res["model"].str.startswith(w)]
        cols = [c for c in res.columns if c.startswith("bias_")]
        spread = riv[riv["model"].str.startswith(w)]["qe_ug_g_at_0.001"]
        print(f"{w}: bias range {part[cols].min().min():.2f}-{part[cols].max().max():.2f}; "
              f"qe at 0.001 mg/L across Ce0 choices {spread.min():.2f}-{spread.max():.2f} ug/g")


if __name__ == "__main__":
    main()
