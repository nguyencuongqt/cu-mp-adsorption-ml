"""Can an ML model be the selected predictor? Scenarios compared on every criterion used for model selection.

Candidates
  freundlich_pooled  linear isotherm-form model (currently selected)
  xgb_monotone       monotone XGBoost (best inside the laboratory domain)
  lgbm_splice        isotherm-constrained LightGBM (Freundlich tail below the 5th percentile of Ce)
  xgb_splice         the same with XGBoost as the ML body
  xgb_blend          XGBoost where data are dense, Freundlich where they are not, smooth transition (5th-25th percentile of Ce)
  lgbm_blend         the same with LightGBM

Criteria: median fold error in the three validation designs with the ratio to the Freundlich model (bootstrap over
studies); bias below 0.1 and 0.3 mg/L of models trained above; error with one local measurement per isotherm;
monotonicity along Ce; field-to-laboratory ratio and global estimate.

Output: outputs/reports/ml_role_scenarios.csv
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
from src.analysis.faf_scaling import PRIMARY_SUBSET, load_field_studies, study_level_faf, study_ratios  # noqa: E402
from src.analysis.global_monte_carlo import GLOBAL_POLYMERS, GLOBAL_WEIGHTS, lab_curve, polymer_curves  # noqa: E402
from src.analysis.grouped_validation import export_tuned_params, load_data  # noqa: E402
from src.analysis.transfer_analysis import TIERS, fit_predict, iter_splits, log10  # noqa: E402

MODELS = ["freundlich_pooled", "xgb_monotone", "lgbm_splice", "xgb_splice", "xgb_blend", "lgbm_blend"]
REFERENCE = "freundlich_pooled"
N_BOOT = 2000
ISOTHERM_COL = "curve_id"
CE_WORLD_MG_L = 23.3e-9 * 63.546 * 1e3
MP_FLUX_KT = 53.0


def main() -> None:
    df = load_data("all").reset_index(drop=True)
    rf = export_tuned_params()["RandomForest"]
    rng = np.random.default_rng(RANDOM_SEED)
    y = np.asarray(log10(df[TARGET]))
    out = {m: {} for m in MODELS}

    # 1. validation designs
    loso_pred = {}
    for tier in TIERS:
        pred = {m: np.zeros(len(df)) for m in MODELS}
        count = np.zeros(len(df))
        for _, tr, te in iter_splits(tier, df):
            for m in MODELS:
                pred[m][te] += np.log10(fit_predict(m, df.iloc[tr], df.iloc[te], rf))
            count[te] += 1
        err = pd.DataFrame({m: np.abs(v / count - y) for m, v in pred.items()})
        err[GROUP_COL] = df[GROUP_COL].to_numpy()
        parts = {s: g for s, g in err.groupby(GROUP_COL)}
        boot = pd.DataFrame([pd.concat([parts[s] for s in rng.choice(list(parts), len(parts))])[MODELS].median() for _ in range(N_BOOT)])
        for m in MODELS:
            d = boot[m] - boot[REFERENCE]
            out[m][f"mfe_{tier}"] = 10 ** err[m].median()
            out[m][f"ratio_{tier}"] = f"{10 ** (err[m].median() - err[REFERENCE].median()):.2f} ({10 ** np.percentile(d, 2.5):.2f}-{10 ** np.percentile(d, 97.5):.2f})"
        if tier == "leave_one_study_out":
            loso_pred = {m: v / count for m, v in pred.items()}
        print(f"{tier} done", flush=True)

    # 2. one local measurement per isotherm (same anchor point for every model)
    anchors = {}
    for iso, g in df.groupby(ISOTHERM_COL):
        if len(g) >= 2:
            anchors[iso] = int(rng.choice(g.index.to_numpy()))
    for m in MODELS:
        errs = []
        for iso, g in df.groupby(ISOTHERM_COL):
            if iso in anchors:
                a = anchors[iso]
                shift = y[a] - loso_pred[m][a]
                rest = g.index.to_numpy()[g.index.to_numpy() != a]
                errs.append(np.abs(loso_pred[m][rest] + shift - y[rest]))
        out[m]["mfe_one_local_measurement"] = 10 ** np.median(np.concatenate(errs))

    # 3. low-concentration hold-out
    for threshold in (0.1, 0.3):
        low, high = df[df["Ce"] <= threshold], df[df["Ce"] > threshold]
        for m in MODELS:
            same = np.log10(fit_predict(m, high, low, rf)) - log10(low[TARGET])
            new = np.concatenate([np.log10(fit_predict(m, high[high[GROUP_COL] != s], g, rf)) - log10(g[TARGET]) for s, g in low.groupby(GROUP_COL)])
            out[m][f"bias_below_{threshold}_same_lab"] = 10 ** np.mean(same)
            out[m][f"bias_below_{threshold}_new_lab"] = 10 ** np.mean(new)
        print(f"low-Ce {threshold} done", flush=True)

    # 4. monotonicity, field ratio and global estimate (model fitted to all data)
    studies = load_field_studies()
    grid = np.logspace(-4, 2.3, 127)
    combos = df[["ReT", "AgS", "AdC"]].drop_duplicates()
    frame = combos.loc[np.repeat(combos.index, len(grid))].assign(Ce=np.tile(grid, len(combos)))
    for m in MODELS:
        p = np.log10(fit_predict(m, df, frame, rf)).reshape(len(combos), len(grid))
        out[m]["share_of_steps_decreasing"] = float(np.mean(np.diff(p, axis=1) < -1e-6))
        out[m]["slope_below_0.01_mg_L"] = float(np.mean(np.diff(p, axis=1)[:, grid[1:] < 0.01] / np.diff(np.log10(grid))[grid[1:] < 0.01]))
        ratio = study_level_faf(study_ratios(m, df, studies), PRIMARY_SUBSET)
        gm = float(np.exp(np.log(ratio).mean()))
        qe = float(lab_curve(polymer_curves(m, df, GLOBAL_POLYMERS), GLOBAL_POLYMERS, GLOBAL_WEIGHTS)(np.array([CE_WORLD_MG_L]))[0])
        out[m].update(field_ratio_gm=gm, field_ratio_max_over_min=float(ratio.max() / ratio.min()), qe_lab_ug_g_at_1p48=qe * 1e3,
                      global_kg_yr_53kt=MP_FLUX_KT * qe * gm * 1e3)
    table = pd.DataFrame(out).T
    table.to_csv(REPORTS_DIR / "ml_role_scenarios.csv")
    pd.set_option("display.width", 250)
    pd.set_option("display.max_columns", 40)
    num = table.apply(pd.to_numeric, errors="coerce")
    print(num.dropna(axis=1, how="all").round(3).T.to_string())
    print(table[[c for c in table.columns if c.startswith("ratio_")]].T.to_string())


if __name__ == "__main__":
    main()
