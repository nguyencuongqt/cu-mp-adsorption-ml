"""Do particle size or specific surface area (SSA) improve the model? Restricted-sample sensitivity analysis.

For each variable, only the isotherm series with a reported value are used (no imputation), and the
core model (polymer, ageing status, system, Ce) is compared with core + log10(variable) on exactly
the same observations:

1. pooled Freundlich model: coefficient of log10(variable) with a 95% interval from a bootstrap over
   studies; leave-one-study-out (LOSO) error; between-study SD of the residuals (how much of the
   difference between laboratories the variable explains);
2. monotone LightGBM: LOSO error with and without the variable.

Outputs (outputs/reports): size_ssa_sensitivity.csv
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor
from sklearn.linear_model import LinearRegression

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from config import CURVE_COL, DATA_DIR, GROUP_COL, RANDOM_SEED, REPORTS_DIR, TARGET  # noqa: E402
from src.analysis.grouped_validation import export_tuned_params, load_data  # noqa: E402

N_BOOT = 1000
CATS = ["ReT", "AgS", "AdC"]
CASES = {  # label: (column, filter on metadata)
    "SSA": ("ssa_m2_g", None),
    "size, all assigned series": ("size_um", None),
    "size, without upper limits": ("size_um", lambda m: m["size_kind"].isin(["value", "range"])),
}


def design(df: pd.DataFrame, extra: bool, columns: list[str] | None = None) -> pd.DataFrame:
    x = pd.get_dummies(df[CATS], dtype=float)
    x.insert(0, "logCe", np.log10(df["Ce"].to_numpy()))
    if extra:
        x.insert(1, "logX", np.log10(df["x"].to_numpy()))
    return x if columns is None else x.reindex(columns=columns, fill_value=0.0)


def loso(df: pd.DataFrame, extra: bool, kind: str, params: dict) -> np.ndarray:
    err = np.empty(len(df))
    y = np.log10(df[TARGET].to_numpy())
    groups = df[GROUP_COL].to_numpy()
    for g in np.unique(groups):
        tr, te = groups != g, groups == g
        xtr = design(df[tr], extra)
        xte = design(df[te], extra, list(xtr.columns))
        if kind == "freundlich":
            model = LinearRegression().fit(xtr, y[tr])
        else:
            model = LGBMRegressor(**params, monotone_constraints=[1] + [0] * (xtr.shape[1] - 1), random_state=RANDOM_SEED,
                                  n_jobs=2, verbose=-1).fit(xtr.to_numpy(), y[tr])
            xte = xte.to_numpy()
        err[te] = model.predict(xte) - y[te]
    return err


def between_study_sd(df: pd.DataFrame, extra: bool) -> float:
    x, y = design(df, extra), np.log10(df[TARGET].to_numpy())
    resid = pd.Series(y - LinearRegression().fit(x, y).predict(x))
    return float(resid.groupby(df[GROUP_COL].to_numpy()).mean().std(ddof=1))


def coefficient(df: pd.DataFrame) -> tuple[float, float, float]:
    def coef(d: pd.DataFrame) -> float:
        x = design(d, True)
        return float(LinearRegression().fit(x, np.log10(d[TARGET].to_numpy())).coef_[list(x.columns).index("logX")])

    rng = np.random.default_rng(RANDOM_SEED)
    parts = {s: g for s, g in df.groupby(GROUP_COL)}
    names = list(parts)
    boot = [coef(pd.concat([parts[s] for s in rng.choice(names, len(names))], ignore_index=True)) for _ in range(N_BOOT)]
    return coef(df), float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5))


def main() -> None:
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    data = load_data("all")
    meta = pd.read_csv(DATA_DIR / "size_ssa_metadata.csv")
    params = export_tuned_params()["LightGBM"]
    rows = []
    for label, (column, keep) in CASES.items():
        m = meta if keep is None else meta[keep(meta)]
        m = m[m[column].notna()]
        df = data.merge(m[[CURVE_COL, column]].rename(columns={column: "x"}), on=CURVE_COL, how="inner", validate="m:1")
        y = np.log10(df[TARGET].to_numpy())
        b, lo, hi = coefficient(df)
        row = {"variable": label, "n_studies": df[GROUP_COL].nunique(), "n_series": df[CURVE_COL].nunique(), "n_points": len(df),
               "range_of_variable": f"{df['x'].min():.3g}-{df['x'].max():.3g}", "freundlich_coef_log10": b, "coef_ci95_low": lo,
               "coef_ci95_high": hi, "interval_excludes_zero": bool(lo > 0 or hi < 0)}
        for kind in ["freundlich", "lgbm_monotone"]:
            for extra in [False, True]:
                e = loso(df, extra, kind, params)
                tag = f"{kind}_{'plus' if extra else 'core'}"
                row[f"{tag}_loso_fold_error"] = float(10 ** np.median(np.abs(e)))
                row[f"{tag}_loso_r2_log10"] = float(1 - np.sum(e ** 2) / np.sum((y - y.mean()) ** 2))
        row["between_study_sd_core"], row["between_study_sd_plus"] = between_study_sd(df, False), between_study_sd(df, True)
        rows.append(row)
        print(f"{label}: done", flush=True)
    out = pd.DataFrame(rows)
    out.to_csv(REPORTS_DIR / "size_ssa_sensitivity.csv", index=False)
    pd.set_option("display.width", 250)
    print(out.set_index("variable").T.to_string(float_format=lambda v: f"{v:.3g}"))


if __name__ == "__main__":
    main()
