"""Additional checks made after the internal review of 9 October 2026 (rerun 17 Oct 2026 on the 961-point dataset; nothing else is refitted).

A - Field-to-laboratory ratio for alternative sets of field records (isotherm-form model): without the study whose
    water Cu fraction is not stated, without the estuarine records, without the PET-dominated study.
B - Coverage of the study-level 90% interval of the isotherm-form model for observations below 0.1 mg/L, evaluated
    in the same outer leave-one-study-out loop as Table S8, and the leave-one-study-out error in that range.
C - Variability of the random cross-validation between its three repeats (median fold error and R2 per repeat).

Outputs (outputs/reports): field_ratio_sensitivity.csv, interval_coverage_by_ce.csv, random_cv_by_repeat.csv
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from config import GROUP_COL, REPORTS_DIR, TARGET  # noqa: E402
from src.analysis.faf_scaling import study_level, summarise_faf  # noqa: E402
from src.analysis.final_model_validation import balanced_quantile, model_err  # noqa: E402
from src.analysis.grouped_validation import export_tuned_params, load_data  # noqa: E402
from src.analysis.transfer_analysis import fit_predict, iter_splits, log10  # noqa: E402

MODEL = "freundlich_pooled"
FIELD_SETS = {  # field records as of 17 Oct 2026 (field-study search of 16 Oct 2026 added 7 records, 2 studies)
    "main: six studies with dissolved Cu": lambda d: d[d["study"] != "Purwiyanto_2020"],
    "without Ta and Babel (2020a, b): water Cu fraction not stated":
        lambda d: d[~d["study"].isin(["Purwiyanto_2020", "Ta_Babel_2020", "Ta_Babel_2020b"])],
    "without estuarine records (Ta and Babel, 2020a; Samut Prakan zone of Ta and Babel, 2023a, b)":
        lambda d: d[~d["study"].isin(["Purwiyanto_2020", "Ta_Babel_2020"]) & ~d["reference"].str.contains("Samut Prakan")],
    "without Zhang et al. (2025a): PET-dominated": lambda d: d[~d["study"].isin(["Purwiyanto_2020", "Zhang_2025a"])],
    "records available at first submission (four studies)":
        lambda d: d[~d["study"].isin(["Purwiyanto_2020", "Ta_Babel_2020b", "Ling_2026"]) & ~d["reference"].str.contains("2023b")],
    "all seven studies": lambda d: d,
}
CE_BANDS = [("all observations", 0.0, np.inf), ("Ce < 0.1 mg/L", 0.0, 0.1), ("Ce >= 0.1 mg/L", 0.1, np.inf)]
MODELS_CV = ["freundlich_pooled", "xgb_blend", "xgb_monotone", "lgbm_monotone", "rf_raw"]


def field_sensitivity() -> pd.DataFrame:
    rec = pd.read_csv(REPORTS_DIR / "faf_scaling_by_study.csv")
    rec = rec[rec["lab_model"] == MODEL]
    rows = []
    for label, keep in FIELD_SETS.items():
        part = keep(rec)
        s = summarise_faf(study_level(part).to_numpy())
        rows.append({"field_records": label, "n_records": len(part), "n_studies": s["n_field_studies"],
                     "geometric_mean": s["geometric_mean"], "gm_ci95_low": s["gm_ci95_low"], "gm_ci95_high": s["gm_ci95_high"],
                     "between_study_sd_log10": s["between_study_sd_log10"]})
    return pd.DataFrame(rows)


def coverage_by_ce(df: pd.DataFrame, level: float = 0.90) -> pd.DataFrame:
    studies = sorted(df[GROUP_COL].unique())
    parts = []
    for s in studies:
        test, rest = df[df[GROUP_COL] == s], df[df[GROUP_COL] != s]
        cal, grp = [], []
        for t in [x for x in studies if x != s]:
            e = np.abs(model_err(rest[rest[GROUP_COL] != t], rest[rest[GROUP_COL] == t], MODEL))
            cal.append(e)
            grp.append(np.full(len(e), t))
        q = balanced_quantile(np.concatenate(cal), np.concatenate(grp), level)
        e_test = model_err(rest, test, MODEL)
        parts.append(pd.DataFrame({GROUP_COL: s, "Ce": test["Ce"].to_numpy(), "err_log10": e_test, "q_log10": q}))
    pts = pd.concat(parts, ignore_index=True)
    rows = []
    for label, lo, hi in CE_BANDS:
        g = pts[(pts["Ce"] >= lo) & (pts["Ce"] < hi)]
        a = np.abs(g["err_log10"])
        rows.append({"range": label, "n_points": len(g), "n_studies": g[GROUP_COL].nunique(), "nominal": level,
                     "coverage_points": float(np.mean(a <= g["q_log10"])),
                     "coverage_mean_over_studies": float((a <= g["q_log10"]).groupby(g[GROUP_COL]).mean().mean()),
                     "median_fold_error": float(10 ** a.median()), "p90_fold_error": float(10 ** a.quantile(0.9)),
                     "bias_as_factor": float(10 ** g["err_log10"].mean()),
                     "interval_factor_mean": float(10 ** g["q_log10"].mean())})
    return pd.DataFrame(rows)


def random_cv_by_repeat(df: pd.DataFrame) -> pd.DataFrame:
    rf_params = export_tuned_params()["RandomForest"]
    y = np.asarray(log10(df[TARGET]))
    pred = {}
    for rep, tr, te in iter_splits("random", df):
        for m in MODELS_CV:
            pred.setdefault((m, rep), np.full(len(df), np.nan))[te] = np.log10(fit_predict(m, df.iloc[tr], df.iloc[te], rf_params))
        print(f"random CV repeat {rep}: fold done", flush=True)
    rows = []
    for (m, rep), p in pred.items():
        rows.append({"model": m, "repeat": rep, "median_fold_error": 10 ** np.median(np.abs(p - y)),
                     "r2_log10": 1 - ((p - y) ** 2).sum() / ((y - y.mean()) ** 2).sum()})
    by_rep = pd.DataFrame(rows)
    agg = by_rep.groupby("model").agg(mfe_mean=("median_fold_error", "mean"), mfe_sd=("median_fold_error", "std"),
                                      mfe_min=("median_fold_error", "min"), mfe_max=("median_fold_error", "max"),
                                      r2_mean=("r2_log10", "mean"), r2_sd=("r2_log10", "std")).reset_index()
    return by_rep.merge(agg, on="model")


def shap_xgb(df: pd.DataFrame) -> pd.DataFrame:
    """SHAP shares of monotone XGB, the tree component of the isotherm-anchored model (same procedure as for LightGBM)."""
    import shap
    from xgboost import XGBRegressor

    from config import RANDOM_SEED
    from src.analysis.grouped_validation import make_preprocessor
    from src.analysis.transfer_analysis import FEATURES

    def shares(part: pd.DataFrame) -> pd.Series:
        pre = make_preprocessor(FEATURES, log_ce=True)
        x = pre.fit_transform(part[FEATURES])
        onehot = pre.named_transformers_["cat"].named_steps["onehot"]
        owner = ["Ce"] + [c for c, levels in zip([c for c in FEATURES if c != "Ce"], onehot.categories_) for _ in levels]
        reg = XGBRegressor(**export_tuned_params()["XGBoost"], objective="reg:squarederror", tree_method="hist", random_state=RANDOM_SEED,
                           n_jobs=-1, monotone_constraints=tuple([1] + [0] * (x.shape[1] - 1))).fit(x, log10(part[TARGET]))
        values = shap.TreeExplainer(reg).shap_values(x)
        imp = pd.Series({f: np.abs(values[:, [o == f for o in owner]].sum(axis=1)).mean() for f in FEATURES})
        return imp / imp.sum() * 100

    full = shares(df)
    loo = pd.DataFrame([shares(df[df[GROUP_COL] != s]) for s in df[GROUP_COL].unique()])
    order = full.sort_values(ascending=False).index
    return pd.DataFrame({"predictor": order, "share_pct": full[order].to_numpy(), "share_pct_min_leave_one_study_out": loo[order].min().to_numpy(),
                         "share_pct_max_leave_one_study_out": loo[order].max().to_numpy(),
                         "rank_changes_in_leave_one_study_out": [int((loo.rank(axis=1, ascending=False)[f] != i + 1).sum())
                                                                 for i, f in enumerate(order)]})


def main() -> None:
    pd.set_option("display.width", 220)
    if "--shap-only" in sys.argv:
        d = shap_xgb(load_data("all"))
        d.to_csv(REPORTS_DIR / "interpretation_shap_importance_xgb.csv", index=False)
        print(d.to_string(index=False, float_format=lambda v: f"{v:.3g}"))
        return
    fmt = lambda v: f"{v:.3g}"  # noqa: E731
    a = field_sensitivity()
    a.to_csv(REPORTS_DIR / "field_ratio_sensitivity.csv", index=False)
    print(a.to_string(index=False, float_format=fmt))
    if "--field-only" in sys.argv:
        return
    df = load_data("all")
    b = coverage_by_ce(df)
    b.to_csv(REPORTS_DIR / "interval_coverage_by_ce.csv", index=False)
    print(b.to_string(index=False, float_format=fmt))
    if "--skip-cv" not in sys.argv:
        c = random_cv_by_repeat(df)
        c.to_csv(REPORTS_DIR / "random_cv_by_repeat.csv", index=False)
        print(c.drop_duplicates("model").drop(columns=["repeat", "median_fold_error", "r2_log10"]).to_string(index=False, float_format=fmt))
        print(c.pivot(index="model", columns="repeat", values="median_fold_error").to_string(float_format=fmt))


if __name__ == "__main__":
    main()
