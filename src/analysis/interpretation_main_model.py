"""Interpretation of the ML models (SHAP of monotone LightGBM; response curves of the isotherm-anchored XGBoost, ``xgb_blend``).

1. SHAP values of the monotone LightGBM body (target log10 qe), summed over the one-hot
   columns of each predictor. Because the target is on the log scale, a SHAP value s is a
   multiplicative factor 10**s on qe. Importance is the mean |SHAP| per predictor.
2. Stability of that importance when one study at a time is left out of the fit.
3. Response curves of the full model (ML body + Freundlich tail) against Ce for every polymer
   with at least three source studies, for virgin and aged particles.

Outputs: interpretation_shap_importance.csv, interpretation_shap_ce_dependence.csv,
interpretation_response_curves.csv (outputs/reports); interpretation_main_model.png (outputs/figures).
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
from lightgbm import LGBMRegressor

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from config import FIGURES_DIR, GROUP_COL, RANDOM_SEED, REPORTS_DIR, TARGET  # noqa: E402
from src.analysis.grouped_validation import export_tuned_params, load_data, make_preprocessor  # noqa: E402
from src.analysis.transfer_analysis import BLEND_HIGH, BLEND_LOW, FEATURES, fit_predict, log10  # noqa: E402

LABELS = {"Ce": "Equilibrium concentration (Ce)", "ReT": "Polymer type", "AgS": "Aging status", "AdC": "Adsorption system"}
MIN_STUDIES = 3
CE_GRID = np.logspace(-3.5, 2, 111)
PARAMS = export_tuned_params()


def shap_by_predictor(df: pd.DataFrame) -> pd.DataFrame:
    """Per-sample SHAP value of each original predictor (log10 qe units)."""
    pre = make_preprocessor(FEATURES, log_ce=True)  # first column is log10(Ce)
    x = pre.fit_transform(df[FEATURES])
    onehot = pre.named_transformers_["cat"].named_steps["onehot"]
    cat_cols = [c for c in FEATURES if c != "Ce"]
    owner = ["Ce"] + [c for c, levels in zip(cat_cols, onehot.categories_) for _ in levels]
    assert len(owner) == x.shape[1]
    reg = LGBMRegressor(**PARAMS["LightGBM"], monotone_constraints=[1] + [0] * (x.shape[1] - 1),
                        random_state=RANDOM_SEED, n_jobs=2, verbose=-1).fit(x, log10(df[TARGET]))
    values = shap.TreeExplainer(reg).shap_values(x)
    return pd.DataFrame({f: values[:, [o == f for o in owner]].sum(axis=1) for f in FEATURES}, index=df.index)


def importance(sv: pd.DataFrame) -> pd.Series:
    return sv.abs().mean().sort_values(ascending=False)


def importance_table(df: pd.DataFrame, sv: pd.DataFrame) -> pd.DataFrame:
    full = importance(sv)
    share = full / full.sum() * 100
    loo = pd.DataFrame([importance(shap_by_predictor(df[df[GROUP_COL] != s])) for s in df[GROUP_COL].unique()])
    loo_share = loo.div(loo.sum(axis=1), axis=0) * 100
    return pd.DataFrame({"predictor": full.index, "label": [LABELS[f] for f in full.index],
                         "mean_abs_shap_log10": full.to_numpy(), "typical_factor_on_qe": 10 ** full.to_numpy(),
                         "share_pct": share.to_numpy(),
                         "share_pct_min_leave_one_study_out": loo_share[full.index].min().to_numpy(),
                         "share_pct_max_leave_one_study_out": loo_share[full.index].max().to_numpy(),
                         "rank_changes_in_leave_one_study_out": [int((loo.rank(axis=1, ascending=False)[f]
                                                                     != i + 1).sum()) for i, f in enumerate(full.index)]})


def response_curves(df: pd.DataFrame) -> pd.DataFrame:
    counts = df.groupby("ReT")[GROUP_COL].nunique()
    polymers = sorted(counts[counts >= MIN_STUDIES].index)
    frame = pd.DataFrame([{"ReT": p, "AgS": a, "AdC": "Single-ion", "Ce": c}
                          for p in polymers for a in ["Virgin", "Aged"] for c in CE_GRID])
    frame["qe_mg_g"] = fit_predict("xgb_blend", df, frame, PARAMS["RandomForest"])
    frame["n_studies"] = frame["ReT"].map(counts)
    for p in polymers:  # laboratory Ce range of each polymer, to mark where curves are supported by data
        ce = df.loc[df["ReT"] == p, "Ce"]
        frame.loc[frame["ReT"] == p, ["Ce_data_min", "Ce_data_max"]] = [ce.min(), ce.max()]
    return frame.rename(columns={"Ce": "Ce_mg_L"})


def save_figure(df: pd.DataFrame, table: pd.DataFrame, sv: pd.DataFrame, curves: pd.DataFrame, zone: tuple[float, float]) -> Path:
    plt.rcParams.update({"font.family": "Times New Roman", "mathtext.fontset": "custom", "mathtext.rm": "Times New Roman", "mathtext.it": "Times New Roman:italic", "mathtext.bf": "Times New Roman:bold", "mathtext.default": "regular", "axes.spines.top": False, "axes.spines.right": False})
    fig, axes = plt.subplots(1, 3, figsize=(15.5, 4.6))
    part = table.iloc[::-1]
    axes[0].barh(part["label"].str.replace("(Ce)", "($C_e$)", regex=False), part["share_pct"], color="#2f6f9f",
                 xerr=[part["share_pct"] - part["share_pct_min_leave_one_study_out"],
                       part["share_pct_max_leave_one_study_out"] - part["share_pct"]], capsize=3)
    axes[0].set_xlabel("Share of mean |SHAP| (%); bars: range when one study is left out")
    axes[0].set_title("a", loc="left", fontsize=12, fontweight="bold")
    axes[1].scatter(df["Ce"], 10 ** sv["Ce"], s=6, color="#2f6f9f", alpha=0.5)
    axes[1].axhline(1.0, color="0.4", linestyle="--", linewidth=0.8)
    axes[1].set_xscale("log"); axes[1].set_yscale("log")
    axes[1].set_xlabel("$C_e$ (mg L$^{-1}$)"); axes[1].set_ylabel("SHAP effect of $C_e$ (factor on $q_e$)")
    axes[1].set_title("b", loc="left", fontsize=12, fontweight="bold")
    aged = curves[curves["AgS"] == "Aged"]
    for p, g in aged.groupby("ReT"):
        line, = axes[2].plot(g["Ce_mg_L"], g["qe_mg_g"], label=f"{p} (n = {int(g['n_studies'].iloc[0])})")
        v = curves[(curves["AgS"] == "Virgin") & (curves["ReT"] == p)]
        axes[2].plot(v["Ce_mg_L"], v["qe_mg_g"], color=line.get_color(), linestyle=":", linewidth=1)
    axes[2].axvspan(*zone, color="0.5", alpha=0.10)
    axes[2].axvspan(0.001, 0.01, color="tab:blue", alpha=0.10)
    axes[2].set_xscale("log"); axes[2].set_yscale("log")
    axes[2].set_xlabel("$C_e$ (mg L$^{-1}$)"); axes[2].set_ylabel("Predicted $q_e$ (mg g$^{-1}$); solid aged, dotted virgin")
    axes[2].set_title("c", loc="left", fontsize=12, fontweight="bold")
    axes[2].legend(frameon=False, fontsize=8, title="Polymer (studies)")
    fig.tight_layout()
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    path = FIGURES_DIR / "interpretation_main_model.png"
    fig.savefig(path, dpi=600)
    fig.savefig(path.with_suffix(".tiff"), dpi=600, pil_kwargs={"compression": "tiff_lzw"})
    plt.close(fig)
    return path


def main() -> None:
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    df = load_data("all")
    zone = tuple(float(v) for v in np.percentile(df["Ce"], [BLEND_LOW, BLEND_HIGH]))
    sv = shap_by_predictor(df)
    table = importance_table(df, sv)
    table.to_csv(REPORTS_DIR / "interpretation_shap_importance.csv", index=False)
    dep = pd.DataFrame({GROUP_COL: df[GROUP_COL], "ReT": df["ReT"], "Ce_mg_L": df["Ce"],
                        "shap_Ce_log10": sv["Ce"], "factor_on_qe": 10 ** sv["Ce"]})
    dep.to_csv(REPORTS_DIR / "interpretation_shap_ce_dependence.csv", index=False)
    curves = response_curves(df)
    curves.to_csv(REPORTS_DIR / "interpretation_response_curves.csv", index=False)

    pd.set_option("display.width", 220)
    fmt = lambda v: f"{v:.3g}"  # noqa: E731
    print(f"Transition zone of the isotherm-anchored model: {zone[0]:.3g}-{zone[1]:.3g} mg/L")
    print(table.drop(columns="predictor").to_string(index=False, float_format=fmt))
    for name, sub in [("polymer", "ReT"), ("aging", "AgS"), ("system", "AdC")]:
        eff = (10 ** sv[sub].groupby(df[sub]).mean()).sort_values()
        print(f"\nMean SHAP effect by {name} (factor on qe): " + ", ".join(f"{k} {v:.2f}" for k, v in eff.items()))
    at = curves[(curves["AgS"] == "Aged") & (np.isclose(curves["Ce_mg_L"], CE_GRID[np.argmin(np.abs(CE_GRID - 1.0))]))]
    print("\nPredicted qe at 1 mg/L, aged (mg/g): " + ", ".join(f"{r.ReT} {r.qe_mg_g:.3f}" for r in at.itertuples()))
    ratio = curves.pivot_table(index=["ReT", "Ce_mg_L"], columns="AgS", values="qe_mg_g")
    print("Aged / virgin ratio (median over Ce): " + ", ".join(
        f"{p} {g['Aged'].div(g['Virgin']).median():.2f}" for p, g in ratio.groupby(level=0)))
    # Fig. 3 is drawn from the CSV files above by scripts/make_fig3_interpretation.py (layout of the revision, Oct 2026);
    # save_figure() keeps the earlier one-row layout and is no longer called.
    import runpy
    runpy.run_path(str(PROJECT_ROOT / "scripts" / "make_fig3_interpretation.py"), run_name="__main__")


if __name__ == "__main__":
    main()
