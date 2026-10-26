"""Draw the main figures of the revised manuscript from the analysis outputs.

Fig. 2  Model validation: accuracy inside the laboratory domain and for a new laboratory,
        observed vs predicted with study-level intervals, low-Ce extrapolation, response to Ce.
Fig. 3  Interpretation of the main model (copied from interpretation_main_model.py).
Fig. 5  Field comparison and scale-up: field vs laboratory prediction, field-to-laboratory
        ratio by study and model, global flux scenarios, river application.

Run the analysis scripts first; this script only reads outputs/reports and refits the models
once for the response curves. Output: Manuscript/Submission_Env pollution/Revision/Figures_revised/
(PNG and LZW-compressed TIFF at 600 dpi).
"""

from __future__ import annotations

import shutil
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.analysis.grouped_validation import export_tuned_params, load_data  # noqa: E402
from src.analysis.transfer_analysis import BLEND_HIGH, BLEND_LOW, fit_predict  # noqa: E402

REP = ROOT / "outputs" / "reports"
OUT = ROOT / "Manuscript" / "Submission_Env pollution" / "Revision" / "02_Figures"
DPI = 600
plt.rcParams.update({"font.family": "Times New Roman", "mathtext.fontset": "custom", "mathtext.rm": "Times New Roman", "mathtext.it": "Times New Roman:italic", "mathtext.bf": "Times New Roman:bold", "font.size": 8, "axes.labelsize": 8, "axes.titlesize": 9,
                     "legend.fontsize": 7, "xtick.labelsize": 7.5, "ytick.labelsize": 7.5,
                     "axes.spines.top": False, "axes.spines.right": False})

MAIN = "freundlich_pooled"   # model selected by study-grouped validation
MODELS = {"freundlich_pooled": ("Isotherm-form linear model", "#1b4f72"),
          "xgb_blend": ("Isotherm-anchored XGBoost", "#5dade2"),
          "lgbm_monotone": ("Monotone LightGBM", "#f39c12"),
          "rf_raw": ("Random Forest, unconstrained", "#c0392b")}
STUDY_STYLE = {"Ta_Babel_2020": ("Ta & Babel (2020a)", "o"), "Ta_Babel_2020b": ("Ta & Babel (2020b)", "P"),
               "Ta_Babel_2023": ("Ta & Babel (2023a, b)", "s"), "Ling_2026": ("Ling et al. (2026)", "v"),
               "Zhang_2025a": ("Zhang et al. (2025a)", "^"), "Zhang_2025b": ("Zhang et al. (2025b)", "D"),
               "Purwiyanto_2020": ("Purwiyanto et al. (2020)", "X")}
SENSITIVITY_ONLY = {"Purwiyanto_2020"}
RIVER_BAND = (0.001, 0.01)


def panel_label(ax, letter: str) -> None:
    ax.text(-0.14, 1.04, letter, transform=ax.transAxes, fontsize=11, fontweight="bold", va="bottom")


def save(fig, name: str) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT / f"{name}.png", dpi=DPI)
    fig.savefig(OUT / f"{name}.tiff", dpi=DPI, pil_kwargs={"compression": "tiff_lzw"})
    plt.close(fig)
    print(f"saved {name}")


# ------------------------------------------------------------------ Fig. 2
def validation_numbers() -> pd.DataFrame:
    """Median fold error by design and low-concentration bias for the four models shown (fixed hyperparameters)."""
    t1 = pd.read_csv(REP / "table1_model_performance.csv").pivot(index="model", columns="tier", values="median_fold_error")
    low = pd.read_csv(REP / "low_ce_holdout_final.csv")
    low = low[np.isclose(low["threshold_mg_L"], 0.1)].pivot(index="model", columns="design", values="bias_as_factor")
    return pd.DataFrame({m: {"random": t1.loc[m, "random"], "new_lab": t1.loc[m, "leave_one_study_out"],
                             "bias_same": low.loc[m, "same_lab"], "bias_new": low.loc[m, "new_lab"]} for m in MODELS}).T


def response_curves(df: pd.DataFrame) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    ce = np.logspace(-3.6, 2.2, 160)
    polymers = ["PE", "PP", "PS", "PET"]
    frame = pd.DataFrame([{"ReT": p, "AgS": "Aged", "AdC": "Mixed", "Ce": c} for p in polymers for c in ce])
    rf_params = export_tuned_params()["RandomForest"]
    return ce, {m: 10 ** np.log10(fit_predict(m, df, frame, rf_params)).reshape(len(polymers), len(ce)).mean(axis=0)
                for m in MODELS}


def figure_2(df: pd.DataFrame) -> None:
    nums = validation_numbers()
    intervals = pd.read_csv(REP / "final_prediction_intervals.csv")
    main_int = intervals[(intervals["model"] == MAIN) & np.isclose(intervals["nominal"], 0.9)].set_index("case")
    nested = pd.read_csv(REP / "final_nested_loso.csv").set_index("model")
    pred = pd.read_csv(REP / "final_nested_loso_predictions.csv")
    fig, axes = plt.subplots(2, 2, figsize=(7.4, 6.6))

    ax = axes[0, 0]
    x, width = np.arange(2), 0.2
    for i, (m, (label, color)) in enumerate(MODELS.items()):
        vals = [nums.loc[m, "random"], nums.loc[m, "new_lab"]]
        bars = ax.bar(x + (i - 1.5) * width, vals, width, color=color, label=label)
        ax.bar_label(bars, fmt="%.2f", fontsize=6, padding=1)
    ax.set_xticks(x); ax.set_xticklabels(["Random split\n(same laboratories)", "Leave-one-study-out\n(new laboratory)"])
    ax.set_ylabel("Median fold error (×)"); ax.set_ylim(1, 3.6)
    ax.legend(frameon=False, loc="upper left")
    panel_label(ax, "a")

    ax = axes[0, 1]
    obs, est = 10 ** pred["log_obs"], 10 ** pred[MAIN]
    lim = np.array([min(obs.min(), est.min()) * 0.6, max(obs.max(), est.max()) * 1.6])
    f_new, f_anchor = main_int.loc["new_laboratory", "interval_factor_final"], main_int.loc["one_anchor_per_isotherm", "interval_factor_final"]
    ax.fill_between(lim, lim / f_new, lim * f_new, color="#1b4f72", alpha=0.10, label=f"90% interval, new laboratory (×/÷{f_new:.0f})")
    ax.fill_between(lim, lim / f_anchor, lim * f_anchor, color="#1b4f72", alpha=0.22,
                    label=f"90% interval with one local point (×/÷{f_anchor:.1f})")
    ax.plot(lim, lim, color="black", linewidth=0.8)
    ax.scatter(obs, est, s=5, color="#1b4f72", alpha=0.55, linewidths=0)
    ax.set_xscale("log"); ax.set_yscale("log"); ax.set_xlim(lim); ax.set_ylim(lim)
    ax.set_xlabel("Observed $q_e$ (mg g$^{-1}$)"); ax.set_ylabel("Predicted $q_e$, study left out (mg g$^{-1}$)")
    ax.text(0.97, 0.05, f"median fold error {nested.loc[MAIN, 'median_fold_error']:.2f}×\n"
            f"{len(pred)} points, {pred['source_id'].nunique()} studies", transform=ax.transAxes, ha="right", fontsize=6.5)
    ax.legend(frameon=False, loc="upper left")
    panel_label(ax, "b")

    ax = axes[1, 0]
    for i, (m, (label, color)) in enumerate(MODELS.items()):
        vals = [nums.loc[m, "bias_same"], nums.loc[m, "bias_new"]]
        bars = ax.bar(x + (i - 1.5) * width, vals, width, color=color)
        ax.bar_label(bars, fmt="%.1f", fontsize=6, padding=1)
    ax.axhline(1.0, color="black", linewidth=0.8, linestyle="--")
    ax.set_yscale("log"); ax.set_ylim(0.7, 30)
    ax.set_xticks(x); ax.set_xticklabels(["Same laboratories", "New laboratory"])
    ax.set_ylabel("Bias at $C_e$ ≤ 0.1 mg L$^{-1}$ (predicted / observed)\nmodels trained on $C_e$ > 0.1 mg L$^{-1}$ only")
    panel_label(ax, "c")

    ax = axes[1, 1]
    ce, curves = response_curves(df)
    ax.scatter(df["Ce"], df["qe"], s=3, color="0.78", linewidths=0, label="Laboratory data", zorder=1)
    for m, (label, color) in MODELS.items():
        ax.plot(ce, curves[m], color=color, linewidth=1.6 if m == MAIN else 1.0, zorder=3 if m == MAIN else 2)
    ax.axvspan(*RIVER_BAND, color="#2e86c1", alpha=0.12, label="Typical river $C_e$")
    ca, cb = np.percentile(df["Ce"], [BLEND_LOW, BLEND_HIGH])
    ax.axvspan(ca, cb, color="0.5", alpha=0.10, label="Transition zone ($C_a$–$C_b$)")
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlabel("$C_e$ (mg L$^{-1}$)"); ax.set_ylabel("$q_e$ (mg g$^{-1}$)")
    ax.legend(frameon=False, loc="lower right")
    panel_label(ax, "d")
    fig.tight_layout()
    save(fig, "Fig_2_model_validation")


# ------------------------------------------------------------------ Fig. 5
def figure_5() -> None:
    by = pd.read_csv(REP / "faf_scaling_by_study.csv")
    summ = pd.read_csv(REP / "faf_scaling_summary.csv")
    flux = pd.read_csv(REP / "global_flux_constrained.csv")
    rivers = pd.read_csv(REP / "river_application_main.csv")
    fig, axes = plt.subplots(2, 2, figsize=(7.4, 6.6))

    ax = axes[0, 0]
    main = by[by["lab_model"] == MAIN]
    lim = np.array([1e-4, 0.3])
    ax.plot(lim, lim, color="black", linewidth=0.8)
    for factor in (10, 100):
        ax.plot(lim, lim / factor, color="0.6", linewidth=0.6, linestyle="--")
        ax.text(lim[1] * 0.8, lim[1] / factor * 0.8, f"1:{factor}", fontsize=6, color="0.4", ha="right", va="top")
    for study, (label, marker) in STUDY_STYLE.items():
        g = main[main["study"] == study]
        hollow = study in SENSITIVITY_ONLY
        ax.scatter(g["qe_lab_mg_g"], g["qe_field_mg_g"], marker=marker, s=34, facecolor="none" if hollow else "#1b4f72",
                   edgecolor="#1b4f72", linewidths=0.9, label=label + (" (total Cu in water)" if hollow else ""), zorder=3)
    ax.set_xscale("log"); ax.set_yscale("log"); ax.set_xlim(lim); ax.set_ylim(2e-5, 20)
    ax.set_xlabel("Laboratory-model $q_e$ at field $C_e$ (mg g$^{-1}$)"); ax.set_ylabel("Cu on field microplastics (mg g$^{-1}$)")
    ax.legend(frameon=False, loc="upper left", fontsize=6.3, handletextpad=0.2, borderaxespad=0.2)
    panel_label(ax, "a")

    ax = axes[0, 1]
    order = list(MODELS)
    for i, m in enumerate(order):
        part = by[by["lab_model"] == m]
        for study, (_, marker) in STUDY_STYLE.items():
            g = part[part["study"] == study]
            ax.scatter(g["faf"], np.full(len(g), i), marker=marker, s=24, edgecolor=MODELS[m][1], linewidths=0.8,
                       facecolor="none" if study in SENSITIVITY_ONLY else MODELS[m][1], zorder=3)
        row = summ[(summ["lab_model"] == m) & summ["primary"]].iloc[0]
        ax.plot([row["gm_ci95_low"], row["gm_ci95_high"]], [i + 0.3] * 2, color="black", linewidth=1.2)
        ax.scatter([row["geometric_mean"]], [i + 0.3], marker="|", s=70, color="black", zorder=4)
        ax.text(row["gm_ci95_high"] * 1.8, i + 0.3, f"{row['geometric_mean']:.2g}", fontsize=6.3, va="center")
    ax.axvline(1.0, color="0.4", linestyle="--", linewidth=0.8)
    ax.set_xscale("log"); ax.set_xlim(3e-4, 40)
    ax.set_yticks(range(len(order))); ax.set_yticklabels([MODELS[m][0].replace(", ", ",\n") for m in order])
    ax.set_ylim(len(order) - 0.3, -0.6)
    ax.set_xlabel("Field-to-laboratory ratio\n(black: geometric mean, 95% CI)")
    panel_label(ax, "b")

    ax = axes[1, 0]
    world = flux[flux["Ce_case"] == "world_average"].set_index("lab_model")
    scen = [("Cu_kg_yr_low_MP_flux", "Microplastics\n6.1 kt yr$^{-1}$"), ("Cu_kg_yr_central_MP_flux", "Microplastics\n53 kt yr$^{-1}$"),
            ("Cu_kg_yr_high_MP_flux", "All plastics\n500 kt yr$^{-1}$\n(upper bound)")]
    xs = np.arange(len(scen))
    bars = ax.bar(xs, [world.loc[MAIN, c] for c, _ in scen], 0.55, color=MODELS[MAIN][1])
    ax.bar_label(bars, labels=[f"{float(f'{b.get_height():.2g}'):,.0f}" for b in bars], fontsize=6.5, padding=2)   # two significant figures, as in the text
    for m in [k for k in MODELS if k != MAIN]:
        ax.scatter(xs + 0.36, [world.loc[m, c] for c, _ in scen], s=14, color=MODELS[m][1], zorder=3, label=MODELS[m][0])
    dissolved_kg = world.loc[MAIN, "Cu_kg_yr_central_MP_flux"] / world.loc[MAIN, "pct_of_dissolved_central"] * 100
    ax.set_yscale("log"); ax.set_ylim(5, 2e6)
    ax.set_xticks(xs); ax.set_xticklabels([s for _, s in scen])
    ax.set_ylabel("MP-bound Cu exported by rivers (kg yr$^{-1}$)")
    ax.text(0.03, 0.70, f"Dissolved Cu flux: {dissolved_kg / 1e3:,.0f} t yr$^{{-1}}$\nMP-bound share ≤ "
            f"{world.loc[MAIN, 'pct_of_dissolved_high']:.4f}%", transform=ax.transAxes, va="top", fontsize=6.5)
    ax.legend(frameon=False, loc="upper left", title="Other laboratory models", title_fontsize=6.5, fontsize=6.3)
    panel_label(ax, "c")

    ax = axes[1, 1]
    names = [r.split(" (")[0].replace("Pearl River system", "Pearl") for r in rivers["river"]]
    cases = [("12.5ug", "12.5 µg per particle", "#1b4f72"), ("170ug", "170 µg per particle", "#5dade2"), ("230ug", "230 µg per particle", "#aed6f1")]
    xr = np.arange(len(rivers))
    for i, (key, label, color) in enumerate(cases):
        mid = rivers[f"Cu_kg_yr_{key}"]
        err = [mid - rivers[f"Cu_kg_yr_{key}_low"], rivers[f"Cu_kg_yr_{key}_high"] - mid]
        ax.bar(xr + (i - 1) * 0.27, mid, 0.27, color=color, label=label, yerr=err, error_kw={"linewidth": 0.6, "capsize": 1.5})
    ax.set_yscale("log"); ax.set_xticks(xr); ax.set_xticklabels(names)
    ax.set_ylabel("MP-bound Cu (kg yr$^{-1}$)")
    ax.legend(frameon=False, loc="upper right", title="Assumed particle mass", title_fontsize=6.5)
    panel_label(ax, "d")
    fig.tight_layout()
    save(fig, "Fig_5_field_and_scaleup")


def main() -> None:
    df = load_data("all")
    figure_2(df)
    # Fig. 5 is drawn by scripts/make_fig5_field_scaleup.py (layout of the revision, Oct 2026); figure_5() is kept for reference.
    import runpy
    runpy.run_path(str(ROOT / "scripts" / "make_fig5_field_scaleup.py"), run_name="__main__")
    OUT.mkdir(parents=True, exist_ok=True)
    shutil.copy(ROOT / "outputs" / "figures" / "interpretation_main_model.png", OUT / "Fig_3_interpretation.png")
    shutil.copy(ROOT / "outputs" / "figures" / "interpretation_main_model.tiff", OUT / "Fig_3_interpretation.tiff")
    print("copied Fig_3_interpretation")


if __name__ == "__main__":
    main()
