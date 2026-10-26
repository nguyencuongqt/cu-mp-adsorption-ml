"""Fig. 5 (field comparison and scale-up), redrawn for the revision from the saved analysis outputs.

(a) Cu on field MPs versus the laboratory-model qe at the field Ce; (b) field-to-laboratory ratio of each field
record for four laboratory models; (c) global MP-bound Cu flux scenarios; (d) the four rivers with measured inputs.
Layout and data as in make_revised_figures.figure_5(); larger axis text, model names as in the text and Table 1
(no "monotone"), two-line labels in panel b. Reads only outputs/reports; no model is refitted.
Output: Revision/02_Figures/Fig_5_field_and_scaleup.png / .tiff (600 dpi)
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
REP = ROOT / "outputs" / "reports"
OUT = ROOT / "Manuscript" / "Submission_Env pollution" / "Revision" / "02_Figures"
DPI = 600
MAIN = "freundlich_pooled"
MODELS = {"freundlich_pooled": ("Isotherm-form model", "Isotherm-form\nmodel", "#1b4f72"),
          "xgb_blend": ("Isotherm-anchored XGB", "Isotherm-anchored\nXGB", "#5dade2"),
          "lgbm_monotone": ("LightGBM", "LightGBM", "#f39c12"),
          "rf_raw": ("RF, untransformed $q_e$", "RF, untrans-\nformed $q_e$", "#c0392b")}
STUDY_STYLE = {"Ta_Babel_2020": ("Ta & Babel (2020a)", "o"), "Ta_Babel_2020b": ("Ta & Babel (2020b)", "P"),
               "Ta_Babel_2023": ("Ta & Babel (2023a, b)", "s"), "Ling_2026": ("Ling et al. (2026)", "v"),
               "Zhang_2025a": ("Zhang et al. (2025a)", "^"), "Zhang_2025b": ("Zhang et al. (2025b)", "D"),
               "Purwiyanto_2020": ("Purwiyanto et al. (2020)", "X")}
SENSITIVITY_ONLY = {"Purwiyanto_2020"}

AX, TICK, NOTE = 10, 9.5, 8      # axis labels, tick labels, notes (legends, value labels, annotations)
plt.rcParams.update({
    "font.family": "serif", "font.serif": ["Times New Roman", "Liberation Serif", "DejaVu Serif"],
    "mathtext.fontset": "custom", "mathtext.rm": "Times New Roman", "mathtext.it": "Times New Roman:italic",
    "mathtext.bf": "Times New Roman:bold",
    "font.size": NOTE, "axes.labelsize": AX, "xtick.labelsize": TICK, "ytick.labelsize": TICK, "legend.fontsize": NOTE,
    "axes.spines.top": False, "axes.spines.right": False,
})
if not any("Times New Roman" in f.name for f in matplotlib.font_manager.fontManager.ttflist):
    plt.rcParams.update({"mathtext.rm": "Liberation Serif", "mathtext.it": "Liberation Serif:italic",
                         "mathtext.bf": "Liberation Serif:bold"})


def panel_label(ax, letter: str, x: float = -0.2) -> None:
    ax.text(x, 1.04, letter, transform=ax.transAxes, fontsize=12, fontweight="bold", va="bottom")


def main() -> None:
    by = pd.read_csv(REP / "faf_scaling_by_study.csv")
    summ = pd.read_csv(REP / "faf_scaling_summary.csv")
    flux = pd.read_csv(REP / "global_flux_constrained.csv")
    rivers = pd.read_csv(REP / "river_application_main.csv")
    fig, axes = plt.subplots(2, 2, figsize=(7.48, 7.0))

    # (a) field vs laboratory
    ax = axes[0, 0]
    main = by[by["lab_model"] == MAIN]
    lim = np.array([1e-4, 0.3])
    ax.plot(lim, lim, color="black", linewidth=0.8)
    for factor in (10, 100):
        ax.plot(lim, lim / factor, color="0.6", linewidth=0.6, linestyle="--")
        ax.text(lim[1] * 0.75, lim[1] / factor * 0.55, f"1:{factor}", fontsize=NOTE, color="0.4", ha="right", va="top")
    for study, (label, marker) in STUDY_STYLE.items():
        g = main[main["study"] == study]
        hollow = study in SENSITIVITY_ONLY
        ax.scatter(g["qe_lab_mg_g"], g["qe_field_mg_g"], marker=marker, s=30, facecolor="none" if hollow else "#1b4f72",
                   edgecolor="#1b4f72", linewidths=0.9, label=label + ("\n(total Cu in water)" if hollow else ""), zorder=3)
    ax.set_xscale("log"); ax.set_yscale("log"); ax.set_xlim(lim); ax.set_ylim(2e-5, 300)
    ax.set_xlabel("Laboratory-model $q_e$ at field $C_e$ (mg g$^{-1}$)")
    ax.set_ylabel("Cu on field microplastics (mg g$^{-1}$)")
    ax.legend(frameon=False, loc="upper left", handletextpad=0.2, borderaxespad=0.2, labelspacing=0.3)
    panel_label(ax, "a")

    # (b) ratio by field record and laboratory model
    ax = axes[0, 1]
    order = list(MODELS)
    for i, m in enumerate(order):
        col = MODELS[m][2]
        part = by[by["lab_model"] == m]
        for study, (_, marker) in STUDY_STYLE.items():
            g = part[part["study"] == study]
            ax.scatter(g["faf"], np.full(len(g), i), marker=marker, s=22, edgecolor=col, linewidths=0.8,
                       facecolor="none" if study in SENSITIVITY_ONLY else col, zorder=3)
        row = summ[(summ["lab_model"] == m) & summ["primary"]].iloc[0]
        ax.plot([row["gm_ci95_low"], row["gm_ci95_high"]], [i + 0.3] * 2, color="black", linewidth=1.2)
        ax.scatter([row["geometric_mean"]], [i + 0.3], marker="|", s=70, color="black", zorder=4)
        ax.text(25, i + 0.3, f"{row['geometric_mean']:.2g}", fontsize=NOTE, va="center", ha="center")
    ax.axvline(1.0, color="0.4", linestyle="--", linewidth=0.8)
    ax.set_xscale("log"); ax.set_xlim(3e-4, 60)
    ax.text(25, -0.45, "GM", fontsize=NOTE, va="center", ha="center", color="0.3")
    ax.set_yticks(range(len(order))); ax.set_yticklabels([MODELS[m][1] for m in order])
    ax.set_ylim(len(order) - 0.3, -0.6)
    ax.set_xlabel("Field-to-laboratory ratio\n(black: geometric mean, 95% CI)")
    panel_label(ax, "b", x=-0.42)

    # (c) global flux scenarios
    ax = axes[1, 0]
    world = flux[flux["Ce_case"] == "world_average"].set_index("lab_model")
    scen = [("Cu_kg_yr_low_MP_flux", "Microplastics\n6.1 kt yr$^{-1}$"), ("Cu_kg_yr_central_MP_flux", "Microplastics\n53 kt yr$^{-1}$"),
            ("Cu_kg_yr_high_MP_flux", "All plastics\n500 kt yr$^{-1}$\n(upper bound)")]
    xs = np.arange(len(scen))
    bars = ax.bar(xs, [world.loc[MAIN, c] for c, _ in scen], 0.55, color=MODELS[MAIN][2])
    ax.bar_label(bars, labels=[f"{float(f'{b.get_height():.2g}'):,.0f}" for b in bars], fontsize=NOTE, padding=2)
    for m in [k for k in MODELS if k != MAIN]:
        ax.scatter(xs + 0.36, [world.loc[m, c] for c, _ in scen], s=14, color=MODELS[m][2], zorder=3, label=MODELS[m][0])
    dissolved_kg = world.loc[MAIN, "Cu_kg_yr_central_MP_flux"] / world.loc[MAIN, "pct_of_dissolved_central"] * 100
    ax.set_yscale("log"); ax.set_ylim(5, 5e6)
    ax.set_xticks(xs); ax.set_xticklabels([s for _, s in scen])
    ax.set_ylabel("MP-bound Cu exported by rivers (kg yr$^{-1}$)")
    ax.legend(frameon=False, loc="upper left", title="Other laboratory models", title_fontsize=NOTE,
              borderaxespad=0.2, handletextpad=0.2, labelspacing=0.3)
    ax.text(0.98, 0.98, f"Dissolved Cu flux:\n{dissolved_kg / 1e3:,.0f} t yr$^{{-1}}$\nMP-bound share\n≤ "
            f"{world.loc[MAIN, 'pct_of_dissolved_high']:.4f}%", transform=ax.transAxes, va="top", ha="right", fontsize=NOTE)
    panel_label(ax, "c")

    # (d) rivers with measured inputs
    ax = axes[1, 1]
    names = [r.split(" (")[0].replace("Pearl River system", "Pearl") for r in rivers["river"]]
    cases = [("12.5ug", "12.5 µg per particle", "#1b4f72"), ("170ug", "170 µg per particle", "#5dade2"),
             ("230ug", "230 µg per particle", "#aed6f1")]
    xr = np.arange(len(rivers))
    for i, (key, label, color) in enumerate(cases):
        mid = rivers[f"Cu_kg_yr_{key}"]
        err = [mid - rivers[f"Cu_kg_yr_{key}_low"], rivers[f"Cu_kg_yr_{key}_high"] - mid]
        ax.bar(xr + (i - 1) * 0.27, mid, 0.27, color=color, label=label, yerr=err, error_kw={"linewidth": 0.6, "capsize": 1.5})
    ax.set_yscale("log"); ax.set_xticks(xr); ax.set_xticklabels(names)
    lo, hi = ax.get_ylim(); ax.set_ylim(lo, hi * 25)
    ax.set_ylabel("MP-bound Cu (kg yr$^{-1}$)")
    ax.legend(frameon=False, loc="upper right", title="Assumed particle mass", title_fontsize=NOTE,
              borderaxespad=0.2, labelspacing=0.3)
    panel_label(ax, "d", x=-0.17)

    fig.tight_layout(h_pad=1.6, w_pad=1.2)
    OUT.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT / "Fig_5_field_and_scaleup.png", dpi=DPI)
    fig.savefig(OUT / "Fig_5_field_and_scaleup.tiff", dpi=DPI, pil_kwargs={"compression": "tiff_lzw"})
    plt.close(fig)
    print("saved Fig_5_field_and_scaleup")


if __name__ == "__main__":
    main()
