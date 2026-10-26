"""Fig. 3 (model interpretation), redrawn for the revision from the saved analysis outputs.

Layout: (a) global SHAP importance and (b) SHAP effect of Ce on the top row; (c) response curves of
the isotherm-anchored XGBoost centred on the bottom row. Reads only outputs/reports (written by
src/analysis/interpretation_main_model.py); no model is refitted.

Outputs: outputs/figures/interpretation_main_model.{png,tiff} and
Manuscript/Submission_Env pollution/Revision/02_Figures/Fig_3_interpretation.{png,tiff} (600 dpi).
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.gridspec import GridSpec
from matplotlib.lines import Line2D

ROOT = Path(__file__).resolve().parents[1]
REP = ROOT / "outputs" / "reports"
FIG_DIRS = [ROOT / "outputs" / "figures", ROOT / "Manuscript" / "Submission_Env pollution" / "Revision" / "02_Figures"]
NAMES = ["interpretation_main_model", "Fig_3_interpretation"]
DPI = 600
BLUE = "#2f6f9f"
RIVER_BAND = (0.001, 0.01)
BLEND_PCT = (5, 25)                # transition zone of the isotherm-anchored model (percentiles of Ce)
POLY_COLORS = {"PA": "#1b9e77", "PE": "#d95f02", "PLA": "#7570b3", "PP": "#e7298a", "PS": "#66a61e", "PVC": "#a6761d"}

plt.rcParams.update({
    "font.family": "serif", "font.serif": ["Times New Roman", "Liberation Serif", "DejaVu Serif"],
    "mathtext.fontset": "custom", "mathtext.rm": "Times New Roman", "mathtext.it": "Times New Roman:italic",
    "mathtext.bf": "Times New Roman:bold",
    "font.size": 10, "axes.labelsize": 10.5, "xtick.labelsize": 9.5, "ytick.labelsize": 9.5,
    "legend.fontsize": 8.5, "axes.linewidth": 0.9, "xtick.major.width": 0.9, "ytick.major.width": 0.9,
    "axes.spines.top": False, "axes.spines.right": False,
})
if not any("Times New Roman" in f.name for f in matplotlib.font_manager.fontManager.ttflist):
    plt.rcParams.update({"mathtext.rm": "Liberation Serif", "mathtext.it": "Liberation Serif:italic",
                         "mathtext.bf": "Liberation Serif:bold"})


def panel_label(fig, ax, letter: str, x: float) -> None:
    """Label in figure coordinates, at a fixed height above the axes."""
    fig.text(x, ax.get_position().y1 + 0.012, letter, fontsize=13, fontweight="bold", va="bottom", ha="left")


def main() -> None:
    imp = pd.read_csv(REP / "interpretation_shap_importance.csv")
    dep = pd.read_csv(REP / "interpretation_shap_ce_dependence.csv")
    cur = pd.read_csv(REP / "interpretation_response_curves.csv")
    zone = tuple(np.percentile(dep["Ce_mg_L"], BLEND_PCT))

    fig = plt.figure(figsize=(7.48, 7.2))
    gs = GridSpec(2, 4, figure=fig, height_ratios=[1, 1.05], hspace=0.42, wspace=0.95,
                  left=0.17, right=0.985, top=0.955, bottom=0.075)
    ax_a, ax_b, ax_c = fig.add_subplot(gs[0, :2]), fig.add_subplot(gs[0, 2:]), fig.add_subplot(gs[1, 1:3])
    # centre panel c with its axis label and legend: same width as panel b
    pb, pc = ax_b.get_position(), ax_c.get_position()
    ylab, leg = 0.085, 0.18          # approximate widths of the y-axis label and legend (figure fraction)
    ax_c.set_position([0.5 - (pb.width + leg - ylab) / 2, pc.y0, pb.width, pc.height])

    # (a) global importance
    names = {"Ce": "Equilibrium\nconcentration ($C_e$)", "ReT": "Polymer type", "AgS": "Aging status",
             "AdC": "Adsorption\nsystem"}
    part = imp.iloc[::-1]
    y = np.arange(len(part))
    err = [part["share_pct"] - part["share_pct_min_leave_one_study_out"],
           part["share_pct_max_leave_one_study_out"] - part["share_pct"]]
    ax_a.barh(y, part["share_pct"], height=0.62, color=BLUE, xerr=err,
              error_kw={"elinewidth": 1.0, "capsize": 3, "capthick": 1.0})
    for yi, (v, hi) in enumerate(zip(part["share_pct"], part["share_pct_max_leave_one_study_out"])):
        ax_a.text(hi + 1.2, yi, f"{v:.1f}%", va="center", fontsize=9)
    ax_a.set_yticks(y)
    ax_a.set_yticklabels([names[p] for p in part["predictor"]])
    ax_a.set_xlim(0, 60)
    ax_a.set_xlabel("Share of mean |SHAP| (%)")
    panel_label(fig, ax_a, "a", 0.012)

    # (b) SHAP effect of Ce
    ax_b.scatter(dep["Ce_mg_L"], dep["factor_on_qe"], s=9, color=BLUE, alpha=0.5, linewidths=0)
    ax_b.axhline(1.0, color="0.45", linestyle="--", linewidth=0.9)
    ax_b.set_xscale("log"); ax_b.set_yscale("log")
    ax_b.set_xlabel("$C_e$ (mg L$^{-1}$)")
    ax_b.set_ylabel("SHAP effect of $C_e$\n(factor on $q_e$)")
    panel_label(fig, ax_b, "b", ax_b.get_position().x0 - 0.115)

    # (c) response curves of the isotherm-anchored XGBoost
    ax_c.axvspan(*RIVER_BAND, color="#2e86c1", alpha=0.13, lw=0)
    ax_c.axvspan(*zone, color="0.5", alpha=0.13, lw=0)
    for p, g in cur.groupby("ReT"):
        col = POLY_COLORS.get(p, "0.3")
        a = g[g["AgS"] == "Aged"]; v = g[g["AgS"] == "Virgin"]
        ax_c.plot(a["Ce_mg_L"], a["qe_mg_g"], color=col, linewidth=1.6)
        ax_c.plot(v["Ce_mg_L"], v["qe_mg_g"], color=col, linewidth=1.1, linestyle=":")
    ax_c.set_xscale("log"); ax_c.set_yscale("log")
    ax_c.set_xlabel("$C_e$ (mg L$^{-1}$)"); ax_c.set_ylabel("Predicted $q_e$ (mg g$^{-1}$)")
    ylim = ax_c.get_ylim()
    ax_c.text(np.sqrt(RIVER_BAND[0] * RIVER_BAND[1]), ylim[1], "River", ha="center", va="top", fontsize=8.5, color="#1f618d")
    ax_c.text(np.sqrt(zone[0] * zone[1]), ylim[1], "Transition", ha="center", va="top", fontsize=8.5, color="0.35")
    counts = cur.groupby("ReT")["n_studies"].first()
    poly_handles = [Line2D([], [], color=POLY_COLORS.get(p, "0.3"), lw=1.8, label=f"{p} ({int(n)})") for p, n in counts.items()]
    style_handles = [Line2D([], [], color="0.2", lw=1.6, label="Aged"),
                     Line2D([], [], color="0.2", lw=1.1, ls=":", label="Virgin")]
    leg1 = ax_c.legend(handles=poly_handles, title="Polymer (studies)", title_fontsize=8.5, frameon=False,
                       loc="upper left", bbox_to_anchor=(1.02, 1.0), borderaxespad=0)
    ax_c.add_artist(leg1)
    ax_c.legend(handles=style_handles, frameon=False, loc="lower left", bbox_to_anchor=(1.02, 0.0), borderaxespad=0)
    # centre panel c on the page: shift it so that the whole block (y-axis label, axes and legends) is centred
    fig.canvas.draw()
    bb = ax_c.get_tightbbox(fig.canvas.get_renderer()).transformed(fig.transFigure.inverted())
    pos = ax_c.get_position()
    ax_c.set_position([pos.x0 + 0.5 - (bb.x0 + bb.x1) / 2, pos.y0, pos.width, pos.height])
    panel_label(fig, ax_c, "c", ax_c.get_position().x0 - 0.115)

    for d in FIG_DIRS:
        d.mkdir(parents=True, exist_ok=True)
    for d, n in zip(FIG_DIRS, NAMES):
        fig.savefig(d / f"{n}.png", dpi=DPI)
        fig.savefig(d / f"{n}.tiff", dpi=DPI, pil_kwargs={"compression": "tiff_lzw"})
        print(f"saved {d / n}.png/.tiff")
    plt.close(fig)


if __name__ == "__main__":
    main()
