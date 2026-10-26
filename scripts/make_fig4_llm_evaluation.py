"""Fig. 4 of the revised manuscript: evaluation of the LLM explanation system (30 cases, conditions C0-C3).

(a) Overall score by condition for the two LLM judges, with 95% bootstrap intervals over cases.
(b) Scores of the independent judge by rubric dimension.
(c) Measures that need no judge.

Layout of the revision (Oct 2026): a and b on the top row, c centred on the bottom row; one legend for the
four conditions at the top. Reads only outputs/reports; no model or judge is rerun.
Output: Revision/02_Figures/Fig_4_llm_evaluation.png / .tiff (600 dpi)
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.gridspec import GridSpec
from matplotlib.patches import Patch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
REPORTS_DIR = PROJECT_ROOT / "outputs" / "reports"
OUT = PROJECT_ROOT / "Manuscript" / "Submission_Env pollution" / "Revision" / "02_Figures"
DPI = 600
COLORS = {"C0": "#bab0ac", "C1": "#5dade2", "C2": "#f39c12", "C3": "#1b4f72"}
COND_LABELS = {"C0": "C0 numbers only", "C1": "C1 rule-based", "C2": "C2 retrieval",
               "C3": "C3 retrieval + knowledge graph"}
DIMS = {"accuracy": "Accuracy", "mechanistic_depth": "Mechanistic\ndepth", "literature_grounding": "Literature\ngrounding",
        "condition_specificity": "Condition\nspecificity", "readability": "Read-\nability",
        "limits_communicated": "Limits\ncommuni-\ncated"}
JUDGES = {"independent_provider": "Independent judge\n(Claude Haiku 4.5)",
          "same_model_as_generator": "Generator as judge\n(GPT-4o mini)"}
BASE = 0.85            # bars start slightly below the lowest score (1) so that a score of 1 remains visible

plt.rcParams.update({
    "font.family": "serif", "font.serif": ["Times New Roman", "Liberation Serif", "DejaVu Serif"],
    "mathtext.fontset": "custom", "mathtext.rm": "Times New Roman", "mathtext.it": "Times New Roman:italic",
    "mathtext.bf": "Times New Roman:bold",
    "font.size": 10, "axes.labelsize": 10.5, "xtick.labelsize": 9.5, "ytick.labelsize": 9.5, "legend.fontsize": 9.5,
    "axes.linewidth": 0.9, "xtick.major.width": 0.9, "ytick.major.width": 0.9,
    "axes.spines.top": False, "axes.spines.right": False,
})
if not any("Times New Roman" in f.name for f in matplotlib.font_manager.fontManager.ttflist):
    plt.rcParams.update({"mathtext.rm": "Liberation Serif", "mathtext.it": "Liberation Serif:italic",
                         "mathtext.bf": "Liberation Serif:bold"})


def panel_label(fig, ax, letter: str, x: float) -> None:
    fig.text(x, ax.get_position().y1 + 0.012, letter, fontsize=13, fontweight="bold", va="bottom", ha="left")


def main() -> None:
    scores = pd.read_csv(REPORTS_DIR / "llm_eval_v2_scores_final.csv")
    diff = pd.read_csv(REPORTS_DIR / "llm_eval_v2_condition_differences_final.csv")
    summ = pd.read_csv(REPORTS_DIR / "llm_eval_v2_summary_final.csv").set_index("condition")
    conds = list(COLORS)

    fig = plt.figure(figsize=(7.48, 7.3))
    gs = GridSpec(2, 2, figure=fig, width_ratios=[1, 1.6], height_ratios=[1, 0.82], hspace=0.42, wspace=0.3,
                  left=0.085, right=0.985, top=0.905, bottom=0.07)
    ax_a, ax_b, ax_c = fig.add_subplot(gs[0, 0]), fig.add_subplot(gs[0, 1]), fig.add_subplot(gs[1, :])

    # (a) overall score, two LLM judges
    w = 0.19
    for j, judge in enumerate(JUDGES):
        for i, c in enumerate(conds):
            r = diff[(diff["judge"] == judge) & (diff["quantity"] == f"overall {c}")].iloc[0]
            x = j + (i - 1.5) * w
            ax_a.bar(x, r["mean"] - BASE, w, bottom=BASE, color=COLORS[c])
            ax_a.errorbar(x, r["mean"], yerr=[[r["mean"] - r["ci_low"]], [r["ci_high"] - r["mean"]]],
                          color="k", lw=0.9, capsize=2)
            ax_a.text(x, r["ci_high"] + 0.06, f"{round(round(r['mean'], 6), 2):.2f}", ha="center", va="bottom",
                      fontsize=8)
    ax_a.set_xticks(range(len(JUDGES)), JUDGES.values())
    ax_a.set_xlim(-0.5, 1.5)
    ax_a.set_ylim(BASE, 5.45)
    ax_a.set_yticks([1, 2, 3, 4, 5])
    ax_a.set_ylabel("Overall score (1–5)")

    # (b) independent judge by criterion
    ind = scores[scores["judge"] == "independent_provider"].groupby("condition")[list(DIMS)].mean()
    wb = 0.2
    for i, c in enumerate(conds):
        ax_b.bar(np.arange(len(DIMS)) + (i - 1.5) * wb, ind.loc[c] - BASE, wb, bottom=BASE, color=COLORS[c])
    ax_b.set_xticks(range(len(DIMS)), DIMS.values(), fontsize=9)
    ax_b.set_xlim(-0.55, len(DIMS) - 0.45)
    ax_b.set_ylim(BASE, 5.2)
    ax_b.set_yticks([1, 2, 3, 4, 5])
    ax_b.set_ylabel("Score, independent judge (1–5)")

    # (c) judge-independent measures
    rows = {"Predicted values\nstated correctly": lambda r: 100 * min(r["qe_stated"], r["interval_stated"], r["field_value_stated"]),
            "Extrapolation warning\ngiven when it applies": lambda r: 100 * r["range_warning_recall"],
            "Sparse-polymer warning\ngiven when it applies": lambda r: 100 * r["sparse_warning_recall"],
            "Explanations citing\n≥ 2 corpus articles": lambda r: 100 * r["two_or_more_sources"],
            "Citations to an article\non the polymer of the case": lambda r: (100 * r["citations_naming_polymer"] / r["citations_total"]
                                                                               if r["citations_total"] else np.nan)}
    hc = 0.19
    for i, c in enumerate(conds):
        vals = np.array([f(summ.loc[c]) for f in rows.values()], dtype=float)
        y = np.arange(len(rows)) + (i - 1.5) * hc
        ax_c.barh(y, np.nan_to_num(vals), hc, color=COLORS[c])
        for yy, v in zip(y, vals):
            if v == 0:
                ax_c.text(0.8, yy, "0", va="center", ha="left", fontsize=8, color="0.4")
    ax_c.text(0.8, len(rows) - 1 - 1.0 * hc, "C0, C1: no citations", va="center", ha="left", fontsize=8.5, color="0.4")
    ax_c.set_yticks(range(len(rows)), rows.keys())
    ax_c.invert_yaxis()
    ax_c.set_xlim(0, 100)
    ax_c.set_xticks([0, 20, 40, 60, 80, 100])
    ax_c.set_xlabel("Share (%)")

    # centre panel c (axes plus tick labels) on the page; width as panel b
    pb, pc = ax_b.get_position(), ax_c.get_position()
    ax_c.set_position([pc.x0, pc.y0, pb.width * 0.82, pc.height])
    fig.canvas.draw()
    bb = ax_c.get_tightbbox(fig.canvas.get_renderer()).transformed(fig.transFigure.inverted())
    pos = ax_c.get_position()
    ax_c.set_position([pos.x0 + 0.5 - (bb.x0 + bb.x1) / 2, pos.y0, pos.width, pos.height])

    handles = [Patch(color=COLORS[c], label=COND_LABELS[c]) for c in conds]
    fig.legend(handles=handles, loc="upper center", ncol=4, frameon=False, bbox_to_anchor=(0.5, 0.995),
               handlelength=1.2, columnspacing=1.4, handletextpad=0.5)
    panel_label(fig, ax_a, "a", 0.012)
    panel_label(fig, ax_b, "b", ax_b.get_position().x0 - 0.075)
    panel_label(fig, ax_c, "c", ax_c.get_position().x0 - 0.21)

    OUT.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT / "Fig_4_llm_evaluation.png", dpi=DPI)
    fig.savefig(OUT / "Fig_4_llm_evaluation.tiff", dpi=DPI, pil_kwargs={"compression": "tiff_lzw"})
    plt.close(fig)
    print("saved Fig_4_llm_evaluation")


if __name__ == "__main__":
    main()
