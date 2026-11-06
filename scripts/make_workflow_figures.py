"""Fig. 1 (workflow), Fig. S2 (LLM interpretation and evaluation) and the graphical abstract of the revised manuscript.

Numbers shown in the figures are read from the result files where they exist.
Outputs: Revision/02_Figures/Fig_1_workflow, Fig_S2_llm_workflow (.png, .tiff) and Graphical_abstract (.png, .tiff, .svg with editable text)
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from config import REPORTS_DIR  # noqa: E402

OUT = ROOT / "Manuscript" / "Submission_Env pollution" / "Revision" / "02_Figures"
NAVY, BLUE, ORANGE, GREEN, GREY, RED = "#1b4f72", "#5dade2", "#f39c12", "#2e7d32", "#566573", "#c0392b"
FILL = {"data": "#eaf2f8", "model": "#fef5e7", "field": "#e8f5e9", "llm": "#f4ecf7", "plain": "#f4f6f7"}
EDGE = {"data": NAVY, "model": "#b9770e", "field": GREEN, "llm": "#6c3483", "plain": GREY}


def box(ax, x, y, w, h, title, lines=(), kind="plain", fs=7.6, title_fs=8.6):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.004,rounding_size=0.012", fc=FILL[kind], ec=EDGE[kind], lw=1.1))
    ax.text(x + w / 2, y + h - 0.012, title, ha="center", va="top", fontsize=title_fs, fontweight="bold", color=EDGE[kind])
    if lines:
        ax.text(x + 0.012, y + h - 0.012 - 0.028 * (1 + title.count("\n")) - 0.008, "\n".join(lines), ha="left", va="top", fontsize=fs,
                linespacing=1.4, color="#1c2833")


def arrow(ax, p0, p1, color=GREY, style="-|>", rad=0.0, lw=1.2):
    ax.add_patch(FancyArrowPatch(p0, p1, arrowstyle=style, mutation_scale=11, color=color, lw=lw,
                                 connectionstyle=f"arc3,rad={rad}", shrinkA=1, shrinkB=1))


def numbers() -> dict:
    d = pd.read_csv(ROOT / "data" / "dataset_grouped.csv")
    faf = pd.read_csv(REPORTS_DIR / "faf_scaling_summary.csv")
    faf = faf[(faf["lab_model"] == "freundlich_pooled") & (faf["field_subset"] == "without_Purwiyanto_2020")].iloc[0]
    iv = pd.read_csv(REPORTS_DIR / "final_prediction_intervals.csv")
    iv = iv[(iv["model"] == "freundlich_pooled") & np.isclose(iv["nominal"], 0.9)].set_index("case")["interval_factor_final"]
    gl = pd.read_csv(REPORTS_DIR / "global_flux_constrained.csv")
    gl = gl[(gl["lab_model"] == "freundlich_pooled") & (gl["Ce_case"] == "world_average")].iloc[0]
    t1 = pd.read_csv(REPORTS_DIR / "table1_model_performance.csv").set_index(["tier", "model"])["median_fold_error"]
    llm = pd.read_csv(REPORTS_DIR / "llm_eval_v2_condition_differences_final.csv")
    llm = llm[llm["judge"] == "independent_provider"].set_index("quantity")["mean"]
    pc = pd.read_csv(ROOT / "Manuscript" / "Submission_Env pollution" / "Revision" / "05_Literature_search" / "PRISMA" / "prisma_counts.csv",
                     index_col="item")["n"]
    return {"screened": int(pc["records_screened"]), "full_text": int(pc["full_text_assessed"]),
            "n": len(d), "studies": d["source_id"].nunique(), "isotherms": d["curve_id"].nunique(), "ratio": faf["geometric_mean"],
            "ratio_lo": faf["gm_ci95_low"], "ratio_hi": faf["gm_ci95_high"], "f_new": iv["new_laboratory"], "f_anchor": iv["one_anchor_per_isotherm"],
            "g_low": gl["Cu_kg_yr_low_MP_flux"], "g_mid": gl["Cu_kg_yr_central_MP_flux"], "g_pct": gl["pct_of_dissolved_central"],
            "mfe_loso": t1[("leave_one_study_out", "freundlich_pooled")], "mfe_ml": t1[("random", "xgb_monotone")],
            "c0": llm["overall C0"], "c1": llm["overall C1"], "c3": llm["overall C3"]}


def save(fig, name: str, dpi: int = 600, svg: bool = False) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT / f"{name}.png", dpi=dpi, bbox_inches="tight", facecolor="white")
    fig.savefig(OUT / f"{name}.tiff", dpi=dpi, bbox_inches="tight", facecolor="white", pil_kwargs={"compression": "tiff_lzw"})
    if svg:
        with plt.rc_context({"svg.fonttype": "none"}):    # text stays as editable text, not outlines
            fig.savefig(OUT / f"{name}.svg", bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print("saved", name)


def fig1(n: dict) -> None:
    fig, ax = plt.subplots(figsize=(12.2, 5.4))
    ax.set_xlim(-0.004, 1.006)
    ax.set_ylim(0, 1)
    ax.axis("off")
    w, h, y1, y2 = 0.232, 0.40, 0.58, 0.02
    xs = [0.005, 0.259, 0.513, 0.767]
    box(ax, xs[0], y1, w, h, "1. Laboratory data", ["Scopus, PubMed and Web of Science", f"{n['screened']} records screened, {n['full_text']} read in full",
                                                     f"{n['studies']} studies, {n['isotherms']} isotherms,", f"{n['n']} equilibrium observations",
                                                     "Mass-balance check of every study:", "C$_0$ = C$_e$ + dose × q$_e$"], "data")
    box(ax, xs[1], y1, w, h, "2. Models", ["Target log$_{10}$ q$_e$; predictors log$_{10}$ C$_e$,", "polymer, aging status, system",
                                            "ML: RF, monotone XGB and LightGBM,", "isotherm-anchored XGBoost",
                                            "Linear isotherm-form model", "(pooled Freundlich)"], "model")
    box(ax, xs[2], y1, w, h, "3. Validation by study", ["Random cross-validation", "Leave-one-isotherm-out", "Leave-one-study-out (new laboratory)",
                                                         "Low-concentration hold-out", "One local measurement (anchored)",
                                                         "→ domain of each model"], "model")
    box(ax, xs[3], y1, w, h, "4. Uncertainty and\ninterpretation", ["Study-level conformal intervals:", f"×/÷{n['f_new']:.0f} for a new laboratory,",
                                                                   f"×/÷{n['f_anchor']:.1f} with one local measurement", "SHAP importance and its stability",
                                                                   "Effects of C$_e$, polymer, aging"], "model")
    box(ax, xs[0], y2, w, h, "5. Field comparison", ["Cu on microplastics collected in rivers", "paired with dissolved Cu in the water:",
                                                      "13 records from 6 studies", "Field-to-laboratory ratio R =",
                                                      "q$_{field}$ / q$_{e,lab}$(C$_e$)", f"R = {n['ratio']:.2f} (95% CI {n['ratio_lo']:.2f}–{n['ratio_hi']:.1f})"], "field")
    box(ax, xs[1], y2, w, h, "6. River and global scale", ["M$_{Cu}$ = F$_{MP}$ × q$_{e,lab}$(C$_e$) × R", "F$_{MP}$: published microplastic mass",
                                                            "fluxes (6.1–53 kt yr$^{-1}$)", "Four rivers with measured inputs",
                                                            "Compared with the dissolved Cu flux"], "field")
    box(ax, xs[2], y2, w, h, "7. LLM Interpretation System", ["C1 rule-based; C2 retrieval (RAG);", "C3 retrieval + knowledge graph",
                                                               "Corpus: 41 source articles", "States interval, validity warnings,",
                                                               "mechanisms and sources", "Numerical check against the model"], "llm")
    box(ax, xs[3], y2, w, h, "8. Evaluation and application", ["30 cases × 4 conditions", "Independent LLM judge and", "generator model as judge",
                                                                "Domain experts (blinded subset)", "Judge-independent faithfulness", "Streamlit application, reports"], "llm")
    for i in range(3):
        arrow(ax, (xs[i] + w, y1 + h / 2), (xs[i + 1], y1 + h / 2))
        arrow(ax, (xs[i] + w, y2 + h / 2), (xs[i + 1], y2 + h / 2))
    ax.plot([xs[0] + w / 2, xs[3] + w / 2], [y1 - 0.045, y1 - 0.045], color=GREY, lw=1.2)
    for x in (xs[0] + w / 2, xs[3] + w / 2):
        ax.plot([x, x], [y1, y1 - 0.045], color=GREY, lw=1.2)
    for x in (xs[0] + w / 2, xs[2] + w / 2):
        arrow(ax, (x, y1 - 0.045), (x, y2 + h))
    ax.text(0.5, y1 - 0.045, "selected model, prediction intervals and validity limits", ha="center", va="center", fontsize=8, color=GREY,
            bbox={"fc": "white", "ec": "none", "pad": 2})
    save(fig, "Fig_1_workflow")


def fig_s2(n: dict) -> None:
    """Two input rows (case -> model output; corpus -> context) feed generation, then the explanations and their evaluation.
    Axis units are inches, so box sizes and font sizes scale together; all arrows are straight."""
    fs, tfs = 10.0, 10.8
    W, H = 12.25, 3.75
    fig, ax = plt.subplots(figsize=(W, H))
    fig.subplots_adjust(0, 0, 1, 1)
    ax.set_xlim(0, W)
    ax.set_ylim(0, H)
    ax.axis("off")

    def card(x, y, w, h, title, lines, kind):
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0,rounding_size=0.08", fc=FILL[kind], ec=EDGE[kind], lw=1.2))
        ax.text(x + w / 2, y + h - 0.12, title, ha="center", va="top", fontsize=tfs, fontweight="bold", color=EDGE[kind])
        ax.text(x + 0.14, y + h - 0.46, "\n".join(lines), ha="left", va="top", fontsize=fs, linespacing=1.35, color="#1c2833")

    g, hr, y_bot = 0.36, 1.55, 0.06                    # gap between columns, row height, bottom of the lower row
    y_top = y_bot + hr + 0.32
    tall = y_top + hr - y_bot
    xs, ws = [0.04], [2.05, 2.30, 2.15, 1.95, 2.25]
    for w in ws[:-1]:
        xs.append(xs[-1] + w + g)
    card(xs[0], y_top, ws[0], hr, "Case", ["Polymer, aging status,", "adsorption system, C$_e$", "Optional: one local", "(C$_e$, q$_e$) measurement"], "data")
    card(xs[0], y_bot, ws[0], hr, "Corpus", ["41 full-text articles", "(32 laboratory and", "7 field studies)", "1,219 embedded passages",
                                             "Polymer–mechanism relations"], "data")
    card(xs[1], y_top, ws[1], hr, "Model output (facts)", ["q$_e$ with 90% interval", "Field-scale value (q$_e$ × R)", "Contribution of each input",
                                                           "Validity warnings"], "model")
    card(xs[1], y_bot, ws[1], hr, "Context", ["C2, C3: five passages most", "similar to the case", "C3: mechanisms reported for", "the polymer, with supporting",
                                              "sentences"], "llm")
    card(xs[2], y_bot, ws[2], tall, "Generation and check", ["GPT-4o mini writes the", "explanation from the facts", "and the context", "",
                                                             "Cites only sources in", "the context", "", "Values in mg g$^{-1}$ checked",
                                                             "against the model output;", "regenerated if different"], "llm")
    card(xs[3], y_bot, ws[3], tall, "Explanations", ["C0  numbers only", "C1  rule-based text", "      (no LLM)", "C2  LLM + passages", "C3  LLM + passages", "      + knowledge graph"], "llm")
    card(xs[4], y_bot, ws[4], tall, "Evaluation", ["30 cases × 4 conditions", "= 120 explanations", "", "Two blinded LLM judges:", "Claude Haiku 4.5", "GPT-4o mini (generator)",
                                                   "", "Two blinded experts:", "15 cases", "", "Judge-independent", "measures (Table S13)"], "plain")
    ym_top, ym_bot = y_top + hr / 2, y_bot + hr / 2
    arrow(ax, (xs[0] + ws[0], ym_top), (xs[1], ym_top))
    arrow(ax, (xs[0] + ws[0], ym_bot), (xs[1], ym_bot), color=EDGE["llm"])
    arrow(ax, (xs[1] + ws[1], ym_top), (xs[2], ym_top))
    arrow(ax, (xs[1] + ws[1], ym_bot), (xs[2], ym_bot))
    arrow(ax, (xs[0] + ws[0] / 2, y_top), (xs[0] + ws[0] / 2, y_bot + hr), color=EDGE["llm"])
    ax.text(xs[0] + ws[0] / 2 + 0.08, (y_top + y_bot + hr) / 2, "query", fontsize=fs - 1.5, color=EDGE["llm"], ha="left", va="center")
    for k in (2, 3):
        arrow(ax, (xs[k] + ws[k], y_bot + tall / 2), (xs[k + 1], y_bot + tall / 2))
    save(fig, "Fig_S2_llm_workflow")


def graphical_abstract(n: dict) -> None:
    # Drawn at the size at which Elsevier checks readability: 13 cm wide (template 531 x 1328 px, h x w), readable on a
    # 96-dpi screen, so the font sizes below are the displayed sizes; the smallest text is 8 pt. The three columns share
    # one content band (top of the plot frame to the bottom of the x-axis label), so frames and boxes line up.
    cm = 1 / 2.54
    W, H = 13 * cm, 5.2 * cm
    fig = plt.figure(figsize=(W, H))
    FS, FS_HEAD, FS_TITLE, INK = 8, 8.5, 9, "#1c2833"
    top, bottom = 0.83, 0.05                     # content band (figure fraction)
    cols = [(0.095, 0.315), (0.385, 0.655), (0.725, 0.995)]
    title_y = 0.935

    def column_axes(x0, x1):
        ax = fig.add_axes([x0, bottom, x1 - x0, top - bottom])
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.axis("off")
        return ax

    def panel(ax, y0, y1, kind, lw=0.9):
        ax.add_patch(FancyBboxPatch((0.012, y0), 0.976, y1 - y0, boxstyle="round,pad=0,rounding_size=0.045",
                                    fc=FILL[kind], ec=EDGE[kind], lw=lw, mutation_aspect=(W * 0.27) / (H * 0.75)))

    for (x0, x1), title in zip(cols, (f"{n['n']} lab data, {n['studies']} studies", "ML model + LLM", "River scale")):
        fig.text((x0 + x1) / 2, title_y, title, fontsize=FS_TITLE, fontweight="bold", color=NAVY, ha="center", va="center")

    # 1. laboratory isotherms; the plot frame spans the upper part of the band, the axis label its lower part
    ax = fig.add_axes([cols[0][0], 0.27, cols[0][1] - cols[0][0], top - 0.27])
    d = pd.read_csv(ROOT / "data" / "dataset_grouped.csv")
    ax.scatter(d["Ce"], d["qe"], s=1.0, color=BLUE, alpha=0.5, lw=0)
    ce = np.logspace(-4, 2.5, 50)
    pred = pd.read_csv(REPORTS_DIR / "interpretation_freundlich_effects.csv").set_index("effect")["estimate"]
    line = pred.iloc[0] * 3.0 * ce ** pred["Freundlich exponent n"]
    ax.fill_between(ce, line / n["f_new"], line * n["f_new"], color=NAVY, alpha=0.10, lw=0)
    ax.plot(ce, line, color=NAVY, lw=1.0)
    ax.axvspan(1e-3, 1e-2, color=ORANGE, alpha=0.30, lw=0)
    ax.text(3.2e-3, 1.5e1, "rivers", color="#b9770e", fontsize=FS, ha="center", fontweight="bold")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim(1e-4, 3e2)
    ax.set_ylim(1e-4, 1e2)
    sup = {-4: "10⁻⁴", -2: "10⁻²", 0: "10⁰", 2: "10²"}                    # Unicode, so that the SVG text stays editable
    ax.set_xticks([1e-4, 1e-2, 1e0, 1e2], [sup[k] for k in (-4, -2, 0, 2)])
    ax.set_yticks([1e-4, 1e0], [sup[k] for k in (-4, 0)])
    ax.tick_params(labelsize=FS, length=2, pad=1.5, width=0.6)
    ax.minorticks_off()
    ax.set_xlabel("Cu in water (mg L⁻¹)", fontsize=FS, labelpad=1.5)
    ax.set_ylabel("Cu on MPs (mg g⁻¹)", fontsize=FS, labelpad=1)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_linewidth(0.6)

    # 2. model and explanation: three boxes of equal height
    ax = column_axes(*cols[1])
    gap = 0.045
    hb = (1 - 2 * gap) / 3
    rows = [("New laboratory", f"median fold error {n['mfe_loso']:.1f}", "model"),
            ("90% interval", f"×/÷{n['f_new']:.0f} → ×/÷{n['f_anchor']:.1f} with\none local measurement", "data"),
            ("LLM explanation", f"score {n['c3']:.1f} of 5;\npredictions unchanged", "llm")]
    for k, (label, value, kind) in enumerate(rows):
        y1 = 1 - k * (hb + gap)
        panel(ax, y1 - hb, y1, kind)
        ax.text(0.07, y1 - 0.045, label, fontsize=FS_HEAD, fontweight="bold", color=EDGE[kind], va="top")
        ax.text(0.07, y1 - 0.125, value, fontsize=FS, color=INK, va="top", linespacing=1.1)

    # 3. river scale: two boxes of equal height, aligned with the column of boxes
    ax = column_axes(*cols[2])
    hb2 = (1 - gap) / 2
    g_low, g_mid = (float(f"{n[k]:.2g}") for k in ("g_low", "g_mid"))        # two significant figures, as in the text
    pct = float(f"{n['g_pct']:.1g}")                                        # one significant figure, as in the Abstract
    boxes = [("Field MPs carry", f"{n['ratio']:.2f}", "of lab-predicted Cu", "field", GREEN, INK),
             ("MP-bound Cu", f"{g_low:.0f}–{g_mid:.0f} kg yr⁻¹", f"≤{pct:g}% of dissolved Cu", "data", NAVY, RED)]
    for k, (head, big, foot, kind, big_color, foot_color) in enumerate(boxes):
        y1 = 1 - k * (hb2 + gap)
        panel(ax, y1 - hb2, y1, kind)
        ax.text(0.5, y1 - 0.05, head, fontsize=FS_HEAD, ha="center", va="top", color=INK)
        ax.text(0.5, y1 - hb2 / 2 - 0.01, big, fontsize=13, ha="center", va="center", color=big_color, fontweight="bold")
        ax.text(0.5, y1 - hb2 + 0.055, foot, fontsize=FS, ha="center", va="bottom", color=foot_color)

    for (a, b) in ((cols[0][1], cols[1][0]), (cols[1][1], cols[2][0])):
        fig.text((a + b) / 2 + 0.003, (top + bottom) / 2, "→", fontsize=14, color=GREY, ha="center", va="center")
    save(fig, "Graphical_abstract", dpi=600, svg=True)


def main() -> None:
    plt.rcParams.update({"font.family": "Times New Roman", "mathtext.fontset": "custom", "mathtext.rm": "Times New Roman", "mathtext.it": "Times New Roman:italic", "mathtext.bf": "Times New Roman:bold", "mathtext.default": "regular"})
    if "--s2-only" in sys.argv:                  # Fig. S2 uses no result files
        fig_s2({})
        return
    if "--ga-only" in sys.argv:
        graphical_abstract(numbers())
        return
    n = numbers()
    fig1(n)
    fig_s2(n)
    graphical_abstract(n)


if __name__ == "__main__":
    main()
