"""Fig. 1 (workflow), Fig. S2 (LLM interpretation and evaluation) and the graphical abstract of the revised manuscript.

Numbers shown in the figures are read from the result files where they exist.
Outputs: Revision/Figures_revised/Fig_1_workflow, Fig_S2_llm_workflow, Graphical_abstract (.png and .tiff)
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
            "g_low": gl["Cu_kg_yr_low_MP_flux"], "g_mid": gl["Cu_kg_yr_central_MP_flux"], "g_pct": gl["pct_of_dissolved_high"],
            "mfe_loso": t1[("leave_one_study_out", "freundlich_pooled")], "mfe_ml": t1[("random", "xgb_monotone")],
            "c0": llm["overall C0"], "c1": llm["overall C1"], "c3": llm["overall C3"]}


def save(fig, name: str, dpi: int = 600) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT / f"{name}.png", dpi=dpi, bbox_inches="tight", facecolor="white")
    fig.savefig(OUT / f"{name}.tiff", dpi=dpi, bbox_inches="tight", facecolor="white", pil_kwargs={"compression": "tiff_lzw"})
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
    fig, ax = plt.subplots(figsize=(12.2, 5.6))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    box(ax, 0.01, 0.60, 0.20, 0.33, "Case", ["Polymer", "Aging status", "Adsorption system", "C$_e$ (mg L$^{-1}$)", "Optional: one local", "(C$_e$, q$_e$) measurement"], "data")
    box(ax, 0.25, 0.60, 0.24, 0.33, "Model output (facts)", ["q$_e$ with 90% prediction interval", "Field-scale value (q$_e$ × R)", "Contribution of each input",
                                                             "Warnings: outside laboratory range,", "sparse data, polymer with < 3 studies"], "model")
    box(ax, 0.01, 0.10, 0.20, 0.38, "Corpus", ["32 laboratory and 7 field", "studies (41 articles, full text)", "1,219 passages, embedded", "Relations at sentence level:",
                                               "polymer – mechanism –", "aging – article"], "data")
    box(ax, 0.25, 0.10, 0.24, 0.38, "Context", ["C2 and C3: five passages most", "similar to the case (polymer", "named first, ≤ 2 per article)",
                                                "C3 only: mechanisms reported", "for the polymer, number of", "articles, supporting sentences"], "llm")
    box(ax, 0.53, 0.60, 0.22, 0.33, "Explanation", ["C0  numbers only", "C1  rule-based text (no LLM)", "C2  LLM + retrieved passages", "C3  LLM + passages +",
                                                     "      knowledge-graph relations"], "llm")
    box(ax, 0.53, 0.10, 0.22, 0.38, "Generation and check", ["GPT-4o-mini writes the text", "from the facts and the context", "Cites only sources in the context",
                                                             "Every value in mg g$^{-1}$ compared", "with the model output;", "one regeneration if it differs"], "llm")
    box(ax, 0.79, 0.10, 0.20, 0.83, "Evaluation", ["30 cases × 4 conditions", "= 120 explanations", "", "LLM judges, blinded:", "• Claude Haiku 4.5", "  (independent)",
                                                   "• GPT-4o-mini", "  (generator model)", "", "Domain experts:", "15 cases, blinded", "",
                                                   "Six rubric criteria (1–5)", "", "Judge-independent:", "values, warnings,", "citations, mechanisms"], "plain")
    arrow(ax, (0.21, 0.765), (0.25, 0.765))
    arrow(ax, (0.49, 0.765), (0.53, 0.765))
    arrow(ax, (0.21, 0.29), (0.25, 0.29))
    arrow(ax, (0.49, 0.29), (0.53, 0.29))
    arrow(ax, (0.37, 0.60), (0.60, 0.48), rad=0.0)
    arrow(ax, (0.64, 0.48), (0.64, 0.60))
    arrow(ax, (0.75, 0.765), (0.79, 0.765))
    arrow(ax, (0.11, 0.60), (0.33, 0.48), rad=0.0, color="#6c3483", lw=1.0)
    ax.text(0.185, 0.545, "query", fontsize=7.2, color="#6c3483", ha="center")
    save(fig, "Fig_S2_llm_workflow")


def graphical_abstract(n: dict) -> None:
    fig = plt.figure(figsize=(13.0, 5.2))
    gs = fig.add_gridspec(1, 3, width_ratios=[1.0, 1.05, 1.0], wspace=0.18, left=0.03, right=0.985, top=0.80, bottom=0.14)
    fig.suptitle("From laboratory isotherms to rivers: Cu on microplastics", fontsize=15, fontweight="bold", color=NAVY, y=0.965)

    ax = fig.add_subplot(gs[0])
    d = pd.read_csv(ROOT / "data" / "dataset_grouped.csv")
    ax.scatter(d["Ce"], d["qe"], s=5, color=BLUE, alpha=0.45, lw=0)
    ce = np.logspace(-4, 2.5, 50)
    pred = pd.read_csv(REPORTS_DIR / "interpretation_freundlich_effects.csv").set_index("effect")["estimate"]
    ref, expo = pred.iloc[0], pred["Freundlich exponent n"]
    line = ref * 3.0 * ce ** expo
    ax.fill_between(ce, line / n["f_new"], line * n["f_new"], color=NAVY, alpha=0.10, lw=0)
    ax.plot(ce, line, color=NAVY, lw=2)
    ax.axvspan(1e-3, 1e-2, color=ORANGE, alpha=0.25, lw=0)
    ax.text(3.2e-3, 3e1, "rivers", color="#b9770e", fontsize=10, ha="center", fontweight="bold")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim(1e-4, 3e2)
    ax.set_ylim(1e-4, 1e2)
    ax.set_xlabel("Dissolved Cu, C$_e$ (mg L$^{-1}$)", fontsize=10)
    ax.set_ylabel("Cu on microplastics, q$_e$ (mg g$^{-1}$)", fontsize=10)
    ax.set_title(f"Laboratory data\n{n['n']} observations, {n['studies']} studies", fontsize=11.5, color=NAVY, fontweight="bold")
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)

    ax = fig.add_subplot(gs[1])
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    ax.set_title("Machine learning validated by study\n+ LLM interpretation", fontsize=11.5, color=NAVY, fontweight="bold")
    # a median fold error of x means that half of the predictions are within a factor of x, not all of them
    rows = [("Known laboratories", f"ML: median fold error {n['mfe_ml']:.1f}", FILL["model"], EDGE["model"]),
            ("New laboratory", f"all models: median fold error {n['mfe_loso']:.1f}", FILL["model"], EDGE["model"]),
            ("90% interval", f"×/÷{n['f_new']:.0f}  →  ×/÷{n['f_anchor']:.1f} with one\nlocal measurement", FILL["data"], EDGE["data"]),
            ("Knowledge-graph LLM", f"explanation quality {n['c0']:.1f} → {n['c3']:.1f} of 5\n(independent judge)", FILL["llm"], EDGE["llm"])]
    y = 0.80
    for label, value, fc, ec in rows:
        hgt = 0.215 if "\n" in value else 0.165
        ax.add_patch(FancyBboxPatch((0.02, y - hgt + 0.16), 0.96, hgt, boxstyle="round,pad=0.005,rounding_size=0.02", fc=fc, ec=ec, lw=1.2))
        ax.text(0.06, y + 0.125, label, fontsize=10.5, fontweight="bold", color=ec, va="center")
        ax.text(0.06, y + 0.085, value, fontsize=10, color="#1c2833", va="top", ha="left", linespacing=1.25)
        y -= hgt + 0.04

    ax = fig.add_subplot(gs[2])
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    ax.set_title("River scale", fontsize=11.5, color=NAVY, fontweight="bold")
    ax.add_patch(FancyBboxPatch((0.02, 0.60), 0.96, 0.36, boxstyle="round,pad=0.005,rounding_size=0.02", fc=FILL["field"], ec=GREEN, lw=1.2))
    ax.text(0.5, 0.90, "Field microplastics carry", fontsize=10.5, ha="center", color="#1c2833")
    ax.text(0.5, 0.775, f"{n['ratio']:.2f}", fontsize=27, ha="center", va="center", color=GREEN, fontweight="bold")
    ax.text(0.5, 0.655, f"of the laboratory-predicted Cu (95% CI {n['ratio_lo']:.2f}–{n['ratio_hi']:.1f})", fontsize=9.5, ha="center", color="#1c2833")
    ax.add_patch(FancyBboxPatch((0.02, 0.10), 0.96, 0.42, boxstyle="round,pad=0.005,rounding_size=0.02", fc=FILL["data"], ec=NAVY, lw=1.2))
    ax.text(0.5, 0.455, "Microplastic-bound Cu exported by rivers", fontsize=10.5, ha="center", color="#1c2833")
    g_low, g_mid = (float(f"{n[k]:.2g}") for k in ("g_low", "g_mid"))        # two significant figures, as in the text
    ax.text(0.5, 0.33, f"{g_low:.0f}–{g_mid:.0f} kg yr$^{{-1}}$", fontsize=22, ha="center", va="center", color=NAVY, fontweight="bold")
    ax.text(0.5, 0.185, f"≤ {n['g_pct']:.3f}% of the dissolved Cu flux", fontsize=10.5, ha="center", color=RED, fontweight="bold")
    fig.text(0.34, 0.47, "→", fontsize=30, color=GREY, ha="center", va="center")
    fig.text(0.675, 0.47, "→", fontsize=30, color=GREY, ha="center", va="center")
    save(fig, "Graphical_abstract", dpi=300)


def main() -> None:
    plt.rcParams.update({"font.family": "Times New Roman", "mathtext.fontset": "custom", "mathtext.rm": "Times New Roman", "mathtext.it": "Times New Roman:italic", "mathtext.bf": "Times New Roman:bold", "mathtext.default": "regular"})
    n = numbers()
    fig1(n)
    fig_s2(n)
    graphical_abstract(n)


if __name__ == "__main__":
    main()
