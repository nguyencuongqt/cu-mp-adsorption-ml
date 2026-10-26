"""Analysis of the blinded expert scores (2 raters, 15 cases, 60 explanations) of the LLM evaluation.

Inputs:  data/expert_scores_by_item.csv (scores of raters LVT and PB for the 60 explanations, with case and condition),
         outputs/reports/llm_eval_v2_scores_final.csv (LLM judges, same rubric and weights).
         The CSV is written from the raters' scoring sheets and the blinding key (Revision/04_LLM_evaluation/) when those
         files are present; otherwise it is read as distributed in the repository.
Outputs: outputs/reports/expert_eval_summary.csv          mean scores by rater and condition
         outputs/reports/expert_eval_differences.csv      within-case differences between conditions (mean of raters)
         outputs/reports/expert_eval_agreement.csv        inter-rater agreement and expert-LLM judge agreement
Confidence intervals: 5,000 bootstrap resamples of cases (all items of a case kept together).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

ROOT = Path(__file__).resolve().parents[1]
EVAL = ROOT / "Manuscript" / "Submission_Env pollution" / "Revision" / "04_LLM_evaluation"
REP = ROOT / "outputs" / "reports"
SCORES_CSV = ROOT / "data" / "expert_scores_by_item.csv"
RATERS = {"LVT": EVAL / "expert scoring sheet 15cases_LVT.xlsx",
          "PB": EVAL / "To_send_expert_raters" / "expert_scoring_sheet_final_15cases_PB.xlsx"}
COLS = {"Accuracy": "accuracy", "Mechanistic depth": "mechanistic_depth", "Literature grounding": "literature_grounding",
        "Condition specificity": "condition_specificity", "Readability": "readability",
        "Limits communicated": "limits_communicated"}
DIMS = list(COLS.values())
WEIGHTS = {"accuracy": 0.25, "mechanistic_depth": 0.30, "literature_grounding": 0.20, "condition_specificity": 0.15,
           "readability": 0.10}
N_BOOT, SEED = 5000, 42


def overall(df: pd.DataFrame) -> pd.Series:
    return sum(df[k] * w for k, w in WEIGHTS.items())


def icc_a1(x: np.ndarray) -> float:
    """ICC(2,1), two-way random effects, absolute agreement, single rater (McGraw and Wong, 1996). x: items x raters."""
    n, k = x.shape
    gm = x.mean()
    msr = k * ((x.mean(axis=1) - gm) ** 2).sum() / (n - 1)
    msc = n * ((x.mean(axis=0) - gm) ** 2).sum() / (k - 1)
    sse = ((x - x.mean(axis=1, keepdims=True) - x.mean(axis=0, keepdims=True) + gm) ** 2).sum()
    mse = sse / ((n - 1) * (k - 1))
    return (msr - mse) / (msr + (k - 1) * mse + k * (msc - mse) / n)


def kripp_alpha_interval(x: np.ndarray) -> float:
    """Krippendorff's alpha, interval metric, complete data. x: items x raters."""
    vals = x.ravel()
    n_pairable = vals.size
    d_o = sum(((r[:, None] - r[None, :]) ** 2).sum() / (len(r) - 1) for r in x) / n_pairable
    d_e = ((vals[:, None] - vals[None, :]) ** 2).sum() / (n_pairable * (n_pairable - 1))
    return 1 - d_o / d_e if d_e > 0 else np.nan


def boot_cases(df: pd.DataFrame, stat, rng) -> tuple[float, float]:
    cases = df["case_id"].unique()
    groups = {c: g for c, g in df.groupby("case_id")}
    out = []
    for _ in range(N_BOOT):
        pick = rng.choice(cases, size=len(cases), replace=True)
        out.append(stat(pd.concat([groups[c] for c in pick], ignore_index=True)))
    return tuple(np.nanpercentile(out, [2.5, 97.5]))


def load_scores() -> pd.DataFrame:
    """Expert scores, one row per rater and explanation (120 rows)."""
    key_file = EVAL / "KEY_do_not_share_with_raters_final_15cases.csv"
    if key_file.exists() and all(p.exists() for p in RATERS.values()):
        key = pd.read_csv(key_file)[["item_id", "case_id", "condition"]]
        frames = []
        for rater, path in RATERS.items():
            d = pd.read_excel(path, sheet_name="score_here").rename(columns=COLS)[["item_id"] + DIMS]
            assert len(d) == 60 and d[DIMS].notna().all().all() and d[DIMS].isin([1, 2, 3, 4, 5]).all().all(), rater
            frames.append(d.assign(rater=rater))
        ex = pd.concat(frames).merge(key, on="item_id", how="left")
        ex = ex[["rater", "item_id", "case_id", "condition"] + DIMS].sort_values(["rater", "item_id"]).reset_index(drop=True)
        SCORES_CSV.parent.mkdir(parents=True, exist_ok=True)
        ex.to_csv(SCORES_CSV, index=False)
    else:
        ex = pd.read_csv(SCORES_CSV)
    assert len(ex) == 120 and ex["condition"].notna().all()
    return ex


def main() -> None:
    rng = np.random.default_rng(SEED)
    ex = load_scores()
    ex["overall"] = overall(ex)
    wide = ex.pivot_table(index=["item_id", "case_id", "condition"], columns="rater", values=DIMS + ["overall"]).reset_index()

    # --- mean scores by rater and condition, and for the mean of the two raters (CI over cases)
    mean2 = ex.groupby(["item_id", "case_id", "condition"])[DIMS + ["overall"]].mean().reset_index()
    rows = []
    for who, data in [("LVT", ex[ex.rater == "LVT"]), ("PB", ex[ex.rater == "PB"]), ("mean_of_raters", mean2)]:
        for cond, g in data.groupby("condition"):
            r = {"rater": who, "condition": cond, "n_items": len(g)}
            for v in DIMS + ["overall"]:
                r[v] = g[v].mean()
            if who == "mean_of_raters":
                r["overall_ci_low"], r["overall_ci_high"] = boot_cases(g, lambda s: s["overall"].mean(), rng)
            rows.append(r)
    summary = pd.DataFrame(rows)

    # --- within-case differences (mean of raters)
    diffs = []
    for a, b in [("C1", "C0"), ("C2", "C1"), ("C3", "C2"), ("C3", "C1")]:
        pa = mean2[mean2.condition == a].set_index("case_id")
        pb = mean2[mean2.condition == b].set_index("case_id")
        for v in ["overall", "mechanistic_depth", "literature_grounding", "limits_communicated"]:
            d = (pa[v] - pb[v]).rename("d").reset_index()
            lo, hi = boot_cases(d, lambda s: s["d"].mean(), rng)
            diffs.append({"comparison": f"{a}-{b}", "criterion": v, "mean_difference": d["d"].mean(), "ci_low": lo,
                          "ci_high": hi, "share_cases_higher": (d["d"] > 0).mean(), "n_cases": len(d)})
    diffs = pd.DataFrame(diffs)

    # --- agreement between the two experts, and between experts and LLM judges on the same 60 items
    agr = []
    for v in DIMS + ["overall"]:
        x = wide[[(v, "LVT"), (v, "PB")]].to_numpy(dtype=float)
        sub = pd.DataFrame({"case_id": wide["case_id"], "a": x[:, 0], "b": x[:, 1]})
        f_icc = lambda s: icc_a1(s[["a", "b"]].to_numpy())
        f_alpha = lambda s: kripp_alpha_interval(s[["a", "b"]].to_numpy())
        agr.append({"pair": "LVT vs PB", "criterion": v, "icc_a1": icc_a1(x), "icc_ci": boot_cases(sub, f_icc, rng),
                    "kripp_alpha_interval": kripp_alpha_interval(x), "alpha_ci": boot_cases(sub, f_alpha, rng),
                    "exact_agreement": float((x[:, 0] == x[:, 1]).mean()) if v != "overall" else np.nan,
                    "within_one_point": float((abs(x[:, 0] - x[:, 1]) <= 1).mean()) if v != "overall" else np.nan,
                    "spearman": spearmanr(x[:, 0], x[:, 1]).statistic})
    # agreement among the LLM-written explanations only (C2, C3): the overall ICC is dominated by the differences between conditions
    llm = wide[wide["condition"].isin(["C2", "C3"])]
    sub = pd.DataFrame({"case_id": llm["case_id"], "a": llm[("overall", "LVT")].to_numpy(float), "b": llm[("overall", "PB")].to_numpy(float)})
    agr.append({"pair": "LVT vs PB, C2 and C3 only", "criterion": "overall", "icc_a1": icc_a1(sub[["a", "b"]].to_numpy()),
                "icc_ci": boot_cases(sub, lambda s: icc_a1(s[["a", "b"]].to_numpy()), rng),
                "spearman": spearmanr(sub["a"], sub["b"]).statistic})
    judges = pd.read_csv(REP / "llm_eval_v2_scores_final.csv")
    for j in judges["judge"].unique():
        jj = judges[judges.judge == j].set_index("item_id")
        m = mean2.set_index("item_id").join(jj[DIMS + ["overall_weighted"]], rsuffix="_judge", how="inner")
        sub = pd.DataFrame({"case_id": m["case_id"], "a": m["overall"], "b": m["overall_weighted"]})
        agr.append({"pair": f"mean of experts vs {j}", "criterion": "overall",
                    "icc_a1": icc_a1(sub[["a", "b"]].to_numpy()),
                    "icc_ci": boot_cases(sub, lambda s: icc_a1(s[["a", "b"]].to_numpy()), rng),
                    "spearman": spearmanr(sub["a"], sub["b"]).statistic,
                    "mean_judge_minus_experts": float((sub["b"] - sub["a"]).mean()),
                    "judge_mean_by_condition_same_items": m.groupby("condition")["overall_weighted"].mean().round(2).to_dict()})
    agr = pd.DataFrame(agr)

    REP.mkdir(parents=True, exist_ok=True)
    summary.to_csv(REP / "expert_eval_summary.csv", index=False)
    diffs.to_csv(REP / "expert_eval_differences.csv", index=False)
    agr.to_csv(REP / "expert_eval_agreement.csv", index=False)
    pd.set_option("display.width", 250); pd.set_option("display.max_columns", 30)
    print(summary.round(2).to_string(index=False)); print(); print(diffs.round(2).to_string(index=False)); print()
    print(agr.round(2).to_string(index=False))


if __name__ == "__main__":
    main()
