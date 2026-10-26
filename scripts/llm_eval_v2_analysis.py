"""Summary statistics for the LLM evaluation (src/rag/llm_eval_v2.py).

Paired differences between conditions (bootstrap over cases), the difference between the two judges
(self-preference of the judge that is the same model as the generator), agreement between judges and
checks that the final cases do not overlap with the development cases.

Usage: python scripts/llm_eval_v2_analysis.py --tag _final
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
from config import RANDOM_SEED, REPORTS_DIR  # noqa: E402

DIMENSIONS = ["accuracy", "mechanistic_depth", "literature_grounding", "condition_specificity", "readability", "limits_communicated"]
N_BOOT = 5000


def boot_ci(x: np.ndarray, rng: np.random.Generator) -> tuple[float, float, float]:
    means = x[rng.integers(0, len(x), size=(N_BOOT, len(x)))].mean(axis=1)
    return float(x.mean()), float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tag", default="_final")
    args = parser.parse_args()
    rng = np.random.default_rng(RANDOM_SEED)
    scores = pd.read_csv(REPORTS_DIR / f"llm_eval_v2_scores{args.tag}.csv")
    cases = pd.read_csv(REPORTS_DIR / f"llm_eval_v2_cases{args.tag}.csv")
    scores = scores.merge(cases[["case_id", "stratum"]], on="case_id")
    pd.set_option("display.width", 220)

    key = ["ReT", "AgS", "AdC", "Ce"]
    used = pd.concat([pd.read_csv(f)[key] for f in REPORTS_DIR.glob("llm_eval_v2_cases_*.csv") if f.stem != f"llm_eval_v2_cases{args.tag}"])
    overlap = cases[key].merge(used.drop_duplicates(), on=key)
    print(f"cases: {len(cases)}; cases also used during development: {len(overlap)}")
    print(cases.groupby("stratum").size().to_string(), "\n")

    rows = []
    for judge, g in scores.groupby("judge"):
        wide = g.pivot(index="case_id", columns="condition", values="overall_weighted")
        for cond in wide.columns:
            m, lo, hi = boot_ci(wide[cond].to_numpy(), rng)
            rows.append({"judge": judge, "quantity": f"overall {cond}", "mean": m, "ci_low": lo, "ci_high": hi})
        for a, b in [("C1", "C0"), ("C2", "C1"), ("C3", "C2"), ("C3", "C1")]:
            m, lo, hi = boot_ci((wide[a] - wide[b]).to_numpy(), rng)
            rows.append({"judge": judge, "quantity": f"overall {a} - {b}", "mean": m, "ci_low": lo, "ci_high": hi,
                         "share_cases_higher": float((wide[a] > wide[b]).mean())})
    table = pd.DataFrame(rows)
    table.to_csv(REPORTS_DIR / f"llm_eval_v2_condition_differences{args.tag}.csv", index=False)
    print(table.round(2).to_string(index=False), "\n")

    both = scores.pivot_table(index=["case_id", "condition"], columns="judge", values=DIMENSIONS + ["overall_weighted"])
    rows = []
    for dim in DIMENSIONS + ["overall_weighted"]:
        same, ind = both[(dim, "same_model_as_generator")], both[(dim, "independent_provider")]
        row = {"dimension": dim, "spearman": spearmanr(same, ind).statistic, "exact_agreement": float((same == ind).mean()) if dim != "overall_weighted" else np.nan}
        for cond in ["C0", "C1", "C2", "C3"]:
            diff = (same - ind).xs(cond, level="condition").to_numpy()
            row[f"same_minus_independent_{cond}"] = diff.mean()
        d = (same - ind).unstack("condition")
        m, lo, hi = boot_ci(((d["C2"] + d["C3"]) / 2 - (d["C0"] + d["C1"]) / 2).to_numpy(), rng)
        row.update(extra_for_llm_text=m, ci_low=lo, ci_high=hi)
        rows.append(row)
    bias = pd.DataFrame(rows)
    bias.to_csv(REPORTS_DIR / f"llm_eval_v2_judge_comparison{args.tag}.csv", index=False)
    print("Judge that is the generator model minus independent judge (extra_for_llm_text = mean difference for C2 and C3 minus")
    print("mean difference for C0 and C1, i.e. how much more the generator model favours its own text)")
    print(bias.round(2).to_string(index=False), "\n")

    by = scores.groupby(["judge", "stratum", "condition"])["overall_weighted"].mean().unstack("condition")
    by.to_csv(REPORTS_DIR / f"llm_eval_v2_by_stratum{args.tag}.csv")
    print(by.round(2).to_string())


if __name__ == "__main__":
    main()
