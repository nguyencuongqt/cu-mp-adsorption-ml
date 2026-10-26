"""Compare the two completed check sheets of the knowledge-graph relations.

Outputs (same folder): KG_check_comparison.xlsx with the summary, the per-mechanism table and the rows on which the two
checkers disagree (to be resolved by discussion; fill in the column agreed_correct).
Usage: python scripts/compare_kg_checks.py <file of checker 1> <file of checker 2>
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DIR = ROOT / "Manuscript" / "Submission_Env pollution" / "Revision" / "04_LLM_evaluation" / "knowledge_graph_check"


def load(name: str) -> pd.DataFrame:
    d = pd.read_excel(DIR / name, sheet_name="Relations")
    p = next(c for c in d.columns if c.startswith("polymer_correct"))
    m = next(c for c in d.columns if c.startswith("mechanism_correct"))
    d["P"] = d[p].astype(str).str.strip().str.upper()
    d["M"] = d[m].astype(str).str.strip().str.upper()
    if not (d["P"].isin(["Y", "N"]).all() and d["M"].isin(["Y", "N"]).all()):
        raise SystemExit(f"{name}: blank or invalid answers")
    d["correct"] = (d["P"] == "Y") & (d["M"] == "Y")
    return d


def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    p = k / n
    centre, half = p + z * z / (2 * n), z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return (centre - half) / (1 + z * z / n), (centre + half) / (1 + z * z / n)


def kappa(a: pd.Series, b: pd.Series) -> float:
    po = (a == b).mean()
    pe = a.mean() * b.mean() + (1 - a.mean()) * (1 - b.mean())
    return (po - pe) / (1 - pe)


def main() -> None:
    f1, f2 = sys.argv[1], sys.argv[2]
    a, b = load(f1), load(f2)
    assert (a["item"] == b["item"]).all() and (a["sentence"] == b["sentence"]).all(), "the two files do not hold the same items"
    n = len(a)
    rows = []
    for label, d in [("checker 1", a), ("checker 2", b)]:
        k = int(d["correct"].sum()); lo, hi = wilson(k, n)
        rows.append({"quantity": f"precision, {label}", "value": f"{k}/{n} = {100 * k / n:.0f}% (95% CI {100 * lo:.0f}-{100 * hi:.0f}%)"})
    both, either = int((a["correct"] & b["correct"]).sum()), int((a["correct"] | b["correct"]).sum())
    rows += [{"quantity": "agreement on correct/incorrect", "value": f"{int((a['correct'] == b['correct']).sum())}/{n}; Cohen's kappa {kappa(a['correct'], b['correct']):.2f}"},
             {"quantity": "agreement on the polymer question", "value": f"{int((a['P'] == b['P']).sum())}/{n}"},
             {"quantity": "agreement on the mechanism question", "value": f"{int((a['M'] == b['M']).sum())}/{n}"},
             {"quantity": "correct for both / for at least one", "value": f"{both} / {either}"}]
    summary = pd.DataFrame(rows)
    by = pd.DataFrame({"n": a.groupby("mechanism").size(), "correct, checker 1": a.groupby("mechanism")["correct"].sum(),
                       "correct, checker 2": b.groupby("mechanism")["correct"].sum(),
                       "correct for both": (a["correct"] & b["correct"]).groupby(a["mechanism"]).sum()})
    dis = a.loc[a["correct"] != b["correct"], ["item", "paper", "polymer", "mechanism", "sentence"]].copy()
    for label, d in [("checker 1", a), ("checker 2", b)]:
        dis[f"polymer, {label}"], dis[f"mechanism, {label}"] = d.loc[dis.index, "P"], d.loc[dis.index, "M"]
        dis[f"comment, {label}"] = d.loc[dis.index, "comment"]
    dis["agreed_correct (Y/N)"] = ""
    with pd.ExcelWriter((DIR / f1).parent / "KG_check_comparison.xlsx", engine="openpyxl") as w:
        summary.to_excel(w, sheet_name="Summary", index=False); by.to_excel(w, sheet_name="By mechanism")
        dis.to_excel(w, sheet_name="Disagreements", index=False)
        w.sheets["Summary"].column_dimensions["A"].width = 42; w.sheets["Summary"].column_dimensions["B"].width = 50
        w.sheets["Disagreements"].column_dimensions["E"].width = 100
    print(summary.to_string(index=False)); print(by.to_string())
    print(dis.drop(columns=["sentence"]).to_string(index=False))


if __name__ == "__main__":
    main()
