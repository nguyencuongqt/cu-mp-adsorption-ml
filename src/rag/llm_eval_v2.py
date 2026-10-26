"""Re-evaluation of the LLM Interpretation System (revision of ENVPOL-D-26-10175).

Design, in response to the reviewers:
- N test cases (default 30) stratified by position of Ce relative to the laboratory range, polymer
  support and ageing status;
- four explanation conditions as in the submitted version: C0 numbers only, C1 rule-based,
  C2 retrieval-augmented, C3 retrieval plus knowledge graph;
- two LLM judges that see neither the condition label nor the other explanations of the case, in
  random order: the model that also generates the explanations (the design of the submitted
  version) and a model from a different provider; their difference measures self-preference;
- judge-independent faithfulness measures computed from the text: the predicted values are stated
  correctly, every applicable validity warning is conveyed, and cited sources exist in the corpus;
- a blinded scoring sheet for human experts.

Outputs (outputs/reports): llm_eval_v2_cases.csv, llm_eval_v2_explanations.csv, llm_eval_v2_scores.csv,
llm_eval_v2_faithfulness.csv, llm_eval_v2_summary.csv; expert sheets in Revision/LLM_evaluation/.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
MODULE_DIR = Path(__file__).resolve().parent
for path in (PROJECT_ROOT, MODULE_DIR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from config import MODELS_DIR, RANDOM_SEED, REPORTS_DIR  # noqa: E402
from src.analysis.predictor import model_summary  # noqa: E402
from llm_clients import GENERATOR, JUDGES, available, call_llm  # noqa: E402
from llm_interpretation_v2 import MECHANISM_TERMS, NOT_INPUTS_RE, explain, papers_mentioning, plain_text, unsupported_values  # noqa: E402

CONDITIONS = {"C0": 0, "C1": 1, "C2": 2, "C3": 3}
DIMENSIONS = ["accuracy", "mechanistic_depth", "literature_grounding", "condition_specificity", "readability", "limits_communicated"]
WEIGHTS = {"accuracy": 0.25, "mechanistic_depth": 0.30, "literature_grounding": 0.20, "condition_specificity": 0.15, "readability": 0.10}
EXPERT_DIR = PROJECT_ROOT / "Manuscript" / "Submission_Env pollution" / "Revision" / "04_LLM_evaluation"
RUBRIC = """Score each dimension from 1 to 5.

accuracy: 1 = predicted values absent or wrong; 3 = the predicted qe is stated correctly but the interval or the field-scale
  value is missing or wrong; 5 = predicted qe, prediction interval and field-scale value are all stated and agree with the facts.
mechanistic_depth: 1 = no mechanism; 3 = generic statement (e.g. "adsorption occurs"); 5 = specific named mechanisms
  (electrostatic attraction, surface complexation, ion exchange, ...) tied to this polymer and ageing status.
literature_grounding: 1 = no reference to sources; 3 = vague mention of "literature" or "the dataset"; 5 = at least two
  specific source ids cited with content relevant to the case.
condition_specificity: 1 = statements that would apply to any case; 3 = one input discussed specifically; 5 = concentration
  level, polymer, ageing status and adsorption system all addressed with reasoning specific to this case.
readability: 1 = raw numbers or incomprehensible; 3 = understandable but disorganised; 5 = clear, well structured and usable
  by an environmental scientist.
limits_communicated: 1 = no statement about uncertainty or validity; 3 = uncertainty mentioned but the applicable warnings
  are missing or wrong; 5 = the prediction interval and every warning listed in the facts are conveyed correctly, and no
  warning is raised that does not apply."""
JUDGE_SYSTEM = ("You are an expert reviewer of explanations of Cu(II) sorption predictions for microplastics. Score the explanation "
                "against the facts using the rubric. Identical quality must receive identical scores. Output only a JSON object.")


# ------------------------------------------------------------------ cases
def build_cases(n: int, seed: int = RANDOM_SEED) -> pd.DataFrame:
    """Stratified cases: river-level, low laboratory, in-range and above-range Ce; common and sparse polymers."""
    rng = np.random.default_rng(seed)
    s = model_summary()
    common = [p for p, k in s["studies_per_polymer"].items() if k >= 3]
    sparse = [p for p, k in s["studies_per_polymer"].items() if k < 3]
    strata = [("river_level", (3e-4, 3e-3), 0.27), ("low_laboratory", (5e-3, 0.09), 0.20),
              ("in_range", (0.15, 60.0), 0.46), ("above_range", (250.0, 400.0), 0.07)]
    counts = [max(1, round(n * share)) for _, _, share in strata]
    counts[2] += n - sum(counts)
    rows = []
    for (name, (lo, hi), _), k in zip(strata, counts):
        for i in range(k):
            polymer = rng.choice(sparse) if (i % 5 == 4 and sparse) else common[(len(rows)) % len(common)]
            rows.append({"stratum": name, "ReT": polymer, "AgS": ["Aged", "Virgin"][len(rows) % 2],
                         "AdC": "Mixed" if name == "river_level" or len(rows) % 4 == 3 else "Single-ion",
                         "Ce": float(f"{10 ** rng.uniform(np.log10(lo), np.log10(hi)):.3g}")})
    cases = pd.DataFrame(rows[:n])
    cases.insert(0, "case_id", [f"case_{i + 1:02d}" for i in range(len(cases))])
    return cases


# ------------------------------------------------------------------ faithfulness
NUMBER = re.compile(r"(?<![\w.])(\d+(?:\.\d+)?(?:[eE][-+]?\d+)?)(?:\s*[x×]\s*10\s*\^?\s*\(?(-?\d+)\)?)?")


def numbers_in(text: str) -> list[float]:
    out = []
    for mant, exp in NUMBER.findall(text.replace("−", "-").replace(",", "")):
        try:
            out.append(float(mant) * (10 ** int(exp) if exp else 1.0))
        except ValueError:
            pass
    return out


def stated(values: list[float], target: float, tol: float = 0.06) -> bool:
    """True if the text gives the target in mg/g or in ug/g."""
    return any(abs(v - t) <= tol * abs(t) for v in values for t in (target, target * 1e3))


def faithfulness(text: str, pred: dict[str, Any], corpus_ids: set[str]) -> dict[str, Any]:
    plain = plain_text(text)
    low = plain.lower()
    nums = numbers_in(plain)
    flags = pred["flags"]
    needs_range = flags["below_dense_lab_range"] or flags["outside_lab_range"]
    says_range = bool(re.search(r"extrapolat|outside (of )?the (laboratory|lab|training|experimental|studied)|below the (laboratory|lab|range)|"
                                r"beyond the (laboratory|lab)|reli(es|ance|ant|ying) (heavily |largely |strongly )?on the isotherm|"
                                r"not covered by the laboratory|only \d+ ?% of (the )?laboratory|below (the )?\d*\.?\d+ mg/l", low))
    says_sparse = bool(re.search(r"represented by (only )?(one|two|1|2|a single)|only (one|two|1|2|a single) (laboratory )?stud|"
                                 r"(one|two|1|2|single) (laboratory )?stud(y|ies)", low))
    cited = set(re.findall(r"\b[A-Z][A-Za-z]+(?:_[A-Z][a-z]+)?_(?:19|20)\d\d[a-c]?\b", plain))
    valid = {c for c in cited if c in corpus_ids}
    return {
        "qe_stated": stated(nums, pred["lab_qe"]),
        "interval_stated": (stated(nums, pred["lab_qe_lower"]) and stated(nums, pred["lab_qe_upper"]))
                           or any(abs(v - pred["interval_factor"]) <= 0.06 * pred["interval_factor"] for v in nums),
        "field_value_stated": stated(nums, pred["river_qe"]),
        "range_warning_needed": needs_range, "range_warning_given": says_range,
        "sparse_warning_needed": flags["sparse_polymer"], "sparse_warning_given": says_sparse,
        "n_unsupported_qe_values": len(unsupported_values(plain, pred)),
        "n_citations": len(cited), "n_citations_in_corpus": len(valid), "citations_not_in_corpus": "; ".join(sorted(cited - valid)),
        "n_citations_naming_polymer": len(valid & papers_mentioning(pred["inputs"]["ReT"])),
        "n_mechanisms_named": sum(bool(re.search(pat, plain, re.I)) for pat in MECHANISM_TERMS.values()),
        "uses_non_input_variable": bool(NOT_INPUTS_RE.search(plain)),
        "n_words": len(plain.split()),
    }


# ------------------------------------------------------------------ judging
def judge(label: str, facts: str, explanation: str) -> dict[str, Any]:
    provider, model = JUDGES[label]
    user = (f"=== Facts (ground truth for this case) ===\n{facts}\n\n=== Explanation to evaluate ===\n{plain_text(explanation)[:3500]}\n\n"
            f"=== Rubric ===\n{RUBRIC}\n\nOutput a JSON object with exactly these keys: "
            + ", ".join(f'"{d}"' for d in DIMENSIONS) + ', "rationale" (at most 40 words).')
    raw = call_llm(provider, model, JUDGE_SYSTEM, user, temperature=0.0, max_tokens=700)
    match = re.search(r"\{.*\}", raw, re.DOTALL)
    try:
        parsed = json.loads(match.group()) if match else {}
    except json.JSONDecodeError:
        parsed = {}
    if not all(d in parsed for d in DIMENSIONS):       # truncated or malformed JSON: read the scores one by one
        for d in DIMENSIONS:
            m = re.search(rf'"{d}"\s*:\s*([1-5])', raw)
            if m:
                parsed[d] = int(m.group(1))
        m = re.search(r'"rationale"\s*:\s*"([^"]*)', raw)
        parsed.setdefault("rationale", m.group(1) if m else "")
    out = {d: (int(parsed[d]) if str(parsed.get(d, "")).strip().isdigit() else np.nan) for d in DIMENSIONS}
    out["rationale"] = str(parsed.get("rationale", ""))[:400]
    out["parse_ok"] = all(d in parsed for d in DIMENSIONS)
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--n-cases", type=int, default=30)
    parser.add_argument("--tag", default="", help="Suffix for output files (e.g. _pilot).")
    parser.add_argument("--no-judges", action="store_true")
    parser.add_argument("--cases-file", default="",
                        help="Reuse the cases of an earlier run (CSV with case_id, stratum, ReT, AgS, AdC, Ce) instead of drawing new ones.")
    parser.add_argument("--seed-offset", type=int, default=0,
                        help="Added to the random seed. Development runs use a non-zero offset; the final run uses 0.")
    args = parser.parse_args()
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    corpus = pd.read_csv(MODELS_DIR / "corpus_v2_papers.csv")
    corpus_ids = set(corpus["key"])
    rng = np.random.default_rng(RANDOM_SEED + args.seed_offset)

    if args.cases_file:      # rerun of 16 October 2026: same 30 cases, model refitted on the final dataset (951 observations)
        cases = pd.read_csv(args.cases_file)[["case_id", "stratum", "ReT", "AgS", "AdC", "Ce"]]
    else:
        cases = build_cases(args.n_cases, RANDOM_SEED + args.seed_offset)
    cases.to_csv(REPORTS_DIR / f"llm_eval_v2_cases{args.tag}.csv", index=False)
    items = []
    for case in cases.itertuples():
        values = {"ReT": case.ReT, "AgS": case.AgS, "AdC": case.AdC, "Ce": case.Ce}
        for cond, mode in CONDITIONS.items():
            res = explain(values, mode=mode)
            items.append({"case_id": case.case_id, "stratum": case.stratum, "condition": cond, "facts": res["facts"],
                          "explanation": res["explanation_html"],
                          "sources": "; ".join(sorted({str(s.get("paper_id")) for s in res["sources"]})),
                          "regenerated_after_check": res["generation"].get("attempts", 1) > 1,
                          **faithfulness(res["explanation_html"], res["prediction"], corpus_ids)})
            if mode >= 2:
                time.sleep(0.5)
        print(f"generated {case.case_id}", flush=True)
    items = pd.DataFrame(items)
    items.insert(0, "item_id", [f"item_{i:03d}" for i in rng.permutation(len(items)) + 1])   # blinded, random order
    items.to_csv(REPORTS_DIR / f"llm_eval_v2_explanations{args.tag}.csv", index=False)
    faith_cols = ["item_id", "case_id", "stratum", "condition", "qe_stated", "interval_stated", "field_value_stated",
                  "range_warning_needed", "range_warning_given", "sparse_warning_needed", "sparse_warning_given",
                  "n_unsupported_qe_values", "regenerated_after_check", "n_citations", "n_citations_in_corpus", "citations_not_in_corpus",
                  "n_citations_naming_polymer", "n_mechanisms_named", "uses_non_input_variable", "n_words"]
    items[faith_cols].to_csv(REPORTS_DIR / f"llm_eval_v2_faithfulness{args.tag}.csv", index=False)

    scores = []
    if not args.no_judges:
        for label, (provider, _) in JUDGES.items():
            if not available(provider):
                print(f"judge {label}: no API key for {provider}; skipped")
                continue
            for row in items.sort_values("item_id").itertuples():       # judged in the blinded order
                scores.append({"judge": label, "item_id": row.item_id, "case_id": row.case_id, "condition": row.condition,
                               **judge(label, row.facts, row.explanation)})
                time.sleep(0.3)
            print(f"judge {label}: done", flush=True)
    scores = pd.DataFrame(scores)
    if len(scores):
        scores["overall_weighted"] = sum(scores[d] * w for d, w in WEIGHTS.items())
        scores.to_csv(REPORTS_DIR / f"llm_eval_v2_scores{args.tag}.csv", index=False)

    # blinded sheet for human experts, with the key kept in a separate file
    EXPERT_DIR.mkdir(parents=True, exist_ok=True)
    sheet = items.sort_values("item_id")[["item_id", "facts", "explanation"]].copy()
    sheet["explanation"] = sheet["explanation"].map(plain_text)
    for d in DIMENSIONS:
        sheet[d] = ""
    sheet["comment"] = ""
    with pd.ExcelWriter(EXPERT_DIR / f"expert_scoring_sheet{args.tag}.xlsx") as writer:
        sheet.to_excel(writer, sheet_name="score_here", index=False)
        pd.DataFrame({"rubric": RUBRIC.split("\n")}).to_excel(writer, sheet_name="rubric", index=False)
    items[["item_id", "case_id", "condition"]].to_csv(EXPERT_DIR / f"KEY_do_not_share_with_raters{args.tag}.csv", index=False)

    # summary
    pd.set_option("display.width", 220)
    f = items.copy()
    summary = f.groupby("condition").agg(
        n=("item_id", "size"), qe_stated=("qe_stated", "mean"), interval_stated=("interval_stated", "mean"),
        field_value_stated=("field_value_stated", "mean"), unsupported_values_per_item=("n_unsupported_qe_values", "mean"), regenerated=("regenerated_after_check", "sum"),
        citations_per_item=("n_citations", "mean"),
        citations_in_corpus=("n_citations_in_corpus", "sum"), citations_total=("n_citations", "sum"),
        citations_naming_polymer=("n_citations_naming_polymer", "sum"), two_or_more_sources=("n_citations_in_corpus", lambda x: (x >= 2).mean()),
        mechanisms_named=("n_mechanisms_named", "mean"), uses_non_input_variable=("uses_non_input_variable", "mean"),
        words=("n_words", "mean"))
    need = f[f["range_warning_needed"]].groupby("condition")["range_warning_given"].mean().rename("range_warning_recall")
    false = f[~f["range_warning_needed"]].groupby("condition")["range_warning_given"].mean().rename("range_warning_false_alarm")
    sparse = f[f["sparse_warning_needed"]].groupby("condition")["sparse_warning_given"].mean().rename("sparse_warning_recall")
    summary = summary.join([need, false, sparse])
    summary.to_csv(REPORTS_DIR / f"llm_eval_v2_summary{args.tag}.csv")
    print(f"\nGenerator: {GENERATOR[0]}/{GENERATOR[1]}; cases: {len(cases)}; items: {len(items)}")
    print("\nFaithfulness by condition (shares)")
    print(summary.round(2).to_string())
    if len(scores):
        print("\nJudge scores by condition (mean)")
        print(scores.groupby(["judge", "condition"])[DIMENSIONS + ["overall_weighted"]].mean().round(2).to_string())
        print("parse failures:", int((~scores["parse_ok"]).sum()))


if __name__ == "__main__":
    main()
