"""Sensitivity of the main results to Turner and Holmes (2015), added on 17 Oct 2026 from the field-study search.

Turner_2015 is the only study whose Cu isotherms were measured in natural river water (filtered Plym water, DOC about
2 mg/L, pH 6.5; nine metals added together, coded AdC = Mixed). Its 10 points lie at 1.8-16 ug/L, i.e. at river
concentrations, and it is the study with the largest leave-one-study-out error. Three questions:

A - Field-to-laboratory ratio R (isotherm-form model, six studies with dissolved Cu) when the laboratory model is
    fitted (i) to all 961 points, (ii) without Turner_2015, (iii) with Turner_2015 coded as single-ion.
B - Low-concentration hold-out (train above 0.1 or 0.3 mg/L, test below) with and without Turner_2015.
C - Leave-one-study-out error of the isotherm-form model below and above 0.1 mg/L, with and without the Turner points.

Output: outputs/reports/turner_sensitivity.csv
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from config import GROUP_COL, REPORTS_DIR, TARGET  # noqa: E402
from src.analysis.faf_scaling import load_field_studies, study_level, study_ratios, summarise_faf  # noqa: E402
from src.analysis.grouped_validation import export_tuned_params, load_data  # noqa: E402
from src.analysis.transfer_analysis import fit_predict, log10  # noqa: E402

SID = "Turner_2015"
MODELS = ["freundlich_pooled", "xgb_blend", "xgb_monotone", "lgbm_monotone", "rf_raw"]


def main() -> None:
    df = load_data("all")
    rf = export_tuned_params()["RandomForest"]
    rows = []

    # A - field-to-laboratory ratio
    records = load_field_studies()
    single = df.copy()
    single.loc[single[GROUP_COL] == SID, "AdC"] = "Single-ion"
    assert set(df["AdC"]) == {"Single-ion", "Mixed"}, set(df["AdC"])
    for label, lab in [("all 961 points", df), ("without Turner_2015", df[df[GROUP_COL] != SID]),
                       ("Turner_2015 coded single-ion", single)]:
        r = study_ratios("freundlich_pooled", lab, records)
        r = r[r["study"] != "Purwiyanto_2020"]
        s = summarise_faf(study_level(r).to_numpy())
        rows.append({"check": "A field ratio, six studies", "variant": label, "n_points": len(lab),
                     "value": s["geometric_mean"], "ci95_low": s["gm_ci95_low"], "ci95_high": s["gm_ci95_high"]})
        print(rows[-1], flush=True)

    # B - low-concentration hold-out
    prev = pd.read_csv(REPORTS_DIR / "low_ce_holdout_final.csv")      # with Turner_2015: same computation (table1_model_performance.py)
    for r in prev.itertuples():
        rows.append({"check": f"B low-Ce hold-out {r.threshold_mg_L} mg/L", "variant": "with Turner_2015", "model": r.model,
                     "design": r.design, "n_points": r.n_test, "value": r.bias_as_factor, "median_fold_error": r.median_fold_error})
    for label, data in [("without Turner_2015", df[df[GROUP_COL] != SID])]:
        for threshold in (0.1, 0.3):
            low, high = data[data["Ce"] <= threshold], data[data["Ce"] > threshold]
            for m in MODELS:
                same = np.log10(fit_predict(m, high, low, rf)) - np.asarray(log10(low[TARGET]))
                new = np.concatenate([np.log10(fit_predict(m, high[high[GROUP_COL] != s], g, rf)) - np.asarray(log10(g[TARGET]))
                                      for s, g in low.groupby(GROUP_COL)])
                for design, e in (("same_lab", same), ("new_lab", new)):
                    rows.append({"check": f"B low-Ce hold-out {threshold} mg/L", "variant": label, "model": m, "design": design,
                                 "n_points": len(e), "n_studies": low[GROUP_COL].nunique(), "value": 10 ** float(np.mean(e)),
                                 "median_fold_error": 10 ** float(np.median(np.abs(e)))})
        print(f"B {label} done", flush=True)

    # C - leave-one-study-out error of the isotherm-form model by concentration band
    err = []
    for s, test in df.groupby(GROUP_COL):
        e = np.log10(fit_predict("freundlich_pooled", df[df[GROUP_COL] != s], test, rf)) - np.asarray(log10(test[TARGET]))
        err.append(pd.DataFrame({GROUP_COL: s, "Ce": test["Ce"].to_numpy(), "e": e}))
    err = pd.concat(err, ignore_index=True)
    for label, part in [("with Turner_2015", err), ("without Turner_2015", err[err[GROUP_COL] != SID])]:
        for band, mask in [("Ce < 0.1 mg/L", part["Ce"] < 0.1), ("Ce >= 0.1 mg/L", part["Ce"] >= 0.1), ("all", part["Ce"] > 0)]:
            g = part[mask]
            rows.append({"check": f"C leave-one-study-out, {band}", "variant": label, "model": "freundlich_pooled",
                         "n_points": len(g), "n_studies": g[GROUP_COL].nunique(),
                         "median_fold_error": 10 ** float(np.median(np.abs(g["e"]))), "value": 10 ** float(g["e"].mean())})
    out = pd.DataFrame(rows)
    out.to_csv(REPORTS_DIR / "turner_sensitivity.csv", index=False)
    pd.set_option("display.width", 220)
    print(out.round(3).to_string(index=False))


if __name__ == "__main__":
    main()
