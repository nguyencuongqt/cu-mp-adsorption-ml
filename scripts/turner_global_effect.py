"""Effect of Turner & Holmes (2015) on the global field-scale qe (Section 3.2.2; response R1-6).

The isotherm-form model (ordinary least squares of log10 qe on log10 Ce and one-hot polymer, aging status and
adsorption system, as LinearRegression in src/analysis/transfer_analysis.py) is fitted with and without the ten
observations of Turner & Holmes (2015). It predicts qe for aged PE and PP in equal proportions in a mixed-solute
system at the two dissolved Cu concentrations of the global estimate, and the field-scale value is qe x R, with R
taken from outputs/reports/turner_sensitivity.csv. The global estimate is proportional to the field-scale value, so its
relative change is the same.

Check: with all 961 observations the laboratory qe must reproduce Table S14 (3.58 and 9.71 ug/g).

Input : data/dataset_grouped.csv, outputs/reports/turner_sensitivity.csv
Output: outputs/reports/turner_global_effect.csv
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "dataset_grouped.csv"
RATIOS = ROOT / "outputs" / "reports" / "turner_sensitivity.csv"
OUT = ROOT / "outputs" / "reports" / "turner_global_effect.csv"
TURNER = "Turner_2015"
CE_UG_L = (1.48, 8.94)                 # world-average and median compiled dissolved Cu (Text S2)
POLYMERS = ("PE", "PP")                # equal proportions, as in the global estimate (Section 2.5)
TABLE_S14 = {1.48: 3.58, 8.94: 9.71}   # laboratory qe (ug/g) with all observations


def fit(df: pd.DataFrame) -> tuple[pd.Index, np.ndarray]:
    x = pd.get_dummies(df[["ReT", "AgS", "AdC"]].astype(str)).astype(float)
    x.insert(0, "logCe", np.log10(df["Ce"]))
    x.insert(0, "const", 1.0)
    beta, *_ = np.linalg.lstsq(x.to_numpy(), np.log10(df["qe"].to_numpy()), rcond=None)
    return x.columns, beta


def predict_ug_g(cols: pd.Index, beta: np.ndarray, ce_ug_l: float) -> float:
    """Mean qe (ug/g) of aged PE and PP in a mixed-solute system."""
    values = []
    for polymer in POLYMERS:
        x = pd.Series(0.0, index=cols)
        x["const"], x["logCe"] = 1.0, np.log10(ce_ug_l / 1000)
        for key in (f"ReT_{polymer}", "AgS_Aged", "AdC_Mixed"):
            x[key] = 1.0
        values.append(10 ** float(x.to_numpy() @ beta) * 1000)
    return float(np.mean(values))


def main() -> None:
    data = pd.read_csv(DATA)
    ratios = pd.read_csv(RATIOS)
    field = ratios[ratios["check"] == "A field ratio, six studies"].set_index("variant")["value"]
    r = {"with": float(field["all 961 points"]), "without": float(field["without Turner_2015"])}
    rows = []
    for variant, df in (("with", data), ("without", data[data["source_id"] != TURNER])):
        cols, beta = fit(df)
        for ce in CE_UG_L:
            qe = predict_ug_g(cols, beta, ce)
            rows.append({"variant": f"{variant} Turner_2015", "n_observations": len(df),
                         "freundlich_exponent": beta[list(cols).index("logCe")], "Ce_ug_L": ce,
                         "qe_lab_ug_g": qe, "field_to_lab_ratio": r[variant], "qe_field_scale_ug_g": qe * r[variant]})
    out = pd.DataFrame(rows)
    for ce, expected in TABLE_S14.items():
        got = out[(out["variant"] == "with Turner_2015") & (out["Ce_ug_L"] == ce)]["qe_lab_ug_g"].iloc[0]
        if abs(got - expected) > 0.01:
            raise SystemExit(f"laboratory qe {got:.3f} at {ce} ug/L does not reproduce Table S14 ({expected})")
    base = out[out["variant"] == "with Turner_2015"].set_index("Ce_ug_L")
    alt = out[out["variant"] == "without Turner_2015"].set_index("Ce_ug_L")
    out["lab_qe_factor_without_vs_with"] = out["Ce_ug_L"].map(alt["qe_lab_ug_g"] / base["qe_lab_ug_g"])
    out["field_scale_change_pct"] = out["Ce_ug_L"].map(100 * (alt["qe_field_scale_ug_g"] / base["qe_field_scale_ug_g"] - 1))
    out.to_csv(OUT, index=False)
    print(out.round(3).to_string(index=False))


if __name__ == "__main__":
    main()
