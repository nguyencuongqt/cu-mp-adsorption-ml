"""Prediction backend of the revised model for the application and the LLM interpretation layer.

One call returns everything an explanation has to convey:

- laboratory-equilibrium qe from the selected model (pooled Freundlich: linear model of log10 qe on
  log10 Ce, polymer, ageing status and adsorption system);
- the study-level 90% prediction interval (prediction x/÷ factor) for a new laboratory, and the
  narrower interval that applies when one local measurement is available;
- validity flags: Ce below the range where laboratory data are dense (extrapolation along the
  isotherm), Ce outside the laboratory range, polymer represented by fewer than three studies;
- the contribution of each predictor to the prediction (multiplicative factors relative to the
  average laboratory observation);
- the field-scale value: laboratory qe times the field-to-laboratory ratio, with its confidence interval.

Numbers that come from the validation (interval factors, field ratio) are read from outputs/reports,
so the application always reports the validated values.
"""

from __future__ import annotations

import sys
from functools import lru_cache
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from config import DATA_GROUPED_FILE, GROUP_COL, REPORTS_DIR, TARGET  # noqa: E402

MODEL_NAME = "freundlich_pooled"
CATS = ["ReT", "AgS", "AdC"]
MIN_STUDIES = 3
DENSE_PERCENTILE = 5          # laboratory data are sparse below this percentile of Ce
FIELD_SUBSET = "without_Purwiyanto_2020"
ALIASES = {"AgS": {"aged": "Aged", "virgin": "Virgin", "pristine": "Virgin"},
           "AdC": {"single-ion": "Single-ion", "single": "Single-ion", "mixed": "Mixed", "multi-solute": "Mixed"}}


def load_data() -> pd.DataFrame:
    """The 961 laboratory observations (same file as src/analysis/grouped_validation.load_data("all"), read here
    without the machine-learning libraries so that the application has few dependencies)."""
    return pd.read_csv(DATA_GROUPED_FILE)


def sig(x: float, n: int = 2) -> str:
    """x with n significant figures, without scientific notation (convention of the manuscript, Section 2.4)."""
    x = float(x)
    if x == 0 or not np.isfinite(x):
        return f"{x:g}"
    d = n - 1 - int(np.floor(np.log10(abs(x))))
    r = round(x, d)
    if r != 0 and int(np.floor(np.log10(abs(r)))) > int(np.floor(np.log10(abs(x)))):   # 0.0996 -> 0.10
        d -= 1
        r = round(x, d)
    return f"{r:.{max(d, 0)}f}"


def qe_unit(reference_mg_g: float) -> tuple[float, str]:
    """Scale and unit for a qe value: ug/g below 0.1 mg/g, otherwise mg/g."""
    return (1000.0, "\u00b5g/g") if reference_mg_g < 0.1 else (1.0, "mg/g")


def fmt_qe(value_mg_g: float, reference_mg_g: float | None = None) -> str:
    """qe with two significant figures and its unit; the unit follows ``reference_mg_g`` (e.g. the central value)."""
    scale, unit = qe_unit(value_mg_g if reference_mg_g is None else reference_mg_g)
    return f"{sig(value_mg_g * scale)} {unit}"


def fmt_range(low: float, high: float, reference_mg_g: float) -> str:
    scale, unit = qe_unit(reference_mg_g)
    return f"{sig(low * scale)}\u2013{sig(high * scale)} {unit}"


@lru_cache(maxsize=1)
def _state() -> dict[str, Any]:
    df = load_data()
    x = pd.get_dummies(df[CATS], dtype=float)
    x.insert(0, "logCe", np.log10(df["Ce"].to_numpy()))
    y = np.log10(df[TARGET].to_numpy())
    model = LinearRegression().fit(x, y)
    intervals = pd.read_csv(REPORTS_DIR / "final_prediction_intervals.csv")
    intervals = intervals[(intervals["model"] == MODEL_NAME) & np.isclose(intervals["nominal"], 0.9)].set_index("case")
    faf = pd.read_csv(REPORTS_DIR / "faf_scaling_summary.csv")
    faf = faf[(faf["lab_model"] == MODEL_NAME) & (faf["field_subset"] == FIELD_SUBSET)].iloc[0]
    return {
        "model": model, "columns": list(x.columns), "x_mean": x.mean(), "y_mean": float(y.mean()),
        "levels": {c: sorted(df[c].unique()) for c in CATS},
        "studies_per_polymer": df.groupby("ReT")[GROUP_COL].nunique().to_dict(),
        "ce_min": float(df["Ce"].min()), "ce_max": float(df["Ce"].max()),
        "ce_dense": float(np.percentile(df["Ce"], DENSE_PERCENTILE)),
        "n_points": len(df), "n_studies": int(df[GROUP_COL].nunique()),
        "factor_new_lab": float(intervals.loc["new_laboratory", "interval_factor_final"]),
        "factor_anchored": float(intervals.loc["one_anchor_per_isotherm", "interval_factor_final"]),
        "field_ratio": float(faf["geometric_mean"]), "field_ratio_low": float(faf["gm_ci95_low"]),
        "field_ratio_high": float(faf["gm_ci95_high"]), "n_field_studies": int(faf["n_field_studies"]),
        "field_ratio_sd_log10": float(faf["between_study_sd_log10"]),
        "exponent": float(model.coef_[0]),
    }


def model_summary() -> dict[str, Any]:
    """Facts about the model that do not depend on the input (for prompts and the user interface)."""
    s = _state()
    return {k: s[k] for k in ["n_points", "n_studies", "ce_min", "ce_max", "ce_dense", "factor_new_lab", "factor_anchored",
                              "field_ratio", "field_ratio_low", "field_ratio_high", "n_field_studies", "exponent",
                              "field_ratio_sd_log10",
                              "studies_per_polymer", "levels"]}


def canonical_input(values: dict[str, Any]) -> dict[str, Any]:
    """Validate and normalise the four model inputs; other keys (pH, Temp, rpm) are ignored."""
    s = _state()
    out: dict[str, Any] = {}
    for col in CATS:
        if col not in values:
            raise ValueError(f"Missing input: {col}")
        raw = str(values[col]).strip()
        value = ALIASES.get(col, {}).get(raw.lower(), raw)
        lookup = {str(v).upper(): v for v in s["levels"][col]}
        if value.upper() not in lookup:
            raise ValueError(f"{col}={raw} is not in the laboratory dataset; choose from {s['levels'][col]}.")
        out[col] = lookup[value.upper()]
    ce = float(values.get("Ce", np.nan))
    if not np.isfinite(ce) or ce <= 0:
        raise ValueError("Ce must be a positive concentration in mg/L.")
    out["Ce"] = ce
    return out


def _design(values: dict[str, Any]) -> pd.Series:
    s = _state()
    row = pd.Series(0.0, index=s["columns"])
    row["logCe"] = np.log10(values["Ce"])
    for col in CATS:
        row[f"{col}_{values[col]}"] = 1.0
    return row


def field_interval_factor(lab_factor: float, level: float = 0.90) -> float:
    """Multiplicative half-width of the interval of the field-scale value for a river not among the field studies.

    Two independent errors are combined on the log10 scale: the error of the laboratory prediction (``lab_factor``,
    conformal 90% interval for a new laboratory or with one local measurement) and the variation of the
    field-to-laboratory ratio between rivers (prediction interval for a new site, t * s * sqrt(1 + 1/n); Section 2.4).
    The interval of the mean ratio (95% CI) alone would understate the uncertainty for a single river.
    """
    from scipy.stats import t

    s = _state()
    n = s["n_field_studies"]
    half_ratio = t.ppf(0.5 + level / 2, n - 1) * s["field_ratio_sd_log10"] * np.sqrt(1 + 1 / n)
    return float(10 ** np.hypot(np.log10(lab_factor), half_ratio))


def predict(values: dict[str, Any], anchor: tuple[float, float] | None = None) -> dict[str, Any]:
    """Prediction with interval, validity flags, drivers and field-scale value.

    ``anchor`` = (Ce, qe) of one local measurement for the same material: the prediction is shifted on the
    log scale to pass through it and the narrower, anchored interval applies.
    """
    s = _state()
    v = canonical_input(values)
    row = _design(v)
    log_qe = float(s["model"].predict(pd.DataFrame([row]))[0])
    factor, anchored = s["factor_new_lab"], False
    if anchor is not None:
        ce_a, qe_a = float(anchor[0]), float(anchor[1])
        if not (np.isfinite(ce_a) and np.isfinite(qe_a) and ce_a > 0 and qe_a > 0):
            raise ValueError("The local measurement needs a positive Ce (mg/L) and a positive qe (mg/g).")
        log_at_anchor = float(s["model"].predict(pd.DataFrame([_design({**v, "Ce": ce_a})]))[0])
        log_qe += np.log10(qe_a) - log_at_anchor
        factor, anchored = s["factor_anchored"], True
    qe = 10 ** log_qe
    river_factor = field_interval_factor(factor)

    contrib = pd.Series(s["model"].coef_, index=s["columns"]) * (row - s["x_mean"])
    drivers = {"Ce": float(contrib["logCe"])}
    for col in CATS:
        drivers[col] = float(contrib[[c for c in s["columns"] if c.startswith(col + "_")]].sum())
    ranked = sorted(drivers.items(), key=lambda kv: abs(kv[1]), reverse=True)

    n_pol = int(s["studies_per_polymer"].get(v["ReT"], 0))
    flags = {
        "below_dense_lab_range": bool(v["Ce"] < s["ce_dense"]),
        "outside_lab_range": bool(v["Ce"] < s["ce_min"] or v["Ce"] > s["ce_max"]),
        "sparse_polymer": bool(n_pol < MIN_STUDIES),
    }
    warnings = []
    if flags["outside_lab_range"]:
        warnings.append(f"Ce = {v['Ce']:.4g} mg/L is outside the laboratory range ({s['ce_min']:.2g}-{s['ce_max']:.3g} mg/L); "
                        "the value is an extrapolation along the fitted isotherm.")
    elif flags["below_dense_lab_range"]:
        warnings.append(f"Ce = {v['Ce']:.4g} mg/L is below {s['ce_dense']:.2g} mg/L, where only {DENSE_PERCENTILE}% of the laboratory "
                        "observations lie; the value relies on the isotherm form.")
    if anchored and abs(np.log10(float(anchor[0]) / v["Ce"])) > 2:
        warnings.append(f"The local measurement (Ce = {float(anchor[0]):.4g} mg/L) is more than 100-fold away from the requested Ce; "
                        "the shift relies on the pooled Freundlich exponent over this distance.")
    if flags["sparse_polymer"]:
        warnings.append(f"{v['ReT']} is represented by {n_pol} laboratory stud{'y' if n_pol == 1 else 'ies'}; "
                        "its effect cannot be separated from the laboratory that measured it.")
    return {
        "inputs": v, "model": MODEL_NAME, "anchored": anchored,
        "lab_qe": qe, "lab_qe_lower": qe / factor, "lab_qe_upper": qe * factor, "interval_factor": factor, "interval_level": 0.90,
        "interval_factor_with_one_local_measurement": s["factor_anchored"],
        "river_qe": qe * s["field_ratio"], "river_qe_lower": qe * s["field_ratio"] / river_factor,
        "river_qe_upper": qe * s["field_ratio"] * river_factor, "river_interval_factor": river_factor,
        "field_ratio": s["field_ratio"], "field_ratio_ci": (s["field_ratio_low"], s["field_ratio_high"]),
        "n_field_studies": s["n_field_studies"],
        "drivers": [{"predictor": k, "log10_effect": e, "factor_on_qe": 10 ** e,
                     "direction": "increases" if e > 0 else "decreases"} for k, e in ranked],
        "freundlich_exponent": s["exponent"], "polymer_n_studies": n_pol, "flags": flags, "warnings": warnings,
    }


if __name__ == "__main__":
    pd.set_option("display.width", 200)
    print({k: (round(v, 4) if isinstance(v, float) else v) for k, v in model_summary().items() if k != "levels"})
    for case in [{"ReT": "PP", "AgS": "aged", "AdC": "mixed", "Ce": 0.00148}, {"ReT": "PS", "AgS": "Virgin", "AdC": "Single-ion", "Ce": 2.0},
                 {"ReT": "TPU", "AgS": "Aged", "AdC": "Mixed", "Ce": 300.0}]:
        r = predict(case)
        print(f"\n{case}\n  lab qe {r['lab_qe']:.4g} mg/g (90% interval {r['lab_qe_lower']:.3g}-{r['lab_qe_upper']:.3g}); "
              f"river {r['river_qe']:.3g} ({r['river_qe_lower']:.2g}-{r['river_qe_upper']:.2g}) mg/g")
        print("  drivers:", [(d["predictor"], round(d["factor_on_qe"], 2)) for d in r["drivers"]])
        print("  warnings:", r["warnings"])
    a = predict({"ReT": "PS", "AgS": "Virgin", "AdC": "Single-ion", "Ce": 2.0}, anchor=(1.0, 0.05))
    print(f"\nanchored: {a['lab_qe']:.4g} mg/g (x/÷ {a['interval_factor']:.2f})")
