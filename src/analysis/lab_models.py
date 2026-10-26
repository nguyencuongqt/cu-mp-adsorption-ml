"""Laboratory reference models used for lab-to-field scaling.

Three models of laboratory equilibrium uptake are fitted on the study-grouped
dataset and share one prediction interface, so that the field scaling factor
and the global estimate can be computed with each of them:

- ``qrf_raw``: Quantile Random Forest on qe (the originally submitted model);
- ``qrf_log``: Quantile Random Forest on log10(qe);
- ``freundlich``: mixed-effects Freundlich isotherm,
  ``log10 qe = b0 + b_ReT + b_AgS + b_AdC + n * log10 Ce + u_study``,
  with a random intercept and slope per study.

Tree ensembles return the mean/quantile of the training leaves and therefore
cannot extrapolate below the lowest Ce they were trained on. The Freundlich
model keeps a monotonic power-law dependence on Ce and supplies the
isotherm-constrained comparator for field-level concentrations.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from config import CAT_COLS, GROUP_COL, NUM_COLS, REPORTS_DIR, RIVER_DEFAULTS, TARGET  # noqa: E402
from src.analysis.grouped_validation import (  # noqa: E402
    CE_FLOOR,
    QE_FLOOR,
    export_tuned_params,
    make_pipeline,
)

LAB_MODELS = ["qrf_raw", "qrf_log", "freundlich"]
FEATURES = CAT_COLS + NUM_COLS


def _log_frame(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["log_ce"] = np.log10(np.clip(out["Ce"].astype(float), CE_FLOOR, None))
    if TARGET in out.columns:
        out["log_qe"] = np.log10(np.clip(out[TARGET].astype(float), QE_FLOOR, None))
    return out


@dataclass
class LabModel:
    """Fitted laboratory model with a common ``predict`` method (mg/g)."""

    name: str
    estimator: Any
    between_study_sd_log10: float  # study-to-study SD of log10 qe at fixed inputs
    known_polymers: tuple[str, ...]

    def predict(self, x: pd.DataFrame) -> np.ndarray:
        if self.name == "freundlich":
            frame = _log_frame(x)
            # Polymers absent from the laboratory data take the reference level.
            frame["ReT"] = frame["ReT"].where(frame["ReT"].isin(self.known_polymers),
                                              self.known_polymers[0])
            return np.power(10.0, np.asarray(self.estimator.predict(frame)))
        pred = np.asarray(self.estimator.predict(x[FEATURES], quantiles=[0.5])).reshape(-1)
        return np.power(10.0, pred) if self.name == "qrf_log" else pred


def fit_freundlich(df: pd.DataFrame) -> LabModel:
    frame = _log_frame(df)
    formula = "log_qe ~ log_ce + C(ReT) + C(AgS) + C(AdC)"
    try:
        fit = smf.mixedlm(formula, frame, groups=frame[GROUP_COL], re_formula="~log_ce").fit(
            method="lbfgs", reml=True)
        between = float(np.sqrt(fit.cov_re.iloc[0, 0]))
    except Exception:  # fall back to a random intercept if the slope term fails
        fit = smf.mixedlm(formula, frame, groups=frame[GROUP_COL]).fit(reml=True)
        between = float(np.sqrt(fit.cov_re.iloc[0, 0]))
    return LabModel("freundlich", fit, between, tuple(sorted(frame["ReT"].unique())))


def fit_qrf(df: pd.DataFrame, target: str) -> LabModel:
    params = export_tuned_params()["RandomForest"]
    pipe = make_pipeline("QRF", params, FEATURES, target)
    y = np.log10(np.clip(df[TARGET], QE_FLOOR, None)) if target == "log" else df[TARGET]
    pipe.fit(df[FEATURES], y)
    return LabModel(f"qrf_{target}", pipe, _loso_between_study_sd(target),
                    tuple(sorted(df["ReT"].unique())))


def _loso_between_study_sd(target: str) -> float:
    """SD of study-mean log10 errors from the leave-one-study-out predictions."""
    path = REPORTS_DIR / "grouped_cv_loso_predictions.csv"
    if not path.exists():
        raise FileNotFoundError(f"{path} not found. Run src/analysis/grouped_validation.py first.")
    loso = pd.read_csv(path)
    loso = loso[(loso["model"] == "RandomForest") & (loso["target"] == target)]
    err = np.log10(np.clip(loso["y"], QE_FLOOR, None)) - np.log10(np.clip(loso["pred"], QE_FLOOR, None))
    return float(err.groupby(loso[GROUP_COL]).mean().std(ddof=1))


def fit_lab_model(name: str, df: pd.DataFrame) -> LabModel:
    if name == "freundlich":
        return fit_freundlich(df)
    if name in {"qrf_raw", "qrf_log"}:
        return fit_qrf(df, name.split("_")[1])
    raise ValueError(f"Unknown laboratory model: {name}")


def field_scenario_frame(polymers: list[str], ce_mg_l: np.ndarray) -> pd.DataFrame:
    """Predictor rows for every polymer x Ce pair under the shared field scenario."""
    ce = np.atleast_1d(np.asarray(ce_mg_l, dtype=float))
    rows = pd.DataFrame({"ReT": np.repeat(polymers, len(ce)), "Ce": np.tile(ce, len(polymers))})
    for key, value in RIVER_DEFAULTS.items():
        rows[key] = value
    return rows[FEATURES]


def predict_mixture_qe(
    model: LabModel, polymers: list[str], weights: list[float], ce_mg_l: np.ndarray
) -> np.ndarray:
    """Composition-weighted laboratory qe (mg/g) at the given Ce values."""
    ce = np.atleast_1d(np.asarray(ce_mg_l, dtype=float))
    w = np.asarray(weights, dtype=float)
    w = w / w.sum()
    pred = model.predict(field_scenario_frame(polymers, ce)).reshape(len(polymers), len(ce))
    return (w[:, None] * pred).sum(axis=0)
