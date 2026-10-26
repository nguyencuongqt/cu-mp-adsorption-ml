"""Empirical field-to-laboratory scaling factor with study-level inference.

For each field record k (one site or season of a field study), the scaling factor is
the Cu loading measured on environmental microplastics divided by the laboratory-model
prediction at the Cu concentration measured in the surrounding water:

    ratio_k = Cu_MP,field,k / sum_p( w_p,k * qe_lab(polymer p, Ce_k) )

where w_p,k are the reported polymer fractions of the surface-water microplastics.
Records of the same field study are first combined (geometric mean), so that every
study contributes one value to the summary statistics; treating sites or seasons of
one study as independent would be pseudo-replication.

The laboratory prediction uses the reduced predictor set (aging status, polymer,
adsorption system, Ce) under the field scenario "aged, mixed system" and is computed
with several laboratory models, so the dependence of the factor on that choice is
reported explicitly. ``freundlich_pooled`` (linear model of log10 qe on log10 Ce,
polymer, ageing status and system) is the model selected by study-grouped validation
(decision of 2026-10-07): it is as accurate as the tree ensembles for a new laboratory
and the least biased below the laboratory Ce range (``model_role_comparison.py``,
``splice_anchor_test.py``). ``lgbm_splice`` (monotone LightGBM with a Freundlich tail),
the monotone boosted models and the unconstrained ``rf_raw`` are the ML benchmarks.

Outputs
-------
- ``faf_scaling_by_study.csv``: Ce, field loading, laboratory qe and ratio per record and model;
- ``faf_scaling_summary.csv``: study-level geometric mean, between-study SD, 95% CI of the
  geometric mean, 95% prediction interval for a new site, cluster bootstrap CI of the
  median and leave-one-study-out range, for all studies and for sensitivity subsets;
- ``faf_scaling.png``.
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from config import (  # noqa: E402
    FAF_BOOTSTRAP_ITERS,
    FIELD_DATA_FILE,
    FIGURES_DIR,
    RANDOM_SEED,
    REPORTS_DIR,
    RIVER_DEFAULTS,
)
from src.analysis.grouped_validation import export_tuned_params, load_data  # noqa: E402
from src.analysis.transfer_analysis import fit_predict  # noqa: E402

RECORD_COL = "reference"
STUDY_COL = "study"
LAB_MODELS = ["freundlich_pooled", "xgb_blend", "xgb_monotone", "lgbm_monotone", "rf_raw"]
REFERENCE_MODEL = "freundlich_pooled"
MODEL_LABELS = {"lgbm_splice": "Isotherm-constrained LightGBM",
                "freundlich_pooled": "Isotherm-form linear model (selected)", "xgb_monotone": "Monotone XGBoost",
                "lgbm_monotone": "Monotone LightGBM", "rf_raw": "Random Forest (unconstrained)"}
FIELD_SCENARIO = {"AgS": RIVER_DEFAULTS["AgS"], "AdC": RIVER_DEFAULTS["AdC"]}
# Sensitivity subsets of field studies (None = all)
SUBSETS = {
    "all": None,
    "without_Zhang_2025b": ["Zhang_2025b"],          # wastewater-dominated river, water Cu far above typical levels
    "without_Purwiyanto_2020": ["Purwiyanto_2020"],  # water Cu is unfiltered (total), not dissolved
}
# Main result: field studies that report dissolved Cu; "all" is the sensitivity case
PRIMARY_SUBSET = "without_Purwiyanto_2020"


def study_level_faf(by_record: pd.DataFrame, subset: str = PRIMARY_SUBSET) -> np.ndarray:
    """Study-level ratios of one laboratory model for a subset of field studies."""
    dropped = SUBSETS[subset]
    kept = by_record if dropped is None else by_record[~by_record[STUDY_COL].isin(dropped)]
    return study_level(kept).to_numpy()


def load_field_studies(path: Path = FIELD_DATA_FILE) -> list[dict]:
    """One record per field site or season, with the study it belongs to."""
    field = pd.read_csv(path)
    if STUDY_COL not in field.columns:
        field[STUDY_COL] = field[RECORD_COL]
    records = []
    for ref, part in field.groupby(RECORD_COL, sort=False):
        if part["real_qe"].nunique() != 1 or part["Ce"].nunique() != 1 or part[STUDY_COL].nunique() != 1:
            raise ValueError(f"{ref}: expected one field loading, one Ce and one study per record.")
        records.append({
            RECORD_COL: ref,
            STUDY_COL: part[STUDY_COL].iloc[0],
            "Ce": float(part["Ce"].iloc[0]),
            "qe_field": float(part["real_qe"].iloc[0]),
            "polymers": part["ReT"].tolist(),
            "weights": part["Resin_pct"].astype(float).tolist(),
        })
    return records


def lab_qe(model_name: str, lab_df: pd.DataFrame, records: list[dict]) -> np.ndarray:
    """Composition-weighted laboratory qe (mg/g) for every field record."""
    rows, owner = [], []
    for i, rec in enumerate(records):
        for polymer in rec["polymers"]:
            rows.append({"ReT": polymer, "Ce": rec["Ce"], **FIELD_SCENARIO})
            owner.append(i)
    frame = pd.DataFrame(rows)
    pred = fit_predict(model_name, lab_df, frame, export_tuned_params()["RandomForest"])
    owner = np.asarray(owner)
    out = np.empty(len(records))
    for i, rec in enumerate(records):
        w = np.asarray(rec["weights"], dtype=float)
        out[i] = float(np.sum(w / w.sum() * pred[owner == i]))
    return out


def study_ratios(model_name: str, lab_df: pd.DataFrame, records: list[dict]) -> pd.DataFrame:
    qe = lab_qe(model_name, lab_df, records)
    return pd.DataFrame([
        {"lab_model": model_name, STUDY_COL: rec[STUDY_COL], RECORD_COL: rec[RECORD_COL], "Ce_mg_L": rec["Ce"],
         "qe_field_mg_g": rec["qe_field"], "qe_lab_mg_g": q, "faf": rec["qe_field"] / q,
         "Kd_field_L_g": rec["qe_field"] / rec["Ce"]}
        for rec, q in zip(records, qe)
    ])


def study_level(by_record: pd.DataFrame) -> pd.Series:
    """Geometric mean of the record ratios within each field study."""
    return np.power(10.0, np.log10(by_record["faf"]).groupby(by_record[STUDY_COL]).mean())


def summarise_faf(faf: np.ndarray, n_boot: int = FAF_BOOTSTRAP_ITERS) -> dict[str, float]:
    """Summary of study-level ratios on the log10 scale (one value per field study)."""
    log_faf = np.log10(faf)
    k = len(log_faf)
    mean, sd = float(log_faf.mean()), float(log_faf.std(ddof=1))
    t = stats.t.ppf(0.975, k - 1)
    rng = np.random.default_rng(RANDOM_SEED)
    boot = np.median(rng.choice(faf, size=(n_boot, k), replace=True), axis=1)
    loo = [10 ** np.delete(log_faf, i).mean() for i in range(k)]
    return {
        "n_field_studies": k,
        "median": float(np.median(faf)),
        "geometric_mean": 10 ** mean,
        "between_study_sd_log10": sd,
        "gm_ci95_low": 10 ** (mean - t * sd / np.sqrt(k)),
        "gm_ci95_high": 10 ** (mean + t * sd / np.sqrt(k)),
        "new_site_pi95_low": 10 ** (mean - t * sd * np.sqrt(1 + 1 / k)),
        "new_site_pi95_high": 10 ** (mean + t * sd * np.sqrt(1 + 1 / k)),
        "median_boot_ci95_low": float(np.percentile(boot, 2.5)),
        "median_boot_ci95_high": float(np.percentile(boot, 97.5)),
        "leave_one_study_out_gm_min": float(min(loo)),
        "leave_one_study_out_gm_max": float(max(loo)),
        "max_to_min_ratio": float(faf.max() / faf.min()),
    }


def save_plot(by_record: pd.DataFrame, summary: pd.DataFrame) -> Path:
    fig, ax = plt.subplots(figsize=(9.2, 5.0))
    records = list(by_record[RECORD_COL].unique())
    markers = ["o", "s", "^", "D", "v", "P", "X", "*", "<", ">"]
    for i, name in enumerate(LAB_MODELS):
        part = by_record[by_record["lab_model"] == name]
        for j, record in enumerate(records):
            value = part.loc[part[RECORD_COL] == record, "faf"]
            ax.scatter(value, [i] * len(value), marker=markers[j % len(markers)], s=55, color="#2f6f9f",
                       edgecolor="white", zorder=3, label=record if i == 0 else None)
        row = summary[(summary["lab_model"] == name) & (summary["field_subset"] == PRIMARY_SUBSET)].iloc[0]
        ax.plot([row["gm_ci95_low"], row["gm_ci95_high"]], [i + 0.25] * 2, color="#c43c39", linewidth=2)
        ax.scatter([row["geometric_mean"]], [i + 0.25], color="#c43c39", marker="|", s=160, zorder=4)
    ax.set_xscale("log")
    ax.set_yticks(range(len(LAB_MODELS)))
    ax.set_yticklabels([MODEL_LABELS.get(m, m) for m in LAB_MODELS])
    ax.set_ylim(-0.6, len(LAB_MODELS) - 0.2)
    ax.axvline(1.0, color="0.4", linestyle="--", linewidth=0.8)
    ax.set_xlabel("Field-to-laboratory ratio")
    ax.grid(axis="x", which="major", linestyle=":", alpha=0.5)
    ax.legend(frameon=False, fontsize=7, loc="upper left", bbox_to_anchor=(1.0, 1.0), title="Field record")
    fig.tight_layout()
    output = FIGURES_DIR / "faf_scaling.png"
    fig.savefig(output, dpi=300)
    plt.close(fig)
    return output


def run() -> tuple[pd.DataFrame, pd.DataFrame]:
    lab_df = load_data("all")
    records = load_field_studies()
    by_record = pd.concat([study_ratios(m, lab_df, records) for m in LAB_MODELS], ignore_index=True)
    rows = []
    for model in LAB_MODELS:
        part = by_record[by_record["lab_model"] == model]
        for subset in SUBSETS:
            rows.append({"lab_model": model, "field_subset": subset, "primary": subset == PRIMARY_SUBSET,
                         **summarise_faf(study_level_faf(part, subset))})
    return by_record, pd.DataFrame(rows)


def main() -> None:
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    by_record, summary = run()
    by_record.to_csv(REPORTS_DIR / "faf_scaling_by_study.csv", index=False)
    summary.to_csv(REPORTS_DIR / "faf_scaling_summary.csv", index=False)

    fmt = lambda v: f"{v:.3g}"  # noqa: E731
    ref = by_record[by_record["lab_model"] == REFERENCE_MODEL]
    print(f"Field records with the reference laboratory model ({REFERENCE_MODEL})")
    print(ref[[RECORD_COL, "Ce_mg_L", "qe_field_mg_g", "qe_lab_mg_g", "faf"]].to_string(index=False, float_format=fmt))
    print("\nRatio by field record and laboratory model")
    print(by_record.pivot(index=RECORD_COL, columns="lab_model", values="faf")[LAB_MODELS].to_string(float_format=fmt))
    print("\nStudy-level summary (one value per field study)")
    print(summary[["lab_model", "field_subset", "n_field_studies", "median", "geometric_mean", "gm_ci95_low",
                   "gm_ci95_high", "new_site_pi95_low", "new_site_pi95_high", "between_study_sd_log10",
                   "max_to_min_ratio"]].to_string(index=False, float_format=fmt))
    print(f"Saved figure: {save_plot(by_record, summary)}")


if __name__ == "__main__":
    main()
