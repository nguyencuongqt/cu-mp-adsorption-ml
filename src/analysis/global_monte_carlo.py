"""Monte Carlo estimate of Cu bound to riverine microplastics, global and per river.

Annual mass of MP-bound Cu exported by rivers:

    M [kg/yr] = Q [m3/yr] * N [items/m3] * m_p [g/item] * qe_river [mg/g] * 1e-6 [kg/mg]
    qe_river  = qe_lab(polymer mix, Ce) * FAF

The submitted estimate multiplied median inputs by total discharge and
propagated only the FAF interval. A flux is an arithmetic sum over rivers, and
N, Ce and FAF are strongly right-skewed, so the product of medians understates
the mean (Jensen's inequality). This module therefore uses a two-level Monte
Carlo:

- inner level (variability): virtual rivers are drawn from the distributions
  of N, Ce and FAF, and the global flux uses the mean of N * qe_river over
  them, assuming concentrations are independent of discharge;
- outer level (uncertainty): the distribution parameters, the field studies
  behind the FAF (study-level bootstrap), particle mass and total discharge
  are redrawn.

Laboratory qe comes from the reduced-predictor models of ``transfer_analysis.py``
under the field scenario of ``faf_scaling.py`` (aged, mixed system); the main model
is the isotherm-constrained LightGBM (``lgbm_splice``). The FAF values are study-level ratios (one per field study),
by default from the studies that report dissolved Cu (``--field-subset``).

The FAF is the ratio of field qe to the prediction of the same laboratory
model, so the absolute level of the laboratory model cancels in qe_river and
its study-to-study transfer error is already contained in the between-study
spread of the FAF. ``--include-lab-transfer`` adds that error once more as an
independent factor (between-study SD of the pooled Freundlich residuals); this
is a deliberately conservative upper scenario.

Reported: the median-input estimate (comparable with the submitted value),
the Monte Carlo distribution of the global flux, its ratio to the dissolved
Cu flux, a freeze-one-factor sensitivity analysis, and per-river estimates
with a measured/default flag for every input.

N and Ce distributions come from ``data/global_mp_abundance.csv`` (column
``items_m3``) and ``data/global_dissolved_cu.csv`` (column ``cu_mg_l``) when
these files exist; otherwise a log-normal distribution is reconstructed from
the published median and interquartile range and truncated to its 1st-99th
percentiles.
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
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
    DATA_DIR,
    FIGURES_DIR,
    GLOBAL_CU_MG_L,
    GLOBAL_DISCHARGE_M3_YR,
    GROUP_COL,
    PARTICLE_MASS_G,
    RANDOM_SEED,
    REPORTS_DIR,
    RIVER_DATA_FILE,
    TARGET,
)
from src.analysis.faf_scaling import (  # noqa: E402
    FIELD_SCENARIO,
    LAB_MODELS,
    PRIMARY_SUBSET,
    REFERENCE_MODEL,
    SUBSETS,
    load_field_studies,
    study_level_faf,
    study_ratios,
)
from src.analysis.global_estimation import _normalize_mp_type  # noqa: E402
from src.analysis.grouped_validation import export_tuned_params, load_data  # noqa: E402
from src.analysis.transfer_analysis import fit_predict, log10  # noqa: E402

MP_RAW_FILE = DATA_DIR / "global_mp_abundance.csv"
CU_RAW_FILE = DATA_DIR / "global_dissolved_cu.csv"

# Published summary statistics (SI Text S2) used when raw datasets are absent.
MP_MEDIAN_ITEMS_M3, MP_IQR_ITEMS_M3, MP_N_OBS = 480.0, 3040.0, 768      # Tan et al. (2023)
CU_MEDIAN_MG_L, CU_IQR_MG_L, CU_N_OBS = GLOBAL_CU_MG_L, 40.2e-3, 1628   # Jin et al. (2025)

# Assumed uncertainty of inputs that have a single literature value. These are
# explicit scenario settings and should be replaced by sourced distributions.
PARTICLE_MASS_GSD = 3.0
DISCHARGE_REL_RANGE = 0.07
TRUNCATION = (0.01, 0.99)

GLOBAL_POLYMERS, GLOBAL_WEIGHTS = ["PP", "PE"], [0.5, 0.5]
CE_GRID = np.logspace(-6, 1, 141)
FACTORS = ["mp_abundance", "dissolved_cu", "faf", "particle_mass", "lab_model_transfer", "discharge"]


@dataclass
class Skewed:
    """Site-to-site distribution of a positive quantity."""

    median: float
    sigma_ln: float
    n_obs: int
    raw: np.ndarray | None = None

    @classmethod
    def from_median_iqr(cls, median: float, iqr: float, n_obs: int) -> "Skewed":
        # Log-normal: IQR = 2 * median * sinh(z75 * sigma)
        sigma = float(np.arcsinh(iqr / (2 * median)) / stats.norm.ppf(0.75))
        return cls(median, sigma, n_obs)

    @classmethod
    def from_file(cls, path: Path, column: str) -> "Skewed":
        values = pd.to_numeric(pd.read_csv(path)[column], errors="coerce").dropna().to_numpy()
        values = values[values > 0]
        return cls(float(np.median(values)), float(np.log(values).std(ddof=1)), len(values), values)

    def mean_truncated(self) -> float:
        return float(self.sample_sites(np.random.default_rng(0), 200_000).mean())

    def redraw(self, rng: np.random.Generator) -> "Skewed":
        """One outer draw: sampling uncertainty of the distribution itself."""
        if self.raw is not None:
            boot = rng.choice(self.raw, size=len(self.raw), replace=True)
            return Skewed(float(np.median(boot)), float(np.log(boot).std(ddof=1)), self.n_obs, boot)
        shift = rng.normal(0.0, self.sigma_ln / np.sqrt(self.n_obs))
        return Skewed(self.median * np.exp(shift), self.sigma_ln, self.n_obs)

    def sample_sites(self, rng: np.random.Generator, size: int) -> np.ndarray:
        if self.raw is not None:
            lo, hi = np.quantile(self.raw, TRUNCATION)
            return rng.choice(self.raw[(self.raw >= lo) & (self.raw <= hi)], size=size, replace=True)
        u = rng.uniform(*TRUNCATION, size=size)
        return self.median * np.exp(self.sigma_ln * stats.norm.ppf(u))


def load_input_distributions() -> tuple[Skewed, Skewed, str]:
    mp = (Skewed.from_file(MP_RAW_FILE, "items_m3") if MP_RAW_FILE.exists()
          else Skewed.from_median_iqr(MP_MEDIAN_ITEMS_M3, MP_IQR_ITEMS_M3, MP_N_OBS))
    cu = (Skewed.from_file(CU_RAW_FILE, "cu_mg_l") if CU_RAW_FILE.exists()
          else Skewed.from_median_iqr(CU_MEDIAN_MG_L, CU_IQR_MG_L, CU_N_OBS))
    source = (f"MP: {'raw data' if mp.raw is not None else 'log-normal from median/IQR'}; "
              f"Cu: {'raw data' if cu.raw is not None else 'log-normal from median/IQR'}")
    return mp, cu, source


def polymer_curves(model_name: str, lab_df: pd.DataFrame, polymers: list[str]) -> dict[str, np.ndarray]:
    """Laboratory qe (mg/g) on CE_GRID for each polymer under the field scenario."""
    frame = pd.DataFrame([{"ReT": p, "Ce": c, **FIELD_SCENARIO} for p in polymers for c in CE_GRID])
    pred = fit_predict(model_name, lab_df, frame, export_tuned_params()["RandomForest"])
    return {p: pred[i * len(CE_GRID):(i + 1) * len(CE_GRID)] for i, p in enumerate(polymers)}


def lab_curve(curves: dict[str, np.ndarray], polymers: list[str], weights: list[float]):
    """Interpolator of mixture laboratory qe (mg/g) against Ce."""
    w = np.asarray(weights, dtype=float) / np.sum(weights)
    log_qe = np.log10(sum(wi * curves[p] for wi, p in zip(w, polymers)))
    return lambda ce: np.power(10.0, np.interp(np.log10(np.clip(ce, CE_GRID[0], CE_GRID[-1])),
                                               np.log10(CE_GRID), log_qe))


def lab_between_study_sd(lab_df: pd.DataFrame) -> float:
    """SD (log10) of the study-mean residuals of the pooled Freundlich model."""
    pred = fit_predict(REFERENCE_MODEL, lab_df, lab_df, export_tuned_params()["RandomForest"])
    resid = pd.Series(log10(lab_df[TARGET]) - log10(pred), index=lab_df.index)
    return float(resid.groupby(lab_df[GROUP_COL]).mean().std(ddof=1))


def simulate_global(
    curve, faf: np.ndarray, mp: Skewed, cu: Skewed, lab_sd_log10: float,
    n_outer: int, n_inner: int, frozen: tuple[str, ...] = (), seed: int = RANDOM_SEED,
) -> pd.DataFrame:
    """Two-level Monte Carlo; ``frozen`` factors are held at their median value."""
    rng = np.random.default_rng(seed)
    rows = []
    for _ in range(n_outer):
        mp_b = mp if "mp_abundance" in frozen else mp.redraw(rng)
        cu_b = cu if "dissolved_cu" in frozen else cu.redraw(rng)
        n_sites = (np.full(n_inner, mp.median) if "mp_abundance" in frozen
                   else mp_b.sample_sites(rng, n_inner))
        ce_sites = (np.full(n_inner, cu.median) if "dissolved_cu" in frozen
                    else cu_b.sample_sites(rng, n_inner))
        if "faf" in frozen:
            faf_sites = np.full(n_inner, np.median(faf))
        else:  # study-level bootstrap, then site variability among the resampled studies
            faf_sites = rng.choice(rng.choice(faf, size=len(faf), replace=True), size=n_inner)
        mass = 1.0 if "particle_mass" in frozen else np.exp(rng.normal(0, np.log(PARTICLE_MASS_GSD)))
        lab = 1.0 if "lab_model_transfer" in frozen else 10 ** rng.normal(0, lab_sd_log10)
        q = 1.0 if "discharge" in frozen else rng.uniform(1 - DISCHARGE_REL_RANGE, 1 + DISCHARGE_REL_RANGE)

        discharge = GLOBAL_DISCHARGE_M3_YR * q
        bound_mg_m3 = np.mean(n_sites * PARTICLE_MASS_G * mass * curve(ce_sites) * lab * faf_sites)
        mp_bound_kg = discharge * bound_mg_m3 / 1e6
        dissolved_kg = discharge * np.mean(ce_sites) / 1e3  # mg/L = g/m3
        rows.append((mp_bound_kg, dissolved_kg))
    out = pd.DataFrame(rows, columns=["mp_bound_kg_yr", "dissolved_kg_yr"])
    out["pct_of_dissolved"] = out["mp_bound_kg_yr"] / out["dissolved_kg_yr"] * 100
    return out


def describe(draws: pd.Series) -> dict[str, float]:
    q = draws.quantile([0.025, 0.25, 0.5, 0.75, 0.975])
    return {"p2.5": q[0.025], "p25": q[0.25], "median": q[0.5], "p75": q[0.75],
            "p97.5": q[0.975], "mean": float(draws.mean())}


def sensitivity(curve, faf, mp, cu, lab_sd, n_outer, n_inner, full: pd.DataFrame) -> pd.DataFrame:
    """Freeze one factor at a time and record the change in spread and centre."""
    base_log = np.log10(full["mp_bound_kg_yr"])
    rows = []
    for factor in [f for f in FACTORS if f != "lab_model_transfer" or lab_sd > 0]:
        draws = simulate_global(curve, faf, mp, cu, lab_sd, n_outer, n_inner, frozen=(factor,))
        log = np.log10(draws["mp_bound_kg_yr"])
        rows.append({"factor_frozen": factor,
                     "variance_share_log10": float(1 - log.var() / base_log.var()),
                     "median_kg_yr_when_frozen": float(draws["mp_bound_kg_yr"].median()),
                     "median_ratio_full_to_frozen": float(full["mp_bound_kg_yr"].median()
                                                          / draws["mp_bound_kg_yr"].median())})
    return pd.DataFrame(rows).sort_values("variance_share_log10", ascending=False)


def river_estimates(curves, faf, mp, cu, lab_sd, n_draws: int) -> pd.DataFrame:
    """Per-river mass with measured/default flags and Monte Carlo intervals."""
    rng = np.random.default_rng(RANDOM_SEED)
    rivers = pd.read_csv(RIVER_DATA_FILE)
    rows = []
    for _, r in rivers.iterrows():
        polymers = _normalize_mp_type(r["MP_type"])
        curve = lab_curve(curves, polymers, [1.0] * len(polymers))
        mp_measured = not np.isclose(r["MP_items_m3"], MP_MEDIAN_ITEMS_M3)
        ce_measured = not np.isclose(r["Ce_mg_L"], GLOBAL_CU_MG_L)
        n = np.full(n_draws, r["MP_items_m3"]) if mp_measured else mp.sample_sites(rng, n_draws)
        ce = np.full(n_draws, r["Ce_mg_L"]) if ce_measured else cu.sample_sites(rng, n_draws)
        mass = np.exp(rng.normal(0, np.log(PARTICLE_MASS_GSD), n_draws))
        lab = 10 ** rng.normal(0, lab_sd, n_draws)
        draws = r["Q_m3_yr"] * n * PARTICLE_MASS_G * mass * curve(ce) * lab * rng.choice(faf, n_draws) / 1e6
        central = (r["Q_m3_yr"] * r["MP_items_m3"] * PARTICLE_MASS_G
                   * float(curve(np.array([r["Ce_mg_L"]]))[0]) * np.median(faf) / 1e6)
        rows.append({"river": r["river"], "country": r["country"], "polymers": "/".join(polymers),
                     "MP_items_m3": r["MP_items_m3"], "MP_source": "measured" if mp_measured else "default",
                     "Ce_mg_L": r["Ce_mg_L"], "Ce_source": "measured" if ce_measured else "default",
                     "Q_m3_yr": r["Q_m3_yr"], "mass_kg_yr_central": central,
                     "mass_kg_yr_p2.5": float(np.percentile(draws, 2.5)),
                     "mass_kg_yr_median": float(np.median(draws)),
                     "mass_kg_yr_p97.5": float(np.percentile(draws, 97.5)),
                     "qc_note": "Ce far above global range; verify source units" if r["Ce_mg_L"] > 1 else ""})
    return pd.DataFrame(rows).sort_values("mass_kg_yr_central", ascending=False)


def save_plot(results: dict[str, pd.DataFrame], sens: pd.DataFrame, primary: str) -> Path:
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11.5, 4.6))
    colors = {"lgbm_splice": "#000000", "rf_raw": "#c43c39", "lgbm_monotone": "#f58518", "xgb_monotone": "#54a24b",
              "freundlich_pooled": "#2f6f9f"}
    bins = np.linspace(0, 8, 81)
    for name, draws in results.items():
        ax1.hist(np.log10(draws["mp_bound_kg_yr"]), bins=bins, histtype="step", linewidth=1.8,
                 color=colors[name], label=name)
    ax1.set_xlabel("log10 MP-bound Cu flux (kg/yr)")
    ax1.set_ylabel("Monte Carlo draws")
    ax1.set_title("a  Global flux by laboratory model", loc="left", fontsize=11)
    ax1.legend(frameon=False, fontsize=9)
    part = sens.sort_values("variance_share_log10")
    ax2.barh(part["factor_frozen"], part["variance_share_log10"] * 100, color="#2f6f9f")
    ax2.set_xlabel("Reduction in variance of log10 flux when frozen (%)")
    ax2.set_title(f"b  Sensitivity ({primary})", loc="left", fontsize=11)
    ax2.grid(axis="x", linestyle=":", alpha=0.5)
    fig.tight_layout()
    output = FIGURES_DIR / "global_monte_carlo.png"
    fig.savefig(output, dpi=300)
    plt.close(fig)
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--n-outer", type=int, default=2000)
    parser.add_argument("--n-inner", type=int, default=2000)
    parser.add_argument("--field-subset", default=PRIMARY_SUBSET, choices=list(SUBSETS),
                        help="Field studies behind the FAF.")
    parser.add_argument("--include-lab-transfer", action="store_true",
                        help="Add the laboratory between-study error as an extra factor (conservative).")
    parser.add_argument("--primary", default=REFERENCE_MODEL, choices=LAB_MODELS,
                        help="Laboratory model used for sensitivity and river tables.")
    parser.add_argument("--tag", default="", help="Suffix for output file names (sensitivity runs).")
    args = parser.parse_args()

    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    lab_df = load_data("all")
    studies = load_field_studies()
    river_polymers = {p for t in pd.read_csv(RIVER_DATA_FILE)["MP_type"] for p in _normalize_mp_type(t)}
    polymers = sorted(river_polymers | set(GLOBAL_POLYMERS))
    missing = set(polymers) - set(lab_df["ReT"])
    if missing:
        raise ValueError(f"Polymers without laboratory data: {sorted(missing)}")
    lab_sd = lab_between_study_sd(lab_df) if args.include_lab_transfer else 0.0
    mp, cu, source = load_input_distributions()
    print(f"Input distributions - {source}")
    print(f"MP: median {mp.median:.0f} items/m3, sigma_ln {mp.sigma_ln:.2f}, truncated mean {mp.mean_truncated():.0f}")
    print(f"Cu: median {cu.median*1e3:.2f} ug/L, sigma_ln {cu.sigma_ln:.2f}, truncated mean {cu.mean_truncated()*1e3:.1f} ug/L")

    summary, results, fitted = [], {}, {}
    for name in LAB_MODELS:
        curves = polymer_curves(name, lab_df, polymers)
        faf = study_level_faf(study_ratios(name, lab_df, studies), args.field_subset)
        curve = lab_curve(curves, GLOBAL_POLYMERS, GLOBAL_WEIGHTS)
        fitted[name] = (curves, faf, curve)
        qe_lab = float(curve(np.array([cu.median]))[0])
        median_inputs = (GLOBAL_DISCHARGE_M3_YR * mp.median * PARTICLE_MASS_G * qe_lab * np.median(faf) / 1e6)
        draws = simulate_global(curve, faf, mp, cu, lab_sd, args.n_outer, args.n_inner)
        results[name] = draws
        summary.append({"lab_model": name, "field_subset": args.field_subset, "n_field_studies": len(faf),
                        "lab_transfer_sd_log10": lab_sd,
                        "qe_lab_at_median_Ce_mg_g": qe_lab, "faf_median": float(np.median(faf)),
                        "qe_river_median_inputs_mg_g": qe_lab * float(np.median(faf)),
                        "flux_median_inputs_kg_yr": median_inputs,
                        **{f"flux_mc_{k}_kg_yr": v for k, v in describe(draws["mp_bound_kg_yr"]).items()},
                        **{f"pct_of_dissolved_{k}": v for k, v in describe(draws["pct_of_dissolved"]).items()
                           if k in {"p2.5", "median", "p97.5"}}})
    summary_df = pd.DataFrame(summary)
    summary_df.to_csv(REPORTS_DIR / f"global_mc_summary{args.tag}.csv", index=False)

    curves, faf, curve = fitted[args.primary]
    sens = sensitivity(curve, faf, mp, cu, lab_sd, args.n_outer, args.n_inner, results[args.primary])
    sens.to_csv(REPORTS_DIR / f"global_mc_sensitivity{args.tag}.csv", index=False)
    rivers = river_estimates(curves, faf, mp, cu, lab_sd, 20000)
    rivers.to_csv(REPORTS_DIR / f"river_mc_results{args.tag}.csv", index=False)

    fmt = lambda v: f"{v:.3g}"  # noqa: E731
    print("\nGlobal MP-bound Cu flux (kg/yr)")
    print(summary_df[["lab_model", "qe_lab_at_median_Ce_mg_g", "faf_median", "flux_median_inputs_kg_yr",
                      "flux_mc_p2.5_kg_yr", "flux_mc_median_kg_yr", "flux_mc_p97.5_kg_yr",
                      "pct_of_dissolved_median", "pct_of_dissolved_p97.5"]].to_string(index=False, float_format=fmt))
    print(f"\nSensitivity ({args.primary})")
    print(sens.to_string(index=False, float_format=fmt))
    print(f"\nRivers ({args.primary}); measured MP in {int((rivers['MP_source'] == 'measured').sum())}, "
          f"measured Ce in {int((rivers['Ce_source'] == 'measured').sum())} of {len(rivers)}")
    print(rivers[["river", "MP_source", "Ce_source", "mass_kg_yr_central", "mass_kg_yr_p2.5",
                  "mass_kg_yr_p97.5", "qc_note"]].to_string(index=False, float_format=fmt))
    if not args.tag:
        print(f"Saved figure: {save_plot(results, sens, args.primary)}")


if __name__ == "__main__":
    main()
