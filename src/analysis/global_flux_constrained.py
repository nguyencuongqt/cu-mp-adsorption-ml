"""Global MP-bound Cu flux constrained by published riverine plastic mass fluxes.

    M_Cu [kg/yr] = F_MP [kt/yr] * qe_river [mg/g]          (1 kt * 1 mg/g = 1000 kg)
    qe_river     = qe_lab(PP/PE, Ce) * FAF

The count-based route of ``global_monte_carlo.py`` multiplies literature MP counts by
one particle mass and by global discharge. This script first reports the plastic mass
flux that the count-based route implies; with the compiled count data it exceeds
published estimates of total riverine plastic export, because the compiled counts are
biased towards polluted sites and depend on the lower size limit of each survey.
Equation (1) replaces counts, particle mass and discharge by an independent plastic
mass flux, and uses the discharge-weighted world-average dissolved Cu concentration
instead of the median of a literature compilation.

Outputs: global_flux_constrained.csv, global_flux_implied_mp_mass.csv (outputs/reports).
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from config import GLOBAL_DISCHARGE_M3_YR, PARTICLE_MASS_G, RANDOM_SEED, REPORTS_DIR  # noqa: E402
from src.analysis.faf_scaling import LAB_MODELS, PRIMARY_SUBSET, load_field_studies, study_level_faf, study_ratios  # noqa: E402
from src.analysis.global_monte_carlo import (  # noqa: E402
    GLOBAL_POLYMERS,
    GLOBAL_WEIGHTS,
    lab_curve,
    load_input_distributions,
    polymer_curves,
)
from src.analysis.grouped_validation import load_data  # noqa: E402

# Riverine plastic mass flux to the ocean (kt/yr). "low" and "central" are microplastic
# estimates and span the Monte Carlo range; "high" is the export of plastics of all sizes
# and is reported as an upper bound only.
MP_FLUX_KT_YR = {
    "low": (6.1, "Weiss et al. (2021), microplastics, best estimate (Table 1, r2 = 0.80); bootstrap interval 3.4-13.9"),
    "central": (53.0, "Strokal et al. (2023), microplastics; van Wijnen et al. (2019) give 47"),
    "high": (500.0, "Strokal et al. (2023), macro- plus microplastics (upper bound)"),
}
# Not used as a scenario: Mai et al. (2020) give 57-265 kt/yr (median 134) for plastics of all
# sizes collected with 0.1-1 mm nets, of which microplastics are about 2.2% by mass.
# Dissolved Cu (mg/L): world-average river water, 23.3 nmol/kg (Gaillardet et al., 2014)
CE_WORLD_AVERAGE_MG_L = 23.3e-9 * 63.546 * 1e3
N_DRAWS = 20000


def implied_mp_mass_flux() -> pd.DataFrame:
    """Plastic mass flux (Mt/yr) implied by the count-based inputs."""
    mp, _, source = load_input_distributions()
    raw_mean = float(mp.raw.mean()) if mp.raw is not None else np.nan
    rows = [("median", mp.median), ("mean, 1st-99th percentile", mp.mean_truncated()), ("mean, all data", raw_mean)]
    return pd.DataFrame([{"mp_count_statistic": label, "items_m3": n,
                          "implied_mp_mass_flux_Mt_yr": GLOBAL_DISCHARGE_M3_YR * n * PARTICLE_MASS_G / 1e12,
                          "input_source": source} for label, n in rows])


def main() -> None:
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    implied = implied_mp_mass_flux()
    implied.to_csv(REPORTS_DIR / "global_flux_implied_mp_mass.csv", index=False)

    lab_df, studies = load_data("all"), load_field_studies()
    _, cu, _ = load_input_distributions()
    dissolved_t_yr = GLOBAL_DISCHARGE_M3_YR * CE_WORLD_AVERAGE_MG_L / 1e6  # mg/L = g/m3
    ce_cases = {"world_average": CE_WORLD_AVERAGE_MG_L, "compilation_median": cu.median}
    rng = np.random.default_rng(RANDOM_SEED)
    low, high = MP_FLUX_KT_YR["low"][0], MP_FLUX_KT_YR["central"][0]
    rows = []
    for model in LAB_MODELS:
        curve = lab_curve(polymer_curves(model, lab_df, GLOBAL_POLYMERS), GLOBAL_POLYMERS, GLOBAL_WEIGHTS)
        faf = study_level_faf(study_ratios(model, lab_df, studies), PRIMARY_SUBSET)
        for ce_label, ce in ce_cases.items():
            qe_lab = float(curve(np.array([ce]))[0])
            qe_river = qe_lab * float(np.exp(np.log(faf).mean()))   # geometric mean, as in faf_scaling
            # MP flux: log-uniform between the two microplastic estimates; FAF: bootstrap of study-level geometric means
            flux = np.exp(rng.uniform(np.log(low), np.log(high), N_DRAWS))
            gm = np.exp(np.log(rng.choice(faf, size=(N_DRAWS, len(faf)))).mean(axis=1))
            draws = flux * qe_lab * gm * 1e3
            row = {"lab_model": model, "Ce_case": ce_label, "Ce_ug_L": ce * 1e3, "qe_lab_ug_g": qe_lab * 1e3,
                   "faf_geometric_mean": float(np.exp(np.log(faf).mean())), "qe_river_ug_g": qe_river * 1e3}
            for name, (value, _) in MP_FLUX_KT_YR.items():
                row[f"Cu_kg_yr_{name}_MP_flux"] = value * qe_river * 1e3
            # share of the dissolved Cu flux at the same Cu concentration (until 9 October 2026 the world-average flux was
            # used as denominator for both concentrations)
            dissolved_case_t_yr = GLOBAL_DISCHARGE_M3_YR * ce / 1e6
            row.update({"Cu_kg_yr_mc_p2.5": float(np.percentile(draws, 2.5)), "Cu_kg_yr_mc_median": float(np.median(draws)),
                        "Cu_kg_yr_mc_p97.5": float(np.percentile(draws, 97.5)),
                        "pct_of_dissolved_central": row["Cu_kg_yr_central_MP_flux"] / (dissolved_case_t_yr * 1e3) * 100,
                        "pct_of_dissolved_high": row["Cu_kg_yr_high_MP_flux"] / (dissolved_case_t_yr * 1e3) * 100})
            rows.append(row)
    out = pd.DataFrame(rows)
    out.to_csv(REPORTS_DIR / "global_flux_constrained.csv", index=False)

    fmt = lambda v: f"{v:.3g}"  # noqa: E731
    print("Plastic mass flux implied by the count-based inputs")
    print(implied.drop(columns="input_source").to_string(index=False, float_format=fmt))
    print("\nPublished riverine plastic mass flux scenarios (kt/yr)")
    for name, (value, ref) in MP_FLUX_KT_YR.items():
        print(f"  {name}: {value} - {ref}")
    print(f"\nDissolved Cu flux at the world-average concentration ({CE_WORLD_AVERAGE_MG_L*1e3:.2f} ug/L): {dissolved_t_yr:,.0f} t/yr")
    print("\nMP-bound Cu flux (kg/yr)")
    print(out.to_string(index=False, float_format=fmt))


if __name__ == "__main__":
    main()
