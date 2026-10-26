"""Illustrative application of the field-calibrated model to individual rivers.

    M_Cu [kg/yr] = Q [m3/yr] * N [items/m3] * m_p [g/item] * qe_lab(polymers, Ce) [mg/g] * FAF * 1e-6

Only rivers with both a measured MP abundance and a measured dissolved Cu concentration
are reported in the main table; the others rely on global default inputs and go to the
supplementary table with a flag for every input. The estimate is not a ranking of rivers.

Because MP counts depend on the lower size limit of each survey, the particle mass m_p is
an assumption; results are given for the value used in the submitted manuscript and for
the median masses of net-sampled riverine microplastics reported by Mai et al. (2020) and
Weiss et al. (2021). The range for each river comes from the lowest and highest
study-level field-to-laboratory ratio.

Outputs (outputs/reports): river_application_main.csv, river_application_SI.csv.
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

from config import PARTICLE_MASS_G, REPORTS_DIR, RIVER_DATA_FILE  # noqa: E402
from src.analysis.faf_scaling import PRIMARY_SUBSET, REFERENCE_MODEL, load_field_studies, study_level_faf, study_ratios  # noqa: E402
from src.analysis.global_estimation import _normalize_mp_type  # noqa: E402
from src.analysis.global_monte_carlo import lab_curve, polymer_curves  # noqa: E402
from src.analysis.grouped_validation import load_data  # noqa: E402

# Particle mass (g/item) and its source
PARTICLE_MASS_CASES = {
    "12.5ug": (PARTICLE_MASS_G, "submitted manuscript (Koelmans et al., 2020)"),
    "170ug": (170e-6, "Mai et al. (2020), median for riverine microplastics"),
    "230ug": (230e-6, "Weiss et al. (2021), median for non-fibre microplastics in plankton nets"),
}


def main() -> None:
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    lab_df, studies = load_data("all"), load_field_studies()
    rivers = pd.read_csv(RIVER_DATA_FILE)
    # Which inputs are measured and which are defaults (medians of data/global_dissolved_cu.csv and
    # data/global_mp_abundance.csv) is recorded in data/river_input_sources.csv.
    flags = pd.read_csv(Path(RIVER_DATA_FILE).with_name("river_input_sources.csv")).set_index("river")
    polymer_lists = [_normalize_mp_type(t) for t in rivers["MP_type"]]
    curves = polymer_curves(REFERENCE_MODEL, lab_df, sorted({p for ps in polymer_lists for p in ps}))
    faf = study_level_faf(study_ratios(REFERENCE_MODEL, lab_df, studies), PRIMARY_SUBSET)
    faf_mid, faf_lo, faf_hi = float(np.exp(np.log(faf).mean())), float(faf.min()), float(faf.max())   # geometric mean; range of studies

    rows = []
    for (_, r), polymers in zip(rivers.iterrows(), polymer_lists):
        qe_lab = float(lab_curve(curves, polymers, [1.0] * len(polymers))(np.array([r["Ce_mg_L"]]))[0])
        mp_measured = flags.loc[r["river"], "MP_source"] == "measured"
        ce_measured = flags.loc[r["river"], "Ce_source"] == "measured"
        dissolved_kg = r["Q_m3_yr"] * r["Ce_mg_L"] / 1e3
        row = {"river": r["river"], "country": r["country"], "polymers": "/".join(polymers),
               "MP_items_m3": r["MP_items_m3"], "MP_source": "measured" if mp_measured else "global default",
               "Ce_ug_L": r["Ce_mg_L"] * 1e3, "Ce_source": "measured" if ce_measured else "global default",
               "Q_km3_yr": r["Q_m3_yr"] / 1e9, "qe_lab_ug_g": qe_lab * 1e3, "qe_river_ug_g": qe_lab * faf_mid * 1e3,
               "qe_river_low_ug_g": qe_lab * faf_lo * 1e3, "qe_river_high_ug_g": qe_lab * faf_hi * 1e3,
               "dissolved_Cu_t_yr": dissolved_kg / 1e3}
        for label, (mass, _) in PARTICLE_MASS_CASES.items():
            base = r["Q_m3_yr"] * r["MP_items_m3"] * mass * qe_lab / 1e6
            row[f"Cu_kg_yr_{label}"] = base * faf_mid
            row[f"Cu_kg_yr_{label}_low"] = base * faf_lo
            row[f"Cu_kg_yr_{label}_high"] = base * faf_hi
        row["pct_of_dissolved_12.5ug"] = row["Cu_kg_yr_12.5ug"] / dissolved_kg * 100
        row["pct_of_dissolved_230ug"] = row["Cu_kg_yr_230ug"] / dissolved_kg * 100
        rows.append(row)
    table = pd.DataFrame(rows)
    both = (table["MP_source"] == "measured") & (table["Ce_source"] == "measured")
    main_table = table[both].sort_values("Cu_kg_yr_12.5ug", ascending=False)
    table["in_main_table"] = both
    main_table.to_csv(REPORTS_DIR / "river_application_main.csv", index=False)
    table.sort_values(["in_main_table", "Cu_kg_yr_12.5ug"], ascending=False).to_csv(
        REPORTS_DIR / "river_application_SI.csv", index=False)

    fmt = lambda v: f"{v:.3g}"  # noqa: E731
    print(f"Laboratory model: {REFERENCE_MODEL}; field-to-laboratory ratio: geometric mean {faf_mid:.3g} "
          f"(range {faf_lo:.3g}-{faf_hi:.3g}, {len(faf)} field studies)")
    for label, (mass, ref) in PARTICLE_MASS_CASES.items():
        print(f"  particle mass {label}: {ref}")
    print(f"\nMain table: {len(main_table)} rivers with measured MP abundance and measured dissolved Cu")
    print(main_table[["river", "MP_items_m3", "Ce_ug_L", "Q_km3_yr", "qe_river_ug_g", "Cu_kg_yr_12.5ug",
                      "Cu_kg_yr_12.5ug_low", "Cu_kg_yr_12.5ug_high", "Cu_kg_yr_170ug", "Cu_kg_yr_230ug",
                      "dissolved_Cu_t_yr", "pct_of_dissolved_12.5ug", "pct_of_dissolved_230ug"]].to_string(index=False, float_format=fmt))
    rest = table[~both]
    print(f"\nSupplementary table: {len(rest)} further rivers; measured MP only in "
          f"{int((rest['MP_source'] == 'measured').sum())}, both inputs default in "
          f"{int(((rest['MP_source'] != 'measured') & (rest['Ce_source'] != 'measured')).sum())}")
    print(rest[["river", "MP_source", "Ce_source", "Cu_kg_yr_12.5ug", "Cu_kg_yr_230ug"]].to_string(index=False, float_format=fmt))


if __name__ == "__main__":
    main()
