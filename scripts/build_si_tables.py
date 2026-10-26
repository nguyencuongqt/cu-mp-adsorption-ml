"""Build the Supplementary Information tables of the revised manuscript.

Reads the curated data and the analysis outputs; writes CSV files and one Excel workbook to
Manuscript/Submission_Env pollution/Revision/SI_tables/.

- Table_S_lab_studies: one row per laboratory study (source of the isotherm dataset).
- Table_S_field_tier1: paired field records (Cu on MPs and Cu in water) with the laboratory
  prediction of the main model and the field-to-laboratory ratio.
- Table_S_field_tier2: field records with Cu on MPs only (supporting evidence).
- Table_S_field_excluded: field papers read in full and not used, with the reason.
- Table_S_river_inputs: river application, inputs flagged as measured or default, with sources.
"""

from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
REV = ROOT / "Manuscript" / "Submission_Env pollution" / "Revision"
OUT = REV / "03_SI_tables"
REPORTS = ROOT / "outputs" / "reports"
MAIN_MODEL = "freundlich_pooled"
PRIMARY_EXCLUDED = {"Purwiyanto_2020": "water Cu is unfiltered (total recoverable); sensitivity analysis only"}


def lab_studies() -> pd.DataFrame:
    s = pd.read_csv(REV / "Data_summary" / "studies_summary.csv")
    s = s.sort_values(["year", "source_id"]).reset_index(drop=True)
    s.insert(0, "No", np.arange(1, len(s) + 1))
    s["polymers"] = s["polymers"].astype(object)
    s.loc[s["source_id"] == "Lin_2022", "polymers"] = "PS (with 2% HBCD as additive)"
    return s.rename(columns={
        "reference": "Reference", "doi": "DOI", "n_points": "Data points", "n_series": "Isotherms",
        "polymers": "Polymers", "ageing_types": "Aging treatment", "systems": "Adsorption system",
        "Temp_C": "Temperature (°C)", "rpm": "Shaking (rpm)", "Ce_min_mg_L": "Ce min (mg/L)",
        "Ce_max_mg_L": "Ce max (mg/L)", "qe_min_mg_g": "qe min (mg/g)", "qe_max_mg_g": "qe max (mg/g)",
        "median_log10_Kd_L_kg": "Median log10 Kd (L/kg)"}).drop(columns=["year"])


def field_tables() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    tiers = pd.read_csv(ROOT / "data" / "field_evidence_tiers.csv", dtype={"tier": str})
    faf = pd.read_csv(REPORTS / "faf_scaling_by_study.csv")
    faf = faf[faf["lab_model"] == MAIN_MODEL]
    t1 = tiers[tiers["tier"] == "1"].copy()
    lab, ratio = [], []
    for r in t1.itertuples():
        hit = faf[np.isclose(faf["Ce_mg_L"], r.water_Cu_mg_L) & np.isclose(faf["qe_field_mg_g"], r.Cu_on_MP_ug_g / 1e3)]
        if len(hit) != 1:
            raise ValueError(f"{r.study_id}: expected one matching field record, found {len(hit)}")
        lab.append(hit["qe_lab_mg_g"].iloc[0] * 1e3)
        ratio.append(hit["faf"].iloc[0])
    t1["Laboratory prediction (µg/g)"], t1["Field/laboratory ratio"] = lab, ratio
    t1["Used in main estimate"] = ["no: " + PRIMARY_EXCLUDED[k] if (k := next((p for p in PRIMARY_EXCLUDED if s.startswith(p)), None))
                                   else "yes" for s in t1["study_id"]]
    common = {"reference": "Reference", "doi": "DOI", "site": "Site", "country": "Country", "water_body": "Water body",
              "sample_matrix": "Microplastic sample", "n_sites": "Sites or samples", "polymers": "Polymer composition",
              "metal_extraction": "Extraction and analysis", "Cu_on_MP_ug_g": "Cu on MPs (µg/g)",
              "Cu_on_MP_min_ug_g": "Cu on MPs, min (µg/g)", "Cu_on_MP_max_ug_g": "Cu on MPs, max (µg/g)",
              "Cu_on_MP_statistic": "Statistic", "notes": "Notes"}
    t1_out = t1.sort_values("Field/laboratory ratio", ascending=False).rename(columns={
        **common, "water_Cu_mg_L": "Cu in water (mg/L)", "water_Cu_fraction": "Water Cu fraction",
        "co_located_water_and_mp": "Pairing of water and MPs"})
    t1_out = t1_out[["Reference", "DOI", "Site", "Country", "Water body", "Microplastic sample", "Sites or samples",
                     "Polymer composition", "Extraction and analysis", "Cu on MPs (µg/g)", "Statistic", "Cu in water (mg/L)",
                     "Water Cu fraction", "Pairing of water and MPs", "Laboratory prediction (µg/g)",
                     "Field/laboratory ratio", "Used in main estimate", "Notes"]]
    t2 = tiers[tiers["tier"] == "2"].rename(columns=common)
    t2 = t2[["Reference", "DOI", "Site", "Country", "Water body", "Microplastic sample", "Sites or samples",
             "Polymer composition", "Extraction and analysis", "Cu on MPs (µg/g)", "Cu on MPs, min (µg/g)",
             "Cu on MPs, max (µg/g)", "Statistic", "Notes"]]
    ex = tiers[tiers["tier"] == "excluded"].rename(columns={**common, "notes": "Reason for exclusion"})
    ex = ex[["Reference", "DOI", "Site", "Country", "Water body", "Reason for exclusion"]]
    return t1_out, t2, ex


def river_inputs() -> pd.DataFrame:
    res = pd.read_csv(REPORTS / "river_application_SI.csv")
    src = pd.read_csv(ROOT / "data" / "river_input_sources.csv")
    m = res.merge(src[["river", "Q_source", "MP_reference", "Ce_reference", "note"]], on="river", how="left", validate="1:1")
    m["Table"] = np.where(m["in_main_table"], "main", "supplementary")
    return m.rename(columns={
        "river": "River", "country": "Country", "polymers": "Polymers assumed", "MP_items_m3": "MP abundance (items/m³)",
        "MP_source": "MP input", "MP_reference": "MP reference", "Ce_ug_L": "Dissolved Cu (µg/L)", "Ce_source": "Cu input",
        "Ce_reference": "Cu reference", "Q_km3_yr": "Discharge (km³/yr)", "Q_source": "Discharge reference",
        "qe_river_ug_g": "River qe (µg/g)", "qe_river_low_ug_g": "River qe, low (µg/g)", "qe_river_high_ug_g": "River qe, high (µg/g)",
        "Cu_kg_yr_12.5ug": "MP-bound Cu (kg/yr), 12.5 µg/item", "Cu_kg_yr_12.5ug_low": "low", "Cu_kg_yr_12.5ug_high": "high",
        "Cu_kg_yr_170ug": "MP-bound Cu (kg/yr), 170 µg/item", "Cu_kg_yr_230ug": "MP-bound Cu (kg/yr), 230 µg/item",
        "dissolved_Cu_t_yr": "Dissolved Cu flux (t/yr)", "pct_of_dissolved_12.5ug": "% of dissolved flux, 12.5 µg/item",
        "pct_of_dissolved_230ug": "% of dissolved flux, 230 µg/item", "note": "Notes"})[[
        "Table", "River", "Country", "Discharge (km³/yr)", "Discharge reference", "MP abundance (items/m³)", "MP input",
        "MP reference", "Polymers assumed", "Dissolved Cu (µg/L)", "Cu input", "Cu reference", "River qe (µg/g)",
        "River qe, low (µg/g)", "River qe, high (µg/g)", "MP-bound Cu (kg/yr), 12.5 µg/item", "low", "high",
        "MP-bound Cu (kg/yr), 170 µg/item", "MP-bound Cu (kg/yr), 230 µg/item", "Dissolved Cu flux (t/yr)",
        "% of dissolved flux, 12.5 µg/item", "% of dissolved flux, 230 µg/item", "Notes"]]


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    t1, t2, ex = field_tables()
    tables = {"Table_S_lab_studies": lab_studies(), "Table_S_field_tier1": t1, "Table_S_field_tier2": t2,
              "Table_S_field_excluded": ex, "Table_S_river_inputs": river_inputs()}
    with pd.ExcelWriter(OUT / "SI_tables.xlsx") as writer:
        for name, table in tables.items():
            table.to_csv(OUT / f"{name}.csv", index=False, encoding="utf-8-sig")
            table.to_excel(writer, sheet_name=name[8:], index=False)
            print(f"{name}: {table.shape[0]} rows, {table.shape[1]} columns")
    lab = tables["Table_S_lab_studies"]
    print(f"Laboratory: {lab['Data points'].sum()} points, {lab['Isotherms'].sum()} isotherms, {len(lab)} studies")
    print(t1[["Reference", "Site", "Cu on MPs (µg/g)", "Cu in water (mg/L)", "Laboratory prediction (µg/g)",
              "Field/laboratory ratio", "Used in main estimate"]].to_string(index=False, float_format=lambda v: f"{v:.3g}"))


if __name__ == "__main__":
    main()
