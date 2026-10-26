"""Build the raw global input tables used by src/analysis/global_monte_carlo.py.

Sources (kept in Manuscript/Submission_Env pollution/Revision/new papers):
- Jin et al. (2025), Commun. Earth Environ. 6, 520; data: figshare 10.6084/m9.figshare.29338148.v1
  (HMsdata.csv). Kept: type = river, D_T = Dissolved, Cu > 0. Cu is in ug/L in the source.
- Tan et al. (2023), Sci. Total Environ. 904, 165979; Supplementary data mmc2.xlsx, sheet "Water PLIi"
  (abundance in items/m3 with the size range of each study). Dropped: canals, wastewater effluent
  outlets and zero abundances.

Outputs: data/global_dissolved_cu.csv (column cu_mg_l) and data/global_mp_abundance.csv (column items_m3).
"""

from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "Manuscript" / "Submission_Env pollution" / "Revision" / "07_Papers" / "new papers"
JIN_FILE = SRC / "Jin_2025_Heavymetal_DATA" / "Heavymetal_DATA" / "data" / "HMsdata.csv"
TAN_FILE = SRC / "Tan_2023_extracted" / "1-s2.0-S0048969723046041-mmc2.xlsx"
NOT_RIVER = ("canal", "wastewater")


def build_cu() -> pd.DataFrame:
    hm = pd.read_csv(JIN_FILE, encoding="latin1", low_memory=False)
    hm["Cu"] = pd.to_numeric(hm["Cu"], errors="coerce")
    keep = (hm["type"].astype(str).str.strip().str.lower() == "river") & (hm["D_T"].astype(str).str.strip() == "Dissolved")
    out = hm[keep & (hm["Cu"] > 0)].copy()
    out["cu_mg_l"] = out["Cu"] / 1000.0
    return out[["cu_mg_l", "Reference", "Citation", "Year", "landuse"]].rename(
        columns={"Reference": "reference_id", "Citation": "citation", "Year": "year"})


def build_mp() -> pd.DataFrame:
    p = pd.read_excel(TAN_FILE, sheet_name="Water PLIi")
    p.columns = ["river", "sample", "items_m3", "size_max_um", "size_min_um", "alpha", "CF",
                 "items_m3_rescaled_by_Tan", "background", "CFi", "PLIi"]
    p["items_m3"] = pd.to_numeric(p["items_m3"], errors="coerce")
    p["river"] = p["river"].astype(str).str.strip()
    is_river = ~p["river"].str.lower().str.contains("|".join(NOT_RIVER))
    out = p[is_river & (p["items_m3"] > 0)]
    return out[["items_m3", "river", "sample", "size_min_um", "size_max_um", "items_m3_rescaled_by_Tan"]]


def main() -> None:
    cu, mp = build_cu(), build_mp()
    cu.to_csv(ROOT / "data" / "global_dissolved_cu.csv", index=False)
    mp.to_csv(ROOT / "data" / "global_mp_abundance.csv", index=False)
    for name, s, unit in [("Dissolved Cu", cu["cu_mg_l"] * 1000, "ug/L"), ("MP abundance", mp["items_m3"], "items/m3")]:
        print(f"{name}: n = {len(s)}, median = {s.median():.4g}, mean = {s.mean():.4g}, "
              f"IQR = {s.quantile(0.75) - s.quantile(0.25):.4g} {unit}")
    print(f"Cu: {cu['reference_id'].nunique()} source publications; MP: {mp['river'].nunique()} river entries")


if __name__ == "__main__":
    main()
