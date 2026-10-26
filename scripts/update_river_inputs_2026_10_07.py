"""One-off update of data/river_data.csv and data/river_input_sources.csv (2026-10-07).

1. Xi, Dong and Zhujiang (three catchments of Lebreton et al., 2017, all draining to the Pearl River
   delta) are merged into one "Pearl River system" row. Discharge is the sum of the three; the MP
   abundance (Wu et al., 2025) and dissolved Cu (Niu et al., 2021) were measured for the Pearl basin
   and estuary and were previously attached to the small Zhujiang sub-catchment only.
2. Solo and Hanjiang: the MP abundances came from other rivers (Surabaya River, a Brantas distributary;
   Han River in Hubei, a Yangtze tributary). Both are reset to the global default inputs.
"""

from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RIVERS, SOURCES = ROOT / "data" / "river_data.csv", ROOT / "data" / "river_input_sources.csv"
PEARL_PARTS = ["Xi", "Dong", "Zhujiang (Pearl)"]
PEARL = "Pearl River system (Xi, Dong, Zhujiang)"
DEFAULT_MP_ITEMS, DEFAULT_MP_MASS, DEFAULT_TYPE = 480.0, 6.0, "½ PE and ½  PP"

rivers = pd.read_csv(RIVERS)
if PEARL in set(rivers["river"]):
    raise SystemExit("river_data.csv has already been updated.")
parts = rivers[rivers["river"].isin(PEARL_PARTS)]
assert len(parts) == 3
zhu = parts[parts["river"] == "Zhujiang (Pearl)"].iloc[0].copy()
zhu["river"], zhu["Q_m3_yr"] = PEARL, parts["Q_m3_yr"].sum()
first = rivers.index[rivers["river"] == "Xi"][0]
rivers.loc[first] = zhu
rivers = rivers[~rivers["river"].isin(PEARL_PARTS)]
for old, new in [("(Surabaya) Solo", "Solo"), ("Hanjiang", "Hanjiang")]:
    m = rivers["river"] == old
    assert m.sum() == 1
    rivers.loc[m, ["river", "MP_type", "MP_items_m3", "MP_mass_mg_m3"]] = [new, DEFAULT_TYPE, DEFAULT_MP_ITEMS, DEFAULT_MP_MASS]
rivers.to_csv(RIVERS, index=False)

src = pd.read_csv(SOURCES)
zrow = src[src["river"] == "Zhujiang (Pearl)"].iloc[0].copy()
zrow["river"] = PEARL
zrow["Q_source"] = "Lebreton et al. (2017) Table 1: sum of Xi (5530), Dong (854) and Zhujiang (133 m3/s)"
zrow["note"] = "Three Lebreton catchments merged on 2026-10-07; MP and Cu sources refer to the Pearl basin and estuary"
src = src[~src["river"].isin(PEARL_PARTS)]
src = pd.concat([src.iloc[:3], zrow.to_frame().T, src.iloc[3:]], ignore_index=True)
for old, new, why in [("(Surabaya) Solo", "Solo", "Lestari et al. (2020) measured the Surabaya River, a distributary of the Brantas"),
                      ("Hanjiang", "Hanjiang", "Chen et al. (2023) measured the Han River in Hubei, a Yangtze tributary")]:
    m = src["river"] == old
    src.loc[m, ["river", "MP_source", "MP_reference", "check_status", "note"]] = [
        new, "global default", "Tan et al. (2023) median", "", f"MP reset to default on 2026-10-07: {why}"]
src.to_csv(SOURCES, index=False)
print(f"{len(rivers)} rivers; Pearl system Q = {zhu['Q_m3_yr'] / 1e9:.1f} km3/yr")
print(rivers[["river", "MP_type", "MP_items_m3", "Ce_mg_L", "Q_m3_yr"]].to_string(index=False))
