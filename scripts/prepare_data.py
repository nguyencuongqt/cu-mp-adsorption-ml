"""Prepare required CSV inputs for reproducible project runs.

This script writes the three small curated CSV files needed by the pipeline:
field calibration rows from Table S1, top-20 river rows from Table 2, and 30
evaluation cases for the LLM Interpretation System ablation study.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from config import DATA_DIR, REPORTS_DIR  # noqa: E402


FIELD_ROWS = [{'AgS': 'Aged', 'ReT': 'PP', 'Resin_pct': 37.0, 'Temp': 25, 'pH': 7, 'rpm': 167, 'AdC': 'Mixed', 'Ce': 0.15776, 'modeled_qe': 0.39, 'lo_qe': 0.07, 'hi_qe': 0.8, 'real_qe': 0.000341, 'reference': '(Zhang et al., 2025a)'}, {'AgS': 'Aged', 'ReT': 'PE', 'Resin_pct': 24.0, 'Temp': 25, 'pH': 7, 'rpm': 167, 'AdC': 'Mixed', 'Ce': 0.15776, 'modeled_qe': 0.33, 'lo_qe': 0.05, 'hi_qe': 0.87, 'real_qe': 0.000341, 'reference': '(Zhang et al., 2025a)'}, {'AgS': 'Aged', 'ReT': 'PET', 'Resin_pct': 18.0, 'Temp': 25, 'pH': 7, 'rpm': 167, 'AdC': 'Mixed', 'Ce': 0.15776, 'modeled_qe': 0.3, 'lo_qe': 0.05, 'hi_qe': 0.64, 'real_qe': 0.000341, 'reference': '(Zhang et al., 2025a)'}, {'AgS': 'Aged', 'ReT': 'PS', 'Resin_pct': 17.0, 'Temp': 25, 'pH': 7, 'rpm': 167, 'AdC': 'Mixed', 'Ce': 0.15776, 'modeled_qe': 0.32, 'lo_qe': 0.06, 'hi_qe': 0.64, 'real_qe': 0.000341, 'reference': '(Zhang et al., 2025a)'}, {'AgS': 'Aged', 'ReT': 'PET', 'Resin_pct': 100.0, 'Temp': 25, 'pH': 7, 'rpm': 167, 'AdC': 'Mixed', 'Ce': 0.000218, 'modeled_qe': 0.34, 'lo_qe': 0.01, 'hi_qe': 3.31, 'real_qe': 0.0002905, 'reference': '(Zhang et al., 2025b)'}, {'AgS': 'Aged', 'ReT': 'PP', 'Resin_pct': 52.22, 'Temp': 25, 'pH': 7, 'rpm': 167, 'AdC': 'Mixed', 'Ce': 0.04, 'modeled_qe': 0.29, 'lo_qe': 0.08, 'hi_qe': 0.75, 'real_qe': 0.0005, 'reference': '(Ta & Babel, 2023)'}, {'AgS': 'Aged', 'ReT': 'PE', 'Resin_pct': 21.0, 'Temp': 25, 'pH': 7, 'rpm': 167, 'AdC': 'Mixed', 'Ce': 0.04, 'modeled_qe': 0.23, 'lo_qe': 0.06, 'hi_qe': 0.87, 'real_qe': 0.0005, 'reference': '(Ta & Babel, 2023)'}, {'AgS': 'Aged', 'ReT': 'PS', 'Resin_pct': 19.0, 'Temp': 25, 'pH': 7, 'rpm': 167, 'AdC': 'Mixed', 'Ce': 0.04, 'modeled_qe': 0.21, 'lo_qe': 0.07, 'hi_qe': 0.54, 'real_qe': 0.0005, 'reference': '(Ta & Babel, 2023)'}, {'AgS': 'Aged', 'ReT': 'PE', 'Resin_pct': 50.0, 'Temp': 25, 'pH': 7, 'rpm': 167, 'AdC': 'Mixed', 'Ce': 0.005, 'modeled_qe': 0.25, 'lo_qe': 0.07, 'hi_qe': 1.11, 'real_qe': 0.0145, 'reference': '(Ta & Babel, 2020)'}, {'AgS': 'Aged', 'ReT': 'PP', 'Resin_pct': 21.0, 'Temp': 25, 'pH': 7, 'rpm': 167, 'AdC': 'Mixed', 'Ce': 0.005, 'modeled_qe': 0.3, 'lo_qe': 0.08, 'hi_qe': 0.75, 'real_qe': 0.0145, 'reference': '(Ta & Babel, 2020)'}, {'AgS': 'Aged', 'ReT': 'PS', 'Resin_pct': 7.0, 'Temp': 25, 'pH': 7, 'rpm': 167, 'AdC': 'Mixed', 'Ce': 0.005, 'modeled_qe': 0.23, 'lo_qe': 0.08, 'hi_qe': 0.87, 'real_qe': 0.0145, 'reference': '(Ta & Babel, 2020)'}, {'AgS': 'Aged', 'ReT': 'PP', 'Resin_pct': 36.6, 'Temp': 25, 'pH': 7, 'rpm': 167, 'AdC': 'Mixed', 'Ce': 0.0138, 'modeled_qe': 0.29, 'lo_qe': 0.08, 'hi_qe': 0.75, 'real_qe': 9.1e-05, 'reference': '(Purwiyanto et al., 2020)'}, {'AgS': 'Aged', 'ReT': 'PE', 'Resin_pct': 27.7, 'Temp': 25, 'pH': 7, 'rpm': 167, 'AdC': 'Mixed', 'Ce': 0.0138, 'modeled_qe': 0.23, 'lo_qe': 0.07, 'hi_qe': 0.87, 'real_qe': 9.1e-05, 'reference': '(Purwiyanto et al., 2020)'}, {'AgS': 'Aged', 'ReT': 'PVC', 'Resin_pct': 9.09, 'Temp': 25, 'pH': 7, 'rpm': 167, 'AdC': 'Mixed', 'Ce': 0.0138, 'modeled_qe': 0.2, 'lo_qe': 0.08, 'hi_qe': 0.51, 'real_qe': 9.1e-05, 'reference': '(Purwiyanto et al., 2020)'}]

RIVER_ROWS = [{'river': 'Yangtze', 'country': 'China', 'MP_type': 'PP', 'MP_items_m3': 2904.12, 'MP_mass_mg_m3': 36.3015, 'Ce_mg_L': 0.00305, 'Q_m3_yr': 498000000000, 'reference': 'Xuan Cuong Nguyen (DTU). Version: 1.3.0'}, {'river': 'Ganges', 'country': 'India, Bangladesh', 'MP_type': '½ PE and ½  PP', 'MP_items_m3': 16.775, 'MP_mass_mg_m3': 0.2096875, 'Ce_mg_L': 2.54, 'Q_m3_yr': 656000000000, 'reference': 'Xuan Cuong Nguyen (DTU). Version: 1.3.0'}, {'river': 'Xi', 'country': 'China', 'MP_type': '½ PE and ½  PP', 'MP_items_m3': 480.0, 'MP_mass_mg_m3': 6.0, 'Ce_mg_L': 0.00877, 'Q_m3_yr': 174000000000, 'reference': 'Xuan Cuong Nguyen (DTU). Version: 1.3.0'}, {'river': 'Huangpu', 'country': 'China', 'MP_type': 'PP', 'MP_items_m3': 1666.7, 'MP_mass_mg_m3': 20.833750000000002, 'Ce_mg_L': 0.00877, 'Q_m3_yr': 12700000000, 'reference': 'Xuan Cuong Nguyen (DTU). Version: 1.3.0'}, {'river': 'Cross', 'country': 'Nigeri, Cameroon', 'MP_type': '½ PE and ½  PP', 'MP_items_m3': 480.0, 'MP_mass_mg_m3': 6.0, 'Ce_mg_L': 0.00877, 'Q_m3_yr': 7570000000, 'reference': 'Xuan Cuong Nguyen (DTU). Version: 1.3.0'}, {'river': 'Brantas', 'country': 'Indonesia', 'MP_type': '½ PE and ½  PP', 'MP_items_m3': 3467.41, 'MP_mass_mg_m3': 43.342625, 'Ce_mg_L': 0.00877, 'Q_m3_yr': 25800000000, 'reference': 'Xuan Cuong Nguyen (DTU). Version: 1.3.0'}, {'river': 'Amazon', 'country': 'Brazil, Peru, Colombia, Ecuador', 'MP_type': '½ PE and ½  PP', 'MP_items_m3': 57.0, 'MP_mass_mg_m3': 0.7125, 'Ce_mg_L': 0.00248, 'Q_m3_yr': 4410000000000, 'reference': 'Xuan Cuong Nguyen (DTU). Version: 1.3.0'}, {'river': 'Pasig', 'country': 'Philippines', 'MP_type': '½ PE and ½  PP', 'MP_items_m3': 480.0, 'MP_mass_mg_m3': 6.0, 'Ce_mg_L': 0.00877, 'Q_m3_yr': 6530000000, 'reference': 'Xuan Cuong Nguyen (DTU). Version: 1.3.0'}, {'river': 'Irrawaddy', 'country': 'Myanmar', 'MP_type': '½ PE and ½  PP', 'MP_items_m3': 480.0, 'MP_mass_mg_m3': 6.0, 'Ce_mg_L': 0.00877, 'Q_m3_yr': 173000000000, 'reference': 'Xuan Cuong Nguyen (DTU). Version: 1.3.0'}, {'river': '(Surabaya) Solo', 'country': 'Indonesia', 'MP_type': '½ PE and ½  PP', 'MP_items_m3': 13.16, 'MP_mass_mg_m3': 0.1645, 'Ce_mg_L': 0.00877, 'Q_m3_yr': 23500000000, 'reference': 'Xuan Cuong Nguyen (DTU). Version: 1.3.0'}, {'river': 'Mekong', 'country': 'Thailand, Cambodia, Laos, China, Myanmar, Vietnam', 'MP_type': 'PET', 'MP_items_m3': 21.5, 'MP_mass_mg_m3': 0.26875, 'Ce_mg_L': 0.00877, 'Q_m3_yr': 190000000000, 'reference': 'Xuan Cuong Nguyen (DTU). Version: 1.3.0'}, {'river': 'Imo', 'country': 'Nigeria', 'MP_type': '½ PE and ½  PP', 'MP_items_m3': 480.0, 'MP_mass_mg_m3': 6.0, 'Ce_mg_L': 0.00877, 'Q_m3_yr': 8800000000, 'reference': 'Xuan Cuong Nguyen (DTU). Version: 1.3.0'}, {'river': 'Dong', 'country': 'China', 'MP_type': '½ PE and ½  PP', 'MP_items_m3': 480.0, 'MP_mass_mg_m3': 6.0, 'Ce_mg_L': 0.00877, 'Q_m3_yr': 26900000000, 'reference': 'Xuan Cuong Nguyen (DTU). Version: 1.3.0'}, {'river': 'Serayu', 'country': 'Indonesia', 'MP_type': '½ PE and ½  PP', 'MP_items_m3': 480.0, 'MP_mass_mg_m3': 6.0, 'Ce_mg_L': 0.00877, 'Q_m3_yr': 11700000000, 'reference': 'Xuan Cuong Nguyen (DTU). Version: 1.3.0'}, {'river': 'Magdalena', 'country': 'Colombia', 'MP_type': '½ PE and ½  PP', 'MP_items_m3': 480.0, 'MP_mass_mg_m3': 6.0, 'Ce_mg_L': 0.00877, 'Q_m3_yr': 187000000000, 'reference': 'Xuan Cuong Nguyen (DTU). Version: 1.3.0'}, {'river': 'Tamsui', 'country': 'Taiwan', 'MP_type': '½ PE and ½  PP', 'MP_items_m3': 25.4, 'MP_mass_mg_m3': 0.3175, 'Ce_mg_L': 0.00877, 'Q_m3_yr': 3410000000, 'reference': 'Xuan Cuong Nguyen (DTU). Version: 1.3.0'}, {'river': 'Zhujiang (Pearl)', 'country': 'China', 'MP_type': 'PET', 'MP_items_m3': 1000.0, 'MP_mass_mg_m3': 12.5, 'Ce_mg_L': 0.00100427, 'Q_m3_yr': 4200000000, 'reference': 'Xuan Cuong Nguyen (DTU). Version: 1.3.0'}, {'river': 'Hanjiang', 'country': 'China', 'MP_type': 'PET', 'MP_items_m3': 3579.44, 'MP_mass_mg_m3': 44.743, 'Ce_mg_L': 0.00877, 'Q_m3_yr': 23200000000, 'reference': 'Xuan Cuong Nguyen (DTU). Version: 1.3.0'}, {'river': 'Progo', 'country': 'Indonesia', 'MP_type': 'PE', 'MP_items_m3': 167.91, 'MP_mass_mg_m3': 2.098875, 'Ce_mg_L': 0.00877, 'Q_m3_yr': 8800000000, 'reference': 'Xuan Cuong Nguyen (DTU). Version: 1.3.0'}, {'river': 'Kwa Ibo', 'country': 'Nigeria', 'MP_type': '½ PE and ½  PP', 'MP_items_m3': 480.0, 'MP_mass_mg_m3': 6.0, 'Ce_mg_L': 0.00877, 'Q_m3_yr': 6060000000, 'reference': 'Xuan Cuong Nguyen (DTU). Version: 1.3.0'}]


def _load_dataset() -> pd.DataFrame:
    path = DATA_DIR / "dataset.csv"
    if not path.exists():
        legacy = ROOT.parent / "Cu_data.xlsx"
        if legacy.exists():
            df = pd.read_excel(legacy)
            df.to_csv(path, index=False, encoding="utf-8-sig")
            return df
        raise FileNotFoundError("data/dataset.csv not found. Copy your lab dataset there first.")
    return pd.read_csv(path)


def _infer_mechanism(row: pd.Series) -> str | None:
    ret = str(row["ReT"]).upper()
    ags = str(row["AgS"]).lower()
    ce = float(row["Ce"])
    if ags == "aged" and ret == "PP":
        return "UV aging + functional groups"
    if ags == "aged" and ret == "PLA":
        return "biodegradation"
    if ags == "virgin" and ce > 50:
        return "surface saturation"
    return None


def _generate_test_cases() -> pd.DataFrame:
    df = _load_dataset()
    required = ["ReT", "AgS", "AdC", "Temp", "pH", "rpm", "Ce", "qe"]
    missing = [col for col in required if col not in df.columns]
    if missing:
        raise KeyError(f"dataset.csv missing required columns: {missing}")
    for col in ["Temp", "pH", "rpm", "Ce", "qe"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df.dropna(subset=required).copy()
    df["known_mechanism"] = df.apply(_infer_mechanism, axis=1)
    candidates = df.dropna(subset=["known_mechanism"]).copy()
    if candidates.empty:
        raise ValueError("No dataset rows matched the mechanism rules for test case generation.")
    selected = candidates.groupby("known_mechanism", group_keys=False).apply(
        lambda g: g.sort_values("Ce").iloc[np.linspace(0, len(g) - 1, min(len(g), 10), dtype=int)]
    )
    if len(selected) < 30:
        extras = candidates.loc[~candidates.index.isin(selected.index)].sort_values(["ReT", "Ce"])
        selected = pd.concat([selected, extras], axis=0)
    selected = selected.drop_duplicates().head(30)
    if len(selected) < 30:
        selected = pd.concat([selected, selected.sample(30 - len(selected), replace=True, random_state=42)])

    rows = []
    for _, row in selected.reset_index(drop=True).iterrows():
        input_dict = {
            "ReT": str(row["ReT"]),
            "AgS": str(row["AgS"]),
            "AdC": str(row["AdC"]),
            "Temp": float(row["Temp"]),
            "pH": float(row["pH"]),
            "rpm": float(row["rpm"]),
            "Ce": float(row["Ce"]),
        }
        rows.append(
            {
                "input_ReT": input_dict["ReT"],
                "input_AgS": input_dict["AgS"],
                "input_AdC": input_dict["AdC"],
                "input_Temp": input_dict["Temp"],
                "input_pH": input_dict["pH"],
                "input_rpm": input_dict["rpm"],
                "input_Ce": input_dict["Ce"],
                "known_mechanism": row["known_mechanism"],
                "known_top_features": "Ce,pH,ReT",
                "input_dict": json.dumps(input_dict),
                "selection_rationale": (
                    "Selected because it matches a literature-inferred mechanism rule: "
                    f"{row['known_mechanism']}."
                ),
            }
        )
    return pd.DataFrame(rows)


def main() -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    field = pd.DataFrame(FIELD_ROWS)
    river = pd.DataFrame(RIVER_ROWS)
    test_cases = _generate_test_cases()

    field.to_csv(DATA_DIR / "field_data.csv", index=False, encoding="utf-8-sig")
    field.rename(columns={"Resin_pct": "Resin_percentage"}).to_csv(
        REPORTS_DIR / "field_data.csv", index=False, encoding="utf-8-sig"
    )
    river.to_csv(DATA_DIR / "river_data.csv", index=False, encoding="utf-8-sig")
    test_cases.to_csv(DATA_DIR / "test_cases.csv", index=False, encoding="utf-8-sig")

    print(
        "Created field_data.csv (14 rows), river_data.csv (20 rows), "
        "test_cases.csv (30 rows). Ready to run main.py."
    )


if __name__ == "__main__":
    main()
