"""QC of the digitized Cu isotherms of Turner & Holmes (2015), Environ. Chem., doi:10.1071/EN14143.

Input : Revision/06_Data_audit/Digitization_field_search_2026-10/16_EnvChem_2015_Turner_Holmes_PE_pellets/
        Turner2015_Cu_digitization.xlsx, sheet Digitize (Ce_digital = [Me] in ug/L, qe_digital = [MeX]* in ug/g)
Output: QC_Turner_2015.csv            one row per point, with checks and keep = yes/no
        rows_for_master_Turner_2015.csv rows in the format of data/dataset_grouped.csv (Ce mg/L, qe mg/g)

Checks
1. Readings present and inside the axis ranges (x 0-20 ug/L; y 0-0.08 ug/g virgin, 0-0.80 ug/g beached).
2. x increases with the nominal C0 (2, 5, 10, 15, 20 ug/L).
3. y agrees with the authors' Freundlich fit at the same x (Table 7): ratio 0.6-1.6, otherwise flagged.
4. Mass balance: x + dose * y (ug/L) should not exceed 1.15 x nominal C0 + 1 ug/L (Cu recovery 90-100 %;
   the 1 ug/L allows for Cu already present in the river water, < 10 % of the spiked pool).
5. Freundlich refit of the digitized points compared with the reported KF and 1/n.
Points are not dropped automatically for checks 3-5; flagged points are reviewed by hand.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
FOLDER = (ROOT / "Manuscript" / "Submission_Env pollution" / "Revision" / "06_Data_audit"
          / "Digitization_field_search_2026-10" / "16_EnvChem_2015_Turner_Holmes_PE_pellets")
YMAX = {"PE_virgin": 0.08, "PE_beached": 0.80}
FREUNDLICH = {"PE_virgin": (0.00739, 0.715), "PE_beached": (0.0568, 1.04)}  # ug/g at [Me] in ug/L
DOSE = {"PE_virgin": 10.4, "PE_beached": 11.6}  # g/L (20 pellets in 50 mL)
AGEING = {"PE_virgin": ("Virgin", "none"), "PE_beached": ("Aged", "outdoor_weathering")}


def main(folder: Path = FOLDER) -> None:
    raw = pd.read_excel(folder / "Turner2015_Cu_digitization.xlsx", sheet_name="Digitize")
    d = pd.DataFrame({
        "point_id": raw.iloc[:, 0], "series_id": raw.iloc[:, 1],
        "c0_nominal_ug_L": pd.to_numeric(raw.iloc[:, 3], errors="coerce"),
        "point_rank_in_series": raw.iloc[:, 0].str[1:].astype(int),
        "x_read": pd.to_numeric(raw.iloc[:, 4], errors="coerce"),
        "y_read": pd.to_numeric(raw.iloc[:, 5], errors="coerce"),
    })
    d["mp_dose_g_L"] = d["series_id"].map(DOSE)
    rows = []
    for _, r in d.iterrows():
        flags = []
        x, y, s = r["x_read"], r["y_read"], r["series_id"]
        if np.isnan(x) or np.isnan(y):
            rows.append({**r.to_dict(), "flags": "not read", "keep": "no"})
            continue
        if not (0 < x <= 20.5):
            flags.append("x outside 0-20 ug/L")
        if not (0 < y <= YMAX[s] * 1.02):
            flags.append(f"y outside 0-{YMAX[s]} ug/g")
        kf, inv_n = FREUNDLICH[s]
        y_f = kf * x ** inv_n
        ratio = y / y_f
        if not (0.6 <= ratio <= 1.6):
            flags.append(f"y/Freundlich = {ratio:.2f}")
        c0_back = x + r["mp_dose_g_L"] * y
        if c0_back > 1.15 * r["c0_nominal_ug_L"] + 1.0:  # + 1 ug/L for Cu already in the river water
            flags.append(f"mass balance {c0_back:.1f} > 1.15 x C0 + 1 ug/L")
        rows.append({**r.to_dict(), "ce_mg_L": x / 1000.0, "qe_mg_g": y / 1000.0, "qe_freundlich_reported_ug_g": y_f,
                     "ratio_to_freundlich": ratio, "c0_back_ug_L": c0_back, "flags": "; ".join(flags),
                     "keep": "yes" if not flags else "review"})
    qc = pd.DataFrame(rows)
    for s, part in qc[qc["keep"] != "no"].groupby("series_id"):
        xs = part["x_read"].to_numpy(float)
        if len(xs) > 1 and np.any(np.diff(xs[np.argsort(part["point_rank_in_series"].to_numpy())]) <= 0):
            qc.loc[part.index, "flags"] = qc.loc[part.index, "flags"].astype(str) + "; x not increasing with C0"
            qc.loc[part.index, "keep"] = "review"
        if len(part) >= 3:
            slope, intercept = np.polyfit(np.log10(xs), np.log10(part["y_read"].to_numpy(float)), 1)
            kf, inv_n = FREUNDLICH[s]
            print(f"{s}: refit KF = {10 ** intercept:.4g} (reported {kf}), 1/n = {slope:.3f} (reported {inv_n}); "
                  f"n points = {len(part)}")
    qc.to_csv(folder / "QC_Turner_2015.csv", index=False)

    keep = qc[qc["keep"] == "yes"]
    master = pd.DataFrame({
        "source_id": "Turner_2015",
        "curve_id": keep["series_id"].map({"PE_virgin": "Turner_2015_01", "PE_beached": "Turner_2015_02"}),
        "series_label": keep["series_id"].map({"PE_virgin": "PE | virgin pellets", "PE_beached": "PE | beached pellets"}),
        "ageing_type": keep["series_id"].map(lambda s: AGEING[s][1]),
        "matrix": "freshwater", "year": 2015, "doi": "10.1071/EN14143",
        "AgS": keep["series_id"].map(lambda s: AGEING[s][0]), "ReT": "PE", "AdC": "Mixed",
        "Temp": 19.0, "pH": 6.4, "rpm": 150.0, "Ce": keep["ce_mg_L"], "qe": keep["qe_mg_g"],
    })
    master.to_csv(folder / "rows_for_master_Turner_2015.csv", index=False)
    print(qc[["point_id", "series_id", "x_read", "y_read", "ratio_to_freundlich", "c0_back_ug_L", "flags", "keep"]]
          .to_string(index=False))
    print(f"\n{len(master)} points written to rows_for_master_Turner_2015.csv "
          f"({(qc['keep'] == 'review').sum()} to review, {(qc['keep'] == 'no').sum()} not read)")


if __name__ == "__main__":
    main(Path(sys.argv[1]) if len(sys.argv) > 1 else FOLDER)
