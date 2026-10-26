"""Create a row-level, read-only audit for the Sun_2022 and Chen_2021 CSV rows."""

from pathlib import Path
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
INPUT = ROOT / "data" / "dataset_grouped.csv"
OUTPUT = (
    ROOT
    / "Manuscript"
    / "Submission_Env pollution"
    / "Revision"
    / "06_Data_audit"
    / "Sun_Chen_row_level_CSV_audit.csv"
)


def nearest(values: np.ndarray, value: float) -> float:
    return float(values[np.abs(values - value).argmin()])


data = pd.read_csv(INPUT)
data["csv_line"] = data.index + 2  # CSV line 1 is the header
audit = data[data["source_id"].isin(["Sun_2022", "Chen_2021"])].copy()

# Independent visual reads of Sun Fig. 1d (PDF p. 4), ranked left-to-right.
# Values are intentionally coarse; the figure-reading uncertainty is about
# +/-0.2 mg/L for Ce and +/-30 ug/g for qe.
sun_figure = {
    "Virgin": [(0.1, 0.025), (0.7, 0.100), (1.8, 0.135), (5.0, 0.145), (8.0, 0.150), (11.0, 0.155)],
    "Aged": [(0.1, 0.450), (0.6, 0.730), (2.4, 1.050), (6.0, 1.250), (8.5, 1.150), (10.8, 1.190)],
}
sun_rank = {}
for state, rows_for_state in audit[audit["source_id"].eq("Sun_2022")].groupby("AgS"):
    for rank, idx in enumerate(rows_for_state.sort_values("csv_line").index):
        sun_rank[idx] = rank

settings = {
    "Sun_2022": {
        "dose_g_l": 2.0,
        "nominal_c0_mg_l": np.array([0.2, 1.0, 2.0, 5.0, 8.0, 15.0]),
        "reported_kd_range_l_kg": np.nan,
        "assessment": (
            "FAIL: incompatible with Sun Fig. 1d as stored. "
            "CSV Ce is 13-1553 mg/L, above the article's 0.2-15 mg/L C0 range."
        ),
    },
    "Chen_2021": {
        "dose_g_l": 0.5,
        "nominal_c0_mg_l": np.array([0.01, 0.1, 0.2, 0.5, 1.0, 2.0]),
        "reported_kd_range_l_kg": (331.2, 761.3),
        "assessment": (
            "FAIL as a Ce-qe isotherm row: Kd calculated from the stored values "
            "is outside the article's 331.2-761.3 L/kg MP range. "
            "Exact point-by-point source comparison requires missing Figure S3/Table S4."
        ),
    },
}

rows = []
for row in audit.itertuples():
    cfg = settings[row.source_id]
    ce = float(row.Ce)
    qe = float(row.qe)
    dose = cfg["dose_g_l"]
    inferred_c0 = ce + dose * qe
    nominal = nearest(cfg["nominal_c0_mg_l"], inferred_c0)
    kd = 1000.0 * qe / ce

    record = {
        "csv_line": row.csv_line,
        "source_id": row.source_id,
        "polymer": row.ReT,
        "stored_state": row.AgS,
        "csv_Ce_mg_L": ce,
        "csv_qe_mg_g": qe,
        "dose_g_L_from_methods": dose,
        "inferred_C0_mg_L": inferred_c0,
        "nearest_nominal_C0_mg_L": nominal,
        "inferred_to_nearest_C0_ratio": inferred_c0 / nominal,
        "Kd_from_CSV_L_kg": kd,
        "row_assessment": cfg["assessment"],
    }
    if row.source_id == "Sun_2022":
        fig_ce, fig_qe = sun_figure[row.AgS][sun_rank[row.Index]]
        record["source_coordinate_comparison"] = "No match to independently read Sun Fig. 1d point"
        record["source_Fig1d_Ce_est_mg_L"] = fig_ce
        record["source_Fig1d_qe_est_mg_g"] = fig_qe
        record["CSV_to_Fig1d_Ce_fold"] = ce / fig_ce
        record["CSV_to_Fig1d_qe_fold"] = qe / fig_qe
    else:
        record["source_coordinate_comparison"] = (
            "Not determinable exactly: Chen Figure S3/Table S4 are missing; row fails Kd test"
        )
        record["source_Fig1d_Ce_est_mg_L"] = np.nan
        record["source_Fig1d_qe_est_mg_g"] = np.nan
        record["CSV_to_Fig1d_Ce_fold"] = np.nan
        record["CSV_to_Fig1d_qe_fold"] = np.nan
    if row.source_id == "Chen_2021":
        lower, upper = cfg["reported_kd_range_l_kg"]
        record["Kd_vs_reported_lower_fold"] = kd / lower
        record["Kd_vs_reported_upper_fold"] = kd / upper
    else:
        record["Kd_vs_reported_lower_fold"] = np.nan
        record["Kd_vs_reported_upper_fold"] = np.nan
    rows.append(record)

pd.DataFrame(rows).sort_values(["source_id", "polymer", "stored_state", "csv_line"]).to_csv(
    OUTPUT, index=False
)
print(f"Wrote {len(rows)} audited rows to {OUTPUT}")
