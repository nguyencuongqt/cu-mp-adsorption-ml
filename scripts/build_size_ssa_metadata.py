"""Series-level particle size and specific surface area (SSA) for the laboratory dataset.

SSA: BET values transcribed on 2026-10-07 from the article or its supplementary file and assigned only
to the isotherm series whose polymer and treatment match the measured material (no imputation).
Size: one representative dimension per series from the audit workbook
(Revision/Particle_size_SSA_audit.xlsx); ranges are represented by their geometric mean, upper limits
are kept as such and flagged; mesh classes and films are left empty.

Output: data/size_ssa_metadata.csv (one row per isotherm series).
"""

from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]

# curve_id -> (BET SSA m2/g, source locator)
SSA = {}


def add(source: str, values: dict[str, float]) -> None:
    SSA.update({k: (v, source) for k, v in values.items()})


add("Cheng et al. (2025), SI Table S3", {"Cheng_2025_PA_virgin": 2.098, "Cheng_2025_PA_chemox": 3.546, "Cheng_2025_PA_biofilm": 2.214,
                                         "Cheng_2025_PLA_virgin": 1.962, "Cheng_2025_PLA_chemox": 2.930, "Cheng_2025_PLA_biofilm": 2.215})
add("Feng et al. (2025), SI Table S1", {"PLA_virgin": 3.07, "PLA_UV30d": 3.15})
add("Godoy et al. (2019), Table 1", {"Godoy_2019_PE_virgin": 1.4, "Godoy_2019_PP_virgin": 2.7, "Godoy_2019_PS_virgin": 3.2})
add("Hu et al. (2024), Table 1 and text", {"Hu_2024_01": 0.4482, "Hu_2024_02": 0.6716, "Hu_2024_03": 0.2789, "Hu_2024_04": 0.3357})
add("Huang et al. (2024), SI Table S3", {"Huang_2024_01": 0.64, "Huang_2024_02": 1.61, "Huang_2024_03": 0.58, "Huang_2024_04": 3.26})
add("Jiang et al. (2026), SI Table S5", {
    "Jiang_2026_PP_50um_virgin": 7.437, "Jiang_2026_PP_50um_chemox": 1.491, "Jiang_2026_PP_50um_biofilm": 6.158,
    "Jiang_2026_PP_50um_chemox_biofilm": 8.444, "Jiang_2026_PP_200um_virgin": 1.385, "Jiang_2026_PP_200um_chemox": 3.095,
    "Jiang_2026_PP_200um_biofilm": 3.647, "Jiang_2026_PP_200um_chemox_biofilm": 4.817})
add("Lin et al. (2023), Section 3.1", {"CPB_PE_Cu": 0.1755, "BPB_PLA_Cu": 0.4637})
add("Liu et al. (2022), SI Table S2", {"Liu_2022_PP_virgin": 0.2335, "Liu_2022_PS_virgin": 0.7215, "Liu_2022_PVC_virgin": 0.7399})
add("Sun et al. (2022), SI BET table", {"Sun_2022_01": 1.40, "Sun_2022_02": 1.05})
add("Sun et al. (2023), SI BET table", {"Sun_2023_01": 0.1145, "Sun_2023_02": 0.1253})
add("Wang et al. (2020b), SI Table S1", {"Wang_2020b_01": 0.2354})
add("Wang et al. (2021), SI Table S2", {"Wang_2021_01": 0.23, "Wang_2021_05": 0.23, "Wang_2021_02": 0.35, "Wang_2021_06": 0.35,
                                        "Wang_2021_03": 0.76, "Wang_2021_07": 0.76, "Wang_2021_04": 1.16, "Wang_2021_08": 1.16})
add("Yang et al. (2019), Table 1 (virgin materials only)", {"Yang_2019_01": 8.710, "Yang_2019_03": 0.110})
add("Yu et al. (2026), SI Table S8", {"PP_fresh_15C": 14.2922, "PP_fresh_25C": 14.2922, "PP_fresh_35C": 14.2922,
                                      "PP_UV7d_15C": 14.9448, "PP_UV7d_25C": 14.9448, "PP_UV7d_35C": 14.9448})
add("Zou et al. (2020), Table 3", {"Zou_2020_01": 2.3, "Zou_2020_02": 8.9, "Zou_2020_03": 3.1, "Zou_2020_04": 1.3})
add("Xue et al. (2021), Table 1 (pristine value printed as 05430)", {"Xue_2021_01": 0.5430, "Xue_2021_03": 0.5430,
                                                                     "Xue_2021_02": 0.7573, "Xue_2021_04": 0.7573})
add("Liu Y et al. (2025), Table 1; Zhou W et al. (2025), Table 1 (CK values identical)", {
    "LiuZhou_2025_PS_virgin": 0.707, "LiuZhou_2025_PS_thermal": 0.829, "LiuZhou_2025_PS_UV": 1.212, "LiuZhou_2025_PS_acid": 1.041,
    "LiuZhou_2025_PLA_virgin": 0.799, "LiuZhou_2025_PLA_thermal": 1.029, "LiuZhou_2025_PLA_UV": 1.223, "LiuZhou_2025_PLA_acid": 0.854})
SSA_REPORTED_NOT_TRANSCRIBED = {"Xiao_2026": "no numeric SSA: only the increase after biofilm formation is given (values in Fig. 3c)"}

# source_id -> (representative size in um, kind) ; kind: value, range (geometric mean), upper (upper limit)
SIZE = {
    "Mao_2020": (0.1, "value"), "Wang_2020": (1000.0, "value"), "Wang_2020b": (np.sqrt(60 * 150), "range"), "Zou_2020": (280.0, "upper"),
    "Wang_2021": (np.sqrt(75 * 140), "range"), "Qiongjie_2022": (50.0, "value"), "Sathyamoorthy_2022": (125.0, "value"),
    "Sun_2022": (np.sqrt(75 * 150), "range"), "Zhou_2022b": (75.0, "value"), "Sun_2023": (200.0, "upper"), "Hu_2024": (np.sqrt(50 * 100), "range"),
    "Chen_2025": (150.0, "upper"), "Dai_2025": (75.0, "value"), "Xiao_2026": (np.sqrt(74 * 50), "range"), "Yu_2026": (50.0, "value"),
    "Feng_2025": (25.0, "value"), "Zou_2026": (np.sqrt(75 * 150), "range"), "Wang_2023": (np.sqrt(40 * 50), "range"),
    "Godoy_2019": (5000.0, "upper"), "Liu_2022": (100.0, "upper"),
    "Yan_2023": (125.0, "value"), "LiuZhou_2025": (150.0, "value"),  # d50 (Yan, Table 1); average size (Liu/Zhou, 2.1)
    "Turner_2015": (4000.0, "value"),  # virgin pellets ~4 mm; beached 2-6 mm (Turner and Holmes, 2015, Pellet characteristics)
}
SIZE_BY_SERIES = {  # studies with different sizes among their series
    "Yang_2019_01": 69.86, "Yang_2019_02": 69.86, "Yang_2019_03": 74.76, "Yang_2019_04": 74.76,
    "Huang_2024_01": 103.63, "Huang_2024_02": 99.39, "Huang_2024_03": 74.52, "Huang_2024_04": 68.73,
    "Cheng_2025_PA_virgin": 118.0, "Cheng_2025_PA_chemox": 118.0, "Cheng_2025_PA_biofilm": 118.0,
    "Cheng_2025_PLA_virgin": 132.0, "Cheng_2025_PLA_chemox": 132.0, "Cheng_2025_PLA_biofilm": 132.0,
}
SIZE_NOT_SCALAR = {"Xue_2021": "mesh class only", "Lin_2022": "mesh class only", "Tang_2022": "mesh class only",
                   "Lin_2023": "mesh class only", "Zhou_2022": "film (width and thickness)"}


def main() -> None:
    d = pd.read_csv(ROOT / "data" / "dataset_grouped.csv")
    s = d.groupby(["source_id", "curve_id", "series_label", "ReT", "AgS"]).size().rename("n_points").reset_index()
    unknown = set(SSA) - set(s["curve_id"])
    assert not unknown, unknown
    s["ssa_m2_g"] = s["curve_id"].map(lambda c: SSA.get(c, (np.nan, ""))[0])
    s["ssa_source"] = s["curve_id"].map(lambda c: SSA.get(c, (np.nan, ""))[1])
    s.loc[s["ssa_m2_g"].isna(), "ssa_source"] = s["source_id"].map(SSA_REPORTED_NOT_TRANSCRIBED).fillna("")

    def size(row) -> tuple[float, str]:
        if row["curve_id"] in SIZE_BY_SERIES:
            return SIZE_BY_SERIES[row["curve_id"]], "value"
        label = row["series_label"]
        if row["source_id"] == "Jiang_2026":
            return (50.0 if "50um" in label else 200.0), "value"
        if row["source_id"] == "Muthuraja_2024":
            if "250" in label or label.rstrip().endswith(("PPb", "PSb")):   # fraction b = 250-500 um
                return float(np.sqrt(250 * 500)), "range"
            if "1000" in label or "500" in label:
                return float(np.sqrt(500 * 1000)), "range"
            return np.nan, "size fraction not identified in the series label"
        if row["source_id"] in SIZE:
            return SIZE[row["source_id"]]
        return np.nan, SIZE_NOT_SCALAR.get(row["source_id"], "not assigned")

    sized = s.apply(size, axis=1, result_type="expand")
    s["size_um"], s["size_kind"] = sized[0].round(2), sized[1]
    s.to_csv(ROOT / "data" / "size_ssa_metadata.csv", index=False, encoding="utf-8-sig")
    has_ssa, has_size = s["ssa_m2_g"].notna(), s["size_um"].notna()
    print(f"SSA assigned: {has_ssa.sum()} series, {s.loc[has_ssa, 'source_id'].nunique()} studies, {s.loc[has_ssa, 'n_points'].sum()} points")
    print(f"size assigned: {has_size.sum()} series, {s.loc[has_size, 'source_id'].nunique()} studies, {s.loc[has_size, 'n_points'].sum()} points")
    print("size kinds:", s["size_kind"].value_counts().to_dict())
    print(s.loc[s["source_id"] == "Muthuraja_2024", ["series_label", "size_um", "size_kind"]].to_string(index=False))


if __name__ == "__main__":
    main()
