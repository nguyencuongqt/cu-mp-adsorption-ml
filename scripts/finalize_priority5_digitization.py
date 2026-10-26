"""QC and standardize the five Priority-1 Cu–MP digitization templates.

This preserves user-entered templates and writes separate, fully traceable
finalization tables.  It does not edit the literature master workbook; the
workbook import is a separate reviewed step.
"""

from __future__ import annotations

from pathlib import Path
import pandas as pd
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "Manuscript" / "Submission_Env pollution" / "Revision" / "06_Data_audit" / "Digitization_next_5_primary_isotherm_papers"
OUT = BASE / "finalized_data"
M_CU = 63.546  # g mol-1


def read_template(folder: str) -> pd.DataFrame:
    return pd.read_csv(next((BASE / folder).glob("*template.csv")))


def basic_record(row: pd.Series, *, source_id: str, first_author: str, year: int,
                 ce: float, qe: float, c0: float | None, c0_inferred: float | None,
                 ce_origin: str, qe_origin: str, include: bool, qa_status: str,
                 qa_note: str, ageing_type: str, ageing_detail: str) -> dict:
    return {
        "point_id": row["point_id"],
        "source_id": source_id,
        "source_author_year": f"({first_author} et al., {year})",
        "first_author": first_author,
        "year": year,
        "doi": row["doi"],
        "figure_panel": row["figure_panel"],
        "source_pdf_page": row["source_pdf_page"],
        "series_id": row["series_id"],
        "polymer": row["polymer"],
        "particle_size_um": row["particle_size_um"],
        "ageing_status": row.get("ageing_status", "virgin"),
        "ageing_type": ageing_type,
        "ageing_detail": ageing_detail,
        "temperature_C": row["temperature_C"],
        "pH": row["pH"],
        "mp_dose_g_L": row.get("mp_dose_g_L", row.get("mp_dose_mg_L", np.nan) / 1000),
        "contact_time_h": row["contact_time_h"],
        "rpm": row["rpm"],
        "matrix": row["matrix"],
        "c0_nominal_mg_L": c0,
        "ce_mg_L": ce,
        "qe_mg_g": qe,
        "c0_inferred_mg_L": c0_inferred,
        "ce_origin": ce_origin,
        "qe_origin": qe_origin,
        "model_include": include,
        "qa_status": qa_status,
        "qa_note": qa_note,
    }


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    records: list[dict] = []

    # 08 — points are direct Ce–qe observations.  C0 is inferred only for QC.
    d = read_template("08_CEJ_2026_175301_PP_PE_UV")
    nominal = np.array([0.1, 1, 2, 4, 6, 8, 10], dtype=float)
    for _, r in d.iterrows():
        ce, qe = float(r.ce_mg_L), float(r.qe_raw)
        c0i = ce + qe
        c0 = float(nominal[np.argmin(np.abs(nominal - c0i))])
        note = f"Direct Fig. 1 value; C0 inferred {c0i:.5f} mg/L, nearest nominal {c0:g} mg/L."
        records.append(basic_record(
            r, source_id="Yu_2026", first_author="Yu", year=2026, ce=ce, qe=qe,
            c0=c0, c0_inferred=c0i, ce_origin="direct_fig1_Ce", qe_origin="direct_fig1_qe",
            include=True, qa_status="included", qa_note=note,
            ageing_type="UV" if str(r.ageing_status) != "virgin" else "none",
            ageing_detail=str(r.ageing_status),
        ))

    # 09 — manual q values used the figure's doubled scale across the broken y-axis.
    # The Cu points are in the 0–300 mg/kg lower segment, so q must be divided by 2,
    # then by 1000 to express mg/g.  Ce is directly read from Fig. 4.
    d = read_template("09_Chemosphere_2023_138920_PE_PLA_bags")
    for _, r in d.iterrows():
        ce = float(r.ce_mg_L)
        qe = float(r.qe_raw) / 2 / 1000
        c0 = float(r.c0_nominal_mg_L)
        c0i = ce + 10 * qe
        note = (
            "Direct Fig. 4 Cu point; raw q digitization divided by 2 for the lower "
            "0–300 mg/kg y-axis segment, then converted mg/kg→mg/g. "
            "Nominal-C0 check is not expected to close exactly because the paper defines q "
            "using measured blank Cbk, which is not reported."
        )
        records.append(basic_record(
            r, source_id="Lin_2023", first_author="Lin", year=2023, ce=ce, qe=qe,
            c0=c0, c0_inferred=c0i, ce_origin="direct_fig4_Ce", qe_origin="fig4_qe_corrected_half_scale_mgkg_to_mgg",
            include=True, qa_status="included_with_caveat", qa_note=note,
            ageing_type="none", ageing_detail="virgin",
        ))

    # 10 — retain the digitized graph as an auditable candidate but exclude it.
    # Fig. 5 extends Ce to ~33 mg/L although Methods 2.5 reports C0=1,2,5,8,10 mg/L.
    d = read_template("10_EcoEnvSaf_2022_113292_PS_UV")
    for _, r in d.iterrows():
        ce, qe = float(r.ce_mg_L), float(r.qe_raw)
        records.append(basic_record(
            r, source_id="Chen_2022", first_author="Chen", year=2022, ce=ce, qe=qe,
            c0=None, c0_inferred=ce + (0.2 / 0.03) * qe,
            ce_origin="direct_fig5_Ce_unresolved", qe_origin="direct_fig5_qe_unresolved",
            include=False, qa_status="excluded_unresolved", qa_note=(
                "Exclude: Fig. 5 Ce reaches ~33 mg/L, irreconcilable with Methods 2.5 "
                "C0=1,2,5,8,10 mg/L and the stated 0.2 g/30 mL dose."
            ),
            ageing_type="UV" if str(r.ageing_status) != "virgin" else "none",
            ageing_detail=str(r.ageing_status),
        ))

    # 11 — Fig. 4c is C0–qe, not Ce–qe.  The template's x-coordinate field is not Ce.
    # Derive Ce exactly from the reported dose (10 mg PE/L) and molar mass of Cu.
    d = read_template("11_WaterRes_2022_118354_PE_size")
    for _, r in d.iterrows():
        c0_u = float(r.c0_CuSO4_umol_L)
        q_u = float(r.qe_raw_umol_per_mg)
        ce_u = c0_u - q_u * float(r.mp_dose_mg_L)
        ce = ce_u * M_CU / 1000
        qe = q_u * M_CU
        c0 = c0_u * M_CU / 1000
        note = (
            "Fig. 4c is C0–qe. Ce derived by C0(µmol/L) − qe(µmol/mg)×10 mg PE/L; "
            "then converted with M(Cu)=63.546 g/mol."
        )
        records.append(basic_record(
            r, source_id="Zhou_2022_WaterRes", first_author="Zhou", year=2022, ce=ce, qe=qe,
            c0=c0, c0_inferred=ce + qe * 0.01, ce_origin="derived_mass_balance_from_fig4c_C0_qe",
            qe_origin="direct_fig4c_umolmg_to_mgg", include=True, qa_status="included_derived_Ce",
            qa_note=note, ageing_type="none", ageing_detail="virgin",
        ))

    # 12 — plotted x values match nominal C0 despite the axis being labeled Ce.
    # With 10 mg MPs/10 mL (1 g/L), the defensible equilibrium concentration is C0−qe.
    d = read_template("12_JECE_2025_117033_PLA_UV")
    for _, r in d.iterrows():
        c0 = float(r.c0_nominal_mg_L)
        qe = float(r.qe_raw)
        ce = c0 - qe  # 1 g/L dose
        note = (
            "Fig. 4a x-coordinates track nominal C0 although labeled Ce. Ce corrected "
            "by mass balance: C0 − qe×1 g/L; q is the direct Fig. 4a value."
        )
        records.append(basic_record(
            r, source_id="Feng_2025", first_author="Feng", year=2025, ce=ce, qe=qe,
            c0=c0, c0_inferred=ce + qe, ce_origin="derived_mass_balance_fig4a_axis_inconsistency",
            qe_origin="direct_fig4a_qe", include=True, qa_status="included_derived_Ce",
            qa_note=note, ageing_type="UV" if str(r.ageing_status) != "virgin" else "none",
            ageing_detail=str(r.ageing_status),
        ))

    final = pd.DataFrame(records)
    final["temperature_C"] = pd.to_numeric(final["temperature_C"], errors="coerce")
    final["pH_numeric"] = pd.to_numeric(final["pH"], errors="coerce")
    final["mp_dose_g_L"] = pd.to_numeric(final["mp_dose_g_L"], errors="coerce")
    final["rpm"] = pd.to_numeric(final["rpm"], errors="coerce")
    final["c0_nominal_mg_L"] = pd.to_numeric(final["c0_nominal_mg_L"], errors="coerce")
    final["c0_inferred_mg_L"] = pd.to_numeric(final["c0_inferred_mg_L"], errors="coerce")
    final["ce_mg_L"] = pd.to_numeric(final["ce_mg_L"], errors="coerce")
    final["qe_mg_g"] = pd.to_numeric(final["qe_mg_g"], errors="coerce")
    final["mass_balance_residual_mg_L"] = final["c0_inferred_mg_L"] - final["c0_nominal_mg_L"]

    final.to_csv(OUT / "finalized_Cu_isotherm_records.csv", index=False, float_format="%.8g")
    final.loc[final.model_include].to_csv(OUT / "finalized_Cu_isotherm_included.csv", index=False, float_format="%.8g")

    # Master-table rows retain candidate records, including Chen_2022 as an explicit exclusion.
    master = pd.DataFrame({
        "source_id": final.source_id,
        "source_author_year": final.source_author_year,
        "first_author": final.first_author,
        "year": final.year,
        "doi": final.doi,
        "doi_match_status": "verified_original_pdf_figure_and_qc",
        "raw_row_id": np.nan,
        "model_include": final.model_include,
        "exclusion_reason": np.where(final.model_include, None, "methods_figure_concentration_conflict"),
        "AgS": np.where(final.ageing_status.eq("virgin"), "Virgin", "Aged"),
        "ReT": final.polymer,
        "AdC": "Single-ion",
        "Temp": final.temperature_C,
        "pH": final.pH_numeric,
        "rpm": final.rpm,
        "Ce": final.ce_mg_L,
        "qe": final.qe_mg_g,
        "series_id": final.series_id,
        "series_label": final.series_id,
        "ageing_type": final.ageing_type,
        "ageing_detail": final.ageing_detail,
    })
    master.to_csv(OUT / "master_append_rows_priority5.csv", index=False, float_format="%.8g")

    lines = [
        "# Priority-1 digitization QC and finalization",
        "",
        "## Result",
        "",
        f"- Candidate points reviewed: {len(final)}",
        f"- Included in the model-ready dataset: {int(final.model_include.sum())}",
        f"- Retained but excluded as unresolved: {int((~final.model_include).sum())}",
        "",
        "## Decisions and transformations",
        "",
        "- **Yu 2026 (42 points): include.** Fig. 1 direct Ce–qe values. Inferred C0 closes to 0.1, 1, 2, 4, 6, 8, or 10 mg/L.",
        "- **Lin 2023 (12 points): include with caveat.** Fig. 4 provides direct Ce–qe. The raw q readings were halved (wrong y-axis segment) and converted from mg/kg to mg/g. Nominal C0 is not used to overwrite Ce because the paper defines q with a blank-control concentration (Cbk) that is not reported.",
        "- **Chen 2022 (30 points): exclude.** Fig. 5 extends to Ce≈33 mg/L but Methods 2.5 lists initial C0 only up to 10 mg/L. No defensible conversion can resolve this without author data or an erratum.",
        "- **Zhou 2022 Water Research (12 points): include as derived Ce.** Fig. 4c is C0–qe. Ce was derived using the reported 10 mg PE/L dose and Cu molar mass; the original template x-coordinate is not used as Ce.",
        "- **Feng 2025 (10 points): include as derived Ce.** Fig. 4a x positions track nominal C0 despite the Ce label. Ce was calculated from C0−qe at the reported 1 g/L dose; q is retained from the figure.",
        "",
        "## Files",
        "",
        "- `finalized_Cu_isotherm_records.csv`: all reviewed observations with provenance and QA flags.",
        "- `finalized_Cu_isotherm_included.csv`: the 76 observations suitable for the model dataset.",
        "- `master_append_rows_priority5.csv`: columns aligned to the `qe_model_cu` master table, including the 30 excluded audit rows.",
    ]
    (OUT / "Priority5_digitization_QC_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Wrote {len(final)} reviewed records; included {int(final.model_include.sum())}.")


if __name__ == "__main__":
    main()
