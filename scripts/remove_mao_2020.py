"""Remove Mao_2020 from the master workbook (decision of 2026-10-07).

Reason: the polystyrene particles have a specified size of 0.1 um (100 nm), i.e. nanoparticles, which the
inclusion criteria exclude. The study was part of the submitted dataset (70 rows, 7 isotherms). The removed
rows are archived in Revision/Removed_Mao_2020_archive_2026-10-07.csv; a backup of the data files is
written first. Run scripts/build_grouped_dataset.py afterwards.
"""

from __future__ import annotations

import shutil
from pathlib import Path

import pandas as pd
from openpyxl import load_workbook

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
MASTER = DATA / "raw_data_literature_master.xlsx"
BACKUP = DATA / "_backups" / "backup_2026-10-07_before_removing_Mao2020"
ARCHIVE = ROOT / "Manuscript" / "Submission_Env pollution" / "Revision" / "06_Data_audit" / "Removed_Mao_2020_archive_2026-10-07.csv"
STUDY = "Mao_2020"
REASON = "particles of 0.1 um (nanoparticles); outside the inclusion criteria"


def main() -> None:
    before = pd.read_excel(MASTER, sheet_name="qe_model_cu")
    rows = before[before["source_id"] == STUDY]
    if rows.empty:
        raise SystemExit(f"{STUDY} is not in the master workbook.")
    BACKUP.mkdir(exist_ok=True)
    for name in ["raw_data_literature_master.xlsx", "dataset.csv", "dataset_grouped.csv", "paper_metadata.csv"]:
        shutil.copy2(DATA / name, BACKUP / name)
    rows.assign(removal_reason=REASON, removed_on="2026-10-07").to_csv(ARCHIVE, index=False, encoding="utf-8-sig")

    wb = load_workbook(MASTER)
    for sheet in ("qe_model_cu", "literature_metadata"):
        ws = wb[sheet]
        col = [c.value for c in ws[1]].index("source_id") + 1
        for r in range(ws.max_row, 1, -1):
            if ws.cell(row=r, column=col).value == STUDY:
                ws.delete_rows(r)
    after = before[before["source_id"] != STUDY]
    n0, n1 = len(before), len(after)
    s0, s1 = before["source_id"].nunique(), after["source_id"].nunique()
    c0, c1 = before["series_id"].nunique(), after["series_id"].nunique()
    swaps = [(f"{n0} Cu rows", f"{n1} Cu rows"), (f"same {n0} rows", f"same {n1} rows"), (f"{s0} literature studies", f"{s1} literature studies"),
             (f"({c0} isotherm series)", f"({c1} isotherm series)"), (f"All {s0} studies", f"All {s1} studies")]
    for sheet in ("README", "validation_summary"):
        for row in wb[sheet].iter_rows(min_row=2):
            for cell in row:
                if isinstance(cell.value, str):
                    for old, new in swaps:
                        cell.value = cell.value.replace(old, new)
                    cell.value = cell.value.replace("last studies added: Godoy_2019, Liu_2022",
                                                    "last studies added: Godoy_2019, Liu_2022; Mao_2020 removed")
                elif sheet == "validation_summary" and cell.value == n0:
                    cell.value = n1
                elif sheet == "validation_summary" and cell.value == s0:
                    cell.value = s1
    wb.save(MASTER)
    print(f"removed {len(rows)} rows of {STUDY}: rows {n0} -> {n1}; studies {s0} -> {s1}; series {c0} -> {c1}")
    print(f"archive: {ARCHIVE.name}; backup: {BACKUP.name}")


if __name__ == "__main__":
    main()
