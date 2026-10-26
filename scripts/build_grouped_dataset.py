"""Build the study-grouped model dataset from the literature master workbook.

``data/dataset.csv`` holds only predictors and the target, so observations
cannot be assigned to their source study. This script writes
``data/dataset_grouped.csv`` with three extra columns:

- ``source_id``: study identifier, the grouping unit for leave-one-study-out
  and GroupKFold validation;
- ``curve_id``: one isotherm, i.e. one experimental series of the source paper
  (``series_id`` of the master sheet), with ``series_label`` and ``ageing_type``;
- ``matrix``: aqueous matrix of the experiment (``freshwater``/``seawater``),
  used for the matrix sensitivity analysis.

It also runs unit/range checks and writes ``outputs/reports/data_qc_flags.csv``
listing records that must be verified against the original papers.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from config import (  # noqa: E402
    CAT_COLS,
    CURVE_COL,
    DATA_FILE,
    DATA_GROUPED_FILE,
    GROUP_COL,
    MASTER_FILE,
    NUM_COLS,
    REPORTS_DIR,
    TARGET,
)

MASTER_SHEET = "qe_model_cu"
MODEL_COLUMNS = CAT_COLS + NUM_COLS + [TARGET]

# Studies whose isotherms were measured in seawater. Extend this mapping (or
# add a ``matrix`` column to the master sheet) when new studies are extracted.
SEAWATER_STUDIES = {"Chen_2021", "Gao_2021"}

# Plausibility limits used only to flag records for manual verification.
CE_FLAG_LOW_MG_L = 1e-4
CE_FLAG_HIGH_MG_L = 500.0


def load_master(master_file: Path = MASTER_FILE) -> pd.DataFrame:
    master = pd.read_excel(master_file, sheet_name=MASTER_SHEET)
    included = master[master["model_include"].astype(bool)].copy()
    if "matrix" not in included.columns:
        included["matrix"] = included[GROUP_COL].map(
            lambda s: "seawater" if s in SEAWATER_STUDIES else "freshwater"
        )
    return included.reset_index(drop=True)


def add_curve_id(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    if "series_id" in df.columns:
        # One series of the source paper = one isotherm. Predictor columns alone
        # merge series that differ only in ageing treatment or co-solute level.
        df[CURVE_COL] = df["series_id"]
        return df
    keys = [GROUP_COL, "ReT", "AgS", "AdC", "Temp", "pH", "rpm"]
    df[CURVE_COL] = df[keys].apply(lambda r: "|".join(map(str, r.values)), axis=1)
    return df


def qc_flags(df: pd.DataFrame) -> pd.DataFrame:
    """Return records whose units or ranges need checking against the source."""
    reasons = []
    for _, row in df.iterrows():
        reason = []
        if row["Ce"] <= 0:
            reason.append("Ce is zero or negative")
        elif row["Ce"] < CE_FLAG_LOW_MG_L:
            reason.append(f"Ce < {CE_FLAG_LOW_MG_L} mg/L (check ug/L vs mg/L)")
        if row["Ce"] > CE_FLAG_HIGH_MG_L:
            reason.append(f"Ce > {CE_FLAG_HIGH_MG_L} mg/L (check units)")
        reasons.append("; ".join(reason))
    flagged = df.assign(qc_reason=reasons)
    return flagged[flagged["qc_reason"] != ""]


def check_against_dataset_csv(grouped: pd.DataFrame) -> None:
    """Confirm the grouped file reproduces the rows of data/dataset.csv."""
    if not Path(DATA_FILE).exists():
        return
    legacy = pd.read_csv(DATA_FILE)[MODEL_COLUMNS]
    a = legacy.round(8).sort_values(MODEL_COLUMNS).reset_index(drop=True)
    b = grouped[MODEL_COLUMNS].round(8).sort_values(MODEL_COLUMNS).reset_index(drop=True)
    same = len(a) == len(b) and a.fillna(-999).astype(str).equals(b.fillna(-999).astype(str))
    print(f"Rows: dataset.csv={len(a)}, grouped={len(b)}; identical content: {same}")


def sync_dataset_csv(grouped: pd.DataFrame) -> None:
    """Regenerate the model-ready file from the canonical grouped master rows."""
    model_data = grouped[MODEL_COLUMNS].copy()
    model_data.to_csv(DATA_FILE, index=False)
    print(f"Saved {DATA_FILE} ({len(model_data)} rows)")


def main() -> None:
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    grouped = add_curve_id(load_master())
    columns = [GROUP_COL, CURVE_COL, "series_label", "ageing_type", "matrix", "year", "doi"] + MODEL_COLUMNS
    grouped = grouped[[c for c in columns if c in grouped.columns]]
    grouped.to_csv(DATA_GROUPED_FILE, index=False)
    sync_dataset_csv(grouped)
    check_against_dataset_csv(grouped)

    flags = qc_flags(grouped)
    flags_path = REPORTS_DIR / "data_qc_flags.csv"
    flags.to_csv(flags_path, index=False)

    print(f"Saved {DATA_GROUPED_FILE} ({len(grouped)} rows, "
          f"{grouped[GROUP_COL].nunique()} studies, {grouped[CURVE_COL].nunique()} isotherms)")
    print(f"Saved {flags_path} ({len(flags)} flagged rows)")
    if not flags.empty:
        print(flags.groupby([GROUP_COL, "qc_reason"]).size().to_string())


if __name__ == "__main__":
    main()
