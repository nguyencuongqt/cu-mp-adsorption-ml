"""Shared cleaning rules for the Cu2+-MP adsorption model dataset."""

from __future__ import annotations

import pandas as pd

from config import CAT_COLS, NUM_COLS, TARGET


MODEL_COLUMNS = CAT_COLS + NUM_COLS + [TARGET]


def clean_model_data(raw_df: pd.DataFrame) -> pd.DataFrame:
    """Return the model-ready dataset used for training and evaluation.

    The manuscript-level cleaning removes unusable non-positive adsorption
    capacities and duplicate model records while retaining missing predictors
    for imputation inside the modeling pipeline.
    """
    missing = [col for col in MODEL_COLUMNS if col not in raw_df.columns]
    if missing:
        raise KeyError(f"Missing required model columns: {missing}")

    clean = raw_df.copy()
    clean[TARGET] = pd.to_numeric(clean[TARGET], errors="coerce")
    clean = clean.dropna(subset=[TARGET])
    clean = clean[clean[TARGET] > 0]
    clean = clean.drop_duplicates(subset=MODEL_COLUMNS, keep="first")
    return clean.reset_index(drop=True)
