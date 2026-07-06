"""Diagnose the missing-data mechanism for pH before imputation.

pH is a high-impact predictor in the Cu adsorption workflow: it drives OHE
rank #4 with about 12% SHAP contribution. Because pH has substantial
missingness, its missing-data mechanism can affect model interpretation,
uncertainty, and any conclusions drawn from feature importance.
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import missingno as msno
import numpy as np
import pandas as pd
from scipy import stats

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from config import CAT_COLS, DATA_FILE, FIGURES_DIR, NUM_COLS, REPORTS_DIR, TARGET
from src.analysis.data_cleaning import clean_model_data


PH_COL = "pH"
MAR_NUMERIC_COLS = ["Ce", "Temp", "rpm"]


def _ensure_output_dirs() -> None:
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)


def _load_raw_dataset() -> pd.DataFrame:
    data_file = Path(DATA_FILE)
    if not data_file.exists():
        raise FileNotFoundError(
            f"DATA_FILE does not exist: {data_file}. "
            "Place the raw dataset there or update config.DATA_FILE."
        )

    if data_file.suffix.lower() in {".xlsx", ".xls"}:
        return pd.read_excel(data_file)
    if data_file.suffix.lower() == ".csv":
        return pd.read_csv(data_file)

    raise ValueError(f"Unsupported DATA_FILE extension: {data_file.suffix}")


def _missing_summary(df: pd.DataFrame) -> pd.DataFrame:
    summary = pd.DataFrame(
        {
            "column": df.columns,
            "n_missing": df.isna().sum().to_numpy(),
            "pct_missing": (df.isna().mean() * 100).to_numpy(),
        }
    )
    return summary.sort_values("pct_missing", ascending=False).reset_index(drop=True)


def _estimate_mean_cov_em(
    numeric_df: pd.DataFrame, max_iter: int = 500, tol: float = 1e-7
) -> tuple[np.ndarray, np.ndarray]:
    x = numeric_df.to_numpy(dtype=float)
    n_rows, n_cols = x.shape

    mu = np.nanmean(x, axis=0)
    col_means = np.where(np.isnan(mu), 0.0, mu)
    filled = np.where(np.isnan(x), col_means, x)
    sigma = np.cov(filled, rowvar=False, bias=True)
    sigma = np.atleast_2d(sigma)
    sigma += np.eye(n_cols) * 1e-6

    for _ in range(max_iter):
        sum_x = np.zeros(n_cols)
        sum_xx = np.zeros((n_cols, n_cols))

        for row in x:
            observed = ~np.isnan(row)
            missing = ~observed
            expected = row.copy()
            expected_xx = np.zeros((n_cols, n_cols))

            if missing.any():
                if observed.any():
                    sigma_oo = sigma[np.ix_(observed, observed)]
                    sigma_mo = sigma[np.ix_(missing, observed)]
                    sigma_om = sigma[np.ix_(observed, missing)]
                    sigma_mm = sigma[np.ix_(missing, missing)]
                    inv_sigma_oo = np.linalg.pinv(sigma_oo)
                    centered_obs = row[observed] - mu[observed]

                    cond_mean = mu[missing] + sigma_mo @ inv_sigma_oo @ centered_obs
                    cond_cov = sigma_mm - sigma_mo @ inv_sigma_oo @ sigma_om
                else:
                    cond_mean = mu[missing]
                    cond_cov = sigma[np.ix_(missing, missing)]

                expected[missing] = cond_mean
                expected_xx += np.outer(expected, expected)
                missing_idx = np.where(missing)[0]
                expected_xx[np.ix_(missing_idx, missing_idx)] += cond_cov
            else:
                expected_xx = np.outer(expected, expected)

            sum_x += expected
            sum_xx += expected_xx

        new_mu = sum_x / n_rows
        new_sigma = sum_xx / n_rows - np.outer(new_mu, new_mu)
        new_sigma = (new_sigma + new_sigma.T) / 2
        new_sigma += np.eye(n_cols) * 1e-8

        delta = max(
            np.max(np.abs(new_mu - mu)),
            np.max(np.abs(new_sigma - sigma)),
        )
        mu, sigma = new_mu, new_sigma
        if delta < tol:
            break

    return mu, sigma


def little_mcar_test(df: pd.DataFrame) -> tuple[float, int, float, list[str]]:
    """Manual Little's MCAR chi-square test for numeric variables."""
    numeric_df = df.apply(pd.to_numeric, errors="coerce")
    numeric_df = numeric_df.dropna(axis=1, how="all")
    cols = list(numeric_df.columns)
    if len(cols) < 2:
        raise ValueError("Little's MCAR test requires at least two numeric columns.")

    mu, sigma = _estimate_mean_cov_em(numeric_df)
    patterns = numeric_df.isna().astype(int).astype(str).agg("".join, axis=1)

    chi_square = 0.0
    degrees_components = 0

    for _, group in numeric_df.groupby(patterns, sort=False):
        observed_mask = ~group.iloc[0].isna().to_numpy()
        if not observed_mask.any():
            continue

        observed_idx = np.where(observed_mask)[0]
        group_mean = group.iloc[:, observed_idx].mean(skipna=True).to_numpy(dtype=float)
        expected_mean = mu[observed_idx]
        sigma_obs = sigma[np.ix_(observed_idx, observed_idx)]
        diff = group_mean - expected_mean

        chi_square += len(group) * float(diff.T @ np.linalg.pinv(sigma_obs) @ diff)
        degrees_components += len(observed_idx)

    dof = max(degrees_components - len(cols), 1)
    p_value = float(stats.chi2.sf(chi_square, dof))
    return float(chi_square), int(dof), p_value, cols


def _save_missing_matrix(df: pd.DataFrame) -> Path:
    output = FIGURES_DIR / "missing_matrix.png"
    plt.figure(figsize=(10, 6))
    msno.matrix(df, sparkline=False)
    plt.tight_layout()
    plt.savefig(output, dpi=300, bbox_inches="tight")
    plt.close()
    return output


def _save_missing_correlations(correlations: dict[str, tuple[float, float, int]]) -> Path:
    output = FIGURES_DIR / "missing_correlation.png"
    labels = list(correlations)
    values = [correlations[col][0] for col in labels]
    p_values = [correlations[col][1] for col in labels]
    colors = ["#2c7fb8" if p >= 0.05 else "#d95f0e" for p in p_values]

    plt.figure(figsize=(7, 4.5))
    plt.bar(labels, values, color=colors)
    plt.axhline(0, color="black", linewidth=0.8)
    plt.ylabel("Point-biserial correlation with pH_missing")
    plt.xlabel("Numeric predictor")
    plt.tight_layout()
    plt.savefig(output, dpi=300, bbox_inches="tight")
    plt.close()
    return output


def run(df: pd.DataFrame) -> dict[str, object]:
    _ensure_output_dirs()

    lines: list[str] = []

    def emit(message: str = "") -> None:
        print(message)
        lines.append(message)

    emit("Missing-data diagnosis for pH")
    emit("=" * 36)
    emit(f"DATA_FILE: {Path(DATA_FILE)}")
    emit(f"Dataset shape: {df.shape[0]} rows x {df.shape[1]} columns")
    emit("")

    summary = _missing_summary(df)
    emit("Missing-data summary")
    emit(summary.to_string(index=False, formatters={"pct_missing": "{:.2f}".format}))
    emit("")

    matrix_path = _save_missing_matrix(df)
    emit(f"Saved missingness matrix plot: {matrix_path}")
    emit("")

    little_cols = [col for col in NUM_COLS + [TARGET] if col in df.columns]
    chi_square, dof, mcar_p, tested_cols = little_mcar_test(df[little_cols])
    emit("Little's MCAR test")
    emit(f"Variables tested: {', '.join(tested_cols)}")
    emit(f"chi-square = {chi_square:.4f}, df = {dof}, p = {mcar_p:.6g}")
    if mcar_p >= 0.05:
        mcar_conclusion = f"MCAR (p={mcar_p:.6g})"
    else:
        mcar_conclusion = f"Reject MCAR \u2192 likely MAR/MNAR (p={mcar_p:.6g})"
    emit(f"Conclusion: {mcar_conclusion}")
    emit("")

    if PH_COL not in df.columns:
        raise KeyError(f"Required pH column not found: {PH_COL}")

    work_df = df.copy()
    work_df["pH_missing"] = work_df[PH_COL].isna().astype(int)

    emit("MAR investigation: numeric predictors")
    numeric_results: dict[str, tuple[float, float, int]] = {}
    mar_signals: list[str] = []
    for col in MAR_NUMERIC_COLS:
        if col not in work_df.columns:
            emit(f"{col}: skipped, column not found")
            continue

        pair = work_df[["pH_missing", col]].dropna()
        if pair["pH_missing"].nunique() < 2 or pair[col].nunique() < 2:
            numeric_results[col] = (np.nan, np.nan, len(pair))
            emit(f"{col}: skipped, insufficient variation after dropping NaNs")
            continue

        corr, p_value = stats.pointbiserialr(pair["pH_missing"], pair[col])
        numeric_results[col] = (float(corr), float(p_value), len(pair))
        emit(f"{col}: r = {corr:.4f}, p = {p_value:.6g}, n = {len(pair)}")
        if p_value < 0.05:
            mar_signals.append(col)
    emit("")

    corr_path = _save_missing_correlations(numeric_results)
    emit(f"Saved missingness correlation plot: {corr_path}")
    emit("")

    emit("MAR investigation: categorical predictors")
    for col in CAT_COLS:
        if col not in work_df.columns:
            emit(f"{col}: skipped, column not found")
            continue

        contingency = pd.crosstab(work_df[col].astype("string").fillna("<MISSING>"), work_df["pH_missing"])
        if contingency.shape[0] < 2 or contingency.shape[1] < 2:
            emit(f"{col}: skipped, contingency table has insufficient variation")
            continue

        chi2, p_value, cat_dof, _ = stats.chi2_contingency(contingency)
        emit(f"{col}: chi2 = {chi2:.4f}, df = {cat_dof}, p = {p_value:.6g}")
        if p_value < 0.05:
            mar_signals.append(col)
    emit("")

    if mar_signals:
        mar_conclusion = (
            "MAR supported: pH_missing is associated with "
            f"{', '.join(mar_signals)} at p < 0.05."
        )
    else:
        mar_conclusion = "No observed MAR signal at p < 0.05 among tested predictors."
    emit(f"MAR conclusion: {mar_conclusion}")
    emit("")

    emit(
        "MNAR note: MNAR cannot be tested from the observed dataset alone; "
        "external validation data or study-design information would be required. "
        "This remains a limitation."
    )

    report_path = REPORTS_DIR / "missing_diagnosis.txt"
    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8-sig")

    return {
        "missing_summary": summary,
        "little_mcar": {"chi_square": chi_square, "df": dof, "p_value": mcar_p},
        "mar_signals": mar_signals,
        "report_path": report_path,
        "matrix_path": matrix_path,
        "correlation_path": corr_path,
    }


if __name__ == "__main__":
    raw_df = clean_model_data(_load_raw_dataset())
    run(raw_df)
