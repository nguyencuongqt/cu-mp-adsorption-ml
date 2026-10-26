"""Diagnose model behaviour at field-level Cu concentrations.

River Ce (about 0.001-0.01 mg/L) lies at or below the lower edge of the
laboratory data. This module documents how each laboratory model behaves
there:

1. ``low_ce_data_support.csv`` - number of observations and studies per Ce
   decade, i.e. how much laboratory evidence supports each range;
2. ``low_ce_response_curves.csv`` / ``low_ce_response.png`` - predicted qe
   along a Ce grid for PP and PE under the field scenario, for every model;
3. ``low_ce_extrapolation_test.csv`` - a concentration hold-out: models are
   trained only on Ce above a threshold and tested on the observations below
   it, which measures downward extrapolation directly.

Each analysis is run on all studies and on freshwater studies only.
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from config import FIGURES_DIR, GLOBAL_CU_MG_L, GROUP_COL, REPORTS_DIR, TARGET  # noqa: E402
from src.analysis.grouped_validation import load_data, point_metrics  # noqa: E402
from src.analysis.lab_models import LAB_MODELS, field_scenario_frame, fit_lab_model  # noqa: E402

CE_GRID = np.logspace(-4, 2, 61)
FIELD_CE_RANGE = (1e-3, 1e-2)
THRESHOLDS_MG_L = [0.1, 1.0]
POLYMERS = ["PP", "PE"]
MODEL_LABELS = {"qrf_raw": "QRF (qe)", "qrf_log": "QRF (log qe)", "freundlich": "Mixed-effects Freundlich"}
MODEL_COLORS = {"qrf_raw": "#c43c39", "qrf_log": "#f58518", "freundlich": "#2f6f9f"}


def data_support(df: pd.DataFrame) -> pd.DataFrame:
    edges = [0, 1e-3, 1e-2, 1e-1, 1, 10, 100, np.inf]
    rows = []
    for lo, hi in zip(edges[:-1], edges[1:]):
        part = df[(df["Ce"] > lo) & (df["Ce"] <= hi)]
        rows.append({"Ce_low_mg_L": lo, "Ce_high_mg_L": hi, "n_obs": len(part),
                     "n_studies": part[GROUP_COL].nunique(),
                     "studies": "; ".join(sorted(part[GROUP_COL].unique())) if len(part) else "",
                     "qe_median_mg_g": part[TARGET].median()})
    return pd.DataFrame(rows)


def response_curves(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for name in LAB_MODELS:
        model = fit_lab_model(name, df)
        for polymer in POLYMERS:
            pred = model.predict(field_scenario_frame([polymer], CE_GRID))
            rows.append(pd.DataFrame({"model": name, "ReT": polymer, "Ce": CE_GRID, "qe": pred}))
    return pd.concat(rows, ignore_index=True)


def extrapolation_test(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for threshold in THRESHOLDS_MG_L:
        train, test = df[df["Ce"] > threshold], df[(df["Ce"] <= threshold) & (df["Ce"] > 0)]
        if len(test) < 10:
            continue
        for name in LAB_MODELS:
            pred = fit_lab_model(name, train).predict(test)
            bias = np.median(np.log10(np.clip(pred, 1e-9, None)) - np.log10(test[TARGET]))
            rows.append({"train_Ce_above_mg_L": threshold, "model": name, "n_train": len(train),
                         "n_test": len(test), "n_test_studies": test[GROUP_COL].nunique(),
                         "median_bias_factor": float(10 ** bias),
                         **point_metrics(test[TARGET].to_numpy(), pred)})
    return pd.DataFrame(rows)


def save_plot(curves: dict[str, pd.DataFrame], data: dict[str, pd.DataFrame]) -> Path:
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.8), sharey=True)
    for ax, subset in zip(axes, curves):
        df = data[subset]
        obs = df[df["ReT"].isin(POLYMERS) & (df["Ce"] > 0)]
        ax.scatter(obs["Ce"], obs[TARGET], s=9, color="0.6", alpha=0.6, label="Laboratory data (PP, PE)")
        for name in LAB_MODELS:
            part = curves[subset][curves[subset]["model"] == name].groupby("Ce")["qe"].mean()
            ax.plot(part.index, part.values, color=MODEL_COLORS[name], linewidth=2, label=MODEL_LABELS[name])
        ax.axvspan(*FIELD_CE_RANGE, color="#54a24b", alpha=0.15, label="Typical river Ce")
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel("Ce (mg/L)")
        ax.set_title(f"{subset.capitalize()} studies (n = {len(df)}, {df[GROUP_COL].nunique()} studies)", fontsize=11)
        ax.grid(which="major", linestyle=":", alpha=0.5)
    axes[0].set_ylabel("qe (mg/g)")
    axes[0].legend(frameon=False, fontsize=8)
    fig.tight_layout()
    output = FIGURES_DIR / "low_ce_response.png"
    fig.savefig(output, dpi=300)
    plt.close(fig)
    return output


def main() -> None:
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    data = {subset: load_data(subset) for subset in ["all", "freshwater"]}
    support, curves, tests = [], {}, []
    for subset, df in data.items():
        support.append(data_support(df).assign(subset=subset))
        curves[subset] = response_curves(df)
        tests.append(extrapolation_test(df).assign(subset=subset))

    pd.concat(support).to_csv(REPORTS_DIR / "low_ce_data_support.csv", index=False)
    all_curves = pd.concat([c.assign(subset=s) for s, c in curves.items()])
    all_curves.to_csv(REPORTS_DIR / "low_ce_response_curves.csv", index=False)
    test_table = pd.concat(tests, ignore_index=True)
    test_table.to_csv(REPORTS_DIR / "low_ce_extrapolation_test.csv", index=False)

    nearest = CE_GRID[np.argmin(np.abs(CE_GRID - GLOBAL_CU_MG_L))]
    at_field = (all_curves[np.isclose(all_curves["Ce"], nearest)]
                .groupby(["subset", "model"])["qe"].mean().unstack())
    print(f"Laboratory qe (mg/g) for 50:50 PP/PE at Ce = {nearest:.4f} mg/L")
    print(at_field.to_string(float_format=lambda v: f"{v:.4g}"))
    print("\nConcentration hold-out (train above threshold, test below)")
    print(test_table[["subset", "train_Ce_above_mg_L", "model", "n_test", "n_test_studies",
                      "median_bias_factor", "median_fold_error", "r2_log10"]]
          .to_string(index=False, float_format=lambda v: f"{v:.3g}"))
    print(f"Saved figure: {save_plot(curves, data)}")


if __name__ == "__main__":
    main()
