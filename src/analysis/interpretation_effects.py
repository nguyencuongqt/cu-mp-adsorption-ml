"""Interpretable effects of the reduced-predictor models (replaces the Ce50 summary).

1. Pooled Freundlich model: log10(qe) = a + n*log10(Ce) + polymer + aging + system.
   Reported as the exponent n and as multiplicative factors on qe (aged vs pristine,
   mixed vs single-ion, each polymer vs PE), with 95% intervals from a bootstrap that
   resamples whole studies (cluster bootstrap), because points of one study are not
   independent.
2. Partial dependence of qe on Ce for the monotone LightGBM benchmark, compared with
   the Freundlich line and with the range of the laboratory data.

Outputs: interpretation_freundlich_effects.csv, interpretation_partial_dependence.csv
(outputs/reports) and interpretation_effects.png (outputs/figures).
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

from config import FIGURES_DIR, GROUP_COL, RANDOM_SEED, REPORTS_DIR, TARGET  # noqa: E402
from src.analysis.grouped_validation import export_tuned_params, load_data  # noqa: E402
from src.analysis.transfer_analysis import fit_predict  # noqa: E402

N_BOOT = 1000
REFERENCE = {"ReT": "PE", "AgS": "Virgin", "AdC": "Single-ion", "Ce": 1.0}
PD_GRID = np.logspace(-3, 2, 61)
MIN_STUDIES_FOR_POLYMER = 3


def contrast_frame(df: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    """Rows whose predictions give every effect as a ratio to the reference row."""
    rows, labels = [dict(REFERENCE)], ["reference: PE, pristine, single-ion, Ce = 1 mg/L (qe, mg/g)"]
    rows.append({**REFERENCE, "Ce": 10.0}); labels.append("Freundlich exponent n")
    rows.append({**REFERENCE, "AgS": "Aged"}); labels.append("aged vs pristine")
    other_system = [v for v in df["AdC"].unique() if v != REFERENCE["AdC"]]
    for v in other_system:
        rows.append({**REFERENCE, "AdC": v}); labels.append(f"{v} vs single-ion system")
    for p in sorted(df["ReT"].unique()):
        if p != "PE":
            rows.append({**REFERENCE, "ReT": p}); labels.append(f"{p} vs PE")
    return pd.DataFrame(rows), labels


def effects(train: pd.DataFrame, frame: pd.DataFrame) -> np.ndarray:
    """log10 effects; an effect is NaN when its category is absent from the training sample."""
    log_pred = np.log10(fit_predict("freundlich_pooled", train, frame, {}))
    out = log_pred - log_pred[0]
    out[0] = log_pred[0]
    for i, row in frame.iterrows():
        if any(row[c] not in set(train[c]) for c in ["ReT", "AgS", "AdC"]):
            out[i] = np.nan
    return out


def freundlich_table(df: pd.DataFrame) -> pd.DataFrame:
    frame, labels = contrast_frame(df)
    point = effects(df, frame)
    rng = np.random.default_rng(RANDOM_SEED)
    studies = df[GROUP_COL].unique()
    parts = {s: g for s, g in df.groupby(GROUP_COL)}
    boot = np.array([effects(pd.concat([parts[s] for s in rng.choice(studies, len(studies))], ignore_index=True), frame)
                     for _ in range(N_BOOT)])
    lo, hi = np.nanpercentile(boot, 2.5, axis=0), np.nanpercentile(boot, 97.5, axis=0)
    n_studies = {p: df.loc[df["ReT"] == p, GROUP_COL].nunique() for p in df["ReT"].unique()}
    rows = []
    for i, label in enumerate(labels):
        is_slope = label.startswith("Freundlich")
        conv = (lambda v: v) if is_slope else (lambda v: 10 ** v)
        polymer = label.split(" vs ")[0] if label.endswith("vs PE") else None
        rows.append({"effect": label, "estimate": conv(point[i]), "ci95_low": conv(lo[i]), "ci95_high": conv(hi[i]),
                     "scale": "exponent" if is_slope else ("mg/g" if i == 0 else "factor on qe"),
                     "n_studies": n_studies.get(polymer, df[GROUP_COL].nunique()) if polymer else df[GROUP_COL].nunique(),
                     "interval_excludes_no_effect": bool(not is_slope and i > 0 and (lo[i] > 0 or hi[i] < 0)),
                     "note": "fewer than 3 studies: not interpretable" if polymer and n_studies[polymer] < MIN_STUDIES_FOR_POLYMER else ""})
    return pd.DataFrame(rows)


def partial_dependence(df: pd.DataFrame) -> pd.DataFrame:
    """Geometric-mean prediction over the observed polymer/aging/system mix at each Ce."""
    rf_params = export_tuned_params()["RandomForest"]
    base = df[["ReT", "AgS", "AdC"]]
    frame = pd.concat([base.assign(Ce=c) for c in PD_GRID], ignore_index=True)
    out = pd.DataFrame({"Ce_mg_L": PD_GRID})
    for model in ["freundlich_pooled", "lgbm_monotone", "xgb_monotone", "rf_raw"]:
        pred = np.log10(fit_predict(model, df, frame, rf_params)).reshape(len(PD_GRID), len(base))
        out[f"qe_{model}_mg_g"] = 10 ** pred.mean(axis=1)
    return out


def save_figure(df: pd.DataFrame, table: pd.DataFrame, pd_curve: pd.DataFrame) -> Path:
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11.5, 4.6))
    part = table[(table["scale"] == "factor on qe") & (table["note"] == "")].iloc[::-1]
    y = np.arange(len(part))
    ax1.errorbar(part["estimate"], y, xerr=[part["estimate"] - part["ci95_low"], part["ci95_high"] - part["estimate"]],
                 fmt="o", color="#2f6f9f", capsize=3)
    ax1.axvline(1.0, color="0.4", linestyle="--", linewidth=0.8)
    ax1.set_xscale("log"); ax1.set_yticks(y); ax1.set_yticklabels(part["effect"])
    ax1.set_xlabel("Factor on qe (pooled Freundlich; 95% CI, study bootstrap)")
    ax1.set_title("a", loc="left", fontsize=11)
    ax2.scatter(df["Ce"], df[TARGET], s=5, color="0.8", label="laboratory data", zorder=1)
    styles = {"freundlich_pooled": ("Pooled Freundlich", "#2f6f9f"), "lgbm_monotone": ("Monotone LightGBM", "#f58518"),
              "xgb_monotone": ("Monotone XGBoost", "#54a24b"), "rf_raw": ("Random Forest", "#c43c39")}
    for key, (label, color) in styles.items():
        ax2.plot(pd_curve["Ce_mg_L"], pd_curve[f"qe_{key}_mg_g"], label=label, color=color, zorder=2)
    ax2.axvspan(0.001, 0.01, color="tab:blue", alpha=0.10, label="typical river Ce")
    ax2.set_xscale("log"); ax2.set_yscale("log")
    ax2.set_xlabel("Ce (mg/L)"); ax2.set_ylabel("Partial dependence of qe (mg/g)")
    ax2.set_title("b", loc="left", fontsize=11); ax2.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    path = FIGURES_DIR / "interpretation_effects.png"
    fig.savefig(path, dpi=300)
    plt.close(fig)
    return path


def main() -> None:
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    df = load_data("all")
    table = freundlich_table(df)
    table.to_csv(REPORTS_DIR / "interpretation_freundlich_effects.csv", index=False)
    pd_curve = partial_dependence(df)
    pd_curve.to_csv(REPORTS_DIR / "interpretation_partial_dependence.csv", index=False)
    pd.set_option("display.width", 220)
    print(table.to_string(index=False, float_format=lambda v: f"{v:.3g}"))
    slopes = np.diff(np.log10(pd_curve.drop(columns="Ce_mg_L")), axis=0) / np.diff(np.log10(pd_curve["Ce_mg_L"]))[:, None]
    mid = (pd_curve["Ce_mg_L"].to_numpy()[1:] * pd_curve["Ce_mg_L"].to_numpy()[:-1]) ** 0.5
    print("\nLocal log-log slope of the partial dependence on Ce")
    for lo, hi in [(0.001, 0.01), (0.01, 0.1), (0.1, 1), (1, 10), (10, 100)]:
        m = (mid >= lo) & (mid < hi)
        print(f"  Ce {lo}-{hi} mg/L: " + ", ".join(f"{c[3:-5]} {slopes[m, j].mean():.2f}"
                                                 for j, c in enumerate(pd_curve.columns[1:])))
    print(f"Figure: {save_figure(df, table, pd_curve)}")


if __name__ == "__main__":
    main()
