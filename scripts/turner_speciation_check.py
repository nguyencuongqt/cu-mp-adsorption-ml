"""Held-out error for Turner and Holmes (2015), the only Cu isotherms of the dataset measured in natural river water.

The pooled Freundlich model is fitted without Turner_2015 and predicts its 10 points (i) at the measured dissolved Cu and
(ii) at the free Cu2+ concentration, taken as 1.7 % of dissolved Cu (WHAM v7 speciation for Plym river water, DOC about
2 mg/L, pH 6.5; 98.2 % organically complexed; Turner and Holmes, 2015, Table 2). Output: outputs/reports/turner_heldout_speciation.csv
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import src.analysis.faf_scaling as fs  # noqa: E402

FREE_FRACTION = 0.017


def main() -> None:
    lab = fs.load_data("all")
    grouped = pd.read_csv(ROOT / "data" / "dataset_grouped.csv")
    mask = (grouped["source_id"] == "Turner_2015").to_numpy()
    params = fs.export_tuned_params()["RandomForest"]
    rows = []
    for model in ["freundlich_pooled", "xgb_blend"]:
        for label, factor in [("dissolved Cu (as measured)", 1.0), ("free Cu2+ (1.7 % of dissolved)", FREE_FRACTION)]:
            test = lab[mask].copy()
            test["Ce"] = test["Ce"] * factor
            pred = fs.fit_predict(model, lab[~mask], test, params)
            err = np.log10(pred / lab[mask]["qe"].to_numpy())
            rows.append({"model": model, "Ce_basis": label, "n_points": int(mask.sum()),
                         "median_fold_error": 10 ** np.median(np.abs(err)), "mean_log10_error": err.mean(),
                         "bias_as_factor": 10 ** err.mean()})
    out = pd.DataFrame(rows)
    out.to_csv(ROOT / "outputs" / "reports" / "turner_heldout_speciation.csv", index=False)
    print(out.round(3).to_string(index=False))


if __name__ == "__main__":
    main()
