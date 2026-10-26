"""Coverage of quantile-regression-forest (QRF) 90% intervals on the final dataset, by validation design.

The QRF interval is the conventional choice that the study-level conformal interval is compared with
(Section 3.1.2). Coverage is computed for a random split and for studies left out of training, for the seven
predictors of the conventional workflow ("full": polymer, aging, system, Ce, temperature, pH, agitation) and
for the four predictors of the final models ("core"), on the untransformed and on the log10 target.

Output: outputs/reports/qrf_coverage_final_dataset.csv
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.base import clone

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from config import GROUP_COL, REPORTS_DIR, TARGET  # noqa: E402
from src.analysis.grouped_validation import (  # noqa: E402
    FEATURE_SETS, QRF_QUANTILES, _log10, _pow10, export_tuned_params, interval_metrics, iter_splits, load_data, make_pipeline,
    point_metrics,
)

REPEATS = 5


def main() -> None:
    params = export_tuned_params()
    df = load_data("all")
    y, groups = df[TARGET], df[GROUP_COL]
    y_range = float(y.max() - y.min())
    rows = []
    for feature_set in ["full", "core"]:
        features = FEATURE_SETS[feature_set]
        x = df[features]
        for target in ["raw", "log"]:
            pipe = make_pipeline("QRF", params["RandomForest"], features, target)
            y_fit = pd.Series(_log10(y)) if target == "log" else y
            for scheme in ["random", "loso"]:
                parts = []
                for rep, _, tr, te in iter_splits(scheme, y, groups, REPEATS):
                    est = clone(pipe).fit(x.iloc[tr], y_fit.iloc[tr])
                    q = np.asarray(est.predict(x.iloc[te], quantiles=QRF_QUANTILES))
                    q = _pow10(q) if target == "log" else q
                    parts.append(pd.DataFrame({"repeat": rep, GROUP_COL: groups.iloc[te].to_numpy(), "y": y.iloc[te].to_numpy(),
                                               "lo": q[:, 0], "pred": q[:, 1], "hi": q[:, 2]}))
                oof = pd.concat(parts, ignore_index=True)
                inside = (oof["y"] >= oof["lo"]) & (oof["y"] <= oof["hi"])
                by_study = inside.groupby(oof[GROUP_COL]).mean()
                rows.append({"n_points": len(df), "n_studies": int(groups.nunique()), "predictors": feature_set, "target": target,
                             "scheme": scheme, "coverage_points": float(inside.mean()),
                             "coverage_mean_over_studies": float(by_study.mean()),
                             "studies_with_coverage_below_50pct": int((by_study < 0.5).sum()),
                             "niw": interval_metrics(oof["y"].to_numpy(), oof["lo"].to_numpy(), oof["hi"].to_numpy(), y_range)["niw"],
                             "median_fold_error": point_metrics(oof["y"].to_numpy(), oof["pred"].to_numpy())["median_fold_error"]})
                print(f"{feature_set:<5} {target:<3} {scheme:<6} coverage = {100 * inside.mean():.1f}% of points, "
                      f"{100 * by_study.mean():.1f}% mean over studies", flush=True)
    out = pd.DataFrame(rows)
    out.to_csv(REPORTS_DIR / "qrf_coverage_final_dataset.csv", index=False)
    pd.set_option("display.width", 200)
    print(out.round(3).to_string(index=False))


if __name__ == "__main__":
    main()
