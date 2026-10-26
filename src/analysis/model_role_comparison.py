"""How much does the ML model add to the pooled Freundlich model, and where?

1. Paired comparison of absolute log10 errors of the main model (lgbm_splice) and the pooled
   Freundlich model, for random CV, leave-one-isotherm-out and leave-one-study-out, with a 95%
   interval from a bootstrap over studies (points of one study are not independent).
2. A Freundlich model with interactions (polymer x ageing, polymer-specific exponent): does a
   linear model with the same flexibility close the gap inside the laboratory domain?
3. Share of studies in which each model is the better one.

Output: outputs/reports/model_role_comparison.csv, model_role_by_study.csv
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from config import GROUP_COL, RANDOM_SEED, REPORTS_DIR, TARGET  # noqa: E402
from src.analysis.grouped_validation import export_tuned_params, load_data  # noqa: E402
from src.analysis.transfer_analysis import TIERS, fit_predict, iter_splits, log10  # noqa: E402

RF_PARAMS = export_tuned_params()["RandomForest"]
MODELS = ["lgbm_splice", "freundlich_pooled", "freundlich_interactions"]
N_BOOT = 2000


def freundlich_interactions(train: pd.DataFrame, test: pd.DataFrame) -> np.ndarray:
    """log qe ~ polymer-specific intercept and exponent + polymer x ageing + system."""
    def design(d: pd.DataFrame, columns=None) -> pd.DataFrame:
        pol = pd.get_dummies(d["ReT"], dtype=float)
        x = pol.add_prefix("a_")
        x = x.join(pol.mul(np.log10(d["Ce"].to_numpy()), axis=0).add_prefix("n_"))
        x = x.join(pol.mul((d["AgS"] == "Aged").astype(float).to_numpy(), axis=0).add_prefix("aged_"))
        x["mixed"] = (d["AdC"] == "Mixed").astype(float).to_numpy()
        return x if columns is None else x.reindex(columns=columns, fill_value=0.0)

    xtr = design(train)
    model = LinearRegression(fit_intercept=False).fit(xtr, log10(train[TARGET]))
    return model.predict(design(test, list(xtr.columns)))


def predict(model: str, train: pd.DataFrame, test: pd.DataFrame) -> np.ndarray:
    if model == "freundlich_interactions":
        return freundlich_interactions(train, test)
    return np.log10(fit_predict(model, train, test, RF_PARAMS))


def main() -> None:
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    df = load_data("all")
    rng = np.random.default_rng(RANDOM_SEED)
    rows, by_study = [], []
    for tier in TIERS:
        abs_err = {m: np.zeros(len(df)) for m in MODELS}
        count = np.zeros(len(df))
        for _, tr, te in iter_splits(tier, df):
            train, test = df.iloc[tr], df.iloc[te]
            for m in MODELS:
                abs_err[m][te] += np.abs(predict(m, train, test) - log10(test[TARGET]))
            count[te] += 1
        e = pd.DataFrame({m: v / count for m, v in abs_err.items()})
        e[GROUP_COL] = df[GROUP_COL].to_numpy()
        studies = e[GROUP_COL].unique()
        parts = {s: g for s, g in e.groupby(GROUP_COL)}
        med = e.groupby(GROUP_COL)[MODELS].median()
        for other in ["freundlich_pooled", "freundlich_interactions"]:
            diff = []
            for _ in range(N_BOOT):
                b = pd.concat([parts[s] for s in rng.choice(studies, len(studies))])
                diff.append(np.median(b["lgbm_splice"]) - np.median(b[other]))
            lo, hi = np.percentile(diff, [2.5, 97.5])
            rows.append({"tier": tier, "comparison": f"lgbm_splice vs {other}",
                         "fold_error_lgbm_splice": 10 ** e["lgbm_splice"].median(), "fold_error_other": 10 ** e[other].median(),
                         "ratio_of_fold_errors": 10 ** (e["lgbm_splice"].median() - e[other].median()),
                         "ratio_ci95_low": 10 ** lo, "ratio_ci95_high": 10 ** hi,
                         "difference_clear": bool(hi < 0 or lo > 0),
                         "studies_where_lgbm_splice_is_better": int((med["lgbm_splice"] < med[other]).sum()), "n_studies": len(med)})
        by_study.append((10 ** med).assign(tier=tier).reset_index())
        print(f"{tier} done", flush=True)
    out = pd.DataFrame(rows)
    out.to_csv(REPORTS_DIR / "model_role_comparison.csv", index=False)
    pd.concat(by_study).to_csv(REPORTS_DIR / "model_role_by_study.csv", index=False)
    pd.set_option("display.width", 250)
    print(out.to_string(index=False, float_format=lambda v: f"{v:.3g}"))


if __name__ == "__main__":
    main()
