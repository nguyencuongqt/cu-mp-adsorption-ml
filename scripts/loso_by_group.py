"""Leave-one-study-out error by polymer, ageing status and adsorption system (Table S9).

Reads the out-of-study predictions of src/analysis/final_model_validation.py.
Output: outputs/reports/final_nested_loso_by_group.csv
"""

import pandas as pd

p = pd.read_csv("outputs/reports/final_nested_loso_predictions.csv")
d = pd.read_csv("data/dataset_grouped.csv")[["curve_id", "ReT", "AgS", "AdC"]].drop_duplicates("curve_id")
m = p.merge(d, on="curve_id", how="left", validate="m:1")
rows = []
for model in ["freundlich_pooled", "lgbm_splice_nested"]:
    e = m[model] - m["log_obs"]
    for col in ["ReT", "AgS", "AdC"]:
        for k, g in m.groupby(col):
            rows.append({"model": model, "group": col, "level": k, "n_points": len(g), "n_studies": g["source_id"].nunique(),
                         "median_fold_error": 10 ** e[g.index].abs().median(), "bias_as_factor": 10 ** e[g.index].mean()})
out = pd.DataFrame(rows)
out.to_csv("outputs/reports/final_nested_loso_by_group.csv", index=False)
print(out[out.model == "freundlich_pooled"].drop(columns="model").round(2).to_string(index=False))
