"""End-to-end orchestrator for the Cu2+-MP adsorption prediction project."""

from __future__ import annotations

import argparse
import importlib.util
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Callable

import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

import config
from src.analysis.data_cleaning import clean_model_data


if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")

ROOT = Path(__file__).parent
SRC = ROOT / "src"


def load_module(name: str, path: Path) -> Any:
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load {name} from {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module  # required for @dataclass to resolve type annotations
    spec.loader.exec_module(module)
    return module


def stratified_bins(y: pd.Series, n_splits: int = 5) -> pd.Series:
    max_bins = min(10, y.nunique(), len(y) // n_splits)
    for n_bins in range(max_bins, 1, -1):
        bins = pd.qcut(y, q=n_bins, labels=False, duplicates="drop")
        if pd.Series(bins).value_counts().min() >= n_splits:
            return bins.astype(int)
    return pd.Series(np.zeros(len(y), dtype=int))


def load_raw_data() -> pd.DataFrame:
    data_file = Path(config.DATA_FILE)
    if data_file.suffix.lower() in {".xlsx", ".xls"}:
        return pd.read_excel(data_file)
    return pd.read_csv(data_file)


def prepare_split(raw_df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    clean = clean_model_data(raw_df)
    feature_cols = config.CAT_COLS + config.NUM_COLS
    y = pd.to_numeric(clean[config.TARGET], errors="coerce")
    bins = stratified_bins(y, n_splits=5)
    return train_test_split(
        clean[feature_cols],
        y,
        test_size=config.TEST_SIZE,
        random_state=config.RANDOM_SEED,
        stratify=bins,
    )


def run_step(step: int, name: str, func: Callable[[], Any], state: dict[str, Any]) -> Any:
    start = time.perf_counter()
    try:
        result = func()
        elapsed = time.perf_counter() - start
        print(f"[Step {step}] {name} — done in {elapsed:.1f}s")
        state["steps"][step] = {"name": name, "status": "done", "seconds": elapsed}
        return result
    except Exception as exc:
        elapsed = time.perf_counter() - start
        print(f"[Step {step}] {name} — failed in {elapsed:.1f}s: {exc}")
        print(f"Hint: fix the input/artifact above and resume with: python main.py --from {step}")
        state["steps"][step] = {"name": name, "status": "failed", "seconds": elapsed, "error": str(exc)}
        return None


def step_01_prepare(state: dict[str, Any]) -> dict[str, Any]:
    config.DATA_DIR.mkdir(parents=True, exist_ok=True)
    config.FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    config.MODEL_DIR.mkdir(parents=True, exist_ok=True)
    config.REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    prep = load_module("prepare_data", ROOT / "scripts" / "prepare_data.py")
    prep.main()
    raw_df = load_raw_data()
    model_df = clean_model_data(raw_df)
    x_train, x_test, y_train, y_test = prepare_split(raw_df)
    state.update(
        {
            "raw_df": model_df,
            "raw_input_df": raw_df,
            "X_train": x_train,
            "X_test": x_test,
            "y_train": y_train,
            "y_test": y_test,
        }
    )
    return state


def step_02_missing(state: dict[str, Any]) -> Any:
    mod = load_module("missing_diagnosis", SRC / "analysis" / "missing_diagnosis.py")
    result = mod.run(state["raw_df"])
    state["missing"] = result
    return result


def step_03_imputation(state: dict[str, Any]) -> Any:
    mod = load_module("imputation_comparison", SRC / "analysis" / "imputation_comparison.py")
    best_name, imputer = mod.run(state["raw_df"])
    state["best_imputation"] = best_name
    state["best_imputer"] = imputer
    return best_name


def step_04_models(state: dict[str, Any]) -> Any:
    mod = load_module("model_training", SRC / "analysis" / "model_training.py")
    best_model, best_name, results = mod.run(
        state["X_train"], state["y_train"], state["X_test"], state["y_test"]
    )
    state["best_model"] = best_model
    state["best_model_name"] = best_name
    state["model_results"] = results
    return results


def step_05_shap(state: dict[str, Any]) -> Any:
    mod = load_module("shap_analysis", SRC / "analysis" / "shap_analysis.py")
    rf_path = config.MODEL_DIR / "RandomForest_tuned.joblib"
    if rf_path.exists():
        model = joblib.load(rf_path)
    else:
        model = state.get("best_model")
        if model is None:
            raise FileNotFoundError("RandomForest_tuned.joblib not found. Run step 04 first.")
    result = mod.run(model, state["X_train"], state["X_test"], state["y_test"], feature_names=list(state["X_train"].columns))
    state["shap_summary"] = result
    return result


def step_06_qrf(state: dict[str, Any]) -> Any:
    mod = load_module("uncertainty_qrf", SRC / "analysis" / "uncertainty_qrf.py")
    qrf_model, med, lo, hi, metrics = mod.run(state["X_train"], state["y_train"], state["X_test"], state["y_test"])
    state.update({"qrf_model": qrf_model, "y_pred_median": med, "y_pred_lower": lo, "y_pred_upper": hi, "qrf_metrics": metrics})
    return metrics


def step_07_faf(state: dict[str, Any]) -> Any:
    mod = load_module("faf_calibration", SRC / "analysis" / "faf_calibration.py")
    field_data = config.DATA_DIR / "field_data.csv"
    report_data = config.REPORTS_DIR / "field_data.csv"
    if field_data.exists():
        shutil.copy2(field_data, report_data)
    central, ci, river_preds = mod.run(
        X_test=state.get("X_test"),
        y_pred_median=state.get("y_pred_median"),
        y_pred_lower=state.get("y_pred_lower"),
        y_pred_upper=state.get("y_pred_upper"),
        qrf_model=state.get("qrf_model"),
    )
    state.update({"faf_central": central, "faf_ci": ci, "river_predictions": river_preds})
    return central, ci


def step_08_global(state: dict[str, Any]) -> Any:
    mod = load_module("global_estimation", SRC / "analysis" / "global_estimation.py")
    global_mass, river_df = mod.run(
        qrf_model=state.get("qrf_model"),
        faf_central=state.get("faf_central"),
        faf_ci=state.get("faf_ci"),
    )
    state.update({"global_mass_kg": global_mass, "river_df": river_df})
    return global_mass


def step_09_ph(state: dict[str, Any]) -> Any:
    mod = load_module("ph_sensitivity", SRC / "analysis" / "ph_sensitivity.py")
    df, robust = mod.run(state["raw_df"])
    state.update({"ph_sensitivity": df, "ph_robust": robust})
    return robust


def step_10_llm(state: dict[str, Any], only_eval: bool = False) -> Any:
    rag_dir = SRC / "rag"
    if only_eval:
        eval_mod = load_module("ablation_evaluation", rag_dir / "ablation_evaluation.py")
        result = eval_mod.run_evaluation()
        state["ablation"] = result
        return result
    ingest = load_module("document_ingestion", rag_dir / "document_ingestion.py")
    vector = load_module("vector_store", rag_dir / "vector_store.py")
    kg = load_module("knowledge_graph", rag_dir / "knowledge_graph.py")
    chunks = ingest.ingest_all()
    index = vector.build_index()
    graph = kg.build_graph()
    state.update({"chunks": chunks, "faiss": index, "kg": graph})
    return {"chunks": len(chunks)}


def step_11_ui(state: dict[str, Any]) -> Any:
    app = SRC / "ui" / "app.py"
    if not app.exists():
        raise FileNotFoundError(app)
    print("UI ready. Launch with: streamlit run src/ui/app.py")
    return app


def summarize(state: dict[str, Any]) -> None:
    model_results = state.get("model_results", {}).get("models", {})
    best_name = state.get("best_model_name", "NA")
    best_metrics = model_results.get(best_name, {}).get("metrics", {})
    qrf = state.get("qrf_metrics", {})
    missing = state.get("missing", {}).get("little_mcar", {})
    missing_label = "MAR/MNAR" if missing.get("p_value", 1) < 0.05 else "MCAR"
    ph_df = state.get("ph_sensitivity")
    max_delta = np.nan
    if isinstance(ph_df, pd.DataFrame) and not ph_df.empty:
        baseline = ph_df.loc[ph_df["scenario"] == "imputed", "MAE"].iloc[0]
        max_delta = float((ph_df["MAE"] - baseline).abs().max())
    faf_ci = state.get("faf_ci", (np.nan, np.nan))
    river_df = state.get("river_df")
    global_low = global_high = np.nan
    if isinstance(river_df, pd.DataFrame) and "global_mass_kg_yr" in river_df.columns:
        mass = float(river_df["global_mass_kg_yr"].iloc[0])
        low_ratio = state.get("faf_ci", (np.nan, np.nan))[0] / state.get("faf_central", np.nan)
        high_ratio = state.get("faf_ci", (np.nan, np.nan))[1] / state.get("faf_central", np.nan)
        global_low, global_high = mass * low_ratio, mass * high_ratio
    ablation_summary = {"H1": "✗", "H2": "✗", "H3": "✗"}
    summary_path = config.REPORTS_DIR / "ablation_summary.txt"
    if summary_path.exists():
        text = summary_path.read_text(encoding="utf-8", errors="ignore")
        ablation_summary["H1"] = "✓" if "H1" in text and "supported" in text.splitlines()[0] else "✗"
        ablation_summary["H2"] = "✓" if len(text.splitlines()) > 1 and "supported" in text.splitlines()[1] else "✗"
        ablation_summary["H3"] = "✓" if len(text.splitlines()) > 2 and "supported" in text.splitlines()[2] else "✗"

    if summary_path.exists():
        text = summary_path.read_text(encoding="utf-8", errors="ignore")
        hypothesis_lines = [line for line in text.splitlines() if line.startswith("H")]
        for idx, key in enumerate(("H1", "H2", "H3")):
            if idx < len(hypothesis_lines):
                line = hypothesis_lines[idx].lower()
                ablation_summary[key] = "yes" if "not supported" not in line and "supported" in line else "no"

    print("\nSummary")
    print("=" * 72)
    print(f"Best model     : {best_name or 'NA'} | MAE={best_metrics.get('oof_mae', np.nan):.4g} | R²={best_metrics.get('oof_r2', np.nan):.4g}")
    print(f"QRF            : Coverage={qrf.get('coverage', np.nan) * 100:.2f}% | NIW={qrf.get('niw', np.nan):.4g}")
    print(f"Missing (pH)   : {missing_label} | best imputation={state.get('best_imputation', 'NA')}")
    print(f"pH robust      : {'Yes' if state.get('ph_robust') else 'No'} | max ΔMAE={max_delta:.4g}")
    print(f"FAF            : {state.get('faf_central', np.nan):.4g} (CI: {faf_ci[0]:.4g}–{faf_ci[1]:.4g})")
    print(f"Global Cu mass : {state.get('global_mass_kg', np.nan):.4g} kg/yr (95% CI: {global_low:.4g}–{global_high:.4g})")
    print(f"LLM ablation   : H1={ablation_summary['H1']} | H2={ablation_summary['H2']} | H3={ablation_summary['H3']}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the Cu2+-MP adsorption pipeline.")
    parser.add_argument("--skip-llm", action="store_true", help="Skip module 10.")
    parser.add_argument("--skip-ui", action="store_true", help="Skip module 11.")
    parser.add_argument("--only-eval", action="store_true", help="Run only 10f ablation evaluation.")
    parser.add_argument("--from", dest="from_step", type=int, default=1, help="Resume from step N (1-11).")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    state: dict[str, Any] = {"steps": {}}
    if args.only_eval:
        run_step(10, "10f_ablation_evaluation", lambda: step_10_llm(state, only_eval=True), state)
        summarize(state)
        return

    if args.from_step > 1 and args.from_step < 10:
        try:
            raw_df = load_raw_data()
            model_df = clean_model_data(raw_df)
            x_train, x_test, y_train, y_test = prepare_split(raw_df)
            state.update(
                {
                    "raw_df": model_df,
                    "raw_input_df": raw_df,
                    "X_train": x_train,
                    "X_test": x_test,
                    "y_train": y_train,
                    "y_test": y_test,
                }
            )
        except Exception as exc:
            print(f"Could not rebuild in-memory state for --from {args.from_step}: {exc}")
            print("Hint: run python main.py --from 1 to recreate prepared data and split.")

    steps: list[tuple[int, str, Callable[[], Any]]] = [
        (1, "prepare_data_and_split", lambda: step_01_prepare(state)),
        (2, "missing_diagnosis", lambda: step_02_missing(state)),
        (3, "imputation_comparison", lambda: step_03_imputation(state)),
        (4, "model_training", lambda: step_04_models(state)),
        (5, "shap_analysis", lambda: step_05_shap(state)),
        (6, "uncertainty_qrf", lambda: step_06_qrf(state)),
        (7, "faf_calibration", lambda: step_07_faf(state)),
        (8, "global_estimation", lambda: step_08_global(state)),
        (9, "ph_sensitivity", lambda: step_09_ph(state)),
    ]
    if not args.skip_llm:
        steps.append((10, "rag_kg_llm_artifacts", lambda: step_10_llm(state)))
    if not args.skip_ui:
        steps.append((11, "streamlit_ui_ready", lambda: step_11_ui(state)))

    for step, name, func in steps:
        if step < args.from_step:
            continue
        result = run_step(step, name, func, state)
        if step == 1 and result is None:
            break
        if step > 1 and "raw_df" not in state and step < 10:
            print("State is missing raw data. Resume from step 1 to rebuild in-memory outputs.")
            break

    summarize(state)


if __name__ == "__main__":
    main()
