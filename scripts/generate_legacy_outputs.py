"""Generate legacy-style figures and tables from Python pipeline outputs.

This script creates a publication/output bundle that mirrors the old R project
layout without overwriting the R archive. It does not copy old figures as a
substitute for reproducibility; it rebuilds legacy-named outputs from Python
CSV/PNG artifacts when available and writes a manifest showing what was
successfully regenerated.

Recommended order:
    python main.py --skip-llm --skip-ui
    python scripts/generate_legacy_outputs.py
"""

from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path
from typing import Iterable

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageFont, ImageOps

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import config  # noqa: E402


LEGACY_ROOT = ROOT / "outputs" / "legacy_style"
LEGACY_FIGURES = LEGACY_ROOT / "figures"
LEGACY_TABLES = LEGACY_ROOT / "tables"
LEGACY_VALIDATION = LEGACY_ROOT / "validation"

R_ROOT = ROOT.parent / "R_Cu"
R_FIG_DONE = ROOT.parent / "Fig done"
R_PDF_PLOT = ROOT.parent / "Pdf plot"


def ensure_dirs() -> None:
    for path in (LEGACY_ROOT, LEGACY_FIGURES, LEGACY_TABLES, LEGACY_VALIDATION):
        path.mkdir(parents=True, exist_ok=True)


def read_csv_optional(path: Path) -> pd.DataFrame | None:
    if not path.exists():
        return None
    return pd.read_csv(path)


def copy_table(src: Path, dst_name: str, manifest: list[dict[str, str]], required: bool = True) -> None:
    dst = LEGACY_TABLES / dst_name
    if src.exists():
        if src.suffix.lower() == ".txt" or dst.suffix.lower() == ".txt":
            shutil.copy2(src, dst)
        else:
            df = pd.read_csv(src)
            df.to_csv(dst, index=False, encoding="utf-8-sig")
        manifest.append({"artifact": dst_name, "type": "table", "status": "created", "source": str(src), "note": ""})
    else:
        status = "missing_required" if required else "missing_optional"
        manifest.append({"artifact": dst_name, "type": "table", "status": status, "source": str(src), "note": "Run the upstream Python step first."})


def save_table_from_df(df: pd.DataFrame, dst_name: str, manifest: list[dict[str, str]], source: str) -> None:
    dst = LEGACY_TABLES / dst_name
    df.to_csv(dst, index=False, encoding="utf-8-sig")
    manifest.append({"artifact": dst_name, "type": "table", "status": "created", "source": source, "note": ""})


def image_to_tiff(src: Path, dst_name: str, manifest: list[dict[str, str]], required: bool = True) -> None:
    dst = LEGACY_FIGURES / dst_name
    if not src.exists():
        status = "missing_required" if required else "missing_optional"
        manifest.append({"artifact": dst_name, "type": "figure", "status": status, "source": str(src), "note": "Run the upstream Python step first."})
        return
    with Image.open(src) as img:
        rgb = ImageOps.exif_transpose(img).convert("RGB")
        rgb.save(dst, compression="tiff_lzw")
    manifest.append({"artifact": dst_name, "type": "figure", "status": "created", "source": str(src), "note": ""})


def composite_figure_mpl(
    sources: list[Path],
    dst_name: str,
    manifest: list[dict[str, str]],
    labels: list[str] | None = None,
    dpi: int = 300,
    extra_dst: Path | None = None,
    label_fontsize: int = 28,
) -> None:
    """Composite panels with matplotlib imshow — no PIL resize, best quality for print/Word."""
    existing = [s for s in sources if s.exists()]
    dst = LEGACY_FIGURES / dst_name
    if not existing:
        manifest.append({"artifact": dst_name, "type": "figure", "status": "missing_required",
                         "source": "; ".join(map(str, sources)), "note": "No panel images available."})
        return

    labels = labels or [chr(ord("a") + i) for i in range(len(existing))]

    # Load full-resolution images as arrays (lossless, no PIL resize)
    imgs = []
    for src in existing:
        with Image.open(src) as im:
            imgs.append(np.asarray(ImageOps.exif_transpose(im).convert("RGB")))

    heights = [a.shape[0] for a in imgs]
    widths  = [a.shape[1] for a in imgs]

    # Physical dimensions in inches at target DPI
    # Each panel keeps its natural aspect ratio; all panels share the same height
    target_h_in = max(heights) / dpi
    panel_w_in  = [(w / h) * target_h_in for w, h in zip(widths, heights)]
    gap_in      = 0.15 * (len(imgs) - 1)
    total_w_in  = sum(panel_w_in) + gap_in

    fig = plt.figure(figsize=(total_w_in, target_h_in))
    fig.patch.set_facecolor("white")

    # Place each panel as a precisely sized axes
    x = 0.0
    for i, (arr, label, pw) in enumerate(zip(imgs, labels, panel_w_in)):
        left   = x / total_w_in
        width  = pw / total_w_in
        height = (arr.shape[0] / dpi) / target_h_in  # may be < 1 for shorter panels
        bottom = 1.0 - height                          # top-align panels
        ax = fig.add_axes([left, bottom, width, height])
        ax.imshow(arr, interpolation="lanczos", aspect="auto")
        ax.axis("off")
        ax.set_title(label, loc="left", fontsize=label_fontsize, fontweight="bold", pad=8)
        x += pw + (0.15 if i < len(imgs) - 1 else 0)

    save_kw = dict(dpi=dpi, bbox_inches="tight",
                   pil_kwargs={"compression": "tiff_lzw"})
    fig.savefig(str(dst), format="tiff", **save_kw)
    if extra_dst is not None:
        extra_dst.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(str(extra_dst), format="tiff", **save_kw)
    plt.close(fig)
    manifest.append({"artifact": dst_name, "type": "figure", "status": "created",
                     "source": "; ".join(map(str, existing)),
                     "note": "Composite built with matplotlib (high-quality, no PIL resize)."})


def composite_figure(
    sources: list[Path],
    dst_name: str,
    manifest: list[dict[str, str]],
    labels: list[str] | None = None,
    layout: str = "vertical",
    panel_gap: int = 70,
    dpi: int = 300,
    extra_dst: Path | None = None,
) -> None:
    existing = [src for src in sources if src.exists()]
    dst = LEGACY_FIGURES / dst_name
    if not existing:
        manifest.append({"artifact": dst_name, "type": "figure", "status": "missing_required", "source": "; ".join(map(str, sources)), "note": "No panel images available."})
        return
    labels = labels or [chr(ord("a") + i) for i in range(len(existing))]
    images = [ImageOps.exif_transpose(Image.open(src)).convert("RGB") for src in existing]
    try:
        font = ImageFont.truetype("arial.ttf", 52)
    except OSError:
        font = ImageFont.load_default()

    if layout == "horizontal":
        target_height = min(max(img.size[1] for img in images), 1200)
        resized = []
        for img in images:
            scale = target_height / img.size[1]
            resized.append(img.resize((int(img.size[0] * scale), target_height)))
        total_width = sum(img.size[0] for img in resized) + panel_gap * (len(resized) - 1) + 80
        canvas = Image.new("RGB", (total_width, target_height + 95), "white")
        x = 0
        for label, img in zip(labels, resized):
            canvas.paste(img, (x, 70))
            draw = ImageDraw.Draw(canvas)
            draw.text((x + 16, 12), label, fill="black", font=font)
            x += img.size[0] + panel_gap
    else:
        widths, heights = zip(*(img.size for img in images))
        target_width = max(widths)
        resized = []
        for img in images:
            scale = target_width / img.size[0]
            resized.append(img.resize((target_width, int(img.size[1] * scale))))
        total_height = sum(img.size[1] for img in resized) + 70 * len(resized)
        canvas = Image.new("RGB", (target_width, total_height), "white")
        y = 0
        for label, img in zip(labels, resized):
            draw = ImageDraw.Draw(canvas)
            draw.text((12, y + 18), label, fill="black", font=font)
            canvas.paste(img, (0, y + 70))
            y += img.size[1] + 70
    save_kwargs = {"compression": "tiff_lzw", "dpi": (dpi, dpi)}
    canvas.save(dst, **save_kwargs)
    if extra_dst is not None:
        extra_dst.parent.mkdir(parents=True, exist_ok=True)
        canvas.save(extra_dst, **save_kwargs)
    for img in images:
        img.close()
    manifest.append({"artifact": dst_name, "type": "figure", "status": "created", "source": "; ".join(map(str, existing)), "note": "Composite built from Python panels."})


def make_roadmap(manifest: list[dict[str, str]]) -> None:
    dst = LEGACY_FIGURES / "Fig_1_research_roadmap_python.tiff"
    fig, ax = plt.subplots(figsize=(11, 4.5))
    ax.axis("off")
    steps = [
        "Raw Cu dataset",
        "Missing diagnosis\npH MAR check",
        "Imputation\nsensitivity",
        "RF/QRF training",
        "SHAP + Ce50",
        "FAF calibration",
        "River/global\nestimation",
        "LLM Interpretation\nSystem",
    ]
    x = np.linspace(0.06, 0.94, len(steps))
    for i, (xi, label) in enumerate(zip(x, steps)):
        ax.text(xi, 0.55, label, ha="center", va="center", bbox=dict(boxstyle="round,pad=0.35", fc="#e8f1fb", ec="#4c78a8"))
        if i < len(steps) - 1:
            ax.annotate("", xy=(x[i + 1] - 0.045, 0.55), xytext=(xi + 0.045, 0.55), arrowprops=dict(arrowstyle="->", lw=1.2))
    ax.set_title("Python reproducible workflow replacing the R pipeline", fontsize=14, weight="bold")
    fig.tight_layout()
    fig.savefig(dst, dpi=300, bbox_inches="tight")
    plt.close(fig)
    manifest.append({"artifact": dst.name, "type": "figure", "status": "created", "source": "scripts/generate_legacy_outputs.py", "note": "Roadmap recreated in Python."})


def make_model_tables(manifest: list[dict[str, str]]) -> None:
    copy_table(config.REPORTS_DIR / "model_comparison.csv", "Table_1_model_comparison_python.csv", manifest)
    copy_table(config.REPORTS_DIR / "model_comparison_wilcoxon_holm.csv", "Table_S3_wilcoxon_holm_python.csv", manifest, required=False)
    copy_table(config.REPORTS_DIR / "imputation_comparison.csv", "Table_S_imputation_comparison_python.csv", manifest, required=False)
    copy_table(config.REPORTS_DIR / "missing_diagnosis.txt", "missing_diagnosis.txt", manifest, required=False)
    copy_table(config.REPORTS_DIR / "shap_summary.csv", "Table_SHAP_summary_python.csv", manifest, required=False)
    copy_table(config.REPORTS_DIR / "faf_results.csv", "Table_S1_FAF_calibration_python.csv", manifest, required=False)
    copy_table(config.REPORTS_DIR / "river_results.csv", "Table_2_river_results_python.csv", manifest, required=False)
    copy_table(config.REPORTS_DIR / "ph_sensitivity.csv", "Table_S_pH_sensitivity_python.csv", manifest, required=False)


def make_fallback_tables(manifest: list[dict[str, str]]) -> None:
    if not (config.REPORTS_DIR / "river_results.csv").exists() and (config.DATA_DIR / "river_data.csv").exists():
        df = pd.read_csv(config.DATA_DIR / "river_data.csv")
        save_table_from_df(df, "Table_2_river_inputs_reference.csv", manifest, "data/river_data.csv")
    if not (config.REPORTS_DIR / "faf_results.csv").exists() and (config.DATA_DIR / "field_data.csv").exists():
        df = pd.read_csv(config.DATA_DIR / "field_data.csv")
        save_table_from_df(df, "Table_S1_field_inputs_reference.csv", manifest, "data/field_data.csv")


def make_figures(manifest: list[dict[str, str]]) -> None:
    make_roadmap(manifest)
    composite_figure(
        [
            config.FIGURES_DIR / "model_mae_comparison.png",
            config.FIGURES_DIR / "qrf_predictions.png",
        ],
        "Fig_2_model_python.tiff",
        manifest,
        labels=["a", "b"],
        layout="horizontal",
        panel_gap=90,
        dpi=300,
        extra_dst=ROOT / "outputs" / "manuscript" / "All figures_for_word_main_and_SI" / "Fig_2_model_python.tiff",
    )
    composite_figure_mpl(
        [
            config.FIGURES_DIR / "shap_beeswarm.png",
            config.FIGURES_DIR / "shap_global.png",
        ],
        "Fig_3_feature_importance_python.tiff",
        manifest,
        labels=["a", "b"],
        dpi=300,
        label_fontsize=42,
        extra_dst=ROOT / "outputs" / "manuscript" / "All figures_for_word_main_and_SI" / "Fig_3_feature_importance_python.tiff",
    )
    image_to_tiff(config.FIGURES_DIR / "river_barplot.png", "Fig_4_river_global_python.tiff", manifest, required=False)
    composite_figure(
        [
            config.FIGURES_DIR / "missing_matrix.png",
            config.FIGURES_DIR / "missing_correlation.png",
        ],
        "Fig_S_missingness_python.tiff",
        manifest,
        labels=["a", "b"],
    )
    image_to_tiff(config.FIGURES_DIR / "imputation_comparison.png", "Fig_S_imputation_python.tiff", manifest, required=False)
    image_to_tiff(config.FIGURES_DIR / "faf_bootstrap.png", "Fig_S_FAF_bootstrap_python.tiff", manifest, required=False)
    image_to_tiff(config.FIGURES_DIR / "ph_sensitivity.png", "Fig_S_pH_sensitivity_python.tiff", manifest, required=False)
    image_to_tiff(config.FIGURES_DIR / "ablation_accuracy.png", "Fig_S_ablation_accuracy_python.tiff", manifest, required=False)
    image_to_tiff(config.FIGURES_DIR / "retrieval_precision.png", "Fig_S_retrieval_precision_python.tiff", manifest, required=False)


def compare_metric(
    legacy_path: Path,
    python_path: Path,
    metric_name: str,
    manifest: list[dict[str, str]],
) -> None:
    if not legacy_path.exists() or not python_path.exists():
        manifest.append(
            {
                "artifact": metric_name,
                "type": "validation",
                "status": "not_run",
                "source": f"legacy={legacy_path}; python={python_path}",
                "note": "One side missing.",
            }
        )
        return
    try:
        legacy = pd.read_csv(legacy_path)
        python = pd.read_csv(python_path)
        note = f"legacy_shape={legacy.shape}; python_shape={python.shape}"
        status = "available_for_review"
    except Exception as exc:
        note = str(exc)
        status = "read_failed"
    manifest.append({"artifact": metric_name, "type": "validation", "status": status, "source": f"legacy={legacy_path}; python={python_path}", "note": note})


def make_validation(manifest: list[dict[str, str]]) -> None:
    compare_metric(R_ROOT / "rf_test_metrics.csv", config.REPORTS_DIR / "model_comparison.csv", "RF_metrics_R_vs_Python", manifest)
    compare_metric(R_ROOT / "qrf_test_metrics.csv", config.REPORTS_DIR / "river_results.csv", "QRF_or_river_metrics_R_vs_Python", manifest)
    compare_metric(R_ROOT / "R_models_avg_MAE_by_fold.csv", config.REPORTS_DIR / "model_comparison.csv", "CV_MAE_R_vs_Python", manifest)

    legacy_figs = sorted(R_FIG_DONE.glob("*")) if R_FIG_DONE.exists() else []
    pd.DataFrame(
        [{"legacy_figure": path.name, "bytes": path.stat().st_size} for path in legacy_figs]
    ).to_csv(LEGACY_VALIDATION / "legacy_figure_inventory.csv", index=False, encoding="utf-8-sig")
    manifest.append({"artifact": "legacy_figure_inventory.csv", "type": "validation", "status": "created", "source": str(R_FIG_DONE), "note": f"{len(legacy_figs)} legacy figures inventoried."})


def write_manifest(manifest: list[dict[str, str]]) -> None:
    df = pd.DataFrame(manifest)
    df.to_csv(LEGACY_ROOT / "legacy_output_manifest.csv", index=False, encoding="utf-8-sig")
    summary = df.groupby(["type", "status"]).size().reset_index(name="n")
    summary.to_csv(LEGACY_ROOT / "legacy_output_summary.csv", index=False, encoding="utf-8-sig")
    lines = [
        "Legacy-style Python output bundle",
        "=" * 42,
        summary.to_string(index=False),
        "",
        "Notes:",
        "- Created files are generated from Python artifacts, not copied from R outputs.",
        "- Missing files indicate the corresponding upstream Python step has not been run yet.",
        "- Use this bundle for manuscript/SI replacement and R-vs-Python review.",
    ]
    (LEGACY_ROOT / "README.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create legacy-style output bundle from Python artifacts.")
    parser.add_argument("--run-pipeline", action="store_true", help="Run main.py --skip-llm --skip-ui first.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    ensure_dirs()
    if args.run_pipeline:
        import subprocess

        subprocess.run([sys.executable, str(ROOT / "main.py"), "--skip-llm", "--skip-ui"], check=False)

    manifest: list[dict[str, str]] = []
    make_model_tables(manifest)
    make_fallback_tables(manifest)
    make_figures(manifest)
    make_validation(manifest)
    write_manifest(manifest)
    print(f"Legacy-style outputs written to: {LEGACY_ROOT}")
    print(f"Manifest: {LEGACY_ROOT / 'legacy_output_manifest.csv'}")


if __name__ == "__main__":
    main()
