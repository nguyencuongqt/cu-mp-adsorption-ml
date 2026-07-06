"""Build manuscript Figure 5 from the LLM-as-Judge panels.

Figure 5 layout:
    - Figure 5a: llm_judge_bars.png
    - Figure 5b: llm_judge_radar.png

The script keeps a reproducible path from the source PNG files to the final
combined TIFF/PNG so the figure can be regenerated or adjusted later.

Usage:
    python scripts/make_figure5_llm_judge.py
"""

from __future__ import annotations

import shutil
import sys
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont, ImageOps

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import config  # noqa: E402


MANUSCRIPT_DIR = ROOT / "outputs" / "manuscript"
MAIN_FIGURES_DIR = MANUSCRIPT_DIR / "main_figures"
WORD_FIGURES_DIR = MANUSCRIPT_DIR / "figures_for_word"
TMP_DIR = ROOT / "tmp" / "docs"

BAR_FIG = config.FIGURES_DIR / "llm_judge_bars.png"
RADAR_FIG = config.FIGURES_DIR / "llm_judge_radar.png"

OUT_TIFF = MAIN_FIGURES_DIR / "Fig_5_llm_judge_python.tiff"
OUT_PNG = WORD_FIGURES_DIR / "Fig_5_llm_judge_python.png"
OUT_PDF = MAIN_FIGURES_DIR / "Fig_5_llm_judge_python.pdf"
OUT_MANIFEST = MANUSCRIPT_DIR / "figure5_llm_judge_sources.txt"


def _load_font(size: int) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    for name in ("arial.ttf", "calibri.ttf", "DejaVuSans-Bold.ttf"):
        try:
            return ImageFont.truetype(name, size)
        except OSError:
            continue
    return ImageFont.load_default()


def _resize_to_common_height(images: list[Image.Image], target_height: int) -> list[Image.Image]:
    resized: list[Image.Image] = []
    for img in images:
        scale = target_height / img.height
        resized.append(img.resize((int(img.width * scale), target_height)))
    return resized


def build_figure() -> Path:
    missing = [str(path) for path in (BAR_FIG, RADAR_FIG) if not path.exists()]
    if missing:
        raise FileNotFoundError(f"Missing source figure(s): {missing}")

    MAIN_FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    WORD_FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    TMP_DIR.mkdir(parents=True, exist_ok=True)

    with Image.open(BAR_FIG) as bar_img, Image.open(RADAR_FIG) as radar_img:
        panels = [
            ImageOps.exif_transpose(bar_img).convert("RGB"),
            ImageOps.exif_transpose(radar_img).convert("RGB"),
        ]

        target_height = 1650
        panels = _resize_to_common_height(panels, target_height)

        left_margin = 60
        right_margin = 40
        top_margin = 85
        bottom_margin = 30
        gap = 70

        total_width = left_margin + sum(img.width for img in panels) + gap + right_margin
        total_height = top_margin + target_height + bottom_margin
        canvas = Image.new("RGB", (total_width, total_height), "white")

        label_font = _load_font(92)
        x = left_margin
        for label, panel in zip(("a", "b"), panels):
            canvas.paste(panel, (x, top_margin))
            draw = ImageDraw.Draw(canvas)
            draw.text((x + 8, 12), label, fill="black", font=label_font)
            x += panel.width + gap

        canvas.save(OUT_TIFF, compression="tiff_lzw", dpi=(300, 300))
        canvas.save(OUT_PNG, dpi=(300, 300))
        canvas.save(OUT_PDF, "PDF", resolution=300.0)

    OUT_MANIFEST.write_text(
        "\n".join(
            [
                "Figure 5 source manifest",
                f"panel a: {BAR_FIG}",
                f"panel b: {RADAR_FIG}",
                f"combined tiff: {OUT_TIFF}",
                f"combined png: {OUT_PNG}",
                f"combined pdf: {OUT_PDF}",
            ]
        ),
        encoding="utf-8",
    )

    # Keep a copy in figures_for_word for quick manuscript insertion.
    shutil.copy2(OUT_TIFF, WORD_FIGURES_DIR / OUT_TIFF.name)
    return OUT_TIFF


if __name__ == "__main__":
    output = build_figure()
    print(f"Saved Figure 5 to: {output}")
