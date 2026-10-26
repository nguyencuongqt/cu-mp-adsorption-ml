"""Fig. S3: location of the rivers of the river-scale application and of the paired field studies (reviewer comment R2-4).

Base map: Natural Earth 1:110m land polygons (public domain; data/basemap_natural_earth/ne_110m_land.zip, downloaded from
naciscdn.org on 11 October 2026). The shapefile is read directly, so no mapping library is needed. River positions are the
approximate river mouths (decimal degrees) and field positions the approximate study areas; they place the symbols and are
not used in any calculation. River categories and values come from outputs/reports/river_application_SI.csv.

Output: Revision/02_Figures/Fig_S3_river_map.png and .tiff
"""

from __future__ import annotations

import struct
import sys
import zipfile
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.lines import Line2D
from matplotlib.patches import Polygon, Rectangle

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from config import REPORTS_DIR  # noqa: E402

BASEMAP = ROOT / "data" / "basemap_natural_earth" / "ne_110m_land.zip"
OUT = ROOT / "Manuscript" / "Submission_Env pollution" / "Revision" / "02_Figures"
NAVY, BLUE, RED, LAND, EDGE = "#1b4f72", "#5dade2", "#c0392b", "#ecebe6", "#b5b2a8"

# river (name as in river_application_SI.csv) -> short label, longitude, latitude of the mouth (approximate)
RIVERS = {
    "Yangtze": ("Yangtze", 121.8, 31.4), "Amazon": ("Amazon", -50.0, 0.0),
    "Pearl River system (Xi, Dong, Zhujiang)": ("Pearl", 113.6, 22.2), "Ganges": ("Ganges", 90.6, 22.2),
    "Magdalena": ("Magdalena", -74.85, 11.1), "Irrawaddy": ("Irrawaddy", 95.1, 15.8), "Brantas": ("Brantas", 112.8, -7.5),
    "Huangpu": ("Huangpu", 121.5, 31.2), "Solo": ("Solo", 112.55, -6.9), "Hanjiang": ("Hanjiang", 116.75, 23.35),
    "Serayu": ("Serayu", 109.1, -7.7), "Imo": ("Imo", 7.55, 4.6), "Cross": ("Cross", 8.3, 4.9),
    "Pasig": ("Pasig", 120.96, 14.6), "Kwa Ibo": ("Kwa Ibo", 7.98, 4.55), "Mekong": ("Mekong", 106.6, 9.8),
    "Progo": ("Progo", 110.2, -7.98), "Tamsui": ("Tamsui", 121.4, 25.17),
}
# paired field studies (Table S1): label, longitude, latitude of the study area (approximate), water Cu filtered?
FIELD = [("Ta and Babel\n(2020a, b; 2023a, b)\nChao Phraya", 100.55, 13.65, True), ("Zhang et al. (2025a)\nSuzhou rivers", 120.6, 31.3, True),
         ("Zhang et al. (2025b)\nZao River", 108.8, 34.3, True), ("Purwiyanto et al. (2020)\nMusi River", 104.8, -2.98, False),
         ("Ling et al.\n(2026) Huaibei", 116.75, 33.6, True)]
# label offsets in panel b (points) so that names do not overlap
OFFSETS_B = {"Yangtze": (0, 14), "Huangpu": (8, -8), "Tamsui": (7, -1), "Hanjiang": (6, -8), "Pearl": (-6, -11), "Pasig": (7, -1),
             "Mekong": (7, -4), "Irrawaddy": (-7, 2), "Ganges": (-8, 8), "Brantas": (7, -2), "Solo": (0, 8), "Serayu": (-7, -7),
             "Progo": (5, -10)}
FIELD_OFFSETS = [(-8, -5, "right"), (-9, -9, "right"), (-8, 2, "right"), (-8, 0, "right"), (-4, 14, "right")]   # dx, dy (points), alignment
ASIA = (82, 133, -13, 40)                                   # lon min, lon max, lat min, lat max of panel b


def land_polygons() -> list[list[tuple[float, float]]]:
    """Rings of the polygon shapefile (shape type 5), read without a GIS library."""
    with zipfile.ZipFile(BASEMAP) as z:
        data = z.read("ne_110m_land.shp")
    rings, pos = [], 100
    while pos < len(data):
        length = struct.unpack(">i", data[pos + 4:pos + 8])[0] * 2
        rec = data[pos + 8:pos + 8 + length]
        if struct.unpack("<i", rec[:4])[0] == 5:
            n_parts, n_points = struct.unpack("<2i", rec[36:44])
            parts = list(struct.unpack(f"<{n_parts}i", rec[44:44 + 4 * n_parts])) + [n_points]
            xy = struct.unpack(f"<{2 * n_points}d", rec[44 + 4 * n_parts:44 + 4 * n_parts + 16 * n_points])
            pts = list(zip(xy[0::2], xy[1::2]))
            rings += [pts[parts[i]:parts[i + 1]] for i in range(n_parts)]
        pos += 8 + length
    return rings


def draw_base(ax, rings, extent) -> None:
    for ring in rings:
        ax.add_patch(Polygon(ring, closed=True, facecolor=LAND, edgecolor=EDGE, linewidth=0.3, zorder=1))
    ax.set_xlim(extent[0], extent[1]); ax.set_ylim(extent[2], extent[3])
    ax.set_aspect("equal"); ax.set_facecolor("white")
    ax.tick_params(labelsize=6.5, length=2)
    ax.xaxis.set_major_formatter(lambda v, _: f"{abs(v):.0f}°{'E' if v > 0 else 'W' if v < 0 else ''}")
    ax.yaxis.set_major_formatter(lambda v, _: f"{abs(v):.0f}°{'N' if v > 0 else 'S' if v < 0 else ''}")
    for side in ax.spines.values():
        side.set_linewidth(0.6)


def draw_points(ax, rivers: pd.DataFrame, labels: bool, values: bool) -> None:
    for _, r in rivers.iterrows():
        measured = bool(r["in_main_table"])
        ax.plot(r["lon"], r["lat"], "o", ms=5.2 if measured else 4.2, mfc=NAVY if measured else "white", mec=NAVY, mew=0.9, zorder=4)
        if labels:
            dx, dy = OFFSETS_B.get(r["label"], (7, -3))
            text = r["label"] + (f"\n{r['Cu_kg_yr_12.5ug']:.2g} kg yr$^{{-1}}$" if measured and values else "")
            ax.annotate(text, (r["lon"], r["lat"]), xytext=(dx, dy), textcoords="offset points", fontsize=6.3,
                        ha="left" if dx > 0 else "right" if dx < 0 else "center", va="center", color=NAVY,
                        fontweight="bold" if measured else "normal", zorder=5)
    for (label, lon, lat, filtered), (dx, dy, ha) in zip(FIELD, FIELD_OFFSETS):
        ax.plot(lon, lat, "^", ms=5.6, mfc=RED if filtered else "white", mec=RED, mew=0.9, zorder=6)
        if labels:
            ax.annotate(label, (lon, lat), xytext=(dx, dy), textcoords="offset points", fontsize=6.0, ha=ha, va="center",
                        color=RED, zorder=6, linespacing=0.95)


def main() -> None:
    plt.rcParams.update({"font.family": "Times New Roman", "mathtext.fontset": "custom", "mathtext.rm": "Times New Roman", "mathtext.it": "Times New Roman:italic", "mathtext.bf": "Times New Roman:bold", "mathtext.default": "regular"})
    table = pd.read_csv(REPORTS_DIR / "river_application_SI.csv")
    missing = set(table["river"]) ^ set(RIVERS)
    if missing:
        raise SystemExit(f"rivers without coordinates or not in the table: {missing}")
    table["label"] = table["river"].map(lambda k: RIVERS[k][0])
    table["lon"], table["lat"] = table["river"].map(lambda k: RIVERS[k][1]), table["river"].map(lambda k: RIVERS[k][2])
    rings = land_polygons()

    world = (-125, 150, -40, 62)
    ratios = [(e[1] - e[0]) / (e[3] - e[2]) for e in (world, ASIA)]            # equal panel heights
    fig, (ax_a, ax_b) = plt.subplots(1, 2, figsize=(11.0, 3.9), gridspec_kw={"width_ratios": ratios, "wspace": 0.10})
    draw_base(ax_a, rings, world)
    draw_points(ax_a, table, labels=False, values=False)
    ax_a.add_patch(Rectangle((ASIA[0], ASIA[2]), ASIA[1] - ASIA[0], ASIA[3] - ASIA[2], fill=False, edgecolor="0.35", linewidth=0.7,
                             linestyle="--", zorder=7))
    ax_a.text(ASIA[0] + 1.5, ASIA[3] + 1.8, "panel b", fontsize=6.5, color="0.3")
    for name, dx, dy, ha in [("Amazon", 7, 4, "left"), ("Magdalena", 7, 4, "left")]:
        r = table[table["label"] == name].iloc[0]
        text = name + (f"\n{r['Cu_kg_yr_12.5ug']:.2g} kg yr$^{{-1}}$" if r["in_main_table"] else "")
        ax_a.annotate(text, (r["lon"], r["lat"]), xytext=(dx, dy), textcoords="offset points", fontsize=6.3, ha=ha, va="bottom",
                      color=NAVY, fontweight="bold" if r["in_main_table"] else "normal")
    ax_a.annotate("Imo, Kwa Ibo, Cross", (8.0, 4.7), xytext=(0, -11), textcoords="offset points", fontsize=6.3, ha="center", color=NAVY)
    draw_base(ax_b, rings, ASIA)
    draw_points(ax_b, table, labels=True, values=True)
    for ax, letter in [(ax_a, "a"), (ax_b, "b")]:
        ax.text(0.0, 1.03, letter, transform=ax.transAxes, fontsize=11, fontweight="bold", va="bottom")
    handles = [Line2D([], [], marker="o", ls="", ms=5.2, mfc=NAVY, mec=NAVY, label="River with measured MP abundance and dissolved Cu (Table 3);\n"
                      "value: MP-bound Cu for 12.5 µg per particle"),
               Line2D([], [], marker="o", ls="", ms=4.2, mfc="white", mec=NAVY, label="River with at least one default input (Table S12)"),
               Line2D([], [], marker="^", ls="", ms=5.6, mfc=RED, mec=RED, label="Field study with Cu on MPs and dissolved Cu in water (Table S1)"),
               Line2D([], [], marker="^", ls="", ms=5.6, mfc="white", mec=RED, label="Field study with Cu in unfiltered water (sensitivity analysis only)")]
    fig.legend(handles=handles, loc="lower center", bbox_to_anchor=(0.5, 0.0), ncol=2, fontsize=6.6, frameon=False, columnspacing=2.5,
               labelspacing=0.8)
    for ext, kw in [("png", {}), ("tiff", {"pil_kwargs": {"compression": "tiff_lzw"}})]:
        fig.savefig(OUT / f"Fig_S3_river_map.{ext}", dpi=600, bbox_inches="tight", facecolor="white", **kw)
    print(f"saved Fig_S3_river_map ({len(rings)} land polygons, {len(table)} rivers, {len(FIELD)} field study areas)")


if __name__ == "__main__":
    main()
