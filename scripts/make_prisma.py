"""PRISMA 2020 flow diagram for the laboratory dataset, with a record-level ledger.

Counts are derived from the Scopus exports (469 records), the PubMed records (136), the Web of Science searches of
14 October 2026 (main string 409; supplementary S2a 563 and S2b 127) and the literature index (full-text decisions),
so that every box of the diagram can be traced to a list of records.

The Scopus/PubMed part of the ledger is rebuilt by build_ledger(). The Web of Science records were deduplicated
against it by Wos/wos_dedup.py; the merged, screened ledger is kept in WOS_LEDGER and is the record of all decisions
(build it once with --finalize-wos from the draft ledger of 14 October 2026). The ledger and counts before the
Web of Science searches are archived in PRISMA/_before_wos_2026-10-07/.

Outputs (05_Literature_search/PRISMA/): prisma_ledger.csv, prisma_counts.csv, Fig_S1_PRISMA.png / .tiff
"""

from __future__ import annotations

import glob
import re
import shutil
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.patches import FancyBboxPatch

ROOT = Path(__file__).resolve().parents[1]
REV = ROOT / "Manuscript" / "Submission_Env pollution" / "Revision"
OUT = REV / "05_Literature_search" / "PRISMA"
WOS_DRAFT = OUT / "prisma_ledger_with_wos_DRAFT.csv"
WOS_LEDGER = OUT / "prisma_ledger.csv"
# Reports from the Web of Science searches included after digitization and QC (14 October 2026)
WOS_INCLUDED = {"10.1007/s11356-023-27906-1": "Yan_2023", "10.1002/ldr.5392": "LiuZhou_2025",
                "10.1016/j.jece.2025.120355": "LiuZhou_2025"}
# Records of the earlier title-only search (2025) that the supplementary Web of Science searches retrieved again
EARLIER_IN_WOS = {"10.1016/j.chemosphere.2020.126064": "S2a", "10.1016/j.jes.2019.07.006": "S2a", "10.1016/j.jhazmat.2020.122515": "S2b"}

# Reason for exclusion at full text, by DOI (from the full-text notes in literature_index.csv)
REASONS = {
    "seawater or marine medium": ["10.1021/acs.est.1c04278", "10.1016/j.marpolbul.2021.112480", "10.1016/j.marpolbul.2026.120276",
                                  "10.3389/fmars.2021.671802", "10.3389/fmars.2021.770606", "10.3389/fmars.2021.785146"],
    "no equilibrium isotherm (single loading or kinetics only)": [
        "10.1016/j.envpol.2026.128518", "10.1016/j.scitotenv.2022.153265", "10.1016/j.scitotenv.2023.164807",
        "10.1016/j.scitotenv.2024.175537", "10.1186/s12302-025-01061-5", "10.3390/microplastics5020115",
        "10.1007/s11356-025-35923-5", "10.1016/j.cej.2025.166576"],
    "figures or units not reconcilable with the Methods": ["10.1021/acsomega.5c05770", "10.1016/j.ecoenv.2022.113292",
                                                           "10.1016/j.jhazmat.2020.123548", "10.1021/acs.estlett.5c01060",
                                                           "10.1155/2021/4938749"],
    "nano-sized particles": ["10.1016/j.cej.2025.171838", "10.1016/j.jhazmat.2020.122515"],
    "uptake not attributable to sorption on the particles": ["10.1016/j.watres.2022.118354", "10.1039/d3em00354j"],
    "particle dose not determinable": ["10.1016/j.aquatox.2024.106938"],
    "functionalised polymers designed as sorbents": ["10.1021/acsomega.5c11348"],
}
NOT_ENGLISH = ["10.13227/j.hjkx.202002196", "10.13227/j.hjkx.202307137"]
FIRST_VERSION_REMOVED = ["10.1021/acs.est.1c04278", "10.1016/j.marpolbul.2021.112480", "10.1016/j.jhazmat.2020.122515"]


def norm_title(t: str) -> str:
    return re.sub(r"[^a-z0-9]", "", str(t).lower())


def build_ledger() -> pd.DataFrame:
    scopus = pd.concat([pd.read_csv(f) for f in sorted(glob.glob(str(REV / "05_Literature_search" / "export_*.csv")))], ignore_index=True)
    scopus = pd.DataFrame({"doi": scopus["DOI"].str.lower().str.strip(), "title": scopus["Title"], "year": scopus["Year"], "scopus": True})
    pubmed = pd.read_csv(REV / "05_Literature_search" / "PubMed_search" / "pubmed_records_2026-10-07.csv")
    pubmed = pd.DataFrame({"doi": pubmed["doi"].str.lower().str.strip(), "title": pubmed["title"], "year": pubmed["year"], "pubmed": True})
    index = pd.read_csv(REV / "05_Literature_search" / "literature_index.csv")
    index["doi"] = index["doi"].str.lower().str.strip()
    lab = index[index["role"].isin(["laboratory dataset", "laboratory candidate"])]

    scopus["tkey"], pubmed["tkey"] = scopus["title"].map(norm_title), pubmed["title"].map(norm_title)
    scopus["id"] = scopus["doi"].fillna("title:" + scopus["tkey"])
    doi_by_title = scopus.dropna(subset=["doi"]).set_index("tkey")["doi"].to_dict()
    pubmed["id"] = pubmed["doi"].fillna(pubmed["tkey"].map(doi_by_title)).fillna("title:" + pubmed["tkey"])
    ledger = pd.merge(scopus[["id", "title", "year", "scopus"]], pubmed[["id", "title", "year", "pubmed"]], on="id", how="outer",
                      suffixes=("", "_pm"))
    ledger["title"] = ledger["title"].fillna(ledger["title_pm"])
    ledger["year"] = ledger["year"].fillna(ledger["year_pm"])
    ledger = ledger.drop(columns=["title_pm", "year_pm"]).fillna({"scopus": False, "pubmed": False})
    other = lab[~lab["doi"].isin(ledger["id"])]
    ledger = pd.concat([ledger, pd.DataFrame({"id": other["doi"], "title": other["pdf_title"], "scopus": False, "pubmed": False})],
                       ignore_index=True)
    ledger["source"] = ledger.apply(lambda r: "Scopus and PubMed" if r["scopus"] and r["pubmed"] else "Scopus" if r["scopus"]
                                    else "PubMed" if r["pubmed"] else "earlier title-only search (2025)", axis=1)

    status = lab.set_index("doi")["status"].to_dict()
    key = lab.set_index("doi")["key"].to_dict()
    reason = {d: r for r, dois in REASONS.items() for d in dois}

    def decide(doi: str) -> tuple[str, str]:
        if doi in NOT_ENGLISH:
            return "excluded at title and abstract", "not in English"
        if status.get(doi) == "included":
            return "included", ""
        if doi in status:
            return "excluded at full text", reason[doi]
        return "excluded at title and abstract", ""

    ledger[["decision", "reason"]] = [decide(d) for d in ledger["id"]]
    ledger["study_key"] = ledger["id"].map(key)
    ledger["in_first_version_dataset"] = ledger["id"].isin(FIRST_VERSION_REMOVED) | ledger["study_key"].isin(first_version_keys())
    return ledger.sort_values(["decision", "reason", "id"]).reset_index(drop=True)


def first_version_keys() -> set[str]:
    log = pd.read_csv(REV / "06_Data_audit" / "Change_log_vs_submitted_dataset" / "change_log_rows.csv") \
        if (REV / "06_Data_audit" / "Change_log_vs_submitted_dataset" / "change_log_rows.csv").exists() else None
    if log is not None and {"source_id", "status"}.issubset(log.columns):
        return set(log.loc[log["status"] != "new", "source_id"])
    return set()


def box(ax, x, y, w, h, text, face="#f4f6f8", bold_first=False, fs=7.6):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.006,rounding_size=0.012", fc=face, ec="#34495e", lw=0.9))
    ax.text(x + 0.012, y + h - 0.014, text, ha="left", va="top", fontsize=fs, linespacing=1.35)


def arrow(ax, x0, y0, x1, y1):
    ax.annotate("", xy=(x1, y1), xytext=(x0, y0), arrowprops={"arrowstyle": "-|>", "color": "#34495e", "lw": 0.9})


def finalize_wos() -> pd.DataFrame:
    """Turn the draft ledger of 14 October 2026 into the final ledger (eligible reports -> included)."""
    led = pd.read_csv(WOS_DRAFT)
    for doi, key in WOS_INCLUDED.items():
        m = led["id"] == doi
        assert m.sum() == 1 and led.loc[m, "decision"].iloc[0].startswith("eligible"), doi
        led.loc[m, ["decision", "reason", "study_key"]] = ["included", None, key]
    assert not led["decision"].str.startswith("eligible").any()
    for doi, search in EARLIER_IN_WOS.items():
        m = led["id"] == doi
        assert m.sum() == 1 and led.loc[m, "source"].iloc[0] == "earlier title-only search (2025)", doi
        led.loc[m, ["wos", "source"]] = [True, f"Web of Science supplementary ({search}); earlier title-only search (2025)"]
    return led.sort_values(["decision", "reason", "id"]).reset_index(drop=True)


def main() -> None:
    import sys
    OUT.mkdir(parents=True, exist_ok=True)
    if "--finalize-wos" in sys.argv:
        finalize_wos().to_csv(WOS_LEDGER, index=False)
    led = pd.read_csv(WOS_LEDGER)
    assert "wos" in led.columns, "run once with --finalize-wos"
    for c in ("scopus", "pubmed", "wos"):
        led[c] = led[c].astype(bool)

    n_scopus, n_pubmed = int(led["scopus"].sum()), int(led["pubmed"].sum())
    src = led["source"]
    n_other = int((src == "earlier title-only search (2025)").sum())
    # Web of Science records retrieved by each search (from the search logs; before deduplication)
    n_wos_main, n_s2a, n_s2b = 409, 563, 127
    n_identified = n_scopus + n_pubmed + n_wos_main + n_s2a + n_s2b + n_other
    n_screened = len(led)
    n_dup = n_identified - n_screened
    ta = led[led["decision"] == "excluded at title and abstract"]
    sought = led[led["decision"] != "excluded at title and abstract"]
    not_retrieved = led[led["decision"] == "report not retrieved"]
    full = sought[sought["decision"] != "report not retrieved"]
    excl = led[led["decision"] == "excluded at full text"]
    incl = led[led["decision"] == "included"]
    reasons = excl["reason"].value_counts()
    dataset = pd.read_csv(ROOT / "data" / "dataset_grouped.csv")
    n_studies = incl["study_key"].nunique()
    n_ta_ne = int((ta["reason"] == "not in English").sum())
    counts = {"records_scopus": n_scopus, "records_pubmed": n_pubmed, "records_wos_main": n_wos_main, "records_wos_S2a": n_s2a,
              "records_wos_S2b": n_s2b, "records_earlier_search_not_retrieved_again": n_other, "records_identified_total": n_identified,
              "duplicates_removed": n_dup, "records_screened": n_screened, "excluded_title_abstract": len(ta),
              "excluded_title_abstract_not_english": n_ta_ne, "reports_sought": len(sought), "reports_not_retrieved": len(not_retrieved),
              "full_text_assessed": len(full), "excluded_full_text": len(excl),
              **{f"excluded_full_text: {k}": int(v) for k, v in reasons.items()},
              "reports_included": len(incl), "studies_included": n_studies,
              "reports_included_from_wos_searches": int(incl["id"].isin(WOS_INCLUDED).sum()),
              "observations": len(dataset), "isotherms": int(dataset["curve_id"].nunique())}
    pd.Series(counts, name="n").rename_axis("item").to_csv(OUT / "prisma_counts.csv")
    # Identification via other methods: field-study search of 16 Oct 2026 (Scopus + WoS, string v3), screened for both the
    # laboratory dataset and the field comparison (Snowball_field_2026-10/Field_search_*.md, screening_sheet_proposed.xlsx)
    oth = {"records_scopus": 755, "records_wos": 690, "duplicates": 456, "already_screened": 115, "screened": 874,
           "excluded_ta": 839, "sought": 35, "not_retrieved": 3, "assessed": 32, "lab_included": 1, "field_tier1": 3,
           "field_tier2": 8, "excluded_ft": 20}
    oth_reasons = [("Metals only qualitative (EDS, XRF, LIBS)", 5), ("Plastics deployed in situ, or marine", 6),
                   ("Laboratory study without Cu isotherm", 3), ("Cu not on plastics or below detection", 3),
                   ("Other (wastewater ponds, units, additive Cu)", 3)]
    assert sum(n for _, n in oth_reasons) == oth["excluded_ft"]
    assert oth["assessed"] == oth["lab_included"] + oth["field_tier1"] + oth["field_tier2"] + oth["excluded_ft"]
    counts.update({f"other_methods_{k}": v for k, v in oth.items()})
    pd.Series(counts, name="n").rename_axis("item").to_csv(OUT / "prisma_counts.csv")
    other_lab = {"Turner_2015"}
    assert set(incl["study_key"]) | other_lab == set(dataset["source_id"]), set(incl["study_key"]) ^ set(dataset["source_id"])
    assert len(full) == len(excl) + len(incl)

    plt.rcParams.update({"font.family": "Times New Roman", "mathtext.fontset": "custom", "mathtext.rm": "Times New Roman", "mathtext.it": "Times New Roman:italic", "mathtext.bf": "Times New Roman:bold"})
    fig, ax = plt.subplots(figsize=(10.6, 9.4))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    for y, h, label in [(0.765, 0.225, "Identification"), (0.30, 0.455, "Screening"), (0.02, 0.27, "Included")]:
        ax.add_patch(FancyBboxPatch((0.005, y), 0.045, h, boxstyle="round,pad=0.004,rounding_size=0.01", fc="#d6e4f0", ec="none"))
        ax.text(0.0275, y + h / 2, label, rotation=90, ha="center", va="center", fontsize=8.5, fontweight="bold", color="#1b4f72")

    lx, lw, rx, rw = 0.055, 0.30, 0.375, 0.29
    box(ax, lx, 0.82 if not n_other else 0.80, lw, 0.16 if not n_other else 0.18, f"Records identified (n = {n_identified})\n"
                                   f"   Scopus, 6 October 2026 (n = {n_scopus})\n   PubMed, 7 October 2026 (n = {n_pubmed})\n"
                                   f"   Web of Science, 14 October 2026\n      main search (n = {n_wos_main})\n"
                                   f"      supplementary searches (n = {n_s2a} + {n_s2b})"
                                   + (f"\n   Studies from an earlier title-only search\n   (2025) not retrieved again (n = {n_other})" if n_other else ""))
    box(ax, rx, 0.8525, rw, 0.075, f"Records removed before screening\n   Duplicates across databases and\n   searches (n = {n_dup})")
    box(ax, lx, 0.655, lw, 0.07, f"Records screened by title and abstract\n(n = {n_screened})")
    box(ax, rx, 0.635, rw, 0.115, f"Records excluded (n = {len(ta)})\n   Not Cu sorption on plastic particles in water:\n   toxicity, engineered sorbents, soil, sediment,\n"
                                  f"   other metals, field surveys, reviews (n = {len(ta) - n_ta_ne})\n"
                                  f"   Not in English (n = {n_ta_ne})")
    box(ax, lx, 0.525, lw, 0.06, f"Reports sought for retrieval (n = {len(sought)})")
    box(ax, rx, 0.525, rw, 0.06, f"Reports not retrieved (n = {len(not_retrieved)})")
    box(ax, lx, 0.415, lw, 0.06, f"Reports assessed for eligibility (n = {len(full)})")
    short = {"no equilibrium isotherm (single loading or kinetics only)": "No equilibrium isotherm",
             "seawater or marine medium": "Seawater or marine medium",
             "figures or units not reconcilable with the Methods": "Figures not reconcilable with Methods",
             "Cu not studied": "Cu not studied",
             "not in English": "Not in English",
             "nano-sized particles": "Nano-sized particles",
             "uptake not attributable to sorption on the particles": "Uptake not due to sorption on particles",
             "functionalised polymers designed as sorbents": "Functionalized sorbent polymers",
             "particle dose not determinable": "Particle dose not determinable",
             "Ce and qe not reported and not calculable": "Ce and qe not reported or calculable",
             "data not extractable (overlapping markers in the isotherm figure)": "Data not extractable from figures"}
    lines = "\n".join(f"   {short[k]} (n = {v})" for k, v in reasons.items())
    box(ax, rx, 0.245, rw, 0.26, f"Reports excluded (n = {len(excl)})\n{lines}")
    box(ax, lx, 0.205, lw, 0.075, f"Studies included in the dataset\n(n = {n_studies + oth['lab_included']}; {len(incl) + oth['lab_included']} reports;\n"
                                   f"{oth['lab_included']} from other methods)", face="#e8f1e4")
    ox, ow = 0.69, 0.305
    ax.text(ox + ow / 2, 0.995, "Identification of studies via other methods", ha="center", va="top", fontsize=8, fontweight="bold", color="#1b4f72")
    box(ax, ox, 0.80, ow, 0.18, f"Field-study search, 16 October 2026\n   Scopus (n = {oth['records_scopus']})\n   Web of Science (n = {oth['records_wos']})\n"
                                 f"   Duplicates (n = {oth['duplicates']})\n   Already screened in the database\n   searches (n = {oth['already_screened']})")
    box(ax, ox, 0.655, ow, 0.07, f"New records screened (n = {oth['screened']})\n   Excluded (n = {oth['excluded_ta']})")
    box(ax, ox, 0.525, ow, 0.06, f"Reports sought (n = {oth['sought']})\n   Not retrieved (n = {oth['not_retrieved']})")
    olines = "\n".join(f"   {k} (n = {v})" for k, v in oth_reasons)
    box(ax, ox, 0.245, ow, 0.26, f"Reports assessed (n = {oth['assessed']})\n"
                                 f"Field comparison (Tables S1, S13)\n   Tier 1, paired Cu in water (n = {oth['field_tier1']})\n"
                                 f"   Tier 2, Cu on MPs only (n = {oth['field_tier2']})\n"
                                 f"Laboratory dataset (n = {oth['lab_included']})\nExcluded (n = {oth['excluded_ft']})\n{olines}", fs=6.9)
    for y0, y1 in [(0.80, 0.725), (0.655, 0.585), (0.525, 0.505)]:
        arrow(ax, ox + ow / 2, y0, ox + ow / 2, y1)
    ax.plot([ox + ow / 2, ox + ow / 2, lx + lw + 0.03], [0.245, 0.225, 0.225], color="#34495e", lw=0.9)
    arrow(ax, lx + lw + 0.03, 0.225, lx + lw, 0.225)
    box(ax, lx, 0.055, lw, 0.08, f"Dataset after quality control\n   {len(dataset)} equilibrium observations\n   {dataset['curve_id'].nunique()} isotherms", face="#e8f1e4")

    cx = lx + lw / 2
    for y0, y1 in [(0.82 if not n_other else 0.80, 0.725), (0.655, 0.585), (0.525, 0.475), (0.415, 0.28), (0.205, 0.135)]:
        arrow(ax, cx, y0, cx, y1)
    arrow(ax, lx + lw, 0.89, rx, 0.89)
    arrow(ax, lx + lw, 0.69, rx, 0.69)
    arrow(ax, lx + lw, 0.555, rx, 0.555)
    arrow(ax, lx + lw, 0.445, rx, 0.445)
    fig.savefig(OUT / "Fig_S1_PRISMA.png", dpi=600, bbox_inches="tight")
    fig.savefig(OUT / "Fig_S1_PRISMA.tiff", dpi=600, bbox_inches="tight", pil_kwargs={"compression": "tiff_lzw"})
    for ext in ("png", "tiff"):                                  # copy next to the other figures of the submission
        (REV / "02_Figures").mkdir(exist_ok=True)
        shutil.copy2(OUT / f"Fig_S1_PRISMA.{ext}", REV / "02_Figures" / f"Fig_S1_PRISMA.{ext}")
    print(pd.Series(counts).to_string())
    print(led.groupby(["source", "decision"]).size().to_string())


if __name__ == "__main__":
    main()
