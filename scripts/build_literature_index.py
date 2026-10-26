"""Build one index of all literature files and their role in the revision.

No file is moved. The index lists every paper that has a role in the manuscript (laboratory dataset,
full-text exclusions, field evidence, inputs of the scale-up, methods references) and every PDF or
supplementary file found in the literature folders, with its current path.

Output: Manuscript/Submission_Env pollution/Revision/literature_index.csv and .xlsx
"""

from __future__ import annotations

import re
import zipfile
from pathlib import Path

import pandas as pd
import pymupdf

ROOT = Path(__file__).resolve().parents[1]
REV = ROOT / "Manuscript" / "Submission_Env pollution" / "Revision"
FOLDERS = [ROOT / "data" / "papers", REV / "07_Papers" / "additional paper", REV / "07_Papers" / "new papers", REV / "06_Data_audit" / "Lab_candidates_waiting_for_SI"]
DOI = re.compile(r"10\.\d{4,9}/[^\s\"<>|,;\]]+")
# Files whose paper cannot be recognised automatically (no DOI printed, or publisher id absent from the article file name)
MANUAL_PDF = {"science.abe0290.pdf": "Weiss_2021", "Urban Canal.pdf": "Eamrat_2023"}
MANUAL_SI = {"es1c04278_si": "Chen_2021", "es0c02273_si": "Mai_2020", "abe0290_weiss_sm": "Weiss_2021", "Heavymetal_DATA.rar": "Jin_2025",
             "ScienceDirect_files_07Oct2026": "Tan_2023", "Leistenschneider": "Leistenschneider_2026", "Purwiyanto_2020_station": "Purwiyanto_2020"}
REVIEW_FILE = REV / "05_Literature_search" / "Additional_papers_PDF_eligibility_review.md"
SI_HINT = re.compile(r"mmc\d|_si_|[-_ ]SI[-_. ]|supp|Heavymetal|ScienceDirect_files", re.I)


def norm(doi: str) -> str:
    return str(doi).strip().lower().rstrip(".)")


def pdf_info(path: Path) -> tuple[str, str]:
    """DOI printed on the first pages and a short title."""
    try:
        doc = pymupdf.open(path)
        text = " ".join(doc[i].get_text() for i in range(min(2, len(doc))))
        found = [norm(d) for d in DOI.findall(text)]
        title = (doc.metadata or {}).get("title") or ""
        if len(title) < 15:
            title = re.sub(r"\s+", " ", doc[0].get_text())[:140]
        return (found[0] if found else ""), title.strip()
    except Exception as exc:  # unreadable file
        return "", f"unreadable: {exc}"


def file_key(name: str) -> str:
    """Publisher identifier shared by an article and its supplementary files."""
    m = re.search(r"S\d{15}[\dX]", name)                       # Elsevier PII
    if m:
        return m.group(0)
    m = re.search(r"\b([a-z]{2}\d[a-z]\d{5})", name.lower())    # ACS manuscript id, e.g. es1c04278
    return m.group(1) if m else ""


def roles() -> pd.DataFrame:
    rows = []
    lab = pd.read_csv(ROOT / "data" / "dataset_grouped.csv")
    for (sid, doi), g in lab.groupby(["source_id", "doi"]):
        rows.append({"key": sid, "doi": norm(doi), "role": "laboratory dataset", "status": "included",
                     "note": f"{len(g)} points, {g['curve_id'].nunique()} isotherms"})
    excluded = {
        "10.1021/acs.est.1c04278": ("Chen_2021", "seawater; in the submitted dataset, removed"),
        "10.1016/j.marpolbul.2021.112480": ("Gao_2021", "seawater; in the submitted dataset, removed"),
        "10.1016/j.ecoenv.2022.113292": ("Chen_2022", "concentrations in figures inconsistent with Methods"),
        "10.1016/j.watres.2022.118354": ("Zhou_2022_WaterRes", "qe of 183-1378 mg/g at 10 mg/L dose; not surface sorption"),
        "10.1021/acs.estlett.5c01060": ("Ye_2025", "concentrations in figures inconsistent with Methods; solution volume not given"),
        "10.1021/acsomega.5c05770": ("ACSOmega_2025", "concentrations in figures inconsistent with Methods"),
        "10.1016/j.aquatox.2024.106938": ("Aquatox_2024", "particle dose cannot be determined"),
        "10.1007/s11356-025-35923-5": ("ESPR_2025", "uptake against time only; equilibrium not reached at 48 h"),
        "10.1016/j.cej.2025.166576": ("Pan_2025", "not an equilibrium Ce-qe isotherm"),
        "10.1016/j.jhazmat.2020.123548": ("Tang_2021", "concentrations in figures inconsistent with Methods"),
        "10.1016/j.jhazmat.2020.122515": ("Mao_2020", "particles of 0.1 um (nanoparticles); in the submitted dataset, removed"),
        "10.1016/j.cej.2025.171838": ("CEJ_2025_171838", "particles of 500 nm (nanoparticles)"),
        "10.13227/j.hjkx.202002196": ("HJKX_2020", "not in English"),
        "10.13227/j.hjkx.202307137": ("HJKX_2024", "not in English"),
    }
    for doi, (key, why) in excluded.items():
        rows.append({"key": key, "doi": doi, "role": "laboratory candidate", "status": "excluded at full text", "note": why})
    field = pd.read_csv(ROOT / "data" / "field_evidence_tiers.csv", dtype=str)
    for doi, g in field.groupby("doi"):
        tiers = sorted(set(g["tier"]))
        status = "excluded" if tiers == ["excluded"] else "tier " + " and ".join(t for t in tiers if t != "excluded")
        rows.append({"key": re.sub(r"_(dry|wet|Bangkok|SamutPrakan|AngThong)$", "", g["study_id"].iloc[0]), "doi": norm(doi),
                     "role": "field evidence", "status": status, "note": "; ".join(g["site"].astype(str).unique())[:200]})
    for key, doi, note in [
        ("Jin_2025", "10.1038/s43247-025-02508-6", "dissolved Cu in rivers (compilation); data: figshare 10.6084/m9.figshare.29338148.v1"),
        ("Tan_2023", "10.1016/j.scitotenv.2023.165979", "microplastic abundance in rivers (compilation)"),
        ("Su_2024", "10.1016/j.jenvman.2024.121389", "dissolved Cu of the lower Ganges; cites world-average river water"),
        ("Weiss_2021", "10.1126/science.abe0290", "riverine microplastic mass flux, 6.1 kt/yr; particle mass 230 ug"),
        ("Mai_2020", "10.1021/acs.est.0c02273", "riverine plastic outflow (all sizes); particle mass 170 ug"),
        ("Strokal_2023", "10.1038/s41467-023-40501-9", "riverine microplastic (53 kt/yr) and total plastic (500 kt/yr) export"),
        ("Lebreton_2017", "10.1038/ncomms15611", "list of rivers and discharge"),
        ("Gaillardet_2014", "", "world-average dissolved Cu in river water, 1.48 ug/L (Treatise on Geochemistry)"),
        ("Collins_2024", "", "global river discharge, 37,411 km3/yr (already in the submitted reference list)"),
    ]:
        rows.append({"key": key, "doi": doi, "role": "scale-up input", "status": "cited", "note": note})
    for key, note in [("Roberts_2017", "cross-validation for structured data (Ecography)"),
                      ("Varma_Simon_2006", "nested cross-validation (BMC Bioinformatics)"),
                      ("Vovk_2005", "conformal prediction (Algorithmic Learning in a Random World)"),
                      ("Lei_2018", "distribution-free predictive inference (J. Am. Stat. Assoc.)")]:
        rows.append({"key": key, "doi": "", "role": "methods reference", "status": "cited; bibliographic details to verify", "note": note})
    known = {r["doi"] for r in rows}
    heading = ""
    for line in REVIEW_FILE.read_text(encoding="utf8").splitlines():   # papers read in full during the revision
        if line.startswith("#"):
            heading = line.strip("# ").strip()
        for d in DOI.findall(line):
            d = norm(d)
            if d not in known and heading.startswith("Quyết định"):
                known.add(d)
                cells = [c.strip() for c in line.split("|") if c.strip()]
                rows.append({"key": "", "doi": d, "role": "laboratory candidate", "status": "read in full; not in the dataset",
                             "note": (heading.split("—")[-1].strip() + ": " + " / ".join(cells[1:]))[:400]})
    return pd.DataFrame(rows)


def reference_list_text() -> str:
    out = []
    for name in ["Manuscript_revised_figure_quality.docx", "Supplementary_Information.docx"]:
        xml = zipfile.ZipFile(REV.parent / name).read("word/document.xml").decode("utf8")
        paras = ["".join(re.findall(r"<w:t[^>]*>([^<]*)</w:t>", p)) for p in re.findall(r"<w:p[ >].*?</w:p>", xml, flags=re.S)]
        start = max((i for i, t in enumerate(paras) if re.fullmatch(r"\s*References?\s*", t)), default=0)
        out.extend(paras[start:])
    return "\n".join(out)


def main() -> None:
    files = [f for folder in FOLDERS for f in folder.rglob("*") if f.is_file() and "_extracted" not in str(f)
             and "Jin_2025_Heavymetal_DATA" not in str(f)]
    articles, supplements = [], []
    for f in files:
        rel = str(f.relative_to(ROOT)).replace("\\", "/")
        if f.suffix.lower() == ".pdf" and not SI_HINT.search(f.name):
            doi, title = pdf_info(f)
            articles.append({"doi": doi, "pdf_path": rel, "pdf_title": title, "file_key": file_key(f.name)})
        else:
            supplements.append({"si_path": rel, "file_key": file_key(f.name), "name": f.name})
    art, sup = pd.DataFrame(articles), pd.DataFrame(supplements)
    role_keys = roles().set_index("key")["doi"].to_dict()
    for fragment, key in MANUAL_PDF.items():   # give DOI-less PDFs the DOI (or a key marker) of their paper
        hit = art["pdf_path"].str.contains(fragment, regex=False)
        known_doi = str(role_keys.get(key) or "")
        art.loc[hit, "doi"] = known_doi if known_doi.startswith("10.") else f"key:{key}"

    role_table = roles()
    keyed = role_table["doi"].eq("") | ~role_table["doi"].str.startswith("10.")
    role_table.loc[keyed & role_table["key"].isin(MANUAL_PDF.values()), "doi"] = "key:" + role_table["key"]
    with_doi = art[art["doi"] != ""].groupby("doi", as_index=False).agg(
        pdf_path=("pdf_path", "; ".join), pdf_title=("pdf_title", "first"), file_key=("file_key", lambda s: next((k for k in s if k), "")))
    index = pd.concat([role_table[role_table["doi"] != ""].merge(with_doi, on="doi", how="outer"),
                       role_table[role_table["doi"] == ""],            # cited works without a DOI in the index
                       art[art["doi"] == ""]], ignore_index=True)      # PDFs whose DOI could not be read
    index["role"] = index["role"].fillna("not classified")
    index["status"] = index["status"].fillna("file present; role to be assigned")
    index["key"] = index["key"].fillna("")

    def si_for(row) -> str:
        hits = sup[(sup["file_key"] != "") & (sup["file_key"] == row["file_key"])]["si_path"].tolist() if row.get("file_key") else []
        hits += [s for frag, key in MANUAL_SI.items() if key == row["key"] for s in sup["si_path"] if frag in s]
        return "; ".join(dict.fromkeys(hits))

    index["si_paths"] = index.apply(si_for, axis=1)
    refs = reference_list_text()

    def in_refs(key: str) -> str:
        m = re.match(r"([A-Za-z]+)_.*?(\d{4})", key or "")
        if not m:
            return ""
        return "yes" if re.search(rf"{m.group(1)}[^\n]{{0,400}}{m.group(2)}", refs) else "no"

    index["in_submitted_reference_lists"] = index["key"].map(in_refs)
    index["pdf_present"] = index["pdf_path"].notna().map({True: "yes", False: "no"})
    order = {"laboratory dataset": 0, "laboratory candidate": 1, "field evidence": 2, "scale-up input": 3,
             "methods reference": 4, "not classified": 5}
    index = index.sort_values(["role", "key"], key=lambda s: s.map(order) if s.name == "role" else s.str.lower())
    cols = ["key", "doi", "role", "status", "note", "pdf_present", "pdf_path", "si_paths", "in_submitted_reference_lists", "pdf_title"]
    index["doi"] = index["doi"].where(~index["doi"].astype(str).str.startswith("key:"), "")
    index[cols].to_csv(REV / "05_Literature_search" / "literature_index.csv", index=False, encoding="utf-8-sig")
    unmatched = sup[~sup["si_path"].isin("; ".join(index["si_paths"]).split("; "))]
    with pd.ExcelWriter(REV / "05_Literature_search" / "literature_index.xlsx") as writer:
        index[cols].to_excel(writer, sheet_name="papers", index=False)
        unmatched[["si_path"]].to_excel(writer, sheet_name="other_files", index=False)

    print(f"files scanned: {len(files)} ({len(art)} article PDFs, {len(sup)} other files)")
    print(index.groupby(["role", "status"]).agg(papers=("key", "size"), with_pdf=("pdf_present", lambda s: (s == "yes").sum())).to_string())
    lab = index[index["role"] == "laboratory dataset"]
    print("\nLaboratory studies without a PDF:", lab.loc[lab["pdf_present"] == "no", "key"].tolist())
    print("Papers with a role but missing from the submitted reference lists:",
          index.loc[(index["in_submitted_reference_lists"] == "no") & (index["role"] != "not classified"), "key"].tolist())
    print(f"Not classified PDFs: {int((index['role'] == 'not classified').sum())}; other files not linked to a paper: {len(unmatched)}")
    pd.set_option("display.width", 250, "display.max_colwidth", 90)
    print(index.loc[index["role"] == "not classified", ["doi", "pdf_path", "pdf_title"]].to_string(index=False))
    print(unmatched["si_path"].to_string(index=False))
    missing = index[(index["pdf_present"] == "no") & (index["role"] != "methods reference")]
    print("\nWith a role but no PDF:"); print(missing[["key", "doi", "role", "status"]].to_string(index=False))


if __name__ == "__main__":
    main()
