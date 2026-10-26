"""Prepare title/abstract screening of the citation search (snowballing) and of the field-study search.

Input (exports placed by the authors):
    05_Literature_search/Snowball_field_2026-10/exports/backward/      Scopus CSV  ("View references" of the seed set)
    05_Literature_search/Snowball_field_2026-10/exports/forward/       Scopus CSV  ("View cited by" of the seed set)
    05_Literature_search/Snowball_field_2026-10/exports/field_scopus/  Scopus CSV  (field-study search string)
    05_Literature_search/Snowball_field_2026-10/exports/field_wos/     WoS .xls/.txt (optional)

Output:
    05_Literature_search/Snowball_field_2026-10/screening_sheet.xlsx   one row per new record, with empty decision columns
    05_Literature_search/Snowball_field_2026-10/screening_counts.md     counts for the PRISMA "other methods" branch

Records already in the PRISMA ledger (database searches), in literature_index.csv or in the seed set are counted as
duplicates and are not screened again. The keyword flags only sort the sheet; every new record still gets a human
decision.
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
LIT = ROOT / "Manuscript" / "Submission_Env pollution" / "Revision" / "05_Literature_search"
WORK = Path(sys.argv[1]) if len(sys.argv) > 1 else LIT / "Snowball_field_2026-10"
EXPORTS = WORK / "exports"

MP = re.compile(r"micro-?plastic|nano-?plastic|plastic (particle|debris|fragment|pellet)|microbead|micro-?fib(er|re)|"
                r"synthetic fib(er|re)|polymer particle", re.I)
CU = re.compile(r"\bcopper\b|\bCu\b|Cu\(II\)|Cu2\+|Cu²⁺", re.I)
METAL = re.compile(r"heavy metal|trace metal|trace element|potentially toxic element|\bmetal(s|loid)?\b", re.I)
ADS = re.compile(r"adsor|sorp|isotherm|binding|uptake|surface complexation|partition", re.I)
FIELD = re.compile(r"\briver|stream|estuar|lake|reservoir|freshwater|surface water|coast|mangrove|beach|harbou?r|"
                   r"field|in situ|sampl|collected|monitor|survey|occurrence|environmental microplastic", re.I)
REVIEW = re.compile(r"\breview\b|meta-analysis|bibliometric", re.I)


def norm_doi(x) -> str:
    if not isinstance(x, str):
        return ""
    x = x.strip().lower()
    x = re.sub(r"^https?://(dx\.)?doi\.org/", "", x)
    return x.rstrip(".")


def norm_title(x) -> str:
    return re.sub(r"[^a-z0-9]", "", str(x).lower())[:120] if isinstance(x, str) else ""


def col(d: pd.DataFrame, *names: str) -> pd.Series:
    for n in names:
        if n in d.columns:
            return d[n]
    return pd.Series("", index=d.index)


def read_scopus(folder: Path, label: str) -> pd.DataFrame:
    frames = []
    for f in sorted(folder.glob("*.csv")):
        d = pd.read_csv(f, dtype=str)
        frames.append(pd.DataFrame({
            "search": label, "file": f.name, "doi": col(d, "DOI"), "title": col(d, "Title"), "year": col(d, "Year"),
            "journal": col(d, "Source title"), "authors": col(d, "Authors"), "abstract": col(d, "Abstract"),
            "document_type": col(d, "Document Type"), "language": col(d, "Language of Original Document")}))
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def read_wos(folder: Path, label: str) -> pd.DataFrame:
    frames = []
    for f in sorted(list(folder.glob("*.xls")) + list(folder.glob("*.xlsx")) + list(folder.glob("*.txt"))):
        d = pd.read_csv(f, sep="\t", dtype=str, quoting=3) if f.suffix == ".txt" else pd.read_excel(f, dtype=str)
        frames.append(pd.DataFrame({
            "search": label, "file": f.name, "doi": col(d, "DOI", "DI"), "title": col(d, "Article Title", "TI"),
            "year": col(d, "Publication Year", "PY"), "journal": col(d, "Source Title", "SO"),
            "authors": col(d, "Authors", "AU"), "abstract": col(d, "Abstract", "AB"),
            "document_type": col(d, "Document Type", "DT"), "language": col(d, "Language", "LA")}))
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def known_records():
    dois, titles = set(), set()
    ledger = pd.read_csv(LIT / "PRISMA" / "prisma_ledger.csv", dtype=str)
    dois |= {norm_doi(x) for x in ledger["id"]}
    titles |= {norm_title(x) for x in ledger["title"]}
    idx = pd.read_csv(LIT / "literature_index.csv", dtype=str)
    dois |= {norm_doi(x) for x in idx["doi"]}
    titles |= {norm_title(x) for x in idx.get("pdf_title", pd.Series(dtype=str))}
    seeds = WORK / "seed_set.csv"
    if seeds.exists():
        dois |= {norm_doi(x) for x in pd.read_csv(seeds, dtype=str)["doi"]}
    dois.discard("")
    titles.discard("")
    return dois, titles


def flag(row) -> str:
    text = f"{row['title']} {row['abstract']}"
    mp, cu, metal, ads, field = (bool(p.search(text)) for p in (MP, CU, METAL, ADS, FIELD))
    if not mp:
        return "5 no microplastic term"
    if REVIEW.search(str(row["document_type"])) or REVIEW.search(str(row["title"])):
        return "4 review (check its reference list only)"
    if cu and ads:
        return "1 laboratory candidate (MP + Cu + sorption)"
    if (cu or metal) and field:
        return "2 field candidate (MP + metal + field terms)"
    if cu or metal:
        return "3 MP + metal, other"
    return "5 no metal term"


def main():
    parts = [read_scopus(EXPORTS / "backward", "backward citation"),
             read_scopus(EXPORTS / "forward", "forward citation"),
             read_scopus(EXPORTS / "field_scopus", "field search, Scopus"),
             read_wos(EXPORTS / "field_wos", "field search, Web of Science")]
    allrec = pd.concat([p for p in parts if len(p)], ignore_index=True)
    if allrec.empty:
        sys.exit(f"No exports found under {EXPORTS}")
    allrec = allrec.fillna("")
    allrec["doi_n"] = allrec["doi"].map(norm_doi)
    allrec["title_n"] = allrec["title"].map(norm_title)

    counts = allrec.groupby("search").size().rename("records").to_frame()

    # duplicates within the new searches: DOI, then title (a record without DOI takes the DOI of a record with
    # the same title, so a WoS export without DOIs still matches the Scopus export)
    title_to_doi = (allrec[(allrec["doi_n"] != "") & (allrec["title_n"] != "")]
                    .drop_duplicates("title_n").set_index("title_n")["doi_n"])
    fill = allrec["title_n"].map(title_to_doi).fillna("")
    allrec.loc[allrec["doi_n"] == "", "doi_n"] = fill[allrec["doi_n"] == ""]
    for c in ["doi", "year", "journal"]:  # copy bibliographic fields from the matching record when missing
        src = allrec[allrec[c] != ""].drop_duplicates("title_n").set_index("title_n")[c]
        miss = allrec[c] == ""
        allrec.loc[miss, c] = allrec.loc[miss, "title_n"].map(src).fillna("")
    key = allrec["doi_n"].where(allrec["doi_n"] != "", "t:" + allrec["title_n"])
    allrec["key"] = key
    allrec["searches"] = allrec.groupby("key")["search"].transform(lambda s: "; ".join(dict.fromkeys(s)))
    uniq = allrec.drop_duplicates("key").copy()

    dois, titles = known_records()
    uniq["already_known"] = uniq["doi_n"].isin(dois) | uniq["title_n"].isin(titles)
    new = uniq[~uniq["already_known"]].copy()
    new["auto_flag"] = new.apply(flag, axis=1)
    new = new.sort_values(["auto_flag", "year"], ascending=[True, False])
    new["decision_title_abstract"] = ""
    new["reason"] = ""
    new["screener"] = ""
    cols = ["auto_flag", "searches", "doi", "title", "year", "journal", "authors", "document_type", "language",
            "abstract", "decision_title_abstract", "reason", "screener"]
    out = WORK / "screening_sheet.xlsx"
    with pd.ExcelWriter(out, engine="openpyxl") as xw:
        new[cols].to_excel(xw, sheet_name="new_records", index=False)
        ws = xw.sheets["new_records"]
        for col, width in zip("ABCDEFGHIJKLM", [34, 26, 28, 70, 6, 28, 26, 14, 10, 90, 18, 30, 10]):
            ws.column_dimensions[col].width = width
        ws.freeze_panes = "D2"
        ws.auto_filter.ref = ws.dimensions

    lines = ["# Citation search and field-study search: counts", "",
             "| Search | Records exported |", "|---|---:|"]
    lines += [f"| {s} | {n} |" for s, n in counts["records"].items()]
    lines += ["", f"- Records in all exports: {len(allrec)}",
              f"- Duplicates among the new searches: {len(allrec) - len(uniq)}",
              f"- Already screened in the database searches or already in the dataset: {int(uniq['already_known'].sum())}",
              f"- New records to screen by title and abstract: {len(new)}", "", "| Automatic flag | n |", "|---|---:|"]
    lines += [f"| {k} | {v} |" for k, v in new["auto_flag"].value_counts().sort_index().items()]
    (WORK / "screening_counts.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))
    print(f"\nWrote {out}")


if __name__ == "__main__":
    main()
