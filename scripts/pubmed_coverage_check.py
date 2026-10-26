"""PubMed search (NCBI E-utilities) and comparison with the Scopus records.

Runs the PubMed equivalent of the Scopus search string, saves the records to
Revision/PubMed_search/, and lists the PubMed records that are not among the Scopus exports
(matched by DOI, then by normalised title).
"""

from __future__ import annotations

import datetime as dt
import glob
import json
import re
import time
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
REV = ROOT / "Manuscript" / "Submission_Env pollution" / "Revision"
OUT = REV / "05_Literature_search" / "PubMed_search"
BASE = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/"
QUERY = ('(microplastic*[tiab] OR nanoplastic*[tiab] OR "plastic particle*"[tiab] OR "plastic debris"[tiab] '
         'OR microbead*[tiab] OR microfibre*[tiab] OR microfiber*[tiab] OR "synthetic fiber*"[tiab] '
         'OR "synthetic fibre*"[tiab]) AND (copper[tiab] OR "Cu(II)"[tiab] OR "copper ion*"[tiab]) '
         'AND (adsorp*[tiab] OR sorp*[tiab] OR binding[tiab] OR uptake[tiab] OR "surface complexation"[tiab])')


def get(endpoint: str, **params) -> bytes:
    url = BASE + endpoint + "?" + urllib.parse.urlencode({"tool": "cu_mp_revision", **params})
    with urllib.request.urlopen(url, timeout=60) as r:
        return r.read()


def norm_title(t: str) -> str:
    return re.sub(r"[^a-z0-9]", "", str(t).lower())


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    search = json.loads(get("esearch.fcgi", db="pubmed", term=QUERY, retmax=5000, retmode="json"))["esearchresult"]
    ids = search["idlist"]
    print(f"PubMed records: {search['count']}; query translation: {search.get('querytranslation', '')[:300]}")
    rows, raw = [], []
    for i in range(0, len(ids), 200):
        xml = get("efetch.fcgi", db="pubmed", id=",".join(ids[i:i + 200]), retmode="xml")
        raw.append(xml.decode("utf8"))
        for art in ET.fromstring(xml).iter("PubmedArticle"):
            doi = next((e.text for e in art.iter("ArticleId") if e.get("IdType") == "doi"), None)
            year = art.findtext(".//JournalIssue/PubDate/Year") or (art.findtext(".//JournalIssue/PubDate/MedlineDate") or "")[:4]
            rows.append({"pmid": art.findtext(".//PMID"), "doi": (doi or "").lower(), "year": year,
                         "journal": art.findtext(".//Journal/ISOAbbreviation") or art.findtext(".//Journal/Title"),
                         "title": "".join(art.find(".//ArticleTitle").itertext()) if art.find(".//ArticleTitle") is not None else "",
                         "publication_types": "; ".join(e.text for e in art.iter("PublicationType")),
                         "abstract": " ".join("".join(e.itertext()) for e in art.iter("AbstractText"))})
        time.sleep(0.4)
    pm = pd.DataFrame(rows)
    stamp = dt.date.today().isoformat()
    (OUT / f"pubmed_raw_{stamp}.xml").write_text("\n".join(raw), encoding="utf8")

    scopus = pd.concat([pd.read_csv(f) for f in glob.glob(str(REV / "05_Literature_search" / "export_*.csv"))], ignore_index=True)
    s_doi = set(scopus["DOI"].dropna().astype(str).str.lower())
    s_title = set(scopus["Title"].map(norm_title))
    pm["in_scopus"] = pm["doi"].isin(s_doi) | pm["title"].map(norm_title).isin(s_title)
    final = pd.read_csv(ROOT / "data" / "dataset_grouped.csv")[["source_id", "doi"]].drop_duplicates()
    f_doi = dict(zip(final["doi"].str.lower(), final["source_id"]))
    pm["in_final_dataset"] = pm["doi"].map(f_doi).fillna("")
    pm.to_csv(OUT / f"pubmed_records_{stamp}.csv", index=False, encoding="utf-8-sig")
    unique = pm[~pm["in_scopus"]]
    unique.to_csv(OUT / f"pubmed_not_in_scopus_{stamp}.csv", index=False, encoding="utf-8-sig")
    (OUT / "search_log.md").write_text(
        f"# PubMed search log\n\n- Date: {stamp}\n- Interface: NCBI E-utilities (esearch + efetch), database `pubmed`\n"
        f"- Filters: none\n- Records: {search['count']}\n- With DOI: {(pm['doi'] != '').sum()}\n"
        f"- Also in the Scopus export (DOI or title match): {int(pm['in_scopus'].sum())}\n"
        f"- Not in the Scopus export: {len(unique)}\n\n## Query\n\n```text\n{QUERY}\n```\n\n"
        f"## Query translation returned by PubMed\n\n```text\n{search.get('querytranslation', '')}\n```\n", encoding="utf8")

    print(f"fetched {len(pm)}; with DOI {(pm['doi'] != '').sum()}; in Scopus {int(pm['in_scopus'].sum())}; not in Scopus {len(unique)}")
    print("final-dataset studies retrieved by PubMed:", sorted(x for x in pm["in_final_dataset"].unique() if x))
    print("final-dataset studies NOT retrieved:", sorted(set(final["source_id"]) - set(pm["in_final_dataset"])))
    print("publication types of the PubMed-only records:", unique["publication_types"].value_counts().head(6).to_dict())
    pd.set_option("display.width", 250, "display.max_colwidth", 110)
    print(unique[["year", "journal", "title", "doi"]].to_string(index=False))


if __name__ == "__main__":
    main()
