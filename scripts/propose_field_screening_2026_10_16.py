"""Proposed title/abstract decisions for the field-study search of 16 Oct 2026 (Scopus 755, WoS 690).

The decisions are proposals by Claude from titles and abstracts; an author must confirm or change each one in
screening_sheet.xlsx (columns decision_title_abstract, reason, screener) before the counts enter the PRISMA diagram.
Record numbers (i) refer to the row order of screening_sheet.xlsx written by screen_snowball_field.py.
"""
import re
import sys
from pathlib import Path
import pandas as pd

W = Path(sys.argv[1])
d = pd.read_excel(W / "screening_sheet.xlsx").fillna("")
d.insert(0, "i", range(len(d)))

FIELD = {  # field, freshwater or estuarine, metals measured on/in collected or deployed plastics
    36: "Cu on MPs, lower Chao Phraya; check for paired dissolved Cu (possible tier 1)",
    7: "Cu on MPs in surface water of a mining city; check for paired dissolved Cu (possible tier 1)",
    359: "Cu on MPs in surface water and sediment, Subarnarekha and Kharkai rivers; check units and water Cu",
    833: "plastics deployed in the Krka estuary; adsorbed Cu and water-column Cu (possible tier 1, deployed plastics)",
    442: "plastic debris deployed in different freshwaters; metals on plastics and in water column",
    414: "MPs deployed in an estuary; Cu in the plastisphere",
    465: "Cu on EPS MPs from island beaches of the Pearl River Estuary; metals in water also reported",
    473: "Cu in MPs from mangrove sediments, Jinjiang Estuary (tier 2 candidate)",
    37: "MPs and Cu in estuarine sediments of Poyang Lake and the Yangtze; check whether Cu was measured on MPs",
    39: "Cu in MPs from Beijiang River sediments (tier 2 candidate)",
    29: "Cu on MPs collected from the Ofanto River (tier 2 candidate)",
    28: "Cu adsorbed on MPs in an estuary and a coastal site (Croatia)",
    361: "Cu on plastics in urban rivers of Mongolia (tier 2 candidate)",
    851: "Cu on environmental plastics, Lake Como (tier 2 candidate; Cu partly from the polymer matrix)",
    11: "Cu adsorbed on MPs in Lake Iznik sediments; check whether quantitative",
    4: "Cu accumulated on MPs in hyporheic sediments of the Beiluo River",
    145: "MP-bound heavy metals in Noyyal River sediments; check whether Cu was measured",
    149: "MPs and heavy metals in rivers near industrial areas; check whether metals were measured on MPs",
    249: "metals in biofilms on MPs from the Brantas River; check for Cu",
    455: "Cu on MPs in a natural wastewater-treatment wetland (ponds and canals); check matrix",
    241: "Cu on MPs below detection in aquatic sediments; check sites",
    3: "MPs and heavy metals (incl. Cu) in wastewater effluents; check whether metals were measured on MPs",
    38: "Cu on beach MPs in Hong Kong, Pearl River influence; borderline coastal",
    150: "heavy metals on MPs from water and sediments of the Suzhou River; check for Cu",
    189: "MP-associated heavy metals in hyporheic sediments (Beiluo River); check for Cu",
    295: "heavy metals in MPs from urban river sediments, China; check for Cu",
    338: "MPs with heavy metals in the Chao Phraya, spatial and seasonal; check for Cu on MPs and dissolved Cu (possible tier 1)",
    405: "metals in MPs from surface water and sediment of the Buriganga (mg/kg); check for Cu",
    476: "metals incl. Cu within MPs, Sungai Dungun (Malaysia); check units",
    396: "Cu in MPs from a stream receiving urban effluent (LIBS); check whether quantitative",
}
LAB = {
    40: "laboratory adsorption of Cu by virgin and beached pellets in river water; check for isotherm data",
    26: "laboratory adsorption of Cu by PP and PE pellets; check medium and isotherm data",
    45: "Cu adsorption isotherms on MPs in domestic sewage; check matrix and data",
    592: "aged MPs and heavy-metal ions under hydrodynamic conditions; check for Cu isotherms",
    27: "Cu adsorption by MPs under simulated wetland (sediment) conditions; check for aqueous isotherms",
}
MARINE = [5, 6, 23, 24, 34, 41, 43, 44, 99, 101, 151, 181, 209, 239, 242, 259, 285, 286, 313, 332, 340, 370, 413, 435,
          456, 468, 475, 478, 491, 496, 500, 504, 507, 516, 519, 557, 558, 573, 580, 845, 850, 854, 858, 865]
QUAL = [30, 33, 71, 87, 116, 127, 137, 200, 204, 246, 269, 315, 321, 349, 375, 383, 410, 412, 451, 464, 470, 484, 503, 566]
NO_CU = {55: "Cu not analysed", 290: "Cu not analysed (Pb only)", 505: "Cu not analysed (Cd, Pb, Br)",
         253: "Cu not reported on MPs", 65: "Cu not analysed (As, Cd, Pb)", 319: "Cu not analysed (Hg)",
         357: "Cu not analysed (U, As)", 868: "Cu not analysed (Hg)", 494: "Cu not analysed (Se)", 144: "Cd only"}

BIOTA = re.compile(r"fish|mussel|oyster|clam|bivalve|crab|shrimp|prawn|zebrafish|daphnia|toxic|exposure|organism|"
                   r"seabird|bird|turtle|dolphin|eel|catfish|tilapia|carp|biomarker|bioaccum|ingest|frog|amphibian|"
                   r"sponge|zooplankton|copepod|plant|algae|algal|phytoplankton|earthworm|duck|armadillo|insect|"
                   r"gastropod|urchin|perch|seahorse|biota|polychaete|tadpole|food|seafood|human|health",
                   re.I)
TERR = re.compile(r"soil|dust|atmospher|air|landfill|sludge|compost|groundwater|road|snow|peat|incinerat|farmland|"
                  r"agricultur|indoor|e-waste|salt\b|beverage|drinking|tap water", re.I)
MAT = re.compile(r"removal|degradation|catalys|sensor|detection|adsorbent|membrane|biochar|framework|MOF|photo|"
                 r"spectroscop|LIBS|XRF|Raman|method|remediation|treatment|flotation|flocculation|nanocomposite|"
                 r"nanoparticle|nanoplastic|leach|release|life cycle|model", re.I)

dec, why = [], []
for _, r in d.iterrows():
    i, t, f = r["i"], str(r["title"]), str(r["auto_flag"])
    if i in FIELD:
        dec.append("include for full text (field)"); why.append(FIELD[i])
    elif i in LAB:
        dec.append("include for full text (laboratory)"); why.append(LAB[i])
    elif f.startswith("5 no microplastic"):
        dec.append("exclude"); why.append("not microplastics")
    elif f.startswith("5 no metal"):
        dec.append("exclude"); why.append("metals not studied")
    elif f.startswith("4 review"):
        dec.append("exclude"); why.append("review, no primary data")
    elif i in NO_CU:
        dec.append("exclude"); why.append(NO_CU[i])
    elif i in QUAL:
        dec.append("exclude"); why.append("metals on MPs only qualitative (SEM-EDS, EDX or XRF detection)")
    elif i in MARINE + [22, 309, 354, 483, 551, 467]:
        dec.append("exclude"); why.append("marine or open-coast setting")
    elif BIOTA.search(t):
        dec.append("exclude"); why.append("biota, toxicology or health study")
    elif TERR.search(t):
        dec.append("exclude"); why.append("terrestrial, atmospheric, sludge, groundwater or drinking-water matrix")
    elif MAT.search(t):
        dec.append("exclude"); why.append("removal, detection, material or leaching study")
    else:
        dec.append("exclude"); why.append("metals not measured on microplastics (co-occurrence in water, sediment or biota only)")
d["decision_title_abstract"] = dec
d["reason"] = why
d["screener"] = "Claude (proposed; to confirm)"
d["priority"] = d["i"].map({**{k: 1 for k in [36, 338, 7, 833, 442, 465, 359, 405]}}).fillna("")
order = d["decision_title_abstract"].map({"include for full text (field)": 0, "include for full text (laboratory)": 1}).fillna(2)
d = d.assign(_o=order).sort_values(["_o", "priority", "i"], ascending=[True, True, True]).drop(columns="_o")
with pd.ExcelWriter(W / "screening_sheet_proposed.xlsx", engine="openpyxl") as xw:
    d.to_excel(xw, sheet_name="new_records", index=False)
    ws = xw.sheets["new_records"]
    for col, wd in zip("ABCDEFGHIJKLMNOP", [5, 30, 24, 26, 70, 6, 26, 22, 12, 10, 80, 30, 60, 26, 8]):
        ws.column_dimensions[col].width = wd
    ws.freeze_panes = "F2"
    ws.auto_filter.ref = ws.dimensions
s = d.groupby(["decision_title_abstract", "reason"]).size()
summ = d.groupby("decision_title_abstract").size()
print(summ.to_string()); print()
print(d[d.decision_title_abstract == "exclude"].groupby("reason").size().sort_values(ascending=False).to_string())
