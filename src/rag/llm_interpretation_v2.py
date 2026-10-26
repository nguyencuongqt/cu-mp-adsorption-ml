"""LLM Interpretation System for the revised model.

Same three explanation modes as the submitted version (1 = rule-based, 2 = retrieval-augmented,
3 = retrieval plus knowledge graph), now driven by ``src/analysis/predictor.py``: the explanation has
to convey the prediction, its study-level interval, the validity warnings and the field-scale value.
"""

from __future__ import annotations

import html
import json
import re
import sys
from functools import lru_cache
from pathlib import Path
from typing import Any

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
MODULE_DIR = Path(__file__).resolve().parent
for path in (PROJECT_ROOT, MODULE_DIR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import config  # noqa: E402
from config import CHUNKS_FILE  # noqa: E402
from src.analysis.predictor import fmt_qe, fmt_range, model_summary, predict, sig  # noqa: E402
from llm_clients import GENERATOR, call_llm  # noqa: E402

LABELS = {"Ce": "equilibrium concentration", "ReT": "polymer type", "AgS": "ageing status", "AdC": "adsorption system"}
SYSTEM = ("You explain predictions of Cu(II) sorption on microplastics to environmental scientists. Use only the facts and the "
          "context given. Report numbers exactly as given. Never invent a citation: cite only source ids that appear in the context.")


def facts_text(pred: dict[str, Any]) -> str:
    """The facts every explanation must be consistent with (also shown to the judges)."""
    v, s = pred["inputs"], model_summary()
    q, iv = pred["lab_qe"], pred["river_qe"]
    lines = [
        f"Inputs: polymer {v['ReT']}; {v['AgS']} particles; {v['AdC']} system; Ce = {v['Ce']:.4g} mg/L.",
        f"Predicted laboratory-equilibrium qe: {fmt_qe(q)}.",
        f"90% prediction interval {'with the local measurement supplied by the user' if pred['anchored'] else 'for a laboratory not in the training data'}: "
        f"{fmt_range(pred['lab_qe_lower'], pred['lab_qe_upper'], q)} "
        f"(prediction multiplied and divided by {sig(pred['interval_factor'])}).",
        f"With one local measurement for the same material the interval narrows to the prediction multiplied and divided by "
        f"{sig(pred['interval_factor_with_one_local_measurement'])}.",
        f"Field-to-laboratory ratio (dimensionless): {sig(pred['field_ratio'])}, 95% confidence interval "
        f"{sig(pred['field_ratio_ci'][0])} to {sig(pred['field_ratio_ci'][1])}, from {pred['n_field_studies']} field studies.",
        f"Field-scale value (laboratory qe times that ratio): {fmt_qe(iv)}; 90% interval for a river not among the field studies, "
        f"combining the model interval with the variation of the ratio between rivers: {fmt_range(pred['river_qe_lower'], pred['river_qe_upper'], iv)} "
        f"(multiplied and divided by {sig(pred['river_interval_factor'])}).",
        "Effect of each input on the prediction, as a multiplier of qe relative to the average laboratory observation: "
        + "; ".join(f"{LABELS[d['predictor']]} multiplies qe by {sig(d['factor_on_qe'])}" for d in pred["drivers"]) + ".",
        f"Model: pooled Freundlich isotherm fitted to {s['n_points']} observations from {s['n_studies']} studies; "
        f"Freundlich exponent {pred['freundlich_exponent']:.2f}; laboratory Ce range {s['ce_min']:.2g}–{s['ce_max']:.3g} mg/L; "
        f"{v['ReT']} is represented by {pred['polymer_n_studies']} studies.",
    ]
    lines += [f"WARNING: {w}" for w in pred["warnings"]] or ["No validity warning applies to this input."]
    return "\n".join(lines)


def numbers_only(pred: dict[str, Any]) -> str:
    """Condition C0: the numerical output without any explanation."""
    return (f"lab_qe={fmt_qe(pred['lab_qe'])}; 90% PI: {fmt_range(pred['lab_qe_lower'], pred['lab_qe_upper'], pred['lab_qe'])}; "
            f"river_qe={fmt_qe(pred['river_qe'])}")


def rule_based_html(pred: dict[str, Any]) -> str:
    """Mode 1: deterministic explanation generated from the model output only (no LLM, no retrieval)."""
    v = pred["inputs"]
    q, iv = pred["lab_qe"], pred["river_qe"]
    drivers = "".join(f"<li>{html.escape(LABELS[d['predictor']])}: {d['direction']} qe (factor {sig(d['factor_on_qe'])})</li>"
                      for d in pred["drivers"])
    warnings = "".join(f"<li>{html.escape(w)}</li>" for w in pred["warnings"]) or "<li>No validity warning applies.</li>"
    interval = ("with the local measurement" if pred["anchored"] else "for a new laboratory")
    return (
        "<section><h3>Rule-based local interpretation</h3>"
        f"<p>Predicted laboratory qe is <b>{fmt_qe(q)}</b> (90% interval {interval}: "
        f"{fmt_range(pred['lab_qe_lower'], pred['lab_qe_upper'], q)}, i.e. the prediction multiplied and divided by "
        f"{sig(pred['interval_factor'])})"
        + ("" if pred["anchored"] else
           f"; one local measurement would narrow this to a factor of {sig(pred['interval_factor_with_one_local_measurement'])}")
        + ".</p>"
        f"<p>Field-scale value: <b>{fmt_qe(iv)}</b>, using a field-to-laboratory ratio of {sig(pred['field_ratio'])} from "
        f"{pred['n_field_studies']} field studies; 90% interval for a river not among these studies: "
        f"{fmt_range(pred['river_qe_lower'], pred['river_qe_upper'], iv)}.</p>"
        f"<p>Drivers relative to the average laboratory observation:</p><ul>{drivers}</ul>"
        f"<p>{html.escape(v['ReT'])} is represented by {pred['polymer_n_studies']} laboratory "
        f"stud{'y' if pred['polymer_n_studies'] == 1 else 'ies'}.</p>"
        f"<p>Validity checks:</p><ul>{warnings}</ul></section>"
    )


def retrieval_available() -> bool:
    """C2 and C3 need the retrieval index and the text chunks built locally from the source articles (not redistributed)."""
    return config.FAISS_INDEX_FILE.exists() and config.CHUNKS_METADATA_FILE.exists() and CHUNKS_FILE.exists()


# ------------------------------------------------------------------ retrieval (mode 2) and graph evidence (mode 3)
POLYMER_NAMES = {
    "PE": r"\b(?:PE|[HL]DPE|LLDPE)\b|polyethylene(?! terephthalate)", "PP": r"\bPP\b|polypropylene", "PS": r"\bPS\b|polystyrene",
    "PLA": r"\bPLA\b|polylactic|poly\(lactic", "PA": r"\bPA6?6?\b|polyamide|nylon", "PVC": r"\bPVC\b|polyvinyl chloride",
    "PET": r"\bPET\b|polyethylene terephthalate", "PBAT": r"\bPBAT\b", "TPU": r"\bTPU\b|polyurethane",
    "PMMA": r"\bPMMA\b|methyl methacrylate", "CPE": r"\bCPE\b|chlorinated polyethylene",
}
POLYMER_FULL = {"PE": "polyethylene", "PP": "polypropylene", "PS": "polystyrene", "PLA": "polylactic acid", "PA": "polyamide",
                "PVC": "polyvinyl chloride", "PET": "polyethylene terephthalate", "PBAT": "PBAT", "TPU": "thermoplastic polyurethane",
                "PMMA": "polymethyl methacrylate", "CPE": "chlorinated polyethylene"}
MECHANISM_TERMS = {
    "electrostatic attraction": r"electrostatic",
    "surface complexation": r"complexation|complex(?:es|ed|ing)? with|chelat",
    "ion exchange": r"ion[- ]exchange|cation exchange",
    "surface precipitation": r"precipitat",
    "oxygen-containing functional groups": r"oxygen[- ]containing|carboxyl|carbonyl|hydroxyl|C=O|COOH",
    "surface area and pore structure": r"\bsurface area|\b(?:micro|meso|macro)?pores?\b|\bporosity|\bporous\b(?! medi)|\brough|\bcrack",
}
SOLUTION_COMPLEX_RE = re.compile(r"humic|fulvic|\bFA\b|\bHA\b|\bDOM\b|dissolved organic", re.I)   # complexation in solution, not on the particle
PROCESS_MECHANISMS = {"electrostatic attraction", "surface complexation", "ion exchange", "surface precipitation"}
METAL_RE = re.compile(r"\bCu\b|Cu\(II\)|Cu2\+|copper|\bmetals?\b|heavy metal|\bPb\b|\bCd\b|\bZn\b|\bNi\b|cation|metal ion", re.I)
UPTAKE_RE = re.compile(r"adsor|sorption|sorbed|uptake|bind|bound|capacity|affinity|interact|retention|accumul", re.I)
ORGANIC_SORBATE_RE = re.compile(r"tetracycline|\bTC\b|\bOTC\b|\bCIP\b|ciprofloxacin|antibiotic|pharmaceutical|sulfonamide|\bSA\b|\bSMX\b|\bPAE\b|phthalate", re.I)
NEGATED_RE = re.compile(r"repuls|electrostatic potential|not (?:the|a) (?:main|dominant)|exclud|difficult", re.I)
AGEING_RE = re.compile(r"\b(?:aged|aging|ageing|weather\w*|UV|photo\w*|oxidi[sz]\w*|irradiat\w*)\b", re.I)
NOT_INPUTS_RE = re.compile(r"\bpH\b|temperature|ionic strength|salinity|shaking|humic|fulvic|dissolved organic", re.I)
REFERENCE_RE = re.compile(r"et al\.,|doi|https?:|\bvol\.|\bpp\.|\(\d{4}\)\s*\d+|\d{4};\s*\d+|\b\d+\s*\(\d+\)\s*,?\s*\d+", re.I)


def _is_reference_list(text: str) -> bool:
    return len(re.findall(r"\b(?:19|20)\d\d\b", text)) > 14 and len(re.findall(r"et al|https?:|doi", text, re.I)) > 5


@lru_cache(maxsize=1)
def _paper_chunks() -> list[dict[str, Any]]:
    with CHUNKS_FILE.open("r", encoding="utf-8") as handle:
        chunks = [json.loads(line) for line in handle if line.strip()]
    return [c for c in chunks if str(c.get("source_type", "")).endswith("_paper")]


@lru_cache(maxsize=1)
def _relations() -> list[dict[str, Any]]:
    """Sentence-level relations polymer - mechanism - ageing - paper extracted from the corpus."""
    seen, out = set(), []
    for chunk in _paper_chunks():
        if _is_reference_list(chunk["text"]):
            continue
        for sent in re.split(r"(?<=[.;])\s+(?=[A-Z])", chunk["text"]):
            sent = sent.strip()
            if not 80 <= len(sent) <= 420 or REFERENCE_RE.search(sent) or sent[:70] in seen or sent[-70:] in seen:
                continue
            if sum(ch.isalpha() or ch.isspace() for ch in sent) < 0.8 * len(sent):      # table rows and peak lists
                continue
            mechs = [m for m, pat in MECHANISM_TERMS.items() if re.search(pat, sent, re.I)]
            if "surface complexation" in mechs and SOLUTION_COMPLEX_RE.search(sent):
                mechs.remove("surface complexation")
            # Process mechanisms count only when the sentence is about the uptake of a metal (rule added after the
            # first hand check of 100 relations, 10 October 2026); surface properties need no such statement.
            if not (METAL_RE.search(sent) and UPTAKE_RE.search(sent)) or ORGANIC_SORBATE_RE.search(sent) or NEGATED_RE.search(sent):
                mechs = [m for m in mechs if m not in PROCESS_MECHANISMS]
            if not mechs:
                continue
            polymers = [p for p, pat in POLYMER_NAMES.items() if re.search(pat, sent)]
            seen.update((sent[:70], sent[-70:]))
            out.append({"paper": chunk["paper_id"], "text": sent, "mechanisms": mechs, "polymers": polymers,
                        "ageing": bool(AGEING_RE.search(sent)), "copper": bool(re.search(r"\bCu\b|Cu\(II\)|Cu2\+|copper|metal", sent)),
                        "other_factor": bool(NOT_INPUTS_RE.search(sent))})
    return out


def papers_mentioning(polymer: str) -> set[str]:
    """Corpus papers whose text names the polymer (used to check that a citation is relevant to the case)."""
    pat = re.compile(POLYMER_NAMES.get(polymer, rf"\b{re.escape(polymer)}\b"))
    return {c["paper_id"] for c in _paper_chunks() if pat.search(c["text"])}


def graph_evidence(polymer: str, ags: str, max_passages: int = 6) -> tuple[str, list[dict[str, Any]]]:
    """Mechanisms linked to this polymer (and ageing status) in the corpus, with the supporting sentences."""
    aged = str(ags).lower().startswith("aged")
    rel = [r for r in _relations() if polymer in r["polymers"]]
    lines = []
    for mech in MECHANISM_TERMS:
        hits = [r for r in rel if mech in r["mechanisms"]]
        papers = sorted({r["paper"] for r in hits})
        if not papers:
            continue
        with_ageing = sorted({r["paper"] for r in hits if r["ageing"]})
        line = f"{polymer} -[{mech}]-> reported in {len(papers)} paper(s): {', '.join(papers[:6])}"
        if aged and with_ageing:
            line += f"; linked to ageing in {len(with_ageing)}: {', '.join(with_ageing[:6])}"
        lines.append((len(papers), line))
    lines = [text for _, text in sorted(lines, reverse=True)]
    ranked = sorted(rel, key=lambda r: (not r["other_factor"], r["ageing"] == aged, r["copper"], len(r["mechanisms"])), reverse=True)
    passages, per_paper = [], {}
    for r in ranked:
        if per_paper.get(r["paper"], 0) >= 2:
            continue
        per_paper[r["paper"]] = per_paper.get(r["paper"], 0) + 1
        passages.append({"paper_id": r["paper"], "text": r["text"], "chunk_id": f"{r['paper']}::relation"})
        if len(passages) == max_passages:
            break
    if not lines:
        return (f"No mechanism is linked to {polymer} in the corpus; say that the mechanism for this polymer is not documented "
                f"in the corpus instead of assuming one.", [])
    text = ("Mechanism relations for this polymer:\n" + "\n".join(lines) + "\nSupporting sentences:\n"
            + "\n".join(f"[Source: {p['paper_id']}] {p['text']}" for p in passages))
    return text, passages


def retrieve_context(v: dict[str, Any], top_k: int = 5, words_per_passage: int = 200) -> tuple[str, list[dict[str, Any]]]:
    """Vector retrieval for the case; reference lists are dropped and passages naming the polymer come first."""
    from vector_store import retrieve

    state = "aged (weathered, oxidised)" if str(v["AgS"]).lower().startswith("aged") else "virgin (pristine)"
    system = "in the presence of other metal ions (competitive adsorption)" if v["AdC"] == "Mixed" else "from a single-metal solution"
    query = (f"Mechanism of Cu(II) adsorption on {state} {POLYMER_FULL.get(v['ReT'], v['ReT'])} ({v['ReT']}) microplastics "
             f"{system}; surface functional groups and adsorption capacity")
    hits = [h for h in retrieve(query, top_k=30) if str(h.get("source_type", "")).endswith("_paper") and not _is_reference_list(h["text"])]
    pat = re.compile(POLYMER_NAMES.get(v["ReT"], rf"\b{re.escape(v['ReT'])}\b"))
    hits = sorted(hits, key=lambda h: (bool(pat.search(h["text"])), h["score"]), reverse=True)
    chosen, per_paper = [], {}
    for h in hits:
        if per_paper.get(h["paper_id"], 0) >= 2:
            continue
        per_paper[h["paper_id"]] = per_paper.get(h["paper_id"], 0) + 1
        chosen.append(h)
        if len(chosen) == top_k:
            break
    context = "\n\n".join(f"[Source: {h['paper_id']}] " + " ".join(h["text"].split()[:words_per_passage]) for h in chosen)
    return context, chosen


def llm_html(pred: dict[str, Any], mode: int) -> tuple[str, list[dict[str, Any]]]:
    """Modes 2 and 3: explanation written by the LLM from retrieved context (mode 3 adds the graph evidence)."""
    v = pred["inputs"]
    context, sources = retrieve_context(v)
    extra = ""
    if mode == 3:
        graph_text, passages = graph_evidence(v["ReT"], v["AgS"])
        sources = sources + passages
        extra = f"\n\nKNOWLEDGE GRAPH\n{graph_text}"
    task = ("Write 180-240 words of HTML covering, in this order: (1) the predicted qe and its 90% prediction interval; (2) why the value "
            "is what it is: the contribution of each of the four inputs (equilibrium concentration, polymer, ageing status, adsorption "
            "system) with its multiplier, and the sorption mechanism that the context reports for this polymer and this ageing status; "
            "(3) the field-scale value and what the field-to-laboratory ratio means; (4) every WARNING listed in the facts, restated in "
            "plain words, and what one local measurement would change.\n"
            "Rules: cite sources as [Source_id] using only ids that appear in the context, and only for statements taken from a context "
            "passage; cite two or more different sources when the context supports it, each in the form 'Source_id reports that ...' "
            "followed by the finding of that source, and do not cite a passage that concerns another polymer or another metal. The FACTS come from the model and "
            "need no citation. The mechanistic reasoning must agree with the direction of each multiplier: for virgin particles "
            "explain the lower qe by the absence of the surface changes that ageing produces, and never describe a virgin particle as "
            "aged. Do not calculate any value that is not in the FACTS (for example, do not turn the factor for one local measurement "
            "into a range in mg/g). pH, temperature, ionic strength, organic "
            "matter and shaking are not inputs of the model: do not use them to explain the prediction. If the context reports no "
            "mechanism for this polymer, say so instead of assuming one. Give every value with its unit; do not attach the unit mg/g to "
            "the dimensionless ratio. Return the HTML only, without code fences.")
    if mode == 3:
        task += (" Name the mechanisms listed in the KNOWLEDGE GRAPH for this polymer, starting with those reported by most papers, "
                 "and state how they relate to the ageing status of the particles.")
    user = f"FACTS\n{facts_text(pred)}\n\nCONTEXT\n{context}{extra}\n\nTASK\n{task}"
    text = _clean(call_llm(*GENERATOR, system=SYSTEM, user=user, temperature=0.2))
    attempts = 1
    wrong = unsupported_values(plain_text(text), pred)
    if wrong:      # numerical check against the model output: one corrective regeneration
        note = (f"\n\nYour previous answer gave these values, which are not in the FACTS: {', '.join(wrong)}. Write the answer again "
                "using only the values in the FACTS.")
        text = _clean(call_llm(*GENERATOR, system=SYSTEM, user=user + note, temperature=0.0))
        attempts = 2
    LAST_GENERATION.update(attempts=attempts, unsupported_after_check=len(unsupported_values(plain_text(text), pred)))
    return text, sources


LAST_GENERATION: dict[str, int] = {}


def _clean(text: str) -> str:
    return re.sub(r"^\s*```(?:html)?\s*|\s*```\s*$", "", text.strip())   # remove code fences if the model adds them


def unsupported_values(plain: str, pred: dict[str, Any], tol: float = 0.06) -> list[str]:
    """Values given in mg/g (or ug/g) that match none of the predicted quantities."""
    truth = [pred[k] for k in ("lab_qe", "lab_qe_lower", "lab_qe_upper", "river_qe", "river_qe_lower", "river_qe_upper")]
    bad = []
    for m in re.finditer(r"(\d+(?:\.\d+)?(?:[eE][-+]?\d+)?)\s*(?:to|-|\u2013)?\s*(\d+(?:\.\d+)?(?:[eE][-+]?\d+)?)?\s*(mg/g|\u00b5g/g|ug/g|\u03bcg/g)", plain):
        scale = 1.0 if m.group(3) == "mg/g" else 1e-3
        for g in (m.group(1), m.group(2)):
            if g is not None and not any(abs(float(g) * scale - t) <= tol * abs(t) for t in truth):
                bad.append(f"{g} {m.group(3)}")
    return bad


def explain(values: dict[str, Any], mode: int = 1, anchor: tuple[float, float] | None = None) -> dict[str, Any]:
    """Explanation for mode 0 (numbers only), 1, 2 or 3."""
    pred = predict(values, anchor=anchor)
    sources: list[dict[str, Any]] = []
    if mode == 0:
        text = numbers_only(pred)
    elif mode == 1:
        text = rule_based_html(pred)
    else:
        text, sources = llm_html(pred, mode)
    return {"generation": dict(LAST_GENERATION) if mode >= 2 else {}, "mode": mode, "explanation_html": text, "sources": sources, "prediction": pred, "facts": facts_text(pred)}


def plain_text(text: str) -> str:
    return re.sub(r"\s+", " ", re.sub(r"<[^>]+>", " ", str(text))).strip()


def readable_text(text: str) -> str:
    """HTML explanation as text with paragraphs and list items on their own lines (for the reports)."""
    t = re.sub(r"<li[^>]*>", "\n- ", str(text), flags=re.I)
    t = re.sub(r"</(p|h\d|ul|ol|section|div)>|<br\s*/?>", "\n", t, flags=re.I)
    t = html.unescape(re.sub(r"<[^>]+>", "", t))
    lines = [re.sub(r"[ \t]+", " ", line).strip() for line in t.splitlines()]
    return re.sub(r"\n{3,}", "\n\n", "\n".join(lines)).strip()


if __name__ == "__main__":
    demo = {"ReT": "PP", "AgS": "aged", "AdC": "mixed", "Ce": 0.00148}
    for m in (0, 1):
        print(f"\nMODE {m}\n{plain_text(explain(demo, mode=m)['explanation_html'])}")
    print("\nFACTS\n" + explain(demo, mode=0)["facts"])
