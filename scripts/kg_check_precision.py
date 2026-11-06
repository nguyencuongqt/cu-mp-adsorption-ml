"""Precision of the knowledge-graph relations from the hand check by two authors (Text S4).

Input: outputs/reports/kg_relation_check.csv (one row per checked relation; Y/N for the polymer and the mechanism question
for each checker; a relation is correct when both answers are Y).
Rounds: (1) sample of 100 of 338 relations, before the rule for process mechanisms was tightened; 89 of them remain in the
final graph; (2) sample of 100 of the 203 relations of the 292-relation graph that were not in round 1, stratified by
mechanism; (3) all 39 relations of the two studies added after the Web of Science search (final graph: 331 relations).
The precision of the final graph weights each part by its number of relations (89 + 203 + 39 = 331); round 2 is
weighted by mechanism (relations in the graph per mechanism, SAMPLING_FRAME); its confidence interval uses the
stratified sampling variance of round 2.
Output: outputs/reports/kg_relation_check_summary.csv. Standard library only.
"""
from __future__ import annotations

import csv
from math import sqrt
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
IN = ROOT / "outputs" / "reports" / "kg_relation_check.csv"
OUT = ROOT / "outputs" / "reports" / "kg_relation_check_summary.csv"
SAMPLING_FRAME = {"surface area and pore structure": 109, "oxygen-containing functional groups": 63,
                  "electrostatic attraction": 18, "surface complexation": 12, "surface precipitation": 1}  # round 2, 203 relations


def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return c - h, c + h


def kappa(x: list[bool], y: list[bool]) -> float:
    n = len(x)
    po = sum(a == b for a, b in zip(x, y)) / n
    px, py = sum(x) / n, sum(y) / n
    pe = px * py + (1 - px) * (1 - py)
    return (po - pe) / (1 - pe) if pe < 1 else 1.0


def main() -> None:
    rows = list(csv.DictReader(IN.open(encoding="utf8")))
    for r in rows:
        r["ok1"] = r["checker_1_polymer"] == "Y" and r["checker_1_mechanism"] == "Y"
        r["ok2"] = r["checker_2_polymer"] == "Y" and r["checker_2_mechanism"] == "Y"
    out = []
    for rnd in ("1", "2", "3"):
        sub = [r for r in rows if r["round"] == rnd]
        n = len(sub)
        for c in ("1", "2"):
            k = sum(r["ok" + c] for r in sub)
            lo, hi = wilson(k, n)
            out.append({"part": f"round {rnd}", "checker": c, "correct": k, "n": n, "precision": round(k / n, 3),
                        "ci95_low": round(lo, 3), "ci95_high": round(hi, 3),
                        "cohen_kappa": round(kappa([r["ok1"] for r in sub], [r["ok2"] for r in sub]), 2)})
    for c in ("1", "2"):
        r1 = sum(r["ok" + c] for r in rows if r["round"] == "1" and r["in_final_graph"] == "True")
        r2 = var = 0.0
        for mech, n_graph in SAMPLING_FRAME.items():
            sub = [r for r in rows if r["round"] == "2" and r["mechanism"] == mech]
            n, p = len(sub), sum(r["ok" + c] for r in sub) / len(sub)
            r2 += n_graph * p
            if n > 1:   # stratified sampling variance with finite-population correction; rounds 1 (retained) and 3 are complete
                var += n_graph ** 2 * (1 - n / n_graph) * p * (1 - p) / (n - 1)
        r3 = sum(r["ok" + c] for r in rows if r["round"] == "3")
        est, half = (r1 + r2 + r3) / 331, 1.96 * sqrt(var) / 331
        out.append({"part": "final graph (331 relations, weighted)", "checker": c, "correct": round(r1 + r2 + r3, 1), "n": 331,
                    "precision": round(est, 3), "ci95_low": round(est - half, 3), "ci95_high": round(est + half, 3), "cohen_kappa": ""})
    with OUT.open("w", newline="", encoding="utf8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(out[0]))
        w.writeheader()
        w.writerows(out)
    for o in out:
        print(o)


if __name__ == "__main__":
    main()
