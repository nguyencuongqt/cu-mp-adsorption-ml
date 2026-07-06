"""Regenerate LLM-as-Judge plots from saved score CSVs.

This avoids re-running the judge API calls when only the figure styling changes.

Usage:
    python scripts/regenerate_llm_judge_figures.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.rag.llm_judge import _bar_chart, _radar_chart, _stat_tests  # noqa: E402


def main() -> None:
    scores_path = ROOT / "outputs" / "reports" / "llm_judge_scores.csv"
    if not scores_path.exists():
        raise FileNotFoundError(f"Missing score file: {scores_path}")

    scores_df = pd.read_csv(scores_path)
    tests = _stat_tests(scores_df)
    bar_path = _bar_chart(scores_df, tests)
    radar_path = _radar_chart(scores_df)
    print(f"Saved: {bar_path}")
    print(f"Saved: {radar_path}")


if __name__ == "__main__":
    main()
