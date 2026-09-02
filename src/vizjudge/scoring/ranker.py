"""Evaluate and order chart candidates."""

from __future__ import annotations

import pandas as pd

from vizjudge.charts.generator import ChartCandidate
from vizjudge.scoring.rules import judge_candidate


def rank_candidates(
    frame: pd.DataFrame, candidates: list[ChartCandidate], top_k: int = 8
) -> list[ChartCandidate]:
    """Score candidates and return the highest-value subset."""
    if top_k < 1:
        raise ValueError("top_k must be at least 1")
    judged = [judge_candidate(frame, candidate) for candidate in candidates]
    return sorted(judged, key=lambda item: (-item.score, item.title))[:top_k]
