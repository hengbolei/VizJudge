"""Evaluate and order chart candidates."""

from __future__ import annotations

import pandas as pd

from vizjudge.charts.generator import ChartCandidate
from vizjudge.core.validation import validate_top_k
from vizjudge.scoring.metrics import absolute_correlation
from vizjudge.scoring.rules import judge_candidate


def rank_candidates(
    frame: pd.DataFrame,
    candidates: list[ChartCandidate],
    top_k: int = 8,
    *,
    diversity: bool = True,
) -> list[ChartCandidate]:
    """Greedily select charts using explicit repetition and redundancy penalties."""
    validate_top_k(top_k)
    unique = {
        (candidate.kind, candidate.x, candidate.y, candidate.target): candidate
        for candidate in reversed(candidates)
    }
    judged = [judge_candidate(frame, candidate) for candidate in unique.values()]
    selected: list[ChartCandidate] = []
    similarity: dict[tuple[str, str], bool] = {}
    while judged and len(selected) < top_k:
        for item in judged:
            penalty = 0.0
            reasons: list[str] = []
            if diversity:
                columns = {item.x, item.y} - {None, item.target}
                for previous in selected:
                    other = {previous.x, previous.y} - {None, previous.target}
                    if columns & other:
                        penalty = max(penalty, 12.0)
                        reasons.append("A selected chart already covers this feature.")
                    if item.feature and previous.feature and item.feature != previous.feature:
                        pair = tuple(sorted((item.feature, previous.feature)))
                        if pair not in similarity:
                            left, right = frame[pair[0]], frame[pair[1]]
                            if pd.api.types.is_numeric_dtype(
                                left
                            ) and pd.api.types.is_numeric_dtype(right):
                                similarity[pair] = absolute_correlation(left, right) >= 0.95
                            else:
                                complete = left.notna() & right.notna()
                                similarity[pair] = bool(
                                    complete.sum() >= 30
                                    and (
                                        left[complete]
                                        .astype(str)
                                        .eq(right[complete].astype(str))
                                        .mean()
                                        >= 0.95
                                    )
                                )
                        if similarity[pair]:
                            penalty = max(penalty, 20.0)
                            reasons.append(f"Feature is redundant with {previous.feature}.")
                if any(previous.kind == item.kind for previous in selected):
                    penalty += 5.0
                    reasons.append("This chart family is already represented.")
            item.selection_penalty = penalty
            item.selection_score = round(max(0.0, item.score - penalty), 1)
            item.selection_reasons = list(dict.fromkeys(reasons))
        best = min(judged, key=lambda item: (-item.selection_score, -item.score, item.key))
        selected.append(best)
        judged.remove(best)
    return selected
