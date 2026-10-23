"""Evaluate and order chart candidates."""

from __future__ import annotations

import pandas as pd

from vizjudge.charts.generator import ChartCandidate
from vizjudge.core.validation import validate_top_k
from vizjudge.scoring.metrics import absolute_correlation, one_to_one_agreement
from vizjudge.scoring.rules import judge_candidate


def _redundancy(left: pd.Series, right: pd.Series) -> str | None:
    """Name the redundancy between two features, or return None."""
    if pd.api.types.is_numeric_dtype(left) and pd.api.types.is_numeric_dtype(right):
        if absolute_correlation(left, right) >= 0.95:
            return "absolute correlation at least 0.95"
    else:
        complete = left.notna() & right.notna()
        if (
            complete.sum() >= 30
            and left[complete].astype(str).eq(right[complete].astype(str)).mean() >= 0.95
        ):
            return "values match in at least 95% of rows"
    # Re-encodings such as education (strings) vs education-num (integers).
    if one_to_one_agreement(left, right) >= 0.99:
        return "one-to-one level mapping"
    return None


def rank_candidates(
    frame: pd.DataFrame,
    candidates: list[ChartCandidate],
    top_k: int = 8,
    *,
    diversity: bool = True,
) -> list[ChartCandidate]:
    """Greedily select charts using explicit repetition and redundancy penalties.

    Leakage suspects are selected first, whatever their score: the host must see them
    to decide whether the column exists at prediction time.
    """
    validate_top_k(top_k)
    unique = {
        (candidate.kind, candidate.x, candidate.y, candidate.target): candidate
        for candidate in reversed(candidates)
    }
    supervised = any(candidate.target for candidate in unique.values())
    judged = [
        judge_candidate(frame, candidate, supervised=supervised) for candidate in unique.values()
    ]
    selected: list[ChartCandidate] = []
    similarity: dict[tuple[str, str], str | None] = {}

    def redundancy(item: ChartCandidate, previous: ChartCandidate) -> str | None:
        if not item.feature or not previous.feature or item.feature == previous.feature:
            return None
        pair = tuple(sorted((item.feature, previous.feature)))
        if pair not in similarity:
            similarity[pair] = _redundancy(frame[pair[0]], frame[pair[1]])
        return similarity[pair]

    suspects = sorted(
        (item for item in judged if item.metrics.get("leakage_suspect")),
        key=lambda item: (-item.score, item.key),
    )
    for item in suspects:
        if len(selected) >= top_k:
            break
        # One chart per leaking signal; redundant copies compete in the ranking below.
        if any(
            item.feature == previous.feature or redundancy(item, previous) for previous in selected
        ):
            continue
        item.selection_penalty = 0.0
        item.selection_score = item.score
        item.selection_reasons = ["Surfaced ahead of the ranking for leakage review."]
        selected.append(item)
        judged.remove(item)
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
                    reason = redundancy(item, previous)
                    if reason:
                        penalty = max(penalty, 20.0)
                        reasons.append(f"Feature is redundant with {previous.feature} ({reason}).")
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
