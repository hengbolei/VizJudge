"""Explainable heuristics for judging candidate chart value."""

from __future__ import annotations

import pandas as pd

from vizjudge.charts.generator import ChartCandidate
from vizjudge.scoring.metrics import (
    absolute_correlation,
    correlation_ratio,
    cramers_v,
    normalized_entropy,
)


def _coverage(frame: pd.DataFrame, columns: list[str]) -> float:
    return float(frame[columns].notna().all(axis=1).mean())


def _strength_label(value: float) -> str:
    if value >= 0.6:
        return "strong"
    if value >= 0.3:
        return "moderate"
    return "weak"


def judge_candidate(frame: pd.DataFrame, candidate: ChartCandidate) -> ChartCandidate:
    """Mutate a candidate with a 0-100 score and concise evidence."""
    columns = [candidate.x] + ([candidate.y] if candidate.y else [])
    coverage = _coverage(frame, columns)
    sample_count = int(frame[columns].dropna().shape[0])
    sample_factor = min(1.0, sample_count / 100)
    score = 10.0 + 25.0 * coverage + 10.0 * sample_factor
    candidate.reasons.append(f"{coverage:.0%} complete-case coverage across plotted columns")

    if candidate.kind == "histogram":
        values = pd.to_numeric(frame[candidate.x], errors="coerce").dropna()
        skew = float(values.skew()) if len(values) >= 3 else 0.0
        variability = 1.0 if values.nunique() >= 10 else values.nunique() / 10
        score += 20.0 * variability + 20.0 * min(abs(skew) / 2, 1.0)
        candidate.reasons.append(f"absolute skew is {abs(skew):.2f}")
        if abs(skew) >= 1:
            candidate.observations.append(
                f"{candidate.x} is strongly skewed; inspect transformations and robust scaling."
            )
        if coverage < 0.9:
            candidate.observations.append(
                f"{candidate.x} has meaningful missingness; test whether "
                "missingness is informative."
            )
    elif candidate.kind == "category_bar":
        entropy = normalized_entropy(frame[candidate.x])
        levels = int(frame[candidate.x].nunique(dropna=True))
        cardinality_penalty = max(0.0, min(20.0, (levels - 20) * 0.75))
        score += 35.0 * entropy - cardinality_penalty
        candidate.reasons.append(f"normalized category entropy is {entropy:.2f}")
        if levels > 20:
            candidate.observations.append(
                f"{candidate.x} has {levels} levels; group rare levels or use a "
                "high-cardinality encoder."
            )
    elif candidate.kind in {"scatter", "target_scatter"}:
        assert candidate.y is not None
        association = absolute_correlation(frame[candidate.x], frame[candidate.y])
        score += 55.0 * association
        label = _strength_label(association)
        candidate.reasons.append(f"{label} absolute Pearson correlation ({association:.2f})")
        candidate.observations.append(
            f"{candidate.x} and {candidate.y} show {label} linear association ({association:.2f})."
        )
    elif candidate.kind == "target_box":
        assert candidate.y is not None
        association = correlation_ratio(frame[candidate.x], frame[candidate.y])
        score += 55.0 * association
        label = _strength_label(association)
        candidate.reasons.append(f"{label} class separation by eta squared ({association:.2f})")
        candidate.observations.append(
            f"{candidate.y} has {label} separation across {candidate.x} "
            f"classes ({association:.2f})."
        )
    elif candidate.kind == "target_category":
        assert candidate.y is not None
        association = cramers_v(frame[candidate.x], frame[candidate.y])
        score += 55.0 * association
        label = _strength_label(association)
        candidate.reasons.append(
            f"{label} categorical association by Cramer's V ({association:.2f})"
        )
        candidate.observations.append(
            f"{candidate.x} has {label} association with {candidate.y} ({association:.2f})."
        )

    if sample_count < 30:
        score -= 20
        candidate.reasons.append("fewer than 30 complete observations; evidence is unstable")
    candidate.score = round(max(0.0, min(100.0, score)), 1)
    return candidate
