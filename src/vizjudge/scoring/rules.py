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
    candidate.reasons.clear()
    candidate.observations.clear()
    candidate.warnings.clear()
    candidate.metrics.clear()
    if candidate.target:
        candidate.feature = candidate.y if candidate.x == candidate.target else candidate.x
    columns = [candidate.x] + ([candidate.y] if candidate.y else [])
    coverage = _coverage(frame, columns)
    sample_count = int(frame[columns].dropna().shape[0])
    sample_factor = min(1.0, sample_count / 100)
    score = 10.0 + 25.0 * coverage + 10.0 * sample_factor
    candidate.metrics.update(coverage=round(coverage, 6), sample_count=sample_count)
    candidate.reasons.append(f"{coverage:.0%} complete-case coverage across plotted columns")

    if candidate.kind == "histogram":
        values = pd.to_numeric(frame[candidate.x], errors="coerce").dropna()
        skew = float(values.skew()) if len(values) >= 3 else 0.0
        skew = 0.0 if pd.isna(skew) else skew
        candidate.metrics["skew"] = round(skew, 6)
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
        candidate.metrics.update(entropy=round(entropy, 6), levels=levels)
        cardinality_penalty = max(0.0, min(20.0, (levels - 20) * 0.75))
        score += 35.0 * entropy - cardinality_penalty
        candidate.reasons.append(f"normalized category entropy is {entropy:.2f}")
        if levels > 20:
            candidate.warnings.append(
                "Rendered category bars show only the 20 most frequent levels."
            )
            candidate.observations.append(
                f"{candidate.x} has {levels} levels; group rare levels or use a "
                "high-cardinality encoder."
            )
    elif candidate.kind in {"scatter", "target_scatter"}:
        assert candidate.y is not None
        association = absolute_correlation(frame[candidate.x], frame[candidate.y])
        candidate.metrics["absolute_correlation"] = round(association, 6)
        score += 55.0 * association
        label = _strength_label(association)
        candidate.reasons.append(f"{label} absolute Pearson correlation ({association:.2f})")
        candidate.observations.append(
            f"{candidate.x} and {candidate.y} show {label} linear association ({association:.2f})."
        )
    elif candidate.kind in {"target_box", "regression_box"}:
        assert candidate.y is not None
        association = correlation_ratio(frame[candidate.x], frame[candidate.y])
        candidate.metrics["eta_squared"] = round(association, 6)
        groups = frame[columns].dropna().groupby(candidate.x, observed=True).size()
        if len(groups) and groups.min() < 5:
            score -= 15
            candidate.warnings.append(
                "Some groups have fewer than 5 observations; separation may overfit."
            )
        if len(groups) > 15:
            score -= 10
            candidate.warnings.append("Rendered boxes show only the 15 most frequent groups.")
        score += 55.0 * association
        label = _strength_label(association)
        candidate.reasons.append(f"{label} group separation by eta squared ({association:.2f})")
        candidate.observations.append(
            f"{candidate.y} has {label} separation across {candidate.x} groups ({association:.2f})."
        )
    elif candidate.kind == "target_category":
        assert candidate.y is not None
        if frame[candidate.y].nunique() > 20:
            candidate.warnings.append("Rendered class legend groups infrequent classes into Other.")
        levels = max(frame[candidate.x].nunique(), frame[candidate.y].nunique())
        if levels > 100:
            association = 0.0
            score -= 15
            candidate.warnings.append("Categorical association skipped: more than 100 levels.")
        else:
            association = cramers_v(frame[candidate.x], frame[candidate.y])
        candidate.metrics["cramers_v"] = round(association, 6)
        groups = frame[columns].dropna().groupby(candidate.x, observed=True).size()
        if len(groups) and groups.min() < 5:
            score -= 15
            candidate.warnings.append(
                "Some categories have fewer than 5 observations; signal may overfit."
            )
        if len(groups) > 20:
            candidate.warnings.append(
                "Rendered category relationships show only 20 frequent levels."
            )
        score += 55.0 * association
        label = _strength_label(association)
        candidate.reasons.append(
            f"{label} categorical association by Cramer's V ({association:.2f})"
        )
        candidate.observations.append(
            f"{candidate.x} has {label} association with {candidate.y} ({association:.2f})."
        )
    else:
        raise ValueError(f"Unsupported chart kind: {candidate.kind}")

    if sample_count < 30:
        score -= 20
        candidate.reasons.append("fewer than 30 complete observations; evidence is unstable")
        candidate.warnings.append("Fewer than 30 complete observations; evidence is unstable.")
    if (
        candidate.target
        and candidate.metrics.get(
            "absolute_correlation",
            candidate.metrics.get("eta_squared", candidate.metrics.get("cramers_v", 0)),
        )
        >= 0.95
    ):
        candidate.warnings.append(
            "Near-perfect target association; review direct or proxy leakage."
        )
    candidate.metrics["heuristic_score"] = round(max(0.0, min(100.0, score)), 1)
    if candidate.predictive and candidate.predictive.get("status") == "completed":
        gain = float(candidate.predictive["incremental_gain"])
        if candidate.predictive["metric"] == "rmse":
            gain /= max(float(candidate.predictive["baseline_score"]), 1e-12)
        bonus = max(-10.0, min(15.0, 30.0 * gain))
        score += bonus
        candidate.metrics["predictive_adjustment"] = round(bonus, 3)
        candidate.reasons.append(
            f"Held-out {candidate.predictive['metric']} gain over baseline: "
            f"{candidate.predictive['incremental_gain']:.4f} (positive is better)."
        )
    candidate.score = round(max(0.0, min(100.0, score)), 1)
    return candidate
