"""Explainable heuristics for judging candidate chart value."""

from __future__ import annotations

import math

import pandas as pd

from vizjudge.charts.generator import ChartCandidate
from vizjudge.scoring.metrics import (
    absolute_correlation,
    correlation_ratio,
    cramers_v,
    information_coefficient,
    normalized_entropy,
)

# Targeted charts score 55 points of relevance. Relevance blends the chart's linear
# association (|r|, eta = sqrt(eta squared), or Cramer's V; all on the |r| scale) with
# the type-agnostic information coefficient.
RELEVANCE_POINTS = 55.0
LINEAR_WEIGHT = 0.5
# Held-out validation moves relevance toward the cross-validated strength.
PREDICTIVE_WEIGHT = 0.5
PREDICTIVE_CAP = 15.0
# Distribution charts earn up to this many points for an actionable shape (skew, rare
# levels). With a target they carry no target evidence, so a featureless distribution
# scores like a targeted chart with zero relevance instead of outranking real signal.
DISTRIBUTION_POINTS = 20.0
# Any association or missingness association at this level is reviewed as possible
# leakage. Held-out strength needs more: Somers' D 0.95 is ROC-AUC 0.975, which clean,
# strong features reach (worst_perimeter in breast cancer is 0.946); 0.98 is AUC 0.99.
LEAKAGE_THRESHOLD = 0.95
HELD_OUT_LEAKAGE_THRESHOLD = 0.98
LEAKAGE_PREFIX = "Near-perfect target signal"


def _coverage(frame: pd.DataFrame, columns: list[str]) -> float:
    return float(frame[columns].notna().all(axis=1).mean())


def _strength_label(value: float) -> str:
    if value >= 0.6:
        return "strong"
    if value >= 0.3:
        return "moderate"
    return "weak"


def _relevance(
    frame: pd.DataFrame,
    candidate: ChartCandidate,
    linear: float,
    *,
    feature_discrete: bool,
    target_discrete: bool,
) -> float:
    """Blend linear association with the information coefficient for targeted charts."""
    if not candidate.target or candidate.feature is None:
        candidate.metrics["relevance"] = round(linear, 6)
        return linear
    information = information_coefficient(
        frame[candidate.feature],
        frame[candidate.target],
        feature_discrete=feature_discrete,
        target_discrete=target_discrete,
    )
    relevance = LINEAR_WEIGHT * linear + (1.0 - LINEAR_WEIGHT) * information
    candidate.metrics.update(
        information_coefficient=round(information, 6), relevance=round(relevance, 6)
    )
    candidate.reasons.append(
        f"{_strength_label(information)} nonlinear dependence by information coefficient "
        f"({information:.2f}); blended relevance {relevance:.2f}"
    )
    if information >= linear + 0.15:
        candidate.observations.append(
            f"{candidate.feature} carries more nonlinear than linear signal about "
            f"{candidate.target} ({information:.2f} vs {linear:.2f}); consider binning, "
            "splines or tree models."
        )
    return relevance


def _missingness_association(frame: pd.DataFrame, candidate: ChartCandidate) -> float:
    """Association between the feature's missingness and the target, on the |r| scale.

    Complete-case metrics never see this: a column filled only after the outcome is
    missing for one class, so eta and the information coefficient see a single group.
    """
    assert candidate.feature is not None and candidate.target is not None
    rows = frame[candidate.target].notna()
    missing = frame.loc[rows, candidate.feature].isna()
    if missing.nunique() < 2:
        return 0.0
    target = frame.loc[rows, candidate.target]
    if candidate.kind in {"target_box", "target_category"}:
        return cramers_v(missing, target)
    return math.sqrt(correlation_ratio(missing, target))


def _leakage_signal(candidate: ChartCandidate) -> tuple[str, float] | None:
    """Return the strongest near-perfect target signal as (metric, value), if any."""
    names = (
        "absolute_correlation",
        "eta",
        "cramers_v",
        "information_coefficient",
        "missingness_association",
        "held_out_strength",
    )
    flagged = [
        (name, float(candidate.metrics[name]))
        for name in names
        if name in candidate.metrics
        and float(candidate.metrics[name])
        >= (HELD_OUT_LEAKAGE_THRESHOLD if name == "held_out_strength" else LEAKAGE_THRESHOLD)
    ]
    return max(flagged, key=lambda item: item[1], default=None)


def _rare_level_share(series: pd.Series) -> float:
    counts = series.dropna().value_counts()
    return float((counts < 5).mean()) if len(counts) else 0.0


def _strength(metric: str, gain: float, score: float) -> float:
    """Map a held-out gain onto the |r| scale used by relevance.

    ROC-AUC gains become Somers' D (``2 * gain``). RMSE gains become a signed
    ``sqrt(1 - (rmse_with / rmse_without) ** 2)``, i.e. a held-out partial correlation;
    ``score`` is the RMSE with the feature, so the RMSE without it is ``score + gain``.
    """
    if metric == "roc_auc":
        return max(-1.0, min(1.0, 2.0 * gain))
    ratio = score / max(score + gain, 1e-12)
    change = 1.0 - ratio**2
    return max(-1.0, min(1.0, math.copysign(math.sqrt(abs(change)), change)))


def held_out_strength(predictive: dict[str, object]) -> float:
    """Marginal held-out strength, averaged with the conditional (forward-selection)
    strength when available."""
    metric = str(predictive["metric"])
    strength = _strength(
        metric, float(predictive["incremental_gain"]), float(predictive["feature_score"])
    )
    if "conditional_gain" in predictive:
        conditional = _strength(
            metric,
            float(predictive["conditional_gain"]),
            float(predictive["conditional_score"]),
        )
        strength = 0.5 * (strength + conditional)
    return strength


def judge_candidate(
    frame: pd.DataFrame, candidate: ChartCandidate, *, supervised: bool = False
) -> ChartCandidate:
    """Mutate a candidate with a 0-100 score and concise evidence.

    ``supervised`` marks an analysis with a target: distribution charts then score
    only their actionable shape, so they cannot outrank target evidence by default.
    """
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
        shape = min(abs(skew) / 2, 1.0)
        if supervised:
            score += DISTRIBUTION_POINTS * variability * shape
        else:
            score += 20.0 * variability + 20.0 * shape
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
        if supervised:
            # Rare levels and high cardinality call for an encoding decision.
            rare = _rare_level_share(frame[candidate.x])
            candidate.metrics["rare_level_share"] = round(rare, 6)
            interest = max(rare, min(1.0, max(0, levels - 20) / 30))
            score += DISTRIBUTION_POINTS * interest
            candidate.reasons.append(f"{rare:.0%} of levels have fewer than 5 rows")
        else:
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
        label = _strength_label(association)
        candidate.reasons.append(f"{label} absolute Pearson correlation ({association:.2f})")
        candidate.observations.append(
            f"{candidate.x} and {candidate.y} show {label} linear association ({association:.2f})."
        )
        relevance = _relevance(
            frame, candidate, association, feature_discrete=False, target_discrete=False
        )
    elif candidate.kind in {"target_box", "regression_box"}:
        assert candidate.y is not None
        eta_squared = correlation_ratio(frame[candidate.x], frame[candidate.y])
        # eta (not eta squared) is on the same scale as |r| and Cramer's V.
        association = math.sqrt(eta_squared)
        candidate.metrics.update(eta_squared=round(eta_squared, 6), eta=round(association, 6))
        groups = frame[columns].dropna().groupby(candidate.x, observed=True).size()
        if len(groups) and groups.min() < 5:
            # Scaled by the share of sparse groups: one rare level should not sink a feature.
            score -= 15 * float((groups < 5).mean())
            candidate.warnings.append(
                "Some groups have fewer than 5 observations; separation may overfit."
            )
        if len(groups) > 15:
            score -= 10
            candidate.warnings.append("Rendered boxes show only the 15 most frequent groups.")
        label = _strength_label(association)
        candidate.reasons.append(
            f"{label} group separation by eta ({association:.2f}; eta squared {eta_squared:.2f})"
        )
        candidate.observations.append(
            f"{candidate.y} has {label} separation across {candidate.x} groups ({association:.2f})."
        )
        # target_box: x is the class target; regression_box: x is the categorical feature.
        relevance = _relevance(
            frame,
            candidate,
            association,
            feature_discrete=candidate.kind == "regression_box",
            target_discrete=candidate.kind == "target_box",
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
            # Scaled by the share of sparse groups: one rare level should not sink a feature.
            score -= 15 * float((groups < 5).mean())
            candidate.warnings.append(
                "Some categories have fewer than 5 observations; signal may overfit."
            )
        if len(groups) > 20:
            candidate.warnings.append(
                "Rendered category relationships show only 20 frequent levels."
            )
        label = _strength_label(association)
        candidate.reasons.append(
            f"{label} categorical association by Cramer's V ({association:.2f})"
        )
        candidate.observations.append(
            f"{candidate.x} has {label} association with {candidate.y} ({association:.2f})."
        )
        if levels > 100:
            relevance = 0.0
            candidate.metrics["relevance"] = 0.0
        else:
            relevance = _relevance(
                frame, candidate, association, feature_discrete=True, target_discrete=True
            )
    else:
        raise ValueError(f"Unsupported chart kind: {candidate.kind}")
    if candidate.kind not in {"histogram", "category_bar"}:
        if coverage < 1.0:
            missingness = _missingness_association(frame, candidate)
            candidate.metrics["missingness_association"] = round(missingness, 6)
            if missingness >= 0.3:
                candidate.observations.append(
                    f"Whether {candidate.feature} is missing is associated with "
                    f"{candidate.target} ({missingness:.2f}); missingness is informative, "
                    "or the column is only filled after the outcome."
                )
            if missingness > relevance:
                relevance = missingness
                candidate.metrics["relevance"] = round(relevance, 6)
                candidate.reasons.append(
                    f"relevance taken from missingness association ({missingness:.2f})"
                )
        score += RELEVANCE_POINTS * relevance

    if sample_count < 30:
        score -= 20
        candidate.reasons.append("fewer than 30 complete observations; evidence is unstable")
        candidate.warnings.append("Fewer than 30 complete observations; evidence is unstable.")
    candidate.metrics["heuristic_score"] = round(max(0.0, min(100.0, score)), 1)
    if (
        candidate.predictive
        and candidate.predictive.get("status") == "completed"
        and not candidate.predictive.get("in_baseline")
    ):
        strength = held_out_strength(candidate.predictive)
        bonus = PREDICTIVE_WEIGHT * RELEVANCE_POINTS * (strength - candidate.metrics["relevance"])
        bonus = max(-PREDICTIVE_CAP, min(PREDICTIVE_CAP, bonus))
        score += bonus
        candidate.metrics.update(
            held_out_strength=round(strength, 6), predictive_adjustment=round(bonus, 3)
        )
        if "conditional_gain" in candidate.predictive:
            candidate.metrics["conditional_gain"] = candidate.predictive["conditional_gain"]
        candidate.reasons.append(
            f"Held-out {candidate.predictive['metric']} gain over baseline: "
            f"{candidate.predictive['incremental_gain']:.4f} (positive is better); "
            f"held-out strength {strength:.2f} vs in-sample relevance "
            f"{candidate.metrics['relevance']:.2f}."
        )
    leakage = _leakage_signal(candidate) if candidate.target else None
    if leakage:
        candidate.metrics["leakage_suspect"] = 1
        candidate.warnings.append(
            f"{LEAKAGE_PREFIX} ({leakage[0]} {leakage[1]:.2f}); review direct, "
            "proxy or post-outcome leakage before using this feature."
        )
    candidate.score = round(max(0.0, min(100.0, score)), 1)
    return candidate
