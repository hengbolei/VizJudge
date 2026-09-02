"""Modeling recommendations derived from the strongest chart evidence."""

from __future__ import annotations

from vizjudge.charts.generator import ChartCandidate


def recommend_models(
    target: str | None, roles: dict[str, str], ranked: list[ChartCandidate]
) -> list[str]:
    recommendations = [
        "Use a leakage-safe train/validation split before confirming any visual hypothesis."
    ]
    if target is None:
        recommendations.append(
            "Choose a target and task metric to turn unsupervised chart value "
            "into predictive evidence."
        )
        return recommendations

    if roles[target] == "categorical":
        recommendations.append(
            "Start with a regularized linear classifier and a tree-based baseline; "
            "compare calibrated metrics."
        )
    elif roles[target] == "numeric":
        recommendations.append(
            "Start with regularized linear regression and a tree-based regressor; "
            "compare residual behavior."
        )
    strong = [chart for chart in ranked if chart.target and chart.score >= 70]
    if strong:
        names = ", ".join(dict.fromkeys(chart.x for chart in strong))
        recommendations.append(
            "Validate the incremental out-of-sample value of the strongest visual "
            f"signals: {names}."
        )
        recommendations.append(
            "Review unusually strong target relationships for direct, temporal, or proxy leakage."
        )
    return recommendations[:6]
