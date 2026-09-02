"""Feature-engineering recommendations derived from profiles and charts."""

from __future__ import annotations

from typing import Any

from vizjudge.charts.generator import ChartCandidate


def recommend_features(profile: dict[str, Any], ranked: list[ChartCandidate]) -> list[str]:
    recommendations: list[str] = []
    by_name = {column["name"]: column for column in profile["columns"]}
    for chart in ranked:
        if chart.kind == "histogram":
            skew = abs(float(by_name[chart.x].get("statistics", {}).get("skew") or 0.0))
            if skew >= 1:
                recommendations.append(
                    f"Compare raw and monotonic transforms of {chart.x}; "
                    "validate inside cross-validation."
                )
        elif chart.kind == "category_bar" and by_name[chart.x]["unique_count"] > 20:
            recommendations.append(
                f"Group rare {chart.x} levels or compare hashing/target encoding without leakage."
            )
    return list(dict.fromkeys(recommendations))[:8]
