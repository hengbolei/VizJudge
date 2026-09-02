"""Data-quality recommendations."""

from __future__ import annotations

from typing import Any


def recommend_data_quality(profile: dict[str, Any]) -> list[str]:
    recommendations: list[str] = []
    if profile["duplicate_rows"]:
        recommendations.append(
            f"Audit and deduplicate {profile['duplicate_rows']} duplicate rows "
            "before splitting data."
        )
    for column in profile["columns"]:
        rate = float(column["missing_rate"])
        if rate >= 0.4:
            recommendations.append(
                f"Review whether {column['name']} should be retained: "
                f"{rate:.0%} of values are missing."
            )
        elif rate >= 0.05:
            recommendations.append(
                f"Choose and validate an imputation strategy for {column['name']} "
                f"({rate:.0%} missing)."
            )
        if column["role"] == "constant":
            recommendations.append(f"Drop constant column {column['name']}.")
        if column["role"] == "identifier":
            recommendations.append(
                f"Exclude identifier-like column {column['name']} unless it supports "
                "grouping or joins."
            )
    return recommendations[:10]
