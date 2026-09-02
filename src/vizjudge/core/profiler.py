"""Compact, JSON-safe dataset profiling."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from vizjudge.core.type_infer import ROLE_NUMERIC, infer_types


def _scalar(value: Any) -> Any:
    if pd.isna(value):
        return None
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, (pd.Timestamp, pd.Timedelta)):
        return str(value)
    return value


def profile_dataframe(frame: pd.DataFrame, roles: dict[str, str] | None = None) -> dict[str, Any]:
    """Return dataset- and column-level diagnostics without exposing row values."""
    roles = roles or infer_types(frame)
    columns: list[dict[str, Any]] = []
    row_count = len(frame)

    for name in frame.columns:
        series = frame[name]
        non_null = series.dropna()
        item: dict[str, Any] = {
            "name": str(name),
            "dtype": str(series.dtype),
            "role": roles[str(name)],
            "missing_count": int(series.isna().sum()),
            "missing_rate": round(float(series.isna().mean()), 6),
            "unique_count": int(non_null.nunique(dropna=True)),
        }
        if roles[str(name)] == ROLE_NUMERIC and len(non_null):
            numeric = pd.to_numeric(non_null, errors="coerce").dropna()
            item["statistics"] = {
                "min": _scalar(numeric.min()),
                "max": _scalar(numeric.max()),
                "mean": _scalar(numeric.mean()),
                "median": _scalar(numeric.median()),
                "std": _scalar(numeric.std()),
                "skew": _scalar(numeric.skew()),
            }
        else:
            counts = non_null.astype(str).value_counts().head(5)
            item["top_values"] = [
                {"value": str(value), "count": int(count)} for value, count in counts.items()
            ]
        columns.append(item)

    return {
        "row_count": int(row_count),
        "column_count": int(frame.shape[1]),
        "duplicate_rows": int(frame.duplicated().sum()),
        "total_missing_rate": round(float(frame.isna().mean().mean()), 6),
        "columns": columns,
    }
