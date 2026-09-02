"""Infer analysis roles rather than relying only on storage dtypes."""

from __future__ import annotations

import math

import pandas as pd
from pandas.api import types as ptypes

ROLE_NUMERIC = "numeric"
ROLE_CATEGORICAL = "categorical"
ROLE_DATETIME = "datetime"
ROLE_IDENTIFIER = "identifier"
ROLE_CONSTANT = "constant"
ROLE_TEXT = "text"


def infer_series_type(series: pd.Series) -> str:
    """Infer a practical visualization role for one series."""
    values = series.dropna()
    if values.nunique(dropna=True) <= 1:
        return ROLE_CONSTANT

    count = len(values)
    unique = values.nunique(dropna=True)
    uniqueness = unique / max(count, 1)
    name = str(series.name or "").lower()

    if ptypes.is_datetime64_any_dtype(series):
        return ROLE_DATETIME
    if ptypes.is_bool_dtype(series):
        return ROLE_CATEGORICAL

    identifier_name = name == "id" or name.endswith("_id") or name.startswith("id_")
    if uniqueness >= 0.98 and (identifier_name or ptypes.is_integer_dtype(series)):
        return ROLE_IDENTIFIER

    if ptypes.is_numeric_dtype(series):
        categorical_limit = max(10, min(30, int(math.sqrt(max(count, 1)))))
        if unique <= categorical_limit and not ptypes.is_float_dtype(series):
            return ROLE_CATEGORICAL
        return ROLE_NUMERIC

    if ptypes.is_object_dtype(series) or isinstance(series.dtype, pd.StringDtype):
        sample = values.astype(str).head(200)
        parsed = pd.to_datetime(sample, errors="coerce", format="mixed")
        if len(sample) >= 3 and parsed.notna().mean() >= 0.9:
            return ROLE_DATETIME
        mean_length = float(sample.str.len().mean()) if len(sample) else 0.0
        if uniqueness > 0.8 and mean_length > 24:
            return ROLE_TEXT
        return ROLE_CATEGORICAL

    return ROLE_CATEGORICAL


def infer_types(frame: pd.DataFrame) -> dict[str, str]:
    """Infer a visualization role for every column."""
    return {str(column): infer_series_type(frame[column]) for column in frame.columns}
