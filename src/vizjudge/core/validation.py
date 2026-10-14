"""Shared validation for file and in-memory analysis entry points."""

import numpy as np
import pandas as pd
from pandas.api import types as ptypes


def validate_frame(frame: pd.DataFrame) -> None:
    if not isinstance(frame, pd.DataFrame):
        raise ValueError("Dataset must be a pandas DataFrame")
    if frame.empty:
        raise ValueError("Dataset is empty")
    if frame.columns.has_duplicates:
        raise ValueError("Dataset contains duplicate column names")
    if any(not isinstance(name, str) or not name.strip() for name in frame.columns):
        raise ValueError("Column names must be non-empty strings")
    for name, series in frame.items():
        if not all(ptypes.is_scalar(value) for value in series):
            raise ValueError(f"Column must contain scalar values: {name}")
        if ptypes.is_complex_dtype(series):
            raise ValueError(f"Complex values are not supported: {name}")
        if ptypes.is_numeric_dtype(series) and np.isinf(series.dropna()).any():
            raise ValueError(f"Infinite values are not supported: {name}")
        try:
            series.nunique()
        except TypeError as exc:
            raise ValueError(f"Column must contain scalar values: {name}") from exc


def validate_top_k(top_k: int) -> None:
    if isinstance(top_k, bool) or not isinstance(top_k, int) or top_k < 1:
        raise ValueError("top_k must be a positive integer")
