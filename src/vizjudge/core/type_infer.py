"""Infer analysis roles rather than relying only on storage dtypes."""

from __future__ import annotations

import math
from collections.abc import Mapping

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
    if uniqueness >= 0.98 and identifier_name:
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


VALID_ROLES = {
    ROLE_NUMERIC,
    ROLE_CATEGORICAL,
    ROLE_DATETIME,
    ROLE_IDENTIFIER,
    ROLE_CONSTANT,
    ROLE_TEXT,
}


def infer_types(
    frame: pd.DataFrame, role_overrides: dict[str, str] | None = None
) -> dict[str, str]:
    """Infer roles, then apply validated overrides without coercing input values."""
    from vizjudge.core.validation import validate_frame

    validate_frame(frame)
    if role_overrides is not None and not isinstance(role_overrides, Mapping):
        raise ValueError("role_overrides must map column names to roles")
    roles = {column: infer_series_type(frame[column]) for column in frame.columns}
    for name, role in (role_overrides or {}).items():
        if name not in roles:
            raise ValueError(f"Override column not found: {name}")
        if not isinstance(role, str) or role not in VALID_ROLES:
            raise ValueError(f"Invalid role for {name}: {role}")
        if role == ROLE_NUMERIC and not ptypes.is_numeric_dtype(frame[name]):
            raise ValueError(f"Numeric role requires numeric data: {name}")
        if role == ROLE_CONSTANT and frame[name].nunique() > 1:
            raise ValueError(f"Constant role requires at most one distinct value: {name}")
        roles[name] = role
    return roles


def resolve_task(
    frame: pd.DataFrame,
    roles: dict[str, str],
    target: str | None,
    task: str = "auto",
    role_overrides: dict[str, str] | None = None,
) -> str | None:
    """Resolve task independently of storage dtype and align the target role."""
    if task not in {"auto", "classification", "regression"}:
        raise ValueError("task must be auto, classification, or regression")
    if target is None:
        if task != "auto":
            raise ValueError("target is required for an explicit task")
        return None
    if not isinstance(target, str) or target not in frame.columns:
        raise ValueError(f"Target column not found: {target}")
    values = frame[target].dropna()
    if values.nunique() < 2:
        raise ValueError("Target must contain at least two distinct non-missing values")
    override = (role_overrides or {}).get(target)
    if override is not None and override not in {ROLE_NUMERIC, ROLE_CATEGORICAL}:
        raise ValueError("Target role must be numeric or categorical")
    if task == "auto":
        role = override or roles[target]
        if override is None and ptypes.is_numeric_dtype(values) and values.nunique() == 2:
            role = ROLE_CATEGORICAL
        if role not in {ROLE_NUMERIC, ROLE_CATEGORICAL}:
            raise ValueError("Cannot infer target task; specify task or target role explicitly")
        task = "regression" if role == ROLE_NUMERIC else "classification"
    required_role = ROLE_NUMERIC if task == "regression" else ROLE_CATEGORICAL
    if override is not None and override != required_role:
        raise ValueError("Target role override conflicts with task")
    if task == "regression" and not ptypes.is_numeric_dtype(values):
        raise ValueError("Regression target must contain numeric data")
    roles[target] = required_role
    return task
