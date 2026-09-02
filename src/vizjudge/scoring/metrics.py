"""Small dependency-light association metrics used by scoring rules."""

from __future__ import annotations

import math

import numpy as np
import pandas as pd


def normalized_entropy(series: pd.Series) -> float:
    """Return Shannon entropy normalized to [0, 1]."""
    counts = series.dropna().value_counts().to_numpy(dtype=float)
    if len(counts) <= 1:
        return 0.0
    probabilities = counts / counts.sum()
    entropy = -float(np.sum(probabilities * np.log(probabilities)))
    return entropy / math.log(len(counts))


def absolute_correlation(left: pd.Series, right: pd.Series) -> float:
    pair = pd.concat([left, right], axis=1).dropna()
    if len(pair) < 3 or pair.iloc[:, 0].nunique() < 2 or pair.iloc[:, 1].nunique() < 2:
        return 0.0
    value = pair.iloc[:, 0].corr(pair.iloc[:, 1])
    return 0.0 if pd.isna(value) else abs(float(value))


def correlation_ratio(categories: pd.Series, values: pd.Series) -> float:
    """Return eta squared for categorical-to-numeric association."""
    pair = pd.concat([categories, values], axis=1).dropna()
    if len(pair) < 3 or pair.iloc[:, 0].nunique() < 2:
        return 0.0
    numeric = pd.to_numeric(pair.iloc[:, 1], errors="coerce")
    pair = pair.loc[numeric.notna()].copy()
    pair.iloc[:, 1] = numeric.dropna().to_numpy()
    overall = float(pair.iloc[:, 1].mean())
    total = float(((pair.iloc[:, 1] - overall) ** 2).sum())
    if total <= 0:
        return 0.0
    between = sum(
        len(group) * (float(group.iloc[:, 1].mean()) - overall) ** 2
        for _, group in pair.groupby(pair.columns[0], observed=True)
    )
    return max(0.0, min(1.0, float(between / total)))


def cramers_v(left: pd.Series, right: pd.Series) -> float:
    """Return bias-corrected Cramer's V without requiring SciPy."""
    table = pd.crosstab(left, right)
    if table.empty or min(table.shape) < 2:
        return 0.0
    observed = table.to_numpy(dtype=float)
    n = observed.sum()
    if n <= 1:
        return 0.0
    expected = np.outer(observed.sum(axis=1), observed.sum(axis=0)) / n
    valid = expected > 0
    chi2 = float(np.sum(((observed - expected) ** 2 / np.where(valid, expected, 1))[valid]))
    phi2 = chi2 / n
    rows, columns = observed.shape
    corrected = max(0.0, phi2 - ((columns - 1) * (rows - 1)) / (n - 1))
    corrected_rows = rows - ((rows - 1) ** 2) / (n - 1)
    corrected_columns = columns - ((columns - 1) ** 2) / (n - 1)
    denominator = min(corrected_columns - 1, corrected_rows - 1)
    return 0.0 if denominator <= 0 else min(1.0, math.sqrt(corrected / denominator))
