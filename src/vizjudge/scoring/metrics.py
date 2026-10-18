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


def _discretize(series: pd.Series, discrete: bool, bins: int) -> np.ndarray:
    """Integer codes: category levels, or quantile bins for continuous values."""
    if discrete:
        return pd.factorize(series, sort=True)[0]
    values = pd.to_numeric(series, errors="coerce").to_numpy(dtype=float)
    if np.unique(values).size <= bins:
        return np.unique(values, return_inverse=True)[1]
    # Heavy ties (e.g. mostly-zero amounts) share one bin; side="left" keeps values
    # above a tied edge in the next bin instead of merging them with the tie.
    edges = np.unique(np.quantile(values, np.linspace(0, 1, bins + 1)[1:-1]))
    return np.searchsorted(edges, values, side="left")


def information_coefficient(
    feature: pd.Series,
    target: pd.Series,
    *,
    feature_discrete: bool,
    target_discrete: bool,
    max_bins: int = 16,
) -> float:
    """Return a type-agnostic nonlinear association on the |Pearson r| scale.

    Mutual information is estimated from a contingency table (quantile bins for
    continuous values, levels for categories), reduced by the Miller-Madow bias
    term ``(kx - 1)(ky - 1) / 2n``, and mapped to Linfoot's informational
    coefficient ``sqrt(1 - exp(-2 MI))``. For a bivariate normal pair this equals
    |r|, so it is comparable with Pearson correlation, eta and Cramer's V, while also
    detecting non-monotone and heavy-tailed relationships.
    """
    pair = pd.concat([feature, target], axis=1).dropna()
    n = len(pair)
    if n < 30 or pair.iloc[:, 0].nunique() < 2 or pair.iloc[:, 1].nunique() < 2:
        return 0.0
    bins = int(min(max_bins, max(4, round(n ** (1 / 3)))))
    left = _discretize(pair.iloc[:, 0], feature_discrete, bins)
    right = _discretize(pair.iloc[:, 1], target_discrete, bins)
    table = np.zeros((left.max() + 1, right.max() + 1))
    np.add.at(table, (left, right), 1.0)
    table = table[table.sum(axis=1) > 0][:, table.sum(axis=0) > 0]
    joint = table / n
    expected = np.outer(joint.sum(axis=1), joint.sum(axis=0))
    nonzero = joint > 0
    mutual = float(np.sum(joint[nonzero] * np.log(joint[nonzero] / expected[nonzero])))
    mutual -= (table.shape[0] - 1) * (table.shape[1] - 1) / (2 * n)
    if mutual <= 0:
        return 0.0
    return min(1.0, math.sqrt(1.0 - math.exp(-2.0 * mutual)))


def one_to_one_agreement(left: pd.Series, right: pd.Series, *, max_levels: int = 100) -> float:
    """Share of complete rows consistent with a one-to-one level mapping.

    Detects re-encodings such as ``education`` (strings) vs ``education-num``
    (integers): each level of one column maps to a single level of the other, in both
    directions. Returns the weaker of the two directions; 1.0 means a bijection.
    Columns with more than ``max_levels`` levels (or more than a tenth of the rows)
    are not compared, because unique continuous values map one-to-one trivially.
    """
    pair = pd.concat([left, right], axis=1).dropna()
    limit = min(max_levels, len(pair) // 10)
    if len(pair) < 30 or max(pair.iloc[:, 0].nunique(), pair.iloc[:, 1].nunique()) > limit:
        return 0.0
    left_codes = pd.factorize(pair.iloc[:, 0])[0]
    right_codes = pd.factorize(pair.iloc[:, 1])[0]
    if left_codes.max() < 1 or right_codes.max() < 1:
        return 0.0
    table = np.zeros((left_codes.max() + 1, right_codes.max() + 1))
    np.add.at(table, (left_codes, right_codes), 1.0)
    forward = table.max(axis=1).sum() / len(pair)
    backward = table.max(axis=0).sum() / len(pair)
    return float(min(forward, backward))
