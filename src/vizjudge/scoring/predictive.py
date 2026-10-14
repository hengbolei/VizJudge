"""Bounded, deterministic predictive checks with training-fold-only preprocessing.

These ridge baselines measure incremental value for IID splits; they are not a
substitute for grouped/temporal validation or model selection.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

MAX_ROWS = 5000
MAX_FEATURES = 20
MAX_LEVELS = 20


def validate_options(cv_folds: int, seed: int, baseline_features: list[str] | None) -> None:
    if isinstance(cv_folds, bool) or not isinstance(cv_folds, int) or not 2 <= cv_folds <= 10:
        raise ValueError("cv_folds must be an integer between 2 and 10")
    if isinstance(seed, bool) or not isinstance(seed, int) or not 0 <= seed < 2**32:
        raise ValueError("seed must be an integer between 0 and 2**32 - 1")
    if baseline_features is not None and (
        not isinstance(baseline_features, list)
        or any(not isinstance(name, str) for name in baseline_features)
        or len(set(baseline_features)) != len(baseline_features)
        or len(baseline_features) > MAX_FEATURES
    ):
        raise ValueError("baseline_features must be a list of at most 20 distinct column names")


def _design(
    train: pd.DataFrame, test: pd.DataFrame, features: list[str], roles: dict[str, str]
) -> tuple[np.ndarray, np.ndarray]:
    """Fit medians, scales and capped category vocabulary on train alone."""
    train_parts = [np.ones((len(train), 1))]
    test_parts = [np.ones((len(test), 1))]
    for name in features:
        if roles[name] == "numeric":
            left = train[name].to_numpy(dtype=float, na_value=np.nan)
            right = test[name].to_numpy(dtype=float, na_value=np.nan)
            median = float(np.nanmedian(left)) if np.isfinite(left).any() else 0.0
            left, right = np.nan_to_num(left, nan=median), np.nan_to_num(right, nan=median)
            mean, scale = float(left.mean()), float(left.std())
            scale = scale if scale > 0 else 1.0
            train_parts.append(((left - mean) / scale)[:, None])
            test_parts.append(((right - mean) / scale)[:, None])
            train_parts.append(train[name].isna().to_numpy(dtype=float)[:, None])
            test_parts.append(test[name].isna().to_numpy(dtype=float)[:, None])
        else:
            # One-hot category codes keep missingness distinct from literal strings.
            left = train[name]
            right = test[name]
            levels = left.dropna().value_counts().head(MAX_LEVELS).index
            for level in levels:
                train_parts.append(left.eq(level).fillna(False).to_numpy(dtype=float)[:, None])
                test_parts.append(right.eq(level).fillna(False).to_numpy(dtype=float)[:, None])
            train_parts.append(left.isna().to_numpy(dtype=float)[:, None])
            test_parts.append(right.isna().to_numpy(dtype=float)[:, None])
    return np.concatenate(train_parts, axis=1), np.concatenate(test_parts, axis=1)


def _predict(design: np.ndarray, test: np.ndarray, response: np.ndarray) -> np.ndarray:
    penalty = np.eye(design.shape[1])
    penalty[0, 0] = 0.0
    weights = np.linalg.solve(design.T @ design + penalty, design.T @ response)
    return test @ weights


def _score(actual: np.ndarray, prediction: np.ndarray, task: str) -> float:
    if task == "regression":
        return float(np.sqrt(np.mean((actual - prediction) ** 2)))
    predicted = prediction.argmax(axis=1)
    return float(
        np.mean([np.mean(predicted[actual == label] == label) for label in np.unique(actual)])
    )


def validate_predictive_value(
    frame: pd.DataFrame,
    roles: dict[str, str],
    target: str | None,
    task: str | None,
    features: list[str],
    *,
    enabled: bool = True,
    cv_folds: int = 5,
    seed: int = 42,
    baseline_features: list[str] | None = None,
) -> dict[str, Any]:
    """Compare baseline and baseline+feature on identical held-out folds."""
    validate_options(cv_folds, seed, baseline_features)
    baseline = list(baseline_features or [])
    for name in baseline:
        if name == target or roles.get(name) not in {"numeric", "categorical"}:
            raise ValueError(f"Baseline column must be an eligible non-target feature: {name}")
    if baseline and target is None:
        raise ValueError("target is required for baseline_features")
    result: dict[str, Any] = {
        "status": "skipped",
        "reason": None,
        "method": "ridge_baseline",
        "split": "shuffled_stratified" if task == "classification" else "shuffled_kfold",
        "cv_folds": cv_folds,
        "seed": seed,
        "baseline_features": baseline,
        "max_rows": MAX_ROWS,
        "max_features": MAX_FEATURES,
        "row_count": 0,
        "excluded_target_rows": 0,
        "sampled": False,
        "metric": "balanced_accuracy" if task == "classification" else "rmse",
        "features": {},
        "warnings": [
            "IID folds do not protect against temporal, group, duplicate, or semantic leakage.",
            "Fold variation is descriptive, not a confidence interval or significance test.",
        ],
    }
    if not enabled or target is None:
        result["reason"] = "disabled" if not enabled else "no_target"
        return result
    subset = frame.loc[frame[target].notna()].reset_index(drop=True)
    result["excluded_target_rows"] = len(frame) - len(subset)
    if len(subset) > MAX_ROWS:
        subset = subset.sample(MAX_ROWS, random_state=seed).reset_index(drop=True)
        result["sampled"] = True
        result["warnings"].append(f"Predictive validation uses a seeded sample of {MAX_ROWS} rows.")
    result["row_count"] = len(subset)
    if len(subset) < max(30, cv_folds * 2):
        result["reason"] = "insufficient_rows"
        return result
    rng = np.random.default_rng(seed)
    if task == "classification":
        labels, levels = pd.factorize(subset[target], sort=False)
        counts = np.bincount(labels)
        if len(levels) < 2 or len(levels) > MAX_LEVELS or counts.min() < cv_folds:
            result["reason"] = "insufficient_class_support_or_too_many_classes"
            return result
        parts: list[list[np.ndarray]] = [[] for _ in range(cv_folds)]
        for label in range(len(levels)):
            for index, chunk in enumerate(
                np.array_split(rng.permutation(np.flatnonzero(labels == label)), cv_folds)
            ):
                parts[index].append(chunk)
        folds = [np.concatenate(part) for part in parts]
        response = np.eye(len(levels))[labels]
        actual = labels
    else:
        actual = subset[target].to_numpy(dtype=float)
        response = actual
        folds = list(np.array_split(rng.permutation(len(subset)), cv_folds))
    selected = list(dict.fromkeys(features))[:MAX_FEATURES]
    if len(features) > MAX_FEATURES:
        result["warnings"].append(f"Only the first {MAX_FEATURES} ranked features were validated.")
    if not selected:
        result["reason"] = "no_eligible_features"
        return result
    baseline_scores: list[float] = []
    feature_scores: dict[str, list[float]] = {name: [] for name in selected}
    for test_indices in folds:
        train_mask = np.ones(len(subset), dtype=bool)
        train_mask[test_indices] = False
        train_indices = np.flatnonzero(train_mask)
        train, test = subset.iloc[train_indices], subset.iloc[test_indices]
        design, held_out = _design(train, test, baseline, roles)
        base_score = _score(
            actual[test_indices], _predict(design, held_out, response[train_indices]), task
        )
        baseline_scores.append(base_score)
        for name in selected:
            if name in baseline:
                feature_scores[name].append(base_score)
                continue
            design, held_out = _design(train, test, [*baseline, name], roles)
            feature_scores[name].append(
                _score(
                    actual[test_indices], _predict(design, held_out, response[train_indices]), task
                )
            )
    result["status"] = "completed"
    result["baseline_score"] = round(float(np.mean(baseline_scores)), 6)
    for name, scores in feature_scores.items():
        gains = np.array(scores) - np.array(baseline_scores)
        if task == "regression":
            gains = -gains  # Positive gain always means improvement.
        result["features"][name] = {
            "status": "completed",
            "metric": result["metric"],
            "baseline_score": result["baseline_score"],
            "feature_score": round(float(np.mean(scores)), 6),
            "incremental_gain": round(float(gains.mean()), 6),
            "gain_std": round(float(gains.std(ddof=1)), 6),
            "positive_folds": int((gains > 0).sum()),
            "fold_gains": [round(float(gain), 6) for gain in gains],
            "in_baseline": name in baseline,
        }
    return result
