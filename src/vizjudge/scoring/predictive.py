"""Bounded, deterministic predictive checks with training-fold-only preprocessing.

These ridge baselines measure incremental value for IID splits; they are not a
substitute for grouped/temporal validation or model selection. Numeric features are
winsorized and expanded into quantile-bin indicators fit on the training fold, so the
additive ridge model captures nonlinear, tree-like step effects and is not dominated
by outliers. Classification is scored by threshold-free ROC-AUC.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

MAX_ROWS = 5000
MAX_FEATURES = 20
MAX_LEVELS = 20
NUMERIC_BINS = 8
FORWARD_STEPS = 10
NULL_PERMUTATIONS = 49
NULL_ALPHA = 0.05


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
    train: pd.DataFrame,
    test: pd.DataFrame,
    features: list[str],
    roles: dict[str, str],
    *,
    bins: int = 0,
) -> tuple[np.ndarray, np.ndarray]:
    """Fit medians, scales, bin edges and capped category vocabulary on train alone.

    With ``bins > 0`` numeric values are clipped to the training 1st-99th percentiles
    and also encoded as indicators of training-quantile bins.
    """
    train_parts = [np.ones((len(train), 1))]
    test_parts = [np.ones((len(test), 1))]
    for name in features:
        if roles[name] == "numeric":
            left = train[name].to_numpy(dtype=float, na_value=np.nan)
            right = test[name].to_numpy(dtype=float, na_value=np.nan)
            median = float(np.nanmedian(left)) if np.isfinite(left).any() else 0.0
            left, right = np.nan_to_num(left, nan=median), np.nan_to_num(right, nan=median)
            if bins:
                low, high = np.quantile(left, [0.01, 0.99])
                left, right = np.clip(left, low, high), np.clip(right, low, high)
                edges = np.unique(np.quantile(left, np.linspace(0, 1, bins + 1)[1:-1]))
                left_bins = np.searchsorted(edges, left, side="left")
                right_bins = np.searchsorted(edges, right, side="left")
                for index in range(1, len(edges) + 1):
                    train_parts.append((left_bins == index).astype(float)[:, None])
                    test_parts.append((right_bins == index).astype(float)[:, None])
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


class _Fold:
    """One train/test split with every feature encoded once and a cached Gram matrix.

    Fitting a feature subset solves ridge (unit penalty, free intercept) on the
    matching sub-block, which equals encoding and fitting that subset from scratch.
    """

    def __init__(
        self,
        train: pd.DataFrame,
        test: pd.DataFrame,
        names: list[str],
        roles: dict[str, str],
        response: np.ndarray,
        actual: np.ndarray,
        task: str,
    ) -> None:
        train_parts, test_parts = [np.ones((len(train), 1))], [np.ones((len(test), 1))]
        self.columns: dict[str, np.ndarray] = {}
        width = 1
        for name in names:
            left, right = _design(train, test, [name], roles, bins=NUMERIC_BINS)
            train_parts.append(left[:, 1:])
            test_parts.append(right[:, 1:])
            self.columns[name] = np.arange(width, width + left.shape[1] - 1)
            width += left.shape[1] - 1
        self.design = np.concatenate(train_parts, axis=1)
        self.test = np.concatenate(test_parts, axis=1)
        self.gram = self.design.T @ self.design
        self.cross = self.design.T @ response
        self.actual, self.task = actual, task

    def relabeled(self, response: np.ndarray, actual: np.ndarray) -> _Fold:
        """The same fold with a different response; reuses the design and Gram matrix."""
        fold = object.__new__(_Fold)
        fold.columns, fold.design, fold.test, fold.gram = (
            self.columns,
            self.design,
            self.test,
            self.gram,
        )
        fold.cross = self.design.T @ response
        fold.actual, fold.task = actual, self.task
        return fold

    def score(self, names: list[str]) -> float:
        index = np.concatenate([[0], *(self.columns[name] for name in names)]).astype(int)
        penalty = np.eye(len(index))
        penalty[0, 0] = 0.0
        weights = np.linalg.solve(self.gram[np.ix_(index, index)] + penalty, self.cross[index])
        return _score(self.actual, self.test[:, index] @ weights, self.task)


def _auc(positive: np.ndarray, scores: np.ndarray) -> float:
    """Mann-Whitney ROC-AUC with average ranks for ties; constant scores give 0.5."""
    ranks = pd.Series(scores).rank(method="average").to_numpy()
    count = int(positive.sum())
    other = len(positive) - count
    if count == 0 or other == 0:
        return 0.5
    return float((ranks[positive].sum() - count * (count + 1) / 2) / (count * other))


def _score(actual: np.ndarray, prediction: np.ndarray, task: str) -> float:
    if task == "regression":
        return float(np.sqrt(np.mean((actual - prediction) ** 2)))
    if prediction.shape[1] == 2:
        return _auc(actual == 1, prediction[:, 1] - prediction[:, 0])
    # Macro one-vs-rest ROC-AUC over classes present in the held-out fold.
    return float(
        np.mean([_auc(actual == label, prediction[:, label]) for label in np.unique(actual)])
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
    forward_steps: int = 0,
) -> dict[str, Any]:
    """Compare baseline and baseline+feature on identical held-out folds.

    With ``forward_steps > 0`` a greedy forward selection then adds, step by step, the
    feature with the largest held-out gain over the baseline plus earlier picks, and
    records each feature's conditional gain (redundant features gain little,
    complementary ones keep their value).
    """
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
        "method": "binned_ridge_baseline",
        "split": "shuffled_stratified" if task == "classification" else "shuffled_kfold",
        "cv_folds": cv_folds,
        "seed": seed,
        "baseline_features": baseline,
        "max_rows": MAX_ROWS,
        "max_features": MAX_FEATURES,
        "row_count": 0,
        "excluded_target_rows": 0,
        "sampled": False,
        "metric": "roc_auc" if task == "classification" else "rmse",
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
    cache = []
    splits = []
    for test_indices in folds:
        train_mask = np.ones(len(subset), dtype=bool)
        train_mask[test_indices] = False
        train_indices = np.flatnonzero(train_mask)
        splits.append((train_indices, test_indices))
        cache.append(
            _Fold(
                subset.iloc[train_indices],
                subset.iloc[test_indices],
                list(dict.fromkeys([*baseline, *selected])),
                roles,
                response[train_indices],
                actual[test_indices],
                task,
            )
        )
    baseline_scores = [fold.score(baseline) for fold in cache]
    feature_scores = {
        name: baseline_scores
        if name in baseline
        else [fold.score([*baseline, name]) for fold in cache]
        for name in selected
    }
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
    if forward_steps > 0:
        result["forward_selection"] = _forward(
            cache, baseline, baseline_scores, feature_scores, result, forward_steps
        )
    result["null_test"] = _null_test(
        cache, splits, baseline, selected, response, actual, task, result, rng
    )
    return result


def _null_test(
    cache: list[_Fold],
    splits: list[tuple[np.ndarray, np.ndarray]],
    baseline: list[str],
    selected: list[str],
    response: np.ndarray,
    actual: np.ndarray,
    task: str,
    result: dict[str, Any],
    rng: np.random.Generator,
) -> dict[str, Any]:
    """Permutation test of the best feature's mean held-out gain on the same folds.

    Shuffling the target breaks every feature-target link, so the permuted maxima form
    the gain the best of these features reaches by chance. Without it some chart always
    looks "worth seeing", even on pure noise.
    """
    candidates = [name for name in selected if name not in baseline]
    if not candidates:
        return {"status": "skipped", "reason": "no_candidate_features"}

    def best_gain(folds: list[_Fold]) -> float:
        reference = [fold.score(baseline) for fold in folds]
        return max(
            float(_gains([fold.score([*baseline, name]) for fold in folds], reference, task).mean())
            for name in candidates
        )

    observed = max(result["features"][name]["incremental_gain"] for name in candidates)
    exceed = 0
    for _ in range(NULL_PERMUTATIONS):
        order = rng.permutation(len(actual))
        shuffled_response, shuffled_actual = response[order], actual[order]
        folds = [
            fold.relabeled(shuffled_response[train], shuffled_actual[test])
            for fold, (train, test) in zip(cache, splits, strict=True)
        ]
        exceed += best_gain(folds) >= observed
    p_value = (1 + exceed) / (1 + NULL_PERMUTATIONS)
    return {
        "status": "completed",
        "statistic": "max_mean_incremental_gain",
        "observed": round(observed, 6),
        "permutations": NULL_PERMUTATIONS,
        "p_value": round(p_value, 6),
        "alpha": NULL_ALPHA,
        "signal_detected": p_value <= NULL_ALPHA,
    }


def _gains(scores: list[float], reference: list[float], task: str) -> np.ndarray:
    gains = np.array(scores) - np.array(reference)
    return -gains if task == "regression" else gains


def _forward(
    cache: list[_Fold],
    baseline: list[str],
    baseline_scores: list[float],
    feature_scores: dict[str, list[float]],
    result: dict[str, Any],
    steps: int,
) -> dict[str, Any]:
    """Greedy forward selection on the cached folds; annotates conditional gains."""
    task = cache[0].task
    chosen, reference = list(baseline), baseline_scores
    remaining = [name for name in feature_scores if name not in baseline]
    trial = {name: feature_scores[name] for name in remaining}
    path = []
    while remaining and len(path) < steps:
        if path:
            trial = {name: [fold.score([*chosen, name]) for fold in cache] for name in remaining}
        mean_gain = {name: float(_gains(trial[name], reference, task).mean()) for name in remaining}
        for name in remaining:
            result["features"][name].update(
                conditional_gain=round(mean_gain[name], 6),
                conditional_score=round(float(np.mean(trial[name])), 6),
                conditioned_on=list(chosen),
            )
        best = max(remaining, key=lambda name: mean_gain[name])
        if mean_gain[best] <= 0:
            break
        path.append(
            {
                "feature": best,
                "conditional_gain": round(mean_gain[best], 6),
                "score": round(float(np.mean(trial[best])), 6),
            }
        )
        chosen.append(best)
        reference = trial[best]
        remaining.remove(best)
    return {
        "status": "completed",
        "max_steps": steps,
        "steps": path,
        "note": "Each step adds the feature with the largest mean held-out gain over the "
        "baseline plus earlier steps; features never added report their gain over the "
        "last conditioning set.",
    }
