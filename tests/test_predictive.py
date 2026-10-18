import numpy as np
import pandas as pd
import pytest

from vizjudge import analyze_frame
from vizjudge.scoring.predictive import _design


def test_incremental_value_separates_signal_and_noise_and_is_reproducible():
    rng = np.random.default_rng(6)
    signal = rng.normal(size=300)
    frame = pd.DataFrame(
        {
            "signal": signal,
            "noise": rng.normal(size=300),
            "target": 5 * signal + rng.normal(0, 0.1, 300),
        }
    )
    first = analyze_frame(frame, target="target", render=False)
    second = analyze_frame(frame, target="target", render=False)
    assert first["predictive_validation"] == second["predictive_validation"]
    values = first["predictive_validation"]["features"]
    assert values["signal"]["incremental_gain"] > 4
    assert values["signal"]["positive_folds"] == 5
    assert values["noise"]["incremental_gain"] < 0.1
    with_baseline = analyze_frame(
        frame, target="target", render=False, baseline_features=["signal"]
    )
    assert with_baseline["predictive_validation"]["features"]["signal"]["incremental_gain"] == 0
    assert with_baseline["predictive_validation"]["features"]["noise"]["incremental_gain"] < 0.02


def test_multiclass_and_categorical_regression():
    frame = pd.DataFrame({"category": ["a", "b", "c"] * 60, "target": [0.0, 1.0, 2.0] * 60})
    classified = analyze_frame(frame, target="target", task="classification", render=False)
    assert classified["predictive_validation"]["metric"] == "roc_auc"
    assert classified["predictive_validation"]["features"]["category"]["feature_score"] == 1.0
    regressed = analyze_frame(frame, target="target", task="regression", render=False)
    assert any(chart["kind"] == "regression_box" for chart in regressed["ranked_charts"])
    assert regressed["predictive_validation"]["features"]["category"]["incremental_gain"] > 0.7


def test_preprocessing_is_fit_on_training_fold_only():
    train = pd.DataFrame({"x": [0.0, 2.0, None], "cat": ["a", "b", None]})
    test = pd.DataFrame({"x": [None, 100.0], "cat": ["unseen", "a"]})
    left, right = _design(train, test, ["x", "cat"], {"x": "numeric", "cat": "categorical"})
    assert left.shape[1] == right.shape[1] == 6
    assert right[0, 1] == 0.0  # Impute training median, not held-out median.
    assert right[1, 1] > 100
    assert right[0, 3:].sum() == 0.0  # Unseen category does not create a new column.


@pytest.mark.parametrize(
    "options",
    [
        {"cv_folds": 1},
        {"cv_folds": True},
        {"seed": -1},
        {"baseline_features": ["target"]},
        {"top_k": False},
    ],
)
def test_invalid_validation_options(options):
    frame = pd.DataFrame({"x": range(40), "target": [0, 1] * 20})
    with pytest.raises(ValueError):
        analyze_frame(frame, target="target", render=False, **options)


def test_insufficient_support_and_missing_targets_are_reported():
    frame = pd.DataFrame({"x": np.arange(40, dtype=float), "target": [0] * 38 + [1, None]})
    report = analyze_frame(frame, target="target", render=False)
    validation = report["predictive_validation"]
    assert validation["status"] == "skipped"
    assert validation["excluded_target_rows"] == 1
    assert "class_support" in validation["reason"]


def test_classification_uses_threshold_free_roc_auc():
    rng = np.random.default_rng(3)
    signal = rng.normal(size=600)
    # Imbalanced target: a weak feature barely moves argmax accuracy but moves AUC.
    target = (signal + rng.normal(0, 2.0, 600) > 2.2).astype(int)
    frame = pd.DataFrame({"signal": signal, "noise": rng.normal(size=600), "target": target})
    report = analyze_frame(frame, target="target", task="classification", render=False)
    features = report["predictive_validation"]["features"]
    assert report["predictive_validation"]["metric"] == "roc_auc"
    assert features["signal"]["incremental_gain"] > 0.1
    assert abs(features["noise"]["incremental_gain"]) < 0.08


def test_forward_selection_discounts_duplicates_and_keeps_complements():
    rng = np.random.default_rng(4)
    left, right = rng.normal(size=500), rng.normal(size=500)
    frame = pd.DataFrame(
        {
            "left": left,
            "left_copy": left + rng.normal(0, 0.01, 500),
            "right": right,
            "target": 2 * left + right + rng.normal(0, 0.1, 500),
        }
    )
    validation = analyze_frame(frame, target="target", render=False)["predictive_validation"]
    steps = [step["feature"] for step in validation["forward_selection"]["steps"]]
    assert steps[0] in {"left", "left_copy"} and steps[1] == "right"
    duplicate = "left_copy" if steps[0] == "left" else "left"
    features = validation["features"]
    assert features["right"]["conditional_gain"] > 0.5
    assert features[duplicate]["conditional_gain"] < 0.05
    assert steps[0] in features[duplicate]["conditioned_on"]
