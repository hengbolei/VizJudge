import numpy as np
import pandas as pd
import pytest

from vizjudge.charts.generator import ChartCandidate
from vizjudge.scoring.metrics import (
    absolute_correlation,
    correlation_ratio,
    cramers_v,
    information_coefficient,
    one_to_one_agreement,
)
from vizjudge.scoring.ranker import rank_candidates
from vizjudge.scoring.rules import judge_candidate


def test_association_metrics_detect_strong_signal() -> None:
    numeric = pd.Series(range(100))
    assert absolute_correlation(numeric, numeric * 3) > 0.99
    assert correlation_ratio(pd.Series(["a"] * 50 + ["b"] * 50), numeric) > 0.7
    assert cramers_v(pd.Series(["a"] * 50 + ["b"] * 50), pd.Series([0] * 50 + [1] * 50)) > 0.9


def test_ranker_puts_strong_target_relationship_first() -> None:
    frame = pd.DataFrame(
        {
            "signal": range(100),
            "noise": [index % 7 for index in range(100)],
            "target": [index * 2 for index in range(100)],
        }
    )
    candidates = [
        ChartCandidate("target_scatter", "signal", "signal", "target", "target"),
        ChartCandidate("target_scatter", "noise", "noise", "target", "target"),
    ]

    ranked = rank_candidates(frame, candidates, top_k=2)

    assert ranked[0].x == "signal"
    assert ranked[0].score > ranked[1].score


def test_duplicates_and_redundant_features_are_handled_without_name_collisions():
    frame = pd.DataFrame({"a b": range(100), "a_b": range(100), "target": range(100)})
    first = ChartCandidate("target_scatter", "first", "a b", "target", "target")
    duplicate = ChartCandidate("target_scatter", "duplicate", "a b", "target", "target")
    second = ChartCandidate("target_scatter", "second", "a_b", "target", "target")
    ranked = rank_candidates(frame, [first, duplicate, second], top_k=3)
    assert len(ranked) == 2
    assert ranked[1].selection_penalty == 25
    assert any("redundant" in reason for reason in ranked[1].selection_reasons)
    plain = rank_candidates(frame, [first, second], top_k=2, diversity=False)
    assert all(chart.selection_score == chart.score for chart in plain)


def test_information_coefficient_detects_nonmonotone_signal_on_the_r_scale():
    rng = np.random.default_rng(0)
    x = rng.uniform(-1, 1, 2000)
    noise = rng.normal(size=2000)
    parabola = pd.Series(x**2 + rng.normal(0, 0.05, 2000))
    assert absolute_correlation(pd.Series(x), parabola) < 0.1
    assert (
        information_coefficient(
            pd.Series(x), parabola, feature_discrete=False, target_discrete=False
        )
        > 0.8
    )
    assert (
        information_coefficient(
            pd.Series(noise), parabola, feature_discrete=False, target_discrete=False
        )
        < 0.1
    )
    # For a bivariate normal pair the coefficient approximates |r|.
    y = 0.6 * noise + 0.8 * rng.normal(size=2000)
    value = information_coefficient(
        pd.Series(noise), pd.Series(y), feature_discrete=False, target_discrete=False
    )
    assert 0.45 < value < 0.7


def test_box_charts_score_eta_not_eta_squared():
    frame = pd.DataFrame({"x": [0.0, 1.0] * 100, "target": ["a", "b"] * 100})
    frame.loc[::7, "x"] = 0.5
    candidate = judge_candidate(frame, ChartCandidate("target_box", "t", "target", "x", "target"))
    assert candidate.metrics["eta"] == pytest.approx(candidate.metrics["eta_squared"] ** 0.5)
    assert "information_coefficient" in candidate.metrics
    assert "relevance" in candidate.metrics


def test_one_to_one_reencoding_is_redundant():
    levels = ["Bachelors", "HS-grad", "Masters", "Doctorate"]
    names = pd.Series([levels[index % 4] for index in range(200)])
    codes = names.map({"Bachelors": 13, "HS-grad": 9, "Masters": 14, "Doctorate": 16})
    assert one_to_one_agreement(names, codes) == 1.0
    assert one_to_one_agreement(pd.Series(range(200)), pd.Series(range(200))) == 0.0
    frame = pd.DataFrame(
        {"education": names, "education_num": codes, "target": [0, 1, 1, 0, 1] * 40}
    )
    candidates = [
        ChartCandidate("target_category", "e", "education", "target", "target"),
        ChartCandidate("target_box", "n", "target", "education_num", "target"),
    ]
    ranked = rank_candidates(frame, candidates, top_k=2)
    assert any("one-to-one" in reason for reason in ranked[1].selection_reasons)


def test_high_cardinality_category_skips_association_but_keeps_relevance_evidence():
    frame = pd.DataFrame({"name": [f"n{index}" for index in range(300)], "target": [0, 1] * 150})
    candidate = ChartCandidate("target_category", "t", "name", "target", "target")
    candidate.predictive = {
        "status": "completed",
        "metric": "roc_auc",
        "incremental_gain": 0.0,
        "feature_score": 0.5,
        "baseline_score": 0.5,
    }
    judged = judge_candidate(frame, candidate)
    assert judged.metrics["relevance"] == 0.0
    assert "predictive_adjustment" in judged.metrics
