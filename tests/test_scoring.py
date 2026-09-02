import pandas as pd

from vizjudge.charts.generator import ChartCandidate
from vizjudge.scoring.metrics import absolute_correlation, correlation_ratio, cramers_v
from vizjudge.scoring.ranker import rank_candidates


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
