"""Which charts surface, not which features predict: the half the benchmark does not cover."""

import numpy as np
import pandas as pd

from vizjudge.agent.judge import judge_report
from vizjudge.agent.runner import refine_report
from vizjudge.core.report import analyze_frame


def _wide(seed: int = 0, rows: int = 600, noise: int = 19) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    data = {f"noise_{index:02d}": rng.normal(size=rows) for index in range(noise)}
    data["x1_strong"] = rng.normal(size=rows)
    data["x2_moderate"] = rng.normal(size=rows)
    logit = 2.0 * data["x1_strong"] + 0.8 * data["x2_moderate"]
    data["y"] = (rng.random(rows) < 1 / (1 + np.exp(-logit))).astype(int)
    return pd.DataFrame(data)


def _rank(report: dict, kind: str, feature: str) -> int:
    for index, chart in enumerate(report["ranked_charts"], start=1):
        if chart["kind"] == kind and (chart["feature"] or chart["x"]) == feature:
            return index
    raise AssertionError(f"{kind} of {feature} was not ranked")


def test_signal_charts_outrank_noise_distributions_on_a_wide_table():
    report = analyze_frame(
        _wide(), target="y", task="classification", render=False, top_k=40, diversity=False
    )
    noise_histograms = [
        index
        for index, chart in enumerate(report["ranked_charts"], start=1)
        if chart["kind"] == "histogram" and chart["x"].startswith("noise_")
    ]
    assert _rank(report, "target_box", "x1_strong") == 1
    assert _rank(report, "target_box", "x2_moderate") < min(noise_histograms)
    assert report["signal_assessment"]["status"] == "signal"


def test_skewed_distribution_still_surfaces_when_it_is_actionable():
    frame = _wide(noise=4)
    frame["amount"] = np.random.default_rng(1).lognormal(sigma=1.5, size=len(frame))
    report = analyze_frame(frame, target="y", task="classification", render=False, top_k=40)
    plain = _rank(report, "histogram", "noise_00")
    assert _rank(report, "histogram", "amount") < plain


def test_post_outcome_column_is_surfaced_flagged_and_deferred():
    frame = _wide(noise=10)
    rng = np.random.default_rng(2)
    filled = rng.integers(1, 90, size=len(frame)).astype(float)
    frame["days_since_default"] = np.where(frame["y"] == 1, filled, np.nan)
    report = analyze_frame(frame, target="y", task="classification", render=False, top_k=6)
    first = report["ranked_charts"][0]
    assert first["feature"] == "days_since_default"
    assert first["metrics"]["missingness_association"] > 0.95
    assert any("leakage" in warning for warning in first["warnings"])
    assert [alert["feature"] for alert in report["leakage_alerts"]] == ["days_since_default"]

    judgment = judge_report(report)
    review = judgment["reviews"][0]
    assert review["verdict"] == "defer"
    assert review["confidence"]["level"] != "high"
    # Statistical-only refinement must not drop the unreviewed suspect again.
    refined = refine_report(report, judgment, judgment["refinement"])
    assert refined["ranked_charts"][0]["feature"] == "days_since_default"


def test_pure_noise_reports_no_signal_and_low_confidence():
    rng = np.random.default_rng(3)
    frame = pd.DataFrame({f"z{index}": rng.normal(size=600) for index in range(8)})
    frame["y"] = rng.integers(0, 2, size=600)
    report = analyze_frame(frame, target="y", task="classification", render=False, top_k=6)
    assert report["signal_assessment"]["status"] == "no_signal"
    assert report["warnings"][0] == report["signal_assessment"]["message"]
    levels = {review["confidence"]["level"] for review in judge_report(report)["reviews"]}
    assert levels == {"low"}


def test_null_test_is_seeded_and_reproducible():
    frame = _wide(noise=5, rows=300)
    first = analyze_frame(frame, target="y", render=False, seed=7)["predictive_validation"]
    second = analyze_frame(frame, target="y", render=False, seed=7)["predictive_validation"]
    assert first["null_test"] == second["null_test"]
    assert first["null_test"]["signal_detected"]
