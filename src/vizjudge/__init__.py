"""VizJudge: chart-value ranking for machine-learning workflows."""

from vizjudge._version import __version__
from vizjudge.core.report import analyze_dataset, analyze_frame

__all__ = ["analyze_dataset", "analyze_frame", "__version__"]
