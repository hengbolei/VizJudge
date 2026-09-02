"""Numeric chart renderers."""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from vizjudge.charts.generator import ChartCandidate


def render_numeric(frame: pd.DataFrame, candidate: ChartCandidate, path: Path) -> None:
    fig, axis = plt.subplots(figsize=(8, 5))
    if candidate.kind == "histogram":
        values = pd.to_numeric(frame[candidate.x], errors="coerce").dropna()
        axis.hist(values, bins="auto", color="#4C78A8", edgecolor="white")
        axis.set_xlabel(candidate.x)
        axis.set_ylabel("Count")
    else:
        assert candidate.y is not None
        pair = frame[[candidate.x, candidate.y]].dropna()
        axis.scatter(pair[candidate.x], pair[candidate.y], alpha=0.55, s=18, color="#4C78A8")
        axis.set_xlabel(candidate.x)
        axis.set_ylabel(candidate.y)
    axis.set_title(candidate.title)
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)
