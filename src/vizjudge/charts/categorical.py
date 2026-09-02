"""Categorical chart renderers."""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from vizjudge.charts.generator import ChartCandidate


def render_categorical(frame: pd.DataFrame, candidate: ChartCandidate, path: Path) -> None:
    fig, axis = plt.subplots(figsize=(8, 5))
    counts = frame[candidate.x].fillna("<missing>").astype(str).value_counts().head(20)
    counts.sort_values().plot.barh(ax=axis, color="#F58518")
    axis.set_xlabel("Count")
    axis.set_ylabel(candidate.x)
    axis.set_title(candidate.title)
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)
