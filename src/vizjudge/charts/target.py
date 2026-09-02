"""Target-aware chart renderers."""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from vizjudge.charts.generator import ChartCandidate


def render_target(frame: pd.DataFrame, candidate: ChartCandidate, path: Path) -> None:
    fig, axis = plt.subplots(figsize=(8, 5))
    assert candidate.y is not None
    if candidate.kind == "target_box":
        subset = frame[[candidate.x, candidate.y]].dropna()
        groups = list(subset.groupby(candidate.x, observed=True, sort=False))[:15]
        axis.boxplot(
            [group[candidate.y] for _, group in groups], tick_labels=[str(k) for k, _ in groups]
        )
        axis.set_xlabel(candidate.x)
        axis.set_ylabel(candidate.y)
    elif candidate.kind == "target_category":
        table = pd.crosstab(
            frame[candidate.x].fillna("<missing>"),
            frame[candidate.y].fillna("<missing>"),
            normalize="index",
        ).head(20)
        table.plot.bar(stacked=True, ax=axis, colormap="tab10")
        axis.set_ylabel("Within-category share")
        axis.legend(title=candidate.y, bbox_to_anchor=(1.02, 1), loc="upper left")
    else:
        subset = frame[[candidate.x, candidate.y]].dropna()
        axis.scatter(subset[candidate.x], subset[candidate.y], alpha=0.55, s=18, color="#54A24B")
        axis.set_xlabel(candidate.x)
        axis.set_ylabel(candidate.y)
    axis.set_title(candidate.title)
    fig.tight_layout()
    fig.savefig(path, dpi=140, bbox_inches="tight")
    plt.close(fig)
