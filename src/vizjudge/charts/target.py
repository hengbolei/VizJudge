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
    if candidate.kind in {"target_box", "regression_box"}:
        subset = frame[[candidate.x, candidate.y]].dropna()
        levels = subset[candidate.x].value_counts().head(15).index
        groups = [(level, subset.loc[subset[candidate.x] == level]) for level in levels]
        if groups:
            axis.boxplot(
                [group[candidate.y] for _, group in groups],
                tick_labels=[str(k) for k, _ in groups],
            )
        else:
            axis.text(0.5, 0.5, "No complete observations", ha="center")
        axis.set_xlabel(candidate.x)
        axis.set_ylabel(candidate.y)
    elif candidate.kind == "target_category":
        subset = frame[[candidate.x, candidate.y]].dropna()
        levels = subset[candidate.x].value_counts().head(20).index
        subset = subset.loc[subset[candidate.x].isin(levels)]
        table = pd.crosstab(
            subset[candidate.x].astype("string"),
            subset[candidate.y].astype("string"),
            normalize="index",
        )
        # Bound the legend for an explicitly high-cardinality classification target.
        if table.shape[1] > 20:
            keep = subset[candidate.y].astype("string").value_counts().head(19).index
            remaining = table.drop(columns=keep).sum(axis=1)
            table = table.loc[:, keep].copy()
            table["<other classes>"] = remaining
        if not table.empty:
            table.plot.bar(stacked=True, ax=axis, colormap="tab10")
        else:
            axis.text(0.5, 0.5, "No complete observations", ha="center")
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
