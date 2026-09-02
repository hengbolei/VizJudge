"""Generate inexpensive chart specifications before rendering images."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field

import pandas as pd

from vizjudge.core.type_infer import ROLE_CATEGORICAL, ROLE_NUMERIC


@dataclass(slots=True)
class ChartCandidate:
    """A chart proposal plus its eventual value judgment."""

    kind: str
    title: str
    x: str
    y: str | None = None
    target: str | None = None
    score: float = 0.0
    reasons: list[str] = field(default_factory=list)
    observations: list[str] = field(default_factory=list)
    image: str | None = None

    @property
    def key(self) -> str:
        parts = [self.kind, self.x, self.y or "", self.target or ""]
        return "__".join(part.replace(" ", "_") for part in parts if part)

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def generate_candidates(
    frame: pd.DataFrame, roles: dict[str, str], target: str | None = None
) -> list[ChartCandidate]:
    """Create chart specs for distributions and target relationships."""
    if target is not None and target not in frame.columns:
        raise ValueError(f"Target column not found: {target}")

    candidates: list[ChartCandidate] = []
    for name, role in roles.items():
        if name == target:
            continue
        if role == ROLE_NUMERIC:
            candidates.append(ChartCandidate("histogram", f"Distribution of {name}", x=name))
        elif role == ROLE_CATEGORICAL:
            candidates.append(ChartCandidate("category_bar", f"Frequency of {name}", x=name))

    if target is None:
        numeric = [name for name, role in roles.items() if role == ROLE_NUMERIC]
        correlations: list[tuple[float, str, str]] = []
        for index, left in enumerate(numeric):
            for right in numeric[index + 1 :]:
                value = frame[[left, right]].corr().iloc[0, 1]
                if pd.notna(value):
                    correlations.append((abs(float(value)), left, right))
        for _, left, right in sorted(correlations, reverse=True)[:10]:
            candidates.append(ChartCandidate("scatter", f"{left} vs {right}", x=left, y=right))
        return candidates

    target_role = roles[str(target)]
    for name, role in roles.items():
        if name == target:
            continue
        if target_role == ROLE_NUMERIC and role == ROLE_NUMERIC:
            candidates.append(
                ChartCandidate(
                    "target_scatter", f"{name} vs target {target}", x=name, y=target, target=target
                )
            )
        elif target_role == ROLE_CATEGORICAL and role == ROLE_NUMERIC:
            candidates.append(
                ChartCandidate(
                    "target_box", f"{name} by target {target}", x=target, y=name, target=target
                )
            )
        elif target_role == ROLE_CATEGORICAL and role == ROLE_CATEGORICAL:
            candidates.append(
                ChartCandidate(
                    "target_category",
                    f"{name} by target {target}",
                    x=name,
                    y=target,
                    target=target,
                )
            )
    return candidates
