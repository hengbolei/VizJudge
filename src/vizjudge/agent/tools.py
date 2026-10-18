"""Fine-grained, frame-level analysis steps that host agents compose themselves.

Each function is deterministic for a given frame and options, and returns chart
IDs and evidence IDs identical to those used by ``judge_report``.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd

from vizjudge.agent.judge import chart_evidence, chart_key
from vizjudge.charts.generator import ChartCandidate, generate_candidates
from vizjudge.core.report import render_chart, safe_filename
from vizjudge.core.type_infer import ROLE_CATEGORICAL, ROLE_NUMERIC, infer_types, resolve_task
from vizjudge.scoring.predictive import MAX_FEATURES, validate_predictive_value
from vizjudge.scoring.rules import judge_candidate

_SELECTION_FIELDS = ("image", "selection_score", "selection_penalty", "selection_reasons")


def describe_chart(chart: dict[str, Any]) -> dict[str, Any]:
    """Attach the stable chart ID and citable evidence to a serialized chart."""
    return {"chart_id": chart_key(chart), **chart, "evidence": chart_evidence(chart)}


def _judged(
    frame: pd.DataFrame, roles: dict[str, str], target: str | None, max_candidates: int
) -> list[ChartCandidate]:
    candidates = generate_candidates(frame, roles, target, max_candidates=max_candidates)
    for candidate in candidates:
        judge_candidate(frame, candidate)
    return candidates


def list_candidates(
    frame: pd.DataFrame,
    *,
    target: str | None = None,
    task: str = "auto",
    role_overrides: dict[str, str] | None = None,
    max_candidates: int = 200,
) -> dict[str, Any]:
    """Generate and rule-score every candidate, ordered by heuristic score."""
    roles = infer_types(frame, role_overrides)
    resolved = resolve_task(frame, roles, target, task, role_overrides)
    candidates = sorted(
        _judged(frame, roles, target, max_candidates), key=lambda item: (-item.score, item.key)
    )
    charts = []
    for candidate in candidates:
        chart = describe_chart(candidate.to_dict())
        for name in _SELECTION_FIELDS:
            chart.pop(name)
        charts.append(chart)
    return {
        "target": target,
        "task": resolved,
        "roles": roles,
        "candidate_count": len(charts),
        "candidates": charts,
    }


def validate_features(
    frame: pd.DataFrame,
    *,
    target: str,
    features: list[str],
    task: str = "auto",
    role_overrides: dict[str, str] | None = None,
    baseline_features: list[str] | None = None,
    cv_folds: int = 5,
    seed: int = 42,
) -> dict[str, Any]:
    """Measure held-out incremental gain of each named feature over the baseline."""
    if not isinstance(target, str):
        raise ValueError("target is required for feature validation")
    if (
        not isinstance(features, list)
        or not features
        or any(not isinstance(name, str) for name in features)
        or len(set(features)) != len(features)
        or len(features) > MAX_FEATURES
    ):
        raise ValueError(f"features must be a list of 1 to {MAX_FEATURES} distinct column names")
    roles = infer_types(frame, role_overrides)
    resolved = resolve_task(frame, roles, target, task, role_overrides)
    for name in features:
        if name not in roles:
            raise ValueError(f"Feature column not found: {name}")
        if name == target:
            raise ValueError("The target cannot be validated as its own feature")
        if roles[name] not in {ROLE_NUMERIC, ROLE_CATEGORICAL}:
            raise ValueError(
                f"Feature {name} has role {roles[name]}; "
                "use role_overrides to mark it numeric or categorical"
            )
    predictive = validate_predictive_value(
        frame,
        roles,
        target,
        resolved,
        features,
        cv_folds=cv_folds,
        seed=seed,
        baseline_features=baseline_features,
    )
    # Targeted candidates precede distributions, so one per column is enough.
    charts = {
        candidate.feature: candidate
        for candidate in generate_candidates(frame, roles, target, max_candidates=len(roles))
        if candidate.target is not None
    }
    results = []
    for name in features:
        outcome = predictive["features"].get(
            name, {"status": "skipped", "reason": predictive["reason"]}
        )
        candidate = charts[name]
        judge_candidate(frame, candidate)
        candidate.predictive = outcome
        chart = candidate.to_dict()
        results.append(
            {
                "feature": name,
                "chart_id": chart_key(chart),
                "evidence": chart_evidence(chart),
                **outcome,
            }
        )
    summary = {key: value for key, value in predictive.items() if key != "features"}
    return {"target": target, "task": resolved, "validation": summary, "features": results}


def render_chart_ids(
    frame: pd.DataFrame,
    *,
    chart_ids: list[str],
    output_dir: str,
    max_charts: int,
    target: str | None = None,
    task: str = "auto",
    role_overrides: dict[str, str] | None = None,
    max_candidates: int = 200,
) -> dict[str, Any]:
    """Render previously generated chart IDs in the given order, within the image budget."""
    if (
        not isinstance(chart_ids, list)
        or not chart_ids
        or any(not isinstance(item, str) for item in chart_ids)
        or len(set(chart_ids)) != len(chart_ids)
    ):
        raise ValueError("chart_ids must be a non-empty list of distinct chart IDs")
    for item in chart_ids:
        try:
            parsed = json.loads(item)
        except json.JSONDecodeError as exc:
            raise ValueError(f"Malformed chart ID: {item}") from exc
        if not isinstance(parsed, list) or len(parsed) != 4:
            raise ValueError(f"Malformed chart ID: {item}")
    roles = infer_types(frame, role_overrides)
    resolve_task(frame, roles, target, task, role_overrides)
    pool = {
        chart_key(candidate.to_dict()): candidate
        for candidate in generate_candidates(frame, roles, target, max_candidates=max_candidates)
    }
    unknown = [item for item in chart_ids if item not in pool]
    if unknown:
        raise ValueError(
            "Chart IDs were not generated for this target, task and role_overrides: "
            + ", ".join(unknown)
        )
    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    rendered, skipped = [], []
    for index, chart_id in enumerate(chart_ids, start=1):
        if index > max_charts:
            skipped.append(chart_id)
            continue
        candidate = pool[chart_id]
        filename = f"{index:02d}-{safe_filename(candidate.key)}.png"
        render_chart(frame, candidate, destination / filename)
        rendered.append(
            {
                "chart_id": chart_id,
                "title": candidate.title,
                "image": filename,
                "path": str(destination / filename),
            }
        )
    return {"output_dir": str(destination), "rendered": rendered, "skipped_by_budget": skipped}
