"""Run the complete VizJudge workflow and serialize its evidence."""

from __future__ import annotations

import json
import re
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pandas as pd

from vizjudge._version import SCHEMA_VERSION, __version__
from vizjudge.charts.categorical import render_categorical
from vizjudge.charts.generator import ChartCandidate, generate_candidates
from vizjudge.charts.numeric import render_numeric
from vizjudge.charts.target import render_target
from vizjudge.core.loader import load_dataset
from vizjudge.core.profiler import profile_dataframe
from vizjudge.core.type_infer import infer_types, resolve_task
from vizjudge.core.validation import validate_top_k
from vizjudge.recommendations.data_quality import recommend_data_quality
from vizjudge.recommendations.feature_engineering import recommend_features
from vizjudge.recommendations.modeling import recommend_models
from vizjudge.scoring.predictive import (
    FORWARD_STEPS,
    validate_options,
    validate_predictive_value,
)
from vizjudge.scoring.ranker import rank_candidates
from vizjudge.scoring.rules import judge_candidate


def safe_filename(value: str) -> str:
    """Turn a chart key into a short, filesystem-safe PNG stem."""
    cleaned = re.sub(r"[^a-zA-Z0-9._-]+", "-", value).strip("-_").lower()
    return cleaned[:100] or "chart"


def render_chart(frame: pd.DataFrame, candidate: ChartCandidate, path: Path) -> None:
    """Render one candidate with the renderer matching its chart kind."""
    if candidate.kind in {"histogram", "scatter"}:
        render_numeric(frame, candidate, path)
    elif candidate.kind == "category_bar":
        render_categorical(frame, candidate, path)
    else:
        render_target(frame, candidate, path)


def analyze_frame(
    frame: pd.DataFrame,
    *,
    target: str | None = None,
    task: str = "auto",
    role_overrides: dict[str, str] | None = None,
    validation: bool = True,
    cv_folds: int = 5,
    seed: int = 42,
    baseline_features: list[str] | None = None,
    diversity: bool = True,
    max_candidates: int = 200,
    output_dir: str | Path | None = None,
    top_k: int = 8,
    render: bool = True,
    source: str | None = None,
) -> dict[str, Any]:
    """Analyze an in-memory DataFrame and optionally write a report directory."""
    validate_top_k(top_k)
    validate_top_k(max_candidates)
    validate_options(cv_folds, seed, baseline_features)
    if any(not isinstance(value, bool) for value in (validation, diversity, render)):
        raise ValueError("validation, diversity, and render must be booleans")
    if render and output_dir is None:
        raise ValueError("output_dir is required when render=True")
    roles = infer_types(frame, role_overrides)
    resolved_task = resolve_task(frame, roles, target, task, role_overrides)
    candidates = generate_candidates(frame, roles, target, max_candidates=max_candidates)
    for candidate in candidates:
        judge_candidate(frame, candidate)
    features = list(
        dict.fromkeys(
            candidate.feature
            for candidate in sorted(candidates, key=lambda item: (-item.score, item.key))
            if candidate.feature is not None
        )
    )
    predictive = validate_predictive_value(
        frame,
        roles,
        target,
        resolved_task,
        features,
        enabled=validation,
        cv_folds=cv_folds,
        seed=seed,
        baseline_features=baseline_features,
        forward_steps=FORWARD_STEPS,
    )
    for candidate in candidates:
        if candidate.feature:
            candidate.predictive = predictive["features"].get(
                candidate.feature,
                {
                    "status": "skipped",
                    "reason": predictive["reason"] or "feature_budget",
                },
            )
    ranked = rank_candidates(frame, candidates, top_k=top_k, diversity=diversity)
    profile = profile_dataframe(frame, roles)

    destination = Path(output_dir).expanduser().resolve() if output_dir else None
    if destination is not None:
        destination.mkdir(parents=True, exist_ok=True)
    if render and destination is not None:
        for rank, candidate in enumerate(ranked, start=1):
            filename = f"{rank:02d}-{safe_filename(candidate.key)}.png"
            render_chart(frame, candidate, destination / filename)
            candidate.image = filename

    recommendations = {
        "data_quality": recommend_data_quality(profile),
        "feature_engineering": recommend_features(profile, ranked),
        "modeling": recommend_models(target, roles, ranked),
    }
    report: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "vizjudge_version": __version__,
        "generated_at": datetime.now(UTC).isoformat(),
        "source": source,
        "target": target,
        "task": resolved_task,
        "role_overrides": dict(role_overrides or {}),
        "roles": roles,
        "profile": profile,
        "candidate_count": len(candidates),
        "configuration": {
            "top_k": top_k,
            "render": render,
            "diversity": diversity,
            "validation": validation,
            "cv_folds": cv_folds,
            "seed": seed,
            "baseline_features": list(baseline_features or []),
            "max_candidates": max_candidates,
        },
        "predictive_validation": predictive,
        "warnings": list(
            dict.fromkeys(warning for candidate in ranked for warning in candidate.warnings)
        ),
        "ranked_charts": [candidate.to_dict() for candidate in ranked],
        "recommendations": recommendations,
    }
    if destination is not None:
        (destination / "report.json").write_text(
            json.dumps(report, indent=2, ensure_ascii=False, allow_nan=False), encoding="utf-8"
        )
        (destination / "summary.md").write_text(markdown_summary(report), encoding="utf-8")
    return report


def analyze_dataset(
    path: str | Path,
    *,
    target: str | None = None,
    task: str = "auto",
    role_overrides: dict[str, str] | None = None,
    validation: bool = True,
    cv_folds: int = 5,
    seed: int = 42,
    baseline_features: list[str] | None = None,
    diversity: bool = True,
    max_candidates: int = 200,
    output_dir: str | Path = "outputs/vizjudge-report",
    top_k: int = 8,
    render: bool = True,
) -> dict[str, Any]:
    """Load and analyze a local dataset."""
    source = Path(path).expanduser().resolve()
    frame = load_dataset(source)
    return analyze_frame(
        frame,
        target=target,
        task=task,
        role_overrides=role_overrides,
        validation=validation,
        cv_folds=cv_folds,
        seed=seed,
        baseline_features=baseline_features,
        diversity=diversity,
        max_candidates=max_candidates,
        output_dir=output_dir,
        top_k=top_k,
        render=render,
        source=str(source),
    )


def markdown_summary(report: dict[str, Any]) -> str:
    """Summarize a report as Markdown for summary.md."""
    lines = [
        "# VizJudge report",
        "",
        f"- Version: {report['vizjudge_version']} (schema {report['schema_version']})",
        f"- Source: `{report['source'] or 'in-memory DataFrame'}`",
        f"- Target: `{report['target'] or 'not specified'}`",
        (
            f"- Shape: {report['profile']['row_count']} rows × "
            f"{report['profile']['column_count']} columns"
        ),
        f"- Task: {report['task'] or 'unsupervised'}",
        f"- Candidate charts judged: {report['candidate_count']}",
        "",
        "## Predictive validation",
        "",
        f"- Status: {report['predictive_validation']['status']}",
        f"- Reason: {report['predictive_validation']['reason'] or 'completed'}",
        f"- Metric: {report['predictive_validation']['metric']}",
        f"- Folds: {report['configuration']['cv_folds']}; seed: {report['configuration']['seed']}",
        *(f"- {warning}" for warning in report["predictive_validation"]["warnings"]),
        "",
        "## Highest-value charts",
        "",
    ]
    if not report["ranked_charts"]:
        lines.append("No eligible chart candidates were generated.")
    for index, chart in enumerate(report["ranked_charts"], start=1):
        lines.extend(
            [
                f"### {index}. {chart['title']} — {chart['score']}/100",
                "",
                *(f"- {reason}" for reason in chart["reasons"]),
                *(f"- Observation: {item}" for item in chart["observations"]),
                *(f"- Warning: {item}" for item in chart["warnings"]),
                f"- Selection score: {chart['selection_score']} "
                f"(diversity penalty: {chart['selection_penalty']})",
                *(f"- Selection: {item}" for item in chart["selection_reasons"]),
                "",
            ]
        )
        if chart["image"]:
            lines.extend([f"![{chart['title']}]({chart['image']})", ""])
    lines.extend(["## Recommended next steps", ""])
    for category, items in report["recommendations"].items():
        lines.extend([f"### {category.replace('_', ' ').title()}", ""])
        lines.extend(f"- {item}" for item in items)
        lines.append("")
    return "\n".join(lines).rstrip() + "\n"
