"""Run the complete VizJudge workflow and serialize its evidence."""

from __future__ import annotations

import json
import re
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pandas as pd

from vizjudge.charts.categorical import render_categorical
from vizjudge.charts.generator import ChartCandidate, generate_candidates
from vizjudge.charts.numeric import render_numeric
from vizjudge.charts.target import render_target
from vizjudge.core.loader import load_dataset
from vizjudge.core.profiler import profile_dataframe
from vizjudge.core.type_infer import infer_types
from vizjudge.recommendations.data_quality import recommend_data_quality
from vizjudge.recommendations.feature_engineering import recommend_features
from vizjudge.recommendations.modeling import recommend_models
from vizjudge.scoring.ranker import rank_candidates


def _safe_filename(value: str) -> str:
    cleaned = re.sub(r"[^a-zA-Z0-9._-]+", "-", value).strip("-_").lower()
    return cleaned[:100] or "chart"


def _render(frame: pd.DataFrame, candidate: ChartCandidate, path: Path) -> None:
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
    output_dir: str | Path | None = None,
    top_k: int = 8,
    render: bool = True,
    source: str | None = None,
) -> dict[str, Any]:
    """Analyze an in-memory DataFrame and optionally write a report directory."""
    roles = infer_types(frame)
    if target is not None and target not in frame.columns:
        raise ValueError(f"Target column not found: {target}")
    candidates = generate_candidates(frame, roles, target)
    ranked = rank_candidates(frame, candidates, top_k=top_k) if candidates else []
    profile = profile_dataframe(frame, roles)

    destination = Path(output_dir).expanduser().resolve() if output_dir else None
    if render and destination is None:
        raise ValueError("output_dir is required when render=True")
    if destination is not None:
        destination.mkdir(parents=True, exist_ok=True)
    if render and destination is not None:
        for rank, candidate in enumerate(ranked, start=1):
            filename = f"{rank:02d}-{_safe_filename(candidate.key)}.png"
            _render(frame, candidate, destination / filename)
            candidate.image = filename

    recommendations = {
        "data_quality": recommend_data_quality(profile),
        "feature_engineering": recommend_features(profile, ranked),
        "modeling": recommend_models(target, roles, ranked),
    }
    report: dict[str, Any] = {
        "schema_version": "0.1",
        "generated_at": datetime.now(UTC).isoformat(),
        "source": source,
        "target": target,
        "roles": roles,
        "profile": profile,
        "candidate_count": len(candidates),
        "ranked_charts": [candidate.to_dict() for candidate in ranked],
        "recommendations": recommendations,
    }
    if destination is not None:
        (destination / "report.json").write_text(
            json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8"
        )
        (destination / "summary.md").write_text(_markdown_summary(report), encoding="utf-8")
    return report


def analyze_dataset(
    path: str | Path,
    *,
    target: str | None = None,
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
        output_dir=output_dir,
        top_k=top_k,
        render=render,
        source=str(source),
    )


def _markdown_summary(report: dict[str, Any]) -> str:
    lines = [
        "# VizJudge report",
        "",
        f"- Source: `{report['source'] or 'in-memory DataFrame'}`",
        f"- Target: `{report['target'] or 'not specified'}`",
        (
            f"- Shape: {report['profile']['row_count']} rows × "
            f"{report['profile']['column_count']} columns"
        ),
        f"- Candidate charts judged: {report['candidate_count']}",
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
                "",
            ]
        )
    lines.extend(["## Recommended next steps", ""])
    for category, items in report["recommendations"].items():
        lines.extend([f"### {category.replace('_', ' ').title()}", ""])
        lines.extend(f"- {item}" for item in items)
        lines.append("")
    return "\n".join(lines).rstrip() + "\n"
