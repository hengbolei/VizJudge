"""Expose VizJudge through the stable MCP Python SDK v2."""

from __future__ import annotations

import base64
import functools
import hashlib
import json
from collections.abc import Callable
from dataclasses import asdict
from pathlib import Path
from typing import Annotated, Any

import pydantic_core
from mcp.server import MCPServer
from mcp.server.mcpserver.exceptions import ToolError
from mcp.types import CallToolResult, ImageContent, TextContent

from vizjudge.agent import ExecutionBudget
from vizjudge.agent import run_agent as run_agent_analysis
from vizjudge.agent.judge import chart_key, judge_report
from vizjudge.agent.runner import BudgetExceeded, Executor, refine_report
from vizjudge.agent.tools import describe_chart
from vizjudge.core.report import analyze_dataset as run_analysis
from vizjudge.core.validation import validate_top_k
from vizjudge.recommendations.data_quality import recommend_data_quality
from vizjudge.scoring.predictive import validate_options

server = MCPServer(
    "VizJudge",
    instructions=(
        "VizJudge supplies deterministic statistical evidence; you (the host model) are the "
        "Judge. Compose profile_dataset -> generate_candidates -> validate_features -> "
        "select_charts -> render_charts, cite the returned evidence IDs, and never raise "
        "statistical priority or confidence. Pass the same target, task, role_overrides, seed "
        "and budget.max_rows to every call so chart and evidence IDs stay consistent. "
        "analyze_dataset and run_agent remain available as one-shot pipelines."
    ),
)


def _reported(tool: Callable[..., Any]) -> Callable[..., Any]:
    """Surface anticipated failures to the host model instead of a generic crash."""

    @functools.wraps(tool)
    def wrapper(*args: Any, **kwargs: Any) -> Any:
        try:
            return tool(*args, **kwargs)
        except (ValueError, TypeError, OSError, BudgetExceeded) as exc:
            raise ToolError(str(exc)) from exc

    return wrapper


# Inline PNGs let hosts (Claude, DeepSeek Harness attachments) look at what they judge.
# Caps keep a tool result well inside typical MCP message and context limits.
MAX_IMAGE_BLOCKS = 8
MAX_IMAGE_BYTES = 1_000_000
MAX_TOTAL_IMAGE_BYTES = 4_000_000

ToolResult = Annotated[CallToolResult, dict[str, Any]]


def _with_images(
    result: dict[str, Any], images: list[tuple[str, str, Path]], include: bool
) -> CallToolResult:
    """Return the structured result plus capped PNG image blocks for (id, title, path)."""
    blocks: list[TextContent | ImageContent] = []
    attached, omitted = [], []
    total = 0
    for chart_id, title, path in images if include else []:
        try:
            size = path.stat().st_size
        except OSError:
            omitted.append({"chart_id": chart_id, "reason": "file_missing"})
            continue
        if len(attached) >= MAX_IMAGE_BLOCKS:
            reason = "max_image_blocks"
        elif size > MAX_IMAGE_BYTES:
            reason = "max_image_bytes"
        elif total + size > MAX_TOTAL_IMAGE_BYTES:
            reason = "max_total_image_bytes"
        else:
            reason = None
        if reason:
            omitted.append({"chart_id": chart_id, "reason": reason})
            continue
        total += size
        attached.append(chart_id)
        blocks.append(TextContent(type="text", text=f"Chart {chart_id}: {title}"))
        blocks.append(
            ImageContent(
                type="image",
                data=base64.b64encode(path.read_bytes()).decode("ascii"),
                mime_type="image/png",
            )
        )
    if images:
        result["image_content"] = {
            "included": include,
            "attached": attached,
            "omitted": omitted,
        }
    structured = json.loads(pydantic_core.to_json(result, fallback=str))
    text = pydantic_core.to_json(result, fallback=str, indent=2).decode()
    return CallToolResult(
        content=[TextContent(type="text", text=text), *blocks], structured_content=structured
    )


def _report_images(report: dict[str, Any], output_dir: Path) -> list[tuple[str, str, Path]]:
    return [
        (chart_key(chart), chart["title"], output_dir / chart["image"])
        for chart in report.get("ranked_charts", [])
        if chart.get("image")
    ]


def _limits(budget: dict[str, Any] | None) -> ExecutionBudget:
    if budget is not None and not isinstance(budget, dict):
        raise ValueError("budget must be an object of ExecutionBudget fields")
    limits = ExecutionBudget(**(budget or {}))
    limits.validate()
    return limits


def _execute(
    tool: str,
    path: str,
    limits: ExecutionBudget,
    reason: str,
    *,
    seed: int,
    **request: Any,
) -> dict[str, Any]:
    """Run one allowlisted tool in a bounded worker and attach its provenance."""
    validate_options(5, seed, None)
    source = Path(path).expanduser().resolve()
    if not source.is_file():
        raise FileNotFoundError(source)
    if source.stat().st_size > limits.max_input_bytes:
        raise ValueError("Dataset exceeds max_input_bytes")
    digest = hashlib.sha256(source.read_bytes()).hexdigest()
    executor = Executor(limits)
    result = executor.call(
        tool,
        {
            "path": str(source),
            "dataset_sha256": digest,
            "max_input_bytes": limits.max_input_bytes,
            "max_rows": limits.max_rows,
            "seed": seed,
            **request,
        },
        reason,
    )
    result["execution"] = {
        "dataset_sha256": digest,
        "budget": asdict(limits),
        "trace": executor.trace,
    }
    return result


@server.tool()
@_reported
def profile_dataset(
    path: str,
    role_overrides: dict[str, str] | None = None,
    seed: int = 42,
    budget: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Profile column roles, missingness and duplicates; suggest data-quality actions.

    Use the roles and column names to choose the target and role_overrides and to
    spot columns that may leak the target. Large files are row-sampled per budget.max_rows.
    """
    result = _execute(
        "profile",
        path,
        _limits(budget),
        "Profile dataset quality for the host Judge.",
        seed=seed,
        role_overrides=role_overrides,
    )
    result["data_quality"] = recommend_data_quality(result["profile"])
    return result


@server.tool()
@_reported
def generate_candidates(
    path: str,
    target: str | None = None,
    task: str = "auto",
    role_overrides: dict[str, str] | None = None,
    max_candidates: int | None = None,
    seed: int = 42,
    budget: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """List every candidate chart with heuristic score, warnings, chart_id and evidence IDs.

    No predictive validation or diversity selection is applied; see validate_features
    and select_charts.
    """
    limits = _limits(budget)
    if max_candidates is not None:
        validate_top_k(max_candidates)
    return _execute(
        "candidates",
        path,
        limits,
        "Generate and rule-score candidate charts.",
        seed=seed,
        options={
            "target": target,
            "task": task,
            "role_overrides": role_overrides,
            "max_candidates": min(max_candidates or limits.max_candidates, limits.max_candidates),
        },
    )


@server.tool()
@_reported
def validate_features(
    path: str,
    target: str,
    features: list[str],
    task: str = "auto",
    role_overrides: dict[str, str] | None = None,
    baseline_features: list[str] | None = None,
    cv_folds: int = 5,
    seed: int = 42,
    budget: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Measure cross-validated incremental gain of each named feature over a ridge baseline.

    Positive incremental_gain means improvement for both balanced accuracy and RMSE.
    Each result carries the chart_id and evidence IDs of the feature's target chart.
    """
    validate_options(cv_folds, seed, baseline_features)
    return _execute(
        "validate",
        path,
        _limits(budget),
        "Validate incremental predictive value of named features.",
        seed=seed,
        options={
            "target": target,
            "features": features,
            "task": task,
            "role_overrides": role_overrides,
            "baseline_features": baseline_features,
            "cv_folds": cv_folds,
            "seed": seed,
        },
    )


@server.tool()
@_reported
def select_charts(
    path: str,
    target: str | None = None,
    task: str = "auto",
    role_overrides: dict[str, str] | None = None,
    top_k: int = 6,
    diversity: bool = True,
    validation: bool = True,
    cv_folds: int = 5,
    seed: int = 42,
    baseline_features: list[str] | None = None,
    reviews: list[dict[str, Any]] | None = None,
    refinement: str | None = None,
    refine: bool = True,
    budget: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Select a small, diverse, validated chart set and apply optional host verdicts.

    Call once without reviews to obtain chart_ids and evidence IDs, then optionally
    again with reviews for any subset of charts: [{chart_id, evidence_ids, priority
    0..100, verdict keep|defer, rationale, next_step}], plus refinement (none,
    prefer_validated, increase_diversity, reduce_charts). Host priority can only lower
    combined_priority, and a statistical defer cannot be reversed.
    """
    limits = _limits(budget)
    validate_top_k(top_k)
    validate_options(cv_folds, seed, baseline_features)
    if any(not isinstance(value, bool) for value in (diversity, validation, refine)):
        raise ValueError("diversity, validation and refine must be booleans")
    if reviews is not None and not isinstance(reviews, list):
        raise ValueError("reviews must be a list of review objects")
    report = _execute(
        "analyze",
        path,
        limits,
        "Select diverse, validated charts without rendering.",
        seed=seed,
        options={
            "target": target,
            "task": task,
            "role_overrides": role_overrides,
            "validation": validation,
            "cv_folds": cv_folds,
            "seed": seed,
            "baseline_features": baseline_features,
            "diversity": diversity,
            "top_k": min(top_k, limits.max_candidates),
            "max_candidates": limits.max_candidates,
            "render": False,
        },
    )
    host = None
    if reviews is not None or refinement is not None:
        response = {"reviews": reviews or [], "refinement": refinement or "none"}

        def host(context: dict[str, Any]) -> dict[str, Any]:
            return response

    judgment = judge_report(report, host, partial=True)
    if host is not None:
        judgment["mode"] = "statistical_and_host"
    before = [describe_chart(chart)["chart_id"] for chart in report["ranked_charts"]]
    applied = refine and judgment["refinement"] != "none"
    if applied:
        report = refine_report(report, judgment, judgment["refinement"])
    by_id = {review["chart_id"]: review for review in judgment["reviews"]}
    charts = []
    for chart in report["ranked_charts"]:
        described = describe_chart(chart)
        described["judgment"] = by_id[described["chart_id"]]
        charts.append(described)
    report["ranked_charts"] = charts
    report["selection"] = {
        "judgment_mode": judgment["mode"],
        "refinement": judgment["refinement"],
        "refinement_applied": applied,
        "before": before,
        "after": [chart["chart_id"] for chart in charts],
    }
    return report


@server.tool()
@_reported
def render_charts(
    path: str,
    chart_ids: list[str],
    output_dir: str = "outputs/vizjudge-charts",
    target: str | None = None,
    task: str = "auto",
    role_overrides: dict[str, str] | None = None,
    seed: int = 42,
    budget: dict[str, Any] | None = None,
    include_images: bool = True,
) -> ToolResult:
    """Render chart IDs, in the given order, to PNG files up to budget.max_rendered_charts.

    Pass the same target, task and role_overrides that produced the chart IDs. Unless
    include_images is false, the PNGs are also returned as image content (at most 8,
    1 MB each, 4 MB total) after the JSON result, each preceded by its chart_id.
    """
    limits = _limits(budget)
    if not isinstance(include_images, bool):
        raise ValueError("include_images must be a boolean")
    result = _execute(
        "render_ids",
        path,
        limits,
        "Render host-selected charts within the image budget.",
        seed=seed,
        output_dir=str(Path(output_dir).expanduser().resolve()),
        max_charts=limits.max_rendered_charts,
        options={
            "chart_ids": chart_ids,
            "target": target,
            "task": task,
            "role_overrides": role_overrides,
            "max_candidates": limits.max_candidates,
        },
    )
    images = [(item["chart_id"], item["title"], Path(item["path"])) for item in result["rendered"]]
    return _with_images(result, images, include_images)


@server.tool()
def analyze_dataset(
    path: str,
    target: str | None = None,
    output_dir: str = "outputs/vizjudge-report",
    top_k: int = 8,
    render: bool = True,
    task: str = "auto",
    role_overrides: dict[str, str] | None = None,
    validation: bool = True,
    cv_folds: int = 5,
    seed: int = 42,
    baseline_features: list[str] | None = None,
    diversity: bool = True,
    include_images: bool = True,
) -> ToolResult:
    """Judge useful charts in a local dataset and recommend the next ML actions.

    When rendering, the PNGs are also returned as capped image content unless
    include_images is false.
    """
    report = run_analysis(
        path,
        target=target,
        task=task,
        role_overrides=role_overrides,
        validation=validation,
        cv_folds=cv_folds,
        seed=seed,
        baseline_features=baseline_features,
        diversity=diversity,
        output_dir=output_dir,
        top_k=top_k,
        render=render,
    )
    images = _report_images(report, Path(output_dir).expanduser().resolve()) if render else []
    return _with_images(report, images, include_images is True)


@server.tool()
def run_agent(
    path: str,
    target: str | None = None,
    output_dir: str = "outputs/vizjudge-agent",
    task: str = "auto",
    role_overrides: dict[str, str] | None = None,
    top_k: int = 6,
    render: bool = True,
    validation: bool = True,
    cv_folds: int = 5,
    seed: int = 42,
    baseline_features: list[str] | None = None,
    budget: dict[str, Any] | None = None,
    judge: dict[str, str] | None = None,
    include_images: bool = True,
) -> ToolResult:
    """Plan, execute bounded Python tools, judge cited evidence, and refine charts.

    Judge defaults to statistical-only; file replays or HTTP calls must be explicit.
    When rendering, the PNGs are also returned as capped image content unless
    include_images is false.
    """
    report = run_agent_analysis(
        path,
        target=target,
        output_dir=output_dir,
        task=task,
        role_overrides=role_overrides,
        top_k=top_k,
        render=render,
        validation=validation,
        cv_folds=cv_folds,
        seed=seed,
        baseline_features=baseline_features,
        budget=ExecutionBudget(**(budget or {})),
        judge=judge,
    )
    images = _report_images(report, Path(output_dir).expanduser().resolve()) if render else []
    return _with_images(report, images, include_images is True)


def main() -> None:
    """Run the VizJudge MCP server over stdio."""
    server.run()


if __name__ == "__main__":
    main()
