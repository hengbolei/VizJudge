"""Expose VizJudge through the stable MCP Python SDK v2."""

from __future__ import annotations

from typing import Any

from mcp.server import MCPServer

from vizjudge.agent import ExecutionBudget
from vizjudge.agent import run_agent as run_agent_analysis
from vizjudge.core.report import analyze_dataset as run_analysis

server = MCPServer(
    "VizJudge",
    instructions=(
        "Use analyze_dataset to rank high-value visual evidence in local tabular data and "
        "translate it into data-quality, feature-engineering, and modeling actions. "
        "Use run_agent for bounded planning, Python tool execution, cited judging and refinement."
    ),
)


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
) -> dict[str, Any]:
    """Judge useful charts in a local dataset and recommend the next ML actions."""
    return run_analysis(
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
) -> dict[str, Any]:
    """Plan, execute bounded Python tools, judge cited evidence, and refine charts.

    Judge defaults to statistical-only; file replays or HTTP calls must be explicit.
    """
    return run_agent_analysis(
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


def main() -> None:
    """Run the VizJudge MCP server over stdio."""
    server.run()


if __name__ == "__main__":
    main()
