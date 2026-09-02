"""Expose VizJudge through the stable MCP Python SDK v2."""

from __future__ import annotations

from typing import Any

from mcp.server import MCPServer

from vizjudge.core.report import analyze_dataset as run_analysis

server = MCPServer(
    "VizJudge",
    instructions=(
        "Use analyze_dataset to rank high-value visual evidence in local tabular data and "
        "translate it into data-quality, feature-engineering, and modeling actions."
    ),
)


@server.tool()
def analyze_dataset(
    path: str,
    target: str | None = None,
    output_dir: str = "outputs/vizjudge-report",
    top_k: int = 8,
    render: bool = True,
) -> dict[str, Any]:
    """Judge useful charts in a local dataset and recommend the next ML actions."""
    return run_analysis(
        path,
        target=target,
        output_dir=output_dir,
        top_k=top_k,
        render=render,
    )


def main() -> None:
    """Run the VizJudge MCP server over stdio."""
    server.run()


if __name__ == "__main__":
    main()
