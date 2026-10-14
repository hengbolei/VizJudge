import asyncio

import pandas as pd
from mcp import Client

from vizjudge.mcp.server import server


def test_mcp_analyze_dataset_tool(tmp_path) -> None:
    data_path = tmp_path / "mcp-data.csv"
    output_path = tmp_path / "mcp-report"
    pd.DataFrame(
        {
            "feature": [float(index) for index in range(40)],
            "target": [index % 2 for index in range(40)],
        }
    ).to_csv(data_path, index=False)

    async def call_tool() -> None:
        async with Client(server) as client:
            result = await client.call_tool(
                "analyze_dataset",
                {
                    "path": str(data_path),
                    "target": "target",
                    "output_dir": str(output_path),
                    "top_k": 2,
                    "render": False,
                },
            )
            assert not result.is_error
            assert result.structured_content is not None
            assert result.structured_content["schema_version"] == "0.4"

    asyncio.run(call_tool())


def test_mcp_agent_tool_preserves_budget_and_trace(tmp_path) -> None:
    data_path = tmp_path / "data.csv"
    pd.DataFrame({"x": [float(i) for i in range(40)], "target": [0, 1] * 20}).to_csv(
        data_path, index=False
    )

    async def call_tool() -> None:
        async with Client(server) as client:
            result = await client.call_tool(
                "run_agent",
                {
                    "path": str(data_path),
                    "target": "target",
                    "render": False,
                    "output_dir": str(tmp_path / "report"),
                    "budget": {"max_tool_calls": 1},
                },
            )
            assert not result.is_error
            assert result.structured_content["agent"]["status"] == "budget_exhausted"
            assert result.structured_content["agent"]["trace"][0]["tool"] == "profile"

    asyncio.run(call_tool())
