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

    asyncio.run(call_tool())
