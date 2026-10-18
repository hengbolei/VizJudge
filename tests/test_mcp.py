import asyncio
import base64
import json
from pathlib import Path

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
            assert result.structured_content["schema_version"] == "0.5"

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


def _leaky_dataset(tmp_path):
    rows = 120
    path = tmp_path / "leaky.csv"
    pd.DataFrame(
        {
            "signal": [float(i % 2) + (i % 7) / 10 for i in range(rows)],
            "noise": [float((i * 37) % 11) for i in range(rows)],
            "segment": ["a", "b", "c"] * (rows // 3),
            "target": [i % 2 for i in range(rows)],
        }
    ).to_csv(path, index=False)
    return path


def _call(name: str, arguments: dict):
    async def call():
        async with Client(server) as client:
            return await client.call_tool(name, arguments)

    return asyncio.run(call())


def test_mcp_lists_fine_grained_tools() -> None:
    async def names():
        async with Client(server) as client:
            return {tool.name for tool in (await client.list_tools()).tools}

    assert {
        "analyze_dataset",
        "run_agent",
        "profile_dataset",
        "generate_candidates",
        "validate_features",
        "select_charts",
        "render_charts",
    } <= asyncio.run(names())


def test_mcp_profile_and_candidates_share_evidence_ids(tmp_path) -> None:
    path = _leaky_dataset(tmp_path)
    profile = _call("profile_dataset", {"path": str(path), "budget": {"max_rows": 60}})
    assert not profile.is_error
    content = profile.structured_content
    assert content["sampling"] == {
        "original_rows": 120,
        "analyzed_rows": 60,
        "sampled": True,
        "seed": 42,
    }
    assert content["execution"]["trace"][0]["tool"] == "profile"
    assert len(content["execution"]["dataset_sha256"]) == 64
    assert isinstance(content["data_quality"], list)

    result = _call(
        "generate_candidates",
        {"path": str(path), "target": "target", "max_candidates": 3},
    )
    assert not result.is_error
    candidates = result.structured_content["candidates"]
    assert result.structured_content["task"] == "classification"
    assert len(candidates) == 3
    assert [item["score"] for item in candidates] == sorted(
        (item["score"] for item in candidates), reverse=True
    )
    for item in candidates:
        assert json.loads(item["chart_id"])[0] == item["kind"]
        assert all(entry["id"].startswith("C") for entry in item["evidence"])
        assert "selection_score" not in item

    selection = _call(
        "select_charts", {"path": str(path), "target": "target", "validation": False}
    ).structured_content
    by_id = {chart["chart_id"]: chart for chart in selection["ranked_charts"]}
    shared = set(by_id) & {item["chart_id"] for item in candidates}
    assert shared
    for item in candidates:
        if item["chart_id"] in by_id:
            assert item["evidence"] == by_id[item["chart_id"]]["evidence"]


def test_mcp_validate_features_reports_gain_with_evidence(tmp_path) -> None:
    path = _leaky_dataset(tmp_path)
    result = _call(
        "validate_features",
        {"path": str(path), "target": "target", "features": ["signal", "segment"]},
    )
    assert not result.is_error
    content = result.structured_content
    assert content["validation"]["status"] == "completed"
    signal = content["features"][0]
    assert signal["feature"] == "signal"
    assert signal["incremental_gain"] > 0
    ids = {item["id"] for item in signal["evidence"]}
    prefix = next(iter(ids)).split(".")[0]
    assert f"{prefix}.incremental_gain" in ids

    selection = _call("select_charts", {"path": str(path), "target": "target"}).structured_content
    chart = next(c for c in selection["ranked_charts"] if c["chart_id"] == signal["chart_id"])
    assert chart["predictive"]["incremental_gain"] == signal["incremental_gain"]

    for features in (["target"], ["missing"], []):
        bad = _call(
            "validate_features",
            {"path": str(path), "target": "target", "features": features},
        )
        assert bad.is_error


def test_mcp_select_charts_host_review_only_lowers_priority(tmp_path) -> None:
    path = _leaky_dataset(tmp_path)
    first = _call(
        "select_charts", {"path": str(path), "target": "target", "refine": False}
    ).structured_content
    assert first["selection"]["judgment_mode"] == "statistical_only"
    chart = first["ranked_charts"][0]
    statistical = chart["judgment"]["statistical_priority"]
    review = {
        "chart_id": chart["chart_id"],
        "evidence_ids": [chart["evidence"][0]["id"]],
        "priority": 100,
        "verdict": "defer",
        "rationale": "Column semantics suggest post-outcome information.",
        "next_step": "Drop the column and re-validate on a grouped split.",
    }
    second = _call(
        "select_charts",
        {"path": str(path), "target": "target", "reviews": [review], "refine": False},
    ).structured_content
    assert second["selection"]["judgment_mode"] == "statistical_and_host"
    reviewed = next(c for c in second["ranked_charts"] if c["chart_id"] == chart["chart_id"])
    assert reviewed["judgment"]["combined_priority"] == statistical
    assert reviewed["judgment"]["verdict"] == "defer"
    assert second["selection"]["refinement"] == "prefer_validated"

    refined = _call(
        "select_charts",
        {"path": str(path), "target": "target", "reviews": [review]},
    ).structured_content
    assert refined["selection"]["refinement_applied"]
    assert chart["chart_id"] not in refined["selection"]["after"]

    invented = {**review, "evidence_ids": ["C000000000000.made_up"]}
    bad = _call("select_charts", {"path": str(path), "target": "target", "reviews": [invented]})
    assert bad.is_error
    assert "cite existing evidence" in bad.content[0].text


def test_mcp_render_charts_respects_image_budget(tmp_path) -> None:
    path = _leaky_dataset(tmp_path)
    selection = _call(
        "select_charts", {"path": str(path), "target": "target", "validation": False}
    ).structured_content
    ids = selection["selection"]["after"][:3]
    result = _call(
        "render_charts",
        {
            "path": str(path),
            "target": "target",
            "chart_ids": ids,
            "output_dir": str(tmp_path / "charts"),
            "budget": {"max_rendered_charts": 2},
        },
    )
    assert not result.is_error
    content = result.structured_content
    assert [item["chart_id"] for item in content["rendered"]] == ids[:2]
    assert content["skipped_by_budget"] == ids[2:]
    assert all((tmp_path / "charts" / item["image"]).is_file() for item in content["rendered"])

    unknown = json.dumps(["scatter", "signal", "noise", None])
    bad = _call(
        "render_charts",
        {"path": str(path), "target": "target", "chart_ids": [unknown]},
    )
    assert bad.is_error
    assert "were not generated" in bad.content[0].text


def test_mcp_fine_grained_tools_enforce_budget(tmp_path) -> None:
    path = _leaky_dataset(tmp_path)
    result = _call(
        "generate_candidates",
        {"path": str(path), "budget": {"max_input_bytes": 10}},
    )
    assert result.is_error
    result = _call("profile_dataset", {"path": str(path), "budget": {"max_rows": 0}})
    assert result.is_error


def test_mcp_render_charts_returns_png_image_content(tmp_path) -> None:
    path = _leaky_dataset(tmp_path)
    selection = _call(
        "select_charts", {"path": str(path), "target": "target", "validation": False}
    ).structured_content
    ids = selection["selection"]["after"][:2]
    arguments = {
        "path": str(path),
        "target": "target",
        "chart_ids": ids,
        "output_dir": str(tmp_path / "charts"),
    }
    result = _call("render_charts", arguments)
    assert not result.is_error
    content = result.structured_content
    # The structured result is unchanged apart from the additive image summary.
    assert json.loads(result.content[0].text) == content
    assert [item["chart_id"] for item in content["rendered"]] == ids
    assert content["image_content"] == {"included": True, "attached": ids, "omitted": []}
    images = [block for block in result.content if block.type == "image"]
    captions = [block.text for block in result.content[1:] if block.type == "text"]
    assert len(images) == 2
    assert all(block.mime_type == "image/png" for block in images)
    for block, item in zip(images, content["rendered"], strict=True):
        assert base64.b64decode(block.data) == Path(item["path"]).read_bytes()
    assert captions == [
        f"Chart {item['chart_id']}: {item['title']}" for item in content["rendered"]
    ]
    trace = content["execution"]["trace"]
    assert trace[0]["cache"]["status"] == "hit"  # select_charts already cached the frame.

    plain = _call("render_charts", {**arguments, "include_images": False})
    assert [block.type for block in plain.content] == ["text"]
    assert plain.structured_content["image_content"]["attached"] == []


def test_mcp_image_content_respects_size_caps(tmp_path, monkeypatch) -> None:
    from vizjudge.mcp import server as module

    path = _leaky_dataset(tmp_path)
    selection = _call(
        "select_charts", {"path": str(path), "target": "target", "validation": False}
    ).structured_content
    monkeypatch.setattr(module, "MAX_IMAGE_BYTES", 10)
    result = _call(
        "render_charts",
        {
            "path": str(path),
            "target": "target",
            "chart_ids": selection["selection"]["after"][:2],
            "output_dir": str(tmp_path / "charts"),
        },
    )
    assert not result.is_error
    assert not [block for block in result.content if block.type == "image"]
    omitted = result.structured_content["image_content"]["omitted"]
    assert {item["reason"] for item in omitted} == {"max_image_bytes"}


def test_mcp_agent_returns_rendered_images(tmp_path) -> None:
    path = _leaky_dataset(tmp_path)
    result = _call(
        "run_agent",
        {
            "path": str(path),
            "target": "target",
            "output_dir": str(tmp_path / "agent"),
            "validation": False,
            "budget": {"max_rendered_charts": 2},
        },
    )
    assert not result.is_error
    images = [block for block in result.content if block.type == "image"]
    rendered = [c for c in result.structured_content["ranked_charts"] if c["image"]]
    assert len(images) == len(rendered) == 2
    statuses = [
        event.get("cache", {}).get("status")
        for event in result.structured_content["agent"]["trace"]
    ]
    assert statuses[0] == "miss" and "hit" in statuses
