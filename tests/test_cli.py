import json
import tomllib
from pathlib import Path

import pandas as pd
import pytest

from vizjudge import __version__, analyze_frame
from vizjudge.cli import main
from vizjudge.core.loader import load_dataset


def test_versions_and_report_contract_match(capsys):
    metadata = tomllib.loads((Path(__file__).parents[1] / "pyproject.toml").read_text())
    assert metadata["project"]["version"] == __version__ == "0.4.0"
    with pytest.raises(SystemExit) as result:
        main(["--version"])
    assert result.value.code == 0
    assert "0.4.0" in capsys.readouterr().out
    report = analyze_frame(pd.DataFrame({"x": [1.0, 2.0, 3.0]}), render=False)
    assert {
        "schema_version",
        "generated_at",
        "source",
        "target",
        "roles",
        "profile",
        "candidate_count",
        "ranked_charts",
        "recommendations",
    } <= set(report)
    json.dumps(report, allow_nan=False)
    assert report["predictive_validation"]["reason"] == "no_target"


def test_analyze_cli_overrides_and_agent_budget_exit(tmp_path):
    path = tmp_path / "data.csv"
    pd.DataFrame({"x": range(40), "target": [0, 1] * 20}).to_csv(path, index=False)
    output = tmp_path / "analysis"
    assert (
        main(
            [
                "analyze",
                str(path),
                "--target",
                "target",
                "--role",
                "x=numeric",
                "--no-render",
                "--no-validation",
                "--output",
                str(output),
            ]
        )
        == 0
    )
    report = json.loads((output / "report.json").read_text())
    assert report["roles"]["x"] == "numeric"
    assert report["predictive_validation"]["reason"] == "disabled"
    assert (
        main(
            [
                "agent",
                str(path),
                "--target",
                "target",
                "--no-render",
                "--max-tool-calls",
                "1",
                "--output",
                str(tmp_path / "agent"),
            ]
        )
        == 3
    )


def test_csv_duplicate_headers_are_not_silently_renamed(tmp_path):
    path = tmp_path / "duplicate.csv"
    path.write_text("value,value\n1,2\n3,4\n")
    with pytest.raises(ValueError, match="duplicate"):
        load_dataset(path)


@pytest.mark.parametrize(
    "frame",
    [
        pd.DataFrame(),
        pd.DataFrame({"x": [float("inf")]}),
        pd.DataFrame({"x": [(1, 2), (2, 3)]}),
        pd.DataFrame({1: [1, 2]}),
    ],
)
def test_invalid_in_memory_inputs(frame):
    with pytest.raises(ValueError):
        analyze_frame(frame, render=False)
