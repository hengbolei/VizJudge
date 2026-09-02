import json

import pandas as pd

from vizjudge.core.report import analyze_dataset


def test_report_pipeline_writes_structured_outputs(tmp_path) -> None:
    data_path = tmp_path / "data.csv"
    output_path = tmp_path / "report"
    frame = pd.DataFrame(
        {
            "age": [20.0 + index for index in range(80)],
            "segment": ["a", "b"] * 40,
            "target": [0, 1] * 40,
        }
    )
    frame.to_csv(data_path, index=False)

    report = analyze_dataset(
        data_path, target="target", output_dir=output_path, top_k=3, render=True
    )

    assert report["candidate_count"] >= 2
    assert len(report["ranked_charts"]) <= 3
    assert (output_path / "report.json").is_file()
    assert (output_path / "summary.md").is_file()
    assert list(output_path.glob("*.png"))
    serialized = json.loads((output_path / "report.json").read_text(encoding="utf-8"))
    assert serialized["schema_version"] == "0.1"
