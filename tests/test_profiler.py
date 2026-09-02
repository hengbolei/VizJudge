import pandas as pd

from vizjudge.core.profiler import profile_dataframe


def test_profile_is_compact_and_json_safe() -> None:
    frame = pd.DataFrame({"value": [1.0, 2.0, None], "group": ["a", "a", "b"]})

    profile = profile_dataframe(frame)

    assert profile["row_count"] == 3
    assert profile["column_count"] == 2
    assert profile["total_missing_rate"] == 0.166667
    value = next(column for column in profile["columns"] if column["name"] == "value")
    assert value["missing_count"] == 1
    assert value["statistics"]["median"] == 1.5
