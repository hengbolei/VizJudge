import pandas as pd

from vizjudge.core.type_infer import infer_types


def test_infer_practical_roles() -> None:
    frame = pd.DataFrame(
        {
            "user_id": range(100),
            "amount": [float(index) / 3 for index in range(100)],
            "segment": ["a", "b"] * 50,
            "constant": [1] * 100,
            "event_date": pd.date_range("2025-01-01", periods=100),
        }
    )

    roles = infer_types(frame)

    assert roles == {
        "user_id": "identifier",
        "amount": "numeric",
        "segment": "categorical",
        "constant": "constant",
        "event_date": "datetime",
    }
