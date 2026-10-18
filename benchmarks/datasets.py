"""Public tabular datasets used by the selection benchmark.

Every loader returns ``(frame, target, task)`` with categorical columns as plain
``object`` strings, so every selector sees identical inputs. Downloads are cached
under ``benchmarks/.cache`` (git-ignored); nothing large is committed.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

CACHE = Path(__file__).resolve().parent / ".cache"


@dataclass(frozen=True)
class Dataset:
    name: str
    task: str
    source: str
    licence: str
    loader: Callable[[], tuple[pd.DataFrame, str]]
    notes: str = ""


def _openml(data_id: int) -> pd.DataFrame:
    from sklearn.datasets import fetch_openml

    bunch = fetch_openml(data_id=data_id, as_frame=True, data_home=str(CACHE), parser="auto")
    return bunch.frame.copy()


def _normalize(frame: pd.DataFrame, target: str, task: str) -> pd.DataFrame:
    frame = frame.copy()
    for name in frame.columns:
        series = frame[name]
        if not pd.api.types.is_numeric_dtype(series) or pd.api.types.is_bool_dtype(series):
            frame[name] = series.astype(object).map(
                lambda value: None if pd.isna(value) else str(value)
            )
    if task == "classification":
        frame[target] = frame[target].astype(str)
    else:
        frame[target] = pd.to_numeric(frame[target]).astype(float)
    frame = frame.loc[frame[target].notna()].reset_index(drop=True)
    return frame


def _adult() -> tuple[pd.DataFrame, str]:
    return _openml(1590), "class"


BANK_COLUMNS = [
    "age", "job", "marital", "education", "default", "balance", "housing", "loan",
    "contact", "day", "month", "duration", "campaign", "pdays", "previous", "poutcome",
]  # fmt: skip


def _bank() -> tuple[pd.DataFrame, str]:
    frame = _openml(1461)
    frame.columns = [*BANK_COLUMNS, "y"]
    # UCI notes `duration` is only known after the call ends: direct target leakage.
    return frame.drop(columns=["duration"]), "y"


def _credit_g() -> tuple[pd.DataFrame, str]:
    return _openml(31), "class"


def _titanic() -> tuple[pd.DataFrame, str]:
    # boat/body are recorded after the outcome (leakage); name is a unique free-text label.
    return _openml(40945).drop(columns=["boat", "body", "name"]), "survived"


def _breast_cancer() -> tuple[pd.DataFrame, str]:
    from sklearn.datasets import load_breast_cancer

    frame = load_breast_cancer(as_frame=True).frame
    frame["target"] = frame["target"].map({0: "malignant", 1: "benign"})
    return frame, "target"


def _california() -> tuple[pd.DataFrame, str]:
    from sklearn.datasets import fetch_california_housing

    return fetch_california_housing(as_frame=True, data_home=str(CACHE)).frame, "MedHouseVal"


def _diabetes() -> tuple[pd.DataFrame, str]:
    from sklearn.datasets import load_diabetes

    return load_diabetes(as_frame=True, scaled=False).frame, "target"


def _wine_red() -> tuple[pd.DataFrame, str]:
    frame = _openml(40691).rename(columns={"class": "quality"})
    frame["quality"] = pd.to_numeric(frame["quality"].astype(str))
    return frame, "quality"


def _abalone() -> tuple[pd.DataFrame, str]:
    return _openml(183).rename(columns={"Class_number_of_rings": "rings"}), "rings"


def _bike() -> tuple[pd.DataFrame, str]:
    return _openml(42712), "count"


DATASETS: dict[str, Dataset] = {
    item.name: item
    for item in [
        Dataset("adult", "classification", "OpenML 1590", "Public (UCI, CC BY 4.0)", _adult),
        Dataset(
            "bank_marketing", "classification", "OpenML 1461", "Public (UCI, CC BY 4.0)", _bank,
            "`duration` dropped (post-outcome leakage).",
        ),
        Dataset("credit_g", "classification", "OpenML 31", "Public (UCI, CC BY 4.0)", _credit_g),
        Dataset(
            "titanic", "classification", "OpenML 40945", "Public", _titanic,
            "`boat`, `body` (leakage) and `name` dropped.",
        ),
        Dataset(
            "breast_cancer", "classification", "scikit-learn", "UCI, CC BY 4.0", _breast_cancer
        ),
        Dataset("california_housing", "regression", "scikit-learn", "StatLib", _california),
        Dataset("diabetes", "regression", "scikit-learn", "BSD (scikit-learn bundle)", _diabetes),
        Dataset("wine_quality_red", "regression", "OpenML 40691", "UCI, CC BY 4.0", _wine_red),
        Dataset("abalone", "regression", "OpenML 183", "UCI, CC BY 4.0", _abalone),
        Dataset("bike_sharing", "regression", "OpenML 42712", "UCI, CC BY 4.0", _bike),
    ]
}  # fmt: skip


def load(name: str) -> tuple[pd.DataFrame, str, str]:
    spec = DATASETS[name]
    frame, target = spec.loader()
    return _normalize(frame, target, spec.task), target, spec.task
