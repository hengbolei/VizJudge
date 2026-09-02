"""Load supported tabular datasets."""

from pathlib import Path

import pandas as pd

SUPPORTED_SUFFIXES = {".csv", ".json", ".jsonl", ".ndjson", ".parquet"}


def load_dataset(path: str | Path) -> pd.DataFrame:
    """Load a local CSV, JSON/JSONL, or Parquet file into a DataFrame."""
    source = Path(path).expanduser().resolve()
    if not source.is_file():
        raise FileNotFoundError(f"Dataset does not exist or is not a file: {source}")

    suffix = source.suffix.lower()
    if suffix not in SUPPORTED_SUFFIXES:
        allowed = ", ".join(sorted(SUPPORTED_SUFFIXES))
        raise ValueError(f"Unsupported dataset format {suffix!r}; expected one of: {allowed}")

    if suffix == ".csv":
        frame = pd.read_csv(source)
    elif suffix in {".jsonl", ".ndjson"}:
        frame = pd.read_json(source, lines=True)
    elif suffix == ".json":
        frame = pd.read_json(source)
    else:
        frame = pd.read_parquet(source)

    if frame.empty:
        raise ValueError(f"Dataset is empty: {source}")
    if frame.columns.has_duplicates:
        raise ValueError("Dataset contains duplicate column names")
    return frame
