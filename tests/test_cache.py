import os
import time

import pandas as pd
import pytest

from vizjudge.agent import ExecutionBudget
from vizjudge.agent.runner import Executor
from vizjudge.core import cache
from vizjudge.core.cache import load_sampled, prune


def _write(path, rows=50, offset=0):
    pd.DataFrame(
        {"x": [float(i + offset) for i in range(rows)], "target": [i % 2 for i in range(rows)]}
    ).to_csv(path, index=False)
    return path


def test_cache_miss_then_hit_returns_identical_sample(tmp_path):
    path = _write(tmp_path / "data.csv")
    first, rows, meta = load_sampled(path, max_rows=20, seed=7)
    assert meta["status"] == "miss"
    assert rows == 50 and len(first) == 20
    second, rows, meta = load_sampled(path, max_rows=20, seed=7)
    assert meta["status"] == "hit"
    assert rows == 50
    pd.testing.assert_frame_equal(first, second)
    expected = pd.read_csv(path).sample(20, random_state=7)
    pd.testing.assert_frame_equal(second, expected)


def test_cache_key_covers_budget_and_seed(tmp_path):
    path = _write(tmp_path / "data.csv")
    load_sampled(path, max_rows=20, seed=7)
    assert load_sampled(path, max_rows=20, seed=8)[2]["status"] == "miss"
    assert load_sampled(path, max_rows=30, seed=7)[2]["status"] == "miss"
    assert load_sampled(path, max_rows=20, seed=7)[2]["status"] == "hit"


def test_cache_invalidated_when_source_changes(tmp_path):
    path = _write(tmp_path / "data.csv")
    load_sampled(path, max_rows=100, seed=1)
    stat = path.stat()
    _write(path, offset=1000)
    # Restore the mtime so only the content hash can tell the versions apart.
    os.utime(path, ns=(stat.st_atime_ns, stat.st_mtime_ns))
    frame, _, meta = load_sampled(path, max_rows=100, seed=1)
    assert meta["status"] == "miss"
    assert frame["x"].iloc[0] == 1000.0


def test_cache_rejects_source_that_differs_from_expected_digest(tmp_path):
    path = _write(tmp_path / "data.csv")
    with pytest.raises(ValueError, match="changed"):
        load_sampled(path, max_rows=10, seed=1, expected_sha256="0" * 64)


def test_corrupt_entry_is_a_miss_and_is_replaced(tmp_path):
    path = _write(tmp_path / "data.csv")
    _, _, meta = load_sampled(path, max_rows=10, seed=1)
    entries = list(cache.cache_dir().glob("*.pkl"))
    assert len(entries) == 1
    entries[0].write_bytes(b"not a pickle")
    assert load_sampled(path, max_rows=10, seed=1)[2]["status"] == "miss"
    assert load_sampled(path, max_rows=10, seed=1)[2]["status"] == "hit"


def test_cache_can_be_disabled(tmp_path, monkeypatch):
    monkeypatch.setenv("VIZJUDGE_NO_CACHE", "1")
    path = _write(tmp_path / "data.csv")
    assert load_sampled(path, max_rows=10, seed=1)[2]["status"] == "disabled"
    assert load_sampled(path, max_rows=10, seed=1)[2]["status"] == "disabled"


def test_prune_enforces_size_cap_least_recently_used_first(tmp_path):
    directory = tmp_path / "entries"
    directory.mkdir()
    now = time.time()
    for index, name in enumerate(("old", "mid", "new")):
        entry = directory / f"{name}.pkl"
        entry.write_bytes(b"x" * 100)
        os.utime(entry, (now + index, now + index))
    prune(directory, max_bytes=250)
    assert sorted(entry.stem for entry in directory.glob("*.pkl")) == ["mid", "new"]
    stale = directory / "stale.pkl"
    stale.write_bytes(b"x")
    os.utime(stale, (1_000, 1_000))
    prune(directory, max_bytes=10_000)
    assert not stale.exists()


def test_executor_trace_records_cache_hits(tmp_path):
    path = _write(tmp_path / "data.csv")
    executor = Executor(ExecutionBudget())
    request = {"path": str(path), "max_input_bytes": 10_000_000, "max_rows": 30, "seed": 3}
    first = executor.call("profile", request, "first")
    second = executor.call("profile", request, "second")
    assert [event["cache"]["status"] for event in executor.trace] == ["miss", "hit"]
    assert first == second
    assert len(executor.trace[0]["cache"]["dataset_sha256"]) == 64
