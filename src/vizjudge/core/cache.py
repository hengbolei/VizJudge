"""Per-user disk cache of loaded and budget-sampled datasets.

Every fine-grained MCP tool runs in a fresh worker process, so without a cache one
profile -> candidates -> validate -> select -> render flow parses the same file five
times. Entries are keyed by the source's SHA-256, size, mtime, the sampling budget
and seed, and the pandas version, so any change to the source or the sample misses.

Entries are pickles written only by this module into the current user's cache
directory; never point ``VIZJUDGE_CACHE_DIR`` at a location other users can write.
"""

from __future__ import annotations

import hashlib
import os
import pickle
import sys
import tempfile
import time
from pathlib import Path
from typing import Any

import pandas as pd

from vizjudge.core.loader import load_dataset

CACHE_FORMAT = 1
DEFAULT_MAX_BYTES = 512 * 1024 * 1024
MAX_AGE_SECONDS = 7 * 24 * 3600


def cache_dir() -> Path | None:
    """Return the cache directory, or ``None`` when caching is disabled."""
    if os.environ.get("VIZJUDGE_NO_CACHE", "").strip() not in {"", "0"}:
        return None
    if os.environ.get("VIZJUDGE_CACHE_DIR"):
        return Path(os.environ["VIZJUDGE_CACHE_DIR"]).expanduser()
    if sys.platform == "win32" and os.environ.get("LOCALAPPDATA"):
        root = Path(os.environ["LOCALAPPDATA"])
    elif os.environ.get("XDG_CACHE_HOME"):
        root = Path(os.environ["XDG_CACHE_HOME"])
    else:
        root = Path.home() / ".cache"
    return root / "vizjudge" / "frames"


def _max_bytes() -> int:
    try:
        return max(0, int(os.environ.get("VIZJUDGE_CACHE_MAX_BYTES", DEFAULT_MAX_BYTES)))
    except ValueError:
        return DEFAULT_MAX_BYTES


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _sample(frame: pd.DataFrame, max_rows: int, seed: int) -> pd.DataFrame:
    if len(frame) > max_rows:
        return frame.sample(max_rows, random_state=seed)
    return frame


def prune(directory: Path, max_bytes: int | None = None, keep: Path | None = None) -> None:
    """Drop stale entries and evict least-recently-used ones until under the size cap."""
    limit = _max_bytes() if max_bytes is None else max_bytes
    now = time.time()
    entries = []
    for entry in directory.glob("*.pkl"):
        try:
            stat = entry.stat()
        except OSError:
            continue
        if now - stat.st_mtime > MAX_AGE_SECONDS and entry != keep:
            entry.unlink(missing_ok=True)
            continue
        entries.append((stat.st_mtime, stat.st_size, entry))
    # Abandoned temporary files from interrupted writes.
    for entry in directory.glob("*.tmp"):
        try:
            if now - entry.stat().st_mtime > 3600:
                entry.unlink(missing_ok=True)
        except OSError:
            continue
    total = sum(size for _, size, _ in entries)
    for _, size, entry in sorted(entries, key=lambda item: item[0]):
        if total <= limit:
            break
        if entry == keep:
            continue
        try:
            entry.unlink(missing_ok=True)
        except OSError:
            continue
        total -= size


def load_sampled(
    path: str | Path,
    *,
    max_rows: int,
    seed: int,
    expected_sha256: str | None = None,
) -> tuple[pd.DataFrame, int, dict[str, Any]]:
    """Load a dataset sampled to ``max_rows``, reusing a cached copy when valid.

    Returns the frame, the original row count and cache metadata for the trace.
    ``expected_sha256`` (the digest the caller recorded) makes a source change
    between the caller's check and this load an error instead of a silent mismatch.
    """
    source = Path(path).expanduser().resolve()
    before = source.stat()
    sha256 = _sha256(source)
    if expected_sha256 is not None and sha256 != expected_sha256:
        raise ValueError("Dataset changed during execution")
    key_fields = {
        "format": CACHE_FORMAT,
        "sha256": sha256,
        "size": before.st_size,
        "mtime_ns": before.st_mtime_ns,
        "max_rows": max_rows,
        "seed": seed,
        "pandas": pd.__version__,
    }
    key = hashlib.sha256(repr(sorted(key_fields.items())).encode("utf-8")).hexdigest()
    meta: dict[str, Any] = {"key": key[:16], "dataset_sha256": sha256}
    directory = cache_dir()
    if directory is None:
        frame = load_dataset(source)
        return _sample(frame, max_rows, seed), len(frame), {**meta, "status": "disabled"}
    entry = directory / f"{key}.pkl"
    try:
        with entry.open("rb") as handle:
            payload = pickle.load(handle)
        if payload.get("key") == key_fields:
            os.utime(entry)  # Mark as recently used for LRU eviction.
            return payload["frame"], payload["original_rows"], {**meta, "status": "hit"}
    except Exception:
        # A missing, corrupt or incompatible entry is a miss; it is overwritten below.
        pass
    frame = load_dataset(source)
    original_rows = len(frame)
    frame = _sample(frame, max_rows, seed)
    after = source.stat()
    if (after.st_size, after.st_mtime_ns) != (before.st_size, before.st_mtime_ns):
        # Never cache a parse of a file that changed while it was being read.
        return frame, original_rows, {**meta, "status": "miss_uncached"}
    stored = _store(directory, entry, key_fields, frame, original_rows)
    meta["status"] = "miss" if stored else "miss_uncached"
    return frame, original_rows, meta


def _store(
    directory: Path, entry: Path, key_fields: dict, frame: pd.DataFrame, original_rows: int
) -> bool:
    temporary = None
    try:
        directory.mkdir(parents=True, exist_ok=True)
        payload = {"key": key_fields, "frame": frame, "original_rows": original_rows}
        handle, temporary = tempfile.mkstemp(dir=directory, suffix=".tmp")
        with os.fdopen(handle, "wb") as stream:
            pickle.dump(payload, stream, protocol=pickle.HIGHEST_PROTOCOL)
        os.replace(temporary, entry)
        temporary = None
        prune(directory, keep=entry)
        return entry.exists()
    except Exception:
        # Caching is an optimization; an unwritable cache must never fail analysis.
        return False
    finally:
        if temporary is not None:
            Path(temporary).unlink(missing_ok=True)
