import pytest


@pytest.fixture(autouse=True)
def _isolated_frame_cache(tmp_path_factory, monkeypatch):
    """Keep tests out of the user's real dataset cache."""
    monkeypatch.setenv("VIZJUDGE_CACHE_DIR", str(tmp_path_factory.mktemp("frame-cache")))
    monkeypatch.delenv("VIZJUDGE_NO_CACHE", raising=False)
    monkeypatch.delenv("VIZJUDGE_CACHE_MAX_BYTES", raising=False)
