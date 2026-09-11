import pytest


@pytest.fixture(autouse=True)
def isolated_xdg_cache(tmp_path, monkeypatch):
    """Keep process coordination locks out of the developer's cache during tests."""
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path / "xdg-cache"))
