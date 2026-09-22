import sys
from types import ModuleType

from semif_phase1 import mlx_backend


def test_clear_memory_cache_releases_inactive_mlx_allocations(monkeypatch):
    calls = []
    core = ModuleType("mlx.core")
    core.clear_cache = lambda: calls.append("cleared")
    package = ModuleType("mlx")
    package.__path__ = []
    package.core = core
    monkeypatch.setitem(sys.modules, "mlx", package)
    monkeypatch.setitem(sys.modules, "mlx.core", core)

    mlx_backend.clear_memory_cache()

    assert calls == ["cleared"]
