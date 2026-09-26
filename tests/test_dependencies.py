"""Repository-wide import resolution checks that run on every platform."""
import ast
from importlib import import_module
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
SOURCE_DIRS = ("src", "tests", "benchmarks", "exl3-bridge", "webgpu-demo")
ROOTS = ("mlx", "mlx_lm")


def _optional_imports():
    found = []
    for folder in SOURCE_DIRS:
        for path in sorted((ROOT / folder).rglob("*.py")):
            tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
            for node in ast.walk(tree):
                if not isinstance(node, ast.ImportFrom) or node.level:
                    continue
                if (node.module or "").split(".")[0] in ROOTS:
                    relative = path.relative_to(ROOT)
                    for alias in node.names:
                        found.append(f"{relative}:{node.lineno} {node.module}.{alias.name}")
    return sorted(found)


OPTIONAL_IMPORTS = _optional_imports()


def test_optional_import_sweep_still_matches_the_tree():
    assert OPTIONAL_IMPORTS, "no mlx imports found; the sweep no longer matches the tree"


def test_optional_import_names_resolve_in_the_installed_extra():
    pytest.importorskip("mlx_lm", reason="install the [mlx] extra to resolve MLX imports")
    missing = []
    for entry in OPTIONAL_IMPORTS:
        module, _, name = entry.partition(" ")[2].rpartition(".")
        if not hasattr(import_module(module), name):
            missing.append(entry)
    assert not missing, "imports the pinned MLX extra does not provide: " + ", ".join(missing)


def test_mlx_extra_pins_an_immutable_revision():
    try:
        import tomllib
    except ModuleNotFoundError:
        tomllib = pytest.importorskip("tomli")
    pinned = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    requirements = pinned["project"]["optional-dependencies"]["mlx"]
    revisions = [requirement.partition(";")[0].strip().rsplit("@", 1)[-1] for requirement in requirements]
    assert any("@" in requirement and len(revision) == 40 for requirement, revision in zip(requirements, revisions))
