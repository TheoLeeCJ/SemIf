import importlib.util
import sys
from pathlib import Path

import pytest


def benchmark(name):
    path = Path(__file__).resolve().parents[1] / "benchmarks" / f"{name}.py"
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def xpu_benchmark(monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1] / "benchmarks"))
    return benchmark("xpu_benchmark")


def test_field_name_is_backend_specific(xpu_benchmark):
    """peak_xpu_bytes, not peak_cuda_bytes: see the docstring's Apple Silicon precedent."""
    source = Path(xpu_benchmark.__file__).read_text()
    assert '"peak_xpu_bytes"' in source
    assert '"peak_cuda_bytes"' not in source


def test_choice_returns_argmax_option_id(xpu_benchmark):
    row = {"option_ids": ["yes", "no"], "probabilities": [0.3, 0.7]}
    assert xpu_benchmark.choice(row) == "no"


def test_compare_detects_flips_and_max_difference(xpu_benchmark):
    reference = [
        {"id": "a", "option_ids": ["yes", "no"], "probabilities": [0.9, 0.1]},
        {"id": "b", "option_ids": ["yes", "no"], "probabilities": [0.2, 0.8]},
    ]
    candidates = [
        {"id": "a", "option_ids": ["yes", "no"], "probabilities": [0.4, 0.6]},  # flips
        {"id": "b", "option_ids": ["yes", "no"], "probabilities": [0.25, 0.75]},  # no flip
    ]
    result = xpu_benchmark.compare(reference, candidates)
    assert result["rows"] == 2
    assert result["argmax_flips"] == ["a"]
    assert result["max_probability_difference"] == pytest.approx(0.5)


def test_compare_rejects_mismatched_ids(xpu_benchmark):
    reference = [{"id": "a", "option_ids": ["yes", "no"], "probabilities": [0.9, 0.1]}]
    candidates = [{"id": "b", "option_ids": ["yes", "no"], "probabilities": [0.9, 0.1]}]
    with pytest.raises(ValueError, match="identical decision IDs"):
        xpu_benchmark.compare(reference, candidates)


def test_rejects_existing_output_directory_before_loading(monkeypatch, tmp_path, xpu_benchmark):
    output = tmp_path / "run"
    output.mkdir()
    monkeypatch.setattr(sys, "argv", ["xpu_benchmark", "--output", str(output)])
    loader_calls = []
    monkeypatch.setattr(xpu_benchmark, "load_causal_model", lambda *a, **k: loader_calls.append(1))
    with pytest.raises(FileExistsError):
        xpu_benchmark.main()
    assert loader_calls == []
