import pytest
import torch

from openjev_phase1 import core


def _set_cuda(monkeypatch, available, count=1):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: available)
    monkeypatch.setattr(torch.cuda, "device_count", lambda: count)


def test_explicit_cpu_always_resolves(monkeypatch):
    _set_cuda(monkeypatch, False, 0)
    monkeypatch.setattr(core, "mps_available", lambda: False)
    assert core.resolve_device("cpu").type == "cpu"


def test_explicit_mps_requires_availability(monkeypatch):
    _set_cuda(monkeypatch, False, 0)
    monkeypatch.setattr(core, "mps_available", lambda: False)
    with pytest.raises(ValueError, match="MPS"):
        core.resolve_device("mps")


def test_explicit_mps_resolves_when_available(monkeypatch):
    _set_cuda(monkeypatch, False, 0)
    monkeypatch.setattr(core, "mps_available", lambda: True)
    assert core.resolve_device("mps").type == "mps"


def test_unknown_device_rejected(monkeypatch):
    with pytest.raises(ValueError, match="Unknown device"):
        core.resolve_device("tpu")


def test_auto_prefers_single_cuda(monkeypatch):
    _set_cuda(monkeypatch, True, 1)
    monkeypatch.setattr(core, "mps_available", lambda: True)
    assert str(core.resolve_device(None)) == "cuda:0"


def test_auto_rejects_multiple_cuda(monkeypatch):
    _set_cuda(monkeypatch, True, 2)
    monkeypatch.setattr(core, "mps_available", lambda: True)
    with pytest.raises(ValueError, match="exactly one CUDA GPU"):
        core.resolve_device(None)


def test_auto_falls_back_mps_then_cpu(monkeypatch):
    _set_cuda(monkeypatch, False, 0)
    monkeypatch.setattr(core, "mps_available", lambda: True)
    assert core.resolve_device(None).type == "mps"
    monkeypatch.setattr(core, "mps_available", lambda: False)
    assert core.resolve_device(None).type == "cpu"


def test_dtype_candidates_cuda_strict():
    assert core.dtype_candidates(torch.device("cuda"), None) == [torch.bfloat16]


def test_dtype_candidates_cpu_skips_float16():
    device = torch.device("cpu")
    # Simulate CUDA semantics via explicit candidates check on dtype policy:
    assert core.resolve_dtype(device, None) == torch.bfloat16
    assert core.dtype_candidates(device, None) == [torch.bfloat16, torch.float32]


def test_dtype_candidates_mps_includes_fallbacks():
    candidates = core.dtype_candidates(torch.device("mps"), None)
    assert candidates[0] == torch.bfloat16
    assert torch.float16 in candidates and torch.float32 in candidates


def test_explicit_dtype_respected():
    assert core.resolve_dtype(torch.device("mps"), "float32") == torch.float32
    assert core.resolve_dtype(torch.device("cpu"), "float16") == torch.float16
    with pytest.raises(ValueError, match="Unknown dtype"):
        core.resolve_dtype(torch.device("cpu"), "int8")


def test_synchronize_never_raises():
    core.synchronize(torch.device("cpu"))
    core.synchronize(torch.device("mps"))
    core.synchronize(torch.device("cuda"))
    core.synchronize(None)


def test_describe_hardware_is_string():
    assert isinstance(core.describe_hardware(), str)


def test_peak_memory_none_off_cuda():
    assert core.peak_memory_bytes(torch.device("cpu")) is None


def test_load_rejects_bad_revision_before_weights():
    with pytest.raises(ValueError, match="pinned 40-character"):
        core.load_causal_model("some-remote-model", "short", device="cpu")
