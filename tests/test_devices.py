from types import SimpleNamespace
from unittest.mock import Mock
import sys

import pytest

from semif_phase1.core import load_causal_model, resolve_device, synchronize


@pytest.fixture
def torch_stub(monkeypatch):
    torch = SimpleNamespace(
        cuda=SimpleNamespace(is_available=Mock(return_value=False), device_count=Mock(return_value=0),
                             synchronize=Mock()),
        backends=SimpleNamespace(mps=SimpleNamespace(is_available=Mock(return_value=True))),
        mps=SimpleNamespace(synchronize=Mock()),
        device=lambda name: SimpleNamespace(type=name.split(":")[0], name=name),
    )
    monkeypatch.setitem(sys.modules, "torch", torch)
    return torch


def test_auto_uses_mps_without_cuda(torch_stub):
    assert resolve_device().name == "mps"


def test_cuda_requires_exactly_one_gpu(torch_stub):
    torch_stub.cuda.is_available.return_value = True
    torch_stub.cuda.device_count.return_value = 1
    assert resolve_device().name == "cuda:0"
    torch_stub.cuda.device_count.return_value = 2
    with pytest.raises(ValueError, match="exactly one CUDA"):
        resolve_device()
    with pytest.raises(ValueError, match="exactly one CUDA"):
        resolve_device("cuda")


def test_mps_availability_is_checked(torch_stub):
    torch_stub.backends.mps.is_available.return_value = False
    with pytest.raises(ValueError, match="MPS is unavailable"):
        resolve_device("mps")


@pytest.mark.parametrize("backend", ["cuda", "mps", "cpu"])
def test_synchronization_dispatch(torch_stub, backend):
    device = SimpleNamespace(type=backend)
    synchronize(device)
    if backend == "cuda":
        torch_stub.cuda.synchronize.assert_called_once_with(device)
    if backend == "mps":
        torch_stub.mps.synchronize.assert_called_once_with()


def test_loader_passes_device_dtype_and_revision(torch_stub, monkeypatch):
    model = Mock()
    factory = SimpleNamespace(from_pretrained=Mock(return_value=(model, {})))
    transformers = SimpleNamespace(
        AutoConfig=SimpleNamespace(from_pretrained=Mock(return_value=SimpleNamespace(model_type="other"))),
        AutoTokenizer=SimpleNamespace(from_pretrained=Mock(return_value=object())),
        AutoModelForCausalLM=factory, __version__="test",
    )
    monkeypatch.setitem(sys.modules, "transformers", transformers)
    torch_stub.device = lambda name: name
    torch_stub.float16 = "float16"
    torch_stub.__version__ = "test"
    revision = "a" * 40
    _, _, metadata = load_causal_model("test/model", revision, "mps", "float16")
    kwargs = factory.from_pretrained.call_args.kwargs
    assert kwargs["device_map"] == {"": "mps"}
    assert kwargs["dtype"] == "float16"
    assert kwargs["revision"] == revision
    assert kwargs["trust_remote_code"] is False
    assert metadata["device"] == "mps" and metadata["dtype"] == "float16"
    model.eval.assert_called_once()


def test_loader_defaults_to_float16_on_mps(torch_stub, monkeypatch):
    model = Mock()
    factory = SimpleNamespace(from_pretrained=Mock(return_value=(model, {})))
    transformers = SimpleNamespace(
        AutoConfig=SimpleNamespace(from_pretrained=Mock(return_value=SimpleNamespace(model_type="other"))),
        AutoTokenizer=SimpleNamespace(from_pretrained=Mock(return_value=object())),
        AutoModelForCausalLM=factory, __version__="test",
    )
    monkeypatch.setitem(sys.modules, "transformers", transformers)
    torch_stub.device = lambda name: SimpleNamespace(type=name.split(":")[0], name=name)
    torch_stub.float16 = "float16"
    torch_stub.bfloat16 = "bfloat16"
    torch_stub.__version__ = "test"
    revision = "a" * 40
    # On MPS (or auto resolving to MPS), dtype defaults to float16
    _, _, metadata = load_causal_model("test/model", revision, "mps")
    kwargs = factory.from_pretrained.call_args.kwargs
    assert kwargs["dtype"] == "float16"
    assert metadata["dtype"] == "float16"

    # On CUDA, dtype defaults to bfloat16
    torch_stub.cuda.is_available.return_value = True
    torch_stub.cuda.device_count.return_value = 1
    _, _, metadata_cuda = load_causal_model("test/model", revision, "cuda")
    kwargs_cuda = factory.from_pretrained.call_args.kwargs
    assert kwargs_cuda["dtype"] == "bfloat16"
    assert metadata_cuda["dtype"] == "bfloat16"

