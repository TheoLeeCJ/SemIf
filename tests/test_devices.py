from types import SimpleNamespace
from unittest.mock import Mock
import sys

import pytest

from openjev_phase1.core import load_causal_model, resolve_device, synchronize


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


def test_auto_preserves_single_cuda_requirement(torch_stub):
    torch_stub.cuda.is_available.return_value = True
    torch_stub.cuda.device_count.return_value = 1
    assert resolve_device().name == "cuda:0"
    torch_stub.cuda.device_count.return_value = 2
    with pytest.raises(ValueError, match="exactly one CUDA"):
        resolve_device()


def test_explicit_unavailable_devices_fail(torch_stub):
    with pytest.raises(ValueError, match="exactly one CUDA"):
        resolve_device("cuda")
    torch_stub.backends.mps.is_available.return_value = False
    with pytest.raises(ValueError, match="MPS is unavailable"):
        resolve_device("mps")
    with pytest.raises(ValueError, match="Device must"):
        resolve_device("cpu")


@pytest.mark.parametrize("backend", ["cuda", "mps", "cpu"])
def test_synchronization_dispatch(torch_stub, backend):
    device = SimpleNamespace(type=backend)
    synchronize(device)
    if backend == "cuda":
        torch_stub.cuda.synchronize.assert_called_once_with(device)
    else:
        torch_stub.cuda.synchronize.assert_not_called()
    if backend == "mps":
        torch_stub.mps.synchronize.assert_called_once_with()
    else:
        torch_stub.mps.synchronize.assert_not_called()


def test_loader_passes_device_dtype_and_revision(torch_stub, monkeypatch):
    model = Mock()
    factory = SimpleNamespace(from_pretrained=Mock(return_value=(model, {})))
    config = SimpleNamespace(model_type="other")
    transformers = SimpleNamespace(
        AutoConfig=SimpleNamespace(from_pretrained=Mock(return_value=config)),
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
    assert metadata["device"] == "mps"
    assert metadata["dtype"] == "float16"
    model.eval.assert_called_once()


def test_direct_scorer_synchronizes_before_and_after_forward(monkeypatch):
    import torch
    from openjev_phase1 import direct

    events = []

    class Model:
        def parameters(self):
            yield torch.zeros(1)

        def forward(self, input_ids, attention_mask, use_cache, return_dict, logits_to_keep):
            events.append("forward")
            return SimpleNamespace(logits=torch.tensor([[[1.0, 2.0]]]))

        __call__ = forward

    monkeypatch.setattr(direct, "encode_prompt", lambda *args: ([1], [0, 1], "hash"))
    monkeypatch.setattr(direct, "synchronize", lambda device: events.append("sync"))
    result = direct.score(Model(), None, {"id": "x", "options": [{"id": "a"}, {"id": "b"}]}, {})
    assert events == ["sync", "forward", "sync"]
    assert sum(result["probabilities"]) == pytest.approx(1.0)
    assert result["option_logits"] == [1.0, 2.0]
