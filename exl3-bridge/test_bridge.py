"""CI-safe tests for the exl3 bridge: pure logic only, no exllamav3/GPU needed.

exllamav3 is stubbed in sys.modules before importing the runner so the
module-level imports succeed in environments without exl3.
"""
from __future__ import annotations

import sys
import types
from pathlib import Path

import torch

_STUB = types.ModuleType("exllamav3")
_STUB.model_init = types.SimpleNamespace(add_args=lambda p, **k: None, init=lambda *a, **k: None)
_STUB.Generator = object
_STUB.Job = object
_STUB.version = types.ModuleType("exllamav3.version")
sys.modules.setdefault("exllamav3", _STUB)
sys.modules.setdefault("exllamav3.version", _STUB.version)

sys.path.insert(0, str(Path(__file__).resolve().parent))
import exl3_runner  # noqa: E402


class FakeTokenizer:
    """Byte-level fake: every char is one token; decode round-trips tensors."""

    def __init__(self, prefix_breaks: bool = False):
        self.prefix_breaks = prefix_breaks

    def encode(self, text):
        return [ord(c) for c in text]

    def decode(self, ids):
        assert isinstance(ids, torch.Tensor), "ids must be Tensor"
        return "".join(chr(int(i)) for i in ids.flatten().tolist())

    def single_id(self, token):
        assert len(token) == 1
        return ord(token)


def test_encode_ids_normalizes_list_tuple_tensor_returns():
    tok = FakeTokenizer()
    assert exl3_runner.encode_ids(tok, "hi") == [104, 105]  # plain list return
    class TupleTok(FakeTokenizer):
        def encode(self, text):
            return ([104, 105], text)  # (ids, text) return
    assert exl3_runner.encode_ids(TupleTok(), "hi") == [104, 105]
    class TensorTok(FakeTokenizer):
        def encode(self, text):
            return torch.tensor([[104, 105]])
    assert exl3_runner.encode_ids(TensorTok(), "x") == [104, 105]


def test_slot_ids_pass_roundtrip_and_prefix():
    tok = FakeTokenizer()
    slots = exl3_runner.slot_ids(tok, 2, "abc", [97, 98, 99])
    assert slots == [65, 66]


def test_slot_ids_rejects_decode_mismatch():
    class BadDecode(FakeTokenizer):
        def decode(self, ids):
            return "?"
    try:
        exl3_runner.slot_ids(BadDecode(), 2, "abc", [97])
        assert False, "expected ValueError"
    except ValueError as error:
        assert "decode" in str(error)


def test_slot_ids_rejects_prefix_instability():
    tok = FakeTokenizer(prefix_breaks=True)
    class PrefixBreak(FakeTokenizer):
        def encode(self, text):
            ids = [ord(c) for c in text]
            if text.endswith("A"):
                ids = ids[:-1]  # appended slot letter changes tokenization
            return ids
    try:
        exl3_runner.slot_ids(PrefixBreak(), 2, "abc", [97, 98, 99])
        assert False, "expected ValueError"
    except ValueError as error:
        assert "boundary" in str(error)


def test_result_row_carries_direct_mode_contract_fields():
    """Bridge rows must be substitutable for direct-mode rows in analysis."""
    required = {
        "id", "option_ids", "probabilities", "option_logits", "input_tokens",
        "prompt_sha256", "prompt_version", "model", "readout", "probability_status",
    }
    import json
    results = Path(__file__).resolve().parent / "results" / "probe-27b-exl3-501.jsonl"
    if not results.exists():
        raise SystemExit(0)  # results bundle absent in partial checkouts
    with results.open() as handle:
        row = json.loads(handle.readline())
    assert required <= set(row), required - set(row)
    assert row["status"] == "ok"
    assert abs(sum(row["probabilities"]) - 1.0) < 1e-9
