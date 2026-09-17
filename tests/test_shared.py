from types import SimpleNamespace

import pytest

torch = pytest.importorskip("torch")

from openjev_phase1.shared import _suffix_layout, score_shared  # noqa: E402


def test_suffix_padding_follows_real_tokens():
    layout, ends = _suffix_layout([[3, 4], [5]], 7, 0)
    assert layout["input_ids"] == [[3, 4], [5, 0]]
    assert layout["attention_mask"] == [[1] * 9, [1] * 8 + [0]]
    assert layout["position_ids"] == [[7, 8], [7, 0]]
    assert ends == [1, 0]


def test_empty_suffix_is_rejected():
    with pytest.raises(ValueError):
        _suffix_layout([[1], []], 7, 0)


class FakeCache:
    """Native-cache stand-in: mutable length, records reorder calls."""

    def __init__(self, length):
        self.length = length
        self.orders = []

    def get_seq_length(self):
        return self.length

    def reorder_cache(self, order):
        self.orders.append(order.tolist() if hasattr(order, "tolist") else list(order))

    def __deepcopy__(self, memo):
        clone = FakeCache(self.length)
        memo[id(self)] = clone
        return clone


class FakeModel:
    """Forward with a real signature so selective-logits detection succeeds."""

    def __init__(self, calls, device):
        self.calls = calls
        self._device = device

    def parameters(self):
        yield SimpleNamespace(device=self._device)

    def eval(self):
        return self

    def forward(self, input_ids=None, attention_mask=None, use_cache=False,
                return_dict=True, logits_to_keep=1, past_key_values=None):
        width = int(input_ids.shape[1])
        entry_length = None if past_key_values is None else past_key_values.length
        self.calls.append({
            "batch": int(input_ids.shape[0]),
            "width": width,
            "logits_to_keep": logits_to_keep,
            "cache_entry_length": entry_length,
            "reorders": None if past_key_values is None else len(past_key_values.orders),
        })
        if past_key_values is None:
            past_key_values = FakeCache(width)
        else:
            past_key_values.length += width
        logits = torch.arange(width + 4, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
        return SimpleNamespace(logits=logits, past_key_values=past_key_values)

    __call__ = forward


def test_score_shared_looped_path_forwards_one_row_per_call(monkeypatch):
    if not torch.backends.mps.is_available():
        pytest.skip("MPS unavailable")
    rows_data = [
        {"id": "a", "state": "s", "question": "q1", "options": [
            {"id": "x", "description": "X"}, {"id": "y", "description": "Y"}]},
        {"id": "b", "state": "s", "question": "q2", "options": [
            {"id": "x", "description": "X"}, {"id": "y", "description": "Y"}]},
    ]

    class FakeTok:
        pad_token_id = 0
        eos_token_id = 1

    calls = []

    def fake_encode_prompt(tokenizer, row, max_tokens):
        return ([1] * 8, [2, 3], f"hash-{row['id']}")

    def fake_state_prefix(tokenizer, state):
        return [1] * 6

    import openjev_phase1.shared as shared_module
    monkeypatch.setattr(shared_module, "encode_prompt", fake_encode_prompt)
    monkeypatch.setattr(shared_module, "_state_prefix", fake_state_prefix)
    results, timing = score_shared(FakeModel(calls, torch.device("mps")), FakeTok(), rows_data, {}, 4096)

    assert [r["id"] for r in results] == ["a", "b"]
    assert timing["batch_size"] == 2
    assert timing["suffix_forward_seconds"] > 0
    assert timing["replicate_seconds"] > 0
    assert all(r["model"]["serving_config"] == "native-state-prefix-looped-v1" for r in results)
    # Looped suffixes are unpadded: padded count equals true suffix tokens.
    assert timing["padded_suffix_tokens"] == timing["true_suffix_tokens"] == 4
    assert len(calls) == 3  # one prefill + one suffix forward per row
    prefill, *suffixes = calls
    assert prefill["cache_entry_length"] is None
    for call in suffixes:
        assert call["batch"] == 1
        assert call["logits_to_keep"] == 1
        # Deep-copied branches are already independent batch-1 rows.
        assert call["reorders"] == 0
        # Each branch starts at prefix length: deep-copied, not shared state.
        assert call["cache_entry_length"] == 6
    for result in results:
        assert sum(result["probabilities"]) == pytest.approx(1.0)
