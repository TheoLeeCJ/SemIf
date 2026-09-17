from types import SimpleNamespace

import pytest


@pytest.mark.parametrize("tied", [False, True])
def test_qwen35_last_logits_uses_language_model(tied):
    mx = pytest.importorskip("mlx.core")
    from openjev_phase1.mlx_backend import _last_logits

    calls = []
    hidden = mx.arange(24).reshape(1, 3, 8).astype(mx.float32)
    cache = [object()]

    class TextModel:
        def __call__(self, ids, cache=None):
            calls.append(cache)
            return hidden

    def project(last):
        assert last.shape == (1, 1, 8)
        return last * 2

    text = TextModel()
    def wrong_projection(last):
        pytest.fail("Used the wrong output projection")

    text.embed_tokens = SimpleNamespace(as_linear=project if tied else wrong_projection)
    # Deliberately no top-level .model: mlx-lm 0.31.3 Qwen3.5 uses this wrapper.
    model = SimpleNamespace(language_model=SimpleNamespace(
        model=text, args=SimpleNamespace(tie_word_embeddings=tied),
        lm_head=wrong_projection if tied else project,
    ))
    logits = _last_logits(model, mx.array([[1, 2, 3]]), cache)
    assert calls == [cache]
    assert logits.tolist() == (hidden[:, -1, :] * 2).tolist()


def test_serial_branches_preserve_hybrid_state(monkeypatch):
    mx = pytest.importorskip("mlx.core")
    cache_module = pytest.importorskip("mlx_lm.models.cache")
    from openjev_phase1 import mlx_backend as backend

    monkeypatch.setattr(cache_module, "make_prompt_cache", lambda model: [cache_module.ArraysCache(size=2)])
    monkeypatch.setattr(backend, "_state_prefix", lambda *args: [1, 2])
    monkeypatch.setattr(backend, "encode_prompt", lambda tok, row, limit: (
        [1, 2, row["token"]], [0, 1], "hash-" + row["id"],
    ))
    entries = []

    def forward(model, ids, cache=None):
        entries.append(None if cache[0][0] is None else cache[0][0].tolist())
        value = mx.sum(ids).astype(mx.float32)
        cache[0][0] = mx.array([value])
        cache[0][1] = mx.array([value + 1])
        return mx.stack([value, -value])[None, :]

    monkeypatch.setattr(backend, "_last_logits", forward)
    scorer = backend.SerialPrefixScorer(object(), None, {})
    row = {"id": "a", "state": {"s": 1}, "token": 3, "options": [{"id": "a"}, {"id": "b"}]}
    first = scorer.score(row)
    repeated = scorer.score(row)
    other = scorer.score({**row, "id": "b", "token": 4})
    assert entries == [None, [3.0], [3.0], [3.0]]
    assert first["option_logits"] == repeated["option_logits"]
    assert first["answer_token_ids"] == [0, 1]
    assert first["full_vocab_argmax_id"] == 0
    assert first["allowed_token_mass"] == pytest.approx(1.0)
    assert not first["cache_hit"] and repeated["cache_hit"] and other["cache_hit"]
    assert scorer.cache[0][0].tolist() == [3.0]
    assert scorer.cache[0][1].tolist() == [4.0]
    row["state"]["s"] = 2
    assert not scorer.score(row)["cache_hit"]


def test_loader_rejects_unpinned_revision():
    from openjev_phase1.mlx_backend import load_causal_model

    with pytest.raises(ValueError, match="pinned"):
        load_causal_model("not-a-local-model", "main")


@pytest.mark.parametrize("config,message", [
    ({"model_type": "qwen3_5_text"}, "top-level Qwen3.5"),
    ({"model_type": "qwen3_5", "model_file": "custom.py"}, "custom model_file"),
    ({"model_type": "qwen3_5", "quantization": {"bits": 4}}, "unquantized"),
    ({"model_type": "qwen3_5", "quantization_config": {"bits": 4}}, "unquantized"),
])
def test_loader_rejects_unsupported_checkpoint(tmp_path, monkeypatch, config, message):
    import json
    import sys
    from unittest.mock import Mock

    from openjev_phase1.mlx_backend import load_causal_model

    mx = SimpleNamespace(metal=SimpleNamespace(is_available=lambda: True),
                         gpu=object(), set_default_device=lambda device: None)
    loader = Mock(side_effect=AssertionError("Unsafe checkpoint reached the model loader"))
    monkeypatch.setitem(sys.modules, "mlx", SimpleNamespace(core=mx))
    monkeypatch.setitem(sys.modules, "mlx.core", mx)
    monkeypatch.setitem(sys.modules, "mlx.utils", SimpleNamespace(tree_flatten=Mock()))
    monkeypatch.setitem(sys.modules, "mlx_lm", SimpleNamespace())
    monkeypatch.setitem(sys.modules, "mlx_lm.utils", SimpleNamespace(load_model=loader))
    (tmp_path / "config.json").write_text(json.dumps(config))
    with pytest.raises(ValueError, match=message):
        load_causal_model(str(tmp_path), "local-manifest")


def test_shared_encodes_once_before_forwarding(monkeypatch):
    from openjev_phase1 import mlx_backend as backend

    events = []
    rows = [{"id": key, "state": "same"} for key in ("a", "b")]

    def encode(tokenizer, row, limit):
        events.append(("encode", row["id"]))
        return [1, 2, 3], [0, 1], row["id"]

    class Scorer:
        def __init__(self, *args):
            self.metadata = {}

        def score(self, row, encoded=None):
            events.append(("score", row["id"]))
            assert encoded == ([1, 2, 3], [0, 1], row["id"])
            return {"input_tokens": 3, "prefix_tokens": 2, "prefill_seconds": 0.1,
                    "copy_seconds": 0.2, "suffix_forward_seconds": 0.3}

    monkeypatch.setattr(backend, "encode_prompt", encode)
    monkeypatch.setattr(backend, "SerialPrefixScorer", Scorer)
    results, timing = backend.score_shared(None, None, rows, {})
    assert events == [("encode", "a"), ("encode", "b"), ("score", "a"), ("score", "b")]
    assert len(results) == 2
    assert timing["replicate_seconds"] == pytest.approx(0.4)
    assert timing["suffix_forward_seconds"] == pytest.approx(0.6)
    assert timing["true_suffix_tokens"] == timing["padded_suffix_tokens"] == 2
    assert timing["encode_seconds"] > 0


def test_shared_rejects_mixed_states():
    from openjev_phase1.mlx_backend import score_shared
    with pytest.raises(ValueError, match="exact state"):
        score_shared(None, None, [{"state": "a"}, {"state": "b"}], {})
