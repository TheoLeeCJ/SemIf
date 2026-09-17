from types import SimpleNamespace

import pytest


@pytest.mark.parametrize("tied", [False, True])
def test_last_logits_uses_language_model(tied):
    mx = pytest.importorskip("mlx.core")
    from openjev_phase1.mlx_backend import _last_logits

    hidden = mx.arange(24).reshape(1, 3, 8).astype(mx.float32)
    cache = [object()]

    class TextModel:
        def __call__(self, ids, cache=None):
            assert cache is cache
            return hidden

    def project(last):
        assert last.shape == (1, 1, 8)
        return last * 2

    def wrong_projection(last):
        pytest.fail("Used the wrong output projection")

    text = TextModel()
    text.embed_tokens = SimpleNamespace(as_linear=project if tied else wrong_projection)
    # Deliberately no top-level .model: mlx-lm 0.31.3 Qwen3.5 uses this wrapper.
    model = SimpleNamespace(language_model=SimpleNamespace(
        model=text, args=SimpleNamespace(tie_word_embeddings=tied),
        lm_head=wrong_projection if tied else project,
    ))
    logits = _last_logits(model, mx.array([[1, 2, 3]]), cache)
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
    assert not first["cache_hit"] and repeated["cache_hit"] and other["cache_hit"]
    row["state"]["s"] = 2
    assert not scorer.score(row)["cache_hit"]


def test_loader_rejects_unpinned_revision():
    from openjev_phase1.mlx_backend import load_causal_model

    with pytest.raises(ValueError, match="pinned"):
        load_causal_model("not-a-local-model", "main")


@pytest.mark.parametrize("config", [
    {"model_type": "qwen3_5_text"},
    {"model_type": "qwen3_5", "model_file": "custom.py"},
    {"model_type": "qwen3_5", "quantization": {"bits": 4}},
])
def test_loader_rejects_unsupported_checkpoint(tmp_path, config):
    import json

    from openjev_phase1.mlx_backend import load_causal_model

    (tmp_path / "config.json").write_text(json.dumps(config))
    with pytest.raises(ValueError):
        load_causal_model(str(tmp_path), "local-manifest")


def test_shared_rejects_mixed_states():
    from openjev_phase1.mlx_backend import score_shared
    with pytest.raises(ValueError, match="exact state"):
        score_shared(None, None, [{"state": "a"}, {"state": "b"}], {})
