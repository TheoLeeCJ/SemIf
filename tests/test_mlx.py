"""Real, tiny hybrid-model regressions; no network or large weights required."""
import copy
import platform

import pytest

if platform.system() != "Darwin" or platform.machine() != "arm64":
    pytest.skip("MLX requires Apple Silicon", allow_module_level=True)
mx = pytest.importorskip("mlx.core")
pytest.importorskip("mlx_lm")

from mlx_lm.models.qwen3_5 import Model, ModelArgs
from mlx_lm.models.gemma4 import Model as Gemma4Model
from mlx_lm.models.gemma4 import ModelArgs as Gemma4ModelArgs
from mlx_lm.models.muse_glimmer import Model as MuseGlimmerModel
from mlx_lm.models.muse_glimmer import ModelArgs as MuseGlimmerModelArgs
from semif_phase1 import mlx_backend as backend


class Tokenizer:
    pad_token_id = 0
    eos_token_id = 1

    def apply_chat_template(self, turns, **kwargs):
        return "\n".join(turn["content"] for turn in turns) + "\nAssistant:"

    def encode(self, text, add_special_tokens=False):
        return list(text.encode())

    def decode(self, ids):
        return bytes(ids).decode()


@pytest.fixture(scope="module")
def model():
    mx.random.seed(17)
    result = Model(ModelArgs(model_type="qwen3_5", text_config={
        "hidden_size": 64, "intermediate_size": 128, "num_hidden_layers": 4,
        "num_attention_heads": 4, "num_key_value_heads": 2, "head_dim": 16,
        "vocab_size": 256, "linear_num_key_heads": 2, "linear_num_value_heads": 4,
        # The native Metal delta kernel requires production-sized head widths.
        "linear_key_head_dim": 128, "linear_value_head_dim": 128,
        "full_attention_interval": 2,
    }))
    result.eval()
    mx.eval(result.parameters())
    return result


@pytest.fixture(scope="module")
def gemma4_model():
    mx.random.seed(19)
    result = Gemma4Model(Gemma4ModelArgs(
        model_type="gemma4_unified",
        vocab_size=256,
        text_config={
            "model_type": "gemma4_unified_text",
            "hidden_size": 64,
            "intermediate_size": 128,
            "num_hidden_layers": 2,
            "num_attention_heads": 4,
            "num_key_value_heads": 2,
            "num_global_key_value_heads": 2,
            "head_dim": 16,
            "global_head_dim": 16,
            "vocab_size": 256,
            "vocab_size_per_layer_input": 256,
            "hidden_size_per_layer_input": 0,
            "num_kv_shared_layers": 0,
            "layer_types": ["sliding_attention", "full_attention"],
            "sliding_window": 64,
            "use_double_wide_mlp": False,
        },
    ))
    result.eval()
    mx.eval(result.parameters())
    return result


@pytest.fixture(scope="module")
def muse_glimmer_model():
    mx.random.seed(23)
    result = MuseGlimmerModel(MuseGlimmerModelArgs(
        model_type="muse_glimmer",
        hidden_size=64,
        intermediate_size=128,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=16,
        vocab_size=256,
        layer_types=["sliding_attention", "full_attention"],
        layer_rope_theta=[500000.0, 0],
        sliding_window=64,
    ))
    result.eval()
    mx.eval(result.parameters())
    return result


@pytest.fixture
def rows():
    return [dict(id=str(i), state={"evidence": "A deployment succeeded."}, question=question,
                 options=[{"id": "yes", "description": "Yes"}, {"id": "no", "description": "No"}])
            for i, question in enumerate(["Success?", "Did the deployment succeed in the supplied evidence?", "Failure?"])]


def assert_same(left, right):
    assert left["id"] == right["id"]
    for key in ("option_ids", "answer_token_ids", "input_ids_sha256", "prompt_sha256"):
        assert left[key] == right[key]
    # Full and split prefill use different Metal reduction shapes, even in FP32.
    assert left["probabilities"] == pytest.approx(right["probabilities"], abs=1e-4)


def test_hybrid_cache_branches_padding_and_order(model, rows):
    tokenizer = Tokenizer()
    fresh = [backend.score(model, tokenizer, row, {}) for row in rows]
    serial = backend.SerialPrefixScorer(model, tokenizer, {})
    for index in (0, 2, 1, 0):
        actual = serial.score(rows[index])
        assert_same(fresh[index], actual)
    for order in (rows, rows[::-1], rows[:1]):
        results, timing = backend.score_shared(model, tokenizer, order, {})
        assert timing["batch_size"] == len(order)
        for row in results:
            assert_same(fresh[int(row["id"])], row)


def test_gemma4_cache_branches_match_direct_scoring(gemma4_model, rows):
    tokenizer = Tokenizer()
    fresh = [backend.score(gemma4_model, tokenizer, row, {}) for row in rows]
    shared, timing = backend.score_shared(gemma4_model, tokenizer, rows, {})
    assert timing["batch_size"] == len(rows)
    for expected, actual in zip(fresh, shared, strict=True):
        assert_same(expected, actual)


def test_muse_glimmer_cache_branches_match_direct_scoring(muse_glimmer_model, rows):
    tokenizer = Tokenizer()
    fresh = [backend.score(muse_glimmer_model, tokenizer, row, {}) for row in rows]
    shared, timing = backend.score_shared(muse_glimmer_model, tokenizer, rows, {})
    assert timing["batch_size"] == len(rows)
    for expected, actual in zip(fresh, shared, strict=True):
        assert_same(expected, actual)


def test_serial_invalidates_cache_for_mutated_structured_state(model, rows):
    row = copy.deepcopy(rows[0])
    serial = backend.SerialPrefixScorer(model, Tokenizer(), {})
    assert not serial.score(row)["cache_hit"]
    assert serial.score(row)["cache_hit"]
    row["state"]["evidence"] = "The deployment failed."
    actual = serial.score(row)
    assert not actual["cache_hit"]
    assert_same(backend.score(model, Tokenizer(), row, {}), actual)


def test_shared_rejects_mixed_states_and_duplicate_ids(model, rows):
    with pytest.raises(ValueError, match="unique"):
        backend.score_shared(model, Tokenizer(), [rows[0], rows[0]], {})
    rows[1]["state"] = "different"
    with pytest.raises(ValueError, match="exact state"):
        backend.score_shared(model, Tokenizer(), rows, {})


def test_token_limit_prevents_inference(model, rows):
    with pytest.raises(ValueError, match="no truncation"):
        backend.score(model, Tokenizer(), rows[0], {}, max_tokens=2)


@pytest.mark.parametrize("config", [
    {"model_type": "qwen3_5", "model_file": "custom.py"},
    {"model_type": "unrecognized"},
])
def test_loader_rejects_custom_or_unsupported_models(tmp_path, config):
    import json

    (tmp_path / "config.json").write_text(json.dumps(config))
    with pytest.raises(ValueError, match="native Qwen3.5, Gemma 4, and Muse Glimmer"):
        backend.load_model(str(tmp_path), "local-fixture")


@pytest.mark.parametrize("model_type", ["qwen3_5", "gemma4", "gemma4_unified", "muse_glimmer"])
def test_loader_accepts_supported_native_model_types(tmp_path, monkeypatch, model, model_type):
    import json
    import mlx_lm

    (tmp_path / "config.json").write_text(json.dumps({"model_type": model_type}))
    monkeypatch.setattr(mlx_lm, "load", lambda *args, **kwargs: (model, Tokenizer()))
    _, _, metadata = backend.load_model(str(tmp_path), "local-fixture")
    assert metadata["model_type"] == model_type


def test_remote_model_requires_immutable_revision():
    with pytest.raises(ValueError, match="40-character"):
        backend.load_model("Qwen/Qwen3.5-4B", "main")


def test_recurrent_qk_normalization_matches_reference_l2_epsilon():
    # Small q/k expose the release-0.31.3 sum-versus-mean epsilon mismatch.
    from mlx_lm.models.gated_delta import normalize_qk

    width = 128
    values = mx.full((1, 2, width), 1e-4)
    q, k = normalize_qk(values, values, inv_scale=width**-0.5, eps=1e-6)
    expected = values * mx.rsqrt(mx.sum(values * values, axis=-1, keepdims=True) + 1e-6)
    assert mx.max(mx.abs(k - expected)).item() < 1e-6
    assert mx.max(mx.abs(q - expected * width**-0.5)).item() < 1e-6


def test_serial_keys_by_exact_tokens_not_python_value_equality(model, rows):
    row = copy.deepcopy(rows[0])
    row["state"] = {"value": True}
    serial = backend.SerialPrefixScorer(model, Tokenizer(), {})
    serial.score(row)
    row["state"] = {"value": 1}  # Python compares True == 1, JSON prompts differ.
    actual = serial.score(row)
    assert not actual["cache_hit"]
    assert_same(backend.score(model, Tokenizer(), row, {}), actual)


@pytest.mark.parametrize('limit_mib', [None, 0, 64])
def test_loader_applies_cache_limit_and_records_bytes(tmp_path, monkeypatch, model, limit_mib):
    import json
    import mlx_lm

    (tmp_path / 'config.json').write_text(json.dumps({'model_type': 'qwen3_5'}))
    monkeypatch.setattr(mlx_lm, 'load', lambda *args, **kwargs: (model, Tokenizer()))
    previous = mx.set_cache_limit(32 * 1024 * 1024)
    try:
        kwargs = {} if limit_mib is None else {'cache_limit_mib': limit_mib}
        _, _, metadata = backend.load_model(str(tmp_path), 'local-fixture', **kwargs)
        expected = (256 if limit_mib is None else limit_mib) * 1024 * 1024
        assert metadata['allocator_cache_limit_bytes'] == expected
        assert mx.set_cache_limit(previous) == expected
    finally:
        mx.set_cache_limit(previous)


@pytest.mark.parametrize('limit', [-1, 1.5, True])
def test_loader_rejects_invalid_cache_limit_before_loading(limit):
    with pytest.raises(ValueError, match='nonnegative integer'):
        backend.load_model('Qwen/Qwen3.5-4B', '0' * 40, cache_limit_mib=limit)
