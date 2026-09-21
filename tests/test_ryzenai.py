"""Regression coverage for the Ryzen AI NPU direct-logit boundary.

These tests deliberately replace the OGA runtime at its import boundary: they
exercise the native generator contract without a model download or NPU.
"""

import json
import sys
from types import SimpleNamespace

import numpy as np
import pytest

from semif_phase1 import ryzenai_backend as backend
from semif_phase1.core import LETTERS, direct_messages, validate_row


ROW = {
    "id": "decision-1",
    "state": {"result": "passed"},
    "question": "Did it pass?",
    "options": [{"id": "yes", "description": "Yes"}, {"id": "no", "description": "No"}],
}


def _row_with_options(count):
    return {
        "id": f"decision-{count}", "state": "evidence", "question": "Choose one.",
        "options": [{"id": f"option-{index}", "description": f"Option {index}"} for index in range(count)],
    }


def _config(provider=None):
    return {
        "model": {
            "context_length": 40960,
            "decoder": {
                "session_options": {"provider_options": provider if provider is not None else [{"RyzenAI": {
                    "hybrid_opt_token_backend": "npu",
                }}]},
            },
        }, "search": {"max_length": 4096},
    }


def _write_config(directory, config):
    (directory / "genai_config.json").write_text(json.dumps(config), encoding="utf-8")


@pytest.mark.parametrize(("provider", "message"), [
    ([{"RyzenAI": {"hybrid_opt_token_backend": "cpu"}}], "hybrid_opt_token_backend"),
    ([{"DmlExecutionProvider": {}}], "exactly one RyzenAI"),
    ([{"RyzenAI": {"hybrid_opt_token_backend": "npu", "device": "cuda"}}], "hybrid GPU"),
    ([], "exactly one RyzenAI"),
])
def test_npu_config_rejects_hybrid_or_missing_npu_provider(tmp_path, provider, message):
    _write_config(tmp_path, _config(provider))

    with pytest.raises(ValueError, match=message):
        backend._load_npu_config(tmp_path)


def test_npu_loader_rejects_unsupported_oga_runtime_before_importing_sdk(tmp_path, monkeypatch):
    _write_config(tmp_path, _config())
    monkeypatch.setattr(backend, "_package_version", lambda _: "0.13.0")

    with pytest.raises(RuntimeError, match="onnxruntime-genai-directml-ryzenai==0.14.0"):
        backend.load_model(str(tmp_path), "fixture-revision")


def test_load_model_keeps_local_config_and_records_artifact_identity(tmp_path, monkeypatch):
    _write_config(tmp_path, _config())
    (tmp_path / "model.onnx").write_bytes(b"fixture graph")
    original_config = (tmp_path / "genai_config.json").read_bytes()
    calls = []
    model = SimpleNamespace(device_type="fixture-npu")
    reference = object()
    tokenizer = object()

    def load_reference(source, **kwargs):
        calls.append((source, kwargs))
        return reference

    def load_native(source):
        assert source == str(tmp_path)
        return model

    monkeypatch.setattr(backend, "_package_version", lambda _: backend.OGA_VERSION)
    monkeypatch.setitem(sys.modules, "transformers", SimpleNamespace(
        __version__="4.57.6", AutoTokenizer=SimpleNamespace(from_pretrained=load_reference),
    ))
    monkeypatch.setitem(sys.modules, "onnxruntime_genai", SimpleNamespace(
        Model=load_native, Tokenizer=lambda _: tokenizer,
    ))
    loaded, adapter, metadata = backend.load_model(str(tmp_path), "fixture-revision")

    assert loaded is model and adapter.reference is reference and adapter.oga is tokenizer
    assert calls == [(str(tmp_path), {
        "local_files_only": True, "trust_remote_code": False, "fix_mistral_regex": False,
    })]
    assert (tmp_path / "genai_config.json").read_bytes() == original_config
    assert metadata["revision"] == "fixture-revision"
    assert metadata["context_ceiling"] == 4096
    assert metadata["generation_reserve_tokens"] == 0
    assert metadata["oga_version"] == backend.OGA_VERSION
    assert metadata["source_artifact_sha256"] == backend._artifact_hashes(tmp_path)
    assert set(metadata["source_artifact_sha256"]) == {"genai_config.json", "model.onnx"}


@pytest.mark.parametrize("component", ["encoder", "decoder_pipeline"])
def test_npu_config_requires_a_decoder_only_execution_graph(tmp_path, component):
    config = _config()
    config["model"][component] = {"session_options": {"provider_options": []}}
    _write_config(tmp_path, config)

    with pytest.raises(ValueError, match="decoder-only language model"):
        backend._load_npu_config(tmp_path)


def test_npu_config_rejects_decoder_pipeline_nested_in_a_list(tmp_path):
    config = _config()
    config["model"]["decoder"]["pipeline"] = [{"session_options": {
        "provider_options": [{"DmlExecutionProvider": {}}],
    }}]
    _write_config(tmp_path, config)

    with pytest.raises(ValueError, match="decoder-only language model"):
        backend._load_npu_config(tmp_path)


def test_npu_config_allows_gpu_in_a_filename_but_rejects_a_gpu_device(tmp_path):
    harmless = _config()
    harmless["model"]["decoder"]["session_options"]["provider_options"][0]["RyzenAI"]["cache_file"] = "gpu-compiled.bin"
    _write_config(tmp_path, harmless)
    _, options, _ = backend._load_npu_config(tmp_path)
    assert options["cache_file"] == "gpu-compiled.bin"

    forbidden = _config()
    forbidden["model"]["decoder"]["session_options"]["provider_options"][0]["RyzenAI"]["device"] = "cuda"
    _write_config(tmp_path, forbidden)
    with pytest.raises(ValueError, match="hybrid GPU"):
        backend._load_npu_config(tmp_path)


def test_artifact_hashes_include_only_declared_tokenfusion_cache_artifacts(tmp_path):
    cache = tmp_path / "cache"
    cache.mkdir()
    (cache / "txn_bins.zip").write_bytes(b"transaction")
    (cache / "decoder_meta.json").write_text("{}", encoding="utf-8")
    (cache / "unrelated.json").write_text("{}", encoding="utf-8")
    (tmp_path / ".cache").mkdir()
    (tmp_path / ".cache" / "hub_meta.json").write_text("{}", encoding="utf-8")
    hashes = backend._artifact_hashes(tmp_path)
    assert set(hashes) == {"cache/txn_bins.zip", "cache/decoder_meta.json"}


def test_context_ceiling_uses_the_lower_of_hardware_and_config_limits():
    model = _config()["model"]
    options = {"hybrid_opt_token_backend": "npu"}
    # AMD's model advertises a large hardware context, while its generation
    # config makes the supported prompt budget explicit.
    assert backend._context_ceiling(model, options, _config()["search"]) == 4096
    assert backend._context_ceiling(model, {**options, "max_length_for_kv_cache": 2048}, _config()["search"]) == 2048


def test_context_ceiling_accepts_official_chunked_16k_tokenfusion_config():
    model = _config()["model"]
    options = {
        "hybrid_opt_token_backend": "npu", "hybrid_opt_max_seq_length": "4096",
        "hybrid_opt_chunk_context": "1", "hybrid_opt_chunk_context_threshold": "1",
        "external_data_file": "model.pb.bin", "fusion_opt_io_bind_kv_cache": "1",
    }
    search = {"max_length": 16384, "chunk_size": 4096}
    assert backend._context_ceiling(model, options, search) == 16384


@pytest.mark.parametrize(("options", "search", "message"), [
    ({"hybrid_opt_chunk_context": "1"}, {"max_length": 16384, "chunk_size": 4096}, "hybrid_opt_max_seq_length"),
    ({"hybrid_opt_chunk_context": "1", "hybrid_opt_max_seq_length": 4096}, {"max_length": 16384, "chunk_size": 4097}, "chunk_size"),
    ({"hybrid_opt_chunk_context": "yes", "hybrid_opt_max_seq_length": 4096}, {"max_length": 16384, "chunk_size": 4096}, "chunk_context"),
    ({"hybrid_opt_chunk_context": 2, "hybrid_opt_max_seq_length": 4096}, {"max_length": 16384, "chunk_size": 4096}, "chunk_context"),
])
def test_context_ceiling_rejects_invalid_chunk_context_config(options, search, message):
    with pytest.raises(ValueError, match=message):
        backend._context_ceiling(_config()["model"], options, search)


@pytest.mark.parametrize(("model_limit", "search_limit"), [(4096.0, 4096), (4096, True)])
def test_context_ceiling_rejects_float_and_boolean_limits(model_limit, search_limit):
    model = _config()["model"]
    model["context_length"] = model_limit
    with pytest.raises(ValueError, match="must be an integer"):
        backend._context_ceiling(model, {"hybrid_opt_token_backend": "npu"}, {"max_length": search_limit})


@pytest.mark.parametrize("tokens", [np.asarray([-1], dtype=np.int64), np.asarray([2**31], dtype=np.int64)])
def test_token_ids_must_fit_signed_int32(tokens):
    with pytest.raises(ValueError, match="signed int32 range"):
        backend._ids(tokens, label="fixture")


class _ReferenceTokenizer:
    def apply_chat_template(self, messages, **kwargs):
        assert kwargs == {"tokenize": False, "add_generation_prompt": True, "enable_thinking": False}
        return "fixed prompt"

    def encode(self, text, add_special_tokens=False):
        assert not add_special_tokens
        known = {"fixed prompt": [10, 11], "A": [3], "B": [7]}
        if text in known:
            return known[text]
        return [10, 11, {"A": 3, "B": 7}[text[-1]]]

    def decode(self, ids):
        return {3: "A", 7: "B"}[list(ids)[0]]


class _OgaTokenizer(_ReferenceTokenizer):
    def __init__(self, *, drift=None):
        self.drift = drift

    def encode(self, text):
        if text == "fixed prompt" and self.drift == "ids":
            return np.asarray([10, 12], dtype=np.int32)
        if text == "fixed promptA" and self.drift == "boundary":
            return np.asarray([10, 99], dtype=np.int32)
        return np.asarray(super().encode(text), dtype=np.int32)

    def decode(self, ids):
        return super().decode(ids)


class _BoundaryDriftReferenceTokenizer(_ReferenceTokenizer):
    def encode(self, text, add_special_tokens=False):
        if text == "fixed promptA":
            return [10, 99]
        return super().encode(text, add_special_tokens=add_special_tokens)


class _OptionReferenceTokenizer:
    def apply_chat_template(self, messages, **kwargs):
        self.messages = messages
        assert kwargs == {"tokenize": False, "add_generation_prompt": True, "enable_thinking": False}
        return "option prompt"

    def encode(self, text, add_special_tokens=False):
        assert not add_special_tokens
        if text == "option prompt":
            return [1, 2]
        label = text[-1]
        token = 10 + LETTERS.index(label)
        return [token] if text == label else [1, 2, token]

    def decode(self, ids):
        return LETTERS[list(ids)[0] - 10]


class _OptionOgaTokenizer(_OptionReferenceTokenizer):
    def encode(self, text):
        return np.asarray(super().encode(text), dtype=np.int32)


@pytest.mark.parametrize("count", [2, 16])
def test_npu_uses_the_same_option_limit_and_prompt_as_other_backends(monkeypatch, count):
    row = _row_with_options(count)
    with pytest.raises(ValueError, match="2-16"):
        validate_row(_row_with_options(17))

    reference, oga = _OptionReferenceTokenizer(), _OptionOgaTokenizer()
    ids, slots, _ = backend._encode_prompt(backend.RyzenAiTokenizer(reference, oga), row, 4096)
    assert ids == [1, 2] and slots == list(range(10, 10 + count))
    assert reference.messages == direct_messages(row)

    class Params:
        def __init__(self, model):
            pass

        def set_search_options(self, **kwargs):
            pass

    class Generator:
        def __init__(self, model, params):
            pass

        def append_tokens(self, tokens):
            pass

        def get_logits(self):
            return np.arange(64, dtype=np.float32)[None, :]

    monkeypatch.setitem(sys.modules, "onnxruntime_genai", SimpleNamespace(GeneratorParams=Params, Generator=Generator))
    result = backend.score(object(), backend.RyzenAiTokenizer(reference, oga), row, {"context_ceiling": 4096})
    assert result["answer_token_ids"] == list(range(10, 10 + count))
    assert result["prompt_version"] == backend.PROMPT_VERSION


@pytest.mark.parametrize(("drift", "message"), [("ids", "IDs differ"), ("boundary", "boundary changes")])
def test_tokenizer_drift_fails_before_npu_inference(monkeypatch, drift, message):
    calls = []
    fake_oga = SimpleNamespace(
        GeneratorParams=lambda model: calls.append("params"),
        Generator=lambda model, params: calls.append("generator"),
    )
    monkeypatch.setitem(sys.modules, "onnxruntime_genai", fake_oga)
    adapter = backend.RyzenAiTokenizer(_ReferenceTokenizer(), _OgaTokenizer(drift=drift))

    with pytest.raises(ValueError, match=message):
        backend.score(object(), adapter, ROW, {"context_ceiling": 32})
    assert calls == []


def test_reference_tokenizer_boundary_drift_fails_before_npu_inference(monkeypatch):
    calls = []
    monkeypatch.setitem(sys.modules, "onnxruntime_genai", SimpleNamespace(
        GeneratorParams=lambda model: calls.append("params"),
        Generator=lambda model, params: calls.append("generator"),
    ))
    adapter = backend.RyzenAiTokenizer(_BoundaryDriftReferenceTokenizer(), _OgaTokenizer())

    with pytest.raises(ValueError, match="boundary changes"):
        backend.score(object(), adapter, ROW, {"context_ceiling": 32})
    assert calls == []


@pytest.mark.parametrize(("ceiling", "requested"), [(1, 32), (32, 1)])
def test_context_overflow_fails_before_allocating_a_generator(monkeypatch, ceiling, requested):
    calls = []
    monkeypatch.setitem(sys.modules, "onnxruntime_genai", SimpleNamespace(
        GeneratorParams=lambda _: calls.append("params"),
        Generator=lambda *_: calls.append("generator"),
    ))
    adapter = backend.RyzenAiTokenizer(_ReferenceTokenizer(), _OgaTokenizer())
    with pytest.raises(ValueError, match="no truncation allowed"):
        backend.score(object(), adapter, ROW, {"context_ceiling": ceiling}, max_tokens=requested)
    assert calls == []


@pytest.mark.parametrize("shape", [(1, 8), (1, 1, 8), (2, 4), (1, 2, 4)])
def test_score_reads_categorical_logits_without_generating_tokens(monkeypatch, shape):
    events = []

    class Params:
        def __init__(self, model):
            events.append("params")

        def set_search_options(self, **kwargs):
            events.append(("search", kwargs))

    class Generator:
        def __init__(self, model, params):
            events.append("generator")

        def append_tokens(self, tokens):
            assert tokens.dtype == np.int32
            assert tokens.tolist() == [10, 11]
            events.append("append")

        def get_logits(self):
            events.append("logits")
            return np.asarray([0.0, 0.0, 0.0, 2.5, 0.0, 0.0, 0.0, -1.5], dtype=np.float32).reshape(shape)

        def generate_next_token(self):
            raise AssertionError("direct categorical scoring must not generate a token")

    monkeypatch.setitem(sys.modules, "onnxruntime_genai", SimpleNamespace(
        GeneratorParams=Params, Generator=Generator,
    ))
    adapter = backend.RyzenAiTokenizer(_ReferenceTokenizer(), _OgaTokenizer())

    if shape not in ((1, 8), (1, 1, 8)):
        with pytest.raises(RuntimeError, match="Unexpected OGA logits shape"):
            backend.score(object(), adapter, ROW, {"context_ceiling": 4096})
        return
    result = backend.score(object(), adapter, ROW, {"context_ceiling": 4096}, max_tokens=8192)

    assert events == ["params", ("search", {"max_length": 2, "batch_size": 1, "do_sample": False}),
                      "generator", "append", "logits"]
    assert result["option_ids"] == ["yes", "no"]
    assert result["answer_token_ids"] == [3, 7]
    assert result["option_logits"] == pytest.approx([2.5, -1.5])
    assert result["probabilities"] == pytest.approx([0.98201379, 0.01798621])
    assert sum(result["probabilities"]) == pytest.approx(1.0)
    assert result["input_tokens"] == 2
    assert "no generated tokens" in result["readout"]


def test_score_copies_logits_before_the_generator_releases_its_buffer(monkeypatch):
    class Params:
        def __init__(self, model):
            pass

        def set_search_options(self, **kwargs):
            pass

    class Generator:
        def __init__(self, model, params):
            self.logits = np.asarray([[0.0, 0.0, 0.0, 2.5, 0.0, 0.0, 0.0, -1.5]], dtype=np.float32)

        def append_tokens(self, tokens):
            pass

        def get_logits(self):
            return self.logits

        def __del__(self):
            self.logits.fill(-100.0)

    monkeypatch.setitem(sys.modules, "onnxruntime_genai", SimpleNamespace(
        GeneratorParams=Params, Generator=Generator,
    ))
    adapter = backend.RyzenAiTokenizer(_ReferenceTokenizer(), _OgaTokenizer())

    result = backend.score(object(), adapter, ROW, {"context_ceiling": 4096})

    assert result["option_logits"] == pytest.approx([2.5, -1.5])


@pytest.mark.parametrize("invalid", [float("nan"), float("inf"), -float("inf")])
def test_nonfinite_scores_fail_and_release_the_generator(monkeypatch, invalid):
    events = []

    class Generator:
        def __init__(self, model, params):
            pass

        def append_tokens(self, tokens):
            pass

        def get_logits(self):
            return np.asarray([[0, 0, 0, invalid, 0, 0, 0, 1]], dtype=np.float32)

        def __del__(self):
            events.append("released")

    monkeypatch.setitem(sys.modules, "onnxruntime_genai", SimpleNamespace(
        GeneratorParams=lambda _: SimpleNamespace(set_search_options=lambda **_: None),
        Generator=Generator,
    ))
    adapter = backend.RyzenAiTokenizer(_ReferenceTokenizer(), _OgaTokenizer())
    with pytest.raises(RuntimeError, match="non-finite selected logits"):
        backend.score(object(), adapter, ROW, {"context_ceiling": 4096})
    assert events == ["released"]
