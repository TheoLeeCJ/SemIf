import json
import os
import sys
from types import SimpleNamespace

import numpy
import pytest

from semif_phase1.cli import main


@pytest.mark.parametrize("extra,message", [
    (["--mode", "direct", "--gguf", "x.gguf"], "--gguf requires --backend llamacpp"),
    (["--mode", "direct", "--llama-threads", "4"], "--llama-threads requires --backend llamacpp"),
    (["--mode", "direct", "--backend", "llamacpp", "--llama-threads", "0"], "must be positive"),
    (["--mode", "direct", "--llama-gpu-layers", "99"], "--llama-gpu-layers requires --backend llamacpp"),
    (["--mode", "direct", "--backend", "llamacpp", "--llama-gpu-layers", "-1"], "must be nonnegative"),
    (["--mode", "reranker", "--backend", "llamacpp", "--gguf", "x.gguf"], "reranker requires torch"),
    (["--mode", "direct", "--backend", "llamacpp"], "requires --gguf"),
    (["--mode", "direct", "--backend", "llamacpp", "--gguf", "missing.gguf"], "requires --gguf"),
])
def test_invalid_llamacpp_combinations_fail_before_loading(tmp_path, monkeypatch, capsys, extra, message):
    monkeypatch.setattr(sys, "argv", ["semif-score", "--model", "unused", "--revision", "unused",
                                    "--input", "missing.jsonl", "--output", str(tmp_path / "out.jsonl"), *extra])
    with pytest.raises(SystemExit) as error:
        main()
    assert error.value.code == 2
    assert message in capsys.readouterr().err
    assert not (tmp_path / "out.jsonl").exists()


def test_cli_passes_gguf_options_to_loader(tmp_path, monkeypatch):
    import semif_phase1

    fake_backend = SimpleNamespace(
        load_model=lambda source, revision, gguf, *, threads, context_tokens, gpu_layers:
            (None, None, {"gguf": str(gguf), "threads": threads, "context_tokens": context_tokens,
                          "n_gpu_layers": gpu_layers}),
        score=lambda model, tokenizer, row, metadata, max_tokens: metadata,
        SerialPrefixScorer=None, score_shared=None,
    )
    monkeypatch.setattr(semif_phase1, "llamacpp_backend", fake_backend, raising=False)
    source, output, weights = tmp_path / "input.jsonl", tmp_path / "output.jsonl", tmp_path / "model.gguf"
    source.write_text(json.dumps({"id": "test", "state": "Evidence", "question": "Supported?",
                                 "options": [{"id": "yes", "description": "Yes"}, {"id": "no", "description": "No"}]}) + "\n")
    weights.write_bytes(b"gguf")
    monkeypatch.setattr(sys, "argv", [
        "semif-score", "--backend", "llamacpp", "--mode", "direct", "--model", "unused",
        "--revision", "unused", "--gguf", str(weights), "--llama-threads", "7",
        "--llama-gpu-layers", "20", "--input", str(source), "--output", str(output)])
    main()
    record = json.loads(output.read_text())
    assert record["gguf"] == str(weights)
    assert record["threads"] == 7
    assert record["context_tokens"] == 4096
    assert record["n_gpu_layers"] == 20


def test_load_model_rejects_unpinned_remote_revision():
    from semif_phase1 import llamacpp_backend

    with pytest.raises(ValueError, match="pinned"):
        llamacpp_backend.load_model("Qwen/Qwen3.5-4B", "main", "/nonexistent.gguf")


def test_load_model_rejects_missing_gguf(tmp_path):
    from semif_phase1 import llamacpp_backend

    with pytest.raises(ValueError, match="GGUF checkpoint not found"):
        llamacpp_backend.load_model("Qwen/Qwen3.5-4B",
                                    "851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a",
                                    tmp_path / "absent.gguf")


def test_engine_refuses_empty_decode():
    from semif_phase1.llamacpp_backend import _Engine

    engine = _Engine.__new__(_Engine)
    with pytest.raises(ValueError, match="empty"):
        engine._decode([], 0, 0, False)


def test_model_params_initialize_once_and_offload_only_what_is_asked(monkeypatch):
    from semif_phase1 import llamacpp_backend

    calls = []
    library = SimpleNamespace(
        llama_backend_init=lambda: calls.append("init"),
        llama_model_default_params=lambda: SimpleNamespace(n_gpu_layers=-1),
    )
    monkeypatch.setattr(llamacpp_backend, "_BACKEND_INITIALIZED", False)
    first = llamacpp_backend._model_params(library)
    second = llamacpp_backend._model_params(library, 20)
    assert calls == ["init"]
    assert first.n_gpu_layers == 0
    assert second.n_gpu_layers == 20


@pytest.mark.parametrize("layers", [-1, 1.5, True])
def test_load_model_rejects_invalid_gpu_layers(tmp_path, layers):
    from semif_phase1 import llamacpp_backend

    weights = tmp_path / "model.gguf"
    weights.write_bytes(b"gguf")
    with pytest.raises(ValueError, match="gpu_layers"):
        llamacpp_backend.load_model("Qwen/Qwen3.5-4B", "851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a",
                                    weights, gpu_layers=layers)


def test_gpu_layers_on_a_cpu_only_build_fail_instead_of_scoring_on_the_cpu(tmp_path, monkeypatch):
    from semif_phase1 import llamacpp_backend

    weights = tmp_path / "model.gguf"
    weights.write_bytes(b"gguf")
    loaded = []
    cpu_only = SimpleNamespace(llama_supports_gpu_offload=lambda: False,
                               llama_model_load_from_file=lambda *args: loaded.append(args))
    monkeypatch.setitem(sys.modules, "llama_cpp", cpu_only)
    with pytest.raises(RuntimeError, match="GPU offload"):
        llamacpp_backend.load_model("Qwen/Qwen3.5-4B", "851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a",
                                    weights, gpu_layers=20)
    assert not loaded


def test_restore_state_rejects_native_failure():
    from semif_phase1.llamacpp_backend import _Engine

    engine = _Engine.__new__(_Engine)
    engine.context = object()
    engine.memory = object()
    engine.lib = SimpleNamespace(
        llama_memory_seq_rm=lambda *args: True,
        llama_state_seq_set_data=lambda *args: 0,
    )
    with pytest.raises(RuntimeError, match="restore"):
        engine.restore_state((b"state", 5))


def test_engine_records_actual_native_context_size():
    from semif_phase1.llamacpp_backend import _Engine

    context = object()
    library = SimpleNamespace(
        llama_context_default_params=lambda: SimpleNamespace(),
        llama_init_from_model=lambda model, params: context,
        llama_get_memory=lambda value: object(),
        llama_n_ctx=lambda value: 4352,
        llama_model_get_vocab=lambda model: object(),
        llama_n_vocab=lambda vocab: 248320,
    )
    engine = _Engine(library, object(), 4160, 4)
    assert engine.context_tokens == 4352


def test_serial_cache_keys_on_exact_prefix_tokens(monkeypatch):
    from semif_phase1 import llamacpp_backend

    class FakeEngine:
        def __init__(self):
            self.prefills = []

        def clear(self):
            pass

        def prefill(self, prefix):
            self.prefills.append(prefix.copy())

        def save_state(self):
            return b"state", 5

        def restore_state(self, state):
            pass

        def branch_logits(self, prefix_length, suffix):
            return numpy.asarray([2.0, 1.0, 0.0])

    class FakeModel:
        def __init__(self):
            self.engine = FakeEngine()

        def encode_verified(self, row, max_tokens):
            prefix = [1 if key == "a" else 2 for key in row["state"]]
            return prefix + [9], [0, 1], "hash"

    monkeypatch.setattr(
        llamacpp_backend,
        "_state_prefix",
        lambda tokenizer, state: [1 if key == "a" else 2 for key in state],
    )
    model = FakeModel()
    scorer = llamacpp_backend.SerialPrefixScorer(model, object(), {})

    def row(identifier, state):
        return {
            "id": identifier,
            "state": state,
            "question": "Which option?",
            "options": [{"id": "a", "description": "A"}, {"id": "b", "description": "B"}],
        }

    first = scorer.score(row("first", {"a": 1, "b": 2}))
    reordered = {"b": 2, "a": 1}  # Dict-equal state, different serialized/tokenized prefix.
    second = scorer.score(row("second", reordered))
    third = scorer.score(row("third", reordered))
    assert [first["cache_hit"], second["cache_hit"], third["cache_hit"]] == [False, False, True]
    assert model.engine.prefills == [[1, 2], [2, 1]]


@pytest.mark.skipif(not os.environ.get("SEMIF_LLAMACPP_GGUF"), reason="set SEMIF_LLAMACPP_GGUF to a local GGUF path")
def test_real_gguf_scores_direct_serial_and_shared():
    from semif_phase1 import llamacpp_backend

    source = os.environ.get("SEMIF_LLAMACPP_SOURCE", "Qwen/Qwen3.5-4B")
    revision = os.environ.get(
        "SEMIF_LLAMACPP_REVISION", "851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a")
    threads = int(os.environ.get("SEMIF_LLAMACPP_THREADS", "8"))
    gpu_layers = int(os.environ.get("SEMIF_LLAMACPP_GPU_LAYERS", "0"))
    model, tokenizer, metadata = llamacpp_backend.load_model(
        source, revision, os.environ["SEMIF_LLAMACPP_GGUF"], threads=threads, gpu_layers=gpu_layers)
    state = "The deployment completed at 14:02 UTC. Health checks passed in all three zones."
    rows = [
        {
            "id": "deployment",
            "state": state,
            "question": "Is there evidence that the deployment succeeded?",
            "options": [{"id": "yes", "description": "The deployment succeeded."},
                        {"id": "no", "description": "The deployment did not succeed."}],
        },
        {
            "id": "health",
            "state": state,
            "question": "Did the health checks pass?",
            "options": [{"id": "yes", "description": "The checks passed."},
                        {"id": "no", "description": "The checks failed."}],
        },
    ]
    try:
        direct = [llamacpp_backend.score(model, tokenizer, row, metadata) for row in rows]
        serial_scorer = llamacpp_backend.SerialPrefixScorer(model, tokenizer, metadata)
        serial = [serial_scorer.score(row) for row in rows]
        shared, _ = llamacpp_backend.score_shared(model, tokenizer, rows, metadata)
        choices = lambda results: [
            result["option_ids"][int(numpy.argmax(result["probabilities"]))] for result in results
        ]
        assert choices(direct) == choices(serial) == choices(shared) == ["yes", "yes"]
        assert [result["cache_hit"] for result in serial] == [False, True]
        for serial_result, shared_result in zip(serial, shared):
            numpy.testing.assert_allclose(serial_result["option_logits"], shared_result["option_logits"])
        assert metadata["backend"] == "llamacpp"
        assert metadata["n_gpu_layers"] == gpu_layers
        assert metadata["max_prompt_tokens"] == 4096
        assert metadata["context_tokens"] >= metadata["max_prompt_tokens"]
    finally:
        model.close()
