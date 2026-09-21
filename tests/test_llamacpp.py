import json
import os
import sys
from types import SimpleNamespace

import pytest

from semif_phase1.cli import main
from semif_phase1 import llamacpp_backend


@pytest.mark.parametrize("extra,message", [
    (["--mode", "direct", "--gguf", "x.gguf"], "--gguf requires --backend llamacpp"),
    (["--mode", "direct", "--llama-threads", "4"], "--llama-threads requires --backend llamacpp"),
    (["--mode", "direct", "--backend", "llamacpp", "--llama-threads", "0"], "must be positive"),
    (["--mode", "direct", "--llama-gpu-layers", "99"], "--llama-gpu-layers requires --backend llamacpp"),
    (["--mode", "direct", "--backend", "llamacpp", "--llama-gpu-layers", "-2"], "must be -1 or nonnegative"),
    (["--mode", "direct", "--llama-lib", "/tmp/libllama.so"], "--llama-lib requires --backend llamacpp"),
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

    captured = {}

    def load_model(source, revision, gguf, *, threads, context_tokens, n_gpu_layers, llama_lib):
        captured.update(gguf=str(gguf), threads=threads, context_tokens=context_tokens,
                        n_gpu_layers=n_gpu_layers, llama_lib=str(llama_lib) if llama_lib else None)
        return None, None, captured

    fake_backend = SimpleNamespace(
        DEFAULT_GPU_LAYERS=99,
        load_model=load_model,
        score=lambda model, tokenizer, row, metadata, max_tokens: metadata,
        SerialPrefixScorer=None, score_shared=None,
    )
    monkeypatch.setattr(semif_phase1, "llamacpp_backend", fake_backend, raising=False)
    source, output, weights, lib = (
        tmp_path / "input.jsonl", tmp_path / "output.jsonl",
        tmp_path / "model.gguf", tmp_path / "libllama.so")
    source.write_text(json.dumps({"id": "test", "state": "Evidence", "question": "Supported?",
                                 "options": [{"id": "yes", "description": "Yes"}, {"id": "no", "description": "No"}]}) + "\n")
    weights.write_bytes(b"gguf")
    lib.write_bytes(b"so")
    monkeypatch.setattr(sys, "argv", [
        "semif-score", "--backend", "llamacpp", "--mode", "direct", "--model", "unused",
        "--revision", "unused", "--gguf", str(weights), "--llama-threads", "7",
        "--llama-gpu-layers", "99", "--llama-lib", str(lib),
        "--input", str(source), "--output", str(output)])
    main()
    record = json.loads(output.read_text())
    assert record["gguf"] == str(weights)
    assert record["threads"] == 7
    assert record["context_tokens"] == 4096
    assert record["n_gpu_layers"] == 99
    assert record["llama_lib"] == str(lib)


def test_cli_defaults_gpu_layers_to_99(tmp_path, monkeypatch):
    import semif_phase1

    fake_backend = SimpleNamespace(
        DEFAULT_GPU_LAYERS=99,
        load_model=lambda source, revision, gguf, *, threads, context_tokens, n_gpu_layers, llama_lib:
            (None, None, {"n_gpu_layers": n_gpu_layers}),
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
        "--revision", "unused", "--gguf", str(weights),
        "--input", str(source), "--output", str(output)])
    main()
    assert json.loads(output.read_text())["n_gpu_layers"] == 99


def test_load_model_rejects_unpinned_remote_revision():
    with pytest.raises(ValueError, match="pinned"):
        llamacpp_backend.load_model("Qwen/Qwen3.5-4B", "main", "/nonexistent.gguf")


def test_load_model_rejects_missing_gguf(tmp_path):
    with pytest.raises(ValueError, match="GGUF checkpoint not found"):
        llamacpp_backend.load_model("Qwen/Qwen3.5-4B",
                                    "851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a",
                                    tmp_path / "absent.gguf")


def test_engine_refuses_empty_decode():
    engine = llamacpp_backend._Engine.__new__(llamacpp_backend._Engine)
    with pytest.raises(ValueError, match="empty"):
        engine._decode([], 0, 0, False)


def test_ctypes_structs_match_current_llama_abi():
    assert llamacpp_backend.ctypes.sizeof(llamacpp_backend.llama_model_params) == 88
    assert llamacpp_backend.ctypes.sizeof(llamacpp_backend.llama_context_params) == 160
    assert llamacpp_backend.ctypes.sizeof(llamacpp_backend.llama_batch) == 56
    assert llamacpp_backend.llama_model_params.n_gpu_layers.offset == 16
    assert llamacpp_backend.llama_model_params.lazy_mode.offset == 28
    assert llamacpp_backend.llama_model_params.model_shared.offset == 72


def test_resolve_llama_lib_uses_env_then_explicit(tmp_path, monkeypatch):
    missing = tmp_path / "missing.so"
    present = tmp_path / "libllama.so"
    present.write_bytes(b"so")
    monkeypatch.delenv("SEMIF_LLAMA_LIB", raising=False)
    monkeypatch.delenv("LLAMA_CPP_LIB", raising=False)
    monkeypatch.delenv("LLAMA_CPP_LIB_PATH", raising=False)
    monkeypatch.setattr(llamacpp_backend, "_UNSLOTH_LIB", missing)
    assert llamacpp_backend.resolve_llama_lib(present) == present.resolve()
    monkeypatch.setenv("SEMIF_LLAMA_LIB", str(present))
    assert llamacpp_backend.resolve_llama_lib() == present.resolve()
    monkeypatch.delenv("SEMIF_LLAMA_LIB")
    with pytest.raises(RuntimeError, match="No libllama"):
        llamacpp_backend.resolve_llama_lib()


@pytest.mark.skipif(not os.environ.get("SEMIF_LLAMACPP_GGUF"), reason="set SEMIF_LLAMACPP_GGUF to a local GGUF path")
def test_real_gguf_scores_one_decision():
    source = os.environ.get("SEMIF_LLAMACPP_SOURCE", "Qwen/Qwen3.5-4B")
    revision = os.environ.get(
        "SEMIF_LLAMACPP_REVISION", "851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a")
    threads = int(os.environ.get("SEMIF_LLAMACPP_THREADS", "8"))
    layers = int(os.environ.get("SEMIF_LLAMACPP_GPU_LAYERS", "99"))
    model, tokenizer, metadata = llamacpp_backend.load_model(
        source, revision, os.environ["SEMIF_LLAMACPP_GGUF"], threads=threads,
        n_gpu_layers=layers)
    row = {
        "id": "smoke",
        "state": "The deployment completed at 14:02 UTC. Health checks passed in all three zones.",
        "question": "Is there evidence that the deployment succeeded?",
        "options": [{"id": "yes", "description": "The deployment succeeded."},
                    {"id": "no", "description": "The deployment did not succeed."}],
    }
    try:
        result = llamacpp_backend.score(model, tokenizer, row, metadata)
    finally:
        model.close()
    assert abs(sum(result["probabilities"]) - 1.0) < 1e-6
    assert result["probabilities"][0] > result["probabilities"][1]
    assert metadata["backend"] == "llamacpp"
    assert metadata["n_gpu_layers"] == layers
    assert result["readout"].startswith("quantized last-position logits")
