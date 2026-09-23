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
    (["--mode", "reranker", "--backend", "llamacpp", "--gguf", "x.gguf"], "reranker requires torch"),
    (["--mode", "direct", "--backend", "llamacpp"], "requires --gguf"),
    (["--mode", "direct", "--backend", "llamacpp", "--gguf", "missing.gguf"], "requires --gguf"),
    (["--mode", "direct", "--llama-gpu-layers", "99"], "--llama-gpu-layers requires --backend llamacpp"),
    (["--mode", "direct", "--llama-parallel", "4"], "--llama-parallel requires --backend llamacpp"),
    (["--mode", "shared", "--llama-readout", "marginal"], "--llama-readout requires --backend llamacpp"),
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
        load_model=lambda source, revision, gguf, *, threads, context_tokens, gpu_layers, sequences, readout:
            (None, None, {"gguf": str(gguf), "threads": threads, "context_tokens": context_tokens,
                          "gpu_layers": gpu_layers, "sequences": sequences, "readout": readout}),
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
        "--input", str(source), "--output", str(output)])
    main()
    record = json.loads(output.read_text())
    assert record["gguf"] == str(weights)
    assert record["threads"] == 7
    assert record["context_tokens"] == 4096
    assert record["gpu_layers"] == 0 and record["sequences"] == 1 and record["readout"] == "last"


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


def test_model_params_initialize_once_and_apply_offload(monkeypatch):
    from semif_phase1 import llamacpp_backend

    calls = []
    library = SimpleNamespace(
        llama_backend_init=lambda: calls.append("init"),
        llama_model_default_params=lambda: SimpleNamespace(n_gpu_layers=-1),
    )
    monkeypatch.setattr(llamacpp_backend, "_BACKEND_INITIALIZED", False)
    cpu = llamacpp_backend._model_params(library, 0)
    gpu = llamacpp_backend._model_params(library, 99)
    assert calls == ["init"]
    assert cpu.n_gpu_layers == 0 and gpu.n_gpu_layers == 99


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

    # The same decisions through copied-sequence fan-out must match the restore path.
    model, tokenizer, metadata = llamacpp_backend.load_model(
        source, revision, os.environ["SEMIF_LLAMACPP_GGUF"], threads=threads, sequences=3,
        gpu_layers=gpu_layers)
    try:
        assert metadata["n_seq_max"] == 3
        fanned, timing = llamacpp_backend.score_shared(model, tokenizer, rows, metadata)
        assert timing["serving_config"] == "llamacpp-seq-copy-parallel-v1"
        assert timing["branch_state_bytes"] == 0
        assert choices(fanned) == ["yes", "yes"]
        for shared_result, fanned_result in zip(shared, fanned):
            numpy.testing.assert_allclose(
                shared_result["option_logits"], fanned_result["option_logits"], rtol=0, atol=0.5)
        marginal_meta = {**metadata, "readout_mode": "marginal"}
        folded, timing = llamacpp_backend.score_shared(model, tokenizer, rows, marginal_meta)
        assert timing["serving_config"] == "llamacpp-seq-copy-parallel-marginal-v1"
        assert choices(folded) == ["yes", "yes"]
        assert all(0.0 <= result["preamble_mass"] <= 1.0 for result in folded)
    finally:
        model.close()


def test_load_model_validates_new_arguments(tmp_path):
    from semif_phase1 import llamacpp_backend

    weights = tmp_path / "model.gguf"
    weights.write_bytes(b"gguf")
    revision = "851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a"
    with pytest.raises(ValueError, match="gpu_layers"):
        llamacpp_backend.load_model("Qwen/Qwen3.5-4B", revision, weights, gpu_layers=-1)
    with pytest.raises(ValueError, match="sequences"):
        llamacpp_backend.load_model("Qwen/Qwen3.5-4B", revision, weights, sequences=0)
    with pytest.raises(ValueError, match="readout"):
        llamacpp_backend.load_model("Qwen/Qwen3.5-4B", revision, weights, readout="first")


def _fake_engine_for_fanout(sequences: int, vocab_size: int = 8):
    """An _Engine whose native calls are recorded instead of executed."""
    from semif_phase1.llamacpp_backend import _Engine

    calls = {"seq_cp": [], "seq_rm": [], "batches": []}

    class Batch:
        def __init__(self, n, seqs):
            self.token = [0] * n
            self.pos = [0] * n
            self.n_seq_id = [0] * n
            self.seq_id = [[0] * seqs for _ in range(n)]
            self.logits = [0] * n
            self.n_tokens = 0

    def decode(context, batch):
        calls["batches"].append([(batch.token[i], batch.pos[i], batch.seq_id[i][0], batch.logits[i])
                                 for i in range(batch.n_tokens)])
        return 0

    def logits_ith(context, index):
        # Encode the branch id in the logits so the caller can tell rows apart.
        batch = calls["batches"][-1]
        branch = batch[index][2]
        array = (ctypes_float * vocab_size)(*([0.0] * vocab_size))
        array[branch] = 10.0
        return array

    import ctypes as _ctypes
    ctypes_float = _ctypes.c_float
    library = SimpleNamespace(
        llama_batch_init=lambda n, embd, seqs: Batch(n, seqs),
        llama_batch_free=lambda batch: None,
        llama_decode=decode,
        llama_get_logits_ith=logits_ith,
        llama_memory_seq_cp=lambda memory, src, dst, p0, p1: calls["seq_cp"].append((src, dst, p0, p1)),
        llama_memory_seq_rm=lambda memory, seq, p0, p1: calls["seq_rm"].append((seq, p0, p1)) or True,
    )
    engine = _Engine.__new__(_Engine)
    engine.lib = library
    engine.context = object()
    engine.memory = object()
    engine.sequences = sequences
    engine.vocab_size = vocab_size
    return engine, calls


def test_fanout_copies_prefix_decodes_once_and_removes_branches_whole():
    engine, calls = _fake_engine_for_fanout(sequences=4)
    vocabularies = engine.fanout_logits(prefix_length=10, suffixes=[[5, 6], [7], [8, 9, 3]])
    # one copy of sequence 0 per branch, whole range
    assert calls["seq_cp"] == [(0, 1, -1, -1), (0, 2, -1, -1), (0, 3, -1, -1)]
    # a single decode call carrying every suffix token, positions continuing the prefix
    assert len(calls["batches"]) == 1
    batch = calls["batches"][0]
    assert [(t, p, s) for t, p, s, _ in batch] == [
        (5, 10, 1), (6, 11, 1), (7, 10, 2), (8, 10, 3), (9, 11, 3), (3, 12, 3)]
    # logits requested only at each branch's last token
    assert [want for _, _, _, want in batch] == [0, 1, 1, 0, 0, 1]
    # each returned row belongs to its branch
    assert [int(v.argmax()) for v in vocabularies] == [1, 2, 3]
    # branches removed whole afterwards, sequence 0 untouched
    assert calls["seq_rm"] == [(1, -1, -1), (2, -1, -1), (3, -1, -1)]


def test_fanout_refuses_more_branches_than_slots():
    engine, _ = _fake_engine_for_fanout(sequences=2)
    with pytest.raises(ValueError, match="exceed"):
        engine.fanout_logits(3, [[1], [2]])


def test_marginal_readout_folds_preamble_mass_into_slots():
    from semif_phase1.llamacpp_backend import _marginal_readout

    probed = {}

    class Engine:
        def probe_branches(self, positions, tokens):
            probed.update(tokens)
            # After the preamble token the model answers slot 1 (token id 1) with certainty.
            out = numpy.full(8, -20.0)
            out[1] = 20.0
            return {branch: out.copy() for branch in tokens}

    # Branch 1: the model puts its mass on token 7 (a preamble), almost none on the slots 0 and 1.
    vocabulary = numpy.full(8, -20.0)
    vocabulary[7] = 20.0
    vocabulary[0] = 1.0
    vocabulary[1] = 0.0
    (logits, preamble_mass), = _marginal_readout(
        Engine(), branches=[1], prefix_length=10, suffixes=[[4, 4]],
        vocabularies=[vocabulary], slot_lists=[[0, 1]])
    assert probed == {1: 7}
    # Direct read alone would have picked slot 0; the preamble path moves the answer to slot 1.
    assert int(numpy.argmax(logits)) == 1
    assert preamble_mass > 0.99


def test_marginal_readout_skips_branches_that_already_answer_with_a_slot():
    from semif_phase1.llamacpp_backend import _marginal_readout

    class Engine:
        def probe_branches(self, positions, tokens):
            raise AssertionError("no probe should be issued")

    vocabulary = numpy.full(8, -20.0)
    vocabulary[0] = 20.0  # slot 0 is the confident next token; nothing else clears the floor
    (logits, preamble_mass), = _marginal_readout(
        Engine(), branches=[1], prefix_length=3, suffixes=[[9]],
        vocabularies=[vocabulary], slot_lists=[[0, 1]])
    assert int(numpy.argmax(logits)) == 0 and preamble_mass == 0.0
