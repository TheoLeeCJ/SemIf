"""CPU option readout over GGUF checkpoints through llama.cpp.

Prompt construction and answer-slot verification stay on the reference
transformers tokenizer, so prompt_sha256 matches the Torch backend exactly;
llama.cpp only executes the forward pass over the quantized GGUF weights.
Every scored prompt is re-tokenized through the GGUF vocabulary and must
agree with the reference encoding before it is evaluated.

Sequence 0 of the single context holds the prefill; each decision restores the
saved prefix state (whole-sequence save/restore) before decoding its suffix.
The hybrid linear-attention memory of Qwen3.5 supports neither sequence
copies nor partial tail removal, so branch replication goes through the
per-sequence state serialization llama.cpp itself uses for slot caching.
"""

from __future__ import annotations

import ctypes
import hashlib
import json
import os
from pathlib import Path
import re
import time

import numpy

from .core import LETTERS, digest, direct_messages, softmax, validate_row
from .direct import PROMPT_VERSION, encode_prompt
from .reranker import PROMPT_VERSION as RERANKER_PROMPT_VERSION
from .reranker import _answer_ids, render_pair
from .shared import _state_prefix

DECODE_CHUNK = 512


def _render(tokenizer, row: dict) -> str:
    return tokenizer.apply_chat_template(
        direct_messages(row), tokenize=False, add_generation_prompt=True, enable_thinking=False
    )


def _gguf_tokenize(library, vocab, text: str) -> list[int]:
    data = text.encode("utf-8")
    needed = library.llama_tokenize(vocab, data, len(data), None, 0, False, True)
    if needed < 0:
        needed = -needed
    tokens = (library.llama_token * needed)()
    written = library.llama_tokenize(vocab, data, len(data), tokens, needed, False, True)
    if written < 0:
        raise RuntimeError("The GGUF tokenizer rejected the prompt text")
    return list(tokens[:written])


def _gguf_piece(library, vocab, token: int) -> bytes:
    buffer = ctypes.create_string_buffer(64)
    written = library.llama_token_to_piece(vocab, token, buffer, len(buffer), 0, True)
    if written < 0:
        raise RuntimeError("The GGUF tokenizer cannot render a token")
    return buffer.raw[:written]


def _logsumexp(values: numpy.ndarray) -> float:
    peak = float(values.max())
    return peak + float(numpy.log(numpy.exp(values - peak).sum()))


class _Engine:
    """One llama.cpp context bound to a loaded GGUF model."""

    def __init__(self, library, model, context_tokens: int, threads: int):
        params = library.llama_context_default_params()
        params.n_ctx = context_tokens
        params.n_seq_max = 2
        # llama.cpp requires n_outputs_max >= n_seq_max * n_outputs_max_per_seq.
        params.n_outputs_max = 2
        params.n_threads = threads
        params.n_threads_batch = threads
        self.lib = library
        self.model = model
        self.context = library.llama_init_from_model(model, params)
        if not self.context:
            raise RuntimeError("llama.cpp failed to create the scoring context")
        self.memory = library.llama_get_memory(self.context)
        self.vocab_size = library.llama_n_vocab(library.llama_model_get_vocab(model))

    def _decode(self, tokens: list[int], start: int, sequence: int, want_logits: bool):
        if not tokens:
            raise ValueError("Refusing to decode an empty token list")
        total = len(tokens)
        for offset in range(0, total, DECODE_CHUNK):
            chunk = tokens[offset : offset + DECODE_CHUNK]
            batch = self.lib.llama_batch_init(len(chunk), 0, 1)
            try:
                for index in range(len(chunk)):
                    batch.token[index] = chunk[index]
                    batch.pos[index] = start + offset + index
                    batch.n_seq_id[index] = 1
                    batch.seq_id[index][0] = sequence
                    batch.logits[index] = int(want_logits and offset + index == total - 1)
                batch.n_tokens = len(chunk)
                if self.lib.llama_decode(self.context, batch):
                    raise RuntimeError("llama_decode failed; raise --max-tokens if prompts grew")
            finally:
                self.lib.llama_batch_free(batch)
        if not want_logits:
            return None
        pointer = self.lib.llama_get_logits_ith(self.context, -1)
        if not pointer:
            raise RuntimeError("llama.cpp returned no logits for the flagged position")
        return numpy.ctypeslib.as_array(
            ctypes.cast(pointer, ctypes.POINTER(ctypes.c_float)), shape=(self.vocab_size,)
        ).copy()

    def clear(self) -> None:
        self.lib.llama_memory_clear(self.memory, False)

    def prefill(self, prefix: list[int]) -> None:
        self._decode(prefix, 0, 0, False)

    def save_state(self):
        """Snapshot sequence 0 for repeated branch restores."""
        size = self.lib.llama_state_seq_get_size(self.context, 0)
        buffer = (ctypes.c_ubyte * size)()
        if self.lib.llama_state_seq_get_data(self.context, buffer, size, 0) != size:
            raise RuntimeError("llama.cpp wrote an incomplete prefix state")
        return buffer, size

    def restore_state(self, state) -> None:
        buffer, size = state
        if not self.lib.llama_memory_seq_rm(self.memory, 0, -1, -1):
            raise RuntimeError("llama.cpp could not drop the previous scored branch")
        self.lib.llama_state_seq_set_data(self.context, buffer, size, 0)

    def branch_logits(self, prefix_length: int, suffix: list[int]) -> numpy.ndarray:
        return self._decode(suffix, prefix_length, 0, True)

    def full_logits(self, tokens: list[int]) -> numpy.ndarray:
        self.clear()
        return self._decode(tokens, 0, 0, True)


class _Backend:
    """Verified scoring adapter around one llama.cpp engine."""

    def __init__(self, engine: _Engine, vocab, tokenizer):
        self.engine = engine
        self.vocab = vocab
        self.tokenizer = tokenizer

    def encode_verified(self, row: dict, max_tokens: int):
        ids, slots, prompt_hash = encode_prompt(self.tokenizer, row, max_tokens)
        if _gguf_tokenize(self.engine.lib, self.vocab, _render(self.tokenizer, row)) != ids:
            raise ValueError(f"Row {row['id']}: GGUF tokenization disagrees with the reference tokenizer")
        return ids, slots, prompt_hash


def _verify_vocabulary(tokenizer, library, vocab) -> None:
    """Fail early when the GGUF vocabulary is not the tokenizer's own."""
    row = {
        "id": "vocabulary-probe",
        "state": "probe evidence",
        "question": "probe criterion?",
        "options": [{"id": "yes", "description": "Yes."}, {"id": "no", "description": "No."}],
    }
    prompt = _render(tokenizer, row)
    reference = tokenizer.encode(prompt, add_special_tokens=False)
    if _gguf_tokenize(library, vocab, prompt) != reference:
        raise RuntimeError("The GGUF vocabulary disagrees with the reference tokenizer")
    for letter in LETTERS:
        encoded = tokenizer.encode(letter, add_special_tokens=False)
        if len(encoded) != 1 or _gguf_piece(library, vocab, encoded[0]) != letter.encode():
            raise RuntimeError(f"Answer slot {letter!r} is not a shared single token")


def load_model(source: str, revision: str, gguf, *, threads: int | None = None,
               context_tokens: int = 4096):
    """Load one pinned reference tokenizer plus a local GGUF checkpoint for CPU scoring."""
    local = Path(source).is_dir()
    if not local and not re.fullmatch(r"[0-9a-f]{40}", revision or ""):
        raise ValueError("Remote sources require a pinned 40-character revision; local sources require a revision label")
    if local and not revision:
        raise ValueError("Local sources require an explicit revision label")
    gguf = Path(gguf)
    if not gguf.is_file():
        raise ValueError(f"GGUF checkpoint not found: {gguf}")
    if not (isinstance(context_tokens, int) and context_tokens > 0):
        raise ValueError("context_tokens must be a positive integer")
    if threads is None:
        threads = os.cpu_count() or 4
    if not (isinstance(threads, int) and threads >= 1):
        raise ValueError("threads must be a positive integer")
    import transformers
    try:
        import llama_cpp
    except ImportError as error:
        raise RuntimeError("Install the llama.cpp extra: pip install -e '.[test,llamacpp]'") from error

    offline = bool(os.environ.get("HF_HUB_OFFLINE"))
    common = {"revision": None if local else revision,
              "local_files_only": local or offline, "trust_remote_code": False}
    tokenizer = transformers.AutoTokenizer.from_pretrained(source, **common)
    model = llama_cpp.llama_model_load_from_file(str(gguf).encode("utf-8"), llama_cpp.llama_model_default_params())
    if not model:
        raise RuntimeError(f"llama.cpp failed to load the GGUF checkpoint: {gguf}")
    window = context_tokens + 64
    engine = _Engine(llama_cpp, model, window, threads)
    vocab = llama_cpp.llama_model_get_vocab(model)
    _verify_vocabulary(tokenizer, llama_cpp, vocab)
    checksum = hashlib.sha256()
    with gguf.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            checksum.update(block)
    metadata = {
        "source": source,
        "revision": revision,
        "backend": "llamacpp",
        "dtype": "gguf-quantized",
        "gguf": {"file": gguf.name, "bytes": gguf.stat().st_size, "sha256": checksum.hexdigest()},
        "vocab_size": engine.vocab_size,
        "threads": threads,
        "context_tokens": window,
        "decode_chunk": DECODE_CHUNK,
        "llama_cpp_python_version": llama_cpp.__version__,
        "transformers_version": transformers.__version__,
    }
    return _Backend(engine, vocab, tokenizer), tokenizer, metadata


def _result(row: dict, encoded, selected: list[float], vocabulary, metadata: dict, config: str, readout: str) -> dict:
    ids, slots, prompt_hash = encoded
    return {
        "id": row["id"],
        "option_ids": [option["id"] for option in row["options"]],
        "probabilities": softmax(selected),
        "option_logits": selected,
        "answer_token_ids": slots,
        "input_tokens": len(ids),
        "allowed_token_mass": float(numpy.exp(_logsumexp(numpy.asarray(selected)) - _logsumexp(vocabulary))),
        "full_vocab_argmax_id": int(vocabulary.argmax()),
        "prompt_sha256": prompt_hash,
        "prompt_version": PROMPT_VERSION,
        "model": {**metadata, "serving_config": config},
        "readout": readout,
        "probability_status": "conditional option score over quantized weights; uncalibrated as decision confidence",
    }


def score(model, tokenizer, row: dict, metadata: dict, max_tokens: int = 4096) -> dict:
    started = time.perf_counter()
    encoded = model.encode_verified(row, max_tokens)
    mark = time.perf_counter()
    vocabulary = model.engine.full_logits(encoded[0])
    selected = vocabulary[encoded[1]].tolist()
    result = _result(
        row, encoded, selected, vocabulary, metadata, "llamacpp-direct-v1",
        "quantized last-position logits restricted to declared answer slots; no generated tokens",
    )
    result.update(forward_seconds=time.perf_counter() - mark, total_seconds=time.perf_counter() - started)
    return result


def reranker_score(model, tokenizer, row: dict, metadata: dict, max_tokens: int = 4096) -> dict:
    """Score each row/option relevance pair through the quantized reranker prompt."""
    validate_row(row)
    no_id, yes_id = _answer_ids(tokenizer)
    for token, answer in ((no_id, b"no"), (yes_id, b"yes")):
        if _gguf_piece(model.engine.lib, model.vocab, token) != answer:
            raise RuntimeError(f"The GGUF vocabulary does not carry the reranker {answer.decode()!r} answer token")
    started = time.perf_counter()
    scored, forward_seconds = [], 0.0
    for option in row["options"]:
        text = render_pair(row, option)
        ids = tokenizer.encode(text, add_special_tokens=False)
        if not ids or len(ids) > max_tokens:
            raise ValueError(f"Row {row['id']} option {option['id']}: {len(ids)} tokens exceed limit {max_tokens}")
        if _gguf_tokenize(model.engine.lib, model.vocab, text) != ids:
            raise ValueError(f"Row {row['id']} option {option['id']}: GGUF tokenization disagrees with the reference tokenizer")
        mark = time.perf_counter()
        vocabulary = model.engine.full_logits(ids)
        forward_seconds += time.perf_counter() - mark
        odds = float(vocabulary[yes_id]) - float(vocabulary[no_id])
        weights = numpy.exp(numpy.asarray([0.0, odds], dtype=numpy.float64))
        scored.append(
            {
                "option_id": option["id"],
                "log_odds": odds,
                "binary_relevance": float(weights[1] / weights.sum()),
                "input_tokens": len(ids),
                "prompt_sha256": digest(text),
            }
        )
    log_odds = [item["log_odds"] for item in scored]
    binary = [item["binary_relevance"] for item in scored]
    return {
        "id": row["id"],
        "option_ids": [option["id"] for option in row["options"]],
        "probabilities": softmax(log_odds),
        "option_logits": log_odds,
        "independent_binary_relevance": binary,
        "input_tokens": sum(item["input_tokens"] for item in scored),
        "max_option_input_tokens": max(item["input_tokens"] for item in scored),
        "forward_seconds": forward_seconds,
        "total_seconds": time.perf_counter() - started,
        "option_prompt_sha256": [item["prompt_sha256"] for item in scored],
        "pair_batches": [{
            "pair_batch_size": len(scored),
            "forward_seconds": forward_seconds,
            "padded_tokens": sum(item["input_tokens"] for item in scored),
            "execution": "sequential-per-pair",
        }],
        "prompt_version": RERANKER_PROMPT_VERSION,
        "model": {**metadata, "serving_config": "llamacpp-reranker-v1"},
        "readout": "quantized yes/no log-odds per option, normalized only for relative comparison",
        "probability_status": "relative option compatibility over quantized weights; uncalibrated as categorical probability",
    }


class SerialPrefixScorer:
    """Cache the current state once, then score restored-state branch suffixes."""

    def __init__(self, model, tokenizer, metadata: dict, max_tokens: int = 4096):
        self.model = model
        self.tokenizer = tokenizer
        self.metadata = {**metadata, "serving_config": "llamacpp-state-restore-v1"}
        self.max_tokens = max_tokens
        self.state = None
        self.prefix = None
        self.state_data = None

    def score(self, row: dict) -> dict:
        started = time.perf_counter()
        encoded = self.model.encode_verified(row, self.max_tokens)
        ids, slots, _ = encoded
        prefix = _state_prefix(self.tokenizer, row["state"])
        hit = self.state_data is not None and row["state"] == self.state
        if not prefix or ids[: len(prefix)] != prefix or len(ids) <= len(prefix):
            raise ValueError("State prefix does not match the full prompt")
        prefill_seconds = 0.0
        if not hit:
            mark = time.perf_counter()
            self.model.engine.clear()
            self.model.engine.prefill(prefix)
            prefill_seconds = time.perf_counter() - mark
            self.state, self.prefix = row["state"], prefix
            self.state_data = self.model.engine.save_state()
        mark = time.perf_counter()
        self.model.engine.restore_state(self.state_data)
        copy_seconds = time.perf_counter() - mark
        mark = time.perf_counter()
        vocabulary = self.model.engine.branch_logits(len(prefix), ids[len(prefix) :])
        suffix_seconds = time.perf_counter() - mark
        selected = vocabulary[slots].tolist()
        result = _result(
            row, encoded, selected, vocabulary, self.metadata,
            "llamacpp-state-restore-v1", "quantized branch last-position logits over a restored prefix state",
        )
        result.update(
            cache_hit=hit,
            prefix_tokens=len(prefix),
            prefix_sha256=hashlib.sha256(json.dumps(prefix).encode()).hexdigest(),
            branch_state_bytes=self.state_data[1],
            prefill_seconds=prefill_seconds,
            copy_seconds=copy_seconds,
            suffix_forward_seconds=suffix_seconds,
            forward_seconds=prefill_seconds + suffix_seconds,
            total_seconds=time.perf_counter() - started,
        )
        return result


def score_shared(model, tokenizer, rows: list[dict], metadata: dict, max_tokens: int = 4096):
    """Prefill one exact state once, then score every criterion from restored branches."""
    if not rows or any(row["state"] != rows[0]["state"] for row in rows[1:]):
        raise ValueError("Shared scoring requires one nonempty exact state")
    if len({row["id"] for row in rows}) != len(rows):
        raise ValueError("Decision IDs must be unique")
    started = time.perf_counter()
    encoded = [model.encode_verified(row, max_tokens) for row in rows]
    prefix = _state_prefix(tokenizer, rows[0]["state"])
    if not prefix or any(ids[: len(prefix)] != prefix or len(ids) <= len(prefix) for ids, _, _ in encoded):
        raise ValueError("The fixed state prefix does not match every full prompt")
    encode_seconds = time.perf_counter() - started
    mark = time.perf_counter()
    model.engine.clear()
    model.engine.prefill(prefix)
    state_data = model.engine.save_state()
    prefill_seconds = time.perf_counter() - mark
    copy_seconds = suffix_seconds = 0.0
    results = []
    for row, row_encoded in zip(rows, encoded):
        ids, slots, _ = row_encoded
        mark = time.perf_counter()
        model.engine.restore_state(state_data)
        copy_seconds += time.perf_counter() - mark
        mark = time.perf_counter()
        vocabulary = model.engine.branch_logits(len(prefix), ids[len(prefix) :])
        suffix_seconds += time.perf_counter() - mark
        results.append(_result(
            row, row_encoded, vocabulary[slots].tolist(), vocabulary, metadata,
            "llamacpp-state-restore-shared-v1", "quantized branch last-position logits over a restored prefix state",
        ))
    suffix_total = sum(len(ids) - len(prefix) for ids, _, _ in encoded)
    timing = {
        "total_seconds": time.perf_counter() - started,
        "encode_seconds": encode_seconds,
        "prefix_tokens": len(prefix),
        "prefill_seconds": prefill_seconds,
        "replicate_seconds": copy_seconds,
        "suffix_forward_seconds": suffix_seconds,
        "batch_size": len(rows),
        "branch_state_bytes": state_data[1],
        "true_suffix_tokens": suffix_total,
        "padded_suffix_tokens": suffix_total,
    }
    return results, timing
