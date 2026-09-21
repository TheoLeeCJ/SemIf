"""GGUF option readout through llama.cpp (CPU or CUDA).

Prompt construction and answer-slot verification stay on the reference
transformers tokenizer, so prompt_sha256 matches the Torch backend exactly;
llama.cpp only executes the forward pass over the quantized GGUF weights.
Every scored prompt is re-tokenized through the GGUF vocabulary and must
agree with the reference encoding before it is evaluated.

This binds the C API with ctypes against a user-supplied libllama (the
Unsloth CUDA build on this machine). It does not import llama-cpp-python;
the PyPI CUDA wheel SIGILLs on AVX2-only CPUs, and its 0.3.35 structs do
not match current llama.cpp model params.

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

from .core import LETTERS, direct_messages, softmax
from .direct import PROMPT_VERSION, encode_prompt
from .shared import _state_prefix


DECODE_CHUNK = 512
DEFAULT_GPU_LAYERS = 99
_UNSLOTH_LIB = Path.home() / ".unsloth" / "llama.cpp" / "build" / "bin" / "libllama.so.0"

llama_token = ctypes.c_int32
llama_pos = ctypes.c_int32
llama_seq_id = ctypes.c_int32


class llama_model_params(ctypes.Structure):
    _fields_ = [
        ("devices", ctypes.c_void_p),
        ("tensor_buft_overrides", ctypes.c_void_p),
        ("n_gpu_layers", ctypes.c_int32),
        ("split_mode", ctypes.c_int),
        ("load_mode", ctypes.c_int),
        ("lazy_mode", ctypes.c_int),
        ("main_gpu", ctypes.c_int32),
        ("tensor_split", ctypes.POINTER(ctypes.c_float)),
        ("progress_callback", ctypes.c_void_p),
        ("progress_callback_user_data", ctypes.c_void_p),
        ("kv_overrides", ctypes.c_void_p),
        ("model_shared", ctypes.c_void_p),
        ("vocab_only", ctypes.c_bool),
        ("check_tensors", ctypes.c_bool),
        ("use_extra_bufts", ctypes.c_bool),
        ("no_host", ctypes.c_bool),
        ("no_alloc", ctypes.c_bool),
        ("load_mtp", ctypes.c_bool),
    ]


class llama_sampler_seq_config(ctypes.Structure):
    _fields_ = [("seq_id", llama_seq_id), ("sampler", ctypes.c_void_p)]


class llama_context_params(ctypes.Structure):
    _fields_ = [
        ("n_ctx", ctypes.c_uint32),
        ("n_batch", ctypes.c_uint32),
        ("n_ubatch", ctypes.c_uint32),
        ("n_seq_max", ctypes.c_uint32),
        ("n_rs_seq", ctypes.c_uint32),
        ("n_outputs_max", ctypes.c_uint32),
        ("n_outputs_max_per_seq", ctypes.c_uint32),
        ("n_threads", ctypes.c_int32),
        ("n_threads_batch", ctypes.c_int32),
        ("ctx_type", ctypes.c_int),
        ("rope_scaling_type", ctypes.c_int),
        ("pooling_type", ctypes.c_int),
        ("attention_type", ctypes.c_int),
        ("flash_attn_type", ctypes.c_int),
        ("rope_freq_base", ctypes.c_float),
        ("rope_freq_scale", ctypes.c_float),
        ("yarn_ext_factor", ctypes.c_float),
        ("yarn_attn_factor", ctypes.c_float),
        ("yarn_beta_fast", ctypes.c_float),
        ("yarn_beta_slow", ctypes.c_float),
        ("yarn_orig_ctx", ctypes.c_uint32),
        ("defrag_thold", ctypes.c_float),
        ("cb_eval", ctypes.c_void_p),
        ("cb_eval_user_data", ctypes.c_void_p),
        ("type_k", ctypes.c_int),
        ("type_v", ctypes.c_int),
        ("abort_callback", ctypes.c_void_p),
        ("abort_callback_data", ctypes.c_void_p),
        ("embeddings", ctypes.c_bool),
        ("offload_kqv", ctypes.c_bool),
        ("no_perf", ctypes.c_bool),
        ("op_offload", ctypes.c_bool),
        ("swa_full", ctypes.c_bool),
        ("kv_unified", ctypes.c_bool),
        ("samplers", ctypes.POINTER(llama_sampler_seq_config)),
        ("n_samplers", ctypes.c_size_t),
        ("ctx_other", ctypes.c_void_p),
    ]


class llama_batch(ctypes.Structure):
    _fields_ = [
        ("n_tokens", ctypes.c_int32),
        ("token", ctypes.POINTER(llama_token)),
        ("embd", ctypes.POINTER(ctypes.c_float)),
        ("pos", ctypes.POINTER(llama_pos)),
        ("n_seq_id", ctypes.POINTER(ctypes.c_int32)),
        ("seq_id", ctypes.POINTER(ctypes.POINTER(llama_seq_id))),
        ("logits", ctypes.POINTER(ctypes.c_int8)),
    ]


_LIBRARY = None


def resolve_llama_lib(explicit=None) -> Path:
    """Locate libllama.so without importing llama-cpp-python."""
    candidates = []
    if explicit:
        candidates.append(Path(explicit).expanduser())
    for key in ("SEMIF_LLAMA_LIB", "LLAMA_CPP_LIB", "LLAMA_CPP_LIB_PATH"):
        value = os.environ.get(key)
        if value:
            candidates.append(Path(value).expanduser())
    candidates.extend((_UNSLOTH_LIB, _UNSLOTH_LIB.with_name("libllama.so")))
    for path in candidates:
        if path.is_file():
            return path.resolve()
    raise RuntimeError(
        "No libllama shared library found. Pass --llama-lib or set SEMIF_LLAMA_LIB "
        "to a llama.cpp build with CUDA (for example ~/.unsloth/llama.cpp/build/bin/libllama.so.0). "
        "Do not use the pip llama-cpp-python CUDA wheel on AVX2-only CPUs."
    )


def _bind(library):
    library.llama_backend_init.restype = None
    library.llama_model_default_params.restype = llama_model_params
    library.llama_context_default_params.restype = llama_context_params
    library.llama_supports_gpu_offload.restype = ctypes.c_bool
    library.llama_version.restype = ctypes.c_char_p
    library.llama_print_system_info.restype = ctypes.c_char_p
    library.llama_model_load_from_file.argtypes = [ctypes.c_char_p, llama_model_params]
    library.llama_model_load_from_file.restype = ctypes.c_void_p
    library.llama_init_from_model.argtypes = [ctypes.c_void_p, llama_context_params]
    library.llama_init_from_model.restype = ctypes.c_void_p
    library.llama_get_memory.argtypes = [ctypes.c_void_p]
    library.llama_get_memory.restype = ctypes.c_void_p
    library.llama_model_get_vocab.argtypes = [ctypes.c_void_p]
    library.llama_model_get_vocab.restype = ctypes.c_void_p
    library.llama_vocab_n_tokens.argtypes = [ctypes.c_void_p]
    library.llama_vocab_n_tokens.restype = ctypes.c_int32
    library.llama_tokenize.argtypes = [
        ctypes.c_void_p, ctypes.c_char_p, ctypes.c_int32,
        ctypes.POINTER(llama_token), ctypes.c_int32, ctypes.c_bool, ctypes.c_bool,
    ]
    library.llama_tokenize.restype = ctypes.c_int32
    library.llama_token_to_piece.argtypes = [
        ctypes.c_void_p, llama_token, ctypes.c_char_p, ctypes.c_int32, ctypes.c_int32, ctypes.c_bool,
    ]
    library.llama_token_to_piece.restype = ctypes.c_int32
    library.llama_batch_init.argtypes = [ctypes.c_int32, ctypes.c_int32, ctypes.c_int32]
    library.llama_batch_init.restype = llama_batch
    library.llama_batch_free.argtypes = [llama_batch]
    library.llama_batch_free.restype = None
    library.llama_decode.argtypes = [ctypes.c_void_p, llama_batch]
    library.llama_decode.restype = ctypes.c_int32
    library.llama_get_logits_ith.argtypes = [ctypes.c_void_p, ctypes.c_int32]
    library.llama_get_logits_ith.restype = ctypes.POINTER(ctypes.c_float)
    library.llama_memory_clear.argtypes = [ctypes.c_void_p, ctypes.c_bool]
    library.llama_memory_clear.restype = None
    library.llama_memory_seq_rm.argtypes = [ctypes.c_void_p, llama_seq_id, llama_pos, llama_pos]
    library.llama_memory_seq_rm.restype = ctypes.c_bool
    library.llama_state_seq_get_size.argtypes = [ctypes.c_void_p, llama_seq_id]
    library.llama_state_seq_get_size.restype = ctypes.c_size_t
    library.llama_state_seq_get_data.argtypes = [
        ctypes.c_void_p, ctypes.POINTER(ctypes.c_uint8), ctypes.c_size_t, llama_seq_id,
    ]
    library.llama_state_seq_get_data.restype = ctypes.c_size_t
    library.llama_state_seq_set_data.argtypes = [
        ctypes.c_void_p, ctypes.POINTER(ctypes.c_uint8), ctypes.c_size_t, llama_seq_id,
    ]
    library.llama_state_seq_set_data.restype = ctypes.c_size_t
    library.llama_free.argtypes = [ctypes.c_void_p]
    library.llama_free.restype = None
    library.llama_model_free.argtypes = [ctypes.c_void_p]
    library.llama_model_free.restype = None
    return library


def load_llama_library(explicit=None):
    """Load libllama and register ggml backends from the same directory."""
    global _LIBRARY
    path = resolve_llama_lib(explicit)
    if _LIBRARY is not None and _LIBRARY["path"] == str(path):
        return _LIBRARY["llama"], _LIBRARY
    directory = path.parent
    ggml_path = directory / "libggml.so.0"
    if not ggml_path.is_file():
        ggml_path = directory / "libggml.so"
    handles = []
    if ggml_path.is_file():
        ggml = ctypes.CDLL(str(ggml_path), mode=ctypes.RTLD_GLOBAL)
        handles.append(ggml)
        ggml.ggml_backend_load_all_from_path.argtypes = [ctypes.c_char_p]
        ggml.ggml_backend_load_all_from_path.restype = None
        ggml.ggml_backend_load_all_from_path(str(directory).encode())
    llama = _bind(ctypes.CDLL(str(path), mode=ctypes.RTLD_GLOBAL))
    llama.llama_backend_init()
    handles.append(llama)
    _LIBRARY = {"llama": llama, "handles": handles, "path": str(path), "directory": str(directory)}
    return llama, _LIBRARY


def _render(tokenizer, row: dict) -> str:
    return tokenizer.apply_chat_template(
        direct_messages(row), tokenize=False, add_generation_prompt=True, enable_thinking=False
    )


def _gguf_tokenize(library, vocab, text: str) -> list[int]:
    data = text.encode("utf-8")
    needed = library.llama_tokenize(vocab, data, len(data), None, 0, False, True)
    if needed < 0:
        needed = -needed
    tokens = (llama_token * needed)()
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
        vocab = library.llama_model_get_vocab(model)
        self.vocab_size = library.llama_vocab_n_tokens(vocab)

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
        return numpy.ctypeslib.as_array(pointer, shape=(self.vocab_size,)).copy()

    def clear(self) -> None:
        self.lib.llama_memory_clear(self.memory, False)

    def prefill(self, prefix: list[int]) -> None:
        self._decode(prefix, 0, 0, False)

    def save_state(self):
        """Snapshot sequence 0 for repeated branch restores."""
        size = self.lib.llama_state_seq_get_size(self.context, 0)
        buffer = (ctypes.c_uint8 * size)()
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

    def close(self) -> None:
        if self.context:
            self.lib.llama_free(self.context)
            self.context = None
        if self.model:
            self.lib.llama_model_free(self.model)
            self.model = None


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

    def close(self) -> None:
        self.engine.close()


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
               context_tokens: int = 4096, n_gpu_layers: int = DEFAULT_GPU_LAYERS,
               llama_lib=None):
    """Load one pinned reference tokenizer plus a local GGUF checkpoint."""
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
    if not (isinstance(n_gpu_layers, int) and n_gpu_layers >= -1):
        raise ValueError("n_gpu_layers must be -1 or a nonnegative integer")
    if threads is None:
        threads = os.cpu_count() or 4
    if not (isinstance(threads, int) and threads >= 1):
        raise ValueError("threads must be a positive integer")
    import transformers

    llama, loaded = load_llama_library(llama_lib)
    if n_gpu_layers != 0 and not llama.llama_supports_gpu_offload():
        raise RuntimeError(
            "llama.cpp GPU offload is unavailable. Point --llama-lib at a CUDA build, "
            "or pass --llama-gpu-layers 0 for CPU scoring."
        )
    offline = bool(os.environ.get("HF_HUB_OFFLINE"))
    common = {"revision": None if local else revision,
              "local_files_only": local or offline, "trust_remote_code": False}
    tokenizer = transformers.AutoTokenizer.from_pretrained(source, **common)
    params = llama.llama_model_default_params()
    params.n_gpu_layers = n_gpu_layers
    model = llama.llama_model_load_from_file(str(gguf).encode("utf-8"), params)
    if not model:
        raise RuntimeError(f"llama.cpp failed to load the GGUF checkpoint: {gguf}")
    window = context_tokens + 64
    engine = _Engine(llama, model, window, threads)
    vocab = llama.llama_model_get_vocab(model)
    _verify_vocabulary(tokenizer, llama, vocab)
    checksum = hashlib.sha256()
    with gguf.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            checksum.update(block)
    version = llama.llama_version()
    system = llama.llama_print_system_info()
    metadata = {
        "source": source,
        "revision": revision,
        "backend": "llamacpp",
        "dtype": "gguf-quantized",
        "gguf": {"file": gguf.name, "bytes": gguf.stat().st_size, "sha256": checksum.hexdigest()},
        "n_gpu_layers": n_gpu_layers,
        "libllama": loaded["path"],
        "vocab_size": engine.vocab_size,
        "threads": threads,
        "context_tokens": window,
        "decode_chunk": DECODE_CHUNK,
        "llama_version": version.decode() if version else None,
        "llama_system_info": system.decode() if system else None,
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
