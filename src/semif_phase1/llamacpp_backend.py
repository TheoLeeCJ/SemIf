"""Option readout over GGUF checkpoints through llama.cpp, on CPU or with GPU offload.

Prompt construction and answer-slot verification stay on the reference
transformers tokenizer, so prompt_sha256 matches the Torch backend exactly;
llama.cpp only executes the forward pass over the quantized GGUF weights.
Every scored prompt is re-tokenized through the GGUF vocabulary and must
agree with the reference encoding before it is evaluated.

Sequence 0 of the context holds the state prefill. Two ways to branch from it:

* **state restore** (``sequences == 1``): each decision restores the saved
  sequence-0 state, then decodes its suffix. One restore per decision.
* **sequence copy** (``sequences > 1``): the prefix is copied with
  ``llama_memory_seq_cp`` into up to ``sequences - 1`` throwaway sequences and
  all their suffixes are decoded in one batched ``llama_decode``. No state
  serialization, no restore; the branches are removed whole afterwards.

Hybrid linear-attention models such as Qwen3.5 keep a recurrent state that
cannot be partially erased, so ``llama_memory_seq_rm`` refuses to truncate a
tail back to the prefix. They do support whole-sequence removal, state
save/restore, and sequence copies (``llama_memory_recurrent::seq_cp``), which is
what both branching modes rely on. Pure-attention models additionally allow
tail truncation, which the optional marginal readout uses when available.
"""

from __future__ import annotations

import ctypes
import hashlib
import json
import os
from pathlib import Path
import re
import time
import weakref

import numpy

from .core import LETTERS, direct_messages, softmax
from .direct import PROMPT_VERSION, encode_prompt
from .shared import _state_prefix

DECODE_CHUNK = 512
READOUTS = ("last", "marginal")
AUTO = "auto"
#: Upper bound on branches per fan-out when `sequences` is "auto". With a unified KV buffer the
#: bound costs bookkeeping only; the real limit is the token budget, checked per state.
BRANCH_CAP = 32
_BACKEND_INITIALIZED = False


def _model_params(library, gpu_layers: int | None):
    """Initialize llama.cpp once and return model parameters.

    ``None`` keeps the library's own default for ``n_gpu_layers`` (``-1``, every layer, in current
    builds); an integer overrides it. The value that ends up in the params is what the metadata
    reports, so a run is reproducible whichever way it was chosen.
    """
    global _BACKEND_INITIALIZED
    if not _BACKEND_INITIALIZED:
        library.llama_backend_init()
        _BACKEND_INITIALIZED = True
    params = library.llama_model_default_params()
    if gpu_layers is not None:
        params.n_gpu_layers = gpu_layers
    return params


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


def _stable_softmax(values: numpy.ndarray) -> numpy.ndarray:
    shifted = values - values.max()
    weights = numpy.exp(shifted)
    return weights / weights.sum()


class _Engine:
    """One llama.cpp context bound to a loaded GGUF model."""

    def __init__(self, library, model, context_tokens: int, threads: int, sequences: int = 1):
        if not (isinstance(sequences, int) and sequences >= 1):
            raise ValueError("sequences must be a positive integer")
        params = library.llama_context_default_params()
        # Sequence 0 keeps the prefix; the others are branch slots. With a unified KV buffer every
        # sequence sees the whole n_ctx (measured: 16 sequences, n_ctx 4096 -> 4096 per sequence,
        # against 256 without), and copied branches share the prefix's cells, so the context is
        # sized once for the longest prompt and the branch count is decided per state from the
        # token budget. Without a unified buffer llama.cpp splits n_ctx across sequences, and the
        # only safe sizing is the worst case.
        unified = None
        if sequences > 1 and hasattr(params, "kv_unified"):
            params.kv_unified = True
            unified = True
        params.n_ctx = context_tokens if unified or sequences == 1 else context_tokens * sequences
        params.n_seq_max = sequences
        # One flagged output per sequence in a fan-out batch; llama.cpp requires
        # n_outputs_max >= n_seq_max * n_outputs_max_per_seq.
        params.n_outputs_max = sequences
        params.n_outputs_max_per_seq = 1
        params.n_threads = threads
        params.n_threads_batch = threads
        self.lib = library
        self.model = model
        self.sequences = sequences
        self.kv_unified = unified
        self.context = library.llama_init_from_model(model, params)
        if not self.context:
            raise RuntimeError("llama.cpp failed to create the scoring context")
        self.memory = library.llama_get_memory(self.context)
        if not self.memory:
            library.llama_free(self.context)
            self.context = None
            raise RuntimeError("llama.cpp returned no context memory")
        self.context_tokens = int(library.llama_n_ctx(self.context))
        self.vocab_size = library.llama_n_vocab(library.llama_model_get_vocab(model))
        self.can_truncate = None  # learned on first attempt; depends on the model architecture

    def close(self) -> None:
        if self.context:
            self.lib.llama_free(self.context)
            self.context = None
            self.memory = None

    def _logits_at(self, index: int) -> numpy.ndarray:
        pointer = self.lib.llama_get_logits_ith(self.context, index)
        if not pointer:
            raise RuntimeError("llama.cpp returned no logits for the flagged position")
        return numpy.ctypeslib.as_array(
            ctypes.cast(pointer, ctypes.POINTER(ctypes.c_float)), shape=(self.vocab_size,)
        ).copy()

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
        return self._logits_at(-1)

    def _decode_items(self, items: list[tuple[int, int, int, bool]]) -> dict[int, numpy.ndarray]:
        """Decode (token, position, sequence, want_logits) items across sequences in one pass.

        Logits are read right after the chunk that produced them: llama.cpp only exposes the
        outputs of the most recent ``llama_decode`` call.
        """
        if not items:
            raise ValueError("Refusing to decode an empty item list")
        collected: dict[int, numpy.ndarray] = {}
        for offset in range(0, len(items), DECODE_CHUNK):
            chunk = items[offset : offset + DECODE_CHUNK]
            batch = self.lib.llama_batch_init(len(chunk), 0, self.sequences)
            try:
                for index, (token, position, sequence, want) in enumerate(chunk):
                    batch.token[index] = token
                    batch.pos[index] = position
                    batch.n_seq_id[index] = 1
                    batch.seq_id[index][0] = sequence
                    batch.logits[index] = int(want)
                batch.n_tokens = len(chunk)
                if self.lib.llama_decode(self.context, batch):
                    raise RuntimeError("llama_decode failed; raise --max-tokens if prompts grew")
                for index, (_, _, sequence, want) in enumerate(chunk):
                    if want:
                        collected[sequence] = self._logits_at(index)
            finally:
                self.lib.llama_batch_free(batch)
        return collected

    def clear(self) -> None:
        self.lib.llama_memory_clear(self.memory, False)

    def prefill(self, prefix: list[int]) -> None:
        self._decode(prefix, 0, 0, False)

    def save_state(self):
        """Snapshot sequence 0 for repeated branch restores."""
        size = self.lib.llama_state_seq_get_size(self.context, 0)
        if size <= 0:
            raise RuntimeError("llama.cpp returned an empty prefix state")
        buffer = (ctypes.c_ubyte * size)()
        if self.lib.llama_state_seq_get_data(self.context, buffer, size, 0) != size:
            raise RuntimeError("llama.cpp wrote an incomplete prefix state")
        return buffer, size

    def restore_state(self, state) -> None:
        buffer, size = state
        if not self.lib.llama_memory_seq_rm(self.memory, 0, -1, -1):
            raise RuntimeError("llama.cpp could not drop the previous scored branch")
        if self.lib.llama_state_seq_set_data(self.context, buffer, size, 0) == 0:
            raise RuntimeError("llama.cpp could not restore the saved prefix state")

    def branch_logits(self, prefix_length: int, suffix: list[int]) -> numpy.ndarray:
        return self._decode(suffix, prefix_length, 0, True)

    def full_logits(self, tokens: list[int]) -> numpy.ndarray:
        self.clear()
        return self._decode(tokens, 0, 0, True)

    # --- sequence-copy fan-out -------------------------------------------------------------

    def fanout_logits(self, prefix_length: int, suffixes: list[list[int]]) -> list[numpy.ndarray]:
        """Copy the sequence-0 prefix into one branch per suffix and decode them all at once.

        Returns one full-vocabulary logit vector per suffix, then removes the branches whole so
        sequence 0 is left exactly as prefilled. Whole-sequence removal never fails, on any
        memory type.
        """
        if not suffixes or any(not suffix for suffix in suffixes):
            raise ValueError("Every branch needs a nonempty suffix")
        if len(suffixes) > self.sequences - 1:
            raise ValueError(f"{len(suffixes)} branches exceed the {self.sequences - 1} configured slots")
        branches = list(range(1, len(suffixes) + 1))
        for branch in branches:
            self.lib.llama_memory_seq_cp(self.memory, 0, branch, -1, -1)
        items = [
            (token, prefix_length + offset, branch, offset == len(suffix) - 1)
            for branch, suffix in zip(branches, suffixes)
            for offset, token in enumerate(suffix)
        ]
        try:
            logits = self._decode_items(items)
            return [logits[branch] for branch in branches]
        finally:
            for branch in branches:
                self.lib.llama_memory_seq_rm(self.memory, branch, -1, -1)

    def probe_branches(self, positions: dict[int, int], tokens: dict[int, int]) -> dict[int, numpy.ndarray]:
        """Append one token to each listed branch and read the logits after it.

        Used by the marginal readout: ``tokens[branch]`` is the preamble token the model wanted
        to emit first, ``positions[branch]`` the position it goes to. Branches are live sequences
        from the current fan-out; the caller removes them.
        """
        items = [(tokens[branch], positions[branch], branch, True) for branch in sorted(tokens)]
        return self._decode_items(items)


def _free_native(engine: _Engine, library, model) -> None:
    engine.close()
    library.llama_model_free(model)


class _Backend:
    """Verified scoring adapter around one llama.cpp engine."""

    def __init__(self, engine: _Engine, model, vocab, tokenizer):
        self.engine = engine
        self.vocab = vocab
        self.tokenizer = tokenizer
        self._finalizer = weakref.finalize(self, _free_native, engine, engine.lib, model)

    def close(self) -> None:
        self._finalizer()

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
               context_tokens: int = 4096, gpu_layers: int | str | None = AUTO,
               sequences: int | str = AUTO, readout: str = "last"):
    """Load one pinned reference tokenizer plus a local GGUF checkpoint for scoring.

    ``gpu_layers``: ``"auto"`` (or ``None``) leaves llama.cpp's default, which offloads every
    layer when the library can; ``0`` forces CPU; any other integer is passed through.

    ``sequences``: ``"auto"`` opens ``1 + BRANCH_CAP`` sequences over a unified KV buffer and lets
    shared scoring size each fan-out from the state's own token counts; an integer fixes
    ``n_seq_max`` (``1`` keeps the historical state-restore path).
    """
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
    if gpu_layers == AUTO:
        gpu_layers = None
    if gpu_layers is not None and not (isinstance(gpu_layers, int) and gpu_layers >= -1):
        raise ValueError("gpu_layers must be 'auto', -1, or a nonnegative integer")
    auto_sequences = sequences == AUTO
    if auto_sequences:
        sequences = 1 + BRANCH_CAP
    if not (isinstance(sequences, int) and sequences >= 1):
        raise ValueError("sequences must be 'auto' or a positive integer")
    if readout not in READOUTS:
        raise ValueError(f"readout must be one of {READOUTS}")
    import transformers
    try:
        import llama_cpp
    except ImportError as error:
        raise RuntimeError("Install the llama.cpp extra: pip install -e '.[test,llamacpp]'") from error

    offline = bool(os.environ.get("HF_HUB_OFFLINE"))
    common = {"revision": None if local else revision,
              "local_files_only": local or offline, "trust_remote_code": False}
    tokenizer = transformers.AutoTokenizer.from_pretrained(source, **common)
    # Finish file I/O before allocating native resources so read failures cannot leak them.
    checksum = hashlib.sha256()
    with gguf.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            checksum.update(block)
    gguf_record = {"file": gguf.name, "bytes": gguf.stat().st_size,
                   "sha256": checksum.hexdigest()}
    model_params = _model_params(llama_cpp, gpu_layers)
    model = llama_cpp.llama_model_load_from_file(str(gguf).encode("utf-8"), model_params)
    if not model:
        raise RuntimeError(f"llama.cpp failed to load the GGUF checkpoint: {gguf}")
    window = context_tokens + 64
    try:
        engine = _Engine(llama_cpp, model, window, threads, sequences)
        vocab = llama_cpp.llama_model_get_vocab(model)
        _verify_vocabulary(tokenizer, llama_cpp, vocab)
    except Exception:
        if "engine" in locals():
            engine.close()
        llama_cpp.llama_model_free(model)
        raise
    metadata = {
        "source": source,
        "revision": revision,
        "backend": "llamacpp",
        "dtype": "gguf-quantized",
        "gguf": gguf_record,
        "vocab_size": engine.vocab_size,
        "threads": threads,
        "n_gpu_layers": int(model_params.n_gpu_layers),
        "n_gpu_layers_requested": "auto" if gpu_layers is None else gpu_layers,
        "gpu_offload_supported": bool(llama_cpp.llama_supports_gpu_offload()),
        "n_seq_max": sequences,
        "branch_sizing": "auto" if auto_sequences else "fixed",
        "kv_unified": engine.kv_unified,
        "readout_mode": readout,
        "max_prompt_tokens": context_tokens,
        "context_tokens": engine.context_tokens,
        "decode_chunk": DECODE_CHUNK,
        "llama_cpp_python_version": llama_cpp.__version__,
        "transformers_version": transformers.__version__,
    }
    return _Backend(engine, model, vocab, tokenizer), tokenizer, metadata


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
        self.prefix = None
        self.state_data = None

    def score(self, row: dict) -> dict:
        started = time.perf_counter()
        encoded = self.model.encode_verified(row, self.max_tokens)
        ids, slots, _ = encoded
        prefix = _state_prefix(self.tokenizer, row["state"])
        hit = self.state_data is not None and prefix == self.prefix
        if not prefix or ids[: len(prefix)] != prefix or len(ids) <= len(prefix):
            raise ValueError("State prefix does not match the full prompt")
        prefill_seconds = 0.0
        if not hit:
            mark = time.perf_counter()
            self.model.engine.clear()
            self.model.engine.prefill(prefix)
            prefill_seconds = time.perf_counter() - mark
            self.prefix = prefix
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


def _chunks(prefix_length: int, suffix_lengths: list[int], budget: int, cap: int) -> list[list[int]]:
    """Group consecutive rows into fan-out chunks from what the context can hold.

    A chunk holds at most ``cap`` branches and never more than ``budget`` tokens including the
    shared prefix. Everything here is known before the first decode, so the branch count is a
    consequence of the data, not a setting.
    """
    if budget <= prefix_length:
        raise ValueError("The state prefix alone exceeds the context")
    chunks: list[list[int]] = []
    current: list[int] = []
    total = prefix_length
    for index, length in enumerate(suffix_lengths):
        if prefix_length + length > budget:
            raise ValueError(f"Row {index}: prefix plus suffix exceed the context; raise --max-tokens")
        if current and (len(current) >= cap or total + length > budget):
            chunks.append(current)
            current, total = [], prefix_length
        current.append(index)
        total += length
    if current:
        chunks.append(current)
    return chunks


def _marginal_readout(engine: _Engine, branches: list[int], prefix_length: int,
                      suffixes: list[list[int]], vocabularies: list[numpy.ndarray],
                      slot_lists: list[list[int]], probes: int = 2, floor: float = 0.05):
    """Fold one-token preambles back into the answer-slot masses.

    Some checkpoints want to emit a token before the letter — Qwen3-8B puts almost all of its
    mass on the Markdown ``**`` token when the assistant turn starts empty. Reading only the last
    prompt position then reads the preamble, not the answer. For each branch whose most likely
    next tokens are not answer slots, this appends up to ``probes`` such tokens (one extra batched
    decode per probe rank across all branches), reads the slot masses after them, and adds them
    to the direct slot masses weighted by the preamble's probability. The argmax and the reported
    ``option_logits`` become log-masses of that truncated path sum.

    Returns per-branch (option_logits, preamble_mass) with option_logits as natural logs of the
    summed masses, so downstream softmax and temperature scaling keep working unchanged.
    """
    live = {branch: prefix_length + len(suffix) for branch, suffix in zip(branches, suffixes)}
    masses, via = [], []
    for vocabulary, slots in zip(vocabularies, slot_lists):
        probabilities = _stable_softmax(vocabulary.astype(numpy.float64))
        masses.append(probabilities[slots].copy())
        via.append(0.0)
    # Candidate preambles per branch: top tokens that are not answer slots.
    candidates: list[list[int]] = []
    for vocabulary, slots in zip(vocabularies, slot_lists):
        forbidden = set(slots)
        order = numpy.argsort(-vocabulary)[: probes + len(slots)]
        candidates.append([int(token) for token in order if int(token) not in forbidden][:probes])
    weights = [_stable_softmax(vocabulary.astype(numpy.float64)) for vocabulary in vocabularies]
    for rank in range(probes):
        tokens, positions = {}, {}
        for index, branch in enumerate(branches):
            if rank < len(candidates[index]):
                token = candidates[index][rank]
                if weights[index][token] >= floor:
                    tokens[branch] = token
                    positions[branch] = live[branch]
        if not tokens:
            break
        after = engine.probe_branches(positions, tokens)
        for index, branch in enumerate(branches):
            if branch in tokens:
                weight = float(weights[index][tokens[branch]])
                gain = weight * _stable_softmax(after[branch].astype(numpy.float64))[slot_lists[index]]
                masses[index] += gain
                via[index] += float(gain.sum())
                live[branch] += 1
    results = []
    for mass, mass_via in zip(masses, via):
        total = float(mass.sum())
        results.append((numpy.log(numpy.maximum(mass, 1e-300)).tolist(), mass_via / total if total > 0 else 0.0))
    return results


def score_shared(model, tokenizer, rows: list[dict], metadata: dict, max_tokens: int = 4096):
    """Prefill one exact state once, then score every criterion from branches.

    With a single sequence the branches are restored from a saved state, one at a time. With
    ``n_seq_max > 1`` they are sequence copies decoded together, in chunks of ``n_seq_max - 1``.
    """
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
    engine = model.engine
    marginal = metadata.get("readout_mode") == "marginal"
    mark = time.perf_counter()
    engine.clear()
    engine.prefill(prefix)
    if engine.sequences == 1:
        state_data = engine.save_state()
    prefill_seconds = time.perf_counter() - mark
    copy_seconds = suffix_seconds = 0.0
    results = []
    if engine.sequences == 1:
        if marginal:
            raise ValueError("The marginal readout needs sequence copies; load with sequences > 1")
        for row, row_encoded in zip(rows, encoded):
            ids, slots, _ = row_encoded
            mark = time.perf_counter()
            engine.restore_state(state_data)
            copy_seconds += time.perf_counter() - mark
            mark = time.perf_counter()
            vocabulary = engine.branch_logits(len(prefix), ids[len(prefix) :])
            suffix_seconds += time.perf_counter() - mark
            results.append(_result(
                row, row_encoded, vocabulary[slots].tolist(), vocabulary, metadata,
                "llamacpp-state-restore-shared-v1",
                "quantized branch last-position logits over a restored prefix state",
            ))
        branch_bytes = state_data[1]
        config = "llamacpp-state-restore-shared-v1"
    else:
        # Under a unified KV buffer the copied branches share the prefix cells, so the budget is
        # the context minus a small margin; otherwise each sequence has its own slice of n_ctx.
        per_sequence = engine.context_tokens if engine.kv_unified else engine.context_tokens // engine.sequences
        budget = per_sequence - 8 if engine.kv_unified else per_sequence
        chunks = _chunks(len(prefix), [len(ids) - len(prefix) for ids, _, _ in encoded],
                         budget, engine.sequences - 1)
        branch_counts = [len(chunk) for chunk in chunks]
        for chunk in chunks:
            chunk_rows = [rows[i] for i in chunk]
            chunk_encoded = [encoded[i] for i in chunk]
            suffixes = [ids[len(prefix) :] for ids, _, _ in chunk_encoded]
            slot_lists = [slots for _, slots, _ in chunk_encoded]
            branches = list(range(1, len(suffixes) + 1))
            mark = time.perf_counter()
            for branch in branches:
                engine.lib.llama_memory_seq_cp(engine.memory, 0, branch, -1, -1)
            copy_seconds += time.perf_counter() - mark
            mark = time.perf_counter()
            try:
                items = [
                    (token, len(prefix) + offset, branch, offset == len(suffix) - 1)
                    for branch, suffix in zip(branches, suffixes)
                    for offset, token in enumerate(suffix)
                ]
                logits = engine._decode_items(items)
                vocabularies = [logits[branch] for branch in branches]
                if marginal:
                    folded = _marginal_readout(engine, branches, len(prefix), suffixes,
                                               vocabularies, slot_lists)
            finally:
                for branch in branches:
                    engine.lib.llama_memory_seq_rm(engine.memory, branch, -1, -1)
            suffix_seconds += time.perf_counter() - mark
            for index, (row, row_encoded) in enumerate(zip(chunk_rows, chunk_encoded)):
                vocabulary = vocabularies[index]
                if marginal:
                    selected, preamble_mass = folded[index]
                    result = _result(
                        row, row_encoded, selected, vocabulary, metadata,
                        "llamacpp-seq-copy-parallel-marginal-v1",
                        "quantized branch logits over copied prefix sequences, answer-slot masses "
                        "summed over one-token preambles; option_logits are log-masses",
                    )
                    result["preamble_mass"] = preamble_mass
                else:
                    result = _result(
                        row, row_encoded, vocabulary[row_encoded[1]].tolist(), vocabulary, metadata,
                        "llamacpp-seq-copy-parallel-v1",
                        "quantized branch last-position logits over copied prefix sequences; "
                        "one batched decode per chunk",
                    )
                results.append(result)
        branch_bytes = 0
        config = "llamacpp-seq-copy-parallel-marginal-v1" if marginal else "llamacpp-seq-copy-parallel-v1"
    if engine.sequences == 1:
        branch_counts = [1] * len(rows)
    suffix_total = sum(len(ids) - len(prefix) for ids, _, _ in encoded)
    timing = {
        "total_seconds": time.perf_counter() - started,
        "encode_seconds": encode_seconds,
        "prefix_tokens": len(prefix),
        "prefill_seconds": prefill_seconds,
        "replicate_seconds": copy_seconds,
        "suffix_forward_seconds": suffix_seconds,
        "batch_size": len(rows),
        "branch_slots": engine.sequences - 1,
        "branches_per_decode": branch_counts,
        "branch_state_bytes": branch_bytes,
        "serving_config": config,
        "true_suffix_tokens": suffix_total,
        "padded_suffix_tokens": suffix_total,
    }
    return results, timing
