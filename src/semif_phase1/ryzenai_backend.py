"""Direct categorical-logit backend for AMD's official Ryzen AI NPU models.

This module deliberately supports one independent prompt per generator.  Ryzen
AI OGA does not support the continuation/cache modes used by SemIf's serial and
shared scorers, and direct scoring needs no generated token.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
from importlib.metadata import PackageNotFoundError, version
import json
from pathlib import Path
import re
import time

from .core import LETTERS, digest, direct_messages, softmax
from .direct import PROMPT_VERSION


BACKEND = "ryzenai-npu"
OGA_DISTRIBUTION = "onnxruntime-genai-directml-ryzenai"
OGA_VERSION = "0.14.0"


@dataclass(frozen=True)
class RyzenAiTokenizer:
    """The reference chat renderer paired with OGA's model tokenizer."""

    reference: object
    oga: object


def _package_version(name: str) -> str:
    try:
        return version(name)
    except PackageNotFoundError as error:
        raise RuntimeError(f"{name} must be installed for the {BACKEND} backend") from error


def _artifact_hashes(source: Path) -> dict[str, str]:
    """Record the exact local graph, compiled artifacts, and tokenizer inputs."""
    names = {"genai_config.json", "config.json", "tokenizer.json", "tokenizer_config.json",
             "special_tokens_map.json", "added_tokens.json", "chat_template.jinja"}
    model_suffixes = {".onnx", ".data", ".bin", ".fconst", ".state", ".meta", ".super", ".ctrlpkt"}
    result = {}
    for artifact in sorted((path for path in source.rglob("*") if path.is_file()), key=lambda path: str(path)):
        relative = artifact.relative_to(source).as_posix()
        compiled_cache = (relative == "cache/txn_bins.zip" or
                          (relative.startswith("cache/") and artifact.name.endswith("_meta.json")))
        if artifact.name not in names and artifact.suffix.casefold() not in model_suffixes and not compiled_cache:
            continue
        hasher = hashlib.sha256()
        with artifact.open("rb") as handle:
            for chunk in iter(lambda: handle.read(4 * 1024 * 1024), b""):
                hasher.update(chunk)
        result[artifact.relative_to(source).as_posix()] = hasher.hexdigest()
    return result


def _session_option_paths(value, path: tuple[str, ...] = ()) -> list[tuple[str, ...]]:
    if isinstance(value, list):
        result = []
        for index, child in enumerate(value):
            result.extend(_session_option_paths(child, path + (f"[{index}]",)))
        return result
    if not isinstance(value, dict):
        return []
    result = [path + ("session_options",)] if "session_options" in value else []
    for key, child in value.items():
        if key != "session_options":
            result.extend(_session_option_paths(child, path + (key,)))
    return result


def _load_npu_config(source: Path) -> tuple[dict, dict, dict]:
    config_path = source / "genai_config.json"
    if not config_path.is_file():
        raise ValueError(f"{BACKEND} requires local genai_config.json")
    try:
        config = json.loads(config_path.read_text(encoding="utf-8"))
        model = config["model"]
        providers = model["decoder"]["session_options"]["provider_options"]
    except (json.JSONDecodeError, KeyError, TypeError) as error:
        raise ValueError("Invalid genai_config.json: missing decoder provider options") from error
    decoder = model.get("decoder")
    if not isinstance(decoder, dict) or "pipeline" in decoder:
        raise ValueError(f"{BACKEND} supports one decoder-only language model")
    if _session_option_paths(model) != [("decoder", "session_options")]:
        raise ValueError(f"{BACKEND} supports decoder-only language model execution configuration")
    execution_components = {"decoder", "encoder", "decoder_pipeline", "vision", "audio", "embedder",
                            "embedding", "speech"}
    if set(model).intersection(execution_components) != {"decoder"}:
        raise ValueError(f"{BACKEND} supports decoder-only language model execution configuration")
    if not isinstance(providers, list) or len(providers) != 1 or not isinstance(providers[0], dict):
        raise ValueError(f"{BACKEND} requires exactly one RyzenAI decoder provider")
    provider = providers[0]
    if set(provider) != {"RyzenAI"} or not isinstance(provider["RyzenAI"], dict):
        raise ValueError(f"{BACKEND} requires exactly one RyzenAI decoder provider")
    options = provider["RyzenAI"]
    if str(options.get("hybrid_opt_token_backend", "")).casefold() != "npu":
        raise ValueError(f"{BACKEND} requires hybrid_opt_token_backend='npu'")
    for key, value in options.items():
        if key.casefold() in {"device", "device_type", "execution_provider", "provider"}:
            if str(value).casefold() in {"cuda", "dml", "directml", "gpu", "rocm"}:
                raise ValueError(f"{BACKEND} rejects hybrid GPU decoder configuration")
    search = config.get("search", model.get("decoder", {}).get("search"))
    if not isinstance(search, dict) or "max_length" not in search:
        raise ValueError("RyzenAI config must declare search.max_length")
    return model, options, search


def _positive_int(value, name: str) -> int:
    if isinstance(value, bool):
        raise ValueError(f"{name} must be an integer")
    if isinstance(value, int):
        parsed = value
    elif isinstance(value, str) and re.fullmatch(r"[1-9][0-9]*", value):
        parsed = int(value)
    else:
        raise ValueError(f"{name} must be an integer")
    if parsed < 1:
        raise ValueError(f"{name} must be positive")
    return parsed


def _context_ceiling(model: dict, options: dict, search: dict | None = None) -> int:
    search = model.get("search", model.get("decoder", {}).get("search", {})) if search is None else search
    if not isinstance(search, dict):
        raise ValueError("RyzenAI search configuration must be an object")
    if model.get("context_length") is None or search.get("max_length") is None:
        raise ValueError("RyzenAI config must declare model.context_length and search.max_length")
    values = [(model.get("context_length"), "model.context_length"),
              (options.get("max_length_for_kv_cache"), "RyzenAI.max_length_for_kv_cache"),
              (search.get("max_length"), "search.max_length")]
    chunk_context = options.get("hybrid_opt_chunk_context")
    if chunk_context is None:
        chunk_enabled = False
    elif isinstance(chunk_context, bool):
        raise ValueError("RyzenAI.hybrid_opt_chunk_context must be 0 or 1")
    elif (isinstance(chunk_context, int) and chunk_context in {0, 1}) or (
            isinstance(chunk_context, str) and chunk_context in {"0", "1"}):
        chunk_enabled = int(chunk_context) == 1
    else:
        raise ValueError("RyzenAI.hybrid_opt_chunk_context must be 0 or 1")
    if chunk_enabled:
        chunk_limit = _positive_int(options.get("hybrid_opt_max_seq_length"),
                                    "RyzenAI.hybrid_opt_max_seq_length")
        chunk_size = _positive_int(search.get("chunk_size"), "search.chunk_size")
        if chunk_size > chunk_limit:
            raise ValueError("search.chunk_size must not exceed RyzenAI.hybrid_opt_max_seq_length")
    else:
        values.append((options.get("hybrid_opt_max_seq_length"), "RyzenAI.hybrid_opt_max_seq_length"))
    limits = [_positive_int(value, name) for value, name in values if value is not None]
    return min(limits)


def load_model(source: str, revision: str):
    """Load an AMD 1.8 OGA 0.14.0 NPU model without changing its config."""
    source_path = Path(source)
    if not source_path.is_dir():
        raise ValueError(f"{BACKEND} requires a local AMD NPU model directory")
    if not revision:
        raise ValueError("Local models require an explicit manifest/revision string")
    model_config, provider_options, search = _load_npu_config(source_path)
    context_ceiling = _context_ceiling(model_config, provider_options, search)
    if _package_version(OGA_DISTRIBUTION) != OGA_VERSION:
        raise RuntimeError(f"{BACKEND} requires {OGA_DISTRIBUTION}=={OGA_VERSION}")

    import onnxruntime_genai as oga
    import transformers

    reference = transformers.AutoTokenizer.from_pretrained(
        # Preserve AMD's serialized tokenizer. Transformers 4.57.6 can suggest
        # a Mistral regex patch for local Qwen configs without version metadata;
        # applying it would diverge from OGA's tokenizer.
        source, local_files_only=True, trust_remote_code=False, fix_mistral_regex=False
    )
    model = oga.Model(str(source_path))
    tokenizer = RyzenAiTokenizer(reference=reference, oga=oga.Tokenizer(model))
    metadata = {
        "source": source,
        "revision": revision,
        "backend": BACKEND,
        "oga_distribution": OGA_DISTRIBUTION,
        "oga_version": OGA_VERSION,
        "transformers_version": transformers.__version__,
        "model_device_type": getattr(model, "device_type", None),
        "context_ceiling": context_ceiling,
        "generation_reserve_tokens": 0,
        "source_artifact_sha256": _artifact_hashes(source_path),
        "execution": "AMD official RyzenAI NPU configuration, including its CPU graph components; no GPU provider configured",
    }
    return model, tokenizer, metadata


def _ids(tokens, *, label: str) -> list[int]:
    import numpy as np

    array = np.asarray(tokens)
    if array.ndim != 1 or array.dtype.kind not in "iu":
        raise ValueError(f"{label} tokenizer returned non-1D integer token IDs")
    if array.size and (array.min() < 0 or array.max() > np.iinfo(np.int32).max):
        raise ValueError(f"{label} tokenizer returned token IDs outside signed int32 range")
    return array.astype(np.int32, copy=False).tolist()


def _encode_prompt(tokenizer: RyzenAiTokenizer, row: dict, max_tokens: int) -> tuple[list[int], list[int], str]:
    """Render SemIf's exact reference prompt, then require OGA token agreement."""
    import numpy as np

    prompt = tokenizer.reference.apply_chat_template(
        direct_messages(row), tokenize=False, add_generation_prompt=True, enable_thinking=False
    )
    reference_ids = _ids(tokenizer.reference.encode(prompt, add_special_tokens=False), label="reference")
    ids = _ids(tokenizer.oga.encode(prompt), label="OGA")
    if ids != reference_ids:
        raise ValueError("OGA tokenizer IDs differ from the reference SemIf tokenizer")
    if not ids or len(ids) > max_tokens:
        raise ValueError(f"Row {row['id']}: {len(ids)} input tokens exceed limit {max_tokens}; no truncation allowed")
    slots = []
    for letter in LETTERS[:len(row["options"])]:
        reference_slot = _ids(tokenizer.reference.encode(letter, add_special_tokens=False), label="reference")
        oga_slot = _ids(tokenizer.oga.encode(letter), label="OGA")
        if (len(reference_slot) != 1 or oga_slot != reference_slot or
                tokenizer.reference.decode(reference_slot) != letter or
                tokenizer.oga.decode(np.asarray(oga_slot, dtype=np.int32)) != letter):
            raise ValueError(f"Answer slot {letter!r} is not one exact round-trip token")
        reference_boundary = _ids(
            tokenizer.reference.encode(prompt + letter, add_special_tokens=False), label="reference"
        )
        boundary = _ids(tokenizer.oga.encode(prompt + letter), label="OGA")
        if reference_boundary != reference_ids + reference_slot or boundary != ids + oga_slot:
            raise ValueError(f"Answer boundary changes tokenization for slot {letter}")
        slots.append(oga_slot[0])
    if len(slots) != len(set(slots)):
        raise ValueError("Answer-slot tokens collide")
    return ids, slots, digest(prompt)


def score(model, tokenizer: RyzenAiTokenizer, row: dict, metadata: dict, max_tokens: int = 4096) -> dict:
    """Read next-token categorical logits after one NPU prompt append."""
    import numpy as np
    import onnxruntime_genai as oga

    ceiling = metadata.get("context_ceiling")
    if not isinstance(ceiling, int) or ceiling < 1:
        raise ValueError("Missing valid RyzenAI context ceiling in model metadata")
    input_limit = min(max_tokens, ceiling)
    started = time.perf_counter()
    ids, slots, prompt_hash = _encode_prompt(tokenizer, row, input_limit)
    params = oga.GeneratorParams(model)
    # No output token is reserved: direct scoring reads logits immediately after append_tokens.
    params.set_search_options(max_length=len(ids), batch_size=1, do_sample=False)
    generator = oga.Generator(model, params)
    forward_start = time.perf_counter()
    try:
        generator.append_tokens(np.asarray(ids, dtype=np.int32))
        logits = np.asarray(generator.get_logits(), dtype=np.float32)
        # AMD OGA 0.14 returns [batch, last_position, vocabulary]. Other OGA
        # builds omit the singleton position dimension. Never flatten batches.
        if logits.ndim == 3 and logits.shape[:2] == (1, 1):
            logits = logits[:, 0, :]
        if logits.ndim != 2 or logits.shape[0] != 1 or logits.shape[1] <= max(slots):
            raise RuntimeError(f"Unexpected OGA logits shape {tuple(logits.shape)}")
        selected = logits[0, slots].astype(np.float32, copy=True)
        if not np.isfinite(selected).all():
            raise RuntimeError("OGA returned non-finite selected logits")
        selected_values = selected.tolist()
    finally:
        del generator
    return {
        "id": row["id"],
        "option_ids": [option["id"] for option in row["options"]],
        "probabilities": softmax(selected_values),
        "option_logits": selected_values,
        "answer_token_ids": slots,
        "input_tokens": len(ids),
        "input_ids_sha256": hashlib.sha256(json.dumps(ids).encode()).hexdigest(),
        "forward_seconds": time.perf_counter() - forward_start,
        "total_seconds": time.perf_counter() - started,
        "prompt_sha256": prompt_hash,
        "prompt_version": PROMPT_VERSION,
        "model": metadata,
        "readout": "OGA native full-vocabulary last-position logits restricted to declared answer slots; no generated tokens",
        "probability_status": "conditional option score; uncalibrated as decision confidence",
    }
