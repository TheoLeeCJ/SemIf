"""Apple Silicon option readout using MLX-LM's native Qwen3.5 model and caches.

Scores remain conditional on the declared options, not calibrated confidence.
Each suffix owns an independent copy of both attention and recurrent state.
"""

from __future__ import annotations

import copy
import hashlib
from importlib.metadata import distribution, version
import json
from pathlib import Path
import platform
import re
import time

from .core import resolve_prompt, softmax
from .direct import encode_prompt
from .shared import _state_prefix


DEFAULT_CACHE_LIMIT_MIB = 256


def load_model(source: str, revision: str, bits: int | None = None, *,
               cache_limit_mib: int = DEFAULT_CACHE_LIMIT_MIB):
    """Load a pinned checkpoint strictly; optional affine quantization is in memory.

    Use the upstream sanitizer for text weights. Custom model code is prohibited,
    just as it is for the Torch loader. Hash the actual source artifacts, including
    local checkpoints, so a user-supplied local revision is not the only provenance.
    """
    if platform.system() != "Darwin" or platform.machine() != "arm64":
        raise RuntimeError("MLX backend requires macOS on Apple Silicon")
    local = Path(source).is_dir()
    if not revision or (not local and not re.fullmatch(r"[0-9a-f]{40}", revision)):
        raise ValueError("Remote models require a pinned 40-character revision; local models require a revision label")
    if bits not in (None, 4, 8):
        raise ValueError("MLX quantization must be 4 or 8 bits")
    if type(cache_limit_mib) is not int or cache_limit_mib < 0:
        raise ValueError("MLX cache limit must be a nonnegative integer in MiB")
    try:
        import mlx.core as mx
        import mlx.nn as nn
        from mlx.utils import tree_flatten
        from mlx_lm import load
    except ImportError as error:
        raise RuntimeError("Install the MLX extra: pip install -e '.[test,mlx]'") from error
    from huggingface_hub import snapshot_download

    if not mx.metal.is_available():
        raise RuntimeError("MLX Metal GPU is unavailable")
    mx.set_default_device(mx.gpu)
    # MLX defaults to caching almost all system RAM. Bound inactive allocations
    # so repeated variable-length prompts can coexist with other local models.
    cache_limit = cache_limit_mib * 1024 * 1024
    mx.set_cache_limit(cache_limit)
    path = Path(source) if local else Path(snapshot_download(
        source, revision=revision,
        allow_patterns=["*.json", "model*.safetensors", "*.jinja", "*.txt", "*.model"],
    ))
    config = json.loads((path / "config.json").read_text())
    if config.get("model_file") or config.get("model_type") not in {"qwen3_5"}:
        raise ValueError("MLX backend supports native Qwen3.5 text scoring only; custom model code is not allowed")
    if bits and (config.get("quantization") or config.get("quantization_config")):
        raise ValueError("In-memory quantization requires an unquantized source checkpoint")
    artifacts = {}
    for file in sorted(path.iterdir()):
        if file.is_file() and file.suffix in {".json", ".safetensors", ".jinja", ".txt", ".model"}:
            with file.open("rb") as stream:
                checksum = hashlib.sha256()
                for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
                    checksum.update(chunk)
                artifacts[file.name] = checksum.hexdigest()
    model, tokenizer = load(path, tokenizer_config={"trust_remote_code": False})
    if bits:
        nn.quantize(model, group_size=64, bits=bits, mode="affine")
    model.eval()
    mx.eval(model.parameters())
    mx.synchronize()
    metadata = {
        "source": source, "revision": revision, "backend": "mlx",
        "mlx_version": version("mlx"), "mlx_lm_version": version("mlx-lm"),
        "transformers_version": version("transformers"),
        "mlx_lm_source": json.loads(distribution("mlx-lm").read_text("direct_url.json") or "null"),
        "allocator_cache_limit_bytes": cache_limit,
        "dtype": sorted({str(value.dtype) for _, value in tree_flatten(model.parameters())}),
        "quantization": {"bits": bits, "group_size": 64, "mode": "affine"} if bits else config.get("quantization"),
        "source_artifact_sha256": artifacts,
    }
    return model, tokenizer, metadata


def _result(row, encoded, selected, metadata, mode, prompt):
    ids, slots, prompt_hash = encoded
    return {
        "id": row["id"], "option_ids": [option["id"] for option in row["options"]],
        "probabilities": softmax(selected), "option_logits": selected,
        "answer_token_ids": slots, "input_tokens": len(ids),
        "input_ids_sha256": hashlib.sha256(json.dumps(ids).encode()).hexdigest(),
        "prompt_sha256": prompt_hash, "prompt_version": prompt.version,
        "model": {**metadata, "serving_config": f"mlx-{mode}-v1"},
        "readout": "native last-position logits restricted to declared answer slots; no generated tokens",
        "probability_status": "conditional option score; uncalibrated as decision confidence",
    }


def score(model, tokenizer, row, metadata, max_tokens=4096, prompt=None):
    import mlx.core as mx

    prompt = resolve_prompt(prompt)
    mx.synchronize()
    started = time.perf_counter()
    encoded = encode_prompt(tokenizer, row, max_tokens, prompt)
    ids, slots, _ = encoded
    mark = time.perf_counter()
    logits = model(mx.array([ids]))[0, -1].astype(mx.float32)
    selected = logits[mx.array(slots)].tolist()
    mx.synchronize()
    result = _result(row, encoded, selected, metadata, "direct", prompt)
    result.update(forward_seconds=time.perf_counter() - mark, total_seconds=time.perf_counter() - started)
    return result


def _prefill(model, prefix):
    import mlx.core as mx
    from mlx_lm.models.cache import make_prompt_cache

    cache = make_prompt_cache(model)
    # Evaluate the cache, not the unused vocabulary projection for the prefix.
    model(mx.array([prefix]), cache=cache)
    mx.eval([entry.state for entry in cache])
    mx.synchronize()
    return cache


def _check_prefix(prefix, encoded):
    if not prefix or any(ids[:len(prefix)] != prefix or len(ids) <= len(prefix) for ids, _, _ in encoded):
        raise ValueError("The fixed state prefix does not match every full prompt")


class SerialPrefixScorer:
    """Reuse only the current exact state; never mutate the retained prefix."""

    def __init__(self, model, tokenizer, metadata, max_tokens=4096, prompt=None):
        self.model, self.tokenizer, self.metadata = model, tokenizer, metadata
        self.prompt = resolve_prompt(prompt)
        self.max_tokens = max_tokens
        self.prefix = self.cache = None

    def score(self, row):
        import mlx.core as mx

        mx.synchronize()
        started = time.perf_counter()
        encoded = encode_prompt(self.tokenizer, row, self.max_tokens, self.prompt)
        prefix = _state_prefix(self.tokenizer, row["state"], self.prompt)
        hit = self.cache is not None and prefix == self.prefix
        _check_prefix(prefix, [encoded])
        prefill_seconds = 0.0
        if not hit:
            self.cache = self.prefix = None
            mark = time.perf_counter()
            cache = _prefill(self.model, prefix)
            prefill_seconds = time.perf_counter() - mark
            self.cache, self.prefix = cache, prefix
        mark = time.perf_counter()
        branch = copy.deepcopy(self.cache)
        mx.eval([entry.state for entry in branch])
        mx.synchronize()
        copy_seconds = time.perf_counter() - mark
        ids, slots, _ = encoded
        mark = time.perf_counter()
        logits = self.model(mx.array([ids[len(prefix):]]), cache=branch)[0, -1].astype(mx.float32)
        selected = logits[mx.array(slots)].tolist()
        mx.synchronize()
        suffix_seconds = time.perf_counter() - mark
        result = _result(row, encoded, selected, self.metadata, "serial", self.prompt)
        result.update(cache_hit=hit, prefix_tokens=len(prefix), prefill_seconds=prefill_seconds,
                      copy_seconds=copy_seconds, suffix_forward_seconds=suffix_seconds,
                      forward_seconds=prefill_seconds + suffix_seconds,
                      total_seconds=time.perf_counter() - started)
        return result


def score_shared(model, tokenizer, rows, metadata, max_tokens=4096, prompt=None):
    """Prefill once, merge independent cache branches, then score padded suffixes.

    Right padding is masked by native recurrent caches. Causal attention prevents
    later padding from affecting real positions. Read each row's last *real* token;
    the disposable branch caches do not need to be finalized for further decoding.
    """
    import mlx.core as mx

    if not rows or any(row["state"] != rows[0]["state"] for row in rows):
        raise ValueError("Shared scoring requires one nonempty exact state")
    if len({row["id"] for row in rows}) != len(rows):
        raise ValueError("Decision IDs must be unique")
    mx.synchronize()
    started = time.perf_counter()
    prompt = resolve_prompt(prompt)
    encoded = [encode_prompt(tokenizer, row, max_tokens, prompt) for row in rows]
    prefix = _state_prefix(tokenizer, rows[0]["state"], prompt)
    _check_prefix(prefix, encoded)
    suffixes = [ids[len(prefix):] for ids, _, _ in encoded]
    lengths = [len(ids) for ids in suffixes]
    width = max(lengths)
    pad = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id
    if pad is None:
        raise ValueError("Tokenizer requires a padding or EOS token")
    tokens = mx.array([ids + [pad] * (width - len(ids)) for ids in suffixes])
    encode_seconds = time.perf_counter() - started
    mark = time.perf_counter()
    cache = _prefill(model, prefix)
    prefill_seconds = time.perf_counter() - mark
    mark = time.perf_counter()
    branches = [entry.merge([entry] * len(rows)) for entry in cache]
    for entry in branches:
        entry.prepare(lengths=lengths, right_padding=[width - size for size in lengths])
    mx.eval([entry.state for entry in branches])
    mx.synchronize()
    replicate_seconds = time.perf_counter() - mark
    mark = time.perf_counter()
    logits = model(tokens, cache=branches)
    selected = [logits[i, lengths[i] - 1, mx.array(slots)].astype(mx.float32)
                for i, (_, slots, _) in enumerate(encoded)]
    mx.eval(selected)
    mx.synchronize()
    suffix_seconds = time.perf_counter() - mark
    results = [_result(row, enc, values.tolist(), metadata, "shared", prompt)
               for row, enc, values in zip(rows, encoded, selected)]
    timing = {
        "total_seconds": time.perf_counter() - started, "encode_seconds": encode_seconds,
        "prefix_tokens": len(prefix), "prefill_seconds": prefill_seconds,
        "replicate_seconds": replicate_seconds, "suffix_forward_seconds": suffix_seconds,
        "batch_size": len(rows), "true_suffix_tokens": sum(lengths),
        "padded_suffix_tokens": width * len(rows),
    }
    return results, timing
