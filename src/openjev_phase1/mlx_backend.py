"""Optional MLX readout for unquantized Qwen3.5 text decisions."""

from __future__ import annotations

import copy
import hashlib
import json
import re
import time
from importlib.metadata import version
from pathlib import Path

from .core import softmax
from .direct import PROMPT_VERSION, encode_prompt
from .shared import _state_prefix


def _reject_unsupported_checkpoint(config: dict) -> None:
    # mlx-lm can execute a config-declared model_file as Python.
    if config.get("model_file") is not None:
        raise ValueError("Checkpoint declares a custom model_file; refusing to execute it")
    if config.get("model_type") not in {"qwen3_5"}:
        raise ValueError("MLX scoring currently supports top-level Qwen3.5 checkpoints only")
    if config.get("quantization") or config.get("quantization_config"):
        raise ValueError("MLX scoring currently requires an unquantized checkpoint")


def load_causal_model(source: str, revision: str):
    local = Path(source).exists()
    if not local and not re.fullmatch(r"[0-9a-f]{40}", revision or ""):
        raise ValueError("Remote models require a pinned 40-character commit revision")
    if local and not revision:
        raise ValueError("Local models require an explicit manifest/revision string")
    try:
        import mlx.core as mx
        from mlx.utils import tree_flatten
        from mlx_lm.utils import load_model
    except ImportError as error:
        raise RuntimeError("Install the Apple Silicon backend with pip install -e '.[mlx,test]'") from error
    from huggingface_hub import snapshot_download
    from transformers import AutoTokenizer

    if not mx.metal.is_available():
        raise RuntimeError("The MLX backend requires an Apple Metal GPU")
    mx.set_default_device(mx.gpu)
    if local:
        path = Path(source)
        _reject_unsupported_checkpoint(json.loads((path / "config.json").read_text()))
    else:
        # Validate the small config before transferring any checkpoint weights.
        config = Path(snapshot_download(source, revision=revision, allow_patterns=["*.json"]))
        _reject_unsupported_checkpoint(json.loads((config / "config.json").read_text()))
        path = Path(snapshot_download(
            source, revision=revision,
            allow_patterns=["*.json", "*.safetensors", "*.model", "*.txt", "*.jinja"],
        ))
    model, _ = load_model(path, lazy=False, strict=True)
    dtypes = {key: str(value.dtype) for key, value in tree_flatten(model.parameters())}
    # Gated DeltaNet decay (A_log) and norm state parameters run in FP32; all
    # other weights must preserve the checkpoint's BF16 precision.
    unexpected = [
        key for key, dtype in dtypes.items()
        if dtype == "mlx.core.float32"
        and not (".linear_attn." in key and key.endswith((".A_log", ".norm.weight")))
    ]
    if unexpected or not any(dtype == "mlx.core.bfloat16" for dtype in dtypes.values()) \
            or not set(dtypes.values()) <= {"mlx.core.bfloat16", "mlx.core.float32"}:
        raise ValueError(
            "Expected BF16 checkpoint weights with FP32 linear-attention state parameters only, "
            f"got {sorted(set(dtypes.values()))}"
        )
    tokenizer = AutoTokenizer.from_pretrained(path, local_files_only=True, trust_remote_code=False)
    metadata = {
        "source": source,
        "revision": revision,
        "local": local,
        "backend": "mlx",
        "device": "metal",
        "dtype": "bfloat16",
        "parameter_dtypes": sorted(set(dtypes.values())),
        "precision_policy": "checkpoint dtypes preserved; no quantization",
        "mlx_version": version("mlx"),
        "mlx_lm_version": version("mlx-lm"),
        "transformers_version": version("transformers"),
    }
    return model, tokenizer, metadata


def _last_logits(model, ids, cache=None):
    # Qwen3.5's wrapper in mlx-lm 0.31.3 exposes the text backbone here.
    lm = model.language_model
    hidden = lm.model(ids, cache=cache)[:, -1:, :]
    logits = lm.model.embed_tokens.as_linear(hidden) if lm.args.tie_word_embeddings else lm.lm_head(hidden)
    return logits[:, -1, :]


def _result(row, encoded, selected, metadata):
    ids, _, prompt_hash = encoded
    return {
        "id": row["id"],
        "option_ids": [option["id"] for option in row["options"]],
        "probabilities": softmax(selected),
        "option_logits": selected,
        "input_tokens": len(ids),
        "prompt_sha256": prompt_hash,
        "prompt_version": PROMPT_VERSION,
        "model": metadata,
        "readout": "native full-vocabulary last-position logits restricted to declared answer slots",
        "probability_status": "conditional option score; uncalibrated as decision confidence",
    }


def score(model, tokenizer, row: dict, metadata: dict, max_tokens: int = 4096):
    import mlx.core as mx

    started = time.perf_counter()
    encoded = encode_prompt(tokenizer, row, max_tokens)
    ids, slots, _ = encoded
    inputs = mx.array([ids])
    mx.eval(inputs)
    mx.synchronize()
    mark = time.perf_counter()
    selected = _last_logits(model, inputs)[0, mx.array(slots)].astype(mx.float32)
    mx.eval(selected)
    mx.synchronize()
    forward_seconds = time.perf_counter() - mark
    result = _result(row, encoded, selected.tolist(), metadata)
    return {**result, "forward_seconds": forward_seconds, "total_seconds": time.perf_counter() - started}


class SerialPrefixScorer:
    """Branch complete hybrid caches at an exact evidence-token boundary."""

    def __init__(self, model, tokenizer, metadata: dict, max_tokens: int = 4096):
        self.model = model
        self.tokenizer = tokenizer
        self.metadata = {**metadata, "serving_config": "mlx-state-prefix-serial-v1"}
        self.max_tokens = max_tokens
        self.cache = None
        self.state = None
        self.prefix = None

    def score(self, row, encoded=None):
        import mlx.core as mx
        from mlx_lm.models.cache import make_prompt_cache

        started = time.perf_counter()
        if encoded is None:
            encoded = encode_prompt(self.tokenizer, row, self.max_tokens)
        ids, slots, _ = encoded
        hit = self.cache is not None and row["state"] == self.state
        prefix = self.prefix if hit else _state_prefix(self.tokenizer, row["state"])
        if not prefix or ids[:len(prefix)] != prefix or len(ids) <= len(prefix):
            raise ValueError("State prefix does not match the full prompt")
        prefill_seconds = 0.0
        if not hit:
            self.cache = self.state = self.prefix = None
            cache = make_prompt_cache(self.model)
            mx.synchronize()
            mark = time.perf_counter()
            logits = _last_logits(self.model, mx.array([prefix]), cache)
            mx.eval(logits, [layer.state for layer in cache])
            mx.synchronize()
            prefill_seconds = time.perf_counter() - mark
            self.cache, self.state, self.prefix = cache, copy.deepcopy(row["state"]), prefix
            del logits
        mx.synchronize()
        mark = time.perf_counter()
        # Includes both attention KV and convolution/recurrent state; no trimming.
        branch = copy.deepcopy(self.cache)
        mx.eval([layer.state for layer in branch])
        mx.synchronize()
        copy_seconds = time.perf_counter() - mark
        inputs = mx.array([ids[len(prefix):]])
        mx.eval(inputs)
        mx.synchronize()
        mark = time.perf_counter()
        vocabulary = _last_logits(self.model, inputs, branch)[0].astype(mx.float32)
        selected = vocabulary[mx.array(slots)]
        mx.eval(selected, [layer.state for layer in branch])
        mx.synchronize()
        suffix_seconds = time.perf_counter() - mark
        result = _result(row, encoded, selected.tolist(), self.metadata)
        # Audit fields matching the torch serial readout.
        result["answer_token_ids"] = slots
        result["full_vocab_argmax_id"] = int(mx.argmax(vocabulary).item())
        result["allowed_token_mass"] = float(
            mx.exp(mx.logsumexp(selected) - mx.logsumexp(vocabulary)).item()
        )
        del branch
        return {
            **result,
            "cache_hit": hit,
            "prefix_tokens": len(prefix),
            "prefix_sha256": hashlib.sha256(json.dumps(prefix).encode()).hexdigest(),
            "prefill_seconds": prefill_seconds,
            "copy_seconds": copy_seconds,
            "suffix_forward_seconds": suffix_seconds,
            "forward_seconds": prefill_seconds + suffix_seconds,
            "total_seconds": time.perf_counter() - started,
        }


def score_shared(model, tokenizer, rows, metadata, max_tokens=4096):
    if not rows or any(row["state"] != rows[0]["state"] for row in rows[1:]):
        raise ValueError("Shared scoring requires one nonempty exact state")
    if len({row["id"] for row in rows}) != len(rows):
        raise ValueError("Decision IDs must be unique")
    started = time.perf_counter()
    # Encode every prompt once so validation completes before any forward runs.
    encoded = [encode_prompt(tokenizer, row, max_tokens) for row in rows]
    encode_seconds = time.perf_counter() - started
    scorer = SerialPrefixScorer(model, tokenizer, metadata, max_tokens)
    scorer.metadata["serving_config"] = "mlx-state-prefix-shared-loop-v1"
    results = [scorer.score(row, encoded=encoded[index]) for index, row in enumerate(rows)]
    suffix_tokens = sum(r["input_tokens"] - r["prefix_tokens"] for r in results)
    return results, {
        "total_seconds": time.perf_counter() - started,
        "encode_seconds": encode_seconds,
        "prefix_tokens": results[0]["prefix_tokens"],
        "prefill_seconds": sum(r["prefill_seconds"] for r in results),
        "replicate_seconds": sum(r["copy_seconds"] for r in results),
        "suffix_forward_seconds": sum(r["suffix_forward_seconds"] for r in results),
        "batch_size": len(rows),
        "true_suffix_tokens": suffix_tokens,
        "padded_suffix_tokens": suffix_tokens,
    }
