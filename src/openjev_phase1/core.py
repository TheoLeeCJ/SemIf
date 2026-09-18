"""Shared input validation, prompts, model loading, and numeric helpers."""

from __future__ import annotations

import hashlib
import json
import math
import platform
import re
from pathlib import Path

LETTERS = "ABCDEFGHIJKLMNOP"
DIRECT_SYSTEM = (
    "Apply the supplied criterion to the supplied evidence. Choose exactly one listed option. "
    "Respond with only its uppercase letter, with no explanation or reasoning."
)


def validate_row(row: dict) -> None:
    required = {"id", "state", "question", "options"}
    if not required <= row.keys():
        raise ValueError(f"Row is missing fields: {sorted(required - row.keys())}")
    if not all(isinstance(row[key], str) and row[key] for key in ("id", "question")):
        raise ValueError("id and question must be nonempty strings")
    state = row["state"]
    if not isinstance(state, (str, dict, list)) or not state:
        raise ValueError("state must be a nonempty string, object, or array")
    try:
        json.dumps(state, ensure_ascii=False, allow_nan=False)
    except (TypeError, ValueError) as error:
        raise ValueError("state must be finite JSON-compatible data") from error
    options = row["options"]
    if not isinstance(options, list) or not 2 <= len(options) <= len(LETTERS):
        raise ValueError("options must contain 2-16 entries")
    ids = []
    for option in options:
        if not isinstance(option, dict) or not isinstance(option.get("id"), str) or not isinstance(option.get("description"), str):
            raise ValueError("Each option needs string id and description fields")
        ids.append(option["id"])
    if len(ids) != len(set(ids)):
        raise ValueError("Option IDs must be unique")


def direct_messages(row: dict) -> list[dict]:
    validate_row(row)
    payload = {
        "evidence": row["state"],
        "criterion": row["question"],
        "options": [
            {"letter": LETTERS[index], "description": option["description"]}
            for index, option in enumerate(row["options"])
        ],
    }
    return [
        {"role": "system", "content": DIRECT_SYSTEM},
        {"role": "user", "content": json.dumps(payload, ensure_ascii=False)},
    ]


def softmax(values: list[float]) -> list[float]:
    if len(values) < 2 or any(not math.isfinite(value) for value in values):
        raise ValueError("Need at least two finite scores")
    maximum = max(values)
    weights = [math.exp(value - maximum) for value in values]
    total = sum(weights)
    return [weight / total for weight in weights]


def digest(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()


def mps_available() -> bool:
    """Return True when Apple Metal (MPS) acceleration can be used."""
    try:
        import torch

        backends = getattr(torch, "backends", None)
        mps_backend = getattr(backends, "mps", None) if backends is not None else None
        if mps_backend is None:
            return False
        return bool(mps_backend.is_built() and mps_backend.is_available())
    except Exception:
        return False


def resolve_device(preferred=None):
    """Resolve the accelerator: explicit choice else CUDA > MPS > CPU.

    CUDA keeps the historical exactly-one-visible-GPU requirement so
    published timings stay comparable. MPS (Apple Silicon) needs no
    visibility env var; CPU is the final fallback.
    """
    import torch

    if preferred is not None:
        name = str(getattr(preferred, "type", preferred)).split(":")[0].lower()
        if name == "cuda":
            if not torch.cuda.is_available():
                raise ValueError("Requested CUDA device but torch.cuda is not available")
            if torch.cuda.device_count() != 1:
                raise ValueError("Expose exactly one CUDA GPU, for example with CUDA_VISIBLE_DEVICES")
            return torch.device("cuda:0")
        if name == "mps":
            if not mps_available():
                raise ValueError("Requested MPS device but Apple Metal (MPS) is not available")
            return torch.device("mps")
        if name == "cpu":
            return torch.device("cpu")
        raise ValueError(f"Unknown device {preferred!r}; expected cuda, mps, or cpu")
    if torch.cuda.is_available():
        if torch.cuda.device_count() != 1:
            raise ValueError("Expose exactly one CUDA GPU, for example with CUDA_VISIBLE_DEVICES")
        return torch.device("cuda:0")
    if mps_available():
        return torch.device("mps")
    return torch.device("cpu")


def _dtype_from_name(value):
    import torch

    if value is None or isinstance(value, torch.dtype):
        return value
    mapping = {
        "bfloat16": torch.bfloat16,
        "bf16": torch.bfloat16,
        "float16": torch.float16,
        "fp16": torch.float16,
        "half": torch.float16,
        "float32": torch.float32,
        "fp32": torch.float32,
        "float": torch.float32,
    }
    try:
        return mapping[str(value).lower()]
    except KeyError:
        raise ValueError(f"Unknown dtype {value!r}; expected bfloat16, float16, or float32") from None


def resolve_dtype(device=None, requested=None):
    """Preferred dtype: bfloat16 everywhere unless explicitly requested."""
    import torch

    dtype = _dtype_from_name(requested)
    if dtype is not None:
        return dtype
    return torch.bfloat16


def dtype_candidates(device, preferred=None):
    """Ordered dtype attempts: bf16 first, with fp16/fp32 fallback off CUDA."""
    import torch

    first = resolve_dtype(device, preferred)
    device_type = getattr(device, "type", str(device or "").split(":")[0].lower())
    if device_type == "cuda":
        return [first]
    ordered = [first]
    for fallback in (torch.float16, torch.float32):
        if fallback not in ordered:
            # float16 CPU matmuls are often unsupported; keep CPU to bf16->fp32.
            if device_type == "cpu" and fallback == torch.float16:
                continue
            ordered.append(fallback)
    return ordered


def synchronize(device=None) -> None:
    """Block until the accelerator finishes queued work (no-op on CPU)."""
    try:
        import torch
    except Exception:
        return
    kind = getattr(device, "type", None)
    if kind is None and device is not None:
        kind = str(device).split(":")[0].lower()
    try:
        if kind == "cuda" and torch.cuda.is_available():
            torch.cuda.synchronize(device)
        elif kind == "mps" and mps_available():
            mps = getattr(torch, "mps", None)
            sync = getattr(mps, "synchronize", None) if mps is not None else None
            if callable(sync):
                sync()
    except Exception:
        # Timing sync must never break scoring; forwards remain correct.
        pass


def describe_hardware() -> str:
    """Human-readable accelerator label; never raises."""
    try:
        import torch

        if torch.cuda.is_available() and torch.cuda.device_count() >= 1:
            try:
                return str(torch.cuda.get_device_name(0))
            except Exception:
                return "cuda"
        if mps_available():
            return f"apple-mps ({platform.machine()})"
        return f"cpu ({platform.machine()})"
    except Exception:
        return "unknown"


def reset_peak_memory_stats(device=None) -> None:
    """Reset peak allocation counters where the backend supports it."""
    try:
        import torch
    except Exception:
        return
    kind = getattr(device, "type", None)
    if kind is None and device is not None:
        kind = str(device).split(":")[0].lower()
    try:
        if kind == "cuda" and torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats(device)
        # torch.mps exposes current/driver allocation but no peak reset;
        # nothing to reset there or on CPU.
    except Exception:
        pass


def peak_memory_bytes(device=None):
    """Current peak/allocated bytes where measurable, else None."""
    try:
        import torch
    except Exception:
        return None
    kind = getattr(device, "type", None)
    if kind is None and device is not None:
        kind = str(device).split(":")[0].lower()
    try:
        if kind == "cuda" and torch.cuda.is_available():
            return int(torch.cuda.max_memory_allocated(device))
        if kind == "mps" and mps_available():
            mps = getattr(torch, "mps", None)
            current = getattr(mps, "current_allocated_memory", None) if mps is not None else None
            if callable(current):
                return int(current())
    except Exception:
        return None
    return None


def load_causal_model(source: str, revision: str, device=None, dtype=None):
    """Load one pinned causal model on the auto-selected accelerator.

    Device order is CUDA > Apple Metal (MPS) > CPU unless ``device`` is
    given explicitly as ``"cuda"``, ``"mps"``, or ``"cpu"``. The preferred
    dtype is bfloat16; non-CUDA devices fall back to float16/float32 when
    the preferred dtype fails to load. CUDA stays strict bfloat16 so
    published results remain comparable.
    """
    import torch
    import transformers

    local = Path(source).exists()
    if not local and not re.fullmatch(r"[0-9a-f]{40}", revision or ""):
        raise ValueError("Remote models require a pinned 40-character commit revision")
    if local and not revision:
        raise ValueError("Local models require an explicit manifest/revision string")
    resolved = resolve_device(device)
    device_type = resolved.type
    candidates = dtype_candidates(resolved, _dtype_from_name(dtype))
    common = {"revision": None if local else revision, "local_files_only": local, "trust_remote_code": False}
    config = transformers.AutoConfig.from_pretrained(source, **common)
    tokenizer = transformers.AutoTokenizer.from_pretrained(source, **common)
    cls = transformers.AutoModelForCausalLM
    if config.model_type in {"qwen3_5", "qwen3_5_text"}:
        cls = getattr(transformers, "Qwen3_5ForCausalLM", None)
        if cls is None:
            raise RuntimeError("Installed transformers lacks the native Qwen3.5 model")
        config = config.get_text_config()
    device_map = {"": "cuda:0" if device_type == "cuda" else device_type}
    last_error: Exception | None = None
    for attempt in candidates:
        try:
            model, loading = cls.from_pretrained(
                source,
                config=config,
                dtype=attempt,
                device_map=device_map,
                low_cpu_mem_usage=True,
                output_loading_info=True,
                **common,
            )
        except Exception as error:  # noqa: BLE001 - fall through to next dtype
            last_error = error
            if device_type == "cuda":
                raise
            message = str(error).lower()
            retryable = any(
                token in message
                for token in ("bfloat16", "bf16", "dtype", "mps", "metal", "not supported", "not implemented")
            )
            if retryable or attempt != candidates[-1]:
                if attempt == candidates[-1]:
                    raise
                continue
            raise
        if any(loading.get(key) for key in ("missing_keys", "mismatched_keys", "error_msgs")):
            raise RuntimeError(f"Checkpoint did not load completely: {loading}")
        # from_pretrained with a device_map may still leave meta/sharded
        # weights off-device on some backends; ensure placement explicitly.
        try:
            if device_type in {"mps", "cpu"}:
                model.to(resolved)
        except Exception as error:  # noqa: BLE001 - dtype fallback may still help
            last_error = error
            if device_type == "cuda" or attempt == candidates[-1]:
                raise
            continue
        model.eval()
        metadata = {
            "source": source,
            "revision": revision,
            "dtype": str(attempt).replace("torch.", ""),
            "device": device_type,
            "torch_version": torch.__version__,
            "transformers_version": transformers.__version__,
        }
        return model, tokenizer, metadata
    raise RuntimeError(f"Could not load model on {device_type}: {last_error}")
