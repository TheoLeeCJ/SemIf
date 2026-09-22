"""Shared input validation, prompts, model loading, and numeric helpers."""

from __future__ import annotations

import hashlib
import json
import math
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


def normalized_entropy_confidence(probabilities: list[float]) -> float:
    """Return declared-option concentration in [0, 1], not calibrated correctness.

    Zero represents a uniform distribution and one represents all probability
    mass on a single option. The input must be a normalized distribution over
    at least two options.
    """
    if len(probabilities) < 2:
        raise ValueError("Need at least two probabilities")
    if any(
        not math.isfinite(value) or not 0.0 <= value <= 1.0
        for value in probabilities
    ):
        raise ValueError("Probabilities must be finite values between zero and one")
    if not math.isclose(sum(probabilities), 1.0, rel_tol=1e-9, abs_tol=1e-9):
        raise ValueError("Probabilities must sum to one")
    entropy = -sum(value * math.log(value) for value in probabilities if value > 0.0)
    confidence = 1.0 - entropy / math.log(len(probabilities))
    return min(1.0, max(0.0, confidence))


def digest(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()


def resolve_device(device: str = "auto"):
    import torch

    if device not in {"auto", "cuda", "mps"}:
        raise ValueError("Device must be auto, cuda, or mps")
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "mps"
    if device == "cuda":
        if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
            raise ValueError("Expose exactly one CUDA GPU, for example with CUDA_VISIBLE_DEVICES")
        return torch.device("cuda:0")
    if not torch.backends.mps.is_available():
        raise ValueError("MPS is unavailable; use an Apple Silicon Mac with an MPS-enabled PyTorch build")
    return torch.device("mps")


def synchronize(device) -> None:
    import torch

    if device.type == "cuda":
        torch.cuda.synchronize(device)
    elif device.type == "mps":
        torch.mps.synchronize()


def load_causal_model(source: str, revision: str, device: str = "auto", dtype: str = "bfloat16"):
    """Load one pinned causal model on a single CUDA or Apple GPU."""
    import torch
    import transformers

    local = Path(source).exists()
    if not local and not re.fullmatch(r"[0-9a-f]{40}", revision or ""):
        raise ValueError("Remote models require a pinned 40-character commit revision")
    if local and not revision:
        raise ValueError("Local models require an explicit manifest/revision string")
    target = resolve_device(device)
    if dtype not in {"bfloat16", "float16", "float32"}:
        raise ValueError("Dtype must be bfloat16, float16, or float32")
    common = {"revision": None if local else revision, "local_files_only": local, "trust_remote_code": False}
    config = transformers.AutoConfig.from_pretrained(source, **common)
    tokenizer = transformers.AutoTokenizer.from_pretrained(source, **common)
    cls = transformers.AutoModelForCausalLM
    if config.model_type in {"qwen3_5", "qwen3_5_text"}:
        cls = getattr(transformers, "Qwen3_5ForCausalLM", None)
        if cls is None:
            raise RuntimeError("Installed transformers lacks the native Qwen3.5 model")
        config = config.get_text_config()
    model, loading = cls.from_pretrained(
        source,
        config=config,
        dtype=getattr(torch, dtype),
        device_map={"": str(target)},
        low_cpu_mem_usage=True,
        output_loading_info=True,
        **common,
    )
    if any(loading.get(key) for key in ("missing_keys", "mismatched_keys", "error_msgs")):
        raise RuntimeError(f"Checkpoint did not load completely: {loading}")
    model.eval()
    metadata = {
        "source": source,
        "revision": revision,
        "dtype": dtype,
        "device": str(target),
        "torch_version": torch.__version__,
        "transformers_version": transformers.__version__,
    }
    return model, tokenizer, metadata
