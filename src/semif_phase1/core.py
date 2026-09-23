"""Shared input validation, prompts, model loading, and numeric helpers."""

from __future__ import annotations

import dataclasses
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


@dataclasses.dataclass(frozen=True)
class Prompt:
    """The words of one direct-decision prompt: its system instruction and payload key names.

    The payload shape never changes — evidence first, then the criterion, then the lettered
    options — because the serial and shared modes cut the reusable state prefix at the end of the
    evidence. A prompt only changes the words around that shape, so a model can be addressed in
    the language of its evidence. Every prompt carries its own ``version``, written into each
    result next to ``prompt_sha256``; the published numbers all use ``direct-options-v1``.
    """

    version: str
    system: str
    evidence_key: str = "evidence"
    criterion_key: str = "criterion"
    options_key: str = "options"
    letter_key: str = "letter"
    description_key: str = "description"

    def __post_init__(self) -> None:
        if not all(isinstance(value, str) and value.strip() for value in (self.version, self.system)):
            raise ValueError("A prompt needs a nonempty version and system instruction")
        keys = self.keys()
        if len(set(keys)) != len(keys) or not all(isinstance(key, str) and key.strip() for key in keys):
            raise ValueError("Prompt payload keys must be distinct nonempty strings")

    def keys(self) -> tuple[str, ...]:
        return (self.evidence_key, self.criterion_key, self.options_key, self.letter_key, self.description_key)

    def messages(self, row: dict) -> list[dict]:
        validate_row(row)
        payload = {
            self.evidence_key: row["state"],
            self.criterion_key: row["question"],
            self.options_key: [
                {self.letter_key: LETTERS[index], self.description_key: option["description"]}
                for index, option in enumerate(row["options"])
            ],
        }
        return [
            {"role": "system", "content": self.system},
            {"role": "user", "content": json.dumps(payload, ensure_ascii=False)},
        ]

    def evidence_text(self, state) -> str:
        """The user payload up to the end of the evidence: the part every decision on a state shares."""
        return json.dumps({self.evidence_key: state}, ensure_ascii=False)[:-1]


DEFAULT_PROMPT = Prompt("direct-options-v1", DIRECT_SYSTEM)
PROMPTS = {
    "en": DEFAULT_PROMPT,
    "fr": Prompt(
        "direct-options-fr-v1",
        "Applique le critère fourni aux éléments fournis. Choisis exactement une option de la liste. "
        "Réponds uniquement par sa lettre majuscule, sans explication ni raisonnement.",
        evidence_key="elements", criterion_key="critere", options_key="options",
        letter_key="lettre", description_key="description",
    ),
}


def resolve_prompt(spec=None) -> Prompt:
    """A ``Prompt`` from its name (``en``, ``fr``), a JSON file, an existing ``Prompt``, or ``None``.

    A file holds ``{"version": ..., "system": ..., "keys": {"evidence": ..., "criterion": ...,
    "options": ..., "letter": ..., "description": ...}}``; ``keys`` and each of its entries are
    optional and default to the English names. A custom prompt may not reuse a built-in version
    string unless it is that prompt, so ``prompt_version`` in a result always means one wording.
    """
    if spec is None:
        return DEFAULT_PROMPT
    if isinstance(spec, Prompt):
        return spec
    if isinstance(spec, str) and spec in PROMPTS:
        return PROMPTS[spec]
    path = Path(spec)
    if not path.is_file():
        raise ValueError(f"Unknown prompt {spec!r}: expected one of {sorted(PROMPTS)} or a JSON file")
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict) or not {"version", "system"} <= data.keys():
        raise ValueError(f"{path}: a prompt file needs 'version' and 'system'")
    keys = data.get("keys") or {}
    if not isinstance(keys, dict) or not keys.keys() <= {"evidence", "criterion", "options", "letter", "description"}:
        raise ValueError(f"{path}: 'keys' may only rename evidence, criterion, options, letter, description")
    prompt = Prompt(data["version"], data["system"], **{f"{name}_key": value for name, value in keys.items()})
    for built_in in PROMPTS.values():
        if prompt.version == built_in.version and prompt != built_in:
            raise ValueError(f"{path}: version {prompt.version!r} belongs to a built-in prompt with other words")
    return prompt


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


def direct_messages(row: dict, prompt: Prompt | None = None) -> list[dict]:
    return resolve_prompt(prompt).messages(row)


def softmax(values: list[float]) -> list[float]:
    if len(values) < 2 or any(not math.isfinite(value) for value in values):
        raise ValueError("Need at least two finite scores")
    maximum = max(values)
    weights = [math.exp(value - maximum) for value in values]
    total = sum(weights)
    return [weight / total for weight in weights]


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
