"""Backend ownership and execution routing.

One server process owns one loaded backend and one model, matching the
repository's existing rule that one loaded backend owns one stateful scoring
context. Requests are serialized through a lock: concurrency is a queue in
front of the model, not parallel model access.

Routing is specified in docs/JEV_API_COMPAT.md section 7. It is an
optimization and never changes what the model is asked.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import json
import os
from pathlib import Path
import threading
import time

from . import slots

BACKENDS = ("torch", "mlx", "llamacpp")
MODES = ("auto", "direct", "serial", "shared")

JEV_ALIASES = frozenset({"jev-latest", "jev-preview", "jev-1.13.0"})
SEMIF_ALIAS = "semif-latest"


def _env_flag(name: str, default: bool) -> bool:
    raw = os.environ.get(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


def _env_int(name: str, default: int | None) -> int | None:
    raw = os.environ.get(name)
    return default if raw is None or not raw.strip() else int(raw)


@dataclass
class Config:
    """Every knob, resolved. CLI options overwrite these before use."""

    model: str | None = None
    revision: str | None = None
    backend: str = "torch"
    device: str = "auto"
    dtype: str = "bfloat16"
    mlx_bits: int | None = None
    mlx_cache_limit_mib: int | None = None
    gguf: str | None = None
    llama_threads: int | None = None
    max_input_tokens: int = 4096
    mode: str = "auto"
    prompt_version: str = slots.V1
    temperature: float = 1.0
    temperatures: dict = field(default_factory=dict)
    api_key: str | None = None
    accept_jev_aliases: bool = True
    host: str = "127.0.0.1"
    port: int = 8471

    @classmethod
    def from_env(cls) -> "Config":
        config = cls(
            model=os.environ.get("SEMIF_MODEL"),
            revision=os.environ.get("SEMIF_REVISION"),
            backend=os.environ.get("SEMIF_BACKEND", "torch"),
            device=os.environ.get("SEMIF_DEVICE", "auto"),
            dtype=os.environ.get("SEMIF_DTYPE", "bfloat16"),
            mlx_bits=_env_int("SEMIF_MLX_BITS", None),
            mlx_cache_limit_mib=_env_int("SEMIF_MLX_CACHE_LIMIT_MIB", None),
            gguf=os.environ.get("SEMIF_GGUF"),
            llama_threads=_env_int("SEMIF_LLAMA_THREADS", None),
            max_input_tokens=_env_int("SEMIF_MAX_INPUT_TOKENS", 4096),
            mode=os.environ.get("SEMIF_MODE", "auto"),
            prompt_version=os.environ.get("SEMIF_PROMPT_VERSION", slots.V1),
            api_key=os.environ.get("SEMIF_API_KEY"),
            accept_jev_aliases=_env_flag("SEMIF_ACCEPT_JEV_ALIASES", True),
            host=os.environ.get("SEMIF_HOST", "127.0.0.1"),
            port=_env_int("SEMIF_PORT", 8471),
        )
        config.set_temperature(os.environ.get("SEMIF_TEMPERATURE", "1.0"))
        return config

    def set_temperature(self, value) -> None:
        """Accept a global temperature or a path to a per-type JSON map.

        The fitted values in docs/CALIBRATION.md are per-workload. Applying one
        of them globally would correct unrelated traffic, so 1.0 stays the
        default and anything else is an explicit choice.
        """
        if value is None or value == "":
            return
        text = str(value)
        try:
            self.temperature = float(text)
            self.temperatures = {}
            return
        except ValueError:
            pass
        mapping = json.loads(Path(text).read_text())
        if not isinstance(mapping, dict) or not mapping:
            raise ValueError("Temperature file must hold a nonempty object keyed by question type")
        unknown = sorted(set(mapping) - {"noul", "choice", "score", "default"})
        if unknown:
            raise ValueError(f"Temperature file has unknown keys: {unknown}")
        values = {key: float(entry) for key, entry in mapping.items()}
        if any(entry <= 0 for entry in values.values()):
            raise ValueError("Temperatures must be positive")
        self.temperature = values.get("default", 1.0)
        self.temperatures = values

    def temperature_for(self, kind: str) -> float:
        return self.temperatures.get(kind, self.temperature)

    def validate(self) -> None:
        if not self.model or not self.revision:
            raise ValueError("SEMIF_MODEL and SEMIF_REVISION are required")
        if self.backend not in BACKENDS:
            raise ValueError(f"Backend must be one of {list(BACKENDS)}")
        if self.mode not in MODES:
            raise ValueError(f"Mode must be one of {list(MODES)}")
        if self.max_input_tokens < 1:
            raise ValueError("Max input tokens must be positive")
        if self.temperature <= 0:
            raise ValueError("Temperature must be positive")
        if self.mlx_bits is not None and self.backend != "mlx":
            raise ValueError("MLX quantization requires the mlx backend")
        if self.gguf is not None and self.backend != "llamacpp":
            raise ValueError("A GGUF checkpoint requires the llamacpp backend")
        if self.backend == "llamacpp" and not (self.gguf and Path(self.gguf).is_file()):
            raise ValueError("The llamacpp backend requires SEMIF_GGUF pointing at an existing file")
        slots.resolve(self.prompt_version)
        if self.host not in {"127.0.0.1", "localhost", "::1"} and not self.api_key:
            raise ValueError("Set SEMIF_API_KEY before binding a non-loopback address")


def model_id(config: Config) -> str:
    """The identity reported in every response, including quantization."""
    name = Path(config.model).name if Path(config.model).exists() else config.model.rsplit("/", 1)[-1]
    if config.backend == "mlx":
        transform = f"mlx-q{config.mlx_bits}" if config.mlx_bits else "mlx-bf16"
    elif config.backend == "llamacpp":
        transform = "llamacpp-gguf"
    else:
        transform = f"torch-{config.dtype}"
    return f"semif/{name}@{config.revision[:12]}+{transform}"


def accepted_names(config: Config, served: str) -> set:
    """Names the `model` field may carry. `served` is the id that answers."""
    names = {served, SEMIF_ALIAS}
    if config.accept_jev_aliases:
        names |= JEV_ALIASES
    return names


class Runtime:
    """Owns the loaded backend and answers one request at a time."""

    def __init__(self, config: Config):
        config.validate()
        self.config = config
        self.model_id = model_id(config)
        self.lock = threading.Lock()
        self.ready = False
        self.model = None
        self.tokenizer = None
        self.metadata: dict = {}
        self._direct = None
        self._serial = None
        self._shared = None

    def load(self) -> None:
        config = self.config
        if config.backend == "mlx":
            from semif_phase1 import mlx_backend

            limit = (mlx_backend.DEFAULT_CACHE_LIMIT_MIB if config.mlx_cache_limit_mib is None
                     else config.mlx_cache_limit_mib)
            self.model, self.tokenizer, self.metadata = mlx_backend.load_model(
                config.model, config.revision, config.mlx_bits, cache_limit_mib=limit)
            self._direct, self._serial, self._shared = (
                mlx_backend.score, mlx_backend.SerialPrefixScorer, mlx_backend.score_shared)
        elif config.backend == "llamacpp":
            from semif_phase1 import llamacpp_backend

            self.model, self.tokenizer, self.metadata = llamacpp_backend.load_model(
                config.model, config.revision, Path(config.gguf),
                threads=config.llama_threads, context_tokens=config.max_input_tokens)
            self._direct, self._serial, self._shared = (
                llamacpp_backend.score, llamacpp_backend.SerialPrefixScorer, llamacpp_backend.score_shared)
        else:
            from semif_phase1.core import load_causal_model
            from semif_phase1.direct import score as direct_score
            from semif_phase1.serial import SerialPrefixScorer
            from semif_phase1.shared import score_shared

            self.model, self.tokenizer, self.metadata = load_causal_model(
                config.model, config.revision, config.device, config.dtype)
            self._direct, self._serial, self._shared = direct_score, SerialPrefixScorer, score_shared
        self.ready = True

    def _run_direct(self, rows):
        started = time.perf_counter()
        results = [self._direct(self.model, self.tokenizer, row, self.metadata, self.config.max_input_tokens)
                   for row in rows]
        return results, {"total_seconds": time.perf_counter() - started}

    def _run_serial(self, rows):
        started = time.perf_counter()
        scorer = self._serial(self.model, self.tokenizer, self.metadata, self.config.max_input_tokens)
        results = [scorer.score(row) for row in rows]
        return results, {"total_seconds": time.perf_counter() - started}

    def _run_shared(self, rows):
        return self._shared(self.model, self.tokenizer, rows, self.metadata, self.config.max_input_tokens)

    def score(self, rows: list) -> tuple:
        """Return (results, mode, timing, fallback_from)."""
        pinned = self.config.mode
        if pinned != "auto":
            order = [pinned]
        elif len(rows) == 1:
            order = ["direct"]
        else:
            order = ["shared", "serial", "direct"]
        runners = {"direct": self._run_direct, "serial": self._run_serial, "shared": self._run_shared}
        fallback_from = None
        with self.lock:
            for index, mode in enumerate(order):
                try:
                    results, timing = runners[mode](rows)
                except (ValueError, RuntimeError):
                    if index == len(order) - 1:
                        raise
                    fallback_from = mode if fallback_from is None else fallback_from
                    continue
                return results, mode, timing, fallback_from
        raise RuntimeError("No execution mode produced a result")
