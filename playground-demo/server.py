#!/usr/bin/env python3
"""Loopback-only static server and shared MLX/TypeSafe adapter for the playground."""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
import gc
from http import HTTPStatus
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
import inspect
import json
import os
from pathlib import Path
import platform
import threading
import time
from typing import Any, Callable
from urllib.parse import urlsplit


STATIC_DIR = Path(__file__).resolve().parent
DEFAULT_MODEL_SOURCE = "Qwen/Qwen3.5-4B"
DEFAULT_MODEL_REVISION = "851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a"
DEFAULT_LOCAL_MODEL_ID = "qwen3.5-4b"
QWEN_9B_MODEL_SOURCE = "Qwen/Qwen3.5-9B"
QWEN_9B_MODEL_REVISION = "c202236235762e1c871ad0ccb60c8ee5ba337b9a"
QWEN38_27B_MODEL_SOURCE = "mlx-community/Qwen3.8-27B-4bit"
QWEN38_27B_MODEL_REVISION = "10c35caafbb80f7dc6a7a432cdd11af10a6d4818"
MUSE_GLIMMER_30B_MODEL_SOURCE = "mlx-community/Muse-Glimmer-30B-4bit"
MUSE_GLIMMER_30B_MODEL_REVISION = "3e7677d7a40d348a3daba263a2b1c0aa41910710"
GEMMA_12B_MODEL_SOURCE = "mlx-community/gemma-4-12B-it-4bit"
GEMMA_12B_MODEL_REVISION = "73bcf09092aa277861d5a191b989b666f7f32e8f"
GEMMA_26B_A4B_MODEL_SOURCE = "mlx-community/gemma-4-26b-a4b-it-4bit"
GEMMA_26B_A4B_MODEL_REVISION = "0d77464eeb233a2da68ebf9d7dc4edaac7db956d"
GEMMA_31B_MODEL_SOURCE = "mlx-community/gemma-4-31b-it-4bit"
GEMMA_31B_MODEL_REVISION = "696d436c404745a59f30e4939a658162b0a9e57f"
MAX_REQUEST_BYTES = 2 * 1024 * 1024
MAX_QUESTIONS = 32
MAX_API_KEY_CHARS = 512
MAX_PROVIDER_MODEL_CHARS = 128
LOCAL_ORIGIN_HOSTS = {"127.0.0.1", "localhost", "::1"}
BIND_HOSTS = {"127.0.0.1", "localhost"}
BOCHA_JEV_BASE_URL = "https://jev.bocha.cn"
BOCHA_JEV_DEFAULT_MODEL = "bocha-jev-v1"


class RequestError(ValueError):
    """An error safe to return to the local browser client."""


def _reject_json_constant(value: str) -> None:
    raise RequestError(f"JSON constant {value} is not supported")


def _json_compatible(value: Any, label: str) -> None:
    try:
        json.dumps(value, ensure_ascii=False, allow_nan=False)
    except (TypeError, ValueError) as error:
        raise RequestError(f"{label} must be finite JSON-compatible data") from error


def _nonempty_content(value: Any) -> bool:
    if isinstance(value, str):
        return bool(value.strip())
    if isinstance(value, (dict, list)):
        return bool(value)
    return False


def _validated_bocha_api_key(value: Any) -> str:
    if not isinstance(value, str) or not value.strip():
        raise RequestError("Jev Bocha API key is required")
    key = value.strip()
    if len(key) > MAX_API_KEY_CHARS:
        raise RequestError(f"Jev Bocha API key must be at most {MAX_API_KEY_CHARS} characters")
    if any(character.isspace() or ord(character) < 32 or ord(character) > 126 for character in key):
        raise RequestError("Jev Bocha API key must contain printable ASCII without whitespace")
    return key


def _validated_bocha_base_url(value: Any) -> str:
    if not isinstance(value, str) or not value.strip():
        raise RequestError("Jev Bocha base URL is required")
    try:
        parsed = urlsplit(value.strip())
        port = parsed.port
    except ValueError as error:
        raise RequestError("Jev Bocha base URL is invalid") from error
    if (
        parsed.scheme != "https"
        or parsed.hostname != "jev.bocha.cn"
        or port not in {None, 443}
        or parsed.username is not None
        or parsed.password is not None
        or parsed.path not in {"", "/"}
        or parsed.query
        or parsed.fragment
    ):
        raise RequestError(f"Jev Bocha base URL must be {BOCHA_JEV_BASE_URL}")
    return BOCHA_JEV_BASE_URL


def _validated_bocha_model(value: Any) -> str:
    if not isinstance(value, str) or not value.strip():
        raise RequestError("Jev Bocha model is required")
    model = value.strip()
    if len(model) > MAX_PROVIDER_MODEL_CHARS:
        raise RequestError(f"Jev Bocha model must be at most {MAX_PROVIDER_MODEL_CHARS} characters")
    if any(ord(character) < 32 or ord(character) > 126 for character in model):
        raise RequestError("Jev Bocha model must contain printable ASCII")
    return model


def normalize_payload(payload: Any) -> tuple[Any, dict[str, dict[str, Any]], str]:
    """Validate the common browser request shape without importing either SDK."""
    if not isinstance(payload, dict):
        raise RequestError("Request body must be a JSON object")
    state = payload.get("state")
    if not isinstance(state, (str, dict, list)) or not state:
        raise RequestError("state must be a nonempty string, object, or array")
    _json_compatible(state, "state")
    questions = payload.get("questions")
    if not isinstance(questions, dict) or not questions:
        raise RequestError("questions must be a nonempty object")
    if len(questions) > MAX_QUESTIONS:
        raise RequestError(f"questions supports at most {MAX_QUESTIONS} dimensions in this demo")
    normalized: dict[str, dict[str, Any]] = {}
    for question_id, question in questions.items():
        if not isinstance(question_id, str) or not question_id.strip():
            raise RequestError("Every question key must be a nonempty string")
        if not isinstance(question, dict):
            raise RequestError(f"Question {question_id!r} must be an object")
        kind = question.get("type")
        if kind not in {"noul", "choice", "score"}:
            raise RequestError(f"Question {question_id!r} must use type noul, choice, or score")
        instructions = question.get("instructions")
        if not _nonempty_content(instructions):
            raise RequestError(f"Question {question_id!r} needs nonempty instructions")
        _json_compatible(instructions, f"Question {question_id!r} instructions")
        item = {"type": kind, "instructions": instructions}
        criteria = question.get("criteria")
        if kind == "noul":
            if criteria is not None:
                if not isinstance(criteria, dict) or not set(criteria) <= {"true", "false"}:
                    raise RequestError(
                        f"Noul question {question_id!r} criteria may only contain true and false"
                    )
                for key, value in criteria.items():
                    if not _nonempty_content(value):
                        raise RequestError(f"Noul criterion {question_id}.{key} must be nonempty")
                _json_compatible(criteria, f"Question {question_id!r} criteria")
                item["criteria"] = criteria
        elif kind == "choice":
            if not isinstance(criteria, dict) or not 2 <= len(criteria) <= 16:
                raise RequestError(
                    f"Choice question {question_id!r} needs 2 to 16 criteria entries for local comparison"
                )
            for option, description in criteria.items():
                if not isinstance(option, str) or not option.strip():
                    raise RequestError(f"Choice question {question_id!r} has an invalid option key")
                if description is not None and not _nonempty_content(description):
                    raise RequestError(f"Choice criterion {question_id}.{option} must be nonempty or null")
            _json_compatible(criteria, f"Question {question_id!r} criteria")
            item["criteria"] = criteria
        else:
            if not isinstance(criteria, list) or not 2 <= len(criteria) <= 10:
                raise RequestError(f"Score question {question_id!r} needs 2 to 10 ordered criteria")
            if any(not _nonempty_content(level) for level in criteria):
                raise RequestError(f"Score question {question_id!r} has an empty criterion")
            _json_compatible(criteria, f"Question {question_id!r} criteria")
            item["criteria"] = criteria
        normalized[question_id] = item
    model = payload.get("model", "jev-latest")
    if not isinstance(model, str) or not model.strip():
        raise RequestError("model must be a nonempty string")
    return state, normalized, model.strip()


def _content_text(value: Any) -> str:
    if isinstance(value, str):
        return value
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"))


def local_rows(state: Any, questions: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
    """Convert TypeSafe-shaped questions into SemIf's declared-option rows."""
    rows = []
    for question_id, question in questions.items():
        kind = question["type"]
        if kind == "noul":
            criteria = question.get("criteria", {})
            options = [
                {"id": "true", "description": _content_text(criteria.get("true", "Yes / true"))},
                {"id": "false", "description": _content_text(criteria.get("false", "No / false"))},
            ]
        elif kind == "choice":
            options = []
            for option_id, rubric in question["criteria"].items():
                description = option_id if rubric is None else f"{option_id}: {_content_text(rubric)}"
                options.append({"id": option_id, "description": description})
        else:
            options = [
                {"id": str(index), "description": _content_text(rubric)}
                for index, rubric in enumerate(question["criteria"])
            ]
        rows.append(
            {
                "id": question_id,
                "state": state,
                "question": _content_text(question["instructions"]),
                "options": options,
            }
        )
    return rows


def adapt_local_response(
    results: list[dict[str, Any]],
    timing: dict[str, Any],
    questions: dict[str, dict[str, Any]],
    metadata: dict[str, Any],
) -> dict[str, Any]:
    """Return typed answers while preserving the uncalibrated local-score boundary."""
    from semif_phase1.core import normalized_entropy_confidence

    by_id = {result["id"]: result for result in results}
    answers: dict[str, dict[str, Any]] = {}
    readouts = []
    for question_id, question in questions.items():
        result = by_id.get(question_id)
        if result is None:
            raise RuntimeError(f"MLX result is missing question {question_id!r}")
        option_ids = result["option_ids"]
        probabilities = result["probabilities"]
        if len(option_ids) != len(probabilities):
            raise RuntimeError(f"MLX result for {question_id!r} has mismatched options")
        distribution = dict(zip(option_ids, probabilities, strict=True))
        kind = question["type"]
        common = {
            "type": kind,
            "probability_status": "conditional option score; uncalibrated as decision confidence",
        }
        if kind == "noul":
            answer = {**common, "noul": distribution["true"], "probabilities": distribution}
        elif kind == "choice":
            selected = max(distribution, key=distribution.get)
            answer = {
                **common,
                "choice": selected,
                "probabilities": distribution,
                "confidence": normalized_entropy_confidence(probabilities),
                "confidence_method": "normalized_shannon_entropy",
                "confidence_status": "option-distribution concentration; not calibrated correctness",
            }
        else:
            score = sum(int(level) * probability for level, probability in distribution.items())
            answer = {
                **common,
                "score": score,
                "legend": {str(index): _content_text(value) for index, value in enumerate(question["criteria"])},
                "probabilities": distribution,
                "confidence": normalized_entropy_confidence(probabilities),
                "confidence_method": "normalized_shannon_entropy",
                "confidence_status": "option-distribution concentration; not calibrated correctness",
            }
        answers[question_id] = answer
        readouts.append(
            {
                key: value
                for key, value in result.items()
                if key
                in {
                    "id",
                    "option_ids",
                    "probabilities",
                    "option_logits",
                    "answer_token_ids",
                    "input_tokens",
                    "input_ids_sha256",
                    "prompt_sha256",
                    "prompt_version",
                    "readout",
                    "probability_status",
                }
            }
        )
    safe_metadata = {
        key: value
        for key, value in metadata.items()
        if key
        in {
            "source",
            "revision",
            "backend",
            "model_type",
            "mlx_version",
            "mlx_lm_version",
            "transformers_version",
            "allocator_cache_limit_bytes",
            "dtype",
            "quantization",
            "source_artifact_sha256",
        }
    }
    return {
        "model": safe_metadata.get("source", DEFAULT_MODEL_SOURCE),
        "provider": "local SemIf MLX",
        "answers": answers,
        "usage": {
            "input_tokens": sum(int(result.get("input_tokens", 0)) for result in results),
            "output_tokens": 0,
        },
        "timing": timing,
        "model_details": safe_metadata,
        "readouts": readouts,
    }


def typesafe_question_objects(
    questions: dict[str, dict[str, Any]],
    constructors: dict[str, Callable[..., Any]],
) -> dict[str, Any]:
    """Build official SDK objects through injectable constructors for testing."""
    output = {}
    for question_id, question in questions.items():
        kwargs = {"instructions": question["instructions"]}
        if "criteria" in question:
            kwargs["criteria"] = question["criteria"]
        output[question_id] = constructors[question["type"]](**kwargs)
    return output


def _serialize_sdk_response(response: Any) -> dict[str, Any]:
    if hasattr(response, "model_dump"):
        payload = response.model_dump(mode="json")
    elif hasattr(response, "model_dump_json"):
        payload = json.loads(response.model_dump_json())
    elif isinstance(response, dict):
        payload = response
    else:
        raise RuntimeError("TypeSafe SDK returned an unsupported response object")
    if not isinstance(payload, dict):
        raise RuntimeError("TypeSafe SDK response did not serialize to a JSON object")
    return payload


def _attach_client_timing(payload: dict[str, Any], elapsed_seconds: float) -> dict[str, Any]:
    """Add stable end-to-end timing without confusing it with provider latency."""
    existing = payload.get("timing")
    timing: dict[str, Any] = {
        "total_seconds": elapsed_seconds,
        "source": "client_wall_clock",
        "includes": "network and SDK overhead",
    }
    if existing is not None:
        timing["provider"] = existing
    payload["timing"] = timing
    return payload


@dataclass(frozen=True)
class LocalModelSpec:
    id: str
    label: str
    short_label: str
    source: str
    revision: str
    download_gb: float | None = None
    source_quantization_bits: int | None = None


BUILTIN_LOCAL_MODELS = {
    DEFAULT_LOCAL_MODEL_ID: LocalModelSpec(
        id=DEFAULT_LOCAL_MODEL_ID,
        label="Qwen3.5 4B",
        short_label="4B",
        source=DEFAULT_MODEL_SOURCE,
        revision=DEFAULT_MODEL_REVISION,
        download_gb=9.0,
    ),
    "qwen3.5-9b": LocalModelSpec(
        id="qwen3.5-9b",
        label="Qwen3.5 9B",
        short_label="9B",
        source=QWEN_9B_MODEL_SOURCE,
        revision=QWEN_9B_MODEL_REVISION,
        download_gb=19.3,
    ),
    "qwen3.8-27b-4bit": LocalModelSpec(
        id="qwen3.8-27b-4bit",
        label="Qwen3.8 27B · 4-bit",
        short_label="Qwen3.8 27B",
        source=QWEN38_27B_MODEL_SOURCE,
        revision=QWEN38_27B_MODEL_REVISION,
        download_gb=16.1,
        source_quantization_bits=4,
    ),
    "muse-glimmer-30b-4bit": LocalModelSpec(
        id="muse-glimmer-30b-4bit",
        label="Muse Glimmer 30B · 4-bit",
        short_label="Muse Glimmer 30B",
        source=MUSE_GLIMMER_30B_MODEL_SOURCE,
        revision=MUSE_GLIMMER_30B_MODEL_REVISION,
        download_gb=19.4,
        source_quantization_bits=4,
    ),
    "gemma-4-12b-it-4bit": LocalModelSpec(
        id="gemma-4-12b-it-4bit",
        label="Gemma 4 12B IT · 4-bit",
        short_label="Gemma 12B",
        source=GEMMA_12B_MODEL_SOURCE,
        revision=GEMMA_12B_MODEL_REVISION,
        download_gb=6.74,
        source_quantization_bits=4,
    ),
    "gemma-4-26b-a4b-it-4bit": LocalModelSpec(
        id="gemma-4-26b-a4b-it-4bit",
        label="Gemma 4 26B A4B IT · 4-bit",
        short_label="Gemma 26B A4B",
        source=GEMMA_26B_A4B_MODEL_SOURCE,
        revision=GEMMA_26B_A4B_MODEL_REVISION,
        download_gb=15.4,
        source_quantization_bits=4,
    ),
    "gemma-4-31b-it-4bit": LocalModelSpec(
        id="gemma-4-31b-it-4bit",
        label="Gemma 4 31B IT · 4-bit",
        short_label="Gemma 31B",
        source=GEMMA_31B_MODEL_SOURCE,
        revision=GEMMA_31B_MODEL_REVISION,
        download_gb=18.4,
        source_quantization_bits=4,
    ),
}


@dataclass(frozen=True)
class ServiceConfig:
    default_model_id: str = DEFAULT_LOCAL_MODEL_ID
    model_source: str | None = None
    model_revision: str | None = None
    mlx_bits: int | None = None
    mlx_cache_limit_mib: int = 256


class SharedMLXService:
    """Own exactly one MLX model instance and reuse it across browser requests."""

    def __init__(self, config: ServiceConfig):
        self.config = config
        custom = config.model_source is not None or config.model_revision is not None
        if custom and not (config.model_source and config.model_revision):
            raise ValueError("Custom model source and revision must be provided together")
        if custom:
            self._models = {
                "custom": LocalModelSpec(
                    id="custom",
                    label=config.model_source or "Custom model",
                    short_label="custom",
                    source=config.model_source or "",
                    revision=config.model_revision or "",
                )
            }
            self._default_model_id = "custom"
        else:
            if config.default_model_id not in BUILTIN_LOCAL_MODELS:
                raise ValueError(f"Unknown default local model {config.default_model_id!r}")
            self._models = BUILTIN_LOCAL_MODELS
            self._default_model_id = config.default_model_id
        self._condition = threading.Condition()
        self._inference_lock = threading.RLock()
        self._executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="semif-mlx")
        self._status = "unloaded"
        self._error: str | None = None
        self._active_model_id: str | None = None
        self._requested_model_id = self._default_model_id
        self._model: Any = None
        self._tokenizer: Any = None
        self._metadata: dict[str, Any] = {}
        env_bocha_key = os.environ.get("BOCHA_JEV_API_KEY", "").strip()
        self._bocha_api_key = env_bocha_key or None
        self._bocha_base_url = BOCHA_JEV_BASE_URL
        self._bocha_model = _validated_bocha_model(
            os.environ.get("BOCHA_JEV_MODEL", BOCHA_JEV_DEFAULT_MODEL)
        )
        self._bocha_credential_source = "environment" if self._bocha_api_key else "none"

    def _resolve_model(self, model_id: Any = None) -> LocalModelSpec:
        selected = self._default_model_id if model_id is None else model_id
        if not isinstance(selected, str) or selected not in self._models:
            available = ", ".join(self._models)
            raise RequestError(f"Unknown local model {selected!r}; choose one of: {available}")
        return self._models[selected]

    def _release_model(self) -> None:
        model, tokenizer, metadata = self._model, self._tokenizer, self._metadata
        self._model = self._tokenizer = None
        self._metadata = {}
        self._active_model_id = None
        if model is None and tokenizer is None:
            return
        del model, tokenizer
        gc.collect()
        if metadata.get("backend") == "mlx":
            from semif_phase1 import mlx_backend

            clear_cache = getattr(mlx_backend, "clear_memory_cache", None)
            if callable(clear_cache):
                clear_cache()

    def health(self) -> dict[str, Any]:
        with self._condition:
            status = self._status
            error = self._error
            active_model_id = self._active_model_id
            requested_model_id = self._requested_model_id
            bocha_key_configured = bool(self._bocha_api_key)
            bocha_base_url = self._bocha_base_url
            bocha_model = self._bocha_model
            bocha_credential_source = self._bocha_credential_source
        visible = self._models[active_model_id or requested_model_id]
        return {
            "service": "semif-shared-mlx-playground",
            "platform": {"system": platform.system(), "machine": platform.machine()},
            "local_model": {
                "status": status,
                "active_model": active_model_id,
                "requested_model": requested_model_id,
                "default_model": self._default_model_id,
                "available_models": [
                    {
                        "id": spec.id,
                        "label": spec.label,
                        "short_label": spec.short_label,
                        "source": spec.source,
                        "revision": spec.revision,
                        "download_gb": spec.download_gb,
                        "source_quantization_bits": spec.source_quantization_bits,
                    }
                    for spec in self._models.values()
                ],
                "source": visible.source,
                "revision": visible.revision,
                "mlx_bits": self.config.mlx_bits,
                "error": error,
            },
            "typesafe": {
                "sdk_installed": _typesafe_sdk_available(),
                "key_configured": bool(os.environ.get("TYPESAFE_API_KEY")),
                "model": "jev-latest",
            },
            "bocha": {
                "sdk_installed": _typesafe_sdk_available(),
                "key_configured": bocha_key_configured,
                "credential_source": bocha_credential_source,
                "base_url": bocha_base_url,
                "model": bocha_model,
            },
        }

    def configure_bocha(self, payload: Any) -> dict[str, Any]:
        if not isinstance(payload, dict):
            raise RequestError("Jev Bocha configuration must be a JSON object")
        unknown = set(payload) - {"api_key", "base_url", "model"}
        if unknown:
            raise RequestError(f"Unknown Jev Bocha configuration field: {sorted(unknown)[0]}")
        api_key = _validated_bocha_api_key(payload.get("api_key"))
        base_url = _validated_bocha_base_url(payload.get("base_url", BOCHA_JEV_BASE_URL))
        model = _validated_bocha_model(payload.get("model", BOCHA_JEV_DEFAULT_MODEL))
        with self._condition:
            self._bocha_api_key = api_key
            self._bocha_base_url = base_url
            self._bocha_model = model
            self._bocha_credential_source = "setup"
        return self.health()["bocha"]

    def _load_locked(self, model_id: Any = None) -> dict[str, Any]:
        spec = self._resolve_model(model_id)
        with self._condition:
            self._requested_model_id = spec.id
            if self._status == "ready" and self._active_model_id == spec.id:
                return self.health()["local_model"]
            self._status = "loading"
            self._error = None
        try:
            with self._condition:
                self._release_model()
            from semif_phase1 import mlx_backend

            model, tokenizer, metadata = mlx_backend.load_model(
                spec.source,
                spec.revision,
                None if spec.source_quantization_bits is not None else self.config.mlx_bits,
                cache_limit_mib=self.config.mlx_cache_limit_mib,
            )
        except Exception as error:
            with self._condition:
                self._status = "error"
                self._error = str(error)
            raise
        with self._condition:
            self._model = model
            self._tokenizer = tokenizer
            self._metadata = metadata
            self._active_model_id = spec.id
            self._status = "ready"
        return self.health()["local_model"]

    def load(self, model_id: Any = None) -> dict[str, Any]:
        return self._executor.submit(self._load_on_model_thread, model_id).result()

    def _load_on_model_thread(self, model_id: Any = None) -> dict[str, Any]:
        with self._inference_lock:
            return self._load_locked(model_id)

    def evaluate_local(self, payload: Any) -> dict[str, Any]:
        state, questions, _ = normalize_payload(payload)
        rows = local_rows(state, questions)
        return self._executor.submit(
            self._evaluate_local_on_model_thread,
            payload.get("local_model"),
            rows,
            questions,
        ).result()

    def _evaluate_local_on_model_thread(
        self,
        model_id: Any,
        rows: list[dict[str, Any]],
        questions: dict[str, dict[str, Any]],
    ) -> dict[str, Any]:
        from semif_phase1 import mlx_backend

        with self._inference_lock:
            self._load_locked(model_id)
            results, timing = mlx_backend.score_shared(
                self._model, self._tokenizer, rows, self._metadata
            )
        return adapt_local_response(results, timing, questions, self._metadata)

    def close(self) -> None:
        """Release MLX state on its owning thread, then stop that worker."""
        try:
            self._executor.submit(self._release_on_model_thread).result()
        finally:
            self._executor.shutdown(wait=True)

    def _release_on_model_thread(self) -> None:
        with self._inference_lock, self._condition:
            self._release_model()
            self._status = "unloaded"
            self._error = None

    def evaluate_typesafe(self, payload: Any) -> dict[str, Any]:
        state, questions, model = normalize_payload(payload)
        if not os.environ.get("TYPESAFE_API_KEY"):
            raise RequestError(
                "TYPESAFE_API_KEY is not configured. Set it in the server environment and restart."
            )
        try:
            from typesafe_sdk import Choice, Noul, Score, TypeSafeClient
        except ImportError as error:
            raise RuntimeError(
                "TypeSafe SDK is not installed. Install the playground extra: "
                "pip install -e '.[test,mlx,playground]'"
            ) from error
        typed = typesafe_question_objects(
            questions,
            {"noul": Noul, "choice": Choice, "score": Score},
        )
        with TypeSafeClient() as client:
            kwargs = {"state": state, "questions": typed}
            if "model" in inspect.signature(client.system_one).parameters:
                kwargs["model"] = model
            started = time.perf_counter()
            response = client.system_one(**kwargs)
            elapsed_seconds = time.perf_counter() - started
        data = _attach_client_timing(_serialize_sdk_response(response), elapsed_seconds)
        data.setdefault("provider", "TypeSafe Jev")
        return data

    def evaluate_bocha(self, payload: Any) -> dict[str, Any]:
        state, questions, _ = normalize_payload(payload)
        with self._condition:
            api_key = self._bocha_api_key
            base_url = self._bocha_base_url
            model = self._bocha_model
        if not api_key:
            raise RequestError(
                "Jev Bocha is not configured. Add its API key in Setup or set BOCHA_JEV_API_KEY."
            )
        try:
            from typesafe_sdk import Choice, Noul, Score, TypeSafeClient
        except ImportError as error:
            raise RuntimeError(
                "TypeSafe SDK is not installed. Install the playground extra: "
                "pip install -e '.[test,mlx,playground]'"
            ) from error
        typed = typesafe_question_objects(
            questions,
            {"noul": Noul, "choice": Choice, "score": Score},
        )
        with TypeSafeClient(api_key=api_key, base_url=base_url, model=model) as client:
            started = time.perf_counter()
            response = client.system_one(state=state, questions=typed)
            elapsed_seconds = time.perf_counter() - started
        data = _attach_client_timing(_serialize_sdk_response(response), elapsed_seconds)
        data["provider"] = "Jev Bocha"
        return data


def _typesafe_sdk_available() -> bool:
    try:
        import typesafe_sdk  # noqa: F401
    except ImportError:
        return False
    return True


class PlaygroundHTTPServer(ThreadingHTTPServer):
    daemon_threads = True

    def __init__(self, server_address: tuple[str, int], service: SharedMLXService):
        self.service = service
        super().__init__(server_address, PlaygroundHandler)


class PlaygroundHandler(SimpleHTTPRequestHandler):
    server_version = "SemIfPlayground/1.0"

    def __init__(self, *args: Any, **kwargs: Any):
        super().__init__(*args, directory=str(STATIC_DIR), **kwargs)

    @property
    def service(self) -> SharedMLXService:
        return self.server.service  # type: ignore[attr-defined]

    def list_directory(self, path: str) -> None:
        self.send_error(HTTPStatus.NOT_FOUND)

    def _origin_allowed(self) -> bool:
        origin = self.headers.get("Origin")
        if not origin:
            return True
        try:
            return urlsplit(origin).hostname in LOCAL_ORIGIN_HOSTS
        except ValueError:
            return False

    def end_headers(self) -> None:
        origin = self.headers.get("Origin")
        if origin and self._origin_allowed():
            self.send_header("Access-Control-Allow-Origin", origin)
            self.send_header("Vary", "Origin")
        self.send_header("X-Content-Type-Options", "nosniff")
        self.send_header("Referrer-Policy", "no-referrer")
        self.send_header("Cross-Origin-Opener-Policy", "same-origin")
        super().end_headers()

    def do_OPTIONS(self) -> None:  # noqa: N802
        if not self._origin_allowed():
            self._send_json(HTTPStatus.FORBIDDEN, {"error": {"message": "Origin is not allowed"}})
            return
        self.send_response(HTTPStatus.NO_CONTENT)
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.send_header("Access-Control-Max-Age", "600")
        self.end_headers()

    def do_GET(self) -> None:  # noqa: N802
        if self.path.split("?", 1)[0] == "/api/health":
            self._send_json(HTTPStatus.OK, self.service.health())
            return
        super().do_GET()

    def do_POST(self) -> None:  # noqa: N802
        if not self._origin_allowed():
            self._send_json(HTTPStatus.FORBIDDEN, {"error": {"message": "Origin is not allowed"}})
            return
        route = self.path.split("?", 1)[0]
        try:
            if route == "/api/model/load":
                body = self._read_json_body(optional=True)
                if not isinstance(body, dict):
                    raise RequestError("Model load body must be a JSON object")
                data = {"local_model": self.service.load(body.get("model"))}
            elif route == "/api/config/bocha":
                data = {"bocha": self.service.configure_bocha(self._read_json_body())}
            elif route == "/api/evaluate/local":
                data = self.service.evaluate_local(self._read_json_body())
            elif route == "/api/evaluate/bocha":
                data = self.service.evaluate_bocha(self._read_json_body())
            elif route == "/api/evaluate/typesafe":
                data = self.service.evaluate_typesafe(self._read_json_body())
            else:
                self._send_json(HTTPStatus.NOT_FOUND, {"error": {"message": "Unknown API route"}})
                return
        except RequestError as error:
            self._send_json(HTTPStatus.BAD_REQUEST, {"error": {"message": str(error)}})
            return
        except Exception as error:
            self.log_error("API failure on %s: %s", route, error)
            self._send_json(
                HTTPStatus.SERVICE_UNAVAILABLE,
                {"error": {"message": str(error) or error.__class__.__name__}},
            )
            return
        self._send_json(HTTPStatus.OK, data)

    def _read_json_body(self, optional: bool = False) -> Any:
        raw_length = self.headers.get("Content-Length")
        if raw_length is None:
            if optional:
                return {}
            raise RequestError("Content-Length is required")
        try:
            length = int(raw_length)
        except ValueError as error:
            raise RequestError("Invalid Content-Length") from error
        if length < 0 or length > MAX_REQUEST_BYTES:
            raise RequestError(f"Request body must be at most {MAX_REQUEST_BYTES} bytes")
        body = self.rfile.read(length)
        if not body and optional:
            return {}
        try:
            return json.loads(body, parse_constant=_reject_json_constant)
        except (UnicodeDecodeError, json.JSONDecodeError) as error:
            raise RequestError(f"Request body is not valid JSON: {error}") from error

    def _send_json(self, status: HTTPStatus, payload: Any) -> None:
        body = json.dumps(payload, ensure_ascii=False, allow_nan=False).encode("utf-8")
        try:
            self.send_response(status)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(body)
        except BrokenPipeError:
            return


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="127.0.0.1", choices=sorted(BIND_HOSTS))
    parser.add_argument("--port", type=int, default=8090)
    parser.add_argument("--default-model", choices=sorted(BUILTIN_LOCAL_MODELS), default=DEFAULT_LOCAL_MODEL_ID)
    parser.add_argument("--model-source")
    parser.add_argument("--model-revision")
    parser.add_argument("--mlx-bits", type=int, choices=(4, 8))
    parser.add_argument("--mlx-cache-limit-mib", type=int, default=256)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.mlx_cache_limit_mib < 0:
        raise SystemExit("--mlx-cache-limit-mib must be nonnegative")
    if bool(args.model_source) != bool(args.model_revision):
        raise SystemExit("--model-source and --model-revision must be provided together")
    service = SharedMLXService(
        ServiceConfig(
            default_model_id=args.default_model,
            model_source=args.model_source,
            model_revision=args.model_revision,
            mlx_bits=args.mlx_bits,
            mlx_cache_limit_mib=args.mlx_cache_limit_mib,
        )
    )
    server = PlaygroundHTTPServer((args.host, args.port), service)
    print(f"SemIf decision workbench: http://{args.host}:{args.port}")
    print("MLX model loads on demand and is shared for the life of this process.")
    print(
        "TypeSafe Jev comparison: "
        + ("configured" if os.environ.get("TYPESAFE_API_KEY") else "TYPESAFE_API_KEY not set")
    )
    print(
        "Jev Bocha comparison: "
        + ("configured" if service.health()["bocha"]["key_configured"] else "setup required")
    )
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nStopping playground.")
    finally:
        service.close()
        server.server_close()


if __name__ == "__main__":
    main()
