"""TypeSafe System One wire-format adapter for SemIf scorers."""

from __future__ import annotations

from dataclasses import dataclass
import json
import math
from threading import Lock
from typing import Callable


MAX_OPTIONS = 16
CONFIDENCE_METHOD = "one-minus-normalized-entropy"


class SystemOneValidationError(ValueError):
    """A request cannot be represented by the SemIf decision backend."""


@dataclass(frozen=True)
class QuestionSpec:
    id: str
    kind: str
    option_ids: tuple[str, ...]
    legend: tuple[str, ...] = ()


def _error(path: str, message: str):
    raise SystemOneValidationError(f"{path}: {message}")


def _json_text(value, path: str) -> str:
    if not isinstance(value, (str, dict, list)):
        _error(path, "must be a string, object, or array")
    if isinstance(value, str) and not value:
        _error(path, "must not be empty")
    try:
        return value if isinstance(value, str) else json.dumps(value, ensure_ascii=False, allow_nan=False)
    except (TypeError, ValueError) as error:
        raise SystemOneValidationError(f"{path}: must contain finite JSON values") from error


def _description(option_id: str, value, path: str) -> str:
    if value is None:
        return option_id
    return f"{option_id}: {_json_text(value, path)}"


def _instructions(value, kind: str, path: str) -> str:
    if value is not None:
        return _json_text(value, path)
    return {
        "choice": "Which option best matches the supplied state?",
        "noul": "Does the true outcome apply to the supplied state?",
        "score": "Which ordered level best matches the supplied state?",
    }[kind]


def _validate_state(state) -> None:
    if not isinstance(state, (str, dict, list)) or not state:
        _error("state", "must be a nonempty string, object, or array")
    try:
        json.dumps(state, ensure_ascii=False, allow_nan=False)
    except (TypeError, ValueError) as error:
        raise SystemOneValidationError("state: must contain finite JSON values") from error


def request_rows(payload: dict, served_model: str) -> tuple[list[QuestionSpec], list[dict]]:
    """Validate a System One request and convert its questions to SemIf rows."""
    if not isinstance(payload, dict):
        _error("body", "must be a JSON object")
    _validate_state(payload.get("state"))
    model = payload.get("model")
    if model != served_model:
        _error("model", f"must be {served_model!r}; this server does not serve Jev aliases")
    questions = payload.get("questions")
    if not isinstance(questions, dict) or not questions:
        _error("questions", "must be a nonempty object")

    specs, rows = [], []
    for question_id, question in questions.items():
        base = f"questions.{question_id}"
        if not isinstance(question_id, str) or not question_id:
            _error("questions", "question IDs must be nonempty strings")
        if not isinstance(question, dict):
            _error(base, "must be an object")
        kind = question.get("type")
        if kind not in {"choice", "noul", "score"}:
            _error(f"{base}.type", "must be one of: choice, noul, score")
        instructions = _instructions(question.get("instructions"), kind, f"{base}.instructions")
        options: list[dict]
        option_ids: list[str]
        legend: tuple[str, ...] = ()

        if kind == "choice":
            criteria = question.get("criteria")
            if not isinstance(criteria, dict) or not 2 <= len(criteria) <= MAX_OPTIONS:
                _error(f"{base}.criteria", f"must contain 2-{MAX_OPTIONS} options")
            option_ids, options = [], []
            for option_id, value in criteria.items():
                if not isinstance(option_id, str) or not option_id:
                    _error(f"{base}.criteria", "option IDs must be nonempty strings")
                if value is not None and not isinstance(value, (str, dict, list)):
                    _error(f"{base}.criteria.{option_id}", "must be a string, object, array, or null")
                option_ids.append(option_id)
                options.append({
                    "id": option_id,
                    "description": _description(option_id, value, f"{base}.criteria.{option_id}"),
                })
        elif kind == "noul":
            criteria = question.get("criteria")
            if criteria is not None and not isinstance(criteria, dict):
                _error(f"{base}.criteria", "must be an object when supplied")
            criteria = criteria or {}
            unknown = set(criteria) - {"true", "false"}
            if unknown:
                _error(f"{base}.criteria", f"contains unsupported keys: {sorted(unknown)}")
            option_ids = ["true", "false"]
            values = [criteria.get("true", "The answer is yes."), criteria.get("false", "The answer is no.")]
            options = []
            for option_id, value in zip(option_ids, values):
                if value is not None and not isinstance(value, (str, dict, list)):
                    _error(f"{base}.criteria.{option_id}", "must be a string, object, array, or null")
                options.append({
                    "id": option_id,
                    "description": _description(option_id, value, f"{base}.criteria.{option_id}"),
                })
        elif kind == "score":
            criteria = question.get("criteria")
            if not isinstance(criteria, list) or not 2 <= len(criteria) <= 10:
                _error(f"{base}.criteria", "must contain 2-10 ordered levels")
            option_ids, options, rendered = [], [], []
            for index, value in enumerate(criteria):
                path = f"{base}.criteria.{index}"
                text = _json_text(value, path)
                option_id = str(index)
                option_ids.append(option_id)
                rendered.append(text)
                options.append({"id": option_id, "description": f"Level {option_id}: {text}"})
            legend = tuple(rendered)
        else:
            raise AssertionError("validated question type was not handled")

        specs.append(QuestionSpec(question_id, kind, tuple(option_ids), legend))
        rows.append({"id": question_id, "state": payload["state"], "question": instructions, "options": options})
    return specs, rows


def distribution_confidence(probabilities: list[float]) -> float:
    """Return a documented SemIf certainty proxy, not TypeSafe's private statistic."""
    if len(probabilities) < 2:
        raise ValueError("A distribution needs at least two probabilities")
    entropy = -sum(value * math.log(value) for value in probabilities if value > 0)
    return max(0.0, min(1.0, 1.0 - entropy / math.log(len(probabilities))))


def _probabilities(spec: QuestionSpec, result: dict) -> dict[str, float]:
    if result.get("id") != spec.id or result.get("option_ids") != list(spec.option_ids):
        raise RuntimeError(f"Scorer result for {spec.id!r} does not match its declared options")
    values = result.get("probabilities")
    if not isinstance(values, list) or len(values) != len(spec.option_ids):
        raise RuntimeError(f"Scorer result for {spec.id!r} has an invalid probability vector")
    if any(not isinstance(value, (int, float)) or not math.isfinite(value) or value < 0 for value in values):
        raise RuntimeError(f"Scorer result for {spec.id!r} has non-finite or negative probabilities")
    total = sum(values)
    if total <= 0:
        raise RuntimeError(f"Scorer result for {spec.id!r} has zero probability mass")
    normalized = [float(value / total) for value in values]
    return dict(zip(spec.option_ids, normalized))


def response_from_results(served_model: str, specs: list[QuestionSpec], results: list[dict]) -> dict:
    """Convert SemIf row results to the documented System One response shape."""
    if len(results) != len(specs):
        raise RuntimeError("Scorer returned a different number of results than requested")
    answers = {}
    input_tokens = 0
    prompt_versions = set()
    for spec, result in zip(specs, results):
        probabilities = _probabilities(spec, result)
        values = list(probabilities.values())
        winner = max(probabilities, key=probabilities.get)
        if spec.kind == "noul":
            answer = {"type": "noul", "noul": probabilities["true"]}
        elif spec.kind == "choice":
            answer = {
                "type": "choice",
                "choice": winner,
                "probabilities": probabilities,
                "confidence": distribution_confidence(values),
            }
        else:
            answer = {
                "type": "score",
                "score": sum(index * value for index, value in enumerate(values)),
                "legend": {str(index): text for index, text in enumerate(spec.legend)},
                "probabilities": probabilities,
                "confidence": distribution_confidence(values),
            }
        answers[spec.id] = answer
        input_tokens += int(result.get("input_tokens", 0))
        if result.get("prompt_version"):
            prompt_versions.add(result["prompt_version"])
    return {
        "model": served_model,
        "answers": answers,
        "usage": {"input_tokens": input_tokens, "output_tokens": 0},
        "semif": {
            "probability_status": "conditional option scores; uncalibrated as decision confidence",
            "confidence_method": CONFIDENCE_METHOD,
            "prompt_versions": sorted(prompt_versions),
        },
    }


class SystemOneService:
    """Serialize access to one resident model behind the System One wire format."""

    def __init__(
        self,
        score_rows: Callable[[list[dict]], list[dict]],
        served_model: str,
        description: str,
        release_date: str,
    ):
        if not all(isinstance(value, str) and value for value in (served_model, description, release_date)):
            raise ValueError("served model metadata must contain nonempty strings")
        self._score_rows = score_rows
        self._lock = Lock()
        self.served_model = served_model
        self.description = description
        self.release_date = release_date

    def evaluate(self, payload: dict) -> dict:
        specs, rows = request_rows(payload, self.served_model)
        try:
            with self._lock:
                results = self._score_rows(rows)
        except ValueError as error:
            raise SystemOneValidationError(f"questions: {error}") from error
        return response_from_results(self.served_model, specs, results)

    def models(self) -> dict:
        return {"models": [{
            "name": self.served_model,
            "description": self.description,
            "release_date": self.release_date,
        }]}
