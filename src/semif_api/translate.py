"""Jev request -> SemIf rows.

Pure functions over plain dictionaries: no model, no I/O, no global state.
The mapping is specified in docs/JEV_API_COMPAT.md section 4.

The state value is carried through untouched. `shared._state_prefix` locates
the evidence by exact substring match against its JSON serialization, so any
normalization here would silently drop multi-question requests to serial mode.
"""

from __future__ import annotations

from dataclasses import dataclass
import json

from semif_phase1.core import validate_row

from . import slots
from .errors import invalid_request

#: Frozen. Changing either string changes every Noul prompt hash.
NOUL_TRUE_DEFAULT = "The answer to the question is yes."
NOUL_FALSE_DEFAULT = "The answer to the question is no."

NOUL_OPTION_IDS = ("true", "false")
NOUL_CRITERIA_KEYS = frozenset(NOUL_OPTION_IDS)

QUESTION_TYPES = ("noul", "choice", "score")


@dataclass(frozen=True)
class QuestionPlan:
    """What assembly needs to turn one scored row back into a Jev answer."""

    key: str
    kind: str
    option_ids: tuple[str, ...]
    legend: dict[str, str] | None = None


@dataclass(frozen=True)
class Translated:
    model: str
    rows: list[dict]
    plans: list[QuestionPlan]


def render(value, *, param: str, field: str) -> str:
    """Render instructions or a criterion as the nonempty string SemIf needs.

    Insertion order is preserved: structured instructions put the question in
    one field and the data it refers to in others, so key order is meaning.
    """
    if isinstance(value, str):
        if not value.strip():
            raise invalid_request("empty_text", f"{field} is empty.", param)
        return value
    if isinstance(value, (dict, list)) and not isinstance(value, bool):
        if not value:
            raise invalid_request("empty_text", f"{field} is empty.", param)
        try:
            return json.dumps(value, ensure_ascii=False, allow_nan=False)
        except (TypeError, ValueError) as error:
            raise invalid_request(
                "invalid_text", f"{field} is not finite JSON-compatible data: {error}.", param
            ) from error
    raise invalid_request(
        "invalid_text", f"{field} must be a string, object, or array.", param
    )


def _instructions(question: dict, param: str) -> str:
    if "instructions" not in question:
        raise invalid_request("missing_field", "Question is missing 'instructions'.", f"{param}.instructions")
    return render(question["instructions"], param=f"{param}.instructions", field="instructions")


def _noul_options(question: dict, param: str) -> list[dict]:
    criteria = question.get("criteria")
    descriptions = {"true": NOUL_TRUE_DEFAULT, "false": NOUL_FALSE_DEFAULT}
    if criteria is not None:
        if not isinstance(criteria, dict):
            raise invalid_request(
                "invalid_criteria", "Noul criteria must be an object with 'true' and 'false'.", f"{param}.criteria"
            )
        unknown = sorted(set(criteria) - NOUL_CRITERIA_KEYS)
        if unknown:
            raise invalid_request(
                "invalid_criteria",
                f"Noul criteria accepts only 'true' and 'false'; got {unknown}.",
                f"{param}.criteria",
            )
        for key, value in criteria.items():
            if value is not None:
                descriptions[key] = render(
                    value, param=f"{param}.criteria.{key}", field=f"criteria.{key}"
                )
    return [{"id": key, "description": descriptions[key]} for key in NOUL_OPTION_IDS]


def _choice_options(question: dict, param: str, prompt_version: str) -> list[dict]:
    criteria = question.get("criteria")
    if not isinstance(criteria, dict):
        raise invalid_request(
            "invalid_criteria", "Choice criteria must be an object mapping each option to a description.",
            f"{param}.criteria",
        )
    slots.check_count(len(criteria), prompt_version, param=f"{param}.criteria", noun="Choice option")
    options = []
    for key, value in criteria.items():
        if not key.strip():
            raise invalid_request("invalid_criteria", "Choice option names must be nonempty.", f"{param}.criteria")
        # A null or empty description means the option needs no extra detail.
        # The option name is what still has to reach the model.
        described = key if value is None or value == "" else render(
            value, param=f"{param}.criteria.{key}", field=f"criteria.{key}"
        )
        options.append({"id": key, "description": described})
    return options


def _score_options(question: dict, param: str, prompt_version: str) -> tuple[list[dict], dict[str, str]]:
    criteria = question.get("criteria")
    if not isinstance(criteria, list):
        raise invalid_request(
            "invalid_criteria", "Score criteria must be an ordered array of level descriptions.",
            f"{param}.criteria",
        )
    slots.check_count(len(criteria), prompt_version, param=f"{param}.criteria", noun="Score level")
    options, legend = [], {}
    for index, level in enumerate(criteria):
        described = render(level, param=f"{param}.criteria[{index}]", field=f"criteria[{index}]")
        options.append({"id": str(index), "description": described})
        legend[str(index)] = described
    return options, legend


def _check_state(state):
    """Apply core's state rules here so the error points at 'state', not a question."""
    if not isinstance(state, (str, dict, list)) or isinstance(state, bool) or not state:
        raise invalid_request(
            "invalid_state", "'state' must be a nonempty string, object, or array.", "state"
        )
    try:
        json.dumps(state, ensure_ascii=False, allow_nan=False)
    except (TypeError, ValueError) as error:
        raise invalid_request(
            "invalid_state", f"'state' must be finite JSON-compatible data: {error}.", "state"
        ) from error
    return state


def translate(payload, *, prompt_version: str = slots.V1) -> Translated:
    """Turn one decoded request body into rows the SemIf scorers accept."""
    if not isinstance(payload, dict):
        raise invalid_request("invalid_body", "Request body must be a JSON object.")
    for field in ("state", "model", "questions"):
        if field not in payload:
            raise invalid_request("missing_field", f"Request is missing '{field}'.", field)
    model = payload["model"]
    if not isinstance(model, str) or not model.strip():
        raise invalid_request("invalid_model", "'model' must be a nonempty string.", "model")
    questions = payload["questions"]
    if not isinstance(questions, dict) or not questions:
        raise invalid_request("invalid_questions", "'questions' must be an object with at least one entry.", "questions")
    state = _check_state(payload["state"])

    rows, plans = [], []
    for key, question in questions.items():
        param = f"questions.{key}"
        if not key.strip():
            raise invalid_request("invalid_questions", "Question ids must be nonempty.", "questions")
        if not isinstance(question, dict):
            raise invalid_request("invalid_question", "A question must be an object.", param)
        kind = question.get("type")
        if kind not in QUESTION_TYPES:
            raise invalid_request(
                "invalid_question_type",
                f"Question type must be one of {list(QUESTION_TYPES)}; got {kind!r}.",
                f"{param}.type",
            )
        instructions = _instructions(question, param)
        legend = None
        if kind == "noul":
            options = _noul_options(question, param)
        elif kind == "choice":
            options = _choice_options(question, param, prompt_version)
        else:
            options, legend = _score_options(question, param, prompt_version)
        row = {"id": key, "state": state, "question": instructions, "options": options}
        try:
            validate_row(row)
        except ValueError as error:
            raise invalid_request("invalid_question", str(error), param) from error
        rows.append(row)
        plans.append(
            QuestionPlan(key=key, kind=kind, option_ids=tuple(o["id"] for o in options), legend=legend)
        )
    return Translated(model=model, rows=rows, plans=plans)
