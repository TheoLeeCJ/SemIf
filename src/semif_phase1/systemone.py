"""TypeSafe-compatible SystemOne mapping on top of SemIf's direct scorer.

Implements the request/answer shapes from https://docs.typesafe.ai/llms.txt
(``POST /v1/systemone`` with ``{state, model, questions}`` and typed
``choice`` / ``score`` / ``noul`` questions) without any model dependency,
so the mapping is unit-testable without weights.

Honest boundaries vs the hosted API:

- Choice is limited to 2-16 options (SemIf reads single-token answer slots
  A-P; the hosted API accepts up to 255).
- Score is limited to 2-10 ordered levels, matching the hosted API.
- Probabilities are conditional on the supplied options and uncalibrated;
  ``confidence`` is the documented spread statistic
  ``(n * peak - 1) / (n - 1)``, not a calibration claim.
- ``usage.output_tokens`` is always 0: direct readout generates no tokens.
"""

from __future__ import annotations

import json

QUESTION_TYPES = ("choice", "score", "noul")
MAX_OPTIONS = 16  # single-token answer slots A-P in semif_phase1.direct
MAX_LEVELS = 10  # matches the hosted API limit
MAX_QUESTIONS = 256


class SystemOneError(ValueError):
    """Validation failure carrying a loc path for 422 detail responses."""

    def __init__(self, message: str, loc=("body",)):
        super().__init__(message)
        self.loc = list(loc)


def render_content(value, field: str) -> str:
    """Render string|object|array instructions/criteria to prompt text."""
    if isinstance(value, str):
        if value.strip():
            return value
        raise SystemOneError(f"{field} must be a nonempty string, object, or array", loc=("body", field))
    if isinstance(value, (dict, list)) and value:
        try:
            return json.dumps(value, ensure_ascii=False, allow_nan=False)
        except (TypeError, ValueError) as error:
            raise SystemOneError(f"{field} must be finite JSON-compatible data", loc=("body", field)) from error
    raise SystemOneError(f"{field} must be a nonempty string, object, or array", loc=("body", field))


def option_description(value, fallback: str, field: str) -> str:
    """Render one choice option description; null falls back to the option name."""
    if value is None:
        return fallback
    return render_content(value, field)


def confidence(probabilities: list[float]) -> float:
    """Spread statistic from the hosted docs: (n * peak - 1) / (n - 1)."""
    count = len(probabilities)
    if count < 2:
        raise SystemOneError("Need at least two probabilities for confidence", loc=("body",))
    peak = max(probabilities)
    return max(0.0, min(1.0, (count * peak - 1.0) / (count - 1.0)))


def validate_body(body) -> tuple:
    """Return (state, model, questions) or raise SystemOneError."""
    if not isinstance(body, dict):
        raise SystemOneError("Body must be a JSON object with state, model, and questions")
    missing = [key for key in ("state", "model", "questions") if key not in body]
    if missing:
        raise SystemOneError(f"Body is missing fields: {sorted(missing)}")
    state = body["state"]
    if not isinstance(state, (str, dict, list)) or not state:
        raise SystemOneError("state must be a nonempty string, object, or array", loc=("body", "state"))
    try:
        json.dumps(state, ensure_ascii=False, allow_nan=False)
    except (TypeError, ValueError) as error:
        raise SystemOneError("state must be finite JSON-compatible data", loc=("body", "state")) from error
    if not isinstance(body["model"], str) or not body["model"].strip():
        raise SystemOneError("model must be a nonempty string", loc=("body", "model"))
    questions = body["questions"]
    if not isinstance(questions, dict) or not questions:
        raise SystemOneError("questions must be a nonempty map of id to question", loc=("body", "questions"))
    if len(questions) > MAX_QUESTIONS:
        raise SystemOneError(f"questions is limited to {MAX_QUESTIONS} entries", loc=("body", "questions"))
    for qid, question in questions.items():
        validate_question(qid, question)
    return state, body["model"], questions


def validate_question(qid, question) -> None:
    loc = ("body", "questions", qid)
    if not isinstance(qid, str) or not qid:
        raise SystemOneError("Question ids must be nonempty strings", loc=("body", "questions"))
    if not isinstance(question, dict):
        raise SystemOneError(f"Question {qid!r} must be an object", loc=loc)
    qtype = question.get("type")
    if qtype not in QUESTION_TYPES:
        raise SystemOneError(f"Question {qid!r} type must be one of {list(QUESTION_TYPES)}", loc=(*loc, "type"))
    if "instructions" not in question or question["instructions"] is None:
        raise SystemOneError(
            f"Question {qid!r} needs instructions (SemIf has no default question text)", loc=(*loc, "instructions")
        )
    render_content(question["instructions"], f"questions.{qid}.instructions")
    criteria = question.get("criteria")
    if qtype == "choice":
        if not isinstance(criteria, dict) or not 2 <= len(criteria) <= MAX_OPTIONS:
            raise SystemOneError(
                f"Choice question {qid!r} needs 2-{MAX_OPTIONS} criteria options "
                "(SemIf limit; hosted API accepts up to 255)",
                loc=(*loc, "criteria"),
            )
        for option, description in criteria.items():
            if not isinstance(option, str) or not option:
                raise SystemOneError(f"Choice question {qid!r} has an invalid option name", loc=(*loc, "criteria"))
            if description is not None:
                render_content(description, f"questions.{qid}.criteria.{option}")
    elif qtype == "score":
        if not isinstance(criteria, list) or not 2 <= len(criteria) <= MAX_LEVELS:
            raise SystemOneError(
                f"Score question {qid!r} needs 2-{MAX_LEVELS} ordered criteria levels", loc=(*loc, "criteria")
            )
        for index, level in enumerate(criteria):
            if level is None:
                raise SystemOneError(
                    f"Score question {qid!r} level {index} must not be null", loc=(*loc, "criteria", index)
                )
            render_content(level, f"questions.{qid}.criteria[{index}]")
    else:  # noul
        if criteria is not None:
            if not isinstance(criteria, dict) or any(key not in ("true", "false") for key in criteria):
                raise SystemOneError(
                    f"Noul question {qid!r} criteria may only hold 'true'/'false' descriptions",
                    loc=(*loc, "criteria"),
                )
            for key, description in criteria.items():
                if description is not None:
                    render_content(description, f"questions.{qid}.criteria.{key}")


def questions_to_rows(state, questions: dict) -> list[dict]:
    """Convert validated questions to SemIf rows (one per question id)."""
    rows = []
    for qid, question in questions.items():
        text = render_content(question["instructions"], f"questions.{qid}.instructions")
        qtype = question["type"]
        if qtype == "choice":
            options = [
                {"id": option, "description": option_description(desc, option, f"questions.{qid}.criteria.{option}")}
                for option, desc in question["criteria"].items()
            ]
        elif qtype == "score":
            options = [
                {"id": str(index), "description": render_content(level, f"questions.{qid}.criteria[{index}]")}
                for index, level in enumerate(question["criteria"])
            ]
        else:  # noul -> binary yes/no readout; p(yes) is the noul value
            criteria = question.get("criteria") or {}
            options = [
                {
                    "id": "true",
                    "description": option_description(criteria.get("true"), "Yes", f"questions.{qid}.criteria.true"),
                },
                {
                    "id": "false",
                    "description": option_description(criteria.get("false"), "No", f"questions.{qid}.criteria.false"),
                },
            ]
        rows.append({"id": qid, "state": state, "question": text, "options": options})
    return rows


def build_response(model_name: str, questions: dict, results: list[dict]) -> dict:
    """Build a {model, answers, usage} response from scorer results."""
    by_id = {result["id"]: result for result in results}
    if len(by_id) != len(questions):
        raise SystemOneError("Scorer did not return one result per question", loc=("body", "questions"))
    answers = {}
    input_tokens = 0
    for qid, question in questions.items():
        result = by_id[qid]
        probabilities = result["probabilities"]
        input_tokens += int(result.get("input_tokens", 0))
        qtype = question["type"]
        if qtype == "noul":
            answers[qid] = {"type": "noul", "noul": float(probabilities[0])}
        elif qtype == "choice":
            names = list(question["criteria"].keys())
            if len(names) != len(probabilities):
                raise SystemOneError(f"Scorer returned {len(probabilities)} values for {len(names)} options", loc=("body", "questions", qid))
            peak = max(range(len(probabilities)), key=lambda i: probabilities[i])
            answers[qid] = {
                "type": "choice",
                "choice": names[peak],
                "probabilities": {name: float(prob) for name, prob in zip(names, probabilities)},
                "confidence": confidence([float(prob) for prob in probabilities]),
            }
        else:  # score
            levels = question["criteria"]
            if len(levels) != len(probabilities):
                raise SystemOneError("Scorer/level count mismatch", loc=("body", "questions", qid))
            floats = [float(prob) for prob in probabilities]
            answers[qid] = {
                "type": "score",
                "score": float(sum(index * prob for index, prob in enumerate(floats))),
                "legend": {str(index): level for index, level in enumerate(levels)},
                "probabilities": {str(index): prob for index, prob in enumerate(floats)},
                "confidence": confidence(floats),
            }
    return {
        "model": model_name,
        "answers": answers,
        "usage": {"input_tokens": input_tokens, "output_tokens": 0},
    }
