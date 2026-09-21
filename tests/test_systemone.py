"""Pure mapping tests for the TypeSafe-compatible SystemOne layer."""

import pytest

from semif_phase1.systemone import (
    SystemOneError,
    build_response,
    confidence,
    questions_to_rows,
    validate_body,
)


def test_confidence_matches_documented_spread_statistic():
    assert confidence([1.0, 0.0, 0.0]) == 1.0
    assert confidence([1 / 3, 1 / 3, 1 / 3]) == pytest.approx(0.0)
    assert confidence([0.88, 0.12, 0.0]) == pytest.approx((3 * 0.88 - 1) / 2)
    assert confidence([0.7, 0.3]) == pytest.approx(0.4)


def test_choice_option_limit_is_honest_about_slots():
    body = {
        "state": "evidence",
        "model": "m",
        "questions": {
            "q": {
                "type": "choice",
                "instructions": "Pick one",
                "criteria": {f"opt{i}": f"Option {i}" for i in range(17)},
            }
        },
    }
    with pytest.raises(SystemOneError, match="2-16"):
        validate_body(body)


def test_score_rejects_null_levels_and_too_many():
    base = {"state": "evidence", "model": "m"}
    with pytest.raises(SystemOneError, match="must not be null"):
        validate_body({**base, "questions": {"q": {"type": "score", "instructions": "Rate", "criteria": ["a", None]}}})
    with pytest.raises(SystemOneError, match="2-10"):
        validate_body(
            {**base, "questions": {"q": {"type": "score", "instructions": "Rate", "criteria": ["a"] * 11}}}
        )


def test_noul_criteria_keys_are_restricted():
    body = {
        "state": "evidence",
        "model": "m",
        "questions": {"q": {"type": "noul", "instructions": "Yes?", "criteria": {"maybe": "hmm"}}},
    }
    with pytest.raises(SystemOneError, match="true.*false"):
        validate_body(body)


def test_structured_instructions_and_null_choice_description():
    body = {
        "state": {"ticket": "charged twice"},
        "model": "m",
        "questions": {
            "tone": {
                "type": "choice",
                "instructions": {"question": "What tone?", "focus": "Classify it"},
                "criteria": {"calm": None, "angry": "Strong language"},
            }
        },
    }
    state, _, questions = validate_body(body)
    (row,) = questions_to_rows(state, questions)
    assert "What tone?" in row["question"]
    assert row["options"][0] == {"id": "calm", "description": "calm"}


def test_noul_rows_carry_true_false_descriptions():
    body = {
        "state": "I asked three times; talk to a person please.",
        "model": "m",
        "questions": {
            "q": {
                "type": "noul",
                "instructions": "Wants a human?",
                "criteria": {"true": "Asks for a person", "false": "No sign of asking"},
            }
        },
    }
    _, _, questions = validate_body(body)
    (row,) = questions_to_rows("evidence", questions)
    assert row["options"] == [
        {"id": "true", "description": "Asks for a person"},
        {"id": "false", "description": "No sign of asking"},
    ]


def test_build_response_score_math_and_legend():
    questions = {"s": {"type": "score", "instructions": "Rate", "criteria": ["Calm", "Angry"]}}
    response = build_response(
        "fake", questions, [{"id": "s", "probabilities": [0.25, 0.75], "input_tokens": 12}]
    )
    answer = response["answers"]["s"]
    assert answer["score"] == pytest.approx(0.75)
    assert answer["legend"] == {"0": "Calm", "1": "Angry"}
    assert answer["probabilities"] == {"0": 0.25, "1": 0.75}
    assert answer["confidence"] == pytest.approx(0.5)
    assert response["usage"] == {"input_tokens": 12, "output_tokens": 0}
