"""Jev request -> SemIf row translation. Pure: no model, no server, no weights."""

import json

import pytest

from semif_api import slots
from semif_api.errors import ApiError
from semif_api.translate import NOUL_FALSE_DEFAULT, NOUL_TRUE_DEFAULT, render, translate

STATE = "Help! My payouts have been failing for 3 days."


def request(**questions):
    return {"state": STATE, "model": "jev-latest", "questions": questions}


def only(payload):
    translated = translate(payload)
    assert len(translated.rows) == 1
    return translated.rows[0], translated.plans[0]


def test_noul_maps_to_a_two_option_row_with_yes_first():
    row, plan = only(request(is_urgent={
        "type": "noul",
        "instructions": "Does this convey urgency?",
        "criteria": {"true": "Explicitly time-sensitive", "false": "No urgency expressed"},
    }))
    assert row == {
        "id": "is_urgent",
        "state": STATE,
        "question": "Does this convey urgency?",
        "options": [
            {"id": "true", "description": "Explicitly time-sensitive"},
            {"id": "false", "description": "No urgency expressed"},
        ],
    }
    assert plan.kind == "noul" and plan.option_ids == ("true", "false") and plan.legend is None


@pytest.mark.parametrize("criteria", [None, {}, {"true": "Time-sensitive"}])
def test_noul_criteria_are_optional_and_fall_back_to_frozen_defaults(criteria):
    question = {"type": "noul", "instructions": "Urgent?"}
    if criteria is not None:
        question["criteria"] = criteria
    row, _ = only(request(q=question))
    expected_true = criteria.get("true", NOUL_TRUE_DEFAULT) if criteria else NOUL_TRUE_DEFAULT
    assert [option["description"] for option in row["options"]] == [expected_true, NOUL_FALSE_DEFAULT]


def test_noul_rejects_criteria_keys_it_would_otherwise_drop_silently():
    with pytest.raises(ApiError) as error:
        translate(request(q={"type": "noul", "instructions": "Urgent?", "criteria": {"yes": "Time-sensitive"}}))
    assert error.value.status == 422
    assert error.value.code == "invalid_criteria"
    assert error.value.param == "questions.q.criteria"


def test_choice_preserves_criteria_order_and_keys():
    row, plan = only(request(department={
        "type": "choice",
        "instructions": "Which team should handle this?",
        "criteria": {
            "billing": "Payments, invoicing, refunds",
            "technical": "Bugs, outages, integrations",
            "sales": "Pricing, upgrades, new accounts",
        },
    }))
    assert [option["id"] for option in row["options"]] == ["billing", "technical", "sales"]
    assert row["options"][0]["description"] == "Payments, invoicing, refunds"
    assert plan.option_ids == ("billing", "technical", "sales")


@pytest.mark.parametrize("value", [None, ""])
def test_choice_option_without_a_description_still_shows_the_model_its_name(value):
    row, _ = only(request(q={
        "type": "choice", "instructions": "Which?", "criteria": {"billing": value, "sales": "Pricing"},
    }))
    assert row["options"][0] == {"id": "billing", "description": "billing"}


def test_score_levels_become_indexed_options_and_a_legend():
    row, plan = only(request(frustration={
        "type": "score",
        "instructions": "How frustrated is the customer?",
        "criteria": ["Calm", "Frustrated", "Very angry"],
    }))
    assert [option["id"] for option in row["options"]] == ["0", "1", "2"]
    assert plan.legend == {"0": "Calm", "1": "Frustrated", "2": "Very angry"}


def test_structured_instructions_keep_field_order():
    instructions = {
        "potential_duplicate": {"name": "John Smith", "location": "Oakland, California"},
        "question": "Is the resume for the same person as `potential_duplicate`?",
    }
    row, _ = only(request(q={"type": "noul", "instructions": instructions}))
    assert row["question"] == json.dumps(instructions, ensure_ascii=False)
    assert row["question"].index("potential_duplicate") < row["question"].index("question")


def test_structured_criteria_are_rendered_as_json():
    row, _ = only(request(q={
        "type": "choice", "instructions": "Which?",
        "criteria": {"billing": {"covers": ["refunds"], "excludes": ["bugs"]}, "sales": "Pricing"},
    }))
    assert row["options"][0]["description"] == '{"covers": ["refunds"], "excludes": ["bugs"]}'


def test_question_keys_become_row_ids_and_never_reach_the_prompt():
    translated = translate(request(
        a={"type": "noul", "instructions": "First?"},
        b={"type": "noul", "instructions": "Second?"},
    ))
    assert [row["id"] for row in translated.rows] == ["a", "b"]
    assert all(row["id"] not in row["question"] for row in translated.rows)


def test_state_object_is_carried_through_by_identity():
    state = {"ticket": {"subject": "Duplicate charge"}}
    translated = translate({"state": state, "model": "jev-latest",
                            "questions": {"q": {"type": "noul", "instructions": "Refund?"}}})
    assert translated.rows[0]["state"] is state


def test_every_row_shares_one_state_object_so_shared_scoring_applies():
    state = {"ticket": "x"}
    translated = translate({"state": state, "model": "jev-latest", "questions": {
        "a": {"type": "noul", "instructions": "First?"},
        "b": {"type": "choice", "instructions": "Which?", "criteria": {"x": None, "y": None}},
    }})
    assert all(row["state"] is state for row in translated.rows)


@pytest.mark.parametrize("payload,code,param", [
    ({"model": "jev-latest", "questions": {}}, "missing_field", "state"),
    ({"state": STATE, "questions": {}}, "missing_field", "model"),
    ({"state": STATE, "model": "jev-latest"}, "missing_field", "questions"),
    ({"state": "", "model": "jev-latest", "questions": {"q": {}}}, "invalid_state", "state"),
    ({"state": 7, "model": "jev-latest", "questions": {"q": {}}}, "invalid_state", "state"),
    ({"state": STATE, "model": "", "questions": {"q": {}}}, "invalid_model", "model"),
    ({"state": STATE, "model": "jev-latest", "questions": {}}, "invalid_questions", "questions"),
])
def test_top_level_validation(payload, code, param):
    with pytest.raises(ApiError) as error:
        translate(payload)
    assert error.value.status == 422
    assert (error.value.code, error.value.param) == (code, param)


@pytest.mark.parametrize("question,code,param", [
    ({"type": "guess", "instructions": "?"}, "invalid_question_type", "questions.q.type"),
    ({"type": "noul"}, "missing_field", "questions.q.instructions"),
    ({"type": "noul", "instructions": "   "}, "empty_text", "questions.q.instructions"),
    ({"type": "noul", "instructions": {}}, "empty_text", "questions.q.instructions"),
    ({"type": "noul", "instructions": 7}, "invalid_text", "questions.q.instructions"),
    ({"type": "choice", "instructions": "?"}, "invalid_criteria", "questions.q.criteria"),
    ({"type": "choice", "instructions": "?", "criteria": {"only": None}},
     "option_count_unsupported", "questions.q.criteria"),
    ({"type": "score", "instructions": "?", "criteria": {"a": "b"}}, "invalid_criteria", "questions.q.criteria"),
    ({"type": "score", "instructions": "?", "criteria": ["only"]},
     "option_count_unsupported", "questions.q.criteria"),
])
def test_question_validation(question, code, param):
    with pytest.raises(ApiError) as error:
        translate(request(q=question))
    assert error.value.status == 422
    assert (error.value.code, error.value.param) == (code, param)


def test_choice_above_the_slot_ceiling_is_refused_not_truncated():
    criteria = {f"option{index}": None for index in range(slots.CAPACITY[slots.V1] + 1)}
    with pytest.raises(ApiError) as error:
        translate(request(department={"type": "choice", "instructions": "Which?", "criteria": criteria}))
    assert error.value.code == "option_count_unsupported"
    assert "direct-options-v1 supports 2-16" in error.value.message


def test_choice_at_the_slot_ceiling_is_accepted():
    criteria = {f"option{index}": None for index in range(slots.CAPACITY[slots.V1])}
    row, _ = only(request(q={"type": "choice", "instructions": "Which?", "criteria": criteria}))
    assert len(row["options"]) == slots.CAPACITY[slots.V1]


def test_declared_but_unimplemented_prompt_version_is_refused_loudly():
    assert slots.CAPACITY[slots.V2] == 255
    with pytest.raises(ValueError, match="not implemented"):
        slots.resolve(slots.V2)


def test_render_rejects_non_finite_numbers():
    with pytest.raises(ApiError) as error:
        render({"value": float("nan")}, param="questions.q.instructions", field="instructions")
    assert error.value.code == "invalid_text"
