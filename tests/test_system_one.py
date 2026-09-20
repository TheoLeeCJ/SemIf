import math

import pytest

from semif_phase1.system_one import (
    SystemOneService,
    SystemOneValidationError,
    distribution_confidence,
    request_rows,
)


MODEL = "semif-test"


def fake_score(rows):
    distributions = {
        "urgent": [0.8, 0.2],
        "team": [0.1, 0.7, 0.2],
        "severity": [0.1, 0.3, 0.6],
    }
    return [{
        "id": row["id"],
        "option_ids": [option["id"] for option in row["options"]],
        "probabilities": distributions[row["id"]],
        "input_tokens": 10,
        "prompt_version": "test-v1",
    } for row in rows]


def payload():
    return {
        "model": MODEL,
        "state": {"ticket": "Payouts failed for three days"},
        "questions": {
            "urgent": {"type": "noul", "instructions": "Is this urgent?"},
            "team": {
                "type": "choice",
                "instructions": {"question": "Which team?"},
                "criteria": {
                    "billing": "Invoices and charges",
                    "technical": {"covers": "Bugs and integrations"},
                    "other": None,
                },
            },
            "severity": {
                "type": "score",
                "instructions": "How severe is this?",
                "criteria": ["Minor", "Degraded", "Blocking"],
            },
        },
    }


def test_mixed_system_one_questions_are_adapted_and_returned():
    service = SystemOneService(fake_score, MODEL, "Test model", "2026-09-20")
    response = service.evaluate(payload())

    assert response["model"] == MODEL
    assert response["answers"]["urgent"] == {"type": "noul", "noul": pytest.approx(0.8)}
    assert response["answers"]["team"]["choice"] == "technical"
    assert response["answers"]["team"]["probabilities"] == {
        "billing": pytest.approx(0.1),
        "technical": pytest.approx(0.7),
        "other": pytest.approx(0.2),
    }
    assert response["answers"]["severity"]["score"] == pytest.approx(1.5)
    assert response["answers"]["severity"]["legend"] == {
        "0": "Minor", "1": "Degraded", "2": "Blocking"
    }
    assert response["usage"] == {"input_tokens": 30, "output_tokens": 0}
    assert response["semif"]["confidence_method"] == "one-minus-normalized-entropy"


def test_structured_values_and_option_names_are_present_in_rows():
    _, rows = request_rows(payload(), MODEL)
    team = next(row for row in rows if row["id"] == "team")
    assert '"question": "Which team?"' in team["question"]
    assert team["options"][0]["description"].startswith("billing:")
    assert '"covers": "Bugs and integrations"' in team["options"][1]["description"]
    assert team["options"][2]["description"] == "other"


def test_confidence_has_expected_entropy_limits():
    assert distribution_confidence([0.5, 0.5]) == pytest.approx(0.0)
    assert distribution_confidence([1.0, 0.0]) == pytest.approx(1.0)
    assert math.isfinite(distribution_confidence([0.1, 0.7, 0.2]))


@pytest.mark.parametrize("change,message", [
    ({"model": "jev-latest"}, "does not serve Jev aliases"),
    ({"state": ""}, "state"),
    ({"questions": {}}, "questions"),
])
def test_invalid_top_level_request_is_rejected(change, message):
    request = payload()
    request.update(change)
    with pytest.raises(SystemOneValidationError, match=message):
        request_rows(request, MODEL)


def test_choice_backend_limit_is_explicit():
    request = payload()
    request["questions"] = {"wide": {
        "type": "choice",
        "instructions": "Pick one",
        "criteria": {str(index): None for index in range(17)},
    }}
    with pytest.raises(SystemOneValidationError, match="2-16"):
        request_rows(request, MODEL)


def test_optional_instructions_receive_type_specific_fallbacks():
    request = payload()
    request["questions"] = {
        "binary": {"type": "noul"},
        "route": {"type": "choice", "instructions": None, "criteria": {"a": None, "b": None}},
        "rating": {"type": "score", "criteria": ["Low", "High"]},
    }
    _, rows = request_rows(request, MODEL)
    assert [row["question"] for row in rows] == [
        "Does the true outcome apply to the supplied state?",
        "Which option best matches the supplied state?",
        "Which ordered level best matches the supplied state?",
    ]
