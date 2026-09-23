import math

import pytest

from semif_phase1.core import (
    direct_messages,
    normalized_entropy_confidence,
    softmax,
    validate_row,
)


ROW = {
    "id": "x",
    "state": "owned evidence",
    "question": "Which answer follows?",
    "options": [
        {"id": "yes", "description": "Yes."},
        {"id": "no", "description": "No."},
    ],
}


def test_direct_prompt_excludes_extra_fields():
    row = dict(ROW, label="yes", provenance={"secret": "do not leak"})
    rendered = str(direct_messages(row))
    assert "owned evidence" in rendered
    assert "secret" not in rendered
    assert "label" not in rendered


def test_softmax_is_finite_and_normalized():
    values = softmax([1000.0, 999.0, -1000.0])
    assert all(math.isfinite(value) for value in values)
    assert sum(values) == pytest.approx(1.0)
    assert values[0] > values[1] > values[2]


@pytest.mark.parametrize(
    "probabilities, expected",
    [
        ([0.5, 0.5], 0.0),
        ([0.25, 0.25, 0.25, 0.25], 0.0),
        ([1.0, 0.0], 1.0),
        ([0.75, 0.25], 0.18872187554086717),
    ],
)
def test_normalized_entropy_confidence(probabilities, expected):
    assert normalized_entropy_confidence(probabilities) == pytest.approx(expected)


@pytest.mark.parametrize(
    "probabilities",
    [[1.0], [0.4, 0.4], [1.1, -0.1], [float("nan"), float("nan")]],
)
def test_normalized_entropy_confidence_rejects_invalid_distributions(probabilities):
    with pytest.raises(ValueError, match="probabilit|Probabilit|at least two"):
        normalized_entropy_confidence(probabilities)


def test_duplicate_options_rejected():
    row = dict(ROW, options=[ROW["options"][0], ROW["options"][0]])
    with pytest.raises(ValueError, match="unique"):
        validate_row(row)


def test_structured_json_state_is_supported():
    row = dict(ROW, state={"policy": "Never request passwords", "candidate": ["invoice id"]})
    validate_row(row)
    assert '"policy"' in direct_messages(row)[1]["content"]


def test_nonfinite_structured_state_is_rejected():
    with pytest.raises(ValueError, match="finite JSON-compatible"):
        validate_row(dict(ROW, state={"score": float("nan")}))
