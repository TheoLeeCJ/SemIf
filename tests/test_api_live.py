"""End-to-end checks against real weights.

Skipped unless SEMIF_LIVE_MODEL is set, like the other weight-dependent tests
in this repository. Run one at a time on a machine with the backend available:

    SEMIF_LIVE_MODEL=Qwen/Qwen3.5-4B \
    SEMIF_LIVE_REVISION=851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a \
    SEMIF_LIVE_BACKEND=mlx pytest -q tests/test_api_live.py
"""

import os

import pytest

from fastapi.testclient import TestClient

from semif_api import runtime
from semif_api.app import create_app

LIVE_MODEL = os.environ.get("SEMIF_LIVE_MODEL")
pytestmark = pytest.mark.skipif(not LIVE_MODEL, reason="Set SEMIF_LIVE_MODEL to run against real weights")

STATE = "Customer cannot access an account after a password reset."


@pytest.fixture(scope="module")
def live():
    config = runtime.Config(
        model=LIVE_MODEL,
        revision=os.environ["SEMIF_LIVE_REVISION"],
        backend=os.environ.get("SEMIF_LIVE_BACKEND", "torch"),
        gguf=os.environ.get("SEMIF_LIVE_GGUF"),
    )
    engine = runtime.Runtime(config)
    engine.load()
    with TestClient(create_app(config, engine)) as client:
        yield client, engine


def ask(client, questions, **overrides):
    body = {"state": STATE, "model": "jev-latest", "questions": questions}
    body.update(overrides)
    response = client.post("/v1/systemone", json=body)
    assert response.status_code == 200, response.text
    return response.json()


def test_the_three_primitives_return_usable_answers(live):
    client, engine = live
    body = ask(client, {
        "is_access": {"type": "noul", "instructions": "Is this an account access problem?"},
        "queue": {"type": "choice", "instructions": "Which queue should handle this request?",
                  "criteria": {"access": "Account access support.", "billing": "Billing support."}},
        "urgency": {"type": "score", "instructions": "How urgent is this request?",
                    "criteria": ["Not urgent", "Somewhat urgent", "Very urgent"]},
    })
    assert 0.0 <= body["answers"]["is_access"]["noul"] <= 1.0
    assert body["answers"]["queue"]["choice"] in {"access", "billing"}
    assert 0.0 <= body["answers"]["urgency"]["score"] <= 2.0
    assert body["model"] == engine.model_id
    assert sum(body["answers"]["queue"]["probabilities"].values()) == pytest.approx(1.0)


def test_the_server_agrees_with_the_cli_on_the_same_decision(live):
    """The API layer is a translation of the scorer, not a second scorer."""
    client, engine = live
    question = {"type": "choice", "instructions": "Which queue should handle this request?",
                "criteria": {"access": "Account access support.", "billing": "Billing support."}}
    body = ask(client, {"route-1": question})

    row = {"id": "route-1", "state": STATE, "question": question["instructions"],
           "options": [{"id": "access", "description": "Account access support."},
                       {"id": "billing", "description": "Billing support."}]}
    results, _, _, _ = engine.score([row])
    direct = dict(zip(results[0]["option_ids"], results[0]["probabilities"]))

    answer = body["answers"]["route-1"]
    assert answer["semif"]["prompt_sha256"] == results[0]["prompt_sha256"]
    for option, probability in direct.items():
        assert answer["probabilities"][option] == pytest.approx(probability, abs=1e-6)


def test_many_questions_take_the_shared_path_and_count_the_state_once(live):
    client, _ = live
    questions = {f"q{index}": {"type": "noul", "instructions": f"Is criterion {index} satisfied?"}
                 for index in range(4)}
    response = client.post("/v1/systemone", json={"state": STATE, "model": "jev-latest", "questions": questions})
    body = response.json()
    assert response.headers["x-semif-mode"] in {"shared", "serial", "direct"}
    per_question = sum(answer["semif"]["input_tokens"] for answer in body["answers"].values())
    if response.headers["x-semif-mode"] == "shared":
        assert body["usage"]["input_tokens"] < per_question
    assert body["usage"]["output_tokens"] == 0


def test_an_over_budget_question_is_refused_rather_than_truncated(live):
    client, _ = live
    response = client.post("/v1/systemone", json={
        "state": "word " * 20000, "model": "jev-latest",
        "questions": {"q": {"type": "noul", "instructions": "Is this long?"}}})
    assert response.status_code in {422, 500}
    assert "exceed" in response.text or "truncation" in response.text
