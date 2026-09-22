"""Wire conformance for the Jev-compatible surface.

No weights and no network: the backend is a stub that returns fixed logits, so
these exercise the HTTP contract rather than the model. Where the official
SDK is installed, responses are validated against its own models and driven
through a real client over an in-process transport.
"""

import asyncio
import json
from pathlib import Path

import pytest

from fastapi.testclient import TestClient

from semif_api import runtime
from semif_api.app import create_app
from semif_api.translate import translate
from semif_phase1.core import direct_messages

jsonschema = pytest.importorskip("jsonschema")

SCHEMA = json.loads((Path(__file__).resolve().parents[1]
                     / "src/semif_api/schema/systemone.json").read_text())

STATE = "Help! My payouts have been failing for 3 days."
MODEL_ID = "semif/Qwen3.5-4B@851bf6e806ef+torch-bfloat16"

QUESTIONS = {
    "is_urgent": {"type": "noul", "instructions": "Does this convey urgency?",
                  "criteria": {"true": "Explicitly time-sensitive", "false": "No urgency expressed"}},
    "department": {"type": "choice", "instructions": "Which team should handle this?",
                   "criteria": {"billing": "Payments", "technical": "Bugs", "sales": "Pricing"}},
    "frustration": {"type": "score", "instructions": "How frustrated is the customer?",
                    "criteria": ["Calm", "Frustrated", "Very angry"]},
}


class StubEngine:
    """Scores rows with fixed descending logits: option 0 always wins."""

    def __init__(self, mode="shared", fail_modes=()):
        self.ready = True
        self.model_id = MODEL_ID
        self.metadata = {"source": "Qwen/Qwen3.5-4B", "revision": "851bf6e8" + "0" * 32}
        self.mode = mode
        self.fail_modes = set(fail_modes)
        self.seen = []

    def score(self, rows):
        self.seen.append(rows)
        results = []
        for row in rows:
            logits = [4.0 - index for index in range(len(row["options"]))]
            total = sum(2.718281828459045 ** logit for logit in logits)
            results.append({
                "id": row["id"],
                "option_ids": [option["id"] for option in row["options"]],
                "option_logits": logits,
                "probabilities": [2.718281828459045 ** logit / total for logit in logits],
                "input_tokens": 140 + len(row["options"]),
                "prompt_sha256": "0" * 64,
                "prompt_version": "direct-options-v1",
                "model": self.metadata,
                "readout": "native full-vocabulary last-position logits restricted to declared answer slots",
                "probability_status": "conditional option score; uncalibrated as decision confidence",
            })
        return results, self.mode, {"prefix_tokens": 100, "total_seconds": 0.1}, None


def build(**overrides):
    config = runtime.Config(model="Qwen/Qwen3.5-4B", revision="851bf6e8" + "0" * 32, backend="torch")
    for key, value in overrides.items():
        setattr(config, key, value)
    config.validate()
    engine = StubEngine()
    return TestClient(create_app(config, engine), raise_server_exceptions=False), engine


def post(client, **overrides):
    body = {"state": STATE, "model": "jev-latest", "questions": QUESTIONS}
    body.update(overrides)
    return client.post("/v1/systemone", json=body)


def test_a_full_request_answers_every_question_under_its_own_id():
    client, _ = build()
    response = post(client)
    assert response.status_code == 200
    body = response.json()
    assert set(body["answers"]) == set(QUESTIONS)
    assert body["answers"]["is_urgent"]["type"] == "noul"
    assert body["answers"]["department"]["choice"] == "billing"
    assert body["answers"]["frustration"]["legend"] == {"0": "Calm", "1": "Frustrated", "2": "Very angry"}


def test_the_response_validates_against_the_published_schema():
    client, _ = build()
    jsonschema.validate(post(client).json(), SCHEMA)


def test_a_jev_alias_is_served_but_the_real_model_is_reported():
    client, _ = build()
    body = post(client, model="jev-latest").json()
    assert body["model"] == MODEL_ID
    assert not body["model"].startswith("jev")


def test_jev_aliases_can_be_refused():
    client, _ = build(accept_jev_aliases=False)
    response = post(client, model="jev-latest")
    assert response.status_code == 422
    assert response.json()["error"]["code"] == "unknown_model"


def test_the_servers_own_model_id_and_alias_are_accepted():
    client, _ = build()
    for name in (MODEL_ID, "semif-latest"):
        assert post(client, model=name).status_code == 200


def test_usage_counts_the_shared_state_once_and_reports_no_output_tokens():
    client, _ = build()
    usage = post(client).json()["usage"]
    # 3 questions, 142/143/143 input tokens, 100 of them shared prefix.
    assert usage == {"input_tokens": 100 + 42 + 43 + 43, "output_tokens": 0}


def test_headers_carry_a_request_id_and_the_execution_path():
    client, _ = build()
    response = post(client)
    assert response.headers["x-typesafe-request-id"]
    assert response.headers["x-semif-mode"] == "shared"
    assert response.headers["x-semif-backend"] == "torch"


def test_an_unset_api_key_accepts_anything_and_a_set_one_does_not():
    open_client, _ = build()
    assert open_client.post("/v1/systemone", json={"state": STATE, "model": "jev-latest",
                                                   "questions": QUESTIONS}).status_code == 200
    closed_client, _ = build(api_key="secret")
    assert post(closed_client).status_code == 401
    response = closed_client.post("/v1/systemone", headers={"Authorization": "Bearer secret"},
                                  json={"state": STATE, "model": "jev-latest", "questions": QUESTIONS})
    assert response.status_code == 200
    assert closed_client.get("/v1/models", headers={"Authorization": "Bearer wrong"}).status_code == 401


def test_error_bodies_name_the_offending_field():
    client, _ = build()
    response = post(client, questions={"department": {"type": "choice", "instructions": "Which?",
                                                      "criteria": {f"o{index}": None for index in range(42)}}})
    assert response.status_code == 422
    error = response.json()["error"]
    assert error["code"] == "option_count_unsupported"
    assert error["param"] == "questions.department.criteria"
    assert "42" in error["message"]


def test_malformed_json_is_a_422_not_a_crash():
    client, _ = build()
    response = client.post("/v1/systemone", content=b"{not json",
                           headers={"content-type": "application/json"})
    assert response.status_code == 422
    assert response.json()["error"]["code"] == "invalid_body"


def test_unknown_endpoints_return_the_same_error_shape():
    client, _ = build()
    response = client.get("/v1/nope")
    assert response.status_code == 404
    assert response.json()["error"]["type"] == "not_found_error"


def test_models_lists_the_loaded_model():
    client, _ = build()
    models = client.get("/v1/models").json()["models"]
    assert [model["name"] for model in models] == [MODEL_ID]
    assert all(key in models[0] for key in ("name", "description", "release_date"))


def test_healthz_reports_provenance():
    client, _ = build()
    body = client.get("/healthz").json()
    assert body["status"] == "ready"
    assert body["model"] == MODEL_ID
    assert body["prompt_version"] == "direct-options-v1"


def test_a_request_that_is_not_ready_gets_503_with_retry_after():
    client, engine = build()
    engine.ready = False
    response = post(client)
    assert response.status_code == 503
    assert response.headers["retry-after"] == "2"


def test_translated_rows_produce_the_same_prompt_as_the_equivalent_jsonl_row():
    """The API layer is a translation; this is how that is proven, not asserted.

    prompt_sha256 hashes the chat template applied to `direct_messages`, so
    equal messages give an equal hash under any tokenizer.
    """
    translated = translate({"state": STATE, "model": "jev-latest", "questions": {
        "department": QUESTIONS["department"]}})
    jsonl_row = {
        "id": "department",
        "state": STATE,
        "question": "Which team should handle this?",
        "options": [{"id": "billing", "description": "Payments"},
                    {"id": "technical", "description": "Bugs"},
                    {"id": "sales", "description": "Pricing"}],
    }
    assert direct_messages(translated.rows[0]) == direct_messages(jsonl_row)


def test_every_question_in_one_request_reaches_the_backend_together():
    client, engine = build()
    post(client)
    assert len(engine.seen) == 1
    assert [row["id"] for row in engine.seen[0]] == list(QUESTIONS)


def test_a_rejected_question_stops_the_request_before_the_model_is_touched():
    client, engine = build()
    questions = dict(QUESTIONS)
    questions["broken"] = {"type": "choice", "instructions": "Which?", "criteria": {}}
    assert post(client, questions=questions).status_code == 422
    assert engine.seen == []


# --- Official SDK conformance -------------------------------------------------

typesafe_sdk = pytest.importorskip("typesafe_sdk")


def test_the_official_sdk_validates_our_response_models():
    client, _ = build()
    # Validate the bytes on the wire, the way the SDK's own HTTP path does.
    parsed = typesafe_sdk.SystemOneResponse.model_validate_json(post(client).content)
    assert isinstance(parsed.answers["is_urgent"], typesafe_sdk.NoulAnswer)
    assert isinstance(parsed.answers["department"], typesafe_sdk.ChoiceAnswer)
    assert isinstance(parsed.answers["frustration"], typesafe_sdk.ScoreAnswer)
    assert parsed.model == MODEL_ID
    assert parsed.usage.output_tokens == 0


def test_the_official_sdk_validates_our_models_listing():
    client, _ = build()
    parsed = typesafe_sdk.ListModelsResponse.model_validate_json(client.get("/v1/models").content)
    assert parsed.models[0].name == MODEL_ID


def test_a_real_sdk_client_drives_the_server_in_process():
    """The drop-in claim, exercised: unmodified SDK, only the base URL changed."""
    import httpx2

    config = runtime.Config(model="Qwen/Qwen3.5-4B", revision="851bf6e8" + "0" * 32, backend="torch")
    config.validate()
    app = create_app(config, StubEngine())

    async def run():
        client = typesafe_sdk.AsyncTypeSafeClient(
            api_key="local", base_url="http://semif.test", transport=httpx2.ASGITransport(app))
        async with client:
            return await client.system_one(state=STATE, questions={
                "is_urgent": typesafe_sdk.Noul(instructions="Does this convey urgency?"),
                "department": typesafe_sdk.Choice(instructions="Which team?",
                                                  criteria={"billing": "Payments", "sales": "Pricing"}),
                "frustration": typesafe_sdk.Score(instructions="How frustrated?",
                                                  criteria=["Calm", "Frustrated", "Very angry"]),
            })

    response = asyncio.run(run())
    assert 0.0 <= response.answers["is_urgent"].noul <= 1.0
    assert response.answers["department"].choice == "billing"
    assert 0.0 <= response.answers["department"].confidence <= 1.0
    # The wire carries string keys, as the API reference specifies; the SDK
    # surfaces them as ints after parsing.
    assert response.answers["frustration"].legend[0] == "Calm"
    assert response.model == MODEL_ID


def test_the_sdk_raises_its_typed_error_for_a_rejected_request():
    import httpx2

    config = runtime.Config(model="Qwen/Qwen3.5-4B", revision="851bf6e8" + "0" * 32, backend="torch")
    config.accept_jev_aliases = False
    config.validate()
    app = create_app(config, StubEngine())

    async def run():
        client = typesafe_sdk.AsyncTypeSafeClient(
            api_key="local", base_url="http://semif.test", transport=httpx2.ASGITransport(app))
        async with client:
            await client.system_one(state=STATE, model="jev-latest",
                                    questions={"q": typesafe_sdk.Noul(instructions="Urgent?")})

    with pytest.raises(typesafe_sdk.TypeSafeUnprocessableEntityError) as error:
        asyncio.run(run())
    assert error.value.body["error"]["code"] == "unknown_model"
    assert error.value.request_id
