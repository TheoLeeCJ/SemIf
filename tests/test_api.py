import pytest

pytest.importorskip("fastapi")
pytest.importorskip("httpx")

from fastapi.testclient import TestClient

from semif_phase1.api import create_app
from semif_phase1.system_one import SystemOneService


def service():
    def score(rows):
        return [{
            "id": row["id"],
            "option_ids": [option["id"] for option in row["options"]],
            "probabilities": [0.9, 0.1],
            "input_tokens": 12,
            "prompt_version": "test-v1",
        } for row in rows]

    return SystemOneService(score, "semif-test", "Test model", "2026-09-20")


def request():
    return {
        "model": "semif-test",
        "state": "A deployment failed.",
        "questions": {"failed": {"type": "noul", "instructions": "Did it fail?"}},
    }


def test_public_endpoints_and_openapi_schema():
    client = TestClient(create_app(service()))
    assert client.get("/healthz").json() == {"status": "ready", "model": "semif-test"}
    assert client.get("/v1/models").json()["models"][0]["name"] == "semif-test"
    response = client.post("/v1/systemone", json=request())
    assert response.status_code == 200
    assert response.json()["answers"]["failed"] == {"type": "noul", "noul": 0.9}
    paths = client.get("/openapi.json").json()["paths"]
    assert "/v1/systemone" in paths
    assert "/v1/models" in paths


def test_bearer_authentication():
    client = TestClient(create_app(service(), api_key="secret-token"))
    assert client.get("/v1/models").status_code == 401
    assert client.post("/v1/systemone", json=request(), headers={
        "Authorization": "Bearer wrong"
    }).status_code == 401
    response = client.post("/v1/systemone", json=request(), headers={
        "Authorization": "Bearer secret-token"
    })
    assert response.status_code == 200


def test_validation_errors_use_422():
    client = TestClient(create_app(service()))
    invalid = request()
    invalid["model"] = "jev-latest"
    response = client.post("/v1/systemone", json=invalid)
    assert response.status_code == 422
    assert "does not serve Jev aliases" in response.json()["detail"]


def test_scorer_input_errors_use_422():
    def reject(_rows):
        raise ValueError("input exceeds the configured token limit")

    failing = SystemOneService(reject, "semif-test", "Test model", "2026-09-20")
    response = TestClient(create_app(failing)).post("/v1/systemone", json=request())
    assert response.status_code == 422
    assert "token limit" in response.json()["detail"]
