"""Server routing tests with a fake scorer (no model weights needed)."""

import json
import threading
import urllib.error
import urllib.request
from http.server import ThreadingHTTPServer

from semif_phase1.server import ServerState, make_handler


def _fake_result(row):
    count = len(row["options"])
    if count == 2:
        probabilities = [0.7, 0.3]
    elif count == 3:
        probabilities = [0.6, 0.3, 0.1]
    else:
        probabilities = [0.6] + [(0.4 / (count - 1))] * (count - 1)
    return {
        "id": row["id"],
        "option_ids": [o["id"] for o in row["options"]],
        "probabilities": probabilities,
        "input_tokens": 10,
    }


def _fake_state(mode="direct"):
    state = ServerState.__new__(ServerState)
    state.mode = mode
    state.metadata = {"source": "fake", "revision": "test"}
    state.max_tokens = 4096
    state.lock = threading.Lock()
    state._serial = None
    state._shared = None
    state._direct = lambda model, tokenizer, row, metadata, max_tokens: _fake_result(row)
    state.model = state.tokenizer = None
    if mode == "shared":
        state._direct = None
        state._shared = lambda model, tokenizer, rows, metadata, max_tokens: (
            [_fake_result(row) for row in rows],
            {"batch_size": len(rows)},
        )
    return state


def _live_server(state):
    server = ThreadingHTTPServer(("127.0.0.1", 0), make_handler(state))
    server.daemon_threads = True
    threading.Thread(target=server.serve_forever, daemon=True).start()
    return server


def _post(server, path, payload):
    request = urllib.request.Request(
        f"http://127.0.0.1:{server.server_port}{path}",
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(request) as response:
        return response.status, json.loads(response.read())


def _post_raw(server, path, payload):
    request = urllib.request.Request(
        f"http://127.0.0.1:{server.server_port}{path}",
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(request) as response:
            return response.status, json.loads(response.read())
    except urllib.error.HTTPError as error:
        return error.code, json.loads(error.read())


def _get(server, path):
    with urllib.request.urlopen(f"http://127.0.0.1:{server.server_port}{path}") as response:
        return response.status, json.loads(response.read())


ROW = {
    "id": "route-1",
    "state": "Customer cannot access an account after a password reset.",
    "question": "Which queue should handle this request?",
    "options": [
        {"id": "access", "description": "Account access support."},
        {"id": "billing", "description": "Billing support."},
    ],
}

SYSTEMONE = {
    "state": "Our API integration started returning 500 errors 20 minutes ago.",
    "model": "semif-local",
    "questions": {
        "department": {
            "type": "choice",
            "instructions": "Which team should handle this?",
            "criteria": {
                "billing": "Payment or subscription issues",
                "technical": "Bugs or integration problems",
                "sales": "Pricing or account questions",
            },
        },
        "frustration": {
            "type": "score",
            "instructions": "How frustrated the customer appears",
            "criteria": ["Calm, just stating facts", "Frustrated but civil", "Very angry, strong language"],
        },
        "is_urgent": {"type": "noul", "instructions": "The message conveys urgency or time-sensitivity"},
    },
}


def test_score_and_healthz_round_trip():
    server = _live_server(_fake_state())
    try:
        assert _get(server, "/healthz")[0] == 200
        status, body = _post(server, "/score", ROW)
        assert status == 200
        assert body["id"] == "route-1"
        status, body = _post(server, "/score-batch", {"rows": [ROW, dict(ROW, id="route-2")]})
        assert status == 200
        assert [r["id"] for r in body["results"]] == ["route-1", "route-2"]
    finally:
        server.shutdown()


def test_invalid_row_returns_400_not_500():
    server = _live_server(_fake_state())
    try:
        status, body = _post_raw(server, "/score", {"id": "bad"})
        assert status == 400
        assert "error" in body
    finally:
        server.shutdown()


def test_shared_mode_batch_uses_single_prefill():
    server = _live_server(_fake_state("shared"))
    try:
        rows = [ROW, dict(ROW, id="route-2")]
        status, body = _post(server, "/score-batch", {"rows": rows})
        assert status == 200
        assert body["shared_timing"] == {"batch_size": 2}
    finally:
        server.shutdown()


def test_systemone_choice_score_noul_round_trip():
    server = _live_server(_fake_state())
    try:
        status, body = _post(server, "/v1/systemone", SYSTEMONE)
        assert status == 200
        assert body["model"] == "fake"
        department = body["answers"]["department"]
        assert department["type"] == "choice"
        assert department["choice"] == "billing"
        assert department["probabilities"] == {"billing": 0.6, "technical": 0.3, "sales": 0.1}
        assert department["confidence"] == abs(3 * 0.6 - 1) / 2
        frustration = body["answers"]["frustration"]
        assert frustration["type"] == "score"
        assert frustration["score"] == abs(0 * 0.6 + 1 * 0.3 + 2 * 0.1)
        assert frustration["legend"] == {"0": "Calm, just stating facts", "1": "Frustrated but civil", "2": "Very angry, strong language"}
        assert frustration["probabilities"] == {"0": 0.6, "1": 0.3, "2": 0.1}
        urgent = body["answers"]["is_urgent"]
        assert urgent == {"type": "noul", "noul": 0.7}
        assert body["usage"] == {"input_tokens": 30, "output_tokens": 0}
    finally:
        server.shutdown()


def test_systemone_shared_mode_serves_one_prefill():
    server = _live_server(_fake_state("shared"))
    try:
        status, body = _post(server, "/v1/systemone", SYSTEMONE)
        assert status == 200
        assert set(body["answers"]) == {"department", "frustration", "is_urgent"}
    finally:
        server.shutdown()


def test_systemone_validation_returns_422():
    server = _live_server(_fake_state())
    try:
        status, body = _post_raw(server, "/v1/systemone", {"state": "x", "model": "m"})
        assert status == 422
        assert "detail" in body
        status, body = _post_raw(
            server,
            "/v1/systemone",
            {"state": "x", "model": "m", "questions": {"q": {"type": "bogus", "instructions": "y"}}},
        )
        assert status == 422
    finally:
        server.shutdown()


def test_models_list_reports_local_checkpoint():
    server = _live_server(_fake_state())
    try:
        status, body = _get(server, "/v1/models")
        assert status == 200
        assert body["models"][0]["name"] == "fake"
    finally:
        server.shutdown()
