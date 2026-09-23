import importlib.util
import json
from pathlib import Path
import sys
import threading
from types import ModuleType, SimpleNamespace

import pytest


ROOT = Path(__file__).resolve().parent
SPEC = importlib.util.spec_from_file_location("semif_playground_server", ROOT / "server.py")
server = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = server
SPEC.loader.exec_module(server)


PAYLOAD = {
    "state": {"message": "Please fix this today", "attempts": 3},
    "model": "jev-latest",
    "questions": {
        "urgent": {
            "type": "noul",
            "instructions": "The request is urgent.",
            "criteria": {"true": "Time-sensitive", "false": "Can wait"},
        },
        "queue": {
            "type": "choice",
            "instructions": "Choose the queue.",
            "criteria": {"support": "Needs help", "sales": None},
        },
        "severity": {
            "type": "score",
            "instructions": "Rate severity.",
            "criteria": ["low", "medium", "high"],
        },
    },
}


def test_normalize_and_map_all_question_types():
    state, questions, model = server.normalize_payload(PAYLOAD)
    assert state == PAYLOAD["state"]
    assert model == "jev-latest"
    rows = server.local_rows(state, questions)
    assert [row["id"] for row in rows] == ["urgent", "queue", "severity"]
    assert rows[0]["options"] == [
        {"id": "true", "description": "Time-sensitive"},
        {"id": "false", "description": "Can wait"},
    ]
    assert rows[1]["options"] == [
        {"id": "support", "description": "support: Needs help"},
        {"id": "sales", "description": "sales"},
    ]
    assert [option["id"] for option in rows[2]["options"]] == ["0", "1", "2"]
    assert all(row["state"] is state for row in rows)


@pytest.mark.parametrize(
    "mutation, message",
    [
        (lambda data: data.update(state=""), "state"),
        (lambda data: data.update(questions={}), "questions"),
        (
            lambda data: data["questions"].update(
                bad={"type": "choice", "instructions": "choose", "criteria": {"one": None}}
            ),
            "2 to 16",
        ),
        (
            lambda data: data["questions"].update(
                bad={"type": "score", "instructions": "score", "criteria": ["only one"]}
            ),
            "2 to 10",
        ),
    ],
)
def test_invalid_requests_are_rejected(mutation, message):
    data = json.loads(json.dumps(PAYLOAD))
    mutation(data)
    with pytest.raises(server.RequestError, match=message):
        server.normalize_payload(data)


def test_local_response_adds_explicit_entropy_confidence_for_choice_and_score():
    _, questions, _ = server.normalize_payload(PAYLOAD)
    results = [
        {
            "id": "urgent",
            "option_ids": ["true", "false"],
            "probabilities": [0.8, 0.2],
            "option_logits": [4.0, 2.0],
            "input_tokens": 20,
            "prompt_sha256": "a",
        },
        {
            "id": "queue",
            "option_ids": ["support", "sales"],
            "probabilities": [0.7, 0.3],
            "option_logits": [3.0, 2.0],
            "input_tokens": 21,
            "prompt_sha256": "b",
        },
        {
            "id": "severity",
            "option_ids": ["0", "1", "2"],
            "probabilities": [0.1, 0.2, 0.7],
            "option_logits": [0.0, 1.0, 2.0],
            "input_tokens": 22,
            "prompt_sha256": "c",
        },
    ]
    response = server.adapt_local_response(
        results,
        {"total_seconds": 0.25, "batch_size": 3},
        questions,
        {"source": "fixture", "revision": "abc", "backend": "mlx"},
    )
    assert response["answers"]["urgent"]["noul"] == pytest.approx(0.8)
    assert "confidence" not in response["answers"]["urgent"]
    assert response["answers"]["queue"]["choice"] == "support"
    assert response["answers"]["queue"]["confidence"] == pytest.approx(0.1187091007693073)
    assert response["answers"]["queue"]["confidence_method"] == "normalized_shannon_entropy"
    assert "not calibrated" in response["answers"]["queue"]["confidence_status"]
    assert response["answers"]["severity"]["score"] == pytest.approx(1.6)
    assert response["answers"]["severity"]["confidence"] == pytest.approx(0.2701533008379024)
    assert response["usage"] == {"input_tokens": 63, "output_tokens": 0}
    assert "uncalibrated" in response["answers"]["urgent"]["probability_status"]


def test_client_timing_preserves_provider_timing_and_labels_scope():
    payload = {"model": "jev-test", "timing": {"evaluation_time_ms": 125}}
    result = server._attach_client_timing(payload, 0.375)
    assert result["timing"] == {
        "total_seconds": 0.375,
        "source": "client_wall_clock",
        "includes": "network and SDK overhead",
        "provider": {"evaluation_time_ms": 125},
    }


def test_typesafe_evaluation_reports_observed_elapsed_time(monkeypatch):
    captured = {}

    class Question:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

    class Client:
        def __enter__(self):
            return self

        def __exit__(self, *args):
            return None

        def system_one(self, state, questions, *, model=None):
            captured.update(state=state, questions=questions, model=model)
            return {"model": "jev-fixture", "answers": {}, "usage": {}}

    package = ModuleType("typesafe_sdk")
    package.Choice = Question
    package.Noul = Question
    package.Score = Question
    package.TypeSafeClient = Client
    monkeypatch.setitem(sys.modules, "typesafe_sdk", package)
    monkeypatch.setenv("TYPESAFE_API_KEY", "fixture-secret")
    ticks = iter([50.0, 50.625])
    monkeypatch.setattr(server.time, "perf_counter", lambda: next(ticks))
    service = server.SharedMLXService(server.ServiceConfig())
    try:
        response = service.evaluate_typesafe(PAYLOAD)
    finally:
        service.close()

    assert captured["model"] == "jev-latest"
    assert set(captured["questions"]) == {"urgent", "queue", "severity"}
    assert response["timing"]["total_seconds"] == pytest.approx(0.625)
    assert response["timing"]["source"] == "client_wall_clock"
    assert response["provider"] == "TypeSafe Jev"


def test_bocha_evaluation_reuses_typesafe_sdk_with_isolated_configuration(monkeypatch):
    captured = {}

    class Question:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

    class Client:
        def __init__(self, **kwargs):
            captured["client"] = kwargs

        def __enter__(self):
            return self

        def __exit__(self, *args):
            return None

        def system_one(self, state, questions):
            captured.update(state=state, questions=questions)
            return {"model": "bocha-jev-v1", "answers": {}, "usage": {}}

    package = ModuleType("typesafe_sdk")
    package.Choice = Question
    package.Noul = Question
    package.Score = Question
    package.TypeSafeClient = Client
    monkeypatch.setitem(sys.modules, "typesafe_sdk", package)
    monkeypatch.setenv("TYPESAFE_API_KEY", "official-key-must-not-be-reused")
    ticks = iter([70.0, 70.25])
    monkeypatch.setattr(server.time, "perf_counter", lambda: next(ticks))
    service = server.SharedMLXService(server.ServiceConfig())
    try:
        service.configure_bocha(
            {
                "api_key": "bocha-test-secret",
                "base_url": "https://jev.bocha.cn",
                "model": "bocha-jev-v1",
            }
        )
        response = service.evaluate_bocha(PAYLOAD)
    finally:
        service.close()

    assert captured["client"] == {
        "api_key": "bocha-test-secret",
        "base_url": "https://jev.bocha.cn",
        "model": "bocha-jev-v1",
    }
    assert captured["client"]["api_key"] != "official-key-must-not-be-reused"
    assert set(captured["questions"]) == {"urgent", "queue", "severity"}
    assert response["timing"]["total_seconds"] == pytest.approx(0.25)
    assert response["provider"] == "Jev Bocha"


def test_official_sdk_question_construction_is_typed():
    _, questions, _ = server.normalize_payload(PAYLOAD)

    class Captured:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

    typed = server.typesafe_question_objects(
        questions,
        {"noul": Captured, "choice": Captured, "score": Captured},
    )
    assert typed["urgent"].kwargs["criteria"]["true"] == "Time-sensitive"
    assert typed["queue"].kwargs["criteria"]["sales"] is None
    assert typed["severity"].kwargs["criteria"] == ["low", "medium", "high"]


def test_health_reports_configuration_without_exposing_api_key(monkeypatch):
    monkeypatch.setenv("TYPESAFE_API_KEY", "do-not-return-this-secret")
    monkeypatch.setenv("BOCHA_JEV_API_KEY", "do-not-return-this-bocha-secret")
    service = server.SharedMLXService(server.ServiceConfig(model_source="fixture", model_revision="rev"))
    serialized = json.dumps(service.health())
    assert "do-not-return-this-secret" not in serialized
    assert "do-not-return-this-bocha-secret" not in serialized
    assert service.health()["typesafe"]["key_configured"] is True
    assert service.health()["bocha"] == {
        "sdk_installed": server._typesafe_sdk_available(),
        "key_configured": True,
        "credential_source": "environment",
        "base_url": "https://jev.bocha.cn",
        "model": "bocha-jev-v1",
    }


def test_bocha_setup_is_memory_only_and_returns_sanitized_status(monkeypatch):
    monkeypatch.delenv("BOCHA_JEV_API_KEY", raising=False)
    service = server.SharedMLXService(server.ServiceConfig())
    try:
        assert service.health()["bocha"]["key_configured"] is False
        status = service.configure_bocha(
            {
                "api_key": "bocha-test-secret",
                "base_url": "https://jev.bocha.cn/",
                "model": "bocha-jev-v1-preview",
            }
        )
        assert status == {
            "sdk_installed": server._typesafe_sdk_available(),
            "key_configured": True,
            "credential_source": "setup",
            "base_url": "https://jev.bocha.cn",
            "model": "bocha-jev-v1-preview",
        }
        assert "bocha-test-secret" not in json.dumps(status)
    finally:
        service.close()


@pytest.mark.parametrize(
    "base_url",
    [
        "http://jev.bocha.cn",
        "https://jev.bocha.cn.evil.example",
        "https://jev.bocha.cn@evil.example",
        "https://jev.bocha.cn/v1/systemone",
        "https://jev.bocha.cn?next=https://evil.example",
        "https://jev.bocha.cn:8443",
    ],
)
def test_bocha_setup_rejects_non_allowlisted_base_urls(base_url):
    service = server.SharedMLXService(server.ServiceConfig())
    try:
        with pytest.raises(server.RequestError, match="must be https://jev.bocha.cn"):
            service.configure_bocha(
                {"api_key": "bocha-test-secret", "base_url": base_url, "model": "bocha-jev-v1"}
            )
    finally:
        service.close()


def test_bocha_evaluation_requires_configuration(monkeypatch):
    monkeypatch.delenv("BOCHA_JEV_API_KEY", raising=False)
    service = server.SharedMLXService(server.ServiceConfig())
    try:
        with pytest.raises(server.RequestError, match="not configured"):
            service.evaluate_bocha(PAYLOAD)
    finally:
        service.close()


def test_builtin_model_catalog_exposes_pinned_local_model_choices():
    health = server.SharedMLXService(server.ServiceConfig()).health()["local_model"]
    models = {model["id"]: model for model in health["available_models"]}
    assert health["default_model"] == "qwen3.5-4b"
    assert models["qwen3.5-4b"]["source"] == "Qwen/Qwen3.5-4B"
    assert models["qwen3.5-9b"]["source"] == "Qwen/Qwen3.5-9B"
    assert models["qwen3.8-27b-4bit"] == {
        "id": "qwen3.8-27b-4bit",
        "label": "Qwen3.8 27B · 4-bit",
        "short_label": "Qwen3.8 27B",
        "source": "mlx-community/Qwen3.8-27B-4bit",
        "revision": "10c35caafbb80f7dc6a7a432cdd11af10a6d4818",
        "download_gb": 16.1,
        "source_quantization_bits": 4,
    }
    assert models["muse-glimmer-30b-4bit"] == {
        "id": "muse-glimmer-30b-4bit",
        "label": "Muse Glimmer 30B · 4-bit",
        "short_label": "Muse Glimmer 30B",
        "source": "mlx-community/Muse-Glimmer-30B-4bit",
        "revision": "3e7677d7a40d348a3daba263a2b1c0aa41910710",
        "download_gb": 19.4,
        "source_quantization_bits": 4,
    }
    assert models["gemma-4-12b-it-4bit"]["source"] == "mlx-community/gemma-4-12B-it-4bit"
    assert models["gemma-4-12b-it-4bit"]["source_quantization_bits"] == 4
    assert models["gemma-4-26b-a4b-it-4bit"] == {
        "id": "gemma-4-26b-a4b-it-4bit",
        "label": "Gemma 4 26B A4B IT · 4-bit",
        "short_label": "Gemma 26B A4B",
        "source": "mlx-community/gemma-4-26b-a4b-it-4bit",
        "revision": "0d77464eeb233a2da68ebf9d7dc4edaac7db956d",
        "download_gb": 15.4,
        "source_quantization_bits": 4,
    }
    assert models["gemma-4-31b-it-4bit"] == {
        "id": "gemma-4-31b-it-4bit",
        "label": "Gemma 4 31B IT · 4-bit",
        "short_label": "Gemma 31B",
        "source": "mlx-community/gemma-4-31b-it-4bit",
        "revision": "696d436c404745a59f30e4939a658162b0a9e57f",
        "download_gb": 18.4,
        "source_quantization_bits": 4,
    }
    assert all(len(model["revision"]) == 40 for model in models.values())


def test_backend_allows_native_large_model_types():
    from semif_phase1.mlx_backend import SUPPORTED_MODEL_TYPES

    assert {"gemma4", "gemma4_unified", "muse_glimmer"} <= SUPPORTED_MODEL_TYPES


def test_shared_service_switches_models_and_keeps_only_selected_instance(monkeypatch):
    loads = []

    def load_model(source, revision, bits, *, cache_limit_mib):
        model = object()
        loads.append((source, revision, bits, model))
        return model, object(), {"source": source, "revision": revision, "backend": "fixture"}

    package = ModuleType("semif_phase1")
    package.mlx_backend = SimpleNamespace(load_model=load_model)
    monkeypatch.setitem(sys.modules, "semif_phase1", package)
    service = server.SharedMLXService(server.ServiceConfig(mlx_bits=8))

    service.load("qwen3.5-4b")
    first_model = service._model
    service.load("qwen3.5-4b")
    service.load("qwen3.5-9b")
    service.load("qwen3.8-27b-4bit")
    service.load("muse-glimmer-30b-4bit")
    service.load("gemma-4-12b-it-4bit")
    service.load("gemma-4-26b-a4b-it-4bit")
    service.load("gemma-4-31b-it-4bit")

    assert [source for source, _, _, _ in loads] == [
        "Qwen/Qwen3.5-4B",
        "Qwen/Qwen3.5-9B",
        "mlx-community/Qwen3.8-27B-4bit",
        "mlx-community/Muse-Glimmer-30B-4bit",
        "mlx-community/gemma-4-12B-it-4bit",
        "mlx-community/gemma-4-26b-a4b-it-4bit",
        "mlx-community/gemma-4-31b-it-4bit",
    ]
    assert [bits for _, _, bits, _ in loads] == [8, 8, None, None, None, None, None]
    assert service._model is loads[-1][3]
    assert service._model is not first_model
    assert service.health()["local_model"]["active_model"] == "gemma-4-31b-it-4bit"
    service.close()


def test_unknown_local_model_is_rejected_before_loading():
    service = server.SharedMLXService(server.ServiceConfig())
    try:
        with pytest.raises(server.RequestError, match="Unknown local model"):
            service.load("not-a-model")
    finally:
        service.close()


def test_shared_service_loads_one_model_and_reuses_it(monkeypatch):
    calls = {"loads": 0, "scores": 0}
    model_threads = []

    def load_model(source, revision, bits, *, cache_limit_mib):
        calls["loads"] += 1
        model_threads.append(threading.get_ident())
        return object(), object(), {"source": source, "revision": revision, "backend": "mlx"}

    def score_shared(model, tokenizer, rows, metadata):
        calls["scores"] += 1
        model_threads.append(threading.get_ident())
        results = []
        for row in rows:
            results.append(
                {
                    "id": row["id"],
                    "option_ids": [option["id"] for option in row["options"]],
                    "probabilities": [0.75, 0.25],
                    "option_logits": [2.0, 1.0],
                    "input_tokens": 10,
                }
            )
        return results, {"total_seconds": 0.01, "batch_size": len(rows)}

    package = ModuleType("semif_phase1")
    package.mlx_backend = SimpleNamespace(load_model=load_model, score_shared=score_shared)
    monkeypatch.setitem(sys.modules, "semif_phase1", package)
    service = server.SharedMLXService(server.ServiceConfig(model_source="fixture", model_revision="rev"))
    service.load()
    service.load()
    payload = {
        "state": "one shared state",
        "questions": {"decision": {"type": "noul", "instructions": "Is this true?"}},
    }
    first = service.evaluate_local(payload)
    second = service.evaluate_local(payload)
    assert calls == {"loads": 1, "scores": 2}
    assert len(set(model_threads)) == 1
    assert model_threads[0] != threading.get_ident()
    assert first["answers"]["decision"]["noul"] == pytest.approx(0.75)
    assert second["model"] == "fixture"
    service.close()


def test_every_checked_in_template_is_valid_and_mappable():
    index = json.loads((ROOT / "templates" / "index.json").read_text())
    assert len(index["templates"]) >= 3
    seen = set()
    for entry in index["templates"]:
        template = json.loads((ROOT / "templates" / entry["file"]).read_text())
        assert template["id"] not in seen
        seen.add(template["id"])
        assert template["name"] and template["category"] and template["description"]
        state, questions, model = server.normalize_payload(
            {"state": template["state"], "questions": template["questions"], "model": "jev-latest"}
        )
        assert model == "jev-latest"
        assert len(server.local_rows(state, questions)) == len(template["questions"])


def test_static_page_has_comparison_raw_template_and_secret_boundaries():
    html = (ROOT / "index.html").read_text()
    script = (ROOT / "app.js").read_text()
    json_workbench = (ROOT / "json-workbench.js").read_text()
    styles = (ROOT / "style.css").read_text()
    readme = (ROOT / "README.md").read_text()
    for marker in (
        'id="state-editor"',
        'id="questions-editor"',
        'id="comparison-body"',
        'id="response-alert"',
        'id="response-alert-title"',
        'id="response-alert-actions"',
        'id="view-rendered"',
        'id="raw-json"',
        'id="raw-request"',
        'id="raw-curl"',
        'id="raw-request-tab"',
        'id="raw-response-tab"',
        'id="raw-tree-actions"',
        'id="raw-expand-all"',
        'id="raw-collapse-all"',
        'id="state-editor-shell"',
        'id="questions-editor-shell"',
        'id="workbench-resizer"',
        'id="editor-resizer"',
        'id="template-studio"',
        'id="template-trigger"',
        'id="template-popover"',
        'id="template-listbox"',
        'id="challenger-local"',
        'id="challenger-bocha"',
        'id="local-model-select"',
        'id="local-model-size"',
        'id="bocha-api-key"',
        'id="bocha-model"',
        'id="bocha-connection-label"',
        'class="drawer-body"',
    ):
        assert marker in html
    assert 'id="bocha-api-key" type="password"' in html
    assert 'href="style.css?v=20260922-3"' in html
    assert 'src="app.js?v=20260922-3"' in html
    assert 'id="template-toggle" type="button" title="Open template studio"' in html
    assert ">\n          Templates\n        </button>" in html
    assert 'id="template-studio"' in html
    assert 'class="service-chip-label"' in html
    assert "localStorage.setItem" in script
    assert all("bocha" not in line.lower() for line in script.splitlines() if "localStorage.setItem" in line)
    assert "local_model" in script
    assert 'elements.loadModel.setAttribute("aria-busy", local.status === "loading" ? "true" : "false")' in script
    assert 'elements.loadModel.setAttribute("aria-busy", "true")' in script
    assert 'button:disabled[aria-busy="true"]' in styles
    assert "button:disabled {\n  cursor: default;" in styles
    assert all(route in script for route in ("/api/evaluate/local", "/api/evaluate/bocha", "/api/evaluate/typesafe"))
    assert "/api/config/bocha" in script
    assert "function openSettings" in script
    assert "inset: 0 0 0 auto" in styles
    assert "width: min(560px, 100vw)" in styles
    assert "max-width: none" in styles
    assert ".connection-summary" in styles
    assert "Entropy confidence" in script
    assert "timing.total_seconds" in script
    assert "function providerErrorPresentation" in script
    assert "Input is too long for Jev Bocha" in script
    assert 'if (result.status === "error") return { state: "unavailable" };' in script
    assert 'createElement("p", "dimension-instructions", instructionsText(question.instructions))' in script
    assert "Question details" not in script
    assert ".service-strip .icon-button" in styles
    assert "white-space: nowrap" in styles
    assert ".comparison-table th:first-child {\n  width: 22%;" in styles
    assert ".comparison-table th:nth-child(2),\n.comparison-table th:nth-child(3) {\n  width: 29%;" in styles
    assert ".comparison-table th:last-child {\n  width: 20%;" in styles
    assert ".dimension-detail" not in styles
    assert "--workbench-header-height: 52px" in styles
    assert "height: var(--workbench-header-height)" in styles
    assert "--splitter-size: 5px" in styles
    assert "--editor-splitter-size: 5px" in styles
    assert "--editor-pane-min: 150px" in styles
    assert "--state-pane-height: 1fr" in styles
    assert "--input-pane-width: 46.5%" in styles
    assert "--input-pane-min: 460px" in styles
    assert "--response-pane-min: 600px" in styles
    assert "minmax(var(--input-pane-min), var(--input-pane-width))" in styles
    assert ".workbench-resizer" in styles
    assert ".editor-resizer" in styles
    assert "display: none" in styles
    assert 'role="separator"' in html
    assert 'aria-orientation="vertical"' in html
    assert 'aria-orientation="horizontal"' in html
    assert 'data-json-target="state" data-json-action="unfold"' in html
    assert 'data-json-target="state" data-json-action="fold"' in html
    assert 'data-json-target="questions" data-json-action="unfold"' in html
    assert 'data-json-target="questions" data-json-action="fold"' in html
    assert 'window.matchMedia("(max-width: 900px)")' in script
    assert 'event.shiftKey ? 48 : 16' in script
    assert 'resizer.addEventListener("dblclick", reset)' in script
    assert 'resizer.addEventListener("pointerdown"' in script
    assert 'resizer.addEventListener("keydown"' in script
    assert "function initializePaneResizer" in script
    assert "function initializePaneResizers" in script
    assert 'property: "--state-pane-height"' in script
    assert 'axis: "y"' in script
    assert "surface.controller.unfoldAll()" in script
    assert "surface.controller.foldAll()" in script
    assert "function setServiceStatus" in script
    assert 'button.setAttribute("aria-label"' in script
    assert "height: calc(100dvh - var(--topbar-height))" in styles
    assert "max-height: 100dvh" in styles
    assert "grid-template-rows: auto minmax(0, 1fr)" in styles
    assert "overflow-y: auto" in styles
    assert 'from "./json-workbench.js"' in script
    assert 'data-json-mode="tree"' not in html
    assert 'data-json-mode="edit"' not in html
    assert ".json-editor-shell" in styles
    assert ".json-editor-shell > .CodeMirror" in styles
    assert "linear-gradient(to right, #f0f1ee" not in styles
    assert "createJsonEditor" in script
    assert "CodeMirror.fromTextArea" in json_workbench
    assert 'foldGutter: true' in json_workbench
    assert 'fixedGutter: true' in json_workbench
    assert 'lineWrapping: true' in json_workbench
    assert 'change.origin === "paste"' in json_workbench
    for controller_method in ("getValue", "setValue", "lineCount", "focus", "refresh", "foldAll", "unfoldAll", "setInvalid", "onChange", "formatAfterPaste"):
        assert controller_method in json_workbench
    assert 'class="slash-mark brand-mark"' in html
    assert 'class="slash-mark empty-mark"' in html
    for asset in (
        "vendor/codemirror-5.65.21/lib/codemirror.css",
        "vendor/codemirror-5.65.21/addon/fold/foldgutter.css",
        "vendor/codemirror-5.65.21/lib/codemirror.js",
        "vendor/codemirror-5.65.21/mode/javascript/javascript.js",
        "vendor/codemirror-5.65.21/addon/edit/matchbrackets.js",
        "vendor/codemirror-5.65.21/addon/fold/foldcode.js",
        "vendor/codemirror-5.65.21/addon/fold/brace-fold.js",
        "vendor/codemirror-5.65.21/addon/fold/foldgutter.js",
    ):
        assert asset in html
    assert "cdn.jsdelivr.net" not in html
    assert "cdnjs.cloudflare.com" not in html
    assert "formatValidJson" in json_workbench
    assert 'document.createElement("details")' not in json_workbench
    assert 'node("details", "json-branch")' in json_workbench
    assert "textContent" in json_workbench
    assert "innerHTML" not in json_workbench
    assert "_materializeJsonChildren" in json_workbench
    for token_class in (".json-key", ".json-string", ".json-number", ".json-boolean", ".json-null"):
        assert token_class in styles
    assert "rawJsonTree.expandAll()" in script
    assert "rawJsonTree.collapseAll()" in script
    assert "function responseOnlyData()" in script
    assert "rawJsonTree.render(responseOnlyData())" in script
    assert "navigator.clipboard.writeText(content)" in script
    assert "navigator.clipboard.writeText(pretty(app.response))" not in script
    assert 'data-raw-mode="request"' in html
    assert 'data-raw-mode="response"' in html
    assert 'role="listbox"' in html
    assert 'aria-haspopup="listbox"' in html
    assert 'function shellSingleQuote(value)' in script
    assert '"<BOCHA_JEV_API_KEY>"' in script
    assert '"<TYPESAFE_API_KEY>"' in script
    assert '"https://jev.bocha.cn/v1/systemone"' in script
    assert '"https://api.typesafe.ai/v1/systemone"' in script
    assert '"/api/evaluate/local"' in script
    assert "providerRequestBody(request, \"jev\")" in script
    assert "local_model" not in script[script.index("function providerRequestBody"):script.index("function curlCommand")]
    assert "position: sticky" in styles
    assert ".use-case-popover" in styles
    assert ".use-case-option.selected" in styles
    assert ".raw-context-label" in styles
    assert ".raw-curl" in styles
    assert "cannot share model memory" in readme
    assert "never written to browser storage" in readme
    assert "equal half-and-half split" in readme
    assert "bounded to the browser viewport" in readme


def test_vendored_codemirror_is_pinned_and_checksum_complete():
    vendor = ROOT / "vendor" / "codemirror-5.65.21"
    package = json.loads((vendor / "package.json").read_text())
    assert package["name"] == "codemirror"
    assert package["version"] == "5.65.21"
    assert (vendor / "LICENSE").is_file()

    checksum_lines = (vendor / "SHA256SUMS").read_text().splitlines()
    recorded = {line.split("  ", 1)[1] for line in checksum_lines if "  " in line}
    assert recorded == {
        "LICENSE",
        "package.json",
        "lib/codemirror.js",
        "lib/codemirror.css",
        "mode/javascript/javascript.js",
        "addon/edit/matchbrackets.js",
        "addon/fold/foldcode.js",
        "addon/fold/foldgutter.js",
        "addon/fold/brace-fold.js",
        "addon/fold/foldgutter.css",
    }
