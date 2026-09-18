import re
from pathlib import Path

WEBGPU = Path(__file__).resolve().parent

def sources():
    return {name: (WEBGPU / name).read_text() for name in ("index.html", "app.js", "worker.js", "README.md", "_headers")}

def test_static_runtime_and_pins():
    text = sources()
    assert 'src="app.js"' in text["index.html"]
    assert 'href="style.css"' in text["index.html"]
    assert 'new Worker("worker.js", { type: "module" })' in text["app.js"]
    assert 'import("./vendor/wllama/index.js")' in text["worker.js"]
    assert (WEBGPU / "vendor/wllama/wasm/wllama.wasm").stat().st_size > 1_000_000
    assert (WEBGPU / "vendor/wllama/LICENCE").is_file()
    assert "wllama" in text["README.md"] and "3.6.1" in text["README.md"]
    assert "vue@3.5.21" in text["app.js"]
    assert "Material+Symbols+Rounded" in text["index.html"]
    assert "Cross-Origin-Opener-Policy: same-origin" in text["_headers"]
    assert "Cross-Origin-Embedder-Policy: require-corp" in text["_headers"]

def test_three_pinned_model_tiers():
    text = sources()
    for model_id in ("qwen3-0.6b", "minicpm5-2b", "qwen3.5-4b"):
        assert model_id in text["index.html"]
        assert model_id in text["app.js"]
        assert model_id in text["worker.js"]
    for revision in (
        "23749fefcc72300e3a2ad315e1317431b06b590a",
        "2079a22f3beaa4e306449978533478fe0522f4b3",
        "4168f45a16a1290d65a4ec0fa312ae917a4c15d6",
    ):
        assert revision in text["worker.js"]
        assert revision in text["README.md"]
    assert 'modelSelect.value = "minicpm5-2b"' in text["app.js"]
    assert "Phone or small device detected" in text["app.js"]
    assert "Switch to Qwen3 0.6B in the model box" in text["app.js"]
    assert "recommended for phones and small devices" in text["index.html"]
    assert "Smaller model optimized for small devices. Accuracy may be worse." in text["app.js"]
    assert "may not fit on some low-end devices" in text["app.js"]
    assert 'id="model-notice"' in text["index.html"]
    assert 'id="quality-title"' in text["index.html"]
    assert 'owned + public benchmarks' in text["index.html"]
    assert "Published Jev" in text["index.html"]
    for score in ("44.0%", "52.8%", "40.7%", "68.6%", "69.3%", "63.7%", "81.3%", "76.6%", "84.5%", "88.3%"):
        assert score in text["index.html"]
    assert 'row.dataset.qualityModel === modelSelect.value' in text["app.js"]

def test_live_comparison_and_limits():
    text = sources()
    combined = "\n".join(text.values())
    assert "performance.now()" in combined
    assert "createChatCompletion" in text["worker.js"]
    assert "logprobs: true" in text["worker.js"]
    assert "top_logprobs: 20" in text["worker.js"]
    assert "grammar" in text["worker.js"]
    assert "stream: true" in text["worker.js"]
    assert "JSON.parse" in text["worker.js"]
    assert "<think>" in text["worker.js"]
    assert "probabilities must sum to 1" in text["worker.js"]
    assert "Route north" in text["worker.js"]
    assert "const MAX_OPTIONS = 20" in text["app.js"]
    assert "const MIN_OPTIONS = 2" in text["app.js"]
    for phrase in ("conditional probabilities", "not calibrated", "sequential", "warmup", "no backend"):
        assert phrase in combined.lower()
    assert "WebSocket" not in combined
    assert not re.search(r"/api/(?:generate|score)", combined)

def test_identity_notice_and_ui_mode_switch():
    text = sources()
    assert ">SemIf (formerly OpenJev)<" in text["index.html"]
    assert "Independent research project" in text["index.html"]
    assert "Not affiliated with or endorsed by TypeSafe" in text["index.html"]
    assert "Formerly called OpenJev" in text["index.html"]
    assert "No infringement is intended" in text["index.html"]
    assert 'id="ui-mode"' in text["index.html"]
    assert "Unsloppify site" in text["index.html"]
    assert 'document.body.classList.toggle("plain-ui", plain)' in text["app.js"]
    assert "semif-ui-mode" in text["app.js"]
    assert "body.plain-ui" in (WEBGPU / "style.css").read_text()
