"""The pluggable prompt: the default renders the published words, other wordings keep the shape."""

import json
import subprocess
import sys

import pytest

from semif_phase1.core import DEFAULT_PROMPT, DIRECT_SYSTEM, PROMPTS, Prompt, direct_messages, resolve_prompt
from semif_phase1.direct import PROMPT_VERSION
from semif_phase1.serial import _state_prefix
from test_serial import Tokenizer

ROW = {
    "id": "x",
    "state": "owned evidence",
    "question": "Which answer follows?",
    "options": [{"id": "yes", "description": "Yes."}, {"id": "no", "description": "No."}],
}


def test_default_prompt_is_the_published_wording_byte_for_byte():
    payload = {"evidence": "owned evidence", "criterion": "Which answer follows?",
               "options": [{"letter": "A", "description": "Yes."}, {"letter": "B", "description": "No."}]}
    assert direct_messages(ROW) == [
        {"role": "system", "content": DIRECT_SYSTEM},
        {"role": "user", "content": json.dumps(payload, ensure_ascii=False)},
    ]
    assert PROMPT_VERSION == DEFAULT_PROMPT.version == "direct-options-v1"
    assert resolve_prompt(None) is resolve_prompt("en") is DEFAULT_PROMPT


def test_french_prompt_keeps_the_shape_and_changes_the_words():
    fr = resolve_prompt("fr")
    messages = fr.messages(ROW)
    payload = json.loads(messages[1]["content"])
    assert list(payload) == ["elements", "critere", "options"]
    assert payload["elements"] == "owned evidence"
    assert payload["options"][0] == {"lettre": "A", "description": "Yes."}
    assert messages[0]["content"].startswith("Applique")
    assert fr.version != DEFAULT_PROMPT.version
    assert len({prompt.version for prompt in PROMPTS.values()}) == len(PROMPTS)


def test_state_prefix_follows_the_prompt_and_still_stops_before_the_question():
    for name in PROMPTS:
        prompt = PROMPTS[name]
        prefix = bytes(_state_prefix(Tokenizer(), "owned state", prompt)).decode()
        assert prefix.startswith("HEADER\n{\"" + prompt.evidence_key + "\": \"owned state")
        assert "prefix boundary placeholder" not in prefix
        assert prompt.options_key not in prefix
    assert _state_prefix(Tokenizer(), "owned state") == _state_prefix(Tokenizer(), "owned state", "en")


def test_prompt_file_round_trip(tmp_path):
    path = tmp_path / "prompt.json"
    path.write_text(json.dumps({"version": "direct-options-de-test", "system": "Wende das Kriterium an.",
                                "keys": {"evidence": "beleg", "letter": "buchstabe"}}))
    prompt = resolve_prompt(str(path))
    assert prompt == Prompt("direct-options-de-test", "Wende das Kriterium an.", evidence_key="beleg", letter_key="buchstabe")
    assert prompt.evidence_text({"a": 1}) == '{"beleg": {"a": 1}'


@pytest.mark.parametrize("data,message", [
    ({"system": "no version"}, "needs 'version' and 'system'"),
    ({"version": "v", "system": "s", "keys": {"state": "x"}}, "may only rename"),
    ({"version": "v", "system": "s", "keys": {"evidence": "same", "criterion": "same"}}, "distinct"),
    ({"version": "direct-options-v1", "system": "other words"}, "belongs to a built-in prompt"),
    ({"version": " ", "system": "s"}, "nonempty version"),
])
def test_invalid_prompt_files_are_rejected(tmp_path, data, message):
    path = tmp_path / "prompt.json"
    path.write_text(json.dumps(data))
    with pytest.raises(ValueError, match=message):
        resolve_prompt(str(path))


def test_unknown_prompt_name_is_rejected():
    with pytest.raises(ValueError, match="Unknown prompt"):
        resolve_prompt("klingon")


def test_scoring_modules_import_without_torch():
    """The llama.cpp path needs only transformers for its tokenizer; torch stays a lazy import."""
    code = ("import sys; sys.modules['torch'] = None; "
            "import semif_phase1.cli, semif_phase1.core, semif_phase1.direct, semif_phase1.serial, "
            "semif_phase1.shared, semif_phase1.llamacpp_backend; print('ok')")
    completed = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
    assert completed.returncode == 0, completed.stderr
    assert completed.stdout.strip() == "ok"
