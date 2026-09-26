"""Shared state prefixes must be true token prefixes of every full prompt."""
import pytest

from semif_phase1.core import direct_messages
from semif_phase1.serial import _state_prefix as serial_prefix
from semif_phase1.shared import _state_prefix as shared_prefix

PREFIXES = (serial_prefix, shared_prefix)

# The pinned Qwen tokenizer lets a token span the comma that follows the evidence,
# absorbing the two characters in front of it. Whether that happens depends on the
# state text, which is the whole point: a caller cannot predict it. These stubs
# reproduce both extremes byte for byte, so the regression runs with no torch, no
# network and no multi-gigabyte download.
FUSED_TOKEN = 0x110000


class PlainTokenizer:
    """Byte tokenizer that never merges, so the boundary costs exactly one token."""

    def apply_chat_template(self, turns, tokenize=False, add_generation_prompt=True, enable_thinking=False):
        return "HEADER\n" + turns[-1]["content"] + "\nASSISTANT"

    def encode(self, text, add_special_tokens=False):
        return list(text.encode())

    def decode(self, tokens):
        return bytes(token for token in tokens if token != FUSED_TOKEN).decode()


class MergingTokenizer(PlainTokenizer):
    """Byte tokenizer whose tokens straddle the comma after the evidence."""

    def encode(self, text, add_special_tokens=False):
        raw = list(text.encode())
        tokens, index = [], 0
        while index < len(raw):
            if raw[index + 1 : index + 3] in ([ord("}"), ord(",")], [ord('"'), ord(",")]):
                tokens.append(FUSED_TOKEN)
                index += 3
            else:
                tokens.append(raw[index])
                index += 1
        return tokens


TOKENIZERS = (PlainTokenizer, MergingTokenizer)
STATES = [
    "Did the deploy use the same parameters?",
    {"a": "why)"},
    {"a": {"b": "z-"}, "note": "keep"},
    ["y{"],
    "owned state",
    {"priority": "urgent"},
]


def _full_prompt(tokenizer, state):
    row = {
        "id": "x",
        "state": state,
        "question": "Is this urgent?",
        "options": [{"id": "a", "description": "A"}, {"id": "b", "description": "B"}],
    }
    payload = direct_messages(row)[-1]["content"]
    return tokenizer.encode("HEADER\n" + payload + "\nASSISTANT")


@pytest.mark.parametrize("tokenizer", TOKENIZERS, ids=lambda token: token.__name__)
@pytest.mark.parametrize("state", STATES, ids=repr)
@pytest.mark.parametrize("prefix_of", PREFIXES, ids=lambda function: function.__module__)
def test_state_prefix_is_a_token_prefix_of_the_full_prompt(tokenizer, state, prefix_of):
    prefix = prefix_of(tokenizer(), state)
    full = _full_prompt(tokenizer(), state)
    assert prefix
    assert full[: len(prefix)] == prefix
    assert len(full) > len(prefix)


@pytest.mark.parametrize("tokenizer", TOKENIZERS, ids=lambda token: token.__name__)
@pytest.mark.parametrize("state", STATES, ids=repr)
@pytest.mark.parametrize("prefix_of", PREFIXES, ids=lambda function: function.__module__)
def test_state_prefix_never_carries_the_runtime_question_or_options(tokenizer, state, prefix_of):
    decoded = tokenizer().decode(prefix_of(tokenizer(), state))
    assert decoded.startswith('HEADER\n{"evidence": ')
    assert "criterion" not in decoded
    assert "options" not in decoded
