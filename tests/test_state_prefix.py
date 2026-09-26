"""Shared state prefixes must be true token prefixes of every full prompt."""
import json

import pytest

from semif_phase1.serial import _state_prefix as serial_prefix
from semif_phase1.shared import _state_prefix as shared_prefix

PREFIXES = (serial_prefix, shared_prefix)


class ByteTokenizer:
    """Byte tokenizer whose only merges happen at the evidence boundary."""

    def apply_chat_template(self, turns, tokenize=False, add_generation_prompt=True, enable_thinking=False):
        return "HEADER\n" + turns[-1]["content"] + "\nASSISTANT"

    def encode(self, text, add_special_tokens=False):
        raw = list(text.encode())
        tokens, index = [], 0
        while index < len(raw):
            if raw[index : index + 3] == [ord("?"), ord('"'), ord(",")]:
                tokens.append(9001)
                index += 3
            else:
                tokens.append(raw[index])
                index += 1
        return tokens


def _full_prompt(tokenizer, state, question):
    return tokenizer.encode(f"HEADER\n{json.dumps({'evidence': state})[:-1]}" f", ")


@pytest.mark.parametrize("state", ["Did the deploy use the same parameters?", "owned state"])
@pytest.mark.parametrize("prefix_of", PREFIXES, ids=lambda function: function.__module__)
def test_state_prefix_is_a_token_prefix_of_the_full_prompt(state, prefix_of):
    tokenizer = ByteTokenizer()
    prefix = prefix_of(tokenizer, state)
    full = _full_prompt(tokenizer, state, "Is this urgent?")
    assert prefix
    assert full[: len(prefix)] == prefix
    assert len(full) > len(prefix)


@pytest.mark.parametrize("state", ["Did the deploy use the same parameters?", "owned state"])
@pytest.mark.parametrize("prefix_of", PREFIXES, ids=lambda function: function.__module__)
def test_state_prefix_never_carries_the_runtime_question_or_options(state, prefix_of):
    decoded = bytes(prefix_of(ByteTokenizer(), state)).decode()
    assert decoded.startswith('HEADER\n{"evidence": "')
    assert "criterion" not in decoded
    assert "options" not in decoded


@pytest.mark.parametrize("prefix_of", PREFIXES, ids=lambda function: function.__module__)
def test_state_prefix_ends_on_a_token_boundary_when_evidence_merges_forward(prefix_of):
    # The closing quote of the evidence merges with the following comma, so the
    # shared prefix must stop before it and leave the merged token to the suffix.
    decoded = bytes(prefix_of(ByteTokenizer(), "Did the deploy use the same parameters?")).decode()
    assert decoded.endswith("Did the deploy use the same parameters")
    assert not decoded.endswith('"')
