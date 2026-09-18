from semif_phase1.serial import _state_prefix


class Tokenizer:
    def apply_chat_template(self, turns, tokenize=False, add_generation_prompt=True, enable_thinking=False):
        assert tokenize is False and add_generation_prompt is True and enable_thinking is False
        return "HEADER\n" + turns[-1]["content"] + "\nASSISTANT"

    def encode(self, text, add_special_tokens=False):
        assert add_special_tokens is False
        return list(text.encode())


def test_state_prefix_stops_before_runtime_question_and_options():
    prefix = bytes(_state_prefix(Tokenizer(), "owned state")).decode()
    assert "owned state" in prefix
    assert "prefix boundary placeholder" not in prefix
    assert '"options"' not in prefix
