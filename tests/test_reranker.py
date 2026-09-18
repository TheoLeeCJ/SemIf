import pytest

from semif_phase1.reranker import _answer_ids, _encode


class Tokenizer:
    def encode(self, text, add_special_tokens=False):
        assert add_special_tokens is False
        return list(text.encode())


class AnswerTokenizer:
    def encode(self, text, add_special_tokens=False):
        return {"no": [4], "yes": [7]}[text]

    def convert_tokens_to_ids(self, text):
        return {"no": 4, "yes": 7}[text]


def test_reranker_pair_contains_only_declared_semantics():
    row = {
        "id": "case",
        "state": "STATE",
        "question": "QUESTION",
        "options": [{"id": "a", "description": "ANSWER"}, {"id": "b", "description": "OTHER"}],
        "label": "b",
    }
    ids, prompt_hash = _encode(Tokenizer(), row, row["options"][0], 4096)
    prompt = bytes(ids).decode()
    assert all(value in prompt for value in ("STATE", "QUESTION", "ANSWER"))
    assert "label" not in prompt and row["options"][1]["description"] not in prompt
    assert len(prompt_hash) == 64


def test_no_silent_truncation():
    row = {"id": "case", "state": "x" * 100, "question": "q", "options": [{"id": "a", "description": "a"}]}
    with pytest.raises(ValueError, match="no truncation|exceed"):
        _encode(Tokenizer(), row, row["options"][0], 10)


def test_official_yes_no_token_contract_is_checked():
    assert _answer_ids(AnswerTokenizer()) == (4, 7)
