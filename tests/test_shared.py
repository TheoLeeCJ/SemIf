import pytest

from semif_phase1.shared import _suffix_layout


def test_suffix_padding_follows_real_tokens():
    layout, ends = _suffix_layout([[3, 4], [5]], 7, 0)
    assert layout["input_ids"] == [[3, 4], [5, 0]]
    assert layout["attention_mask"] == [[1] * 9, [1] * 8 + [0]]
    assert layout["position_ids"] == [[7, 8], [7, 0]]
    assert ends == [1, 0]


def test_empty_suffix_is_rejected():
    with pytest.raises(ValueError):
        _suffix_layout([[1], []], 7, 0)
