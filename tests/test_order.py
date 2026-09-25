"""CPU unit tests for option-order alignment, averaging, and sensitivity reports."""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import pytest

from semif_phase1.core import softmax
from semif_phase1.order import (
    STABILIZE_PROMPT_VERSION,
    align_values_by_id,
    apply_permutation,
    average_aligned,
    check_stabilize_k,
    report_pair_sensitivity,
    score_with_stabilized_order,
    select_permutations,
    stabilize_from_permutation_results,
    summarize_sensitivity,
    total_variation,
)

ROOT = Path(__file__).resolve().parents[1]


ROW = {
    "id": "demo",
    "state": "owned evidence",
    "question": "Which answer follows?",
    "options": [
        {"id": "yes", "description": "Yes."},
        {"id": "no", "description": "No."},
        {"id": "maybe", "description": "Maybe."},
    ],
    "label": 0,
}


def test_check_stabilize_k_rejects_small_values():
    for bad in (0, 1, -2, 1.5, True):
        with pytest.raises(ValueError):
            check_stabilize_k(bad)


def test_apply_permutation_preserves_gold_id():
    permuted = apply_permutation(ROW, (2, 0, 1))
    assert [option["id"] for option in permuted["options"]] == ["maybe", "yes", "no"]
    assert permuted["options"][permuted["label"]]["id"] == "yes"


def test_select_permutations_includes_identity_and_reverse():
    perms = select_permutations(3, 2, seed=0)
    assert perms[0] == (0, 1, 2)
    assert perms[1] == (2, 1, 0)
    assert select_permutations(3, 6, seed=0)[0] == (0, 1, 2)
    assert len(select_permutations(3, 6, seed=0)) == 6


def test_align_and_average_logits_then_softmax():
    canonical = ["yes", "no", "maybe"]
    # Identity: yes strongest. Reversed display scored with letter-A bias toward maybe.
    identity = align_values_by_id(["yes", "no", "maybe"], [3.0, 1.0, 0.0], canonical)
    reversed_ = align_values_by_id(["maybe", "no", "yes"], [4.0, 1.0, 0.5], canonical)
    mean = average_aligned([identity, reversed_])
    assert mean == pytest.approx([1.75, 1.0, 2.0])
    probs = softmax(mean)
    assert sum(probs) == pytest.approx(1.0)
    assert probs[2] == max(probs)


def test_stabilize_from_fixture_logits_marks_separate_readout():
    canonical = ["yes", "no"]
    results = [
        {"id": "demo", "option_ids": ["yes", "no"], "option_logits": [2.0, 1.0],
         "prompt_sha256": "a", "prompt_version": "direct-options-v1", "total_seconds": 0.1},
        {"id": "demo", "option_ids": ["no", "yes"], "option_logits": [3.0, 0.0],
         "prompt_sha256": "b", "prompt_version": "direct-options-v1", "total_seconds": 0.1},
    ]
    out = stabilize_from_permutation_results(
        canonical, results, permutations=[(0, 1), (1, 0)], k=2, seed=0, base_result=results[0]
    )
    assert out["prompt_version"] == STABILIZE_PROMPT_VERSION
    assert out["stabilize_order"]["cost_multiplier"] == 2
    assert out["option_ids"] == canonical
    # Aligned: identity [2,1], reverse maps to yes=0, no=3 → mean [1, 2]
    assert out["option_logits"] == pytest.approx([1.0, 2.0])
    assert out["probabilities"][1] > out["probabilities"][0]


def test_score_with_stabilized_order_uses_stub_scorer():
    calls = []

    def stub(row):
        calls.append([option["id"] for option in row["options"]])
        # Pure position bias: always prefer display slot 0.
        logits = [5.0] + [0.0] * (len(row["options"]) - 1)
        return {
            "id": row["id"],
            "option_ids": [option["id"] for option in row["options"]],
            "option_logits": logits,
            "probabilities": softmax(logits),
            "prompt_sha256": "stub",
            "prompt_version": "direct-options-v1",
            "total_seconds": 0.01,
        }

    out = score_with_stabilized_order(stub, ROW, k=2, seed=0)
    assert calls == [["yes", "no", "maybe"], ["maybe", "no", "yes"]]
    assert out["prompt_version"] == STABILIZE_PROMPT_VERSION
    # Averaging cancels pure A-bias across identity+reverse for ends; middle stays low.
    assert out["option_ids"] == ["yes", "no", "maybe"]
    assert out["option_logits"][0] == pytest.approx(out["option_logits"][2])
    assert out["option_logits"][1] < out["option_logits"][0]


def test_sensitivity_report_detects_flip_tv_and_position_bias():
    base = {"id": "b", "option_ids": ["yes", "no", "maybe"], "probabilities": [0.7, 0.2, 0.1]}
    # Reverse display order in the variant prediction file (ids follow display).
    variant = {"id": "v", "option_ids": ["maybe", "no", "yes"], "probabilities": [0.6, 0.25, 0.15]}
    pair = report_pair_sensitivity(
        base, variant,
        base_display_ids=["yes", "no", "maybe"],
        variant_display_ids=["maybe", "no", "yes"],
    )
    assert pair["flipped"]
    assert pair["from"] == "yes" and pair["to"] == "maybe"
    assert pair["base_display_position"] == 0
    assert pair["variant_display_position"] == 0
    assert pair["total_variation"] == pytest.approx(total_variation([0.7, 0.2, 0.1], [0.15, 0.25, 0.6]))
    summary = summarize_sensitivity([pair])
    assert summary["argmax_flips"] == 1
    assert summary["position_bias"]["base_position_win_rates"][0] == 1.0
    assert summary["position_bias"]["variant_position_win_rates"][0] == 1.0


def test_harness_on_committed_direct_option_reversal(tmp_path):
    sys.path.insert(0, str(ROOT / "benchmarks"))
    import evaluate_option_order as harness  # noqa: E402

    output = tmp_path / "report.json"
    monkey_argv = [
        "evaluate_option_order.py",
        "--gold", str(ROOT / "benchmarks/data/authored144.jsonl"),
        "--perturbations", str(ROOT / "benchmarks/data/perturbations108.jsonl"),
        "--base-predictions", str(ROOT / "results/raw/predictions/direct-authored144.jsonl"),
        "--perturbation-predictions", str(ROOT / "results/raw/predictions/direct-perturbations108.jsonl"),
        "--output", str(output),
    ]
    old = sys.argv
    try:
        sys.argv = monkey_argv
        harness.main()
    finally:
        sys.argv = old
    report = json.loads(output.read_text())
    assert report["summary"]["rows"] == 36
    assert report["summary"]["argmax_flips"] == 10
    assert 0 < report["summary"]["mean_total_variation"] <= 1
    assert math.isclose(sum(report["summary"]["position_bias"]["base_position_win_rates"]), 1.0)
