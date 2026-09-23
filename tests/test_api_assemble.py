"""SemIf rows -> Jev answers. Pure: no model, no server, no weights."""

import json
import math

import pytest

from semif_api.assemble import build_answer, build_response, confidence, usage
from semif_api.translate import QuestionPlan


def result(option_ids, probabilities, *, row_id="q", logits=None, input_tokens=142):
    return {
        "id": row_id,
        "option_ids": list(option_ids),
        "probabilities": list(probabilities),
        "option_logits": list(logits if logits is not None else [math.log(max(p, 1e-9)) for p in probabilities]),
        "input_tokens": input_tokens,
        "prompt_sha256": "7ac35785",
        "prompt_version": "direct-options-v1",
        "model": {"source": "Qwen/Qwen3.5-4B"},
        "readout": "native full-vocabulary last-position logits restricted to declared answer slots",
        "probability_status": "conditional option score; uncalibrated as decision confidence",
    }


def test_even_split_has_no_confidence_and_one_hot_is_certain():
    assert confidence([0.25] * 4) == 0.0
    assert confidence([1.0, 0.0, 0.0]) == 1.0


@pytest.mark.parametrize("shown_peak,shown_confidence", [(0.88, 0.81), (0.95, 0.92)])
def test_the_published_examples_are_consistent_with_this_formula(shown_peak, shown_confidence):
    """docs.typesafe.ai publishes rounded numbers, so check for consistency.

    The Choice example shows {billing 0.88, technical 0.12, sales 0.0} with
    confidence 0.81, and the Score example shows a 0.95 peak with 0.92. Applying
    the formula to the displayed probabilities gives 0.82 and 0.925, not 0.81 and
    0.92. Both are explained by two-decimal display rounding, but only inside a
    narrow band: this asserts such a band exists rather than pretending the
    published digits reproduce exactly.
    """
    half = 0.005
    peak_from_probability = (shown_peak - half, shown_peak + half)
    # Invert confidence = (3p - 1) / 2 over the displayed confidence interval.
    peak_from_confidence = tuple(((shown_confidence + offset) * 2 + 1) / 3 for offset in (-half, half))
    low = max(peak_from_probability[0], peak_from_confidence[0])
    high = min(peak_from_probability[1], peak_from_confidence[1])
    assert low < high, "no true peak explains both published digits under this formula"
    midpoint = (low + high) / 2
    assert round(confidence([midpoint, 1 - midpoint, 0.0]), 2) == shown_confidence


def test_confidence_stays_a_float_at_the_boundaries():
    # min(1, x) returns the int 1 for x > 1 and breaks strict SDK clients.
    assert isinstance(confidence([1.0, 0.0]), float)
    assert isinstance(confidence([0.5, 0.5]), float)


def test_noul_reads_the_true_option_and_carries_no_confidence():
    plan = QuestionPlan(key="is_urgent", kind="noul", option_ids=("true", "false"))
    answer = build_answer(plan, result(("true", "false"), [0.95, 0.05], row_id="is_urgent"))
    assert answer["type"] == "noul"
    assert answer["noul"] == pytest.approx(0.95)
    assert "confidence" not in answer


def test_choice_reports_the_argmax_and_the_full_distribution():
    plan = QuestionPlan(key="q", kind="choice", option_ids=("billing", "technical", "sales"))
    answer = build_answer(plan, result(plan.option_ids, [0.88, 0.12, 0.0]))
    assert answer["choice"] == "billing"
    assert answer["probabilities"] == {"billing": 0.88, "technical": 0.12, "sales": 0.0}
    assert answer["confidence"] == pytest.approx(0.82)


def test_score_is_the_probability_weighted_level_index():
    # docs.typesafe.ai/api: {0: 0.0, 1: 0.95, 2: 0.05} -> 1.05
    plan = QuestionPlan(key="q", kind="score", option_ids=("0", "1", "2"),
                        legend={"0": "Calm", "1": "Frustrated", "2": "Very angry"})
    answer = build_answer(plan, result(plan.option_ids, [0.0, 0.95, 0.05]))
    assert answer["score"] == pytest.approx(1.05)
    assert answer["legend"] == {"0": "Calm", "1": "Frustrated", "2": "Very angry"}
    assert answer["probabilities"] == {"0": 0.0, "1": 0.95, "2": 0.05}


def test_unit_temperature_reuses_the_backend_probabilities_unchanged():
    plan = QuestionPlan(key="q", kind="choice", option_ids=("a", "b"))
    scored = result(plan.option_ids, [0.73, 0.27], logits=[1.0, 0.0])
    assert build_answer(plan, scored, 1.0)["probabilities"] == {"a": 0.73, "b": 0.27}


def test_temperature_flattens_the_distribution_without_moving_the_argmax():
    plan = QuestionPlan(key="q", kind="choice", option_ids=("a", "b"))
    scored = result(plan.option_ids, [0.9, 0.1], logits=[2.2, 0.0])
    hot = build_answer(plan, scored, 2.5)
    assert hot["choice"] == "a"
    assert hot["probabilities"]["a"] < 0.9
    assert hot["confidence"] < build_answer(plan, scored, 1.0)["confidence"]
    assert hot["semif"]["temperature"] == 2.5


def test_every_numeric_value_serializes_with_a_decimal_point():
    # typesafe_sdk response models are strict: `0` where a float is declared fails.
    plans = [
        QuestionPlan(key="n", kind="noul", option_ids=("true", "false")),
        QuestionPlan(key="c", kind="choice", option_ids=("a", "b")),
        QuestionPlan(key="s", kind="score", option_ids=("0", "1"), legend={"0": "Low", "1": "High"}),
    ]
    results = [
        result(("true", "false"), [1.0, 0.0], row_id="n"),
        result(("a", "b"), [1.0, 0.0], row_id="c"),
        result(("0", "1"), [1.0, 0.0], row_id="s"),
    ]
    body = build_response("semif/x@abc+torch-bfloat16", plans, results,
                          mode="shared", backend="torch", metadata={}, timing={"prefix_tokens": 100})
    for answer in body["answers"].values():
        for field in ("noul", "score", "confidence"):
            if field in answer:
                assert isinstance(answer[field], float)
        for value in answer.get("probabilities", {}).values():
            assert isinstance(value, float)
    encoded = json.dumps(body, allow_nan=False)
    assert '"noul": 1.0' in encoded and '"confidence": 1.0' in encoded


def test_per_type_temperature_is_applied_by_question_kind():
    plans = [
        QuestionPlan(key="c", kind="choice", option_ids=("a", "b")),
        QuestionPlan(key="n", kind="noul", option_ids=("true", "false")),
    ]
    results = [result(("a", "b"), [0.9, 0.1], row_id="c", logits=[2.2, 0.0]),
               result(("true", "false"), [0.9, 0.1], row_id="n", logits=[2.2, 0.0])]
    body = build_response("m", plans, results, temperature=lambda kind: 2.5 if kind == "choice" else 1.0,
                          mode="direct", backend="torch", metadata={})
    assert body["answers"]["c"]["semif"]["temperature"] == 2.5
    assert body["answers"]["n"]["semif"]["temperature"] == 1.0
    assert body["answers"]["n"]["noul"] == pytest.approx(0.9)


def test_shared_usage_counts_the_state_once():
    results = [result(("a", "b"), [0.5, 0.5], row_id="a", input_tokens=120),
               result(("a", "b"), [0.5, 0.5], row_id="b", input_tokens=130)]
    assert usage(results, "shared", {"prefix_tokens": 100}) == {"input_tokens": 150, "output_tokens": 0}
    assert usage(results, "direct", None) == {"input_tokens": 250, "output_tokens": 0}


def test_output_tokens_are_zero_because_no_answer_token_is_generated():
    assert usage([result(("a", "b"), [0.5, 0.5])], "direct", None)["output_tokens"] == 0


def test_answers_are_keyed_by_the_question_ids_that_were_sent():
    plans = [QuestionPlan(key="is_urgent", kind="noul", option_ids=("true", "false")),
             QuestionPlan(key="department", kind="choice", option_ids=("a", "b"))]
    results = [result(("a", "b"), [0.5, 0.5], row_id="department"),
               result(("true", "false"), [0.5, 0.5], row_id="is_urgent")]
    body = build_response("m", plans, results, mode="shared", backend="mlx", metadata={"source": "x"})
    assert set(body["answers"]) == {"is_urgent", "department"}
    assert body["model"] == "m"
    assert body["semif"]["backend"] == "mlx"


def test_a_result_that_does_not_match_its_plan_is_rejected():
    plan = QuestionPlan(key="q", kind="choice", option_ids=("a", "b"))
    with pytest.raises(ValueError, match="planned options"):
        build_answer(plan, result(("a", "c"), [0.5, 0.5]))


def test_the_extension_block_carries_the_projects_own_probability_statement():
    plan = QuestionPlan(key="q", kind="choice", option_ids=("a", "b"))
    block = build_answer(plan, result(plan.option_ids, [0.6, 0.4]))["semif"]
    assert block["probability_status"].endswith("uncalibrated as decision confidence")
    assert block["prompt_version"] == "direct-options-v1"
    assert block["confidence_formula"] == "normalized-peak-v1"
