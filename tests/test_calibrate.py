import json
import sys
from pathlib import Path

import pytest

from semif_phase1 import calibration

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "benchmarks"))
import evaluate  # noqa: E402


def _row(rid, options, label, family="f", group="g"):
    return {
        "id": rid,
        "family": family,
        "group_id": group,
        "label": label,
        "options": [{"id": option} for option in options],
    }


def _pred(rid, option_ids, logits, temperature=None):
    import numpy as np

    values = np.asarray(logits, dtype=float)
    shifted = values - values.max()
    weights = np.exp(shifted)
    probs = (weights / weights.sum()).tolist()
    row = {"id": rid, "option_ids": option_ids, "option_logits": list(logits), "probabilities": probs}
    if temperature is not None:
        row["calibration"] = {"method": "temperature_scaling", "temperature": temperature}
    return row


def test_bad_temperature_rejected():
    for bad in (0, -1.0, float("inf"), float("nan")):
        with pytest.raises(ValueError):
            calibration.check_temperature(bad)


def test_option_set_mismatch_rejected(tmp_path):
    gold = tmp_path / "gold.jsonl"
    pred = tmp_path / "pred.jsonl"
    gold.write_text(json.dumps(_row("x", ["a", "b"], 0)) + "\n")
    pred.write_text(json.dumps({"id": "x", "option_ids": ["a", "c"], "option_logits": [1.0, 2.0]}) + "\n")
    with pytest.raises(ValueError):
        calibration.load_pairs(gold, pred)


def test_duplicate_gold_row_rejected(tmp_path):
    gold = tmp_path / "gold.jsonl"
    pred = tmp_path / "pred.jsonl"
    row = _row("x", ["a", "b"], 0)
    gold.write_text(json.dumps(row) + "\n" + json.dumps(row) + "\n")
    pred.write_text(json.dumps({"id": "x", "option_ids": ["a", "b"], "option_logits": [1.0, 2.0]}) + "\n")
    with pytest.raises(ValueError):
        calibration.load_pairs(gold, pred)


def test_non_finite_logits_rejected(tmp_path):
    gold = tmp_path / "gold.jsonl"
    pred = tmp_path / "pred.jsonl"
    gold.write_text(json.dumps(_row("x", ["a", "b"], 0)) + "\n")
    pred.write_text(json.dumps({"id": "x", "option_ids": ["a", "b"], "option_logits": [1.0, float("inf")]}) + "\n")
    with pytest.raises(ValueError):
        calibration.load_pairs(gold, pred)


def test_argmax_invariant_under_temperature():
    pairs = calibration.load_pairs(ROOT / "benchmarks/data/authored144.jsonl",
                                   ROOT / "results/raw/predictions/direct-authored144.jsonl")
    base = calibration.scored(pairs, 1.0)
    hot = calibration.scored(pairs, calibration.fit_temperature(pairs))
    assert [r["correct"] for r in base] == [r["correct"] for r in hot]


def test_apply_validates_all_rows_before_creating_output(tmp_path):
    predictions = tmp_path / "predictions.jsonl"
    output = tmp_path / "calibrated.jsonl"
    predictions.write_text(
        json.dumps({"id": "good", "option_ids": ["a", "b"], "option_logits": [1.0, 2.0]}) + "\n"
        + json.dumps({"id": "bad", "option_ids": ["a", "b"], "option_logits": [1.0, float("inf")]}) + "\n"
    )
    with pytest.raises(ValueError):
        calibration.apply(predictions, output, 1.5)
    assert not output.exists()


def test_fixed_temperature_cli_does_not_fit(tmp_path, monkeypatch, capsys):
    predictions = tmp_path / "predictions.jsonl"
    output = tmp_path / "calibrated.jsonl"
    predictions.write_text(
        json.dumps({"id": "x", "option_ids": ["a", "b"], "option_logits": [1.0, 2.0]}) + "\n"
    )
    monkeypatch.setattr(calibration, "build_report", lambda *args, **kwargs: pytest.fail("fit was called"))
    monkeypatch.setattr(
        sys,
        "argv",
        ["semif-calibrate", "--predictions", str(predictions), "--temperature", "2.0",
         "--calibrated-out", str(output)],
    )
    calibration.main()
    assert json.loads(output.read_text())["calibration"]["temperature"] == 2.0
    assert json.loads(capsys.readouterr().out)["applied_temperature"] == 2.0


def test_benchmarks_shim_still_exposes_main():
    import calibrate as shim  # noqa: WPS433

    assert shim.main is calibration.main
    assert shim.fit_temperature is calibration.fit_temperature


def test_calibrated_threshold_gate_on_fixture_logits():
    gold = [
        _row("keep", ["yes", "no"], 0, group="g1"),
        _row("review", ["yes", "no"], 1, group="g2"),
        _row("abstain", ["yes", "insufficient"], 1, group="g3"),
    ]
    predictions = [
        _pred("keep", ["yes", "no"], [3.0, 0.0], temperature=1.5),
        _pred("review", ["yes", "no"], [0.2, 0.0], temperature=1.5),
        _pred("abstain", ["yes", "insufficient"], [0.0, 4.0], temperature=1.5),
    ]
    gate = evaluate.calibrated_threshold_gate(gold, predictions, threshold=0.8, abstain_option="insufficient")
    by_id = {row["id"]: row for row in gate["decisions"]}
    assert gate["calibrated_predictions"] is True
    assert gate["applied_temperatures"] == [1.5]
    assert by_id["keep"]["disposition"] == "automatic_semantic_decision"
    assert by_id["review"]["disposition"] == "review"
    assert by_id["abstain"]["disposition"] == "review"
    assert gate["automatic_decisions"] == 1
    assert gate["review_decisions"] == 2


def test_calibrated_threshold_rejects_bad_threshold():
    with pytest.raises(ValueError):
        evaluate.calibrated_threshold_gate([], [], threshold=0)
    with pytest.raises(ValueError):
        evaluate.calibrated_threshold_gate([], [], threshold=1.5)


def test_screening_gate_still_frozen():
    with pytest.raises(ValueError, match="frozen"):
        evaluate.screening_gate([], [], policy="distribution", threshold=0.7)
