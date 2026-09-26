"""Tests for the frozen probe battery and profile aggregation (torch-free)."""

import json
import math

import pytest

from semif_phase1 import profiles
from semif_phase1.profiles import (
    AXES,
    BATTERY_VERSION,
    FEATURE_NAMES,
    OPTION_SCHEMA,
    aggregate,
    battery_sha256,
    detect_record_id,
    features_for_record,
    probe_rows,
)


RECORD = {"id": "rec-1", "state": {"text": "I was charged twice for my invoice"}}


def _scoring(record_id, axis, probabilities=(0.7, 0.2, 0.1)):
    return {
        "id": f"{record_id}::{axis}",
        "probabilities": list(probabilities),
        "options": list(OPTION_SCHEMA),
    }


def _full_scorings(record_id, probabilities=(0.7, 0.2, 0.1)):
    return [_scoring(record_id, axis, probabilities) for axis in AXES]


def test_probe_rows_share_state_and_schema():
    rows = probe_rows(RECORD)
    assert len(rows) == len(AXES)
    assert [row["id"] for row in rows] == [f"rec-1::{axis}" for axis in AXES]
    assert all(row["state"] == RECORD["state"] for row in rows)
    assert all(row["options"] == list(OPTION_SCHEMA) for row in rows)
    assert len({row["question"] for row in rows}) == len(AXES)


def test_probe_rows_validate_record_shape():
    with pytest.raises(ValueError, match="'id' and 'state'"):
        probe_rows({"id": "x"})
    with pytest.raises(ValueError, match="nonempty string"):
        probe_rows({"id": "", "state": "evidence"})
    with pytest.raises(ValueError, match="finite JSON-compatible"):
        probe_rows({"id": "x", "state": {"bad": float("nan")}})


def test_probe_rows_pass_core_validation():
    from semif_phase1.core import validate_row

    for row in probe_rows(RECORD):
        validate_row(row)


def test_battery_hash_is_stable():
    assert battery_sha256() == battery_sha256()


def test_features_are_frozen_order_and_finite():
    result = features_for_record("rec-1", _full_scorings("rec-1"))
    assert result["feature_names"] == list(FEATURE_NAMES)
    assert result["battery_version"] == BATTERY_VERSION
    assert result["battery_sha256"] == battery_sha256()
    assert result["probe_count"] == len(AXES)
    assert all(math.isfinite(value) for value in result["feature"])
    assert result["feature"][0] == pytest.approx(0.7)
    assert result["feature"][1] == pytest.approx(0.2)
    assert result["feature"][2] == pytest.approx(0.1)
    assert result["feature"][3] == pytest.approx(
        -(0.7 * math.log2(0.7) + 0.2 * math.log2(0.2) + 0.1 * math.log2(0.1)))
    assert [entry["axis"] for entry in result["probe_probabilities"]] == list(AXES)


def test_features_reject_missing_duplicate_and_unknown_axes():
    with pytest.raises(ValueError, match="missing scored axes"):
        features_for_record("rec-1", _full_scorings("rec-1")[:-1])
    with pytest.raises(ValueError, match="Duplicate scored row id"):
        features_for_record("rec-1", _full_scorings("rec-1") + [_scoring("rec-1", AXES[0])])
    with pytest.raises(ValueError, match="Unexpected scored row id"):
        features_for_record("rec-1", _full_scorings("rec-1") + [_scoring("rec-1", "bogus")])


def test_features_reject_schema_drift_and_bad_probabilities():
    bad_schema = _scoring("rec-1", AXES[0])
    bad_schema["options"] = list(reversed(OPTION_SCHEMA))
    with pytest.raises(ValueError, match="option ids differ"):
        features_for_record("rec-1", [bad_schema, *_full_scorings("rec-1")[1:]])

    unnormalized = _scoring("rec-1", AXES[0], probabilities=(0.5, 0.2, 0.1))
    with pytest.raises(ValueError, match="sum to 1"):
        features_for_record("rec-1", [unnormalized, *_full_scorings("rec-1")[1:]])

    nonfinite = _scoring("rec-1", AXES[0], probabilities=(float("nan"), 0.0, 0.0))
    with pytest.raises(ValueError, match="non-finite"):
        features_for_record("rec-1", [nonfinite, *_full_scorings("rec-1")[1:]])


def test_aggregate_groups_by_record():
    scorings = _full_scorings("a") + _full_scorings("b", probabilities=(0.1, 0.8, 0.1))
    artifacts = aggregate(scorings)
    assert {artifact["record_id"] for artifact in artifacts} == {"a", "b"}


def test_aggregate_rejects_any_incomplete_record():
    scorings = _full_scorings("a") + _full_scorings("b")[:-1]
    with pytest.raises(ValueError, match="missing scored axes"):
        aggregate(scorings)


def test_detect_record_id_round_trip_and_validation():
    assert detect_record_id("rec-1::billing") == "rec-1"
    with pytest.raises(ValueError, match="not a valid scored probe row id"):
        detect_record_id("rec-1")


def test_probe_rows_json_round_trip(tmp_path):
    path = tmp_path / "rows.jsonl"
    path.write_text("\n".join(json.dumps(row) for row in probe_rows(RECORD)) + "\n")
    loaded = [json.loads(line) for line in path.read_text().splitlines()]
    assert loaded == probe_rows(RECORD)
