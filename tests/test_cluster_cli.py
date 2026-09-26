"""Tests for the semif-cluster CLI (no model, no sklearn needed for profiles)."""

import json
import sys

import pytest

from semif_phase1 import cluster_cli
from semif_phase1.profiles import AXES, OPTION_SCHEMA


RECORDS = [
    {"id": "rec-1", "state": "I was charged twice"},
    {"id": "rec-2", "state": "I cannot log in"},
]


def _run(monkeypatch, argv):
    monkeypatch.setattr(sys, "argv", argv)
    cluster_cli.main()


def test_profiles_expands_records(tmp_path, monkeypatch):
    source = tmp_path / "records.jsonl"
    source.write_text("\n".join(json.dumps(r) for r in RECORDS) + "\n")
    output = tmp_path / "probes.jsonl"
    _run(monkeypatch, ["semif-cluster", "profiles", "--input", str(source), "--output", str(output)])
    rows = [json.loads(line) for line in output.read_text().splitlines()]
    assert len(rows) == len(RECORDS) * len(AXES)
    assert rows[0]["id"] == "rec-1::billing"
    assert rows[0]["options"] == list(OPTION_SCHEMA)
    assert {row["state"] for row in rows} == {"I was charged twice", "I cannot log in"}


def test_profiles_rejects_empty_input(tmp_path, monkeypatch):
    source = tmp_path / "empty.jsonl"
    source.write_text("")
    with pytest.raises(SystemExit):
        _run(monkeypatch, ["semif-cluster", "profiles", "--input", str(source),
                           "--output", str(tmp_path / "out.jsonl")])


def test_profiles_never_overwrites(tmp_path, monkeypatch, capsys):
    source = tmp_path / "records.jsonl"
    source.write_text(json.dumps(RECORDS[0]) + "\n")
    output = tmp_path / "probes.jsonl"
    output.write_text("existing\n")
    with pytest.raises(SystemExit) as error:
        _run(monkeypatch, ["semif-cluster", "profiles", "--input", str(source), "--output", str(output)])
    assert error.value.code == 2
    assert "Output must be new" in capsys.readouterr().err
    assert output.read_text() == "existing\n"


def _scoring_line(record_id, axis, probabilities=(0.7, 0.2, 0.1)):
    return {"id": f"{record_id}::{axis}", "probabilities": list(probabilities),
            "options": list(OPTION_SCHEMA)}


def test_cluster_stub_records_are_aggregated(monkeypatch, tmp_path):
    """Verify full profiles → scorings → cluster wiring, with clustering.assign stubbed."""
    captured = {}

    def fake_assign(artifact, method, **params):
        captured["artifact"] = artifact
        captured["method"] = method
        captured["params"] = params
        return {"ok": True}

    monkeypatch.setattr(cluster_cli.clustering, "assign", fake_assign)
    scorings = tmp_path / "scorings.jsonl"
    lines = [
        _scoring_line(record_id, axis)
        for record_id in ("rec-1", "rec-2")
        for axis in AXES
    ]
    scorings.write_text("\n".join(json.dumps(line) for line in lines) + "\n")
    output = tmp_path / "catalog.json"
    _run(monkeypatch, ["semif-cluster", "cluster", "--input", str(scorings),
                       "--output", str(output), "--method", "kmeans", "--k", "2"])
    assert json.loads(output.read_text()) == {"ok": True, "catalog_version": json.loads(output.read_text())["catalog_version"]}
    assert captured["method"] == "kmeans"
    assert captured["params"] == {"k": 2, "seed": 217}
    assert {entry["record_id"] for entry in captured["artifact"]} == {"rec-1", "rec-2"}


def test_cluster_requires_k_for_kmeans(tmp_path, monkeypatch):
    scorings = tmp_path / "scorings.jsonl"
    scorings.write_text(json.dumps(_scoring_line("r", AXES[0])) + "\n")
    with pytest.raises(SystemExit):
        _run(monkeypatch, ["semif-cluster", "cluster", "--input", str(scorings),
                           "--output", str(tmp_path / "out.json")])


def test_cluster_hdbscan_params_forwarded(monkeypatch, tmp_path):
    captured = {}

    def fake_assign(artifact, method, **params):
        captured.update(method=method, params=params)
        return {"ok": True}

    monkeypatch.setattr(cluster_cli.clustering, "assign", fake_assign)
    scorings = tmp_path / "scorings.jsonl"
    lines = [_scoring_line("rec-1", axis) for axis in AXES]
    scorings.write_text("\n".join(json.dumps(line) for line in lines) + "\n")
    _run(monkeypatch, ["semif-cluster", "cluster", "--input", str(scorings),
                       "--output", str(tmp_path / "out.json"),
                       "--method", "hdbscan", "--min-cluster-size", "3", "--min-samples", "2"])
    assert captured["method"] == "hdbscan"
    assert captured["params"] == {"min_cluster_size": 3, "min_samples": 2, "seed": 217}


def test_cluster_validation_errors_exit_cleanly(tmp_path, monkeypatch, capsys):
    scorings = tmp_path / "scorings.jsonl"
    # Only one axis scored for rec-1: profiles.aggregate must raise.
    scorings.write_text(json.dumps(_scoring_line("rec-1", AXES[0])) + "\n")
    with pytest.raises(SystemExit) as error:
        _run(monkeypatch, ["semif-cluster", "cluster", "--input", str(scorings),
                           "--output", str(tmp_path / "out.json"), "--method", "kmeans", "--k", "2"])
    assert error.value.code == 2
    assert "missing scored axes" in capsys.readouterr().err
