"""Benchmark report schemas must stay consumable by verify_published.py."""
import importlib.util
import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
RAW = ROOT / "results" / "raw"


def _load(name):
    spec = importlib.util.spec_from_file_location(name, ROOT / "benchmarks" / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


SHAPE = _load("shape777")
RERANKER = _load("shape777_reranker")


@pytest.fixture(scope="module")
def direct():
    return json.loads((RAW / "shape777-direct.json").read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def reranker():
    return json.loads((RAW / "shape777-reranker.json").read_text(encoding="utf-8"))


def test_direct_report_top_level_matches_the_script(direct):
    assert list(direct) == list(SHAPE.TOP_LEVEL_FIELDS)
    assert direct["version"] == SHAPE.REPORT_VERSION
    assert direct["max_tokens"] == 4096
    assert direct["timing_scope"] == SHAPE.TIMING_SCOPE


def test_direct_report_modes_and_fields_match_the_script(direct):
    assert [row["mode"] for row in direct["results"]] == list(SHAPE.MODES)
    for row in direct["results"]:
        assert set(row) == {"mode", *SHAPE.RESULT_FIELDS[row["mode"]]}


def test_direct_report_comparisons_match_the_script(direct):
    comparisons = direct["comparisons_to_fresh"]
    assert set(comparisons) == {*SHAPE.MODES[1:], "note"}
    assert comparisons["note"] == SHAPE.COMPARISON_NOTE
    for mode, record in comparisons.items():
        if mode == "note":
            continue
        assert list(record) == list(SHAPE.COMPARISON_FIELDS)
        assert record["tolerance"] == SHAPE.COMPARISON_TOLERANCE


def test_reranker_report_top_level_matches_the_script(reranker):
    assert list(reranker) == list(RERANKER.TOP_LEVEL_FIELDS)
    assert reranker["version"] == RERANKER.REPORT_VERSION
    assert reranker["max_tokens"] == 4096
    assert reranker["timing_scope"] == RERANKER.TIMING_SCOPE


def test_reranker_report_fields_match_the_script(reranker):
    for row in reranker["results"]:
        assert set(row) == set(RERANKER.RESULT_FIELDS)


def test_verifier_reads_only_keys_the_scripts_still_emit():
    verifier = (ROOT / "benchmarks" / "verify_published.py").read_text(encoding="utf-8")
    for summary_key, raw_key in (
        ("wall_seconds", "wall_seconds"),
        ("judgments_per_second", "judgments_per_second"),
        ("state_p50_seconds", "state_latency_p50_seconds"),
        ("peak_cuda_bytes", "peak_cuda_bytes"),
    ):
        assert f'("{summary_key}", "{raw_key}")' in verifier
    for raw_key in ("judgments_per_second", "state_latency_p50_seconds"):
        assert raw_key in SHAPE.RESULT_FIELDS["fresh_batch1"]
        assert raw_key in RERANKER.RESULT_FIELDS
