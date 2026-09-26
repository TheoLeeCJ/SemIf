"""Tests for semantic-profile clustering (torch-free, sklearn mocked where needed)."""

import sys
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from semif_phase1 import clustering
from semif_phase1.clustering import _standardize, assign, feature_matrix
from semif_phase1.profiles import AXES, BATTERY_VERSION, FEATURE_NAMES, OPTION_SCHEMA, battery_sha256


def _feature_entry(record_id, feature):
    return {
        "record_id": record_id,
        "feature": list(feature),
        "feature_names": list(FEATURE_NAMES),
        "battery_version": BATTERY_VERSION,
        "battery_sha256": battery_sha256(),
    }


def _artifact(features_by_id):
    return [_feature_entry(record_id, feature) for record_id, feature in features_by_id]


TWO_GROUPS = [
    ("a1", (0.9, 0.05, 0.05, 0.5)),
    ("a2", (0.85, 0.1, 0.05, 0.6)),
    ("a3", (0.95, 0.03, 0.02, 0.4)),
    ("b1", (0.05, 0.9, 0.05, 0.5)),
    ("b2", (0.1, 0.85, 0.05, 0.6)),
    ("b3", (0.02, 0.95, 0.03, 0.4)),
]


def test_standardize_zero_mean_unit_variance():
    standardized, means, stds = _standardize([[1.0, 4.0], [3.0, 4.0]])
    assert means == [2.0, 4.0]
    assert stds[0] == pytest.approx(1.0)
    assert stds[1] == 0.0  # constant coordinate stays zeroed
    assert standardized == [[-1.0, 0.0], [1.0, 0.0]]


def test_standardize_empty():
    assert _standardize([]) == ([], [], [])


def test_feature_matrix_validates_and_orders():
    ids, matrix = feature_matrix(_artifact(TWO_GROUPS))
    assert ids == [record_id for record_id, _ in TWO_GROUPS]
    assert matrix[0] == list(TWO_GROUPS[0][1])


def test_feature_matrix_rejects_bad_artifacts():
    with pytest.raises(ValueError, match="empty profile artifact"):
        feature_matrix([])

    stale = _artifact(TWO_GROUPS)
    stale[0]["battery_version"] = "older"
    with pytest.raises(ValueError, match="Battery version mismatch"):
        feature_matrix(stale)

    wrong_hash = _artifact(TWO_GROUPS)
    wrong_hash[0]["battery_sha256"] = "0" * 64
    with pytest.raises(ValueError, match="stale"):
        feature_matrix(wrong_hash)

    wrong_order = _artifact(TWO_GROUPS)
    wrong_order[0]["feature_names"] = list(reversed(FEATURE_NAMES))
    with pytest.raises(ValueError, match="Feature order"):
        feature_matrix(wrong_order)

    wrong_length = _artifact(TWO_GROUPS)
    wrong_length[1]["feature"] = [0.1]
    with pytest.raises(ValueError, match="feature length"):
        feature_matrix(wrong_length)

    duplicate = _artifact(TWO_GROUPS)
    duplicate[1]["record_id"] = duplicate[0]["record_id"]
    with pytest.raises(ValueError, match="Duplicate record_id"):
        feature_matrix(duplicate)


def test_assign_rejects_unknown_method():
    with pytest.raises(ValueError, match="Unknown method"):
        assign(_artifact(TWO_GROUPS), method="dbscan")


def test_assign_rejects_importless_sklearn(monkeypatch):
    monkeypatch.setitem(sys.modules, "sklearn", None)
    monkeypatch.setitem(sys.modules, "sklearn.cluster", None)
    with pytest.raises(ImportError, match="semif-phase1\[cluster\]"):
        assign(_artifact(TWO_GROUPS), method="kmeans", k=2)


def test_kmeans_parameters_validated_before_fit():
    pytest.importorskip("sklearn")
    with pytest.raises(ValueError, match="at least 2"):
        assign(_artifact(TWO_GROUPS), method="kmeans", k=1)
    with pytest.raises(ValueError, match="cannot exceed"):
        assign(_artifact(TWO_GROUPS), method="kmeans", k=len(TWO_GROUPS) + 1)


@pytest.mark.skipif(
    __import__("importlib").util.find_spec("sklearn") is None,
    reason="scikit-learn not installed",
)
def test_kmeans_assigns_two_separable_groups():
    result = assign(_artifact(TWO_GROUPS), method="kmeans", k=2, seed=217)
    assert result["method"] == "MiniBatchKMeans"
    assert result["n_records"] == len(TWO_GROUPS)
    assert result["battery_sha256"] == battery_sha256()
    assert len(result["clusters"]) == 2
    a_ids = {f"a{i}" for i in (1, 2, 3)}
    b_ids = {f"b{i}" for i in (1, 2, 3)}
    memberships = {frozenset(m) for m in result["memberships"].values()}
    assert memberships == {frozenset(a_ids), frozenset(b_ids)}
    for info in result["clusters"].values():
        assert info["size"] == 3
        assert info["representative_record_id"] in a_ids | b_ids
    assert len(result["standardization"]["means"]) == len(FEATURE_NAMES)


def test_hdbscan_noise_and_membership_layout(monkeypatch):
    fake_labels = [0, 0, 1, 1, -1, -1]
    fake_model = SimpleNamespace(fit_predict=Mock(return_value=fake_labels))
    fake_hdbscan_cls = Mock(return_value=fake_model)
    fake_module = SimpleNamespace(HDBSCAN=fake_hdbscan_cls)
    monkeypatch.setitem(sys.modules, "sklearn", SimpleNamespace(cluster=fake_module))
    monkeypatch.setitem(sys.modules, "sklearn.cluster", fake_module)

    result = assign(_artifact(TWO_GROUPS), method="hdbscan", min_cluster_size=2)
    fake_hdbscan_cls.assert_called_once_with(min_cluster_size=2)
    assert result["method"] == "HDBSCAN"
    assert sorted(result["noise_record_ids"]) == ["b2", "b3"]
    assert sorted(result["memberships"]) == ["cluster-0", "cluster-1"]
    assert result["memberships"]["cluster-0"] == ["a1", "a2"]
    assert result["clusters"]["cluster-0"]["representative_record_id"] in {"a1", "a2"}


def test_hdbscan_min_cluster_size_validated():
    with pytest.raises(ValueError, match="min_cluster_size"):
        clustering.hdbscan([[0.0], [1.0]], min_cluster_size=1)
