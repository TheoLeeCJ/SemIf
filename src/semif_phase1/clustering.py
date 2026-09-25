"""Semantic-profile clustering.

Stdlib-only import. Clustering itself is lazy-loaded from sklearn
(semif-phase1[cluster]); the module imports cleanly without it so that
tests and plain profile utilities run anywhere numpy or torch are absent.
"""

from __future__ import annotations

import json
import statistics

from .core import digest
from .profiles import BATTERY_VERSION, FEATURE_NAMES, battery_sha256

INSTALL_HINT = "Clustering requires scikit-learn. Install with: pip install 'semif-phase1[cluster]'"


def _standardize(features: list[list[float]]) -> tuple[list[list[float]], list[float], list[float]]:
    """Z-score standardization in pure Python. Returns (standardized, means, stds)."""
    n = len(features)
    if n == 0:
        return [], [], []
    dim = len(features[0])
    means = [statistics.mean(row[i] for row in features) for i in range(dim)]
    stds = [statistics.pstdev(row[i] for row in features) for i in range(dim)]
    # A zero std means this coordinate is constant across records; leave it at 0.
    standardized = [
        [0.0 if stds[i] == 0 else (row[i] - means[i]) / stds[i] for i in range(dim)]
        for row in features
    ]
    return standardized, means, stds


def _squared_distance(left: list[float], right: list[float]) -> float:
    return sum((a - b) ** 2 for a, b in zip(left, right))


def _nearest_to_center(members: list[int], matrix: list[list[float]], center: list[float]) -> int:
    """Matrix index of the member closest to the cluster center."""
    return min(members, key=lambda index: _squared_distance(matrix[index], center))


def _medoid(members: list[int], matrix: list[list[float]]) -> int:
    """Matrix index of the member minimizing total squared distance to other members."""
    return min(
        members,
        key=lambda index: sum(_squared_distance(matrix[index], matrix[other]) for other in members),
    )


def feature_matrix(profile_artifact: list[dict]) -> tuple[list[str], list[list[float]]]:
    """Validate a profile artifact; return (record_ids, frozen-order feature matrix)."""
    if not profile_artifact:
        raise ValueError("Cannot cluster an empty profile artifact")
    first = profile_artifact[0]
    if first.get("battery_version") != BATTERY_VERSION:
        raise ValueError(f"Battery version mismatch: {first.get('battery_version')!r} != {BATTERY_VERSION!r}")
    if first.get("battery_sha256") != battery_sha256():
        raise ValueError("Battery sha256 mismatch; profile artifact is stale")
    if first.get("feature_names") != list(FEATURE_NAMES):
        raise ValueError("Feature order does not match frozen FEATURE_NAMES")
    record_ids: list[str] = []
    matrix: list[list[float]] = []
    for entry in profile_artifact:
        feature = entry["feature"]
        if len(feature) != len(FEATURE_NAMES):
            raise ValueError(f"{entry['record_id']}: feature length does not match FEATURE_NAMES")
        if entry["record_id"] in record_ids:
            raise ValueError(f"Duplicate record_id {entry['record_id']!r} in profile artifact")
        record_ids.append(entry["record_id"])
        matrix.append(list(feature))
    return record_ids, matrix


def kmeans(matrix: list[list[float]], k: int, seed: int = 217) -> dict:
    """Cluster a standardized matrix via MiniBatchKMeans. Lazy sklearn import."""
    try:
        from sklearn.cluster import MiniBatchKMeans
    except ImportError as error:
        raise ImportError(INSTALL_HINT) from error
    if k < 2:
        raise ValueError("k must be at least 2 for MiniBatchKMeans")
    if k > len(matrix):
        raise ValueError("k cannot exceed the number of records")
    model = MiniBatchKMeans(n_clusters=k, random_state=seed, n_init="auto")
    labels = [int(label) for label in model.fit_predict(matrix)]
    centers = [[float(value) for value in row] for row in model.cluster_centers_]
    memberships: dict[str, list[int]] = {}
    for index, label in enumerate(labels):
        memberships.setdefault(f"cluster-{label}", []).append(index)
    return {
        "method": "MiniBatchKMeans",
        "params": {"k": k, "seed": seed},
        "labels": labels,
        "memberships": memberships,
        "centroids": centers,
        "noise": [],
    }


def hdbscan(
    matrix: list[list[float]],
    min_cluster_size: int = 5,
    min_samples: int | None = None,
    seed: int = 217,
) -> dict:
    """Cluster a standardized matrix via HDBSCAN; noise rows land in ``noise``. Lazy import."""
    try:
        from sklearn.cluster import HDBSCAN
    except ImportError as error:
        raise ImportError(INSTALL_HINT) from error
    if min_cluster_size < 2:
        raise ValueError("min_cluster_size must be at least 2")
    kwargs: dict = {"min_cluster_size": min_cluster_size}
    if min_samples is not None:
        kwargs["min_samples"] = min_samples
    model = HDBSCAN(**kwargs)
    labels = [int(label) for label in model.fit_predict(matrix)]
    memberships: dict[str, list[int]] = {}
    noise: list[int] = []
    for index, label in enumerate(labels):
        if label == -1:
            noise.append(index)
        else:
            memberships.setdefault(f"cluster-{label}", []).append(index)
    return {
        "method": "HDBSCAN",
        "params": {"min_cluster_size": min_cluster_size, "min_samples": min_samples, "seed": seed},
        "labels": labels,
        "memberships": memberships,
        "centroids": None,
        "noise": noise,
    }


def assign(profile_artifact: list[dict], method: str = "kmeans", **params) -> dict:
    """Assign records to clusters from a profile artifact.

    Returns a JSON-serializable clustering artifact with provenance. The fitted
    standardization (means/stds) is stored so future single-record assignment
    uses identical coordinates.
    """
    record_ids, matrix = feature_matrix(profile_artifact)
    standardized, means, stds = _standardize(matrix)
    if method == "kmeans":
        clustering = kmeans(standardized, **params)
    elif method == "hdbscan":
        clustering = hdbscan(standardized, **params)
    else:
        raise ValueError(f"Unknown method {method!r}; choose 'kmeans' or 'hdbscan'")

    centroids = clustering["centroids"]
    memberships: dict[str, list[str]] = {}
    cluster_info: dict[str, dict] = {}
    for cluster_id, members in clustering["memberships"].items():
        memberships[cluster_id] = [record_ids[index] for index in members]
        label = int(cluster_id.split("-", 1)[1])
        if centroids is not None:
            representative = record_ids[_nearest_to_center(members, standardized, centroids[label])]
        else:
            representative = record_ids[_medoid(members, standardized)]
        cluster_info[cluster_id] = {
            "size": len(members),
            "representative_record_id": representative,
        }

    source_json = json.dumps([entry["feature"] for entry in profile_artifact], ensure_ascii=False)
    artifact: dict = {
        "battery_version": BATTERY_VERSION,
        "battery_sha256": battery_sha256(),
        "feature_names": list(FEATURE_NAMES),
        "standardization": {
            "means": [float(value) for value in means],
            "stds": [float(value) for value in stds],
        },
        "method": clustering["method"],
        "params": clustering["params"],
        "n_records": len(matrix),
        "source_sha256": digest(source_json),
        "clusters": cluster_info,
        "memberships": memberships,
    }
    if clustering["noise"]:
        artifact["noise_record_ids"] = [record_ids[index] for index in clustering["noise"]]
    return artifact
