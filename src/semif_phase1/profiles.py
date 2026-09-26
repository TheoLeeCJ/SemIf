"""Frozen semantic-probe battery and feature extraction for semantic-profile clustering.

Torch-free and numpy-free: probe rows are ordinary scorer rows, and aggregation only
reads committed prediction dicts. Feature vectors use a frozen coordinate order; any
change to the battery or coordinates requires bumping BATTERY_VERSION.
"""

from __future__ import annotations

import hashlib
import json
import math

from .core import digest, validate_row

BATTERY_VERSION = "semantic-profiles-v1"

OPTION_SCHEMA = (
    {"id": "affirm", "description": "The probe statement holds for the evidence."},
    {"id": "deny", "description": "The probe statement does not hold for the evidence."},
    {"id": "insufficient", "description": "The evidence does not establish either."},
)

PROBE_BATTERY = (
    {"axis": "billing", "question": "Assess the probe: this request is about billing, charges, or payment."},
    {"axis": "account_access", "question": "Assess the probe: this request is about account access or login."},
    {"axis": "urgency", "question": "Assess the probe: this request is urgent or time-sensitive."},
    {"axis": "refund", "question": "Assess the probe: this request asks for a refund or reimbursement."},
    {"axis": "security", "question": "Assess the probe: this request is about security or suspicious activity."},
)

AXES = tuple(probe["axis"] for probe in PROBE_BATTERY)
FEATURE_NAMES = ("affirm_mean", "deny_mean", "insufficient_mean", "entropy_mean")

SEPARATOR = "::"


def battery_sha256() -> str:
    payload = {
        "version": BATTERY_VERSION,
        "probes": list(PROBE_BATTERY),
        "options": list(OPTION_SCHEMA),
        "features": list(FEATURE_NAMES),
    }
    return digest(json.dumps(payload, ensure_ascii=False))


def _state_finite(state) -> None:
    try:
        json.dumps(state, ensure_ascii=False, allow_nan=False)
    except (TypeError, ValueError) as error:
        raise ValueError("state must be finite JSON-compatible data") from error


def probe_rows(record: dict) -> list[dict]:
    """Build scorer rows for one record, one per frozen probe axis.

    All rows share the identical ``state``, so a caller can score them together
    with a single state prefill (e.g. ``shared.score_shared``).
    """
    if not isinstance(record, dict) or "id" not in record or "state" not in record:
        raise ValueError("record must be a dict with 'id' and 'state' fields")
    if not isinstance(record["id"], str) or not record["id"]:
        raise ValueError("record id must be a nonempty string")
    _state_finite(record["state"])

    rows = []
    for probe in PROBE_BATTERY:
        rows.append({
            "id": f"{record['id']}{SEPARATOR}{probe['axis']}",
            "state": record["state"],
            "question": probe["question"],
            "options": list(OPTION_SCHEMA),
        })
    return rows


def _option_ids(options: list[dict]) -> list[str]:
    return [option["id"] for option in options]


def _scoring_option_ids(scoring: dict) -> list[str]:
    """Option ids from native scorer output (option_ids) or committed rows (options)."""
    if "option_ids" in scoring:
        ids = scoring["option_ids"]
        if not isinstance(ids, list) or not all(isinstance(value, str) for value in ids):
            raise ValueError("option_ids must be a list of option-id strings")
        return list(ids)
    options = scoring.get("options")
    if options is None:
        raise ValueError("scoring needs 'option_ids' (native scorer output) or 'options' (committed row)")
    return _option_ids(options)


def _entropy(probabilities: list[float]) -> float:
    """Shannon entropy (base 2) of a probability vector."""
    total = 0.0
    for probability in probabilities:
        if probability > 0:
            total -= probability * math.log2(probability)
    return total


def features_for_record(record_id: str, probe_scorings: list[dict]) -> dict:
    """Build one frozen-order feature vector from scored probe rows for one record.

    Accepts either scoring shape: native scorer output (``option_ids`` as a list of
    option-id strings, as returned by ``direct.score`` and ``score_shared``) or
    committed rows (``options`` as a list of option dicts).
    Strictness: every scored row id must match exactly one expected ``record_id::axis``.
    Unknown ids, missing axes, duplicate axes, wrong option schema/order, or
    non-finite/non-normalized probabilities all raise loudly. This prevents silent
    coordinate misalignment when a record's scored options differ from the frozen schema.
    """
    expected = {f"{record_id}{SEPARATOR}{axis}" for axis in AXES}
    seen: dict[str, dict] = {}
    incomplete: list[str] = []

    for scoring in probe_scorings:
        row_id = scoring["id"]
        if row_id not in expected:
            raise ValueError(f"Unexpected scored row id {row_id!r} for record {record_id!r}")
        if row_id in seen:
            raise ValueError(f"Duplicate scored row id {row_id!r} for record {record_id!r}")

        probabilities = scoring["probabilities"]
        if _scoring_option_ids(scoring) != [option["id"] for option in OPTION_SCHEMA]:
            raise ValueError(f"{row_id}: option ids differ from frozen schema")
        if len(probabilities) != len(OPTION_SCHEMA):
            raise ValueError(f"{row_id}: probability length differs from option count")
        if not all(isinstance(p, (int, float)) and math.isfinite(p) for p in probabilities):
            raise ValueError(f"{row_id}: non-finite probabilities")
        total = sum(probabilities)
        if not math.isclose(total, 1.0, abs_tol=1e-6):
            raise ValueError(f"{row_id}: probabilities do not sum to 1 (got {total})")

        seen[row_id] = {
            "affirm": probabilities[0],
            "deny": probabilities[1],
            "insufficient": probabilities[2],
            "entropy": _entropy(probabilities),
        }

    missing = expected - set(seen)
    if missing:
        incomplete.append(f"record {record_id}: missing scored axes {sorted(missing)}")
    if incomplete:
        raise ValueError("Incomplete profile; " + "; ".join(incomplete))

    ordered = [seen[f"{record_id}{SEPARATOR}{axis}"] for axis in AXES]
    feature = [
        sum(entry["affirm"] for entry in ordered) / len(ordered),
        sum(entry["deny"] for entry in ordered) / len(ordered),
        sum(entry["insufficient"] for entry in ordered) / len(ordered),
        sum(entry["entropy"] for entry in ordered) / len(ordered),
    ]

    return {
        "record_id": record_id,
        "feature": feature,
        "feature_names": list(FEATURE_NAMES),
        "probe_probabilities": [
            {
                "axis": axis,
                "affirm": seen[f"{record_id}{SEPARATOR}{axis}"]["affirm"],
                "deny": seen[f"{record_id}{SEPARATOR}{axis}"]["deny"],
                "insufficient": seen[f"{record_id}{SEPARATOR}{axis}"]["insufficient"],
                "entropy": seen[f"{record_id}{SEPARATOR}{axis}"]["entropy"],
            }
            for axis in AXES
        ],
        "battery_version": BATTERY_VERSION,
        "battery_sha256": battery_sha256(),
        "probe_count": len(ordered),
    }


def detect_record_id(scored_id: str) -> str:
    """Return the record id portion of a scored probe row id."""
    if SEPARATOR not in scored_id:
        raise ValueError(f"{scored_id!r} is not a valid scored probe row id")
    return scored_id[: scored_id.index(SEPARATOR)]


def aggregate(scorings: list[dict]) -> list[dict]:
    """Group scored probe rows by record and return one feature artifact per record.

    Raises if any record has incomplete axes, listing all incomplete records.
    """
    by_record: dict[str, list[dict]] = {}
    for scoring in scorings:
        record_id = detect_record_id(scoring["id"])
        by_record.setdefault(record_id, []).append(scoring)

    artifacts = []
    incomplete: list[str] = []
    for record_id, record_scorings in by_record.items():
        try:
            artifacts.append(features_for_record(record_id, record_scorings))
        except ValueError as error:
            incomplete.append(str(error))
    if incomplete:
        raise ValueError("Incomplete profiles; " + "; ".join(incomplete))
    return artifacts


def score_status(scoring: dict) -> str:
    """Echo the probability_status provenance string from a scoring, if present."""
    return scoring.get("probability_status", "unlabeled profile scoring")


def scoring_record_id(scoring: dict) -> str:
    """Return the record id carried by a single scored probe row."""
    return detect_record_id(scoring["id"])
