"""Option-order sensitivity helpers and opt-in order stabilization.

Letter slots (A/B/C/…) are positional; semantic option IDs are not. These helpers
permute display order, align logits back by option id, and either measure
sensitivity or average across K permutations before softmax.

Stabilized scoring is a separate readout: it does not change the default
``direct-options-v1`` contract or any frozen quality table.
"""

from __future__ import annotations

import copy
import itertools
import math
import random
import time
from typing import Callable, Sequence

from .core import softmax

STABILIZE_PROMPT_VERSION = "direct-options-stabilize-order-v1"
STABILIZE_READOUT = (
    "K-permutation average of native option-slot logits aligned by semantic id, "
    "then softmax; cost is K forward passes per decision"
)


def check_stabilize_k(k: int) -> int:
    if not isinstance(k, int) or isinstance(k, bool) or k < 2:
        raise ValueError(f"--stabilize-order requires an integer K >= 2, got {k!r}")
    return k


def apply_permutation(row: dict, permutation: Sequence[int]) -> dict:
    """Return a deep copy of ``row`` with options reordered by ``permutation`` indices."""
    options = row["options"]
    if sorted(permutation) != list(range(len(options))):
        raise ValueError(f"Permutation must be a rearrangement of 0..{len(options) - 1}")
    altered = copy.deepcopy(row)
    altered["options"] = [options[index] for index in permutation]
    if "label" in altered and isinstance(altered["label"], int) and not isinstance(altered["label"], bool):
        gold_id = options[altered["label"]]["id"]
        altered["label"] = [option["id"] for option in altered["options"]].index(gold_id)
    return altered


def select_permutations(n_options: int, k: int, *, seed: int = 0) -> list[tuple[int, ...]]:
    """Choose ``k`` distinct display permutations, always including identity and reverse when possible.

    Deterministic. When ``n_options!`` is at most ``k``, returns every permutation
    (identity first). Otherwise samples without replacement after locking identity
    and the reversed order.
    """
    check_stabilize_k(k)
    if n_options < 2:
        raise ValueError("Need at least two options to permute")
    identity = tuple(range(n_options))
    reversed_order = tuple(reversed(range(n_options)))
    all_perms = list(itertools.permutations(range(n_options)))
    if k >= len(all_perms):
        # Identity first for stable readout metadata.
        rest = [perm for perm in all_perms if perm != identity]
        return [identity, *rest]
    chosen: list[tuple[int, ...]] = [identity]
    if reversed_order != identity and k >= 2:
        chosen.append(reversed_order)
    remaining = [perm for perm in all_perms if perm not in chosen]
    rng = random.Random(seed)
    rng.shuffle(remaining)
    while len(chosen) < k:
        chosen.append(remaining.pop())
    return chosen


def align_values_by_id(
    source_ids: Sequence[str],
    values: Sequence[float],
    canonical_ids: Sequence[str],
) -> list[float]:
    """Reorder ``values`` so they match ``canonical_ids`` order."""
    if len(source_ids) != len(values):
        raise ValueError("source_ids and values length mismatch")
    if len(set(source_ids)) != len(source_ids):
        raise ValueError("Duplicate source option ids")
    if set(source_ids) != set(canonical_ids):
        raise ValueError("Option id sets differ; cannot align")
    index = {option_id: position for position, option_id in enumerate(source_ids)}
    return [float(values[index[option_id]]) for option_id in canonical_ids]


def average_aligned(vectors: Sequence[Sequence[float]]) -> list[float]:
    if not vectors:
        raise ValueError("Need at least one vector to average")
    width = len(vectors[0])
    if width < 2 or any(len(vector) != width for vector in vectors):
        raise ValueError("Aligned vectors must share the same length (>= 2)")
    if any(not math.isfinite(value) for vector in vectors for value in vector):
        raise ValueError("Non-finite values in aligned vectors")
    scale = len(vectors)
    return [sum(column) / scale for column in zip(*vectors)]


def total_variation(left: Sequence[float], right: Sequence[float]) -> float:
    if len(left) != len(right):
        raise ValueError("Distribution lengths differ")
    return 0.5 * sum(abs(a - b) for a, b in zip(left, right))


def chosen_id(option_ids: Sequence[str], probabilities: Sequence[float]) -> str:
    return option_ids[max(range(len(probabilities)), key=probabilities.__getitem__)]


def distribution_map(option_ids: Sequence[str], probabilities: Sequence[float]) -> dict[str, float]:
    if len(option_ids) != len(probabilities) or len(set(option_ids)) != len(option_ids):
        raise ValueError("Invalid option_ids/probabilities pair")
    return dict(zip(option_ids, probabilities))


def report_pair_sensitivity(
    base: dict,
    variant: dict,
    *,
    base_display_ids: Sequence[str] | None = None,
    variant_display_ids: Sequence[str] | None = None,
) -> dict:
    """Compare one base prediction to one order-perturbed prediction, aligned by option id."""
    left = distribution_map(base["option_ids"], base["probabilities"])
    right = distribution_map(variant["option_ids"], variant["probabilities"])
    if set(left) != set(right):
        raise ValueError(f"Semantic option IDs changed for {base.get('id')}")
    ids = list(base["option_ids"])
    aligned_right = [right[option_id] for option_id in ids]
    base_choice = chosen_id(base["option_ids"], base["probabilities"])
    variant_choice = chosen_id(variant["option_ids"], variant["probabilities"])
    base_pos = list(base_display_ids or base["option_ids"]).index(base_choice)
    variant_pos = list(variant_display_ids or variant["option_ids"]).index(variant_choice)
    return {
        "base_id": base["id"],
        "variant_id": variant["id"],
        "from": base_choice,
        "to": variant_choice,
        "flipped": base_choice != variant_choice,
        "total_variation": total_variation([left[i] for i in ids], aligned_right),
        "base_display_position": base_pos,
        "variant_display_position": variant_pos,
    }


def summarize_sensitivity(rows: Sequence[dict]) -> dict:
    """Aggregate pair-level sensitivity rows into flips, TV, and position bias."""
    if not rows:
        raise ValueError("No sensitivity rows to summarize")
    flips = [row for row in rows if row["flipped"]]
    tvs = [row["total_variation"] for row in rows]
    n_options = 1 + max(max(row["base_display_position"], row["variant_display_position"]) for row in rows)
    base_wins = [0] * n_options
    variant_wins = [0] * n_options
    for row in rows:
        base_wins[row["base_display_position"]] += 1
        variant_wins[row["variant_display_position"]] += 1
    scale = len(rows)
    return {
        "rows": scale,
        "argmax_flips": len(flips),
        "flip_rate": len(flips) / scale,
        "flip_rows": [
            {"base_id": row["base_id"], "variant_id": row["variant_id"], "from": row["from"], "to": row["to"]}
            for row in flips
        ],
        "mean_total_variation": sum(tvs) / scale,
        "max_total_variation": max(tvs),
        "position_bias": {
            "meaning": (
                "Fraction of rows whose argmax lands on each display letter slot "
                "(index 0 = A). Compared before vs after the order perturbation."
            ),
            "base_position_win_rates": [count / scale for count in base_wins],
            "variant_position_win_rates": [count / scale for count in variant_wins],
            "first_position_delta": (variant_wins[0] - base_wins[0]) / scale,
        },
    }


def stabilize_from_permutation_results(
    canonical_ids: Sequence[str],
    permutation_results: Sequence[dict],
    *,
    permutations: Sequence[Sequence[int]],
    k: int,
    seed: int,
    base_result: dict | None = None,
) -> dict:
    """Average logits from already-scored permutations, then softmax in canonical order."""
    if len(permutation_results) != len(permutations):
        raise ValueError("Need one result per permutation")
    aligned = [
        align_values_by_id(result["option_ids"], result["option_logits"], canonical_ids)
        for result in permutation_results
    ]
    mean_logits = average_aligned(aligned)
    started = time.perf_counter() if base_result is None else None
    payload = {
        "id": (base_result or permutation_results[0])["id"],
        "option_ids": list(canonical_ids),
        "probabilities": softmax(mean_logits),
        "option_logits": mean_logits,
        "prompt_version": STABILIZE_PROMPT_VERSION,
        "readout": STABILIZE_READOUT,
        "probability_status": "conditional option score; uncalibrated as decision confidence",
        "stabilize_order": {
            "k": k,
            "seed": seed,
            "cost_multiplier": k,
            "permutations": [list(perm) for perm in permutations],
            "component_prompt_sha256": [result.get("prompt_sha256") for result in permutation_results],
            "component_prompt_version": [
                result.get("prompt_version") for result in permutation_results
            ],
        },
    }
    if base_result is not None:
        for key in ("input_tokens", "model", "forward_seconds", "total_seconds",
                    "cache_hit", "prefix_tokens", "encode_seconds"):
            if key in base_result:
                payload[key] = base_result[key]
        # Wall cost scales with K when each permutation is an independent forward.
        if "total_seconds" in base_result and isinstance(base_result["total_seconds"], (int, float)):
            payload["total_seconds"] = float(base_result["total_seconds"]) * k
            payload["stabilize_order"]["total_seconds_note"] = (
                "total_seconds estimates K * first-permutation wall time"
            )
        payload["prompt_sha256"] = base_result.get("prompt_sha256")
    else:
        payload["total_seconds"] = time.perf_counter() - started
    return payload


def score_with_stabilized_order(
    score_row: Callable[[dict], dict],
    row: dict,
    *,
    k: int,
    seed: int = 0,
) -> dict:
    """Score ``k`` option permutations via ``score_row``, average aligned logits, softmax.

    ``score_row`` must accept one decision dict and return a prediction carrying
    ``option_ids`` and ``option_logits`` (any backend / stub).
    """
    k = check_stabilize_k(k)
    canonical_ids = [option["id"] for option in row["options"]]
    permutations = select_permutations(len(canonical_ids), k, seed=seed)
    results = []
    for permutation in permutations:
        results.append(score_row(apply_permutation(row, permutation)))
    return stabilize_from_permutation_results(
        canonical_ids,
        results,
        permutations=permutations,
        k=k,
        seed=seed,
        base_result=results[0],
    )
