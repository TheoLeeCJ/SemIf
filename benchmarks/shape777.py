"""Reproduce fresh versus parallel shared-state scoring on the owned 37x21 fixture."""

from __future__ import annotations

import argparse
from collections import defaultdict
import hashlib
import json
import statistics
import time
from pathlib import Path

from semif_phase1.core import load_causal_model
from semif_phase1.direct import score
from semif_phase1.serial import SerialPrefixScorer
from semif_phase1.shared import score_shared

REPORT_VERSION = "shape777-direct-v1"
TIMING_SCOPE = (
    "Warm loaded model; complete mode wall time includes tokenization, H2D, forward "
    "and D2H; excludes model load and output write."
)
COMPARISON_TOLERANCE = 1.0
COMPARISON_NOTE = (
    "Tolerance is deliberately non-gating; report exact probability drift and all "
    "argmax flips."
)
COMMON_FIELDS = (
    "judgments",
    "states",
    "wall_seconds",
    "judgments_per_second",
    "state_latency_p50_seconds",
    "state_latency_p95_seconds",
    "peak_cuda_bytes",
)
RESULT_FIELDS = {
    "fresh_batch1": COMMON_FIELDS,
    "serial_prefix": (*COMMON_FIELDS, "cache_hits"),
    "parallel_suffix": (*COMMON_FIELDS, "padded_suffix_tokens", "true_suffix_tokens"),
}
COMPARISON_FIELDS = (
    "passed",
    "max_probability_difference",
    "mean_probability_difference",
    "argmax_flips",
    "tolerance",
)
# The report labels the unbatched mode as fresh_batch1 while its prediction rows
# were committed as fresh, so the two vocabularies are kept distinct.
MODES = ("fresh_batch1", "serial_prefix", "parallel_suffix")
PREDICTION_LABELS = {
    "fresh_batch1": "fresh",
    "serial_prefix": "serial_prefix",
    "parallel_suffix": "parallel_suffix",
}
TOP_LEVEL_FIELDS = (
    "version",
    "input_sha256",
    "model",
    "hardware",
    "max_tokens",
    "timing_scope",
    "results",
    "comparisons_to_fresh",
)


def percentile(values, fraction):
    ordered = sorted(values)
    index = (len(ordered) - 1) * fraction
    lower = int(index)
    return ordered[lower] + (
        ordered[min(lower + 1, len(ordered) - 1)] - ordered[lower]
    ) * (index - lower)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--revision", required=True)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-tokens", type=int, default=4096)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("Output must be new")
    rows = [json.loads(line) for line in args.input.read_text().splitlines() if line.strip()]
    groups = defaultdict(list)
    for row in rows:
        groups[row["group_id"]].append(row)
    if len(rows) != 777 or len(groups) != 37 or any(len(group) != 21 for group in groups.values()):
        parser.error("Expected the committed 37-state x 21-question fixture")
    model, tokenizer, metadata = load_causal_model(args.model, args.revision, "cuda")
    import torch

    first = next(iter(groups.values()))
    score(model, tokenizer, first[0], metadata, args.max_tokens)
    warm_serial = SerialPrefixScorer(model, tokenizer, metadata, args.max_tokens)
    for row in first:
        warm_serial.score(row)
    score_shared(model, tokenizer, first, metadata, args.max_tokens)
    report = {
        "version": REPORT_VERSION,
        "input_sha256": hashlib.sha256(args.input.read_bytes()).hexdigest(),
        "model": metadata,
        "hardware": torch.cuda.get_device_name(0),
        "max_tokens": args.max_tokens,
        "timing_scope": TIMING_SCOPE,
        "results": [],
    }
    predictions = {}
    for mode in MODES:
        torch.cuda.reset_peak_memory_stats()
        started = time.perf_counter()
        values, state_times = [], []
        cache_hits = 0
        padded_suffix_tokens = 0
        true_suffix_tokens = 0
        for group in groups.values():
            mark = time.perf_counter()
            if mode == "fresh_batch1":
                values.extend(score(model, tokenizer, row, metadata, args.max_tokens) for row in group)
            elif mode == "serial_prefix":
                scorer = SerialPrefixScorer(model, tokenizer, metadata, args.max_tokens)
                for row in group:
                    result = scorer.score(row)
                    cache_hits += bool(result["cache_hit"])
                    values.append(result)
            else:
                scored, timing = score_shared(model, tokenizer, group, metadata, args.max_tokens)
                values.extend(scored)
                padded_suffix_tokens += timing["padded_suffix_tokens"]
                true_suffix_tokens += timing["true_suffix_tokens"]
            state_times.append(time.perf_counter() - mark)
        elapsed = time.perf_counter() - started
        predictions[mode] = values
        record = {
            "mode": mode,
            "judgments": len(values),
            "states": len(groups),
            "wall_seconds": elapsed,
            "judgments_per_second": len(values) / elapsed,
            "state_latency_p50_seconds": statistics.median(state_times),
            "state_latency_p95_seconds": percentile(state_times, 0.95),
            "peak_cuda_bytes": torch.cuda.max_memory_allocated(),
        }
        if mode == "serial_prefix":
            record["cache_hits"] = cache_hits
        elif mode == "parallel_suffix":
            record["padded_suffix_tokens"] = padded_suffix_tokens
            record["true_suffix_tokens"] = true_suffix_tokens
        report["results"].append(record)
    reference = {row["id"]: row for row in predictions[MODES[0]]}
    comparisons = {}
    for mode in MODES[1:]:
        flips, maximum, total, cells = [], 0.0, 0.0, 0
        for row in predictions[mode]:
            old = reference[row["id"]]
            differences = [abs(a - b) for a, b in zip(old["probabilities"], row["probabilities"])]
            maximum = max(maximum, *differences)
            total += sum(differences)
            cells += len(differences)
            if max(range(len(old["probabilities"])), key=old["probabilities"].__getitem__) != max(
                range(len(row["probabilities"])), key=row["probabilities"].__getitem__
            ):
                flips.append(row["id"])
        comparisons[mode] = {
            # Non-gating by design, so a pass can only mean an exact reproduction.
            "passed": not flips and maximum == 0.0,
            "max_probability_difference": maximum,
            "mean_probability_difference": total / cells,
            "argmax_flips": flips,
            "tolerance": COMPARISON_TOLERANCE,
        }
    comparisons["note"] = COMPARISON_NOTE
    report["comparisons_to_fresh"] = comparisons
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    args.output.with_suffix(".predictions.jsonl").write_text(
        "".join(
            json.dumps({"mode": PREDICTION_LABELS[mode], **row}, allow_nan=False) + "\n"
            for mode in MODES
            for row in predictions[mode]
        )
    )
    print(json.dumps(report["results"]))


if __name__ == "__main__":
    main()
