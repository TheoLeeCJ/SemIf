"""Reproduce reranker pair batching on the owned 37x21 fixture."""

from __future__ import annotations

import argparse
from collections import defaultdict
import hashlib
import json
import statistics
import time
from pathlib import Path

from openjev_phase1.core import (
    describe_hardware,
    load_causal_model,
    peak_memory_bytes,
    reset_peak_memory_stats,
    softmax,
)
from openjev_phase1.reranker import score_pair_batch


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
    parser.add_argument("--pair-batch-sizes", default="1,4,8")
    parser.add_argument("--max-tokens", type=int, default=4096)
    parser.add_argument(
        "--device",
        choices=("auto", "cuda", "mps", "cpu"),
        default="auto",
        help="Accelerator to use; auto selects CUDA, then Apple Metal (MPS), then CPU.",
    )
    parser.add_argument(
        "--dtype",
        choices=("auto", "bfloat16", "float16", "float32"),
        default="auto",
        help="Model dtype; auto prefers bfloat16 with fallback off CUDA.",
    )
    args = parser.parse_args()
    if args.output.exists():
        parser.error("Output must be new")
    sizes = [int(value) for value in args.pair_batch_sizes.split(",")]
    if not sizes or min(sizes) < 1:
        parser.error("Pair batch sizes must be positive")
    rows = [json.loads(line) for line in args.input.read_text().splitlines() if line.strip()]
    groups = defaultdict(list)
    for row in rows:
        groups[row["group_id"]].append(row)
    if len(rows) != 777 or len(groups) != 37 or any(len(group) != 21 for group in groups.values()):
        parser.error("Expected the committed 37-state x 21-question fixture")
    model, tokenizer, metadata = load_causal_model(
        args.model,
        args.revision,
        device=None if args.device == "auto" else args.device,
        dtype=None if args.dtype == "auto" else args.dtype,
    )

    warm = next(iter(groups.values()))[0]
    score_pair_batch(model, tokenizer, [(warm, option) for option in warm["options"]], args.max_tokens)
    device = next(model.parameters()).device
    report = {
        "version": "shape777-reranker-published-v1",
        "input_sha256": hashlib.sha256(args.input.read_bytes()).hexdigest(),
        "model": metadata,
        "hardware": describe_hardware(),
        "device": device.type,
        "semantic_contract": (
            "Two independent yes/no relevance passes per binary decision; "
            "option log-odds normalized only for relative comparison."
        ),
        "results": [],
    }
    prediction_lines = []
    for size in sizes:
        reset_peak_memory_stats(device)
        started = time.perf_counter()
        state_times, predictions = [], []
        forward_seconds = padded_tokens = 0
        for group in groups.values():
            state_started = time.perf_counter()
            specs = [(row, option) for row in group for option in row["options"]]
            scored = []
            for offset in range(0, len(specs), size):
                batch, timing = score_pair_batch(
                    model, tokenizer, specs[offset : offset + size], args.max_tokens
                )
                scored.extend(batch)
                forward_seconds += timing["forward_seconds"]
                padded_tokens += timing["padded_tokens"]
            cursor = 0
            for row in group:
                count = len(row["options"])
                values = [item["log_odds"] for item in scored[cursor : cursor + count]]
                cursor += count
                predictions.append(
                    {
                        "id": row["id"],
                        "option_ids": [option["id"] for option in row["options"]],
                        "probabilities": softmax(values),
                        "option_logits": values,
                    }
                )
            state_times.append(time.perf_counter() - state_started)
        elapsed = time.perf_counter() - started
        peak = peak_memory_bytes(device)
        record = {
            "pair_batch_size": size,
            "judgments": len(predictions),
            "wall_seconds": elapsed,
            "forward_seconds": forward_seconds,
            "judgments_per_second": len(predictions) / elapsed,
            "state_latency_p50_seconds": statistics.median(state_times),
            "state_latency_p95_seconds": percentile(state_times, 0.95),
            "padded_tokens": padded_tokens,
            "peak_cuda_bytes": peak if device.type == "cuda" else None,
            "peak_memory_bytes": peak,
        }
        report["results"].append(record)
        prediction_lines.extend({"pair_batch_size": size, **row} for row in predictions)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    args.output.with_suffix(".predictions.jsonl").write_text(
        "".join(json.dumps(row, allow_nan=False) + "\n" for row in prediction_lines)
    )
    print(json.dumps(report["results"]))


if __name__ == "__main__":
    main()
