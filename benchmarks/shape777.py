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
from semif_phase1.direct import score as torch_score
from semif_phase1.serial import SerialPrefixScorer as TorchSerialPrefixScorer
from semif_phase1.shared import score_shared as torch_score_shared


def _gpu_name() -> str | None:
    """The NVIDIA device name without importing a CUDA-enabled torch."""
    for info in sorted(Path("/proc/driver/nvidia/gpus").glob("*/information")):
        for line in info.read_text().splitlines():
            if line.startswith("Model:"):
                return line.split(":", 1)[1].strip()
    return None


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--revision", required=True)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-tokens", type=int, default=4096)
    parser.add_argument("--backend", choices=("torch", "llamacpp"), default="torch")
    parser.add_argument("--gguf", type=Path, help="Local GGUF checkpoint for --backend llamacpp")
    parser.add_argument("--llama-threads", type=int, help="CPU threads for --backend llamacpp")
    parser.add_argument("--llama-gpu-layers", type=int, default=0,
                        help="Layers offloaded to the GPU for --backend llamacpp (default: 0)")
    parser.add_argument("--llama-parallel", type=int, default=1,
                        help="Branch slots for llama.cpp shared mode (default: 1 = state restore)")
    args = parser.parse_args()
    if args.output.exists():
        parser.error("Output must be new")
    if args.backend == "llamacpp":
        if args.gguf is None or not args.gguf.is_file():
            parser.error("--backend llamacpp requires --gguf pointing at an existing GGUF file")
        if args.llama_gpu_layers < 0 or args.llama_parallel < 1:
            parser.error("--llama-gpu-layers must be nonnegative and --llama-parallel positive")
    elif args.gguf is not None or args.llama_threads is not None or args.llama_gpu_layers or args.llama_parallel != 1:
        parser.error("llama.cpp options require --backend llamacpp")
    rows = [json.loads(line) for line in args.input.read_text().splitlines() if line.strip()]
    groups = defaultdict(list)
    for row in rows:
        groups[row["group_id"]].append(row)
    if len(rows) != 777 or len(groups) != 37 or any(len(group) != 21 for group in groups.values()):
        parser.error("Expected the committed 37-state x 21-question fixture")
    if args.backend == "llamacpp":
        from semif_phase1 import llamacpp_backend

        model, tokenizer, metadata = llamacpp_backend.load_model(
            args.model, args.revision, args.gguf, threads=args.llama_threads,
            context_tokens=args.max_tokens, gpu_layers=args.llama_gpu_layers,
            sequences=args.llama_parallel)
        score = llamacpp_backend.score
        SerialPrefixScorer = llamacpp_backend.SerialPrefixScorer
        score_shared = llamacpp_backend.score_shared
        cuda = None
        hardware = (_gpu_name() or "unknown GPU") if args.llama_gpu_layers else "CPU"
    else:
        model, tokenizer, metadata = load_causal_model(args.model, args.revision, "cuda")
        import torch as cuda

        score, SerialPrefixScorer, score_shared = torch_score, TorchSerialPrefixScorer, torch_score_shared
        hardware = cuda.cuda.get_device_name(0)

    first = next(iter(groups.values()))
    score(model, tokenizer, first[0], metadata, args.max_tokens)
    warm_serial = SerialPrefixScorer(model, tokenizer, metadata, args.max_tokens)
    for row in first:
        warm_serial.score(row)
    score_shared(model, tokenizer, first, metadata, args.max_tokens)
    report = {
        "version": "shape777-published-v1",
        "input_sha256": hashlib.sha256(args.input.read_bytes()).hexdigest(),
        "model": metadata,
        "hardware": hardware,
        "backend": args.backend,
        "timing_scope": "Warm model; includes prompt construction, tokenization, transfers, forward passes and CPU readout.",
        "results": [],
    }
    predictions = {}
    for mode in ("fresh", "serial_prefix", "parallel_shared"):
        if cuda is not None:
            cuda.cuda.reset_peak_memory_stats()
        started = time.perf_counter()
        values, state_times = [], []
        for group in groups.values():
            mark = time.perf_counter()
            if mode == "fresh":
                values.extend(score(model, tokenizer, row, metadata, args.max_tokens) for row in group)
            elif mode == "serial_prefix":
                scorer = SerialPrefixScorer(model, tokenizer, metadata, args.max_tokens)
                values.extend(scorer.score(row) for row in group)
            else:
                scored, _ = score_shared(model, tokenizer, group, metadata, args.max_tokens)
                values.extend(scored)
            state_times.append(time.perf_counter() - mark)
        elapsed = time.perf_counter() - started
        predictions[mode] = values
        report["results"].append(
            {
                "mode": mode,
                "wall_seconds": elapsed,
                "decisions_per_second": len(values) / elapsed,
                "state_p50_seconds": statistics.median(state_times),
                "peak_cuda_bytes": cuda.cuda.max_memory_allocated() if cuda is not None else None,
            }
        )
    reference = {row["id"]: row for row in predictions["fresh"]}
    report["comparisons_to_fresh"] = {}
    for mode in ("serial_prefix", "parallel_shared"):
        flips, maximum = [], 0.0
        for row in predictions[mode]:
            old = reference[row["id"]]
            maximum = max(
                maximum, *(abs(a - b) for a, b in zip(old["probabilities"], row["probabilities"]))
            )
            if max(range(len(old["probabilities"])), key=old["probabilities"].__getitem__) != max(
                range(len(row["probabilities"])), key=row["probabilities"].__getitem__
            ):
                flips.append(row["id"])
        report["comparisons_to_fresh"][mode] = {
            "max_probability_difference": maximum,
            "argmax_flips": flips,
        }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    args.output.with_suffix(".predictions.jsonl").write_text(
        "".join(
            json.dumps({"mode": mode, **row}, allow_nan=False) + "\n"
            for mode, values in predictions.items()
            for row in values
        )
    )
    print(json.dumps(report["results"]))


if __name__ == "__main__":
    main()
