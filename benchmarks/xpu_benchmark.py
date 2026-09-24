"""Create-only Intel Arc (XPU) evidence: 37x21 drift vs published CUDA rows, and compact generation.

Run from the repository root. Reuses the frozen 37x21 fixture and the
published CUDA row-level predictions; does not modify the CUDA evidence.
Each run requires a new output directory.

This is a dedicated script rather than a `--device xpu` flag on
benchmarks/shape777.py, benchmarks/shape777_reranker.py, or
benchmarks/decision_vs_generation.py, and it reports peak memory as
`peak_xpu_bytes` instead of `peak_cuda_bytes`. Those three scripts stay
CUDA-only, the same way they stayed CUDA-only for Apple Silicon: see
docs/APPLE_SILICON.md ("The reranker and published CUDA benchmark
runners remain CUDA-only") and benchmarks/mlx_benchmark.py, which is the
MLX backend's own dedicated script with its own `peak_mlx_bytes` field.
Reranker is out of scope for the same reason it is out of scope for MPS
and MLX: `semif-score --mode reranker` requires CUDA (see cli.py).
"""
from __future__ import annotations

import argparse
from collections import defaultdict
import hashlib
import json
from pathlib import Path
import platform
import statistics
import time

from decision_vs_generation import run_generation
from semif_phase1.artifacts import write_new_outputs
from semif_phase1.core import load_causal_model
from semif_phase1.direct import score
from semif_phase1.serial import SerialPrefixScorer
from semif_phase1.shared import score_shared

# results/raw/shape777-direct.predictions.jsonl names this mode "parallel_suffix";
# every other script in this repository names it "parallel_shared".
PUBLISHED_MODE_NAMES = {"fresh": "fresh", "serial_prefix": "serial_prefix", "parallel_shared": "parallel_suffix"}


def read_jsonl(path) -> list[dict]:
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]


def write(path: Path, data) -> None:
    write_new_outputs({path: json.dumps(data, indent=2, allow_nan=False) + "\n"})


def write_rows(path: Path, rows: list[dict]) -> None:
    write_new_outputs({path: "".join(json.dumps(row, allow_nan=False) + "\n" for row in rows)})


def choice(row: dict) -> str:
    return row["option_ids"][max(range(len(row["probabilities"])), key=row["probabilities"].__getitem__)]


def compare(reference: list[dict], candidates: list[dict]) -> dict:
    old = {row["id"]: row for row in reference}
    if old.keys() != {row["id"] for row in candidates}:
        raise ValueError("Comparison requires identical decision IDs")
    differences, flips = [], []
    for row in candidates:
        base = old[row["id"]]
        left = dict(zip(base["option_ids"], base["probabilities"]))
        right = dict(zip(row["option_ids"], row["probabilities"]))
        if left.keys() != right.keys():
            raise ValueError("Option IDs differ")
        differences.append(max(abs(left[key] - right[key]) for key in left))
        if choice(row) != choice(base):
            flips.append(row["id"])
    return {"rows": len(candidates), "argmax_flips": flips, "max_probability_difference": max(differences)}


def shape(model, tokenizer, metadata, output: Path) -> list[dict]:
    import torch

    rows = read_jsonl("benchmarks/data/shape777.jsonl")
    groups = defaultdict(list)
    for row in rows:
        groups[row["group_id"]].append(row)
    if len(rows) != 777 or len(groups) != 37 or any(len(group) != 21 for group in groups.values()):
        raise ValueError("Expected 37 groups of 21 decisions")
    published = defaultdict(list)
    for row in read_jsonl("results/raw/shape777-direct.predictions.jsonl"):
        published[row["mode"]].append(row)

    first = next(iter(groups.values()))
    score(model, tokenizer, first[0], metadata)
    warm_serial = SerialPrefixScorer(model, tokenizer, metadata)
    for row in first:
        warm_serial.score(row)
    score_shared(model, tokenizer, first, metadata)

    predictions, reports = {}, []
    for mode in ("fresh", "serial_prefix", "parallel_shared"):
        torch.xpu.reset_peak_memory_stats()
        started = time.perf_counter()
        values, state_times = [], []
        for group in groups.values():
            mark = time.perf_counter()
            if mode == "fresh":
                values.extend(score(model, tokenizer, row, metadata) for row in group)
            elif mode == "serial_prefix":
                scorer = SerialPrefixScorer(model, tokenizer, metadata)
                values.extend(scorer.score(row) for row in group)
            else:
                scored, _ = score_shared(model, tokenizer, group, metadata)
                values.extend(scored)
            state_times.append(time.perf_counter() - mark)
        elapsed = time.perf_counter() - started
        predictions[mode] = values
        write_rows(output / f"shape777-{mode}.jsonl", values)
        reports.append(
            {
                "mode": mode,
                "wall_seconds": elapsed,
                "decisions_per_second": len(values) / elapsed,
                "state_p50_seconds": statistics.median(state_times),
                "peak_xpu_bytes": torch.xpu.max_memory_allocated(),
            }
        )
        print(json.dumps(reports[-1]), flush=True)

    result = {
        "runs": reports,
        "vs_fresh": {
            mode: compare(predictions["fresh"], predictions[mode]) for mode in ("serial_prefix", "parallel_shared")
        },
        "vs_published_cuda": {
            mode: compare(published[PUBLISHED_MODE_NAMES[mode]], predictions[mode])
            for mode in ("fresh", "serial_prefix", "parallel_shared")
        },
    }
    write(output / "shape777.json", result)
    return reports


def generation(model, tokenizer, metadata, output: Path, prefill_chunk_size: int) -> dict:
    import torch

    rows = read_jsonl("benchmarks/data/shape777.jsonl")[:21]
    score_shared(model, tokenizer, rows, metadata)
    warmup = run_generation(model, tokenizer, rows[0]["state"], rows[:1], 16, prefill_chunk_size)
    if not warmup["timeline"]:
        raise RuntimeError("Generation warmup emitted no token events")

    direct_runs, generation_runs = [], []
    direct_outputs = None
    for _ in range(3):
        torch.xpu.reset_peak_memory_stats()
        direct_outputs, timing = score_shared(model, tokenizer, rows, metadata)
        direct_runs.append({**timing, "peak_xpu_bytes": torch.xpu.max_memory_allocated()})
        torch.xpu.reset_peak_memory_stats()
        run = run_generation(model, tokenizer, rows[0]["state"], rows, 128, prefill_chunk_size)
        run["peak_xpu_bytes"] = torch.xpu.max_memory_allocated()
        generation_runs.append(run)

    direct_choices = [choice(row) for row in direct_outputs]
    complete = [run for run in generation_runs if run["choices"] is not None]
    agreement = None
    if complete:
        agreement = sum(a == b for a, b in zip(direct_choices, complete[0]["choices"])) / len(rows)
    direct_median = statistics.median(run["total_seconds"] for run in direct_runs)
    generation_median = statistics.median(run["total_seconds"] for run in generation_runs)
    result = {
        "direct_runs": direct_runs,
        "generation_runs": generation_runs,
        "direct_median_seconds": direct_median,
        "generation_median_seconds": generation_median,
        "generation_over_direct_ratio": generation_median / direct_median,
        "all_generated_arrays_valid": all(run["valid_complete_array"] for run in generation_runs),
        "agreement_with_direct_argmax_first_run": agreement,
    }
    write(output / "generation.json", result)
    return {key: value for key, value in result.items() if not key.endswith("runs")}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="Qwen/Qwen3.5-4B")
    parser.add_argument("--revision", default="851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a")
    parser.add_argument("--dtype", choices=("bfloat16", "float16", "float32"), default="bfloat16")
    parser.add_argument(
        "--attention", choices=("sdpa", "eager"), default="eager",
        help="XPU decode has a separate defect under sdpa; docs/XPU.md recommends eager",
    )
    parser.add_argument(
        "--prefill-chunk-size", type=int, default=512,
        help="Keeps generate()'s internal prefill under the XPU long-forward corruption length",
    )
    parser.add_argument("--suite", choices=("shape", "generation", "all"), default="all")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.prefill_chunk_size < 1:
        parser.error("--prefill-chunk-size must be positive")
    args.output.mkdir(parents=True, exist_ok=False)

    model, tokenizer, metadata = load_causal_model(
        args.model, args.revision, device="xpu", dtype=args.dtype, attn_implementation=args.attention
    )
    import torch

    manifest = {
        "version": "xpu-benchmark-v1",
        "model": metadata,
        "hardware": torch.xpu.get_device_name(0),
        "os": platform.platform(),
        "python": platform.python_version(),
        "suite": args.suite,
        "prefill_chunk_size": args.prefill_chunk_size,
        "source_sha256": {
            str(path): hashlib.sha256(path.read_bytes()).hexdigest()
            for path in [
                Path("benchmarks/xpu_benchmark.py"),
                Path("benchmarks/decision_vs_generation.py"),
                *sorted(Path("src/semif_phase1").glob("*.py")),
            ]
        },
        "timing_scope": "Warm model; includes prompt construction, tokenization, transfers, forward passes and CPU readout.",
        "fixture_sha256": hashlib.sha256(Path("benchmarks/data/shape777.jsonl").read_bytes()).hexdigest(),
    }
    write(args.output / "manifest.json", manifest)
    summaries = {}
    suites = {"all": ("shape", "generation")}.get(args.suite, (args.suite,))
    for name, run in (("shape", shape), ("generation", generation)):
        if name not in suites:
            continue
        summaries[name] = (
            run(model, tokenizer, metadata, args.output, args.prefill_chunk_size)
            if name == "generation"
            else run(model, tokenizer, metadata, args.output)
        )
        print(json.dumps({name: summaries[name]}, default=str), flush=True)
    write(args.output / "summary.json", summaries)


if __name__ == "__main__":
    main()
