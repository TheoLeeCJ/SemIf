"""Compare shared typed readout with compact ordered-array generation."""

from __future__ import annotations

import argparse
import hashlib
import json
import statistics
import time
from pathlib import Path

from openjev_phase1.core import load_causal_model
from openjev_phase1.shared import score_shared


class TimelineStreamer:
    def __init__(self, tokenizer, started: float):
        self.tokenizer = tokenizer
        self.started = started
        self.initial = True
        self.events = []

    def put(self, value):
        if self.initial:  # generate() first sends the input IDs
            self.initial = False
            return
        elapsed = time.perf_counter() - self.started
        for token_id in value.detach().cpu().reshape(-1).tolist():
            self.events.append(
                {
                    "seconds": elapsed,
                    "token_id": token_id,
                    "text": self.tokenizer.decode([token_id], skip_special_tokens=False),
                }
            )

    def end(self):
        pass


def compact_messages(state: str, rows: list[dict]) -> list[dict]:
    request = {
        "state": state,
        "criteria_in_output_order": [row["question"] for row in rows],
        "allowed_values": ["yes", "no"],
    }
    return [
        {
            "role": "system",
            "content": (
                "Apply each criterion independently to the state. Return exactly one JSON array "
                "containing one lowercase yes or no string per criterion, in the supplied order. "
                "Emit no keys, confidence values, markdown, or explanation."
            ),
        },
        {"role": "user", "content": json.dumps(request, ensure_ascii=False)},
    ]


def run_generation(model, tokenizer, state: str, rows: list[dict], max_new_tokens: int) -> dict:
    import torch

    started = time.perf_counter()
    prompt = tokenizer.apply_chat_template(
        compact_messages(state, rows),
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False,
    )
    inputs = tokenizer(prompt, return_tensors="pt").to(next(model.parameters()).device)
    input_tokens = int(inputs["input_ids"].shape[-1])
    streamer = TimelineStreamer(tokenizer, started)
    with torch.inference_mode():
        output = model.generate(
            **inputs,
            do_sample=False,
            max_new_tokens=max_new_tokens,
            streamer=streamer,
            use_cache=True,
        )
    if next(model.parameters()).device.type == "cuda":
        torch.cuda.synchronize()
    total = time.perf_counter() - started
    generated = output[0, input_tokens:].detach().cpu().tolist()
    text = tokenizer.decode(generated, skip_special_tokens=True)
    parsed = None
    try:
        candidate = json.loads(text)
        if (
            isinstance(candidate, list)
            and len(candidate) == len(rows)
            and all(choice in {"yes", "no"} for choice in candidate)
        ):
            parsed = candidate
    except json.JSONDecodeError:
        pass
    return {
        "prompt_sha256": hashlib.sha256(prompt.encode()).hexdigest(),
        "input_tokens": input_tokens,
        "output_tokens": len(generated),
        "time_to_first_token_seconds": streamer.events[0]["seconds"] if streamer.events else None,
        "total_seconds": total,
        "output_text": text,
        "valid_complete_array": parsed is not None,
        "choices": parsed,
        "timeline": streamer.events,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--revision", required=True)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--max-new-tokens", type=int, default=128)
    args = parser.parse_args()
    if args.output.exists() or args.repeats < 1 or args.max_new_tokens < 1:
        parser.error("Output must be new and numeric limits must be positive")
    rows = [json.loads(line) for line in args.input.read_text().splitlines() if line.strip()][:21]
    if len(rows) != 21 or len({row["state"] for row in rows}) != 1:
        parser.error("Input must begin with one complete 21-question shared-state group")

    model, tokenizer, metadata = load_causal_model(args.model, args.revision, "cuda")
    import torch

    # Warm both paths; warmup is excluded from every reported duration.
    score_shared(model, tokenizer, rows, metadata)
    warmup = run_generation(model, tokenizer, rows[0]["state"], rows[:1], 16)
    if not warmup["timeline"]:
        raise RuntimeError("Generation warmup emitted no token events")

    direct_runs = []
    direct_outputs = None
    for _ in range(args.repeats):
        torch.cuda.reset_peak_memory_stats()
        direct_outputs, timing = score_shared(model, tokenizer, rows, metadata)
        direct_runs.append({**timing, "peak_cuda_bytes": torch.cuda.max_memory_allocated()})

    generation_runs = []
    for _ in range(args.repeats):
        torch.cuda.reset_peak_memory_stats()
        run = run_generation(model, tokenizer, rows[0]["state"], rows, args.max_new_tokens)
        run["peak_cuda_bytes"] = torch.cuda.max_memory_allocated()
        generation_runs.append(run)

    direct_choices = [
        row["option_ids"][max(range(len(row["probabilities"])), key=row["probabilities"].__getitem__)]
        for row in direct_outputs
    ]
    complete = [run for run in generation_runs if run["choices"] is not None]
    agreement = None
    if complete:
        agreement = sum(a == b for a, b in zip(direct_choices, complete[0]["choices"])) / len(rows)
    report = {
        "version": "decision-vs-compact-generation-v2",
        "model": metadata,
        "hardware": torch.cuda.get_device_name(0),
        "input": {
            "path": str(args.input),
            "sha256": hashlib.sha256(args.input.read_bytes()).hexdigest(),
            "states": 1,
            "criteria": 21,
        },
        "scope": (
            "Same frozen model, exact state, criteria, hardware and process. Direct path returns "
            "two-option distributions; generation emits only an ordered yes/no JSON array."
        ),
        "direct_parallel": {
            "runs": direct_runs,
            "median_total_seconds": statistics.median(run["total_seconds"] for run in direct_runs),
            "outputs": direct_outputs,
        },
        "compact_generation": {
            "prompt_messages": compact_messages(rows[0]["state"], rows),
            "runs": generation_runs,
            "median_total_seconds": statistics.median(run["total_seconds"] for run in generation_runs),
            "median_output_tokens": statistics.median(run["output_tokens"] for run in generation_runs),
            "all_runs_valid_complete_arrays": all(run["valid_complete_array"] for run in generation_runs),
            "agreement_with_direct_argmax_first_run": agreement,
        },
    }
    report["median_wall_ratio"] = (
        report["compact_generation"]["median_total_seconds"]
        / report["direct_parallel"]["median_total_seconds"]
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, ensure_ascii=False, allow_nan=False) + "\n")
    print(
        json.dumps(
            {
                "direct_median_seconds": report["direct_parallel"]["median_total_seconds"],
                "compact_generation_median_seconds": report["compact_generation"]["median_total_seconds"],
                "ratio": report["median_wall_ratio"],
                "median_output_tokens": report["compact_generation"]["median_output_tokens"],
                "valid": report["compact_generation"]["all_runs_valid_complete_arrays"],
                "agreement": agreement,
            }
        )
    )


if __name__ == "__main__":
    main()
