"""Create-only JSONL command line scorer."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from .core import load_causal_model, validate_row
from .direct import score as direct_score
from .reranker import score as reranker_score
from .serial import SerialPrefixScorer
from .shared import score_shared


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("direct", "serial", "shared", "reranker"), required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--revision", required=True)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
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
        help="Model dtype; auto prefers bfloat16 with float16/float32 fallback off CUDA.",
    )
    args = parser.parse_args()
    if args.output.exists() or args.max_tokens < 1:
        parser.error("Output must be new and max-tokens must be positive")
    rows = [json.loads(line) for line in args.input.read_text().splitlines() if line.strip()]
    if not rows:
        parser.error("Input is empty")
    for row in rows:
        validate_row(row)
    model, tokenizer, metadata = load_causal_model(
        args.model,
        args.revision,
        device=None if args.device == "auto" else args.device,
        dtype=None if args.dtype == "auto" else args.dtype,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x") as destination:
        if args.mode == "shared":
            results, timing = score_shared(model, tokenizer, rows, metadata, args.max_tokens)
            for result in results:
                destination.write(json.dumps({**result, "shared_timing": timing}, allow_nan=False) + "\n")
        elif args.mode == "serial":
            scorer = SerialPrefixScorer(model, tokenizer, metadata, args.max_tokens)
            for row in rows:
                destination.write(json.dumps(scorer.score(row), allow_nan=False) + "\n")
                destination.flush()
        else:
            scorer = direct_score if args.mode == "direct" else reranker_score
            for row in rows:
                destination.write(json.dumps(scorer(model, tokenizer, row, metadata, args.max_tokens), allow_nan=False) + "\n")
                destination.flush()


if __name__ == "__main__":
    main()
