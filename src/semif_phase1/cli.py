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
    parser.add_argument("--backend", choices=("torch", "mlx", "llamacpp"), default="torch")
    parser.add_argument("--mlx-bits", type=int, choices=(4, 8), help="Quantize MLX weights in memory; default preserves source precision")
    parser.add_argument("--mlx-cache-limit-mib", type=int,
                        help="MLX inactive allocation cache in MiB (default: 256; 0 disables caching)")
    parser.add_argument("--gguf", type=Path, help="Local GGUF checkpoint for --backend llamacpp")
    parser.add_argument("--llama-threads", type=int,
                        help="CPU threads for --backend llamacpp (default: all visible cores)")
    parser.add_argument("--llama-gpu-layers", default="auto",
                        help="GPU layers for --backend llamacpp: 'auto' keeps llama.cpp's default (every layer "
                             "when the library can offload), 0 forces CPU, N offloads N layers")
    parser.add_argument("--llama-parallel", default="auto",
                        help="Shared-mode branching for --backend llamacpp: 'auto' sizes each fan-out from the "
                             "state's token counts over a unified KV buffer; N fixes n_seq_max; 1 restores "
                             "state per decision")
    parser.add_argument("--llama-readout", choices=("last", "marginal"), default="last",
                        help="llama.cpp shared-mode readout: last prompt position, or answer-slot masses summed "
                             "over one-token preambles (needs --llama-parallel > 1)")
    parser.add_argument("--model", required=True)
    parser.add_argument("--revision", required=True)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-tokens", type=int, default=4096)
    parser.add_argument("--device", choices=("auto", "cuda", "mps"), default="auto",
                        help="Torch backend GPU (auto prefers CUDA, then Apple MPS)")
    parser.add_argument("--dtype", choices=("bfloat16", "float16", "float32"), default="bfloat16",
                        help="Model precision; changing it can change option scores")
    args = parser.parse_args()
    if args.output.exists() or args.max_tokens < 1:
        parser.error("Output must be new and max-tokens must be positive")
    if args.mlx_bits and args.backend != "mlx":
        parser.error("--mlx-bits requires --backend mlx")
    if args.mlx_cache_limit_mib is not None:
        if args.backend != "mlx":
            parser.error("--mlx-cache-limit-mib requires --backend mlx")
        if args.mlx_cache_limit_mib < 0:
            parser.error("--mlx-cache-limit-mib must be nonnegative")
    if args.gguf is not None and args.backend != "llamacpp":
        parser.error("--gguf requires --backend llamacpp")
    if args.llama_threads is not None:
        if args.backend != "llamacpp":
            parser.error("--llama-threads requires --backend llamacpp")
        if args.llama_threads < 1:
            parser.error("--llama-threads must be positive")
    if args.backend == "mlx" and args.mode == "reranker":
        parser.error("MLX supports direct, serial, and shared modes; reranker requires torch")
    def _auto_or_int(name, value, minimum):
        if value == "auto":
            return "auto"
        try:
            number = int(value)
        except (TypeError, ValueError):
            parser.error(f"{name} must be 'auto' or an integer")
        if number < minimum:
            parser.error(f"{name} must be 'auto' or at least {minimum}")
        return number

    args.llama_gpu_layers = _auto_or_int("--llama-gpu-layers", args.llama_gpu_layers, -1)
    args.llama_parallel = _auto_or_int("--llama-parallel", args.llama_parallel, 1)
    for name, value in (("--llama-gpu-layers", args.llama_gpu_layers), ("--llama-parallel", args.llama_parallel)):
        if args.backend != "llamacpp" and value != "auto":
            parser.error(f"{name} requires --backend llamacpp")
    if args.llama_readout != "last" and args.backend != "llamacpp":
        parser.error("--llama-readout requires --backend llamacpp")
    if args.backend == "llamacpp":
        if args.mode == "reranker":
            parser.error("llama.cpp supports direct, serial, and shared modes; reranker requires torch")
        if args.gguf is None or not args.gguf.is_file():
            parser.error("--backend llamacpp requires --gguf pointing at an existing GGUF file")
        if args.llama_readout == "marginal" and (args.mode != "shared" or args.llama_parallel == 1):
            parser.error("--llama-readout marginal requires --mode shared and --llama-parallel other than 1")
    rows = [json.loads(line) for line in args.input.read_text().splitlines() if line.strip()]
    if not rows:
        parser.error("Input is empty")
    for row in rows:
        validate_row(row)
    direct, serial, shared = direct_score, SerialPrefixScorer, score_shared
    if args.backend == "mlx":
        from . import mlx_backend

        cache_limit_mib = (mlx_backend.DEFAULT_CACHE_LIMIT_MIB if args.mlx_cache_limit_mib is None
                           else args.mlx_cache_limit_mib)
        model, tokenizer, metadata = mlx_backend.load_model(
            args.model, args.revision, args.mlx_bits, cache_limit_mib=cache_limit_mib)
        direct, serial, shared = mlx_backend.score, mlx_backend.SerialPrefixScorer, mlx_backend.score_shared
    elif args.backend == "llamacpp":
        from . import llamacpp_backend

        model, tokenizer, metadata = llamacpp_backend.load_model(
            args.model, args.revision, args.gguf,
            threads=args.llama_threads, context_tokens=args.max_tokens,
            gpu_layers=args.llama_gpu_layers, sequences=args.llama_parallel,
            readout=args.llama_readout)
        direct, serial, shared = (llamacpp_backend.score, llamacpp_backend.SerialPrefixScorer,
                                  llamacpp_backend.score_shared)
    else:
        if args.mode == "reranker":
            if args.device == "mps":
                parser.error("Reranker mode requires CUDA; --device mps is unsupported")
            # Keep auto CUDA-only and let the loader enforce one visible GPU.
            args.device = "cuda"
        model, tokenizer, metadata = load_causal_model(args.model, args.revision, args.device, args.dtype)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x") as destination:
        if args.mode == "shared":
            results, timing = shared(model, tokenizer, rows, metadata, args.max_tokens)
            for result in results:
                destination.write(json.dumps({**result, "shared_timing": timing}, allow_nan=False) + "\n")
        elif args.mode == "serial":
            scorer = serial(model, tokenizer, metadata, args.max_tokens)
            for row in rows:
                destination.write(json.dumps(scorer.score(row), allow_nan=False) + "\n")
                destination.flush()
        else:
            scorer = direct if args.mode == "direct" else reranker_score
            for row in rows:
                destination.write(json.dumps(scorer(model, tokenizer, row, metadata, args.max_tokens), allow_nan=False) + "\n")
                destination.flush()


if __name__ == "__main__":
    main()
