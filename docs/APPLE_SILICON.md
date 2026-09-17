# Apple Silicon

OpenJev runs on Apple Silicon through two backends. The PyTorch backend executes on the
MPS GPU; the optional MLX backend runs the same pinned Qwen3.5-4B checkpoint natively on
Metal with no conversion or quantization. Published CUDA numbers in `results/` are
unaffected: both Apple backends are additive.

## Install

```bash
python -m venv .venv
. .venv/bin/activate
pip install -e '.[test]'
# MLX backend (Apple Silicon only):
pip install -e '.[mlx,test]'
```

## Run

```bash
# PyTorch/MPS:
openjev-score --mode direct --device mps \
  --model Qwen/Qwen3.5-4B --revision 851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a \
  --input examples/decisions.jsonl --output results-mps-direct.jsonl

# MLX (BF16, direct/serial/shared; reranker is CUDA-only):
openjev-score --backend mlx --mode direct \
  --model Qwen/Qwen3.5-4B --revision 851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a \
  --input examples/decisions.jsonl --output results-mlx-direct.jsonl
```

## Behavior and guarantees

- Both backends preserve the prompt contracts: identical chat template, token
  boundaries, and prompt hashes; scores carry the same uncalibrated-probability
  warnings as CUDA output.
- The MLX loader refuses custom Python model files, text-only configurations,
  quantized weights, and non-BF16 weights (FP32 is allowed only for the
  linear-attention decay and norm state parameters, verified by name).
  It records parameter dtypes plus `mlx`/`mlx-lm` versions in every row.
- Remote models require an immutable commit revision. Local checkpoints are trusted
  inputs: their revision is a user-supplied manifest label, not a verified Hub pin.
  Tokenizer chat templates come from the checkpoint and use Transformers' sandbox.
- Prefix reuse deep-copies the full hybrid cache (attention KV plus convolution and
  recurrent state) per branch; caches are never trimmed.
- Shared mode on both Apple backends prefills once, then forwards independent
  batch-1 suffixes rather than a parallel batch. CUDA keeps its batched path.
- Timing fields are device-synchronized on both backends. Shared timing separates
  prompt encoding, prefix prefill, cache replication/copying, and suffix forwards.
- The reranker and published CUDA benchmark runners remain CUDA-only. Use the
  `openjev-score` commands above for Apple Silicon.

## Measured on an M5 (24 GB), pinned Qwen3.5-4B, BF16

Measurements are illustrative, not committed benchmarks; they were taken 2026-09-17
and are hardware- and version-sensitive.

| Path | PyTorch MPS | MLX |
|---|---:|---:|
| Warm direct decision (~140-token prompt) | ~0.6 s | ~0.26 s |
| First forward (shader compilation) | ~8 s | ~7 s |
| One group from the 37×21 fixture (one prefill + 21 suffixes) | 16.4 s | 9.1 s |
| Peak allocation during direct scoring | — | ~8.6 GB |

Known limitations: PyTorch MPS falls back to reference kernels for Qwen3.5's
hybrid attention (`causal_conv1d`, `flash-linear-attention` are CUDA-only), so MPS
should not be compared directly against the committed RTX 3090 numbers. The MLX
backend currently supports unquantized BF16 checkpoints only; 4-bit quantization is
untested and would require its own accuracy validation.
