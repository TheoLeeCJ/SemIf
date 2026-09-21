# Apple Silicon / MLX

The native MLX backend runs SemIf's direct, serial-prefix, and parallel-shared
decision modes on macOS arm64. It uses MLX-LM's Qwen3.5 implementation and the
same prompts and answer-token checks as the Torch backend. No answer token is
generated. The browser demo is a separate implementation.

## Install and score

Use an isolated Python environment on an Apple Silicon Mac with Metal available:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -e '.[test,mlx]'

semif-score --backend mlx --mode direct \
  --model Qwen/Qwen3.5-4B \
  --revision 851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a \
  --input examples/decisions.jsonl \
  --output results-mlx-direct.jsonl
```

The first run downloads the pinned checkpoint into the Hugging Face cache.
Weights are about 9 GB on disk; GPU execution needs additional memory. The
source checkpoint includes vision weights, which MLX-LM's native sanitizer
excludes from the text model. The baseline preserves source precision (BF16
with some FP32 parameters). Runtime pins are MLX 0.32.2 and MLX-LM at
`a63e24c389382619eb6d9af656e3b46024be217a` (package version 0.32.0).
The source pin includes the upstream Qwen recurrent q/k normalization fix;
release 0.31.3 applies the L2 epsilon incorrectly. Install requires Git.
Each prediction records the installed runtime's source commit.

Use `--mode serial` to reuse consecutive identical states. Use `--mode shared`
when every row in the input has the same exact state and a unique decision ID.
The shared mode evaluates all supplied questions in one batch; memory grows
with batch size and suffix length. Input limits are enforced without truncation.

`--mlx-bits 8` or `--mlx-bits 4` applies deterministic affine quantization in
memory, with group size 64. It starts from the same pinned checkpoint and does
not write another model artifact. Results record this transformation and hashes
of source files. Quantization changes the model's probabilities and must be
evaluated separately against the unquantized MLX run. Local model directories require a revision label and are
hashed too. Remote revisions must be immutable 40-character commit IDs.

The default backend remains Torch/CUDA. MLX reranker mode is explicitly
unsupported. Installations on other platforms can continue using the existing
Torch paths without importing MLX. The MLX extra is specific to macOS arm64;
it retains the original Torch dependencies for the precision-probe reference.
The standalone CUDA installation uses the `torch` extra. Keep Ryzen AI in a
separate environment because its NumPy and Transformers versions differ.

The loader caps MLX's inactive allocation cache at 256 MiB by default.
Use `--mlx-cache-limit-mib 512` to change it, or `--mlx-cache-limit-mib 0`
to disable inactive allocation caching. The benchmark and precision-probe
scripts accept the same flag. Python callers can pass `cache_limit_mib=512`
to `mlx_backend.load_model`; prediction metadata records the effective limit
in bytes. This is a process-wide MLX allocator setting, not the prefix cache. MLX's default cache
can otherwise retain almost all system RAM across variable-length prompts,
which is unsuitable when other local models share unified memory. This bounds
the allocation cache, not the active model or batch memory requirement.

## Apple Silicon demo

![Native MLX CLI on an Apple M5 Max](media/openjev-mlx.png)

[Replayable terminal recording and capture details](media/README.md).
The screenshot shows the completed recording of a real local CLI run under
the former OpenJev name. Current commands use `semif-score`.

## Cache correctness

Qwen3.5 combines attention history with recurrent convolution/delta state.
Serial scoring deep-copies the entire native prefix cache for each question.
Parallel scoring merges native cache copies, right-pads question suffixes,
supplies their real lengths to recurrent caches, and reads each suffix's last
real token. A branch's state is never fed into another question.

The retained cache is keyed by exact prefix token IDs: a caller mutating a
previously supplied JSON object cannot accidentally reuse a stale cache. Tests exercise a
small real Qwen3.5 hybrid model, variable-length suffixes, question reordering,
repeat calls, state changes, and invalid inputs without downloading weights.

GPU arithmetic differs across kernels and prompt/batch shapes. The pilot and
review thresholds are frozen in `manifests/mlx-validation.json`. Every changed
choice is reported; typed output does not guarantee semantic correctness, and
softmax scores are not calibrated confidence.

## Compressed retained evidence

Large historical reports are losslessly gzipped to keep the contribution small.
Verification and reference-run readers accept `.gz` files transparently; new
benchmark runs keep the original plain JSON/JSONL format. Original payload
checksums and reproduction details are in [the evidence index](../results/mlx/README.md).

## Reproduce the evidence

Run from the repository root with the environment activated. Every output
directory must be new; interrupted runs remain as partial evidence rather than
being overwritten. Run GPU benchmarks one process at a time.

```bash
python benchmarks/mlx_benchmark.py --suite diagnostic --output results/mlx/my-pilot
python benchmarks/mlx_benchmark.py --suite all --output results/mlx/my-bf16
python benchmarks/mlx_benchmark.py --suite quantization --bits 8 --reference-run results/mlx/my-bf16 --output results/mlx/my-q8
python benchmarks/mlx_benchmark.py --suite quantization --bits 4 --reference-run results/mlx/my-bf16 --output results/mlx/my-q4
```

The runner records:

- **Diagnostic:** exact tokenizer equivalence; fresh, serial, parallel, and
  reordered scores for a small pilot.
- **Quality:** the 144 authored and 108 perturbation cases, existing evaluator
  metrics, missing-evidence behavior, and row-level differences from published
  Torch predictions. Those published predictions used serial prefix reuse;
  this comparison therefore includes backend and execution-shape differences.
- **Systems:** all 777 decisions in fresh, serial, and parallel modes, including
  per-state latency, peak MLX allocation, and every choice change against fresh.
- **Generation:** three repetitions comparing 21 direct distributions against
  the same model writing a compact yes/no array. Records raw output, validity,
  first-token timing, completion timing, and agreement. An invalid generated
  answer is a failure, not an equivalent faster/slower answer.

The quantization suite runs the diagnostic, quality, and generation comparisons.
Use `--suite all --bits 8` (or `4`) to additionally run all 777 decisions through
all three modes for that precision.

Model loading, artifact hashing, and initial warmup are outside timed regions.
Execution measurements include prompt rendering, tokenization, evaluation,
synchronization, and CPU readout. MLX is lazy: evaluating arrays and waiting for
GPU completion is essential. Peak MLX allocation includes model weights and
temporary arrays; it is not total macOS process memory or directly equivalent
to CUDA's allocator metric. CUDA and Mac timings describe different hardware.

## Validation

```bash
pytest -q
(cd results/raw && shasum -a 256 -c SHA256SUMS)
(cd results/mlx && shasum -a 256 -c SHA256SUMS)
python benchmarks/mlx_evidence.py results/mlx/UNCOMPRESSED_SHA256SUMS
python benchmarks/verify_published.py
python benchmarks/verify_mlx.py results/mlx/my-bf16
```

The original CUDA summary and evidence are preserved. Mac measurements live in
their own dated directories under `results/mlx/`. See the [measured results and
run history](../results/mlx/README.md), including numerical differences and
superseded experiments.

For a deeper investigation of probability differences, run the FP32 diagnostic
after the timed GPU benchmarks have finished:

```bash
python benchmarks/mlx_precision_probe.py \
  --run results/mlx/my-bf16 --output results/mlx/my-bf16/precision-probe.json
```

This selects quality rows above the frozen probability review threshold and
any changed choices. It compares native BF16 and FP32 MLX execution with a
PyTorch CPU FP32 reference, including a separate diagnostic for rounding when
MLX-LM folds Qwen's offset RMSNorm weights. It does not alter production model
loading or substitute diagnostic results for the benchmark.

## Runtime references

- [Pinned Qwen3.5 implementation](https://github.com/ml-explore/mlx-lm/blob/a63e24c389382619eb6d9af656e3b46024be217a/mlx_lm/models/qwen3_5.py)
- [Pinned native cache APIs](https://github.com/ml-explore/mlx-lm/blob/a63e24c389382619eb6d9af656e3b46024be217a/mlx_lm/models/cache.py)
- [Upstream normalization fix](https://github.com/ml-explore/mlx-lm/commit/a63e24c389382619eb6d9af656e3b46024be217a)
- [MLX lazy evaluation](https://ml-explore.github.io/mlx/build/html/usage/lazy_evaluation.html)
