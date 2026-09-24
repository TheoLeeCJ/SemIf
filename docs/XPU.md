# Intel Arc (XPU)

The Torch backend runs SemIf's direct, serial-prefix reuse, and parallel
shared-state decision modes on an Intel Arc discrete GPU. CUDA stays the
default backend. With `--device auto` (the default), the loader picks
CUDA when CUDA is available, and tries an Intel GPU only when no CUDA
device is available. Add `--device xpu` to require the Intel GPU
directly, the same way `--device mps` requires Apple Silicon; it
does not fall back to CUDA even when a CUDA device is present. This
applies to any Arc card with enough VRAM to hold the model. The
validation below used one Intel Arc A770 with 16 GB of VRAM.

## Install

Install the XPU build of PyTorch first, from its own package index. Then
install the project:

```bash
pip install torch==2.10.0+xpu --index-url https://download.pytorch.org/whl/xpu
pip install -e '.[test]'
```

The validated versions are `torch==2.10.0+xpu` and `transformers==5.17.0`,
on Python 3.10. Use the same model revisions as the CUDA path. The `xpu`
extra in `pyproject.toml` records this torch pin. It does not fetch from
the separate wheel index; run the two commands above in this order.

## Select one device

Set `ONEAPI_DEVICE_SELECTOR` to expose exactly one Intel GPU per scorer
process:

```bash
export ONEAPI_DEVICE_SELECTOR=level_zero:1
```

Use the index for your own GPU. The index can differ between machines and
between driver updates. Do not assume the index from another guide or from
a past run. Check the device name first:

```bash
ONEAPI_DEVICE_SELECTOR=level_zero:1 python -c \
  "import torch; [print(i, torch.xpu.get_device_properties(i).name) for i in range(torch.xpu.device_count())]"
```

An explicit CUDA or XPU request requires exactly one visible device of
that type. CUDA errors name `CUDA_VISIBLE_DEVICES`; XPU errors name
`ONEAPI_DEVICE_SELECTOR`.

## The long-forward workaround

On XPU, one long forward pass can corrupt the last-token logits. The
failure starts near 1813 input tokens on the Arc A770 test system, on the
Qwen3.5 hybrid architecture. The cause is upstream in PyTorch XPU, not in
this project.

The direct, serial, and shared scorers all split a long XPU forward into
short steps of 1024 tokens. Each step reuses the model's own KV cache.
Each row uses the logits at its last real token, including shared rows
that end before the final chunk. CPU tests compare chunked and full
forwards within floating-point tolerance. The workaround runs on XPU only. It does
not change the CUDA or CPU path in any way.

## Reproducing the evidence: `benchmarks/xpu_benchmark.py`

`benchmarks/shape777.py`, `benchmarks/shape777_reranker.py`, and
`benchmarks/decision_vs_generation.py` are the published CUDA benchmark
runners; they stay CUDA-only, the same way they stay CUDA-only for Apple
Silicon (see [Apple Silicon](APPLE_SILICON.md), which keeps those same
three scripts CUDA-only for MPS too). XPU evidence instead has its own
dedicated script, following the same precedent the MLX backend already
set with `benchmarks/mlx_benchmark.py`: a separate runner with its own
result schema and its own `peak_xpu_bytes` memory field, instead of a
`peak_cuda_bytes` field reused across backends.

```bash
python benchmarks/xpu_benchmark.py --suite shape --output results/xpu/my-a770-shape
python benchmarks/xpu_benchmark.py --suite generation --output results/xpu/my-a770-generation
python benchmarks/xpu_benchmark.py --suite all --output results/xpu/my-a770-run
```

The committed direct and generation evidence predates this runner. It was
produced by earlier XPU-enabled versions of the published benchmark
scripts and retains their original schemas, including `peak_cuda_bytes`.
The current CUDA-only scripts cannot reproduce those XPU runs, and this
runner writes a different schema. The bundle has not been regenerated
with the current code; validating it requires a new Arc hardware run.

Each run requires a new output directory. The runner records:

- **Shape:** all 777 decisions in `fresh`, `serial_prefix`, and
  `parallel_shared` modes, including per-state latency, peak XPU
  allocation, every choice change against the run's own `fresh` mode
  (`vs_fresh`), and every choice change against the published CUDA
  `results/raw/shape777-direct.predictions.jsonl` rows (`vs_published_cuda`).
- **Generation:** three repetitions comparing direct shared-state scoring
  against the same model writing a compact yes/no array, using
  `--prefill-chunk-size 512 --attention eager` by default (both
  overridable) to avoid the two defects described above. Records raw
  output, validity, first-token timing, completion timing, and agreement
  with direct.

The loader is called with `device="xpu"` explicitly, so a run only
proceeds on the Intel GPU; it does not fall back to CUDA. Like
`mlx_benchmark.py`, this runner has no reranker suite: reranker mode is
CUDA-only for XPU too (`semif-score --mode reranker` rejects `--device
xpu`), matching the precedent both MPS and MLX already set.

## Reranker drift on XPU

The committed `results/xpu/` evidence bundle records one reranker
benchmark run from before this repository settled on the CUDA-only
reranker precedent (see above); it predates `xpu_benchmark.py`, which does
not reproduce it. Its choices drift from the published NVIDIA choices.
The reranker readout compares two large yes/no logits, near 17 in
magnitude. Their difference is small, often between 0.1 and 1.0. Small
BF16 rounding differences can flip that small difference's sign. A CPU
FP32 reference matched the published NVIDIA choices, not the A770
choices. This points to readout sensitivity, not a port bug. Treat that
evidence as historical and informational only, not a matched
reproduction, and not something `semif-score` or `xpu_benchmark.py` will
reproduce today.

## What this port validates

Validated: direct, serial, and shared scoring reach drift grade against the
published NVIDIA rows. Between 3 and 7 of 777 choices differ, with a
maximum probability difference of 0.1131. The published CUDA fresh/serial
comparison has 5 choice changes; this single comparison does not establish
a same-GPU drift range. Compact generation with the two settings above
reproduced the published 21-item array exactly.

Blocked: reranker mode is CUDA-only, matching the MPS and MLX precedent.
The one committed reranker run on XPU is historical evidence from before
that decision and drifts from the published NVIDIA choices, for the
readout-sensitivity reason above.

Untested: other Arc cards, multiple GPUs, Windows, and torch versions other
than `2.10.0+xpu`. The Arc B580 is expected to work but was not the
validation target.

## Checks

Run these from the repository root, in an environment installed with
`pip install -e '.[test]'`. No GPU is needed:

```bash
pytest -q
(cd results/raw && sha256sum -c SHA256SUMS)
python benchmarks/verify_published.py
```

Run this check against the committed A770 evidence bundle:

```bash
(cd results/xpu && sha256sum -c SHA256SUMS)
```

See [the A770 evidence](../results/xpu/README.md) for the machine, the
dates, and the claim boundaries behind the numbers above.
