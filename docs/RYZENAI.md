# Ryzen AI NPU backend

Run SemIf's direct scorer on Windows with AMD Ryzen AI Software **1.8.0** and
its **OGA 0.14.0** runtime. The backend uses the existing JSONL input and output
format and reads native next-token logits for the declared options. It does
not generate an answer token. The supported mode is `direct`, with 2–16 options;
serial/shared cache reuse and reranker mode are rejected before model loading.

The model must be a local AMD NPU OGA deployment artifact, rather than a
Transformers checkpoint. These are the pinned Qwen3-4B models used for this
integration:

| AMD artifact | Revision | Context limit |
|---|---|---:|
| [Full Fusion 4K](https://huggingface.co/amd/Qwen3-4B_rai_1.8.0_npu_4K) | `d6fb03663d78ae5034d4594bfe9d92b35a5e213a` | 4096 |
| [Token Fusion 16K](https://huggingface.co/amd/Qwen3-4B_rai_1.8.0_npu_16K) | `715d60818350b685ca2af3566e5ae38f4780daf0` | 16384 |

The configured RyzenAI provider selects NPU execution. AMD's CPU graph
components, including host/prefill/LM-head work, remain part of the official
deployment; this is not a claim that every operator executes on the NPU. GPU
provider configurations are rejected. Consult AMD's
[OGA flow](https://ryzenai.docs.amd.com/en/latest/hybrid_oga.html) for supported
hardware, drivers, and the distinction between Full Fusion and Token Fusion.

## Installation

Install Ryzen AI Software 1.8.0 and the matching NPU driver using AMD's
[installation guide](https://ryzenai.docs.amd.com/en/latest/inst.html).
Models for earlier Ryzen AI releases are incompatible with the 1.8 runtime.
Use the AMD-provided wheels; similarly named public PyPI packages are not
substitutes for these runtime builds.

Create a separate Python 3.12 environment from the SDK interpreter. The
`ryzenai` extra pins NumPy 1.26.4 for the AMD bindings' NumPy 1.x ABI and
Transformers 4.57.6 for reference chat rendering. The `torch` and `mlx` extras
retain the original NumPy 2.2.6 / Transformers 5.17.0 pins and belong in
separate environments. `requirements.txt` describes the published CUDA run,
so use the following installation command for NPU execution instead.

From the repository root, install SemIf and AMD's wheels together so pip can
resolve their dependencies. Torch/Torchvision below match AMD's SDK environment;
SemIf does not use Torch to execute this backend.

```powershell
conda activate ryzen-ai-1.8.0
python -m venv .venv
$sdkDir = 'C:\Program Files\RyzenAI\1.8.0'
$wheelNames = @(
  'onnxruntime_genai_directml_ryzenai-0.14.0-py3-none-win_amd64.whl'
  'onnxruntime_vitisai-1.27.0-py3-none-win_amd64.whl'
  'onnxruntime_providers_ryzenai-1.8.0-py3-none-win_amd64.whl'
  'ryzenai_dynamic_dispatch-1.8.0-py3-none-win_amd64.whl'
  'voe-1.8.0-py3-none-win_amd64.whl'
)
$wheels = $wheelNames | ForEach-Object { Join-Path $sdkDir $_ }
& .\.venv\Scripts\python.exe -m pip install -e '.[test,ryzenai]' `
  'torch==2.4.1' 'torchvision==0.19.1' @wheels
& .\.venv\Scripts\python.exe -m pip check
```

This installs into `.venv`; it does not install the SDK or change machine-wide
environment variables. For a non-default SDK location, adjust `$sdkDir` and
set `$env:RYZEN_AI_INSTALLATION_PATH = $sdkDir` before invoking the CLI directly.
The launcher below uses the default SDK directory and restores the process
environment variable when it exits.

## Score the owned examples

Download the pinned model snapshot into its own directory:

```powershell
& .\.venv\Scripts\hf.exe download amd/Qwen3-4B_rai_1.8.0_npu_4K `
  --revision d6fb03663d78ae5034d4594bfe9d92b35a5e213a `
  --local-dir models\Qwen3-4B-npu-4k
.\run_semif_npu.ps1
```

The launcher sets `--backend ryzenai-npu --mode direct`, reads
`examples/decisions.jsonl`, and creates a new ignored output in `cache/results/`.
To choose explicit paths:

```powershell
.\run_semif_npu.ps1 `
  -Model 'models\Qwen3-4B-npu-4k' `
  -Revision 'd6fb03663d78ae5034d4594bfe9d92b35a5e213a' `
  -Input 'examples\decisions.jsonl' `
  -Output 'cache\results\npu-example.jsonl'
```

The equivalent CLI is:

```powershell
$env:RYZEN_AI_INSTALLATION_PATH = 'C:\Program Files\RyzenAI\1.8.0'
& .\.venv\Scripts\semif-score.exe --backend ryzenai-npu --mode direct `
  --model models\Qwen3-4B-npu-4k `
  --revision d6fb03663d78ae5034d4594bfe9d92b35a5e213a `
  --input examples\decisions.jsonl --output cache\results\npu-cli-example.jsonl
```

Existing output files are never overwritten. Model loading and artifact
hashing happen before per-row timing, and first inference can include kernel
preparation. Each result records the model revision, runtime versions, local
artifact SHA-256 hashes, token count, prompt hash, answer-token IDs, native
option logits, and probabilities normalized over those options. These are
uncalibrated conditional option scores, not generative probability estimates
or guaranteed decision confidence. The README's CUDA quality/speed results
refer to their original model and hardware, not this Qwen3 NPU deployment.

## Long context

The 16K artifact is a separate compiled Token Fusion model. Editing the 4K
artifact's JSON limits does not extend it. Download the pinned 16K snapshot and
select it explicitly:

```powershell
& .\.venv\Scripts\hf.exe download amd/Qwen3-4B_rai_1.8.0_npu_16K `
  --revision 715d60818350b685ca2af3566e5ae38f4780daf0 `
  --local-dir models\Qwen3-4B-npu-16k
.\run_semif_npu.ps1 `
  -Model 'models\Qwen3-4B-npu-16k' `
  -Revision '715d60818350b685ca2af3566e5ae38f4780daf0' `
  -MaxTokens 16384
```

The effective input limit is the minimum of the requested limit and the
model/configured KV-cache limits. With chunking enabled, the chunk size is
validated separately from the total context. Direct scoring reserves zero
output tokens and rejects oversized prompts without truncation. Both
tokenizers must agree on the full prompt, every single-token answer label,
and the prompt/label boundary before inference starts.

## Validation

The NPU unit tests mock the OGA import boundary and run without the SDK or a
model download. For the full repository checks from an installed environment:

```powershell
& .\.venv\Scripts\python.exe -m pytest -q
Push-Location results/raw
& 'C:\Program Files\Git\usr\bin\sha256sum.exe' -c SHA256SUMS
Pop-Location
& .\.venv\Scripts\python.exe benchmarks/verify_published.py
```

Actual NPU execution requires compatible hardware and AMD's runtime; the
commands above do not substitute for running the example scorer on that host.
