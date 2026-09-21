# llama.cpp GGUF backend

Linux (and any host with a llama.cpp shared library) can score the same
`direct-options-v1` prompts from a local GGUF file. Torch still loads native
BF16 only. MLX `--mlx-bits` stays Apple Silicon. This path is quantized GGUF
via llama.cpp last-position logits; it is not the published BF16 ladder
(0.813 / 0.766 / 0.845).

The forward pass is `llama_decode` + `llama_get_logits_ith`. No answer token is
generated. Prompt construction and A–P slot checks stay on the pinned
transformers tokenizer, so `prompt_sha256` matches the Torch backend when the
GGUF vocabulary agrees.

## Install and score

You need a llama.cpp build that exports `libllama` (CUDA for GPU offload). The
pip `llama-cpp-python` CUDA wheel is not used: on AVX2-only CPUs the 0.3.35
cu132 wheel SIGILLs in `ggml_cpu_init`, and its ctypes structs do not match
current `llama_model_params`.

Point the scorer at `libllama.so`:

```bash
export SEMIF_LLAMA_LIB=$HOME/.unsloth/llama.cpp/build/bin/libllama.so.0
# or pass --llama-lib

semif-score --mode direct --backend llamacpp \
  --model Qwen/Qwen3.5-4B \
  --revision 851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a \
  --gguf /path/to/Qwen_Qwen3.5-4B-Q4_K_M.gguf \
  --llama-gpu-layers 99 \
  --input examples/decisions.jsonl \
  --output results-gguf-direct.jsonl
```

`--llama-gpu-layers` defaults to 99. Use `0` for CPU-only scoring, `-1` for
every layer. `--llama-threads` caps CPU threads. `--llama-lib` overrides
`SEMIF_LLAMA_LIB` / `LLAMA_CPP_LIB` / `LLAMA_CPP_LIB_PATH`. If none are set,
the loader also looks for the Unsloth CUDA build under
`~/.unsloth/llama.cpp/build/bin/libllama.so.0`.

Reranker mode stays Torch. Direct, serial, and shared modes are supported.
Qwen3.5 hybrid memory cannot copy or partially drop sequences, so serial and
shared restore a whole-sequence llama.cpp slot-cache snapshot.

Each result records the GGUF path, size, sha256, `n_gpu_layers`, the loaded
`libllama` path, and an explicit uncalibrated-probability warning. llama.cpp
ignores Qwen3.5 MTP tensors (`blk.32.nextn.*`); that is expected.

A live GGUF pytest is gated on `SEMIF_LLAMACPP_GGUF` and does not download
weights in CI.
