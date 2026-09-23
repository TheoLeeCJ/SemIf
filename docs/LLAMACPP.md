# llama.cpp / GGUF

The llama.cpp backend runs SemIf's direct, serial-prefix, and shared decision
modes over a quantized GGUF checkpoint, on CPU or with layers offloaded to a
GPU. Prompts, answer-slot checks and `prompt_sha256` come from the reference
transformers tokenizer, exactly as in the Torch backend; llama.cpp only executes
the forward pass. Every prompt is re-tokenized through the GGUF vocabulary and
must match the reference encoding before it is scored. No answer token is
generated.

## Install and score

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -e '.[test,llamacpp]'          # CPU wheel of llama-cpp-python

semif-score --backend llamacpp --mode direct \
  --model Qwen/Qwen3.5-4B \
  --revision 851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a \
  --gguf /path/to/Qwen_Qwen3.5-4B-Q4_K_M.gguf \
  --input examples/decisions.jsonl \
  --output results-llamacpp-direct.jsonl
```

The pinned tokenizer is downloaded into the Hugging Face cache on first use
(a few megabytes; the model weights come from the GGUF file). The GGUF used for
the results below is `bartowski/Qwen_Qwen3.5-4B-GGUF` at revision
`4168f45a16a1290d65a4ec0fa312ae917a4c15d6`, quantization Q4_K_M, 3 013 027 808
bytes, SHA-256 `13c16f426047e2de38cd075bdade4a7bcbc8c774384876f677740cda65f8a983`.
Each prediction records the file's size and checksum.

### GPU offload

`--llama-gpu-layers N` offloads `N` layers (use a large value such as `99` for
all of them). It needs a llama-cpp-python wheel built with a GPU backend; the
PyPI wheel is CPU-only. For CUDA:

```bash
pip install --force-reinstall --no-deps "llama-cpp-python==0.3.35" \
  --extra-index-url https://abetlen.github.io/llama-cpp-python/whl/cu124
pip install nvidia-cuda-runtime-cu12 nvidia-cublas-cu12 nvidia-cuda-nvrtc-cu12
export LD_LIBRARY_PATH=$(ls -d .venv/lib/python3.*/site-packages/nvidia/*/lib | tr '\n' ':')
```

The metadata records `n_gpu_layers` and whether the loaded library supports
offload at all, so a CPU wheel silently ignoring the flag is visible in the
output.

## Two ways to branch from one state

Sequence 0 of the llama.cpp context holds the prefilled state. Shared mode
scores every question of that state from a branch of it, and there are two
ways to make a branch:

| `--llama-parallel` | branching | decodes per state |
|---|---|---|
| `1` (default) | **state restore** — `llama_state_seq_get_data` after the prefill, `set_data` before each question | one per question |
| `N ≥ 2` | **sequence copy** — `llama_memory_seq_cp` of sequence 0 into up to `N − 1` throwaway sequences, all suffixes in one batched `llama_decode` | one per `N − 1` questions |

State restore is what the backend has always done. Sequence copy is the
llama.cpp equivalent of the Torch backend's `native-state-prefix-parallel-v1`:
the prefix is evaluated once, copied without serialization, and the branch
suffixes share a single decode. The branches are removed whole afterwards
(whole-sequence removal never fails), so sequence 0 is left exactly as
prefilled and the next chunk of questions needs no restore at all.

### Hybrid models

Qwen3.5 is a hybrid architecture: one full-attention layer in four, the rest
Gated DeltaNet with a recurrent state. In llama.cpp that state lives in
`llama_memory_recurrent`, which keeps only the state after the last token and
therefore **cannot be partially erased** — `llama_memory_seq_rm` refuses any
range that includes a sequence's last position (the source says: "models like
Mamba or RWKV can't have a state partially erased at the end of the sequence").
Truncating a branch back to the prefix is thus impossible on these models, and
that is a property of the architecture, not of a build or a binding.

What hybrid memories do support is whole-sequence removal, state save/restore,
and **sequence copies** — `llama_memory_recurrent::seq_cp` is implemented — which
is why both branching modes above work on Qwen3.5. Pure-attention models
additionally allow tail truncation; the backend does not rely on it.

`n_seq_max` is a context parameter (`llama_context_params.n_seq_max`), not a
build option; the same wheel serves both modes. The context is allocated as
`n_seq_max × (max_tokens + 64)` tokens and, when the library exposes it,
`kv_unified` is enabled so the copied branches share the prefix's attention
cells.

## Validation

`tests/test_llamacpp.py::test_real_gguf_scores_direct_serial_and_shared` loads the
real GGUF twice — with one sequence and with three — and checks that direct,
serial, restore-shared, copy-shared and marginal readouts agree on the
decisions, that copy-shared option logits match restore-shared ones within
0.5, and that `preamble_mass` stays in `[0, 1]`. Set `SEMIF_LLAMACPP_GGUF` to run
it, `SEMIF_LLAMACPP_GPU_LAYERS` to offload. It passes on:

- the PyPI CPU wheel of `llama-cpp-python` 0.3.35, CPU only;
- the cu124 wheel of the same version with every layer offloaded to an RTX 3080
  Laptop GPU.

**Known issue, not in this backend:** the cu124 wheel's *CPU* code path raises
`Illegal instruction` on an Intel Xeon W-11955M (AVX-512 without `avx512_bf16`
or AMX) for any decode, single- or multi-sequence, hybrid or pure-attention
model. With that wheel, offload the layers; for CPU scoring, use the PyPI wheel.

## Marginal readout

`--llama-readout marginal` (shared mode, `--llama-parallel ≥ 2`) changes how
the answer is read, not what is asked. Some checkpoints want to emit a token
*before* the letter — Qwen3-8B puts almost all of its next-token mass on the
Markdown `**` token when the assistant turn starts empty, and reading only the
last prompt position then reads that preamble instead of the answer. The
marginal readout takes, for each branch, the most likely non-slot tokens above
a floor (two probes, 5 %), appends each in one extra batched decode across all
branches, reads the slot masses after it, and adds them to the direct slot
masses weighted by the preamble's probability. `option_logits` then hold the
natural logs of the summed masses, so `softmax` and temperature scaling keep
working unchanged; `preamble_mass` records the share that came through a
preamble. On Qwen3.5-4B, which answers with the letter directly, the readout
changes nothing and `preamble_mass` stays at zero.

## Results

<!-- filled from the runs committed under results/raw/ -->

## Reproduce

```bash
# Quality: the owned labeled workload, three serving configurations
for MODE in direct serial shared; do
  semif-score --backend llamacpp --mode $MODE --llama-gpu-layers 99 --llama-parallel 8 \
    --model Qwen/Qwen3.5-4B --revision 851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a \
    --gguf /path/to/Qwen_Qwen3.5-4B-Q4_K_M.gguf \
    --input benchmarks/data/authored144.jsonl \
    --output results/raw/predictions/llamacpp-gguf-cuda-$MODE-authored144.jsonl
done
python benchmarks/evaluate.py --gold benchmarks/data/authored144.jsonl \
  --predictions results/raw/predictions/llamacpp-gguf-cuda-direct-authored144.jsonl

# Systems: the 37x21 fixture, fresh / serial-prefix / parallel-shared
python benchmarks/shape777.py --backend llamacpp --llama-gpu-layers 99 --llama-parallel 8 \
  --model Qwen/Qwen3.5-4B --revision 851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a \
  --gguf /path/to/Qwen_Qwen3.5-4B-Q4_K_M.gguf \
  --input benchmarks/data/shape777.jsonl \
  --output results/raw/shape777-llamacpp-gguf-cuda.json
```

Shared mode requires every row of the input to carry the same exact state;
`shape777.py` groups its fixture by `group_id` and does that for you.
