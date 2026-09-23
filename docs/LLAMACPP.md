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

`--llama-gpu-layers` defaults to `auto`, which leaves llama.cpp's own default —
`-1`, every layer, in current builds — and records what the library did. `0`
forces CPU, `N` offloads `N` layers. Offload needs a llama-cpp-python wheel
built with a GPU backend; the PyPI wheel is CPU-only and ignores the setting
(`gpu_offload_supported` in the metadata says so). For CUDA:

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
| `auto` (default) | **sequence copy, sized per state** — as many of the state's questions per decode as the context holds: `prefix + Σ suffixes ≤ n_ctx`, at most 32 branches | usually one |
| `N ≥ 2` | sequence copy with `n_seq_max = N`, chunks of at most `N − 1` | one per `N − 1` questions |
| `1` | **state restore** — `llama_state_seq_get_data` after the prefill, `set_data` before each question | one per question |

State restore is what the backend has always done. Sequence copy is the
llama.cpp equivalent of the Torch backend's `native-state-prefix-parallel-v1`:
the prefix is evaluated once, copied without serialization, and the branch
suffixes share a single decode. The branches are removed whole afterwards
(whole-sequence removal never fails), so sequence 0 is left exactly as
prefilled and the next chunk of questions needs no restore at all.

Why `auto` needs no number from you: everything that bounds a fan-out is known
before the first decode. `encode_verified` has already tokenized each row, so
the prefix length and every suffix length are in hand, and the context budget
is fixed at load. The branch count is therefore a consequence of the data — on
the `shape777` fixture (states of ~1 800 tokens, suffixes of ~90) all 21
questions of a state fit in one decode. `branches_per_decode` in the shared
timing records what was actually done.

The context is sized **once**, for the longest single prompt (`max_tokens + 64`),
not multiplied by the number of sequences. That relies on `kv_unified`: measured
on this build, with 16 sequences and `n_ctx = 4096`, each sequence sees 4 096
cells under the unified buffer and 256 without it, and copied branches share the
prefix's cells rather than duplicating them. A library without `kv_unified` falls
back to the per-sequence allocation.

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
build option; the same wheel serves both modes.

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

`--llama-readout marginal` (shared mode, any `--llama-parallel` but `1`) changes how
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

RTX 3080 Laptop GPU (16 GB), Q4_K_M, all 33 layers offloaded, `llama-cpp-python` 0.3.35 cu124.

### Quality — `authored144`, direct mode

| backend | mean family balanced accuracy | ECE, own-T out-of-fold | fitted T |
|---|---:|---:|---:|
| Torch BF16, RTX 3090 (committed) | 0.813 | 0.038 | 1.23 |
| **llama.cpp GGUF Q4_K_M, laptop** | **0.796** | **0.063** [0.038, 0.123] | 1.26 |

Coverage 144/144. Median `allowed_token_mass` 0.9997, minimum 0.975, no row
below 0.9 — the quantized model answers with a letter as reliably as the BF16
one. Median forward time 64 ms per decision at a median 147 input tokens.
Row-level predictions are in
`results/raw/predictions/llamacpp-gguf-cuda-direct-authored144.jsonl`, the
report in `results/raw/llamacpp-gguf-cuda-authored144.json`, the calibration
in `results/raw/calibration/llamacpp-gguf-cuda-authored144.json`. The 4-bit
checkpoint gives up 1.7 points of balanced accuracy and is somewhat less
well-calibrated than BF16; its fitted temperature is nearly the same.

### Systems — `shape777`, 37 states × 21 questions

| mode | decisions / s | state p50 | argmax flips vs fresh |
|---|---:|---:|---:|
| fresh | 1.40 | 14.97 s | — |
| serial_prefix (state restore) | 9.21 | 2.28 s | 19 / 777 |
| parallel_shared, `--llama-parallel 8` | 10.88 | 1.93 s | 16 / 777 |
| parallel_shared, `auto` | 10.51 | 2.00 s | 18 / 777 |

The committed Torch BF16 run on an RTX 3090 reports 2.33, 10.75 and 20.03
decisions per second for fresh, serial and parallel.

Two things the numbers say. Fan-out beats state restore by removing the
serialization round-trip, not by batching: one decode per state (`auto`,
21 branches) is no faster than three (8 branches), because llama.cpp splits a
batch into micro-batches of `n_ubatch` tokens either way and the recurrent
memory decodes them with `split_equal`. And the gap to the Torch parallel figure
is not the fan-out's: the Torch backend runs one dense BF16 forward over padded
suffixes, which this 4-bit hybrid path cannot match on a laptop GPU. Reports:
`results/raw/shape777-llamacpp-gguf-cuda.json` (8 branches) and
`results/raw/shape777-llamacpp-gguf-cuda-auto.json`, each with its
`.predictions.jsonl`. The 16–19 argmax flips out of 777 between decode paths
are the quantized model's noise floor — two sequential reads of one prompt
already differ on about 3 % of decisions when only the micro-batch size
changes — not a property of the fan-out.


## Reproduce

```bash
# Quality: the owned labeled workload, three serving configurations
for MODE in direct serial shared; do
  semif-score --backend llamacpp --mode $MODE \
    --model Qwen/Qwen3.5-4B --revision 851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a \
    --gguf /path/to/Qwen_Qwen3.5-4B-Q4_K_M.gguf \
    --input benchmarks/data/authored144.jsonl \
    --output results/raw/predictions/llamacpp-gguf-cuda-$MODE-authored144.jsonl
done
python benchmarks/evaluate.py --gold benchmarks/data/authored144.jsonl \
  --predictions results/raw/predictions/llamacpp-gguf-cuda-direct-authored144.jsonl

# Systems: the 37x21 fixture, fresh / serial-prefix / parallel-shared
python benchmarks/shape777.py --backend llamacpp \
  --model Qwen/Qwen3.5-4B --revision 851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a \
  --gguf /path/to/Qwen_Qwen3.5-4B-Q4_K_M.gguf \
  --input benchmarks/data/shape777.jsonl \
  --output results/raw/shape777-llamacpp-gguf-cuda.json
```

Shared mode requires every row of the input to carry the same exact state;
`shape777.py` groups its fixture by `group_id` and does that for you.
