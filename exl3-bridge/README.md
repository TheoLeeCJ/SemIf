# exl3 bridge (quantized readout)

`exl3-bridge/` is a standalone execution track — like `webgpu-demo/` — for the
direct-mode decision contract, but the LLM forward pass runs through
[exllamav3](https://github.com/turboderp-org/exllamav3) over quantized `.exl3`
checkpoints instead of the pinned BF16 reference. It is **additive**: the
pinned `Qwen/Qwen3.5-4B @ 851bf6e` BF16 claims in `results/`,
`phase1-summary.json`, and the root README remain untouched, and nothing in
`src/`, `benchmarks/`, or `examples/` is modified.

## Contract (unchanged from `src/`)

| Contract | Where |
|---|---|
| Prompt built by `semif_phase1.core` (`direct-options-v1`), same system JSON schema | `exl3_runner.py` imports `semif_phase1.core` |
| Per-row `prompt_sha256` | `semif_phase1.core.encode_prompt` |
| Single-token `A`/`B` slots, validated **and** prefix-stable | `semif_phase1.core` (`encode_prompt`, `find_slot_token_ids`) |
| Readout = softmax over **full-vocabulary** last-position logits restricted to declared options | exllamav3 `Job(return_logits=True)`, identical quantity to `logits[:, -1, :]` |
| No truncation — over-budget rows are refused, never cut | `input_budget_check` re-checked against exllamav3 `model.token_length` |
| Create-only, append-resumable output; one JSONL row per decision | `exl3_runner.py` |

## Results (repository frozen fixtures)

Runs on the project-owned fixtures only (no third-party data). Speeds are
mean per-row `total_seconds` on a single RTX 3090 (this bridge's runtime host).

| Evidence | Rows | 4B BF16 (pinned) | 27B exl3 5.0bpw | 27B exl3 2.0bpw | Gemma-4 26B-A4B exl3 2.10bpw |
|---|---:|---|---:|---:|---:|
| `authored144` balanced accuracy | 144 | 0.813 | **0.9579** | 0.9443 | **0.9552** |
| Speed, `authored144` (~148 tok/row) | 144 | 0.12 s | 0.38 s | 0.39 s | 0.18 s |
| Speed, `shape777` (~1,842 tok/row) | 777 | n/a (no timing committed) | 1.83 s | 1.66 s | **0.37 s** |
| Checkpoint size | | 8.7 GB | 19 GB | 10.8 GB | 10.3 GB |

Row-level evidence + SHA256SUMS ship in `results/`. Caveats:

- Family *and* quantization differ from the pinned 4B baseline, so
  27B/Gemma deltas vs 0.813 are bridge-vs-pinned comparisons. The 27B
  5.0bpw vs 2.0bpw pair *is* a clean quantization ablation (same family,
  same prompts): 2.0bpw costs ~1.4pt balanced accuracy for a 60% smaller
  checkpoint, and all 6 argmax deltas vs 5.0bpw moved away from gold.
- Gemma-4 26B-A4B (Apache-2.0) is a different family, but at 4B active
  parameters (MoE) it reaches 27B-class accuracy at ~5× the long-prompt
  speed of the 27B 5.0bpw and 4.9× at 1/3 the checkpoint size.
- A separate off-repo zero-shot probe on an external cable dataset was also
  run; that data's upstream license is "unknown", so **no probe inputs or
  outputs from it are committed**. An off-repo Qwen3.5-9B exl3 2.0bpw trial
  measured 0.8279 balanced accuracy on `authored144` (≈ the 4B reference);
  its outputs are likewise not committed.

`compare_fixtures.py` recomputes `results/fixture-comparison.json` from the
committed 27b 5.0bpw fixtures (default, no args). To evaluate any other
bridge run, pass its outputs:

```bash
python exl3-bridge/compare_fixtures.py \
  --authored exl3-bridge/results/authored144-gemma4-26b-a4b-2.10bpw.jsonl \
  --shape exl3-bridge/results/shape777-gemma4-26b-a4b-2.10bpw.jsonl \
  [--out some-summary.json]   # omit --out to print only
```

## Reproduce

Requires CUDA + exllamav3 (MIT). The runner is stdlib-only beyond
exllamav3/torch/transformers; it is not part of the pinned reference runtime.
The pinned 1.4.4 build runs the Qwen exl3 checkpoints; **Gemma-4 models
require exllamav3 ≥ 1.5.1** (same cu128/torch2.10 wheel family).

```bash
python -m venv .venv-exl3 && . .venv-exl3/bin/activate
pip install "torch==2.10.0" "transformers==5.17.0"
# Qwen exl3 checkpoints (pinned):
pip install "https://github.com/turboderp-org/exllamav3/releases/download/v1.4.4/exllamav3-1.4.4%2Bcu128.torch2.10.0-cp310-cp310-linux_x86_64.whl"
# Gemma-4 checkpoints need ≥1.5.1, e.g.:
# pip install "https://github.com/turboderp-org/exllamav3/releases/download/v1.5.1/exllamav3-1.5.1%2Bcu128.torch2.10.0-cp312-cp312-linux_x86_64.whl"
python exl3-bridge/exl3_runner.py \
  --model-dir /path/to/model-exl3 \
  --model-source turboderp/Qwen3.8-27B-exl3 \
  --model-revision a35e75a73baee51da709329d19294245cbeeb5d8 \
  --input  benchmarks/data/shape777.jsonl \
  --output exl3-bridge/results/shape777-27b-exl3.jsonl \
  --cache-size 16384 --gpu-split 22.5
(cd exl3-bridge/results && sha256sum -c SHA256SUMS)
pytest exl3-bridge/test_bridge.py -q
python exl3-bridge/compare_fixtures.py
```

Committed fixture provenance (all pinned by `--model-revision`, verified via
SHA256SUMS):

| Result files | Source | Revision |
|---|---|---|
| `*-27b-exl3.jsonl` | `turboderp/Qwen3.8-27B-exl3` (5.00bpw) | `a35e75a73bae` |
| `*-27b-exl3-2.0bpw.jsonl` | `turboderp/Qwen3.8-27B-exl3` (2.00bpw) | `0cd109127229` |
| `*-gemma4-26b-a4b-2.10bpw.jsonl` | `turboderp/gemma-4-26B-A4B-it-exl3` (2.10bpw) | `d20a0b4062b6` |

CI-safe tests: `test_bridge.py` stubs `exllamav3` and validates the runner's
prompt/slot/refusal/resume contract without GPU, weights, or network.

## Limitations

- exllamav3's `return_logits` path returns logits for `max_new_tokens + 1`
  positions; the runner asserts the shape and uses position `-1`.
- Quantization quality depends entirely on the uploaded `.exl3` checkpoint
  (bits, `head_bits`); the runner reports `exl3` metadata in every row but does
  not audit the checkpoint.
- `--input-budget` default (16384) must stay below the KV-cache capacity
  implied by `--cache-size` / `--gpu-split`; rows over budget are refused.
