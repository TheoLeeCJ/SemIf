# exl3 bridge (quantized readout)

`exl3-bridge/` is a standalone execution track — like `webgpu-demo/` — for the
direct-mode decision contract, but the LLM forward pass runs through
[exllamav3](https://github.com/EriqKahn/exllamav3) over quantized `.exl3`
checkpoints instead of the pinned BF16 reference. It is **additive**: the
pinned `Qwen/Qwen3.5-4B @ 851bf6e` BF16 claims in `results/`,
`phase1-summary.json`, and the root README remain untouched, and nothing in
`src/`, `benchmarks/`, or `examples/` is modified.

## Contract (unchanged from `src/`)

| Contract | Where |
|---|---|
| Prompt built by `openjev_phase1.core` (`direct-options-v1`), same system JSON schema | `exl3_runner.py` imports `openjev_phase1.core` |
| Per-row `prompt_sha256` | `openjev_phase1.core.encode_prompt` |
| Single-token `A`/`B` slots, validated **and** prefix-stable | `openjev_phase1.core` (`encode_prompt`, `find_slot_token_ids`) |
| Readout = softmax over **full-vocabulary** last-position logits restricted to declared options | exllamav3 `Job(return_logits=True)`, identical quantity to `logits[:, -1, :]` |
| No truncation — over-budget rows are refused, never cut | `input_budget_check` re-checked against exllamav3 `model.token_length` |
| Create-only, append-resumable output; one JSONL row per decision | `exl3_runner.py` |

## Results (repository frozen fixtures)

Runs on the project-owned fixtures only (no third-party data):

| Evidence | Rows | Bridge (Qwen3.8-27B exl3 5.0bpw) | Pinned 4B BF16 direct (committed) |
|---|---:|---|---|
| `authored144` balanced accuracy | 144 | **0.937** | 0.813 |
| `shape777` argmax agreement vs committed direct-4B row-level predictions | 777 | **0.849** (117 flips) | reference |

Row-level evidence + SHA256SUMS ship in `results/`; `compare_fixtures.py`
recomputes `results/fixture-comparison.json` from committed fixtures. Caveat:
family *and* quantization differ from the pinned baseline, so these deltas are a
bridge-vs-pinned comparison, not a quantization ablation. A separate off-repo
zero-shot probe on an external cable dataset was also run; that data's upstream
license is "unknown", so **no probe inputs or outputs from it are committed**.

## Reproduce

Requires CUDA + exllamav3 (MIT). The runner is stdlib-only beyond
exllamav3/torch/transformers; it is not part of the pinned reference runtime.

```bash
python -m venv .venv-exl3 && . .venv-exl3/bin/activate
pip install "exllamav3==1.4.4+cu128.torch2.10.0" \
  "torch==2.10.0" "transformers==5.17.0"
python exl3-bridge/exl3_runner.py \
  --model-dir /path/to/model-exl3 \
  --input  benchmarks/data/shape777.jsonl \
  --output exl3-bridge/results/shape777-27b-exl3.jsonl \
  --cache-size 16384 --gpu-split 22.5
(cd exl3-bridge/results && sha256sum -c SHA256SUMS)
pytest exl3-bridge/test_bridge.py -q
python exl3-bridge/compare_fixtures.py
```

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
