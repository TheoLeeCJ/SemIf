# Option-order sensitivity

Direct letter-slot readouts assign options to fixed tokens `A`, `B`, `C`, ….
Semantic option IDs are independent of that display order. The Phase 1 results
already record that reversing option order flips about **10/36** direct decisions
on the owned stability set ([RESULTS.md](RESULTS.md)). This document describes
the measurement harness and an **opt-in** mitigation. Neither path rewrites
frozen quality tables or default `direct-options-v1` scores.

## Measurement (CPU, no model required)

`benchmarks/evaluate_option_order.py` joins committed base predictions to the
frozen `option_reversal` perturbation predictions, aligns probabilities by
semantic option ID, and reports:

- argmax flips and flip rate
- mean / max total variation (`0.5 * L1` on the aligned distributions)
- display-position bias (win rate of letter slot A/B/C/… before vs after reverse)

```bash
python benchmarks/evaluate_option_order.py \
  --gold benchmarks/data/authored144.jsonl \
  --perturbations benchmarks/data/perturbations108.jsonl \
  --base-predictions results/raw/predictions/direct-authored144.jsonl \
  --perturbation-predictions results/raw/predictions/direct-perturbations108.jsonl \
  --output /tmp/option-order-direct.json
```

The same command works for reranker prediction files. The harness is analysis-only:
it does not change scorer outputs. Shared math lives in
`semif_phase1.order` and is covered by CPU unit tests with fixture logits (no
weight downloads).

To regenerate fresh permutation scores on CPU without a GPU, use the llama.cpp
backend (`--backend llamacpp --gguf …`) or Torch `--device cpu`, then point this
harness at the new create-only prediction paths.

## Opt-in `--stabilize-order K`

```bash
semif-score --mode direct --stabilize-order 6 \
  --model … --revision … --input decisions.jsonl --output stabilized.jsonl
```

Supported on **direct** and **serial** only (including Torch / MLX / llama.cpp).
Rejected for `shared` and `reranker`.

For each decision the scorer:

1. Selects `K` distinct option permutations (always includes identity; includes
   reverse when `K >= 2`; samples the rest with `--stabilize-order-seed`).
2. Scores each permutation with the ordinary native slot-logit readout.
3. Aligns `option_logits` back to the original semantic option-ID order.
4. Averages the aligned logits, then applies softmax.

Outputs are marked so they cannot be confused with the default readout:

| Field | Value |
|---|---|
| `prompt_version` | `direct-options-stabilize-order-v1` |
| `readout` | describes the K-permutation average |
| `stabilize_order.k` / `cost_multiplier` | `K` |

### Cost

Each decision costs **`K` forward passes** instead of one (same prompt length
class per pass). For the owned 3-option rows, `K=6` covers every permutation;
`K=2` is identity + reverse only. Do **not** put stabilize-order numbers into
frozen quality tables or `results/phase1-summary.json`.
