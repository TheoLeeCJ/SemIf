# Phase 1 results

## Finding

Open components reproduce the *interface pattern* of a semantic decision operator: runtime criteria, typed options, no decoding loop, and shared-state computation. They do not yet reproduce the full economic claim around Jev. Qwen3.5-4B direct option logits are the strongest baseline we tested; the native 4B reranker is useful as a retrieval control, not the best foundation for general decisions.

### Semantic quality

The browser demo now exposes three device tiers. Their base checkpoints were scored with the same native BF16 direct-logit interface before browser quantization:

| Model | Browser artifact | Download | Authored balanced accuracy | Perturbation balanced accuracy | TypeSafe subset agreement |
|---|---|---:|---:|---:|---:|
| Qwen3-0.6B | Q8_0 | 639 MB | 0.440 | 0.528 | 0.407 |
| MiniCPM5-2B | Q4_K_M | 1.56 GB | 0.686 | 0.693 | 0.637 |
| **Qwen3.5-4B** | Q4_K_M | 3.01 GB | **0.813** | **0.766** | 0.845 |
| Published Jev | Closed hosted service | — | — | — | **0.883** |

The owned quality values belong to the native BF16 checkpoints; they isolate model capability and are not presented as measurements of the quantized artifacts. TypeSafe agreement is an equal-case macro over the same selected 102 public rows and 20 cases for all four systems. Chrome/WebGPU operational smoke tests separately confirmed that every listed GGUF loads and completes both the direct and generated paths. Exact revisions, row-level predictions, and smoke timings are in `results/raw/browser-model-ladder.json`.

| Frozen workload | Metric | Direct Qwen3.5-4B | Qwen3-Reranker-4B | Public Jev value |
|---|---|---:|---:|---:|
| Authored, 144 rows | Mean family balanced accuracy | **0.813** | 0.625 | — |
| WANLI, 256 rows | Balanced accuracy | **0.637** | 0.522 | — |
| TypeSafe subset, 102 rows/20 cases | Equal-case reference agreement | **0.845** | 0.560 | 0.883 |
| Every judgment grid, 36 rows | Accuracy | **0.806** | 0.694 | — |
| Every action firewall, 10 actions | Composed action accuracy | 0.700 | 0.700 | — |

The reranker's paired difference from direct logits was -0.188 on the authored workload (95% source-group bootstrap interval -0.256 to -0.120) and -0.115 on WANLI (-0.184 to -0.044). Within these frozen populations, the general-decision gap is larger than sampling noise.

The TypeSafe difference between direct logits and the published Jev values is 3.8 percentage points on modal agreement for this available subset. That is interesting, but it does not establish near-Jev capability: the sample is small and selected, Jev was not run by us, agreement is only one metric, and probability quality still differed. Direct logits had total-variation distance 0.177 from the public target distributions versus Jev's 0.127; the reranker was 0.444.

On the two Every retrieval tasks, both systems had MRR 1.0 and the same Recall@1: 1.0 for code retrieval and 0.929 for company knowledge. The reranker's lower row-level binary accuracies (0.542 and 0.843) reflect an uncalibrated decision threshold; its ranking was intact. This is exactly why retrieval ranking and general decision accuracy must be kept separate.

### Robustness and confidence

On 36 owned base cases, direct logits scored 0.723 mean-family balanced accuracy and the reranker 0.530. For meaning-preserving variants:

| Variant | Direct accuracy | Direct flips | Reranker accuracy | Reranker flips |
|---|---:|---:|---:|---:|
| Option reversal | **0.813** | 10 | 0.498 | 2 |
| Criterion wrapper | **0.706** | 9 | 0.647 | 9 |
| Irrelevant context | **0.821** | 4 | 0.563 | 13 |

The direct model's option-order flips matter even though variant accuracy remained strong; positional wording and probability movements are not solved. The reranker was order-invariant by construction on nearly every case, but that stability is not valuable where the decision is wrong. Each system also made one non-`insufficient` choice above 0.8 score on the 36-row missing-evidence set. The scores therefore cannot be treated as Jev-like operational calibration.

### Systems benchmark

In a focused same-model comparison on one owned state with 21 criteria, parallel direct readout returned 21 probability pairs in a median **1.023 seconds** and generated no answer tokens. The strongest valid naïve baseline requested only an ordered JSON array of `"yes"`/`"no"` strings. It took a median **5.332 seconds**, including 0.489 seconds to first token, and emitted 111 tokens. All three arrays were valid and identical. They agreed with direct argmax on 18/21 criteria. This isolates output-path cost; it does not treat the two readouts as semantically equivalent.

A stricter request for a minified, whitespace-free array was also tested. The model repeated values past the required 21 entries and hit the 128-token cap in all three runs, so it is recorded as a failure rather than used to inflate the speed ratio. The earlier verbose 21-key confidence-object comparison (1.066 versus 18.229 seconds) remains in `results/raw/decision-vs-verbose-json.json`, but it is no longer the headline baseline.

The finalized one-RTX-3090 measurements are recorded in `results/phase1-summary.json`:

| Mode | Wall time | Decisions/s | State p50 | Argmax drift vs batch-1/fresh |
|---|---:|---:|---:|---:|
| Direct, fresh batch 1 | 333.1 s | 2.33 | 8.99 s | reference |
| Direct, serial state-prefix | 72.3 s | 10.75 | 1.93 s | 5/777 |
| Direct, parallel suffixes | **38.8 s** | **20.03** | **1.05 s** | 6/777 |
| Reranker, pair batch 1 | 417.3 s | 1.86 | 11.28 s | reference |
| Reranker, pair batch 4 | 441.7 s | 1.76 | 11.94 s | 51/777 |
| Reranker, pair batch 8 | 435.2 s | 1.79 | 11.76 s | 54/777 |

The reranker performs two full state/question/option evaluations per binary decision. Ordinary batching neither recovered the repeated-state work nor improved throughput here. Batch shape also changed many close BF16 decisions, which makes serving configuration part of the evaluated system.

The fixture and timing scope are described in [METHOD.md](METHOD.md). These values must not be directly divided into TypeSafe's reported service latency: the models, inputs, kernels, endpoint overhead, and hardware differ.

## What was and was not reproduced

Reproduced:

- Natural-language state and criteria mapped directly to typed option scores.
- No autoregressive answer generation or parser.
- Runtime-defined questions rather than a fixed task classifier head.
- A concrete shared-state reuse path across many decisions.
- A strong open semantic baseline at 4B parameters.

Not reproduced or established:

- Jev's undisclosed architecture or its claimed parallel sampler.
- RLCD training, because neither the training data nor a sufficient algorithmic specification is public.
- Calibrated probabilities suitable for operational thresholds.
- Terra-level or frontier-level general semantic ability.
- TypeSafe's advertised latency/cost on an equivalent workload and serving stack.
- The full 711-row TypeSafe benchmark or an independently operated Jev endpoint.

The next justified phase is targeted training for decision semantics and calibration, judged against these frozen baselines. It should proceed only after expanding external gold tasks and defining a held-out operational calibration target. A generic reranker fine-tune would answer the wrong question.

## Appendix — llama.cpp GGUF on a laptop GPU

Added with the GPU/fan-out extension of the llama.cpp backend; the phase-1
claims above are unchanged. Qwen3.5-4B Q4_K_M (`bartowski/Qwen_Qwen3.5-4B-GGUF`,
3 013 027 808 bytes), all layers on an RTX 3080 Laptop (16 GB), `llama-cpp-python`
0.3.35 cu124.

| workload | metric | Torch BF16, 3090 | GGUF Q4_K_M, laptop |
|---|---|---:|---:|
| authored144, direct | mean family balanced accuracy | 0.813 | 0.796 |
| authored144, direct | ECE, own-T out-of-fold | 0.038 | 0.063 |
| shape777, fresh | decisions / s | 2.33 | 1.40 |
| shape777, serial prefix | decisions / s | 10.75 | 9.21 |
| shape777, parallel shared | decisions / s | 20.03 | 10.88 (8 branches) · 10.51 (auto) |
| shape777, parallel vs fresh | argmax flips / 777 | 6 | 16–18 |

Raw: `results/raw/llamacpp-gguf-cuda-authored144.json`,
`results/raw/calibration/llamacpp-gguf-cuda-authored144.json`,
`results/raw/shape777-llamacpp-gguf-cuda.json`,
`results/raw/shape777-llamacpp-gguf-cuda-auto.json`, with row-level predictions
beside them. Method and caveats in [LLAMACPP.md](LLAMACPP.md).
