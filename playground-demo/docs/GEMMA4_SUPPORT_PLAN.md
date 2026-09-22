# Gemma 4 12B MLX support plan

## Goal

Add the instruction-tuned `mlx-community/gemma-4-12B-it-4bit` checkpoint as a
third shared local model in the decision workbench. Preserve immutable model
provenance, one-resident-model switching, direct option-logit scoring, and the
existing TypeSafe comparison boundary.

## Compatibility findings

- Pinned model revision:
  `73bcf09092aa277861d5a191b989b666f7f32e8f`.
- The checkpoint declares `model_type: gemma4_unified` and native MLX affine
  4-bit quantization. The pinned `mlx-lm` runtime remaps that architecture to its
  built-in Gemma 4 text implementation and discards unused vision/audio weights
  for this text-only scorer.
- The production Gemma tokenizer accepts the existing system and user turns,
  renders `enable_thinking=False`, preserves A-F as exact single tokens, and
  yields a nonempty shared state prefix across different questions.
- A tiny native Gemma 4 model produces equivalent direct and shared-prefix
  option probabilities within the existing MLX tolerance.
- Live loading revealed that MLX GPU streams are thread-affine. The shared
  service therefore must dispatch every model load, release, and evaluation to
  one dedicated worker instead of relying only on a cross-thread lock.

## Goals and acceptance checks

1. **Pinned model catalog**
   - Add a Gemma 4 12B IT 4-bit entry with source, immutable revision, download
     size, and source-precision metadata.
   - Expose it through `/api/health`, the existing selector, and
     `--default-model` without hard-coded UI markup.

2. **Backend compatibility**
   - Permit the native `gemma4_unified` architecture while continuing to reject
     custom model code and unknown architectures.
   - Do not re-quantize the already-quantized Gemma checkpoint when the service
     has a global Qwen `--mlx-bits` setting.
   - Preserve MLX stream affinity with one dedicated model worker while HTTP and
     TypeSafe requests remain independently threaded.

3. **Regression coverage**
   - Verify the catalog and switching contract includes Gemma.
   - Exercise shared scoring with a tiny native Gemma 4 model.
   - Run playground tests, repository MLX tests, JavaScript syntax checking,
     evidence verification, and a browser selector check.

4. **Runtime proof**
   - Load the pinned production checkpoint and evaluate a small multi-question
     request when network/model download access is available.
   - Keep this distinct from static/unit validation if the multi-gigabyte
     runtime check cannot be completed.

## Validation result

- `pytest -q`: 97 passed, 1 skipped.
- `tests/test_mlx.py`: 18 passed on the Apple Metal GPU.
- `node --check playground-demo/app.js`: passed.
- `results/raw/SHA256SUMS`: all entries passed.
- `benchmarks/verify_published.py`: 69 summary claims verified.
- The pinned 6.74 GB Gemma checkpoint loaded with MLX 0.32.2 / MLX-LM 0.32.0
  and completed a two-dimension API smoke test using one shared prefix in 1.54s.
- The browser selector displayed the pinned source, revision, and 4-bit source
  precision. The six-dimension Support agent audit rendered locally in 5.62s.
