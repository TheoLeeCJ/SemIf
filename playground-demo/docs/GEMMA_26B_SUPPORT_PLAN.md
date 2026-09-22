# Gemma 4 26B A4B MLX support plan

## Goal

Add the instruction-tuned `mlx-community/gemma-4-26b-a4b-it-4bit`
checkpoint as a pinned local model option in the decision workbench while
preserving the one-resident-model memory boundary.

## Verified compatibility

- The local machine is an Apple M4 Max with 128 GB unified memory and enough
  disk space for the approximately 15.4 GB MLX checkpoint.
- The installed `mlx-lm` includes native `gemma4` text and sparse-MoE model
  implementations.
- The current SemIf backend rejects `gemma4` only because its explicit model
  type allowlist currently contains `gemma4_unified` and `qwen3_5`.
- The chosen checkpoint is instruction-tuned and already quantized at 4-bit,
  so SemIf must use its source quantization rather than applying `--mlx-bits`.

## Implementation

1. Extend the MLX model-type allowlist to include native `gemma4`.
2. Add a pinned catalog entry for the MLX 4-bit instruction-tuned checkpoint:
   revision `0d77464eeb233a2da68ebf9d7dc4edaac7db956d`.
3. Extend catalog, switching, and backend validation tests.
4. Update the playground documentation and CLI option listing.
5. Verify unit tests, repository integrity checks, and the model selector in a
   real browser without disturbing the user's server on port 8090.
6. If the model is available locally, run a real load-and-score smoke test. If
   it is not cached, treat the 15.4 GB download as a separate validation step.

## Validation completed

- Downloaded and loaded the pinned checkpoint on the local 128 GB M4 Max.
- Confirmed the model reports native `gemma4` architecture and affine 4-bit
  quantization, with 8-bit router weights.
- Ran a real shared-prefix MLX batch across Noul, Choice, and Score dimensions.
  The service reached `ready`, returned valid distributions and entropy-based
  confidence values for all three dimensions, and completed in 0.928 seconds.
- Released the loaded model cleanly after the live test.
- Verified the new selector entry and selected-model metadata in a real browser
  on an isolated server without touching the user's server on port 8090.
- Passed the focused suite (`17 passed, 1 skipped`) and full suite
  (`89 passed, 3 skipped`), verified every published raw checksum, and verified
  all 69 published summary claims.

## Acceptance boundaries

- The playground continues to keep only one MLX model resident at a time.
- SemIf uses only the language-model path; multimodal image input is outside
  this playground's state-and-question scoring interface.
- The source revision and 4-bit quantization are immutable catalog metadata.
- A catalog/UI test is not a substitute for a completed live model load.
