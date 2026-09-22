# Muse Glimmer 30B MLX support plan

## Goal

Add the pinned `mlx-community/Muse-Glimmer-30B-4bit` checkpoint as a local
decision-workbench option while preserving SemIf's one-resident-model and
text-only scoring boundaries.

## Verified compatibility

- The local machine is an Apple M4 Max with 128 GB unified memory and sufficient
  disk space for the approximately 19.4 GB checkpoint.
- The checkpoint declares native `muse_glimmer` architecture with affine 4-bit
  quantization, group size 64.
- The pinned MLX-LM dependency includes a native Muse Glimmer language-tower
  implementation and strips the vision tower during text-model sanitization.
- The playground remains a JSON state-and-question scorer. Image input is not
  added by this change.

## Implementation

1. Permit the native `muse_glimmer` model type without enabling remote code.
2. Add immutable source and revision metadata to the shared model catalog.
3. Expose the model through health, the existing selector, and `--default-model`.
4. Extend native architecture, catalog, and one-resident-model regression tests.
5. Update model-size, revision, quantization, and text-only documentation.
6. Download the pinned checkpoint and run a real Noul, Choice, and Score batch.

## Acceptance boundaries

- Use revision `3e7677d7a40d348a3daba263a2b1c0aa41910710` exactly.
- Do not apply in-memory quantization to the already-quantized source.
- A catalog or tiny-model test is not a substitute for loading and scoring the
  production checkpoint on Metal.
- Do not stop or restart the user's server on port 8090.

## Validation completed

- Downloaded the complete pinned checkpoint into the local Hugging Face cache.
- Loaded it through `SharedMLXService` on the 128 GB M4 Max and confirmed native
  `muse_glimmer` architecture with affine 4-bit, group-size-64 quantization.
- Ran every checked-in template. All five returned valid answers and normalized
  option distributions across Noul, Choice, and Score dimensions.
- The mixed ticket-triage batch completed scoring in 3.919 seconds. The remaining
  templates ranged from 2.311 seconds to 22.195 seconds depending on prompt size
  and dimension count.
- Verified the selector, 19.4 GB guidance, source, revision, source quantization,
  ready state, and footer label in a real browser on isolated port 8091.
- Released the isolated model and stopped port 8091 cleanly without touching the
  user's server on port 8090.
- Native Metal tests passed (`21 passed`); the full sandbox suite passed
  (`99 passed, 3 skipped`). The sandbox-only MLX skip is covered by the native
  Metal run.
- JavaScript syntax, `git diff --check`, raw benchmark checksums, CodeMirror
  checksums, and all 69 published summary claims passed.
