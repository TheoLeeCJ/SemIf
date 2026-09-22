# Gemma 4 31B MLX support plan

## Goal

Add the instruction-tuned `mlx-community/gemma-4-31b-it-4bit` checkpoint as
a pinned local model in the decision workbench while preserving the existing
one-resident-model memory and MLX thread-affinity boundaries.

## Compatibility findings

- The local machine is an Apple M4 Max with 128 GB unified memory and enough
  disk space for the approximately 18.4 GB MLX checkpoint.
- The checkpoint declares the native `gemma4` architecture with MLX affine
  4-bit quantization. The current SemIf backend already permits this model type
  and delegates unused multimodal weight removal to MLX-LM for text scoring.
- The instruction-tuned checkpoint includes the chat template required by
  SemIf's system/user prompt and direct option-logit readout.
- Gemma 4 31B is dense, unlike the sparse 26B A4B model, so it should be
  expected to use more compute per token even though both checkpoints fit.

## Implementation

1. Add a catalog entry pinned to revision
   `696d436c404745a59f30e4939a658162b0a9e57f`.
2. Expose the model through `/api/health`, the existing data-driven selector,
   and `--default-model` without changing the UI layout.
3. Extend catalog and one-resident-model switching tests.
4. Update the playground model, revision, quantization, and CLI documentation.
5. Run focused and full tests, JavaScript syntax validation, repository evidence
   checks, and a browser selector check on an isolated server.
6. Download the pinned checkpoint and run a real Noul, Choice, and Score batch
   through `SharedMLXService` when network access is available.

## Acceptance boundaries

- Only one MLX model remains resident at a time.
- The playground continues to use the language-model path; image input remains
  outside the state-and-question scoring interface.
- Source revision and 4-bit source quantization are immutable catalog metadata.
- Catalog, unit, and browser checks are not substitutes for a successful live
  production-checkpoint load and score.

## Validation completed

- Downloaded and loaded the exact pinned 18.4 GB checkpoint on the local
  128 GB M4 Max.
- Confirmed native `gemma4` architecture and MLX affine 4-bit quantization with
  group size 64.
- Ran one shared-prefix batch across Noul, Choice, and Score. The service reached
  `ready`, returned valid option distributions for all three dimensions, and
  completed scoring in 1.842 seconds.
- Released the model cleanly; the service returned to `unloaded`.
- Verified the selector, source, revision, source precision, download guidance,
  load-button label, and footer label in a real browser on an isolated server.
  The only console entry was the pre-existing missing favicon 404.
- Focused tests passed (`17 passed, 1 skipped`); the full suite passed
  (`89 passed, 3 skipped`). The sandbox-only skip was covered by the separate
  real Metal run.
- JavaScript syntax validation and `git diff --check` passed, all raw evidence
  checksums matched, and all 69 published summary claims verified.
