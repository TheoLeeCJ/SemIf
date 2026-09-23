# Qwen3.8 27B MLX support plan

## Goal

Add the pinned `mlx-community/Qwen3.8-27B-4bit` checkpoint as a local model
option in the decision workbench while keeping exactly one MLX model resident.

## Verified compatibility

- The checkpoint is approximately 16.1 GB and the local 128 GB M4 Max has
  ample unified-memory and disk headroom.
- Its configuration declares native `qwen3_5` architecture and affine 4-bit
  quantization with group size 64.
- SemIf's MLX backend already supports `qwen3_5`, so no architecture allowlist
  change is required.
- The playground uses only the text-scoring path; image and video input remain
  outside this state-and-question interface.

## Implementation

1. Add immutable source and revision metadata for the MLX checkpoint.
2. Add Qwen3.8 27B to the built-in selector as a source-quantized model.
3. Extend catalog and one-resident-model switching tests.
4. Update the README and CLI model-choice documentation.
5. Verify the selector in a real browser on an isolated server.
6. Download the pinned checkpoint and run a real shared-prefix MLX smoke test
   across Noul, Choice, and Score dimensions.

## Validation completed

- Downloaded the complete pinned 16.1 GB checkpoint into the local Hugging Face
  cache.
- Loaded it through `SharedMLXService` on the 128 GB M4 Max and confirmed native
  `qwen3_5` architecture with affine 4-bit, group-size-64 quantization.
- Scored a real shared-prefix batch containing Noul, Choice, and Score
  dimensions. All three returned valid option distributions; Choice and Score
  also returned normalized-entropy concentration metrics.
- The three-dimension scoring run completed in 3.810 seconds after load.
- Released the model cleanly; service status returned to `unloaded`.
- Verified the selector, source, revision, source quantization, and download size
  in a real browser on isolated port 8091 without touching port 8090.
- Passed the focused suite (`17 passed, 1 skipped`) and full suite
  (`89 passed, 3 skipped`), verified every published raw checksum, and verified
  all 69 published summary claims.

## Acceptance boundaries

- A selector/catalog test alone is not sufficient for completed support.
- Live validation must use revision
  `10c35caafbb80f7dc6a7a432cdd11af10a6d4818`.
- The model must release cleanly after the live test.
- The user's server on port 8090 must not be stopped or restarted.
