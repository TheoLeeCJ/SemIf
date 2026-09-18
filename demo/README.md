# Decision readout versus generated tokens

Open `index.html` directly in a current browser. It has no dependencies, network requests, or model execution.

The replay shows one measured request on each side: the same frozen BF16 Qwen3.5-4B, owned state, 21 binary criteria, and RTX 3090.

| Path | Completion | Output |
|---|---:|---:|
| Direct typed logits | 1.023 s median | 21 probability pairs; 0 generated tokens |
| Compact JSON array | 5.332 s median | Valid 21-value array; 111 generated tokens |

The generative path emitted its first token at a median 0.489 seconds and then visibly streams its recorded answer. The direct output appears together at its measured completion point. Short labels make the questions readable. Open `index.html` for the interactive replay; the repository media are static previews of that page.

Included project-owned media:

- `assets/semif-phase1-replay.gif` — GitHub README preview.
- `assets/semif-phase1-replay.webm` — 1280 × 720 VP9 preview.
- `assets/semif-phase1-replay-poster.png` — poster frame.
- `assets/semif-social-preview.png` — 1200 × 630 social preview.

The visual compares output paths, not semantic correctness. See the main [results](../docs/RESULTS.md) and [method](../docs/METHOD.md) for quality and scope.
