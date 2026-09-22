# Make provider failures actionable without duplicating them per dimension

Written against: 1f2dea3e25379f9dfc98cb83c324f00ab5deda37

## Evidence chain

- Surface: `playground-demo/` at `/`, Compare view, Support agent audit template, Jev Bocha selected, Jev Bocha request failed while Jev official completed.
- Problem: one request-level Jev Bocha error is rendered in all six dimension rows, and every Comparison cell repeats `Incomplete` plus `Both results are required to compare.` The footer then says to open Rendered or Raw JSON even though Compare already repeats the raw endpoint error.
- Design evidence: the supplied desktop screenshot; `playground-demo/app.js` functions `answerFor`, `renderModelResult`, `comparisonFor`, `renderComparison`, `renderResponses`, and `runRequest`; the existing `--danger`, `--surface-soft`, `--rule`, and provider-color tokens in `playground-demo/style.css`.
- Owner: `playground-demo/index.html`, `playground-demo/app.js`, and `playground-demo/style.css`.
- Scope and affected surfaces: Compare, Rendered, and footer status presentations for Local MLX, Jev Bocha, and Jev official request-level failures. Raw JSON remains the diagnostic source of truth.
- Uncertainty: exact provider error wording can vary; known-error mapping must fall back safely for unknown failures.

## Design decision

Represent a failed provider request once at the response-surface level, because it is one request failure rather than one failure per question. Keep successful results from the other provider visible. Replace repeated raw errors in dimension cells with a compact unavailable state, and reserve the full original error for Raw JSON.

For the known Bocha checkpoint-limit error, show actionable copy: `Input is too long for Jev Bocha. Its current checkpoint accepts at most 512 tokens per expanded candidate. Shorten the shared state or instructions, or compare with Local MLX.` Do not imply that Setup can increase the limit.

## Reuse

- Tokens: `--danger`, `--surface-soft`, `--rule`, `--bocha`, `--local`, and `--jev` from `playground-demo/style.css`.
- Existing compositions: `.response-header`, `.response-body`, `.answer-empty`, `.comparison-result`, and `.run-status`.
- Exemplar: the grouped, single-instance status treatment used by `.connection-summary` in the Setup drawer, adapted to the response surface rather than reused as a provider-configuration component.
- New primitive: add a response-scoped alert because no existing component represents a request-level evaluation failure above all response views. It belongs in the response pane and is shared by Compare and Rendered states.

## Changes

1. `playground-demo/index.html`
   - Change: add one hidden response-alert container between `.response-header` and `.response-body`, with a title, explanatory copy, and a compact recovery-action region. Keep it in the response pane so it remains visible when switching Compare, Rendered, and Raw JSON.
   - Preserve: the existing response tabs, table structure, and raw-response panel.
   - Verify: a single failed provider produces exactly one visible alert above the active response view.

2. `playground-demo/app.js`
   - Change: add a pure provider-error presentation helper that returns user-facing title, explanation, and recovery suggestions for known errors, while retaining a generic fallback. Match the Bocha token-limit response by stable semantic fragments rather than the complete URL or an exact full message.
   - Change: update `renderResponses` to render or clear the response alert from the request-level `challenger` and `jev` statuses.
   - Change: when a provider request has failed, render `Unavailable` in that provider's per-dimension cells without repeating the raw error. Render `Not comparable` once per affected Comparison cell without the redundant `Both results are required` paragraph.
   - Change: keep a missing individual answer distinct from a failed provider request; `Response omitted this dimension.` remains a per-dimension error.
   - Change: make footer failure status concise and consistent with the alert, for example `Jev Bocha failed. Review the response notice or Raw JSON.`
   - Preserve: successful answer distributions, confidence wording, timing, concurrent requests, and complete original errors in `app.response` and Raw JSON.
   - Verify: Jev official results remain readable when Jev Bocha fails; Local MLX failures and Jev official failures use the same request-level pattern with provider-appropriate labels.

3. `playground-demo/style.css`
   - Change: style the response alert with the existing neutral surface, rule, and danger/provider tokens. Keep the visual weight below the page header but above per-dimension results.
   - Change: add compact unavailable and not-comparable variants that do not resemble successful agreement or difference badges.
   - Preserve: current typography, provider colors, table borders, and responsive breakpoints.
   - Verify: the alert wraps cleanly without horizontal overflow at desktop, 900 px, and 620 px widths.

4. `playground-demo/test_playground_demo.py`
   - Change: extend the static-surface test to require the alert container and the known Bocha token-limit copy/mapping without embedding credentials or complete endpoint errors in HTML.
   - Preserve: existing secret-boundary and route assertions.
   - Verify: tests fail if the response alert is removed or raw provider errors are again directly rendered in every model cell.

## Scope

- Inherit: Local MLX, Jev Bocha, and Jev official request-level failure states in Compare and Rendered views.
- Verify: pending states, one-provider success, both-provider failure, one omitted dimension, tab switching, and repeat runs after an error.
- Exclude: provider token-limit changes, automatic truncation, request batching, backend response-shape changes, Setup configuration, and template content changes.

## Validation

- Product: run Support agent audit against Jev Bocha and reproduce the checkpoint-limit error; expect one actionable response alert, intact Jev official results, compact unavailable cells, and the raw provider error only in Raw JSON.
- Interface: inspect successful comparison, challenger-only failure, Jev-official-only failure, both failures, and rerun recovery at desktop, 900 px, and 620 px widths.
- System: confirm all three providers use the same response-alert owner and no provider-specific alert markup is duplicated in templates.
- Repository: `node --check playground-demo/app.js` -> exits successfully.
- Repository: `pytest -q` -> all tests pass.
- Repository: `(cd results/raw && sha256sum -c SHA256SUMS)` -> every published raw artifact reports `OK`.
- Repository: `python benchmarks/verify_published.py` -> published benchmark verification passes.

## Stop conditions

- Stop if the backend no longer preserves the original provider error in `app.response`, if error responses become dimension-specific rather than request-level, or if resolving the issue requires changing provider requests or truncating user input.

## Design documentation

- After acceptance and validation: record in `playground-demo/README.md` that provider request failures are summarized once in the rendered UI while full diagnostics remain in Raw JSON.
