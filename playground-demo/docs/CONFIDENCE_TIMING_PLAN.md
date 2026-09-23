# Jev timing and local distribution confidence plan

## Goal

Make the comparison summary report elapsed time for both local MLX and Jev,
and expose a useful single-number concentration metric for local Choice and
Score distributions without presenting it as calibrated correctness.

## Verified starting point

- The installed TypeSafe SDK `SystemOneResponse` contains `model`, `usage`,
  and `answers`, but no request-duration field. Jev elapsed time therefore has
  to be measured around the SDK call by the loopback server.
- SemIf already returns a softmax distribution over the declared options.
  It does not currently return a separate confidence statistic.
- TypeSafe documents its Choice and Score confidence as a statistic derived
  from the option distribution, but does not publish the exact formula on its
  confidence page. SemIf must not claim that its derived value is equivalent.

## Implementation

1. Add a tested normalized-Shannon-entropy helper to SemIf core:
   `1 - H(p) / log(N)`.
2. Add that value to local Choice and Score playground answers together with
   machine-readable method and calibration-status fields. Leave Noul unchanged:
   its `noul` value is already the binary true probability.
3. Measure Jev client wall-clock duration with a monotonic clock and expose it
   through the same `timing.total_seconds` shape used by local MLX.
4. Render local entropy confidence and Jev provider confidence with distinct,
   explicit labels.
5. Validate unit tests, repository integrity checks, JavaScript syntax, and the
   comparison flow in a real browser against an isolated test server.

## Acceptance boundaries

- Local entropy confidence measures how concentrated the declared-option
  distribution is. It is not calibrated probability that the selected answer
  is correct.
- Jev elapsed time is end-to-end time observed by the local server, including
  network and SDK overhead. It is not provider-side model latency.
- No benchmark evidence, headline result, or checked-in raw report is changed.

## Validation completed

- `pytest -q`: 88 passed, 3 skipped.
- `sha256sum -c SHA256SUMS` in `results/raw`: every published artifact passed.
- `python benchmarks/verify_published.py`: 69 summary claims verified.
- `node --check playground-demo/app.js`: passed.
- Isolated browser fixture: rendered `Local: 5.141 s · Jev: 0.842 s`, local
  entropy confidence with its calibration caveat, and separate Jev confidence.
