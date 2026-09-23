# Constrain the workbench to the viewport and split State from Questions

Written against: `1f2dea3e25379f9dfc98cb83c324f00ab5deda37` plus the current uncommitted `playground-demo/` working tree

## Evidence chain

- Surface: `playground-demo/` at `/`, including the top toolbar, request pane, response pane, and fixed run bar.
- Problem: the workbench uses minimum heights, so expanded JSON and response content can increase the document beyond the viewport. State and Questions also use a fixed `1fr / 1.1fr` split and cannot be resized. The existing request/response rail is visually wider than the user wants.
- Design evidence: `.workbench`, `.input-pane`, `.editor-panel`, `.response-pane`, `.response-body`, `.runbar`, and the existing `#workbench-resizer` implementation in `style.css`, `index.html`, and `app.js`; the user explicitly requested viewport-bounded height, a half/half State/Questions split, and thinner resize treatment.
- Owner: `playground-demo/index.html`, `playground-demo/style.css`, and `playground-demo/app.js`.
- Scope and affected surfaces: the workbench viewport model, both request editors, all response tabs, and the existing request/response splitter. The Setup drawer and Template studio remain independently sized overlays.
- Uncertainty: none. Below 900 px, the request and response sections must remain vertically sequential, so the workbench itself will become the vertical scroll container while the overall document remains viewport-bounded.

## Design decision

Make the application shell exactly one dynamic viewport high: the toolbar occupies `--topbar-height`, the fixed run bar occupies `--runbar-height`, and the workbench owns the remaining height without allowing the document body to grow. On desktop, State, Questions, and Response content scroll inside their own existing content areas. At 900 px and below, keep the current stacked request/response order but scroll that stack inside the bounded workbench.

Add a semantic horizontal separator between State and Questions. Their default grid tracks are equal (`1fr / 1fr`), with the rail between them. Pointer drag adjusts only the State track, Up/Down arrows move it by 16 px, Shift+Arrow by 48 px, and double-click restores the equal split. Clamp both panels to a minimum height that preserves the 52 px header and a usable content region.

Reduce both resize rails to a 5 px layout track with a 1 px visible rule. Preserve a larger invisible pointer target using an overlapping pseudo-element so the rails look thinner without becoming difficult to grab.

## Reuse

- Existing owners: `.workbench`, `.input-pane`, `.editor-panel`, `.response-pane`, `.response-body`, and `.runbar` in `playground-demo/style.css`.
- Existing interaction exemplar: `initializeWorkbenchResizer()` in `playground-demo/app.js`.
- Existing tokens: `--topbar-height`, `--runbar-height`, `--workbench-header-height`, `--rule`, `--rule-strong`, and `--local`.
- New primitive: `#editor-resizer` is required because State and Questions are separate grid tracks with no current controller. It belongs only to `.input-pane` and mirrors the established request/response separator contract.

## Changes

1. `playground-demo/index.html`
   - Change: insert `#editor-resizer` between `.state-panel` and `.questions-panel` with `role="separator"`, `aria-orientation="horizontal"`, keyboard focus, percentage value attributes, and a concise label.
   - Preserve: panel article ownership, editor/tree IDs, labels, validation messages, and request/response separator semantics.
   - Verify: `.input-pane` has exactly State, the horizontal separator, and Questions in that order.

2. `playground-demo/style.css`
   - Change: bind `html` and `body` to `100%`/`100dvh` and prevent document-level overflow; give `.workbench` an exact height of `calc(100dvh - var(--topbar-height))` including its run-bar clearance.
   - Change: keep desktop `.workbench` overflow hidden; at 900 px and below, preserve the stacked layout and make `.workbench` the vertical scroll container so the page itself never exceeds the viewport.
   - Change: make `.input-pane` and `.response-pane` consume one available workbench height. Convert `.response-pane` to a header/content grid and put scrolling on `.response-body`; keep each State/Questions editor or tree as its panel-local scroller.
   - Change: replace the current State/Questions row ratio with `minmax(var(--editor-pane-min), var(--state-pane-height))`, a 5 px horizontal separator, and `minmax(var(--editor-pane-min), 1fr)`, where `--state-pane-height` defaults to `1fr` for an equal split.
   - Change: reduce `--splitter-size` from 9 px to 5 px, use a 1 px visible center rule for both rails, and extend only the invisible pointer target to approximately 9 px.
   - Preserve: the 46.5% default request/response width, pane minimum widths, the fixed run bar, header alignment, breakpoint order, and mobile content order.
   - Verify: no page-level vertical or horizontal scrollbar appears at desktop, 1120 px, 900 px, or 620 px; the intended local or stacked scroll container remains usable.

3. `playground-demo/app.js`
   - Change: factor the shared pointer-capture, clamping, keyboard-step, active-state, and double-click-reset behavior so the existing vertical separator and new horizontal separator follow one interaction contract without sharing dimensions.
   - Change: calculate State height relative to `.input-pane`, clamp both editor panels to `--editor-pane-min`, update `--state-pane-height` and `aria-valuenow`, and reset to the CSS equal split on double-click.
   - Change: preserve the request/response separator's Left/Right behavior and reset its inline size when entering the stacked breakpoint; do not clear the State/Questions split merely because the outer layout stacks.
   - Preserve: editor values, JSON disclosure state, template state, response state, and run behavior during every resize.
   - Verify: drag, keyboard adjustment, reset, viewport resizing, and mode switching never mutate request data.

4. `playground-demo/test_playground_demo.py`
   - Change: assert viewport-bound shell ownership, panel-local response scrolling, the 5 px/1 px rail treatment, horizontal separator markup, equal default tracks, minimum-height clamping, pointer and keyboard directions, and both reset paths.
   - Preserve: existing static-surface, model, provider, template, and secret-boundary checks.
   - Verify: tests fail if the body becomes the desktop scroll owner, either separator loses its semantic orientation, or the default State/Questions split stops being equal.

5. `playground-demo/README.md`
   - Change: document that the workbench is viewport-bounded, which regions scroll, and how both resize rails work and reset.
   - Preserve: existing provider, JSON, template, model, and credential guidance.
   - Verify: documentation does not claim pane sizes persist across reloads.

## Scope

- Inherit: all built-in and browser templates, Tree and Edit modes, empty/running/success/error response states, Compare/Rendered/Raw tabs, and every local model selection.
- Verify: desktop, 1440 px, 1120 px, just above 900 px, 900 px, and 620 px; short and deeply expanded JSON; long comparison tables; pointer and keyboard input.
- Exclude: persisted pane sizes, touch-only resize gestures below the pointer contract, comparison-column resizing, Setup drawer sizing, Template studio sizing, and changes to provider/model behavior.

## Validation

- Product: expand deeply nested State and Questions JSON and run a multi-dimension comparison; expect the browser document to remain one viewport high while each intended region scrolls independently.
- Product: drag State almost to each bound, use Up/Down and Shift+Arrow, then double-click; expect both panels usable and restored to an equal split.
- Product: drag the request/response rail; expect the slimmer rule to remain easy to acquire and preserve the current width constraints.
- Interface: inspect scroll ownership, rail hover/focus/active states, equal split, and stacked behavior at desktop, 1440 px, 1120 px, 900 px, and 620 px.
- System: confirm CSS variables remain the sole pane-size owners and no editor or response value is copied into resizing state.
- Repository: `node --check playground-demo/app.js` -> exits successfully.
- Repository: `PYTHONPATH=src .venv/bin/pytest -q` -> all tests pass, with only environment-dependent skips.
- Repository: `(cd results/raw && sha256sum -c SHA256SUMS)` -> every artifact reports `OK`.
- Repository: `.venv/bin/python benchmarks/verify_published.py` -> published benchmark verification passes.

## Stop conditions

- Stop if viewport bounding would hide content without an explicit local or workbench scroll owner, if a minimum panel height cannot preserve the 52 px header plus usable content, or if the larger invisible hit target intercepts normal pane interactions.

## Design documentation

- After acceptance and validation: update the JSON workbench and pane-sizing section of `playground-demo/README.md` with the viewport and two-separator behavior.
