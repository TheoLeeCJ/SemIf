# Add complete-tree disclosure controls to State and Questions

> Reconciled 2026-09-22: this plan's whole-document disclosure controls are implemented and retained. The later selected change in `semif-unified-json-editor-and-mark.md` supersedes the separate Tree/Edit presentation for State and Questions with one editable folding JSON surface.

Written against: `1f2dea3e25379f9dfc98cb83c324f00ab5deda37` plus the current uncommitted `playground-demo/` working tree

## Evidence chain

- Surface: State and Questions Tree modes in `playground-demo/` at `/`.
- Problem: each object and array already has a native disclosure, but only Raw JSON provides `Expand all` and `Collapse all`. The user explicitly requested equivalent whole-tree controls for State and Questions.
- Design evidence: Tree/Edit controls in `index.html`; `createJsonTree()` already exposes `expandAll()` and `collapseAll()` in `json-workbench.js`; Raw JSON already wires those methods in `app.js` and uses compact quiet buttons in `style.css`.
- Owner: `playground-demo/index.html`, `playground-demo/app.js`, and `playground-demo/style.css`; renderer behavior remains owned by `playground-demo/json-workbench.js`.
- Scope and affected surfaces: State and Questions Tree modes only. Edit mode, parsing, formatting, request payloads, and Raw JSON behavior remain unchanged.
- Uncertainty: none. The required renderer API and an existing response-surface exemplar already exist.

## Design decision

Add two compact, icon-like buttons to each State/Questions heading: `+` for Expand all and `-` for Collapse all, each with an explicit accessible label and tooltip. Keep them next to the existing Tree/Edit mode group so they are associated with the current JSON surface without adding another content row. Enable and show them only while that panel is in Tree mode. Reuse the existing controller methods; disclosure changes presentation only and never update the textarea, dirty state, or evaluation payload.

## Reuse

- Existing renderer/controller: `createJsonTree()` and its `expandAll()` / `collapseAll()` methods in `playground-demo/json-workbench.js`.
- Existing interaction exemplar: Raw JSON controls and event handlers.
- Existing controls/tokens: `.json-mode-tabs`, `.json-mode-button`, `.quiet-button`, `--surface`, `--surface-soft`, `--rule`, and the existing focus-visible treatment.
- No new renderer primitive is required.

## Changes

1. `playground-demo/index.html`
   - Change: add Expand all and Collapse all buttons for State and Questions within each `.panel-heading-tools` group, using `data-json-action`, the matching `data-json-target`, `title`, and `aria-label`.
   - Change: use compact visible symbols while keeping the full action names available to assistive technology and pointer hover.
   - Preserve: Tree/Edit buttons, metadata, editor/tree IDs, field-error relationships, and Raw JSON controls.
   - Verify: every editable JSON surface has exactly one expand and one collapse action tied to the correct target.

2. `playground-demo/app.js`
   - Change: bind `data-json-action="expand"` and `"collapse"` to the existing surface controller methods.
   - Change: update action visibility or disabled state inside `setJsonMode()` so controls cannot operate on a hidden tree in Edit mode.
   - Preserve: lazy branch materialization, current default-open depths, textarea source-of-truth behavior, template dirty state, paste formatting, and request serialization.
   - Verify: expanding or collapsing either tree leaves the editor text byte-for-byte unchanged and affects no other JSON surface.

3. `playground-demo/style.css`
   - Change: style the two actions as compact 24 px controls consistent with the Tree/Edit segment and existing focus treatment; prevent them from increasing the fixed 52 px header.
   - Change: keep headings collision-free at the minimum request-pane width; hide the nonessential line/dimension metadata before truncating the title or JSON controls when space is constrained.
   - Preserve: syntax colors, disclosure indentation, panel alignment, and mode active states.
   - Verify: State and Questions headers remain one line at the minimum desktop pane width and remain usable at 900 px and 620 px.

4. `playground-demo/test_playground_demo.py`
   - Change: assert all four controls, correct target/action attributes, controller wiring, mode-aware state, and reuse of the existing tree methods.
   - Preserve: current safe-rendering, paste-formatting, gutter, Raw JSON, provider, and secret-boundary assertions.
   - Verify: tests fail if an action targets the other panel or starts mutating editor content.

5. `playground-demo/README.md`
   - Change: state that State and Questions provide per-branch disclosure plus whole-tree Expand all/Collapse all controls in Tree mode.
   - Preserve: existing statements that disclosure never changes request data.
   - Verify: documentation distinguishes Tree presentation from Edit content.

## Scope

- Inherit: built-in, imported, and browser templates in State and Questions Tree modes.
- Verify: deeply nested objects, arrays, empty collections, switching Tree/Edit after collapse, mode changes after invalid edits, and minimum-width headers.
- Exclude: expand/collapse depth selectors, remembering disclosure across template changes or reloads, tree editing, JSON search, Raw JSON redesign, and backend changes.

## Validation

- Product: collapse State completely, expand Questions completely, switch both through Edit and back to Tree, and verify each controller remains isolated and parsed values are unchanged.
- Interface: verify button labels/tooltips, focus states, disabled/hidden Edit-mode state, header density, and response to the resized request pane.
- System: confirm all three JSON trees continue to use the single self-hosted renderer and no duplicate recursive traversal is added to `app.js`.
- Repository: `node --check playground-demo/json-workbench.js && node --check playground-demo/app.js` -> both exit successfully.
- Repository: `PYTHONPATH=src .venv/bin/pytest -q` -> all tests pass, with only environment-dependent skips.
- Repository: `(cd results/raw && sha256sum -c SHA256SUMS)` -> every artifact reports `OK`.
- Repository: `.venv/bin/python benchmarks/verify_published.py` -> published benchmark verification passes.

## Stop conditions

- Stop if the controls cannot fit the 52 px heading at the minimum pane width without hiding a primary mode control, or if controller reuse would require synchronizing presentation state into the request payload.

## Design documentation

- After acceptance and validation: update the JSON workbench section of `playground-demo/README.md` with the per-panel whole-tree controls.
