# Unify JSON preview and editing, and promote the two-slash SemIf mark

Written against: `1f2dea3e25379f9dfc98cb83c324f00ab5deda37` plus the current uncommitted `playground-demo/` working tree

## Evidence chain

- Surface: the `playground-demo/` topbar, empty response state, and the State/Questions JSON panels at `/`.
- Problem: the topbar uses a circular `S∕` mark while the response empty state already establishes a more distinctive blue/orange-red `//` motif. State and Questions split JSON inspection and editing into separate Tree/Edit modes, forcing a context switch and losing inline syntax/folding while editing. The supplied reference instead combines line numbers, JSON syntax color, editable text, and disclosure controls in one code surface.
- Design evidence: the supplied screenshots; `.brand-mark` and `.empty-rails` in `index.html`/`style.css`; the separate textarea/tree markup and `setJsonMode()` flow; the existing JSON token palette; the viewport-bounded State/Questions panels; CodeMirror 5's documented JSON mode, line-number/fixed-gutter support, and fold gutter/brace-fold addons.
- Owner: `playground-demo/index.html`, `playground-demo/app.js`, `playground-demo/json-workbench.js`, `playground-demo/style.css`, `playground-demo/test_playground_demo.py`, `playground-demo/README.md`, and a new pinned self-hosted `playground-demo/vendor/codemirror-5.65.21/` dependency boundary.
- Scope and affected surfaces: topbar brand mark, empty-response mark, and editable State/Questions surfaces. Raw Response remains a read-only JSON tree because it is output inspection rather than request authoring.
- Uncertainty: CodeMirror 5 is a legacy, now archived project. Version `5.65.21` is selected because its browser-ready distribution can be self-hosted with no runtime build step and it directly supports the requested folding editor. Implementation must stop if its packaged assets, MIT license, or integrity hashes cannot be verified; do not substitute an unpinned CDN or an ad-hoc `contenteditable` editor.

## Design decision

Replace the two-mode State and Questions experience with one editable folding code editor per panel. Use a pinned, self-hosted CodeMirror `5.65.21` distribution configured for JSON, line numbers, fixed gutters, line wrapping, brace matching, and brace-based fold gutters. The CodeMirror document becomes the interactive source of truth; each original `<textarea>` remains in the DOM as the progressive fallback and compatibility field used by `CodeMirror.fromTextArea()`. Template loading, paste formatting, validation, dirty state, request generation, and downloads must read through a small editor-controller API rather than directly from DOM `.value`.

Remove the Tree/Edit segmented controls from State and Questions. Retain the existing compact `+` / `−` controls, now meaning Unfold all and Fold all in the same editable surface. Individual gutter arrows fold or unfold the corresponding object/array; folded content uses a compact `{…}` or `[…]` placeholder without changing the underlying JSON text. Invalid JSON remains editable and visible, disables structurally unsafe whole-document folding where necessary, and continues to block evaluation with the existing inline error.

Replace the topbar's circular `S∕` with the existing blue/orange-red parallel-slash language. Extract one `.slash-mark` primitive used at two sizes: compact beside `SemIf` in the topbar and large in the response empty state. Reuse `--local` and `--jev` so the mark continues to signify the compared readouts; no new logo asset, font, or decorative color is introduced.

## Reuse

- Brand motif: the two skewed bars currently owned by `.empty-rails i`, plus existing `--local` and `--jev` colors.
- Editor content contract: the existing `#state-editor` and `#questions-editor` textareas, `prettyJson()`, `formatValidJson()`, request validation, template loading, dirty-state behavior, field errors, and editor metadata.
- Editor layout: the viewport-bounded `.editor-panel` rows, State/Questions resizer, `--code-line-height`, `--mono`, JSON token colors, and current field-error row.
- Folding actions: existing `.json-tree-actions`, `.json-tree-action`, button labels/tooltips, and per-panel target wiring.
- Read-only output: existing `createJsonTree()` remains the sole Raw Response renderer.
- New dependency boundary: only the verified CodeMirror core, base CSS, JSON-capable JavaScript mode, `foldcode`, `foldgutter`, `brace-fold`, fold-gutter CSS, and upstream MIT license. No CDN or package manager is required at runtime.

## Changes

1. `playground-demo/vendor/codemirror-5.65.21/`
   - Change: add the browser-ready CodeMirror `5.65.21` core JS/CSS, JavaScript JSON mode, fold-code/fold-gutter/brace-fold JS, fold-gutter CSS, upstream `LICENSE`, and a local `SHA256SUMS` covering every vendored file.
   - Change: preserve upstream file contents; keep product-specific styling outside the vendor directory.
   - Preserve: offline/local operation, Content Security Policy `script-src 'self'`, and the static no-build deployment contract.
   - Verify: version banner/package metadata matches `5.65.21`; MIT license is present; hashes verify; no source map, demo, test, unrelated mode, network loader, or transitive runtime dependency is committed.

2. `playground-demo/index.html`
   - Change: replace the `S∕` text/circle with `<span class="slash-mark brand-mark" aria-hidden="true"><i></i><i></i></span>` and change the empty-state bars to the same `.slash-mark` primitive with a large-size modifier.
   - Change: remove State/Questions Tree/Edit mode groups and the separate visible tree containers. Keep one labeled textarea host per panel and keep compact whole-document fold controls in the headings, relabeled Unfold all (`+`) and Fold all (`−`).
   - Change: load the exact local CodeMirror CSS/JS assets in dependency order before `app.js`: core, mode, fold core, brace fold, fold gutter. Do not weaken CSP or add remote origins.
   - Preserve: textarea IDs, error relationships, panel metadata, State/Questions resizer, Raw JSON markup, and all response views.
   - Verify: each editable panel exposes one editor surface; Raw Response still exposes the read-only JSON tree; the page remains usable when enhanced-editor scripts fail and the textarea fallback remains visible.

3. `playground-demo/json-workbench.js`
   - Change: add `createJsonEditor(textarea, options)` as the single adapter around `window.CodeMirror.fromTextArea()`. Its public contract must include `getValue()`, `setValue(value, { preserveHistory, markClean })`, `focus()`, `lineCount()`, `refresh()`, `foldAll()`, `unfoldAll()`, `setInvalid(message)`, `onChange(handler)`, and `formatAfterPaste(handler)`.
   - Change: configure `{ mode: { name: "javascript", json: true }, lineNumbers: true, fixedGutter: true, lineWrapping: true, matchBrackets: true, foldGutter: true, gutters: ["CodeMirror-foldgutter", "CodeMirror-linenumbers"] }` and a compact collection ellipsis widget.
   - Change: make programmatic template loads transactional: set pretty JSON, clear stale fold/error state, clear history when requested, and suppress dirty callbacks until the operation completes.
   - Change: implement Fold all/Unfold all against CodeMirror's fold API while preserving the document bytes, selection, undo history, and scroll position. Folding must be presentation-only.
   - Change: preserve the existing `createJsonTree()` renderer for Raw Response and keep `prettyJson()` / `formatValidJson()` framework-independent.
   - Preserve: safe `textContent` rendering in the Raw tree and no HTML interpolation of JSON values.
   - Verify: editor-controller methods never reach into product state; two instances remain isolated; a missing `window.CodeMirror` returns a textarea-backed fallback controller rather than breaking application startup.

4. `playground-demo/app.js`
   - Change: replace `initializeEditableJsonSurface()`'s separate tree/textarea state with one `createJsonEditor()` controller. Remove `setJsonMode()`, mode-button wiring, editable-tree rendering, and duplicate tree refresh calls.
   - Change: route `requestFromEditors()`, metadata, current-template export, template apply/import/save, validation focus, and paste formatting through controller methods. No request path may read a stale hidden textarea value.
   - Change: retain current behavior for valid full-JSON paste: format to two spaces after paste, preserve an actionable cursor position, mark dirty once, and keep the editor editable. Leave invalid pasted text unchanged and report the error only at the existing validation points.
   - Change: wire `+` to `unfoldAll()` and `−` to `foldAll()` for the targeted controller. Disable Fold all when the current document cannot be parsed as JSON; individual editor behavior remains available for repair.
   - Change: after State/Questions pane resize, responsive breakpoint change, template switch, or panel visibility change, call the editor's `refresh()` so gutters, wrapping, and scrollbars use the actual panel dimensions.
   - Preserve: exact request JSON values, template dirty confirmation, browser template storage, response reset rules, keyboard run/save shortcuts, pane-resizer semantics, and Raw inspector behavior.
   - Verify: typing immediately affects the next request/export; folding never affects the request; undo/redo works after manual edits; template replacement does not create false dirty state; State actions never affect Questions.

5. `playground-demo/style.css`
   - Change: extract `.slash-mark` from `.empty-rails`, parameterized with custom properties for width, height, gap, and bar width. Size the topbar mark without the old border/circle; retain the larger centered empty-state version.
   - Change: theme CodeMirror using existing SemIf tokens: white surface, `--surface-soft` fixed gutters, `--rule` divider, `--mono`, `--code-line-height`, current JSON key/string/number/boolean/null colors, subtle active line, and `--local` focus rail.
   - Change: make `.CodeMirror`, `.CodeMirror-scroll`, and wrapper containers fill the panel's `minmax(0, 1fr)` row without increasing page height. Fixed gutters must remain clear during horizontal scroll, and wrapped lines must align beneath code rather than beneath line numbers.
   - Change: style fold gutter arrows and `{…}`/`[…]` markers to match the quiet disclosure language in the reference; maintain visible focus and invalid states without overlaying selection/caret.
   - Change: remove obsolete editable-tree/mode-switch styling only after confirming it has no Raw Response consumer. Keep Raw tree token styles under explicit `.json-tree` ownership.
   - Preserve: responsive toolbar, pane splitters, panel header height, field-error row, Raw response styling, and 620/900 px layouts.
   - Verify: no line-number/content overlap, clipped caret, double scrollbar, page-height growth, or header collision at 1600, 1366, 1120, 900, and 620 px.

6. `playground-demo/test_playground_demo.py`
   - Change: verify all vendored hashes and the CodeMirror license/version pin; assert every asset is local and loaded before the module application script.
   - Change: replace Tree/Edit-mode assertions with the unified-editor contract: one controller per textarea, JSON mode, fixed line/fold gutters, fold/unfold isolation, fallback controller, and no editable `createJsonTree()` instances.
   - Change: cover value synchronization for manual edits, programmatic template loads, undo/redo, valid/invalid paste, request creation, template export, dirty-state suppression, invalid-state folding, and resize refresh.
   - Change: assert one `.slash-mark` primitive serves both topbar and empty response, with existing local/Jev color tokens and no image/network asset.
   - Preserve: provider, cURL placeholder, Raw Response tree, secret boundary, pane sizing, template validity, benchmark, and model tests.
   - Verify: tests fail if folded content disappears from payloads, CodeMirror assets load remotely, textarea and editor values diverge, or product CSS modifies vendored files.

7. `playground-demo/README.md`
   - Change: replace Tree/Edit instructions with the unified editable folding editor workflow, per-branch gutter controls, and whole-document `+` / `−` behavior.
   - Change: document CodeMirror `5.65.21` as a pinned, MIT-licensed, self-hosted legacy dependency selected for the no-build static demo; include the integrity-check command and fallback behavior.
   - Change: document that folding affects presentation only and that Raw Response remains a separate read-only tree.
   - Preserve: JSON formatting/validation behavior, template workflow, viewport/pane sizing, API/provider documentation, and secret guarantees.
   - Verify: documentation does not claim CodeMirror 5 is actively maintained or imply that folding changes request data.

8. `playground-demo/docs/design-plans/semif-json-workbench.md`
   - Change: add a reconciliation note that its completed State/Questions disclosure controls are retained, while its separate Tree-mode context is superseded by this unified-editor plan.
   - Preserve: the historical rationale and validation record for whole-document disclosure actions.
   - Verify: the two plans no longer prescribe conflicting editable-surface architectures.

## Scope

- Inherit: every built-in, imported, and browser-saved State/Questions template; long resume state; nested support-agent events; all editable JSON types and Unicode strings.
- Verify: empty objects/arrays, deeply nested collections, long wrapped strings, apostrophes, multi-byte text, selection across wrapped lines, mouse and keyboard folding, invalid JSON repair, undo/redo, full-value paste, pane resizing, template replacement, and browser zoom.
- Exclude: making Raw Response editable, JSON schema completion, lint gutters beyond the existing parse error, search/replace UI, multi-file tabs, minimap, diff view, remote/CDN assets, a general code-editor framework migration, favicon/app-icon work, and backend changes.

## Validation

- Product: edit both State and Questions, fold nested objects, run/export the request, and compare the resulting values byte-for-byte with the unfolded documents; switch templates and verify clean state/history; paste valid and invalid JSON; repair an error and rerun.
- Interface: verify the compact and large `//` marks, cursor/selection, syntax colors, line numbers, fold gutters, placeholders, focus/invalid states, horizontal/wrapped content, and independent panel scrolling at 1600, 1366, 1120, 900, and 620 px.
- Accessibility: keyboard into each editor; move, select, edit, undo, fold/unfold, and reach heading actions without a pointer; verify the decorative slash bars are ignored and editor labels/errors remain associated.
- Dependency: `cd playground-demo/vendor/codemirror-5.65.21 && sha256sum -c SHA256SUMS` -> every vendored asset reports `OK`; scan HTML/CSS/JS for remote CodeMirror URLs -> none.
- System: confirm CodeMirror owns only State/Questions editing, `createJsonTree()` owns only Raw Response, original textareas provide fallback compatibility, and neither surface duplicates JSON state.
- Repository: `node --check playground-demo/json-workbench.js && node --check playground-demo/app.js` -> both exit successfully.
- Repository: `PYTHONPATH=src .venv/bin/pytest -q` -> all tests pass, with only environment-dependent skips.
- Repository: `(cd results/raw && sha256sum -c SHA256SUMS)` -> every benchmark artifact reports `OK`.
- Repository: `.venv/bin/python benchmarks/verify_published.py` -> published benchmark verification passes.

## Stop conditions

- Stop if CodeMirror assets or license cannot be verified, if CSP must be weakened, if implementation requires a runtime build/CDN, if the textarea fallback cannot remain functional, if folds mutate submitted JSON, or if Raw Response would have to become editable to share the implementation.

## Design documentation

- After acceptance and validation: update the JSON workbench/dependency sections of `playground-demo/README.md` with the unified editor, fold semantics, dependency version/license/integrity, fallback, and Raw Response boundary.
