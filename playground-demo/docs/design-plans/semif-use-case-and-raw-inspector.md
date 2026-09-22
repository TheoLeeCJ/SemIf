# Polish use-case selection and turn Raw JSON into a sticky request/response inspector

Written against: `1f2dea3e25379f9dfc98cb83c324f00ab5deda37` plus the current uncommitted `playground-demo/` working tree

## Evidence chain

- Surface: the use-case picker in the `playground-demo/` top toolbar and the `Raw JSON` response tab at `/`.
- Problem: the closed use-case field is styled, but opening it exposes the browser's native grouped menu, which does not match the polished model checkpoint selector. In Raw JSON, the request and both responses are combined into one scrolling tree; its toolbar scrolls away; there is no shareable request representation; and its text Expand/Collapse buttons conflict with the compact `+` / `−` controls already used by State and Questions.
- Design evidence: the supplied screenshots; `.template-picker` and `renderTemplateOptions()` currently own a native grouped `<select>`; `.model-picker-field`, `.select-shell`, and the model control focus treatment establish the desired field appearance; `.raw-toolbar` currently precedes a single `rawJsonTree` rendered from all of `app.response`; State and Questions already use the shared `.json-tree-actions` controls.
- Owner: `playground-demo/index.html`, `playground-demo/app.js`, `playground-demo/style.css`, `playground-demo/test_playground_demo.py`, and `playground-demo/README.md`. The recursive JSON renderer remains owned by `playground-demo/json-workbench.js`.
- Scope and affected surfaces: use-case selection and the Raw JSON tab only. Template data, evaluation routes, model scoring, provider configuration, Compare, Rendered, State, and Questions behavior remain unchanged.
- Uncertainty: TypeSafe's published quickstart defines `POST https://api.typesafe.ai/v1/systemone` with `Authorization: Bearer <API_KEY>`. The Bocha guide was not retrievable during planning, so its direct example must be validated against the configured `https://jev.bocha.cn` base URL and the repository's same-SDK integration before release; do not silently invent a different transport if that check fails.

## Design decision

Replace the native use-case dropdown presentation with a keyboard-accessible, single-select listbox whose trigger reuses the model selector's shell, chevron, border, hover, and focus language at toolbar density. Keep `#template-select` as a visually hidden synchronized native control so existing template application, form semantics, tests, and programmatic selection have one source of truth. The popover will render category headings and options as normal DOM, allowing consistent spacing, typography, active/selected states, and responsive containment instead of relying on an unstyleable operating-system menu.

Turn Raw JSON into a two-mode inspector with a sticky control row: `Request` and `Response`. Default to Response. Response renders only `challenger` and `jev` result state—never the submitted request—while preserving pending, error, timing, provider, usage, and answer diagnostics. Request renders executable cURL for the selected comparison pair. Local MLX uses the demo's loopback `/api/evaluate/local` route. Jev official uses TypeSafe's documented `https://api.typesafe.ai/v1/systemone` endpoint and `Authorization: Bearer <TYPESAFE_API_KEY>`. Jev Bocha uses the configured `https://jev.bocha.cn/v1/systemone` endpoint and `Authorization: Bearer <BOCHA_JEV_API_KEY>` once that same-SDK contract is confirmed. The official and Bocha bodies omit the playground-only `local_model` field; Bocha uses its configured non-secret model alias. No generated command reads or reveals a configured credential.

Keep the Request/Response switcher, context title, and actions sticky at the top of the Raw viewport while only the cURL or JSON content scrolls. In Response mode expose Copy JSON plus the same compact `+` / `−` Expand all and Collapse all controls used by State and Questions. In Request mode replace those tree actions with Copy cURL. All disclosure controls share one component style, symbols, tooltip wording, accessible labels, focus treatment, and disabled/hidden behavior.

## Reuse

- Closed-field visual language: `.select-shell`, its CSS chevron, model selector border radius, hover border, and focus ring.
- Existing template source of truth: `#template-select`, `app.templates`, `renderTemplateOptions()`, and `applyTemplate()`.
- Existing segmented-control language: `.view-tabs`, `.tab`, and `.json-mode-tabs`, adapted to a compact Raw inspector switcher.
- Existing JSON behavior: `createJsonTree()` with `render()`, `expandAll()`, and `collapseAll()`.
- Existing disclosure control: `.json-tree-actions` and `.json-tree-action` from State and Questions.
- Existing request contract: `requestFromEditors()`, `selectedChallenger()`, `/api/evaluate/local`, `/api/evaluate/bocha`, and `/api/evaluate/typesafe`.
- No second JSON renderer, template registry, evaluation client, or credential store is introduced.

## Changes

1. `playground-demo/index.html`
   - Change: add a styled use-case combobox trigger and categorized listbox popover adjacent to the hidden synchronized `#template-select`; include `aria-haspopup="listbox"`, `aria-expanded`, `aria-controls`, listbox/option roles, and a live trigger label.
   - Change: replace the Raw `Request and responses` header with a sticky inspector bar containing a Request/Response segmented switcher, a context label, and context-specific actions.
   - Change: make Response the initially selected inspector mode. Keep one response JSON tree and add a semantic, read-only cURL code surface for Request.
   - Change: render compact `+` and `−` Raw tree controls with the same title and accessible-label pattern as State and Questions; retain a text Copy JSON/Copy cURL action.
   - Preserve: the top-level Compare/Rendered/Raw JSON tabs, response summary, provider alert, all existing IDs needed by request execution, and the native template value as the compatibility/source-of-truth control.
   - Verify: the use-case trigger has exactly one associated listbox; Raw exposes exactly two modes; the request body and response tree are never simultaneously shown.

2. `playground-demo/app.js`
   - Change: refactor `renderTemplateOptions()` to populate both the hidden native select and the styled listbox from the same ordered category map; update selected state, trigger text, `aria-selected`, and active descendant from `app.activeTemplateId`.
   - Change: add combobox/listbox behavior for pointer selection, Arrow Up/Down, Home/End, Enter/Space, Escape, Tab, outside click, focus return, and viewport-aware open/close. Selecting an item must call the existing `applyTemplate()` path, including its dirty-editor confirmation and rollback.
   - Change: add a Raw inspector mode state independent of the top-level response view. Default and reset it to `response`; preserve the selected mode while asynchronous result slots update.
   - Change: derive the response tree through one helper returning `{ challenger, jev }` only. Use this same response-only value for rendering and Copy JSON so the request cannot leak back into Response through a parallel code path.
   - Change: generate cURL from `app.response.request` and the current challenger. Emit two labeled commands: Local MLX or Jev Bocha, then Jev official. Local MLX uses the current loopback origin and `/api/evaluate/local` with the exact playground body. Official uses `https://api.typesafe.ai/v1/systemone`; Bocha uses `https://jev.bocha.cn/v1/systemone` after contract validation. Both direct provider commands include `Content-Type: application/json` and an `Authorization: Bearer <...>` placeholder.
   - Change: derive provider-facing request bodies through one explicit sanitizer: keep `state` and `questions`, set the correct provider model, and omit playground-only `local_model`. Use only the literal placeholders `<BOCHA_JEV_API_KEY>` and `<TYPESAFE_API_KEY>`; never read, interpolate, render, or copy configured credentials or credential-source values.
   - Change: shell-quote the JSON body with a dedicated POSIX single-quote helper so apostrophes in state or questions produce valid cURL. Render cURL with `textContent`, never HTML interpolation.
   - Change: wire Raw `+` / `−` to the existing `rawJsonTree` controller only in Response mode; make the Copy action context-aware and copy exactly the visible JSON or cURL.
   - Preserve: parallel comparison execution, `app.response.request` for diagnostics and cURL generation, pending/error results, provider timing, request validation, and server-side secret handling.
   - Verify: response JSON has no `request` key; each cURL command targets the correct provider and reproduces its evaluated state/questions; provider cURL omits `local_model`; a state containing `'` remains valid shell; no real key string can enter the DOM or clipboard.

3. `playground-demo/style.css`
   - Change: create a toolbar-density variant of the model selector shell for the use-case trigger, matching its custom chevron, border radius, hover border, and focus ring while retaining the current 36 px topbar height.
   - Change: style the use-case listbox as an anchored surface with category captions, 32–36 px options, selected check/rail, hover and keyboard-active states, bounded height, internal scrolling, and the existing surface/rule/shadow tokens.
   - Change: constrain the popover to the viewport and keep it usable in the existing 1120 px, 900 px, and 620 px toolbar layouts; at small widths it aligns to the selector track and never creates page-level horizontal overflow.
   - Change: make the Raw inspector bar `position: sticky; top: 0; z-index` within `.response-body`, with an opaque surface and bottom rule so JSON/cURL scroll beneath without visual bleed.
   - Change: style the Request/Response switcher using the existing compact segmented-control language; reuse `.json-tree-actions` / `.json-tree-action` unchanged for Raw and remove the conflicting text Expand all/Collapse all variant.
   - Change: style cURL as a selectable, horizontally scrollable monospace code block with wrapped labels but unwrapped command lines; keep the content clear of the sticky bar and runbar.
   - Preserve: viewport-bounded workbench sizing, response-header alignment, pane splitters, JSON syntax colors, and all provider identity colors.
   - Verify: the listbox and inspector bar do not clip or overlap at 1600, 1366, 1120, 900, and 620 px; only the content region scrolls; all focus rings remain visible.

4. `playground-demo/test_playground_demo.py`
   - Change: assert the synchronized native selector plus accessible listbox contract, common option source, dirty-state rollback, keyboard handlers, and outside-dismiss behavior.
   - Change: assert that Raw defaults to Response, its render/copy helper excludes `request`, and the Request view produces both currently selected routes with the exact request body.
   - Change: assert the only credential strings allowed in cURL output are the literal placeholders and that configured keys/health values are never read into the renderer.
   - Change: cover POSIX escaping for apostrophes, empty/unrun request presentation, pending/success/error response updates, context-specific copy labels, and shared compact disclosure actions.
   - Change: assert sticky inspector CSS, popover viewport containment, and responsive selector behavior.
   - Preserve: existing template loading, JSON safety, provider, local-model, response rendering, splitter, and secret-boundary assertions.
   - Verify: tests fail if the Response view includes the request, if cURL includes a real key, if a third independent option registry appears, or if Raw returns to text-only disclosure buttons.

5. `playground-demo/README.md`
   - Change: document the styled categorized use-case picker and its keyboard behavior.
   - Change: document that Raw JSON defaults to response-only data and that Request provides copyable Local MLX or provider-facing cURL for both comparison calls with credential placeholders.
   - Change: explicitly state that cURL placeholders are examples, not configured credentials, and that the browser never receives provider API keys.
   - Change: describe the shared `+` / `−` whole-tree actions across State, Questions, and Raw Response.
   - Preserve: current TypeSafe/Bocha SDK guidance, server routes, template format, and credential lifecycle statements.
   - Verify: documentation distinguishes the Local loopback command from provider-facing Jev commands and does not imply that a placeholder is a configured secret.

## Scope

- Inherit: built-in, imported, and browser-saved templates; Local MLX and Jev Bocha challenger selection; official Jev comparison; pending, successful, and failed runs.
- Verify: long template/category names, duplicate display names in different categories, dirty-editor cancellation, rapid open/close, keyboard-only selection, apostrophes/newlines/Unicode in JSON, one failed provider, narrow toolbar layouts, and large response trees.
- Exclude: exposing configured credentials, reverse-engineering SDK-private behavior beyond the documented REST contract, changing API routes, changing comparison execution, adding a third simultaneous result, JSON search, persistent Raw mode, or redesigning the model selector itself.

## Validation

- Product: choose a template by pointer and keyboard, cancel a dirty-editor replacement, complete Local-vs-official and Bocha-vs-official runs, then verify Request shows the two correct cURL commands and Response shows only provider results.
- Interface: scroll a long Raw response and a long cURL at 1600, 1366, 1120, 900, and 620 px; expect the inspector controls to remain pinned, content to stay below them, the use-case listbox to stay inside the viewport, and every disclosure action to use the same compact style.
- Security: configure recognizable test keys, open/copy both Raw modes, and search the DOM/clipboard output; expect only `<BOCHA_JEV_API_KEY>` and `<TYPESAFE_API_KEY>` placeholders and no configured key bytes. Compare the official command with the published TypeSafe quickstart and validate the Bocha command against its provider guide or a non-secret captured request shape.
- System: confirm one template map populates both selector representations, one JSON renderer owns all tree disclosure, and the existing evaluation routes remain the sole request transport.
- Repository: `node --check playground-demo/json-workbench.js && node --check playground-demo/app.js` -> both exit successfully.
- Repository: `PYTHONPATH=src .venv/bin/pytest -q` -> all tests pass, with only environment-dependent skips.
- Repository: `(cd results/raw && sha256sum -c SHA256SUMS)` -> every artifact reports `OK`.
- Repository: `.venv/bin/python benchmarks/verify_published.py` -> published benchmark verification passes.

## Stop conditions

- Stop if preserving the native selector as synchronized source of truth creates two independently mutable template states, if Bocha's provider contract cannot be verified without reverse-engineering private SDK behavior, if any implementation path would expose configured credentials to browser code, or if sticky positioning requires breaking the viewport-bounded response scroll owner.

## Design documentation

- After acceptance and validation: update the use-case picker, Raw inspector, cURL placeholder, and shared disclosure-control contracts in `playground-demo/README.md`.
