# Keep question context visible and align the workbench headers

Written against: 1f2dea3e25379f9dfc98cb83c324f00ab5deda37 plus the current uncommitted `playground-demo/` working tree

## Evidence chain

- Surface: `playground-demo/` at `/`, desktop Compare view and the top edge shared by State and Response.
- Problem: the last UI pass placed each question instruction behind `Question details`, but the supplied screenshot and explicit user direction require the instruction to remain visible. A second supplied crop shows the State and Response bottom rules misaligned.
- Design evidence: the supplied screenshots; `renderComparison` in `playground-demo/app.js`; `.panel-heading` at 51 px and `.response-header` at 52 px in `playground-demo/style.css`; the shared heading typography already applied to both panels.
- Owner: `playground-demo/app.js` and `playground-demo/style.css`.
- Scope and affected surfaces: Compare dimension cells and the desktop State/Response header row. Rendered view remains unchanged.
- Uncertainty: none; the user selected the visible-instruction treatment and the source exposes the one-pixel header mismatch directly.

## Design decision

Render question instructions directly beneath the dimension type in Compare, without a disclosure control. Introduce one shared workbench-header height token and use it for both `.panel-heading` and `.response-header` so their bottom rules land on the same pixel. Preserve automatic height at the narrow breakpoint where the Response tabs stack below the title.

## Reuse

- Existing formatter: `instructionsText` in `playground-demo/app.js`.
- Existing presentation: `.dimension-instructions`, shared heading typography, `--rule`, and `--topbar-height` in `playground-demo/style.css`.
- Exemplar: the always-visible question text already used by `.rendered-card header p`.
- No new primitive is required; this plan removes an unnecessary disclosure and consolidates two existing header dimensions under one token.

## Changes

1. `playground-demo/app.js`
   - Change: remove the `<details>` and `Question details` summary from `renderComparison`; append one always-visible `.dimension-instructions` paragraph after the dimension type.
   - Preserve: question key, type badge, `instructionsText`, successful result distributions, provider-failure presentation, and full Rendered view content.
   - Verify: all six Support agent audit instructions are immediately visible without interaction.

2. `playground-demo/style.css`
   - Change: add `--workbench-header-height: 52px` to `:root`; apply it as the desktop height/minimum height of both `.panel-heading` and `.response-header` with border-box sizing.
   - Change: remove disclosure-specific `.dimension-detail` rules and restore the direct `.dimension-instructions` block treatment.
   - Change: at the 620 px response-header stacking rule, reset fixed height to `auto` while retaining a minimum height that accommodates both title and tabs.
   - Preserve: current 22/29/29/20 table widths, sticky Response header behavior, provider colors, and editor-panel borders.
   - Verify: the State and Response bottom rules align exactly at desktop and tablet widths; mobile stacked tabs do not clip.

3. `playground-demo/test_playground_demo.py`
   - Change: remove assertions for native question disclosures; assert that `renderComparison` directly creates `.dimension-instructions` and that both header classes consume `--workbench-header-height`.
   - Preserve: response-alert, provider, toolbar, secret-boundary, and table-width checks.
   - Verify: tests fail if `Question details` returns or the two desktop header heights diverge.

## Scope

- Inherit: every built-in and browser-authored template rendered in Compare.
- Verify: string and object instructions, one through 32 dimensions, desktop, 1120 px, 900 px, and 620 px widths.
- Exclude: model-result calculations, provider failures, JSON editor rendering, pane resizing, Rendered view structure, and Raw JSON.

## Validation

- Product: load Support agent audit and confirm all six instructions are visible while model answers remain unchanged.
- Interface: compare the State and Response border position at desktop and 1120 px; verify the 900 px stacked layout and 620 px Response tabs remain intact.
- System: confirm `instructionsText` remains the only instruction-to-text conversion and no disclosure-specific CSS remains.
- Repository: `node --check playground-demo/app.js` -> exits successfully.
- Repository: `PYTHONPATH=src .venv/bin/pytest -q` -> all tests pass, with only environment-dependent skips.
- Repository: `(cd results/raw && sha256sum -c SHA256SUMS)` -> every artifact reports `OK`.
- Repository: `.venv/bin/python benchmarks/verify_published.py` -> published benchmark verification passes.

## Stop conditions

- Stop if a fixed shared desktop height clips either heading, or if aligning the rules requires changing the topbar or runbar geometry outside this surface.

## Design documentation

- After acceptance and validation: none; this is a local presentation correction owned by the workbench stylesheet.
