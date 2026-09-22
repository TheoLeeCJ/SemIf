# Make the top toolbar collision-proof across supported widths

Written against: `1f2dea3e25379f9dfc98cb83c324f00ab5deda37` plus the current uncommitted `playground-demo/` working tree

## Evidence chain

- Surface: the `playground-demo/` top toolbar at `/`.
- Problem: in the supplied 1400 px screenshot, the use-case selector and Local status chip overlap because the desktop grid gives the brand and service strip equal flexible tracks even though the service strip has substantially more intrinsic width. Smaller breakpoints compact the chips, but the collision happens before those rules activate.
- Design evidence: the supplied screenshot; `.topbar` currently uses `minmax(220px, 1fr) minmax(240px, 430px) minmax(420px, 1fr)`; `.service-strip` contains three non-wrapping provider chips plus Templates and Setup; existing breakpoints compact provider chips at 1120 px, move the selector to a second row at 900 px, and use three rows at 620 px.
- Owner: `playground-demo/style.css`, with status text updates in `playground-demo/app.js` and stable toolbar markup in `playground-demo/index.html`.
- Scope and affected surfaces: toolbar layout and provider-chip responsive presentation only. Status values, click behavior, template selection, and Setup behavior remain unchanged.
- Uncertainty: dynamic status strings vary in width (`ready`, `setup needed`, `service offline`, model labels), so validation must use the longest current labels rather than only the ready state.

## Design decision

Rebalance the desktop toolbar grid so the service strip receives the largest flexible track and the use-case selector remains bounded instead of encroaching on provider controls. Keep full provider labels at wide desktop sizes, retain the existing compact-chip treatment at 1120 px, retain the two-row layout at 900 px, and retain the three-row mobile layout at 620 px. Every stage must keep one authoritative control for each provider, Templates, Setup, and use-case selection; no overflow menu or duplicate mobile toolbar is introduced.

Within each stage, make grid/flex children explicitly shrink-safe (`min-width: 0`) and keep the selector and status strip inside their assigned tracks. Provider chips remain one line; when compact, their full dynamic status must remain available through an updated accessible label and tooltip rather than color alone.

## Reuse

- Existing owners: `.topbar`, `.template-picker`, `.service-strip`, `.service-chip`, and `.service-strip .icon-button`.
- Existing breakpoints: 1120 px compact chips, 900 px two-row toolbar, and 620 px three-row toolbar.
- Existing behavior: `renderHealth()` owns provider status strings and click targets.
- No parallel toolbar component is required.

## Changes

1. `playground-demo/style.css`
   - Change: replace the equal outer desktop tracks with a brand track, bounded selector track, and larger service/action track sized for the three provider chips plus Templates and Setup; keep the total minimum below the 1120 px compact breakpoint.
   - Change: add explicit `min-width: 0`, overflow containment, and shrink rules to the selector and service strip so one track cannot paint over another.
   - Change: preserve full labels above 1120 px, existing compact chips from 1120 px through 901 px, the two-row toolbar at 900 px, and the three-row toolbar at 620 px.
   - Change: ensure `--topbar-height` exactly matches each responsive toolbar composition so the viewport-bounded workbench begins below it.
   - Preserve: current visual tokens, control heights, one-line Templates/Setup labels, sticky placement, and backdrop treatment.
   - Verify: no overlap, clipping, or page-level horizontal overflow with the longest status strings at 1600 px, 1440 px, 1366 px, 1120 px, 900 px, and 620 px.

2. `playground-demo/app.js`
   - Change: centralize each provider's rendered status string and apply the same full string to visible text, `aria-label`, and `title` whenever health state changes or the service is offline.
   - Preserve: health logic, class names, status-dot colors, Setup focus targets, and polling behavior.
   - Verify: compact chips remain understandable through their accessible name/tooltip and never expose credentials or internal errors.

3. `playground-demo/test_playground_demo.py`
   - Change: assert the rebalanced grid contract, shrink-safe toolbar children, preserved breakpoint stages, exact topbar-height variables, and accessible full provider labels from the shared status updater.
   - Preserve: current Templates, Setup, health, provider, and secret-boundary tests.
   - Verify: tests fail if desktop outer tracks return to equal sizing or compact chips lose their full accessible status.

4. `playground-demo/README.md`
   - Change: none. This is a surface layout correction and does not change user workflow.
   - Preserve: all existing setup and status documentation.
   - Verify: no documentation claims a new overflow menu or alternate toolbar.

## Scope

- Inherit: all local-model labels and every provider health state shown in the toolbar.
- Verify: initial checking, local loading, ready, setup-needed, error, and service-offline labels; template names at the configured maximum length; pointer and keyboard access.
- Exclude: provider status semantics, polling frequency, model selection, Setup drawer design, Template studio design, navigation changes, overflow menus, and backend work.

## Validation

- Product: cycle representative health states and choose the longest template name; expect every control to remain reachable with no collision.
- Interface: inspect at 1600 px, the supplied 1400/1440-class desktop width, 1366 px, 1120 px, 900 px, and 620 px; verify toolbar height, no overlap, no clipping, and no horizontal page overflow.
- System: confirm the existing toolbar remains the only owner and the same full status string drives visible and accessible presentation.
- Repository: `node --check playground-demo/app.js` -> exits successfully.
- Repository: `PYTHONPATH=src .venv/bin/pytest -q` -> all tests pass, with only environment-dependent skips.
- Repository: `(cd results/raw && sha256sum -c SHA256SUMS)` -> every artifact reports `OK`.
- Repository: `.venv/bin/python benchmarks/verify_published.py` -> published benchmark verification passes.

## Stop conditions

- Stop if full provider labels still cannot fit above 1120 px after track rebalancing, if exact toolbar height cannot be expressed by existing breakpoints, or if fixing the collision requires an overflow menu or removing a requested control.

## Design documentation

- After acceptance and validation: none; the responsive toolbar contract remains owned by `playground-demo/style.css`.
