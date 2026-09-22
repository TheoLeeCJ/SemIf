# Setup drawer polish plan

## Goal

Turn the centered Setup modal into a right-side drawer and improve the two configuration surfaces that require the most interaction: local-model selection and Jev Bocha connection setup.

## Design decisions

- Preserve the native `<dialog>` element so Escape, focus trapping, and the backdrop keep working.
- Anchor the dialog to the right edge, use the full viewport height, and keep the header sticky while the configuration sections scroll.
- Keep the drawer to one coherent column at desktop width and make it full-width on small screens.
- Retain the existing SemIf neutral palette and model colors; use spacing, grouped fields, and subtle surfaces instead of introducing a new visual identity.
- Keep Jev official as an environment-only section and keep all credential/security behavior unchanged.

## Changes

1. Restyle `#settings-dialog` as a right drawer with a slide-in transition and a dimmed backdrop.
2. Replace the local-model row with a labeled selection card: clear field label, compact model-size badge, stronger select affordance, and a full-width load action.
3. Recompose Jev Bocha configuration into consistent stacked fields:
   - full-width password field;
   - fixed endpoint and model fields in a responsive grid;
   - explicit memory-only/security note;
   - clearer status facts and save action.
4. Improve section spacing, dividers, focus styles, mobile behavior, and overflow handling.

## Validation

- Verify open, close, Escape, scrolling, and keyboard focus in the browser.
- Inspect desktop and narrow viewport layouts.
- Confirm API-key clearing and Bocha configuration still work.
- Run the playground tests, JavaScript/Python syntax checks, secret scan, and repository-required validation.
