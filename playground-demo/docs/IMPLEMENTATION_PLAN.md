# Shared MLX playground implementation plan

## Goal

Build a local, browser-based workbench for evaluating one JSON state against
runtime-defined Noul, Choice, and Score questions. Run the same request through
SemIf's native MLX readout and, when configured, TypeSafe's official Jev service.
Keep the browser UI framework-free and keep one MLX model loaded in one localhost
Python process so every page that calls the service reuses the same model instance.

## Findings and boundaries

- `webgpu-demo/` is intentionally browser-only. Each tab creates a Web Worker and
  loads a quantized GGUF model into that browser context. It does not use MLX and
  cannot share model memory with another page or process.
- The new reusable boundary is a loopback-only HTTP service backed by
  `semif_phase1.mlx_backend`. One service process owns one loaded model and
  serializes GPU inference; any local page can call its JSON endpoints.
- The TypeSafe SDK runs in the Python service. `TYPESAFE_API_KEY` stays in the
  process environment and is never returned to, stored by, or entered into the
  browser UI.
- Local scores remain conditional option probabilities, not calibrated
  confidence. The UI must label that distinction instead of presenting the
  values as equivalent to Jev confidence.

## Milestones

1. **Shared service**
   - Serve the static playground on `127.0.0.1`.
   - Lazily load the pinned Qwen3.5 4B MLX model once.
   - Expose health, model-load, local-evaluate, and TypeSafe-evaluate endpoints.
   - Convert Noul, Choice, and Score questions into SemIf rows and normalize local
     output into an API-shaped response without hiding raw readout metadata.

2. **Static workbench**
   - Editable State JSON and Questions JSON panes.
   - Local MLX and Jev target controls with independent failure handling.
   - Rendered answers, raw JSON, and a dimension-by-dimension comparison table.
   - Keyboard operation, responsive layout, visible focus, and reduced-motion
     support.

3. **Template workflow**
   - Load checked-in templates from `templates/index.json`.
   - Provide a template studio that uses the current state and questions.
   - Save private drafts to browser storage, import JSON, or download a template
     file for committing to `templates/`.

4. **Validation**
   - Unit-test request normalization, typed-question mapping, local response
     adaptation, template validity, and static security contracts.
   - Syntax-check JavaScript, run repository tests and published-evidence checks,
     and exercise the page in a real local browser without requiring a model
     download or a TypeSafe credential.

## Visual system

- **Color:** paper `#f6f6f2`, ink `#191b1d`, rule `#d6d7d2`, local blue
  `#245e78`, Jev orange `#d9633b`, agreement green `#2e7452`.
- **Type:** the OS humanist sans stack for navigation and instructions; the OS
  monospace stack for JSON, dimension keys, values, and timing.
- **Layout:** the input and response panes share a center rule on wide screens;
  they stack on narrow screens. Results use aligned columns rather than cards as
  the primary organizing device.

```text
+--------------------------------------------------------------------------+
| SemIf / decision workbench   [template selector]   local | Jev | settings|
+--------------------------------------+-----------------------------------+
| State JSON                           | Compare | Rendered | Raw JSON       |
|                                      |                                   |
|                                      | dimension | local | Jev | delta     |
+--------------------------------------+                                   |
| Questions JSON                       |                                   |
|                                      |                                   |
+--------------------------------------+-----------------------------------+
| Template studio / setup drawer                          [Run comparison]  |
+--------------------------------------------------------------------------+
```

The only strong visual gesture is the paired blue/orange comparison rail. The
rest stays quiet and precise so dense JSON and probability distributions remain
readable.
