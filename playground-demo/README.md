# SemIf decision workbench

A static HTML/CSS/JavaScript workbench backed by one loopback-only Python
service. The service loads SemIf's pinned MLX model once, reuses it across
requests and pages, and optionally calls either TypeSafe Jev official or Jev
Bocha through the same TypeSafe Python SDK.

## Why this is separate from `webgpu-demo`

`webgpu-demo/` is intentionally browser-only. Its Web Worker loads a quantized
GGUF model into each browser context through wllama/WebGPU. A browser worker and
a native MLX process cannot share model memory.

This playground instead puts the shared boundary at `server.py`: one native MLX
model is owned by one localhost process. The workbench and any future local page
can call that process. The service permits cross-origin API calls only from
loopback origins (`localhost`, `127.0.0.1`, and `::1`).

## Install and run

From the repository root on Apple Silicon:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -e '.[test,mlx,playground]'
export TYPESAFE_API_KEY="your-key" # optional; required only for Jev
export BOCHA_JEV_API_KEY="your-bocha-key" # optional alternative to Setup
export BOCHA_JEV_MODEL="bocha-jev-v1"      # optional; this is the default
python playground-demo/server.py
```

Open <http://127.0.0.1:8090>. Choose the local model in **Setup**. MLX loads it
lazily when you click its load button or run the local target for the first
time. The built-in choices are immutable revisions of `Qwen/Qwen3.5-4B`
(default, about 9 GB of source weights), `Qwen/Qwen3.5-9B` (about 19.3 GB),
`mlx-community/Qwen3.8-27B-4bit` (about 16.1 GB),
`mlx-community/Muse-Glimmer-30B-4bit` (about 19.4 GB), the instruction-tuned
`mlx-community/gemma-4-12B-it-4bit` (about 6.74 GB),
the sparse-MoE `mlx-community/gemma-4-26b-a4b-it-4bit` (about 15.4 GB),
and the dense `mlx-community/gemma-4-31b-it-4bit` (about 18.4 GB).
Only one model is resident: selecting another model unloads the active model
before loading the new one. The 9B model requires substantially more unified
memory. `--mlx-bits 4` or `--mlx-bits 8` quantizes the selected model in memory
after its source weights are downloaded. Qwen3.8 27B, Muse Glimmer 30B, and all
three Gemma checkpoints are already MLX 4-bit, so the service uses their source
quantization instead of applying that Qwen3.5 setting. The 26B A4B model keeps
all experts resident but
activates only a subset per token. The 31B model is dense and therefore uses
substantially more compute per token. The current 128 GB M4 Max has ample memory
headroom; smaller-memory machines need their own load and workload validation.
Muse Glimmer is multimodal upstream, but this playground intentionally loads only
its language tower for JSON state-and-question scoring; image input is not exposed.

- 4B revision: `851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a`
- 9B revision: `c202236235762e1c871ad0ccb60c8ee5ba337b9a`
- Qwen3.8 27B revision: `10c35caafbb80f7dc6a7a432cdd11af10a6d4818`
- Muse Glimmer 30B revision: `3e7677d7a40d348a3daba263a2b1c0aa41910710`
- Gemma 4 12B IT revision: `73bcf09092aa277861d5a191b989b666f7f32e8f`
- Gemma 4 26B A4B IT revision: `0d77464eeb233a2da68ebf9d7dc4edaac7db956d`
- Gemma 4 31B IT revision: `696d436c404745a59f30e4939a658162b0a9e57f`

The official [TypeSafe quickstart](https://docs.typesafe.ai/introduction/quickstart)
uses `POST https://api.typesafe.ai/v1/systemone`, the `jev-latest` model alias,
and the `TYPESAFE_API_KEY` environment variable. This demo follows that contract
through `typesafe-sdk==0.7.1`. The API key is read only by the Python process. It
is never sent to or stored by the browser.

The [Jev Bocha integration guide](https://bocha-ai.feishu.cn/wiki/PhrPwCrEaiCyyPkkDNecdfRSnHR)
uses the same `typesafe-sdk==0.7.1` client with base URL
`https://jev.bocha.cn` and default model `bocha-jev-v1`. Choose **Jev Bocha**
as the challenger and provide its key in **Setup**, or set
`BOCHA_JEV_API_KEY` before starting the server. A key entered in Setup is sent
once to the loopback service, held only in process memory, and cleared from the
input. It is never written to browser storage or returned by the health API.
Restarting the Python process clears a key entered through Setup. The server
allowlists the documented Bocha HTTPS base URL rather than accepting arbitrary
outbound hosts.

Useful server options:

```text
--host 127.0.0.1
--port 8090
--default-model {gemma-4-12b-it-4bit,gemma-4-26b-a4b-it-4bit,gemma-4-31b-it-4bit,muse-glimmer-30b-4bit,qwen3.5-4b,qwen3.5-9b,qwen3.8-27b-4bit}
--mlx-bits {4,8}
--mlx-cache-limit-mib 256
```

For a single custom model instead of the built-in selector, pass both
`--model-source` and its immutable `--model-revision`.

## Templates

Checked-in templates live in `templates/`. Add the new JSON filename to
`templates/index.json` so the static page can discover it. Each file has this
shape:

```json
{
  "id": "my-use-case",
  "name": "My use case",
  "category": "Custom use cases",
  "description": "What this use case tests",
  "state": { "any": "finite JSON value supported by the API" },
  "questions": {
    "dimension_name": {
      "type": "noul",
      "instructions": "The yes/no statement to evaluate"
    }
  }
}
```

The Template studio can save drafts to browser storage, import a JSON template,
or download the current state and questions as a shareable file. Browser storage
is convenience state, not the source of truth for checked-in use cases. The
categorized **Use case** picker supports pointer selection plus Arrow, Home,
End, Enter, Space, Escape, and Tab keyboard navigation.

## JSON workbench and pane sizing

State and Questions use one editable JSON code surface: syntax color, fixed line
numbers, bracket matching, and fold controls remain available without switching
between Tree and Edit modes. Use the gutter disclosure markers for individual
objects and arrays, or the compact `+` and `−` actions to unfold or fold the
whole document. Long lines wrap beneath the code column while the line-number
and fold gutters remain fixed. Pasting a complete valid JSON value automatically
formats it with two-space indentation after the paste. Invalid JSON is left exactly as
entered so it can be corrected, and validation appears beside the editor and
when running a comparison. Folding changes presentation only; it never changes
the editor value or evaluation payload.

The editor is the MIT-licensed CodeMirror `5.65.21`, vendored locally as a
minimal pinned subset under `vendor/codemirror-5.65.21/`. It does not use a CDN,
and `SHA256SUMS` records every copied upstream file. If the enhancement cannot
load, the underlying textarea remains usable as a plain JSON editor.

Raw JSON opens in **Response** mode and uses the same read-only tree renderer.
Its compact `+` and `−` actions match State and Questions and affect only tree
presentation. The response tree and **Copy JSON** contain provider results only;
the submitted request is intentionally excluded. Switch to **Request** for two
copyable cURL commands: the selected challenger and Jev official. Local MLX uses
the loopback API; provider-facing commands use literal
`<BOCHA_JEV_API_KEY>` or `<TYPESAFE_API_KEY>` placeholders and never expose a
configured key. Request/Response controls remain pinned while their content
scrolls.

The application shell is bounded to the browser viewport. State, Questions, and
the Response region scroll within the workbench instead of extending the browser
document. At 900 px and below, request and response remain stacked in the same
order and that stack scrolls inside the bounded workbench.

State and Questions begin at an equal half-and-half split. Drag their horizontal
rail to resize them, use Up/Down Arrow in 16 px steps or Shift+Arrow in 48 px
steps, and double-click to restore the equal split. On layouts wider than 900 px,
the vertical request/response rail supports the same pointer, keyboard, and
double-click reset behavior with Left/Right Arrow and a default 46.5% request
width. That vertical rail is removed when the workbench stacks. Pane sizes are
intentionally not saved between page loads.

## Question types and comparison boundary

Each run has exactly two result columns. Select either **Local MLX** or **Jev
Bocha** as the challenger; **Jev official** is the fixed baseline. A three-way
view is deliberately deferred so the dimension table and mobile layout remain
readable.

- **Noul:** local output reports the conditional probability of the `true`
  option; Jev returns its official `noul` value.
- **Choice:** both outputs return the selected option and a distribution over
  declared criteria.
- **Score:** both outputs return a probability-weighted score and distribution
  over ordered levels.

SemIf's local probabilities are softmax values conditional on only the declared
options. For Choice and Score, the adapter also reports normalized Shannon
entropy concentration as `confidence`: `1 - H(p) / log(N)`. This is a compact
description of how peaked the declared-option distribution is, not calibrated
probability that the answer is correct. Noul keeps its direct binary `noul`
probability and has no separate confidence field. The UI labels local entropy
confidence separately from Jev confidence and does not claim they are equivalent.

The summary timing for Jev is end-to-end wall-clock time observed by this local
server around the SDK call. It includes network and SDK overhead and is not
provider-side model latency.

Provider request failures are summarized once above the rendered response so a
single upstream error is not repeated for every dimension. Successful results
from the other provider remain visible; provider diagnostics stay available in
Raw JSON without repeating the submitted request.

## Local JSON API

- `GET /api/health` — model, SDK, and key-configuration status; never returns a
  credential.
- `POST /api/model/load` — load or reuse the selected shared MLX model. The
  optional JSON body is `{"model":"qwen3.5-9b"}`.
- `POST /api/config/bocha` — keep a Bocha key, allowlisted base URL, and model
  alias in this server process; the sanitized response never includes the key.
- `POST /api/evaluate/local` — evaluate typed questions through SemIf MLX shared
  state scoring.
- `POST /api/evaluate/bocha` — evaluate through `typesafe-sdk` configured for
  `https://jev.bocha.cn`.
- `POST /api/evaluate/typesafe` — evaluate the same request through the official
  TypeSafe Python SDK.

Both evaluation endpoints accept the TypeSafe request shape:

```json
{
  "state": "Evidence or structured JSON",
  "model": "jev-latest",
  "local_model": "qwen3.5-4b",
  "questions": {
    "is_urgent": {
      "type": "noul",
      "instructions": "The evidence expresses urgency."
    }
  }
}
```

Local MLX inference is serialized because one process owns one stateful GPU
model. The selected challenger request and Jev official request run concurrently
in separate server threads, so the page always presents one challenger against
one baseline without loading the local model twice.

## Validation

```bash
python -m pytest -q playground-demo webgpu-demo
node --check playground-demo/app.js
(cd playground-demo/vendor/codemirror-5.65.21 && sha256sum -c SHA256SUMS)
python benchmarks/verify_published.py
(cd results/raw && sha256sum -c SHA256SUMS)
```

The full repository requirement remains `pytest -q` from an isolated environment.
No live Jev call or multi-gigabyte MLX model download is part of the static test
suite.
