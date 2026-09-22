# Serving SemIf over the Jev HTTP contract

`semif-serve` answers TypeSafe's `POST /v1/systemone` and `GET /v1/models` from
a local SemIf backend, so an application written against the TypeSafe SDKs runs
on an open model by changing one environment variable. The contract, the
mapping, and the known divergences are specified in
[Jev API compatibility](JEV_API_COMPAT.md).

## Install and run

```bash
pip install -e '.[test,serve]'          # add mlx or llamacpp for those backends

SEMIF_BACKEND=mlx \
SEMIF_MODEL=Qwen/Qwen3.5-4B \
SEMIF_REVISION=851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a \
semif-serve
```

The server binds `127.0.0.1:8471`, loads one model, and logs the id it will
report. Every flag also exists as a CLI option, which wins over the environment:
`semif-serve --backend mlx --model Qwen/Qwen3.5-4B --revision 851bf6e8...`.

Check it is up:

```bash
curl -s http://127.0.0.1:8471/healthz
{"status":"ready","model":"semif/Qwen3.5-4B@851bf6e806ef+mlx-bf16","backend":"mlx",
 "prompt_version":"direct-options-v1","max_input_tokens":4096,"mode":"auto"}
```

## Use it from the TypeSafe SDK

Nothing in the application changes except the base URL. The SDK requires a key
to be present before it will send a request; the server ignores it unless
`SEMIF_API_KEY` is set.

```python
import os

os.environ["TYPESAFE_API_KEY"] = "local"
os.environ["TYPESAFE_BASE_URL"] = "http://127.0.0.1:8471"

from typesafe_sdk import TypeSafeClient, Choice, Noul, Score

with TypeSafeClient(timeout=120.0) as client:
    response = client.system_one(
        state={"ticket": "I was charged twice for order A-104. Please refund the duplicate."},
        questions={
            "refund_requested": Noul(instructions="Did the customer request a refund?"),
            "department": Choice(
                instructions="Which team should handle this?",
                criteria={"billing": "Payments, invoicing, refunds",
                          "technical": "Bugs, outages, integrations"},
            ),
            "frustration": Score(
                instructions="How frustrated is the customer?",
                criteria=["Calm and neutral", "Frustrated but civil", "Very angry"],
            ),
        },
    )

print(response.model)                            # the SemIf model that answered
print(response.answers["refund_requested"].noul)
print(response.answers["department"].choice, response.answers["department"].confidence)
print(response.answers["frustration"].score, dict(response.answers["frustration"].legend))
```

The JavaScript SDK takes the same treatment through `TYPESAFE_BASE_URL` or the
client's `baseURL` option. Or skip the SDKs:

```bash
curl -s http://127.0.0.1:8471/v1/systemone \
  -H 'content-type: application/json' \
  -d '{"state":"Help! My payouts have been failing for 3 days.",
       "model":"jev-latest",
       "questions":{"is_urgent":{"type":"noul","instructions":"Does this convey urgency?"}}}'
```

## What the response carries

Answers match the published shapes: `noul`, or `choice` + `probabilities` +
`confidence`, or `score` + `legend` + `probabilities` + `confidence`. Each one
also carries a `semif` block the official SDKs ignore and direct HTTP consumers
can read:

```json
"semif": {
  "option_logits": [26.375, 21.25],
  "prompt_sha256": "239b6987...",
  "prompt_version": "direct-options-v1",
  "input_tokens": 105,
  "temperature": 1.0,
  "confidence_formula": "normalized-peak-v1",
  "probability_status": "conditional option score; uncalibrated as decision confidence"
}
```

Response headers `x-semif-mode` and `x-semif-backend` report how the request was
executed, and `x-typesafe-request-id` is what the SDK surfaces as `request_id`.

## Configuration

| Variable | Default | Meaning |
| -------- | ------- | ------- |
| `SEMIF_MODEL`, `SEMIF_REVISION` | — | Required. Remote models need a 40-character commit id. |
| `SEMIF_BACKEND` | `torch` | `torch`, `mlx`, or `llamacpp`. |
| `SEMIF_DEVICE`, `SEMIF_DTYPE` | `auto`, `bfloat16` | Torch only. |
| `SEMIF_MLX_BITS` | unset | `4` or `8`. In-memory quantization; it changes the answers and shows up in the reported model id. |
| `SEMIF_MLX_CACHE_LIMIT_MIB` | `256` | MLX allocator cache bound. |
| `SEMIF_GGUF`, `SEMIF_LLAMA_THREADS` | — | llama.cpp only. |
| `SEMIF_MAX_INPUT_TOKENS` | `4096` | Per-question budget. Enforced, never truncated. |
| `SEMIF_MODE` | `auto` | Pin `direct`, `serial`, or `shared`. |
| `SEMIF_TEMPERATURE` | `1.0` | A positive float, or a path to a per-question-type JSON map. |
| `SEMIF_API_KEY` | unset | When set, `Authorization: Bearer <value>` is required. |
| `SEMIF_ACCEPT_JEV_ALIASES` | `1` | Whether `jev-latest` and friends are accepted in `model`. |
| `SEMIF_HOST`, `SEMIF_PORT` | `127.0.0.1`, `8471` | A non-loopback bind requires `SEMIF_API_KEY`. |

## Things to know before you rely on it

**Choice is capped at 16 options.** Jev allows 255. A larger Choice is refused
with `422 option_count_unsupported` rather than truncated. See
[section 5](JEV_API_COMPAT.md#5-answer-slots-and-the-option-ceiling).

**Probabilities are not calibrated.** The default temperature is 1.0 and the
fitted per-workload values in [Calibration](CALIBRATION.md) are exactly that —
per workload. Fit your own before thresholding on them, and read
[Results](RESULTS.md) for what this baseline does and does not do well.

**`usage.output_tokens` is always 0.** No answer token is generated. That is the
mechanism, not a reporting gap.

**Model names.** `jev-latest`, `jev-preview`, and `jev-1.13.0` are accepted so
unmodified sample code runs, but the response's `model` field always reports the
SemIf model that answered. Set `SEMIF_ACCEPT_JEV_ALIASES=0` to refuse them.

**One model, one process, one request at a time.** Requests queue in front of the
model. Run one server per GPU.

**This is not Jev.** Agreement with published Jev on the aligned 102-row subset
is 0.845 against 0.883. The API layer changes the interface, not the model.

## Tests

```bash
pytest -q tests/test_api_translate.py tests/test_api_assemble.py \
          tests/test_api_runtime.py tests/test_api_conformance.py

SEMIF_LIVE_MODEL=Qwen/Qwen3.5-4B \
SEMIF_LIVE_REVISION=851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a \
SEMIF_LIVE_BACKEND=mlx pytest -q tests/test_api_live.py
```

The first group needs no weights and no network: it includes a real
`AsyncTypeSafeClient` driving the app over an in-process transport, and a check
that the rows the server builds render to the same prompt as the equivalent
`semif-score` JSONL rows. The second repeats the primitives against real weights
and asserts the server's `prompt_sha256` and probabilities match a direct scorer
call on the same decision.
