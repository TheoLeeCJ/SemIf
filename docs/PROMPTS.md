# Prompts

Every direct-mode result carries `prompt_version` and `prompt_sha256`. Until now there was one
prompt, `direct-options-v1`, written in English with English JSON keys, and the published numbers
all use it. This page describes the pluggable prompt behind `--prompt`: what a prompt may change,
what it may not, and how to add one.

## What a prompt is

A `Prompt` (`semif_phase1.core.Prompt`) is the *words* of the direct prompt: the system instruction
and the five key names of the user payload. The *shape* of the payload is fixed:

```json
{"evidence": <state>, "criterion": <question>, "options": [{"letter": "A", "description": "..."}, ...]}
```

Evidence first, then the criterion, then the lettered options. The serial and shared modes depend
on that order: they cut the reusable prefix at the end of the evidence, so a state can be prefilled
once and every criterion scored from it. A prompt renames the keys and rewrites the instruction; it
cannot reorder the payload, add fields, or change the answer slots, which stay the uppercase
letters `A`–`P` whatever the language.

## Built-in prompts

| name | `prompt_version` | system instruction | keys |
|---|---|---|---|
| `en` (default) | `direct-options-v1` | *Apply the supplied criterion to the supplied evidence. Choose exactly one listed option. Respond with only its uppercase letter, with no explanation or reasoning.* | `evidence`, `criterion`, `options`, `letter`, `description` |
| `fr` | `direct-options-fr-v1` | *Applique le critère fourni aux éléments fournis. Choisis exactement une option de la liste. Réponds uniquement par sa lettre majuscule, sans explication ni raisonnement.* | `elements`, `critere`, `options`, `lettre`, `description` |

`--prompt en` renders byte for byte what the code rendered before this option existed:
`tests/test_prompt.py` pins it, and the published `prompt_sha256` values still verify.

## A prompt from a file

```bash
semif-score --mode shared --backend llamacpp --prompt prompts/de.json ...
```

```json
{
  "version": "direct-options-de-v1",
  "system": "Wende das angegebene Kriterium auf die angegebenen Belege an. Wähle genau eine der aufgeführten Optionen. Antworte nur mit ihrem Großbuchstaben, ohne Erklärung oder Begründung.",
  "keys": {"evidence": "belege", "criterion": "kriterium", "options": "optionen", "letter": "buchstabe", "description": "beschreibung"}
}
```

`keys` and each of its entries are optional and default to the English names. The loader refuses a
file without `version` or `system`, a key it does not know, two keys with the same name, and a
`version` that belongs to a built-in prompt with different words — `prompt_version` in a result
must always mean one wording.

## What changes with the prompt, and what to expect

Only the words change, but option scores are conditional on the words: a different prompt is a
different measurement, with its own `prompt_version`, and numbers from two prompts are not
comparable without saying so. The letter slots are verified for every prompt exactly as before
(single round-trip tokens, boundary check, GGUF agreement).

Whether a model does better when addressed in the evidence's language is a question to measure on
the workload, not to assume. The one measurement made while adding this option, on a private
French-language e-mail set of 300 messages and four yes/no criteria with the pinned Qwen3.5-4B
GGUF, is reported in the pull request that introduced it; it is not part of this repository's
published results.

## In code

```python
from semif_phase1.core import resolve_prompt
from semif_phase1 import llamacpp_backend

prompt = resolve_prompt("fr")                     # or a Prompt(...), or a path
results, timing = llamacpp_backend.score_shared(model, tokenizer, rows, metadata, prompt=prompt)
```

`score`, `SerialPrefixScorer`, `score_shared` and `_state_prefix` in every backend accept
`prompt=`; leaving it out means `en`.
