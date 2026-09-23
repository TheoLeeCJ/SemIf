"""Answer-slot capacity for the API layer.

SemIf reads option probabilities from single-token uppercase letters. The
frozen `direct-options-v1` prompt uses `core.LETTERS`, which holds 16 of them,
so a Choice carries at most 16 options. Jev accepts 255.

`direct-options-v2` in docs/JEV_API_COMPAT.md specifies a wider slot alphabet
that reaches 255. It changes the prompt for every row, including small ones, so
it cannot ship before its own benchmark evidence is committed. It is declared
here and refused, rather than left undefined.
"""

from __future__ import annotations

from semif_phase1.core import LETTERS

from .errors import invalid_request

V1 = "direct-options-v1"
V2 = "direct-options-v2"

#: Maximum options each prompt version can express, by construction.
CAPACITY = {V1: len(LETTERS), V2: 255}

#: Prompt versions this build can actually serve.
IMPLEMENTED = frozenset({V1})

MIN_OPTIONS = 2


def resolve(prompt_version: str) -> str:
    """Return a servable prompt version or explain why it is not servable."""
    if prompt_version not in CAPACITY:
        raise ValueError(f"Unknown prompt version {prompt_version!r}; expected one of {sorted(CAPACITY)}")
    if prompt_version not in IMPLEMENTED:
        raise ValueError(
            f"Prompt version {prompt_version!r} is specified in docs/JEV_API_COMPAT.md but not implemented; "
            "it requires a wider answer-slot alphabet and its own committed benchmark evidence"
        )
    return prompt_version


def check_count(count: int, prompt_version: str, *, param: str, noun: str) -> None:
    """Reject an option or level count this prompt version cannot express."""
    limit = CAPACITY[prompt_version]
    if count < MIN_OPTIONS:
        raise invalid_request(
            "option_count_unsupported",
            f"{noun} count is {count}; at least {MIN_OPTIONS} are required.",
            param,
        )
    if count > limit:
        raise invalid_request(
            "option_count_unsupported",
            f"{noun} count is {count}; {prompt_version} supports {MIN_OPTIONS}-{limit}.",
            param,
        )
