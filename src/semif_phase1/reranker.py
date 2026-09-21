"""Official-style Qwen3 reranker readout adapted to declared decision options."""

from __future__ import annotations

import inspect
import time

from .core import digest, softmax, validate_row

PREFIX = (
    '<|im_start|>system\nJudge whether the Document meets the requirements based on the Query and the '
    'Instruct provided. Note that the answer can only be "yes" or "no".<|im_end|>\n'
    '<|im_start|>user\n'
)
SUFFIX = '<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n'
DECISION_INSTRUCTION = (
    "Given evidence and one possible answer to a question, determine whether the evidence supports that answer "
    "under the question's criterion. Use only the supplied evidence."
)
RETRIEVAL_INSTRUCTION = (
    "Given a search query, determine whether the document is relevant and contains evidence that answers the query."
)
PROMPT_VERSION = "qwen3-reranker-native-options-v1"


def render_pair(row: dict, option: dict) -> str:
    """Render the official-style relevance prompt for one row/option pair."""
    experiment = row.get("provenance", {}).get("experiment")
    instruction = RETRIEVAL_INSTRUCTION if experiment in {"code-rag", "company-brain"} else DECISION_INSTRUCTION
    body = (
        f"<Instruct>: {instruction}\n"
        f"<Query>: Question: {row['question']}\nCandidate answer: {option['description']}\n"
        f"<Document>: {row['state']}"
    )
    return PREFIX + body + SUFFIX


def _encode(tokenizer, row, option, max_tokens):
    text = render_pair(row, option)
    ids = tokenizer.encode(text, add_special_tokens=False)
    if not ids or len(ids) > max_tokens:
        raise ValueError(f"Row {row['id']} option {option['id']}: {len(ids)} tokens exceed limit {max_tokens}")
    return ids, digest(text)


def _answer_ids(tokenizer):
    no = tokenizer.encode("no", add_special_tokens=False)
    yes = tokenizer.encode("yes", add_special_tokens=False)
    if len(no) != 1 or len(yes) != 1 or no == yes:
        raise ValueError("Reranker yes/no answers must be distinct single tokens")
    if tokenizer.convert_tokens_to_ids("yes") != yes[0] or tokenizer.convert_tokens_to_ids("no") != no[0]:
        raise ValueError("Tokenizer conversion differs from the official yes/no token contract")
    return no[0], yes[0]


def score_pair_batch(model, tokenizer, specs, max_tokens: int = 4096):
    """Score independent row/option relevance pairs in one native batch."""
    import torch

    if not specs:
        raise ValueError("Pair batch is empty")
    encoded = [_encode(tokenizer, row, option, max_tokens) for row, option in specs]
    width = max(len(ids) for ids, _ in encoded)
    pad = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id
    if pad is None:
        raise ValueError("Tokenizer requires padding or EOS token")
    device = next(model.parameters()).device
    inputs = {
        "input_ids": torch.tensor([[pad] * (width - len(ids)) + ids for ids, _ in encoded], device=device),
        "attention_mask": torch.tensor(
            [[0] * (width - len(ids)) + [1] * len(ids) for ids, _ in encoded], device=device
        ),
    }
    kwargs = dict(inputs, use_cache=False, return_dict=True)
    if "logits_to_keep" in inspect.signature(model.forward).parameters:
        kwargs["logits_to_keep"] = 1
    no_id, yes_id = _answer_ids(tokenizer)
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    mark = time.perf_counter()
    with torch.inference_mode():
        logits = model(**kwargs).logits[:, -1, :].float()
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    elapsed = time.perf_counter() - mark
    selected = logits[:, [no_id, yes_id]]
    odds = selected[:, 1] - selected[:, 0]
    relevance = selected.softmax(-1)[:, 1]
    return [
        {
            "option_id": option["id"],
            "log_odds": float(odds[index]),
            "binary_relevance": float(relevance[index]),
            "input_tokens": len(encoded[index][0]),
            "prompt_sha256": encoded[index][1],
        }
        for index, (_, option) in enumerate(specs)
    ], {
        "pair_batch_size": len(specs),
        "forward_seconds": elapsed,
        "padded_tokens": int(inputs["input_ids"].numel()),
    }


def score(model, tokenizer, row: dict, metadata: dict, max_tokens: int = 4096) -> dict:
    import torch

    validate_row(row)
    started = time.perf_counter()
    scored, timing = score_pair_batch(
        model, tokenizer, [(row, option) for option in row["options"]], max_tokens
    )
    log_odds = [item["log_odds"] for item in scored]
    binary = [item["binary_relevance"] for item in scored]
    return {
        "id": row["id"],
        "option_ids": [option["id"] for option in row["options"]],
        "probabilities": softmax(log_odds),
        "option_logits": log_odds,
        "independent_binary_relevance": binary,
        "input_tokens": sum(item["input_tokens"] for item in scored),
        "max_option_input_tokens": max(item["input_tokens"] for item in scored),
        "forward_seconds": timing["forward_seconds"],
        "total_seconds": time.perf_counter() - started,
        "option_prompt_sha256": [item["prompt_sha256"] for item in scored],
        "pair_batches": [timing],
        "prompt_version": PROMPT_VERSION,
        "model": metadata,
        "readout": "native yes/no log-odds per option, normalized only for relative comparison",
        "probability_status": "relative option compatibility; uncalibrated as categorical probability",
    }
