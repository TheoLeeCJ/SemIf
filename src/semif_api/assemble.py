"""SemIf rows -> Jev answers.

Pure functions. The mapping is specified in docs/JEV_API_COMPAT.md section 6.

Every probability, noul, score, and confidence value leaves here as a Python
float. `typesafe_sdk`'s response models are strict, so a value serialized as
`0` rather than `0.0` fails SDK validation on the client.
"""

from __future__ import annotations

from semif_phase1.core import softmax

from .translate import QuestionPlan

#: Reported on every Choice and Score answer. TypeSafe publishes this as its
#: explainer's calculation and does not publish the production statistic
#: verbatim, so consumers with thresholds tuned against live Jev can tell which
#: statistic produced a number.
CONFIDENCE_FORMULA = "normalized-peak-v1"


def confidence(probabilities: list[float]) -> float:
    """Collapse a distribution's shape into [0, 1].

    An even split over n options gives 0.0; all mass on one option gives 1.0.
    """
    count = len(probabilities)
    if count < 2:
        raise ValueError("Confidence needs at least two probabilities")
    peak = max(probabilities)
    # Clamp with floats: min(1, 1.2) returns the int 1 and breaks strict clients.
    return min(1.0, max(0.0, (count * peak - 1.0) / (count - 1.0)))


def _probabilities(result: dict, temperature: float) -> list[float]:
    if temperature <= 0:
        raise ValueError("Temperature must be positive")
    if temperature == 1.0:
        # Reuse the backend's own values so the API layer stays bit-identical
        # to the equivalent semif-score row.
        return [float(value) for value in result["probabilities"]]
    return softmax([float(logit) / temperature for logit in result["option_logits"]])


def _extension(result: dict, temperature: float, with_confidence: bool) -> dict:
    block = {
        "option_ids": list(result["option_ids"]),
        "option_logits": [float(value) for value in result["option_logits"]],
        "prompt_sha256": result["prompt_sha256"],
        "prompt_version": result["prompt_version"],
        "input_tokens": int(result["input_tokens"]),
        "temperature": float(temperature),
        "readout": result["readout"],
        "probability_status": result["probability_status"],
    }
    if with_confidence:
        block["confidence_formula"] = CONFIDENCE_FORMULA
    return block


def build_answer(plan: QuestionPlan, result: dict, temperature: float = 1.0) -> dict:
    """Turn one scored row into the answer object for its question type."""
    if tuple(result["option_ids"]) != plan.option_ids:
        raise ValueError(f"Result for {plan.key!r} does not carry the planned options")
    probabilities = _probabilities(result, temperature)
    if plan.kind == "noul":
        index = plan.option_ids.index("true")
        return {
            "type": "noul",
            "noul": float(probabilities[index]),
            "semif": _extension(result, temperature, with_confidence=False),
        }
    distribution = {option: float(value) for option, value in zip(plan.option_ids, probabilities)}
    if plan.kind == "choice":
        best = max(range(len(probabilities)), key=probabilities.__getitem__)
        return {
            "type": "choice",
            "choice": plan.option_ids[best],
            "probabilities": distribution,
            "confidence": confidence(probabilities),
            "semif": _extension(result, temperature, with_confidence=True),
        }
    if plan.kind == "score":
        return {
            "type": "score",
            "score": float(sum(index * value for index, value in enumerate(probabilities))),
            "legend": dict(plan.legend or {}),
            "probabilities": distribution,
            "confidence": confidence(probabilities),
            "semif": _extension(result, temperature, with_confidence=True),
        }
    raise ValueError(f"Unknown question kind {plan.kind!r}")


def usage(results: list[dict], mode: str, timing: dict | None) -> dict:
    """Tokens actually evaluated.

    Shared scoring prefills one state and branches it, so the state is counted
    once. Per-question prompt lengths stay in each answer's `semif` block.
    """
    if mode == "shared" and timing and "prefix_tokens" in timing:
        prefix = int(timing["prefix_tokens"])
        evaluated = prefix + sum(int(result["input_tokens"]) - prefix for result in results)
    else:
        evaluated = sum(int(result["input_tokens"]) for result in results)
    # No answer token is generated. Reporting a non-zero count to look like Jev
    # would misstate the mechanism this project exists to demonstrate.
    return {"input_tokens": evaluated, "output_tokens": 0}


def build_response(model: str, plans: list[QuestionPlan], results: list[dict], *,
                   temperature=1.0, mode: str, backend: str,
                   metadata: dict, timing: dict | None = None,
                   fallback_from: str | None = None) -> dict:
    """Assemble the full response body for one evaluated request.

    `temperature` is a float, or a callable taking a question kind so a
    per-type calibration map can be applied.
    """
    by_id = {result["id"]: result for result in results}
    if len(by_id) != len(plans) or any(plan.key not in by_id for plan in plans):
        raise ValueError("Every question must receive exactly one result")
    resolve = temperature if callable(temperature) else (lambda kind: temperature)
    answers = {plan.key: build_answer(plan, by_id[plan.key], resolve(plan.kind)) for plan in plans}
    return {
        "model": model,
        "answers": answers,
        "usage": usage([by_id[plan.key] for plan in plans], mode, timing),
        "semif": {
            "mode": mode,
            "backend": backend,
            "model": metadata,
            "timing": timing or {},
            "fallback_from": fallback_from,
        },
    }
