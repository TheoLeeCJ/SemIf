"""SemIf-compatible direct-logit readout through exllamav3 (exl3 quantized models).

Same decision contract as semif-score --mode direct:
  - prompt built with SemIf core (system DIRECT_SYSTEM + JSON evidence/criterion/options)
  - chat-template rendered with enable_thinking=False
  - answer slots: single-token 'A'/'B' letters, round-trip + prefix-stable validated
  - readout: full-vocabulary last-position logits restricted to slot tokens, softmax
  - no truncation: rows exceeding the cache budget are refused, never cut

Backend: exllamav3 Generator + Job(max_new_tokens=1, return_logits=True);
result['logits'][0,-1,:] is the last-position logits (same quantity HF gets
from model(**inputs).logits[:, -1, :]).

Intentional deviations from the pinned BF16 path (recorded in output):
quantized exl3 weights (and whatever the arch loader keeps unquantized per
quantization_config), exllamav3 kernels, this model family != pinned 4B.
"""
from __future__ import annotations

import argparse, json, sys, time
from pathlib import Path

_SRC = Path(__file__).resolve().parents[1] / "src"
sys.path.insert(0, str(_SRC))
from semif_phase1.core import LETTERS, direct_messages, softmax, digest  # noqa: E402

import torch  # noqa: E402
from exllamav3 import model_init, Generator, Job  # noqa: E402


def encode_ids(tokenizer, text: str) -> list[int]:
    out = tokenizer.encode(text, encode_special_tokens=True)
    if isinstance(out, tuple):
        out = out[0]
    if isinstance(out, torch.Tensor):
        out = out.flatten().tolist()
    return [int(t) for t in out]


def slot_ids(tokenizer, count: int, prompt: str, ids: list[int]) -> list[int]:
    slots = []
    for letter in LETTERS[:count]:
        tok = int(tokenizer.single_id(letter))
        if tokenizer.decode(torch.tensor([tok])).strip() != letter:
            raise ValueError(f"slot {letter!r} does not decode back")
        if encode_ids(tokenizer, prompt + letter) != ids + [tok]:
            raise ValueError(f"answer boundary changes tokenization for slot {letter}")
        slots.append(tok)
    return slots


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-dir", required=True)
    ap.add_argument("--model-source", default=None, help="Public model ID recorded in output metadata")
    ap.add_argument("--model-revision", default=None, help="Pinned model revision recorded in output metadata")
    ap.add_argument("--input", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--cache-size", type=int, default=16384)
    ap.add_argument("--gpu-split", type=float, default=22.5)
    args = ap.parse_args()

    rows = [json.loads(l) for l in open(args.input) if l.strip()]
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    done = set()
    if out_path.exists():
        done = {json.loads(l)["id"] for l in out_path.read_text().splitlines() if l.strip()}
    rows = [r for r in rows if r["id"] not in done]
    print(f"{len(rows)} rows to score ({len(done)} already done)", file=sys.stderr)

    p = argparse.ArgumentParser()
    model_init.add_args(p, add_draft_model_args=False)
    margs = p.parse_args([
        "-m", args.model_dir, "-gs", str(args.gpu_split), "-cs", str(args.cache_size),
    ])
    model, config, cache, tokenizer = model_init.init(margs, progress=False)
    generator = Generator(model, cache, tokenizer)
    import exllamav3
    import exllamav3.version as ver
    metadata = {
        "source": args.model_source or args.model_dir,
        "revision": args.model_revision,
        "dtype": "exl3 quantized (see quantization_config.json)",
        "runtime": f"exllamav3 {getattr(ver, '__version__', 'unknown')}",
        "torch_version": torch.__version__,
        "readout_backend": "exllamav3 Generator/Job(return_logits=True), exl3 kernels",
    }

    with out_path.open("a") as dest:
        for i, row in enumerate(rows):
            started = time.perf_counter()
            messages = direct_messages(row)  # validates row (2-16 options etc.)
            # enable_thinking is a Qwen-template parameter; templates that don't
            # declare it (e.g. Gemma) reject the kwarg, so fall back without it
            try:
                text = tokenizer.hf_render_chat_template(
                    messages, add_generation_prompt=True, enable_thinking=False
                )
            except Exception:
                text = tokenizer.hf_render_chat_template(
                    messages, add_generation_prompt=True
                )
            ids = encode_ids(tokenizer, text)
            if not ids or len(ids) + 8 > args.cache_size:
                rec = {"id": row["id"], "status": "refused_over_cache_budget",
                       "input_tokens": len(ids), "cache_size": args.cache_size}
                dest.write(json.dumps(rec) + "\n"); dest.flush(); continue
            slots = slot_ids(tokenizer, len(row["options"]), text, ids)

            job = Job(input_ids=torch.tensor([ids], dtype=torch.long),
                      max_new_tokens=1, return_logits=True, seed=53)
            generator.enqueue(job)
            last = None
            while generator.num_remaining_jobs() or generator.num_active_jobs():
                for res in generator.iterate():
                    if res.get("logits") is not None:
                        last = res
            if last is None or last.get("logits") is None:
                rec = {"id": row["id"], "status": "no_logits"}
                dest.write(json.dumps(rec) + "\n"); dest.flush(); continue

            vocab = last["logits"][0, -1].float().cpu()
            selected = [float(vocab[s]) for s in slots]
            dest.write(json.dumps({
                "id": row["id"],
                "status": "ok",
                "option_ids": [o["id"] for o in row["options"]],
                "probabilities": softmax(selected),
                "option_logits": selected,
                "input_tokens": len(ids),
                "total_seconds": time.perf_counter() - started,
                "prompt_sha256": digest(text),
                "prompt_version": "direct-options-v1-exl3-bridge",
                "model": metadata,
                "readout": "native full-vocabulary last-position logits restricted to declared answer slots",
                "probability_status": "conditional option score; uncalibrated as decision confidence",
            }, allow_nan=False) + "\n")
            dest.flush()
            if (i + 1) % 25 == 0:
                print(f"{i+1}/{len(rows)} last={time.perf_counter()-started:.2f}s", file=sys.stderr)
    print("DONE", file=sys.stderr)


if __name__ == "__main__":
    main()
