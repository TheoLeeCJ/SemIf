"""Persistent local HTTP endpoint for SemIf scoring (stdlib only).

The CLI (``semif-score``) loads the model, scores one JSONL file, and exits.
This module loads the model once and keeps it on the single visible CUDA
device, so repeated ``if`` decisions don't pay reload cost:

    CUDA_VISIBLE_DEVICES=0 python -m semif_phase1.server \
      --mode direct \
      --model Qwen/Qwen3.5-4B \
      --revision 851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a \
      --port 8000

Primary API (TypeSafe-compatible shapes, see https://docs.typesafe.ai/llms.txt):

    GET  /healthz       -> {"status": "ok", "mode": ..., "model": {...}}
    GET  /v1/models     -> {"models": [{"name", "description", "release_date"}]}
    POST /v1/systemone  -> {state, model, questions} -> {model, answers, usage}

``questions`` maps an id you choose to one typed question: ``choice``
(picks one option), ``score`` (rates along ordered levels), or ``noul``
(yes/no probability). Answers come back under the same ids. Question ids
are not sent to the model. See ``systemone.py`` for honest boundaries
(2-16 choice options, uncalibrated probabilities, ``output_tokens: 0``).

Legacy SemIf row-schema routes (deprecated, kept for existing scripts):

    POST /score        -> single decision row -> single result object
    POST /score-batch  -> {"rows": [...]} -> {"results": [...], ...}
"""

from __future__ import annotations

import argparse
import json
import threading
import traceback
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from .core import validate_row
from .systemone import SystemOneError

MAX_BODY_BYTES = 10 * 1024 * 1024


class ServerState:
    """Model, tokenizer, and scorer held once for the process lifetime."""

    def __init__(self, model, tokenizer, metadata: dict, mode: str, max_tokens: int):
        from .direct import score as direct_score
        from .reranker import score as reranker_score
        from .serial import SerialPrefixScorer
        from .shared import score_shared

        self.model = model
        self.tokenizer = tokenizer
        self.metadata = metadata
        self.mode = mode
        self.max_tokens = max_tokens
        self.lock = threading.Lock()
        self._direct = direct_score if mode == "direct" else reranker_score if mode == "reranker" else None
        self._shared = score_shared if mode == "shared" else None
        self._serial = SerialPrefixScorer(model, tokenizer, metadata, max_tokens) if mode == "serial" else None

    def resolved_model_name(self, requested: str) -> str:
        source = self.metadata.get("source") if isinstance(self.metadata, dict) else None
        return source or requested

    def models(self) -> dict:
        from .systemone import MAX_LEVELS, MAX_OPTIONS

        name = self.resolved_model_name("semif-local")
        revision = self.metadata.get("revision") if isinstance(self.metadata, dict) else None
        return {
            "models": [
                {
                    "name": name,
                    "description": (
                        f"Local SemIf open baseline ({self.mode} mode, revision {revision}); "
                        f"choice 2-{MAX_OPTIONS} options, score 2-{MAX_LEVELS} levels, "
                        "uncalibrated conditional probabilities, not Jev."
                    ),
                    "release_date": "",
                }
            ]
        }

    def systemone(self, body: dict) -> dict:
        from .systemone import build_response, questions_to_rows, validate_body

        _, requested, questions = validate_body(body)
        state = body["state"]
        rows = questions_to_rows(state, questions)
        with self.lock:
            if self._shared is not None:
                results, _timing = self._shared(self.model, self.tokenizer, rows, self.metadata, self.max_tokens)
            elif self._serial is not None:
                results = [self._serial.score(row) for row in rows]
            else:
                assert self._direct is not None
                results = [
                    self._direct(self.model, self.tokenizer, row, self.metadata, self.max_tokens) for row in rows
                ]
        return build_response(self.resolved_model_name(requested), questions, results)

    def score_one(self, row: dict) -> dict:
        validate_row(row)
        with self.lock:
            if self._serial is not None:
                return self._serial.score(row)
            if self._direct is not None:
                return self._direct(self.model, self.tokenizer, row, self.metadata, self.max_tokens)
            raise ValueError("Shared mode serves POST /score-batch, not POST /score")

    def score_batch(self, rows: list[dict]) -> dict:
        if not isinstance(rows, list) or not rows:
            raise ValueError("Body needs a nonempty 'rows' list")
        for row in rows:
            validate_row(row)
        with self.lock:
            if self._shared is not None:
                results, timing = self._shared(self.model, self.tokenizer, rows, self.metadata, self.max_tokens)
                return {"results": results, "shared_timing": timing}
            if self._serial is not None:
                return {"results": [self._serial.score(row) for row in rows]}
            assert self._direct is not None
            return {
                "results": [
                    self._direct(self.model, self.tokenizer, row, self.metadata, self.max_tokens) for row in rows
                ]
            }


def _send_json(handler: BaseHTTPRequestHandler, status: int, payload: dict) -> None:
    body = json.dumps(payload, allow_nan=False).encode()
    handler.send_response(status)
    handler.send_header("Content-Type", "application/json")
    handler.send_header("Content-Length", str(len(body)))
    handler.end_headers()
    handler.wfile.write(body)


def _detail(message: str, loc=("body",)) -> dict:
    return {"detail": [{"loc": list(loc), "msg": message, "type": "value_error"}]}


def make_handler(state: ServerState):
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args):  # keep logs quiet; override with -v if needed
            pass

        def _read_json(self):
            length = int(self.headers.get("Content-Length") or 0)
            if length < 2 or length > MAX_BODY_BYTES:
                raise ValueError("Body must be 2-10485760 bytes")
            try:
                return json.loads(self.rfile.read(length))
            except (json.JSONDecodeError, OSError) as error:
                raise ValueError(f"Invalid JSON body: {error}") from error

        def do_GET(self):  # noqa: N802
            if self.path in ("/healthz", "/"):
                _send_json(self, 200, {"status": "ok", "mode": state.mode, "model": state.metadata})
            elif self.path == "/v1/models":
                _send_json(self, 200, state.models())
            else:
                _send_json(self, 404, {"error": f"Unknown path {self.path}; try GET /healthz"})

        def do_POST(self):  # noqa: N802
            try:
                if self.path in ("/v1/systemone", "/systemone"):
                    try:
                        _send_json(self, 200, state.systemone(self._read_json()))
                    except SystemOneError as error:
                        _send_json(self, 422, _detail(str(error), error.loc))
                    except ValueError as error:  # scorer limits (e.g. token budget) are also 422
                        _send_json(self, 422, _detail(str(error)))
                elif self.path == "/score":
                    _send_json(self, 200, state.score_one(self._read_json()))
                elif self.path == "/score-batch":
                    body = self._read_json()
                    rows = body.get("rows") if isinstance(body, dict) else None
                    _send_json(self, 200, state.score_batch(rows))
                else:
                    _send_json(self, 404, {"error": f"Unknown path {self.path}; try POST /v1/systemone"})
            except ValueError as error:  # legacy routes keep their 400 shape
                _send_json(self, 400, {"error": str(error)})
            except Exception:  # noqa: BLE001 - never leak a traceback over HTTP
                traceback.print_exc()
                _send_json(self, 500, {"detail": "Internal server error"})

    return Handler


def serve(state: ServerState, host: str, port: int) -> ThreadingHTTPServer:
    server = ThreadingHTTPServer((host, port), make_handler(state))
    server.daemon_threads = True
    return server


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--mode", choices=("direct", "serial", "shared", "reranker"), required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--revision", required=True)
    parser.add_argument("--max-tokens", type=int, default=4096)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()
    if args.max_tokens < 1 or not 1 <= args.port <= 65535:
        parser.error("max-tokens must be positive and port must be 1-65535")

    from .core import load_causal_model

    model, tokenizer, metadata = load_causal_model(args.model, args.revision)
    state = ServerState(model, tokenizer, metadata, args.mode, args.max_tokens)
    server = serve(state, args.host, args.port)
    print(f"semif listening on http://{args.host}:{args.port} (mode={args.mode})", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
