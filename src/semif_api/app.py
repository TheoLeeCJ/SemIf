"""HTTP surface: routes, auth, headers, and error mapping.

Wire contract in docs/JEV_API_COMPAT.md section 3. The request body is decoded
with `json.loads` rather than a request model, because the order of
`questions`, of Choice `criteria`, and of structured `instructions` fields is
semantic and must survive into the prompt unchanged.
"""

from __future__ import annotations

import argparse
import datetime
import json
import logging
import uuid

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

from . import assemble, runtime, slots
from .errors import ApiError, backend_failed, invalid_request, not_found, unauthorized, unavailable
from .translate import translate

logger = logging.getLogger("semif_api")

MODELS_DESCRIPTION = "SemIf direct option readout. Open baseline; not Jev."


def _error_response(error: ApiError, request_id: str) -> JSONResponse:
    headers = {"x-typesafe-request-id": request_id}
    if error.status == 503:
        headers["retry-after"] = "2"
    return JSONResponse(error.body, status_code=error.status, headers=headers)


def _authorize(request: Request, config: runtime.Config) -> None:
    if not config.api_key:
        return
    header = request.headers.get("authorization", "")
    scheme, _, token = header.partition(" ")
    if scheme.lower() != "bearer" or token.strip() != config.api_key:
        raise unauthorized("Missing or invalid API key. Check the Authorization header.")


def create_app(config: runtime.Config, engine: runtime.Runtime | None = None) -> FastAPI:
    """Build the app. `engine` is injectable so tests can run without weights."""
    engine = engine if engine is not None else runtime.Runtime(config)
    app = FastAPI(title="semif-serve", docs_url=None, redoc_url=None)
    app.state.config = config
    app.state.engine = engine

    @app.middleware("http")
    async def tag_request(request: Request, call_next):
        request_id = str(uuid.uuid4())
        request.state.request_id = request_id
        try:
            response = await call_next(request)
        except ApiError as error:
            return _error_response(error, request_id)
        response.headers["x-typesafe-request-id"] = request_id
        response.headers["x-semif-backend"] = config.backend
        return response

    @app.exception_handler(ApiError)
    async def handle_api_error(request: Request, error: ApiError):
        return _error_response(error, getattr(request.state, "request_id", ""))

    @app.get("/healthz")
    async def healthz():
        return {
            "status": "ready" if engine.ready else "loading",
            "model": engine.model_id,
            "backend": config.backend,
            "prompt_version": config.prompt_version,
            "max_input_tokens": config.max_input_tokens,
            "mode": config.mode,
        }

    @app.get("/v1/models")
    async def list_models(request: Request):
        _authorize(request, config)
        return {"models": [{
            "name": engine.model_id,
            "description": MODELS_DESCRIPTION,
            "release_date": datetime.date.today().isoformat(),
        }]}

    @app.post("/v1/systemone")
    async def system_one(request: Request):
        _authorize(request, config)
        raw = await request.body()
        try:
            payload = json.loads(raw)
        except (UnicodeDecodeError, json.JSONDecodeError) as error:
            raise invalid_request("invalid_body", f"Request body is not valid JSON: {error}.") from error

        translated = translate(payload, prompt_version=config.prompt_version)
        # Acceptance and reporting read the same id, so they cannot drift.
        accepted = runtime.accepted_names(config, engine.model_id)
        if translated.model not in accepted:
            raise invalid_request(
                "unknown_model",
                f"Unknown model {translated.model!r}. This server serves {engine.model_id!r}.",
                "model",
            )
        if not engine.ready:
            raise unavailable("The model is still loading.")
        try:
            results, mode, timing, fallback_from = engine.score(translated.rows)
        except ApiError:
            raise
        except (ValueError, RuntimeError) as error:
            raise backend_failed(f"{type(error).__name__}: {error}") from error

        # The response reports the model that answered, never the alias sent.
        body = assemble.build_response(
            engine.model_id, translated.plans, results,
            temperature=config.temperature_for, mode=mode, backend=config.backend,
            metadata=engine.metadata, timing=timing, fallback_from=fallback_from,
        )
        return JSONResponse(body, headers={"x-semif-mode": mode})

    @app.exception_handler(404)
    async def handle_not_found(request: Request, _exception):
        error = not_found(f"Unknown endpoint {request.url.path!r}.")
        return _error_response(error, getattr(request.state, "request_id", ""))

    return app


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="semif-serve", description="Serve SemIf over the Jev HTTP contract.")
    parser.add_argument("--model", help="Hugging Face repo id or local checkpoint directory")
    parser.add_argument("--revision", help="40-character commit id, or a manifest label for local directories")
    parser.add_argument("--backend", choices=runtime.BACKENDS)
    parser.add_argument("--device", choices=("auto", "cuda", "mps"))
    parser.add_argument("--dtype", choices=("bfloat16", "float16", "float32"))
    parser.add_argument("--mlx-bits", type=int, choices=(4, 8))
    parser.add_argument("--mlx-cache-limit-mib", type=int)
    parser.add_argument("--gguf")
    parser.add_argument("--llama-threads", type=int)
    parser.add_argument("--max-input-tokens", type=int)
    parser.add_argument("--mode", choices=runtime.MODES)
    parser.add_argument("--prompt-version", choices=sorted(slots.CAPACITY))
    parser.add_argument("--temperature", help="A positive float, or a path to a per-question-type JSON map")
    parser.add_argument("--host")
    parser.add_argument("--port", type=int)
    return parser


def config_from_args(argv=None) -> runtime.Config:
    args = build_parser().parse_args(argv)
    config = runtime.Config.from_env()
    for name in ("model", "revision", "backend", "device", "dtype", "mlx_bits", "mlx_cache_limit_mib",
                 "gguf", "llama_threads", "max_input_tokens", "mode", "prompt_version", "host", "port"):
        value = getattr(args, name)
        if value is not None:
            setattr(config, name, value)
    if args.temperature is not None:
        config.set_temperature(args.temperature)
    config.validate()
    return config


def main(argv=None) -> None:
    import uvicorn

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s %(message)s")
    config = config_from_args(argv)
    if not config.api_key:
        logger.warning("SEMIF_API_KEY is unset: every Authorization header is accepted on %s:%s",
                       config.host, config.port)
    engine = runtime.Runtime(config)
    logger.info("Loading %s on the %s backend", engine.model_id, config.backend)
    engine.load()
    logger.info("Ready. Point a TypeSafe SDK at http://%s:%s", config.host, config.port)
    uvicorn.run(create_app(config, engine), host=config.host, port=config.port, log_level="info")


if __name__ == "__main__":
    main()
