"""HTTP application exposing the TypeSafe System One request and response shapes."""

from __future__ import annotations

import secrets

from fastapi import Depends, FastAPI, Header, HTTPException
from fastapi.responses import JSONResponse

from .system_one import SystemOneService, SystemOneValidationError


def create_app(service: SystemOneService, api_key: str | None = None) -> FastAPI:
    """Create an app around an already-loaded, single-model service."""
    app = FastAPI(
        title="SemIf System One API",
        version="0.1.0",
        description=(
            "A wire-compatible subset of TypeSafe's System One API backed by SemIf. "
            "It does not serve Jev and its probabilities are not calibrated as Jev probabilities."
        ),
    )

    def authorize(authorization: str | None = Header(default=None)) -> None:
        if api_key is None:
            return
        expected = f"Bearer {api_key}"
        if authorization is None or not secrets.compare_digest(authorization, expected):
            raise HTTPException(
                status_code=401,
                detail="Missing or invalid bearer token",
                headers={"WWW-Authenticate": "Bearer"},
            )

    @app.exception_handler(SystemOneValidationError)
    async def validation_error(_request, error: SystemOneValidationError):
        return JSONResponse(status_code=422, content={"detail": str(error)})

    @app.get("/healthz", include_in_schema=False)
    def health() -> dict:
        return {"status": "ready", "model": service.served_model}

    @app.get("/v1/models", dependencies=[Depends(authorize)])
    def models() -> dict:
        return service.models()

    @app.post("/v1/systemone", dependencies=[Depends(authorize)])
    def system_one(payload: dict) -> dict:
        return service.evaluate(payload)

    return app
