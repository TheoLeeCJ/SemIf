"""Error type and JSON body for the Jev-compatible surface.

TypeSafe does not publish a schema for its error bodies, so this shape is
SemIf's own. It is documented in docs/JEV_API_COMPAT.md so a client can rely
on it.
"""

from __future__ import annotations


class ApiError(Exception):
    """An error with the HTTP status and body the API layer should return."""

    def __init__(self, status: int, kind: str, code: str, message: str, param: str | None = None):
        super().__init__(message)
        self.status = status
        self.kind = kind
        self.code = code
        self.message = message
        self.param = param

    @property
    def body(self) -> dict:
        error = {"type": self.kind, "code": self.code, "message": self.message}
        if self.param is not None:
            error["param"] = self.param
        return {"error": error}


def invalid_request(code: str, message: str, param: str | None = None) -> ApiError:
    return ApiError(422, "invalid_request_error", code, message, param)


def unauthorized(message: str) -> ApiError:
    return ApiError(401, "authentication_error", "invalid_api_key", message)


def not_found(message: str) -> ApiError:
    return ApiError(404, "not_found_error", "unknown_endpoint", message)


def backend_failed(message: str) -> ApiError:
    return ApiError(500, "api_error", "backend_failed", message)


def unavailable(message: str) -> ApiError:
    return ApiError(503, "api_error", "model_unavailable", message)
