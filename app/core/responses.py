"""The response envelope. Every endpoint, no exceptions.

Declared as generic Pydantic models so the OpenAPI schema — and therefore the
TypeScript generated from it — describes the real wire shape rather than just
the payload.
"""

from __future__ import annotations

from typing import Any, Generic, TypeVar

from pydantic import BaseModel, Field

from app.core.context import get_context

T = TypeVar("T")


class PageMeta(BaseModel):
    cursor: str | None = None
    next_cursor: str | None = None
    limit: int
    total: int | None = None


class Meta(BaseModel):
    request_id: str
    page: PageMeta | None = None


class ErrorBody(BaseModel):
    code: str = Field(description="Stable machine-readable error code.")
    message: str
    details: dict[str, Any] | None = None


class Envelope(BaseModel, Generic[T]):
    """Declared on every route as `response_model=Envelope[XOut]`.

    Without it FastAPI emits no response schema, `openapi-typescript` generates
    `unknown`, and the frontend is back to the runtime-reflection problem this
    whole contract exists to remove.
    """

    success: bool
    data: T | None = None
    error: ErrorBody | None = None
    meta: Meta


def ok(data: Any, *, page: PageMeta | None = None) -> dict[str, Any]:
    return {
        "success": True,
        "data": data,
        "meta": {"request_id": get_context().request_id, "page": page},
    }


def fail(
    code: str,
    message: str,
    *,
    details: dict[str, Any] | None = None,
) -> dict[str, Any]:
    return {
        "success": False,
        "error": {"code": code, "message": message, "details": details},
        "meta": {"request_id": get_context().request_id},
    }
