"""Middleware stack, outermost first.

  1. RequestContext  — assigns the request id, writes the access log
  2. CORS            (Starlette's own, added in main.py)
  3. SecurityHeaders

Auth, rate limiting, and quota are FastAPI dependencies rather than middleware,
so they compose per route and are skipped where they do not apply (health,
webhooks).
"""

from __future__ import annotations

import time
import uuid
from collections.abc import Awaitable, Callable

from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import Response
from starlette.types import ASGIApp

from app.core.config import settings
from app.core.context import RequestContext, as_dict, set_context
from app.core.logging import get_logger

logger = get_logger("http")

Next = Callable[[Request], Awaitable[Response]]


class RequestContextMiddleware(BaseHTTPMiddleware):
    """Assigns the request id, and logs the request on the way out.

    These are one middleware rather than two on purpose. `BaseHTTPMiddleware`
    runs the downstream app in its own anyio task, and ContextVar writes made
    downstream do not propagate back out to an enclosing middleware. Split
    across two, the logger would read an empty context and every access line
    would be missing its request id and user — which is exactly the
    correlation the structured log exists to provide.
    """

    async def dispatch(self, request: Request, call_next: Next) -> Response:
        incoming = request.headers.get("X-Request-ID", "")
        request_id = incoming if _looks_like_id(incoming) else f"req_{uuid.uuid4().hex}"

        set_context(RequestContext(request_id=request_id))
        request.state.request_id = request_id

        started = time.perf_counter()
        try:
            response = await call_next(request)
        except Exception:
            logger.exception(
                "request.failed",
                method=request.method,
                path=request.url.path,
                duration_ms=round((time.perf_counter() - started) * 1000, 2),
            )
            raise

        response.headers["X-Request-ID"] = request_id

        # Identity is bound by the auth dependency during the handler, so it is
        # read here at close, when it is known.
        logger.info(
            "request.complete",
            method=request.method,
            path=request.url.path,
            status=response.status_code,
            duration_ms=round((time.perf_counter() - started) * 1000, 2),
            **{k: v for k, v in as_dict().items() if k != "request_id"},
        )
        return response


def _looks_like_id(value: str) -> bool:
    # Echoing an arbitrary client string into every log line and response
    # header is a log-injection vector.
    return 8 <= len(value) <= 64 and value.replace("_", "").replace("-", "").isalnum()


#: `/docs` is the one route that returns a document rather than JSON, and
#: Swagger UI is loaded from a CDN with an inline bootstrap script — so the
#: API-wide CSP below blocks every asset the page needs and renders it blank.
#: These are the only paths that get the relaxed policy, and FastAPI serves
#: none of them when `ENVIRONMENT=production` (`docs_url` is None there), so
#: the loosened directives never reach a production response.
DOCS_PATHS = frozenset({"/docs", "/docs/oauth2-redirect"})

#: Swagger UI needs its stylesheet and bundle from jsdelivr, the FastAPI
#: favicon, `data:` images it inlines itself, and `unsafe-inline` for the
#: bootstrap script FastAPI generates. `connect-src 'self'` is what lets the
#: page fetch /openapi.json and lets "Try it out" call the API.
DOCS_CSP = (
    "default-src 'none'; "
    "script-src 'self' https://cdn.jsdelivr.net 'unsafe-inline'; "
    "style-src 'self' https://cdn.jsdelivr.net 'unsafe-inline'; "
    "img-src 'self' data: https://fastapi.tiangolo.com; "
    "font-src 'self' https://cdn.jsdelivr.net; "
    "connect-src 'self'; "
    "frame-ancestors 'none'"
)


class SecurityHeadersMiddleware(BaseHTTPMiddleware):
    def __init__(self, app: ASGIApp) -> None:
        super().__init__(app)
        self._headers = {
            "X-Content-Type-Options": "nosniff",
            "X-Frame-Options": "DENY",
            "Referrer-Policy": "strict-origin-when-cross-origin",
            "Permissions-Policy": "camera=(), microphone=(), geolocation=(), payment=()",
            # This is a JSON API. Nothing it returns should ever be rendered
            # as a document, so the CSP can be maximally restrictive.
            "Content-Security-Policy": "default-src 'none'; frame-ancestors 'none'",
        }
        if settings.is_production:
            self._headers["Strict-Transport-Security"] = (
                "max-age=63072000; includeSubDomains; preload"
            )

    async def dispatch(self, request: Request, call_next: Next) -> Response:
        response = await call_next(request)
        for key, value in self._headers.items():
            response.headers.setdefault(key, value)
        if request.url.path in DOCS_PATHS:
            response.headers["Content-Security-Policy"] = DOCS_CSP
        return response
