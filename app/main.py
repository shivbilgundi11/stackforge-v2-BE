from __future__ import annotations

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from typing import Any

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from sqlalchemy.exc import IntegrityError
from starlette.exceptions import HTTPException as StarletteHTTPException

from app.api.v1 import agents as agents_router
from app.api.v1 import architect as architect_router
from app.api.v1 import auth as auth_router
from app.api.v1 import catalog as catalog_router
from app.api.v1 import compare as compare_router
from app.api.v1 import cost as cost_router
from app.api.v1 import exports as exports_router
from app.api.v1 import health as health_router
from app.api.v1 import infra as infra_router
from app.api.v1 import projects as projects_router
from app.api.v1 import rag as rag_router
from app.api.v1 import roi as roi_router
from app.api.v1 import runs as runs_router
from app.api.v1 import shares as shares_router
from app.api.v1 import stacks as stacks_router
from app.api.v1 import templates as templates_router
from app.api.v1 import workspace as workspace_router
from app.core.config import settings
from app.core.database import dispose_engine
from app.core.errors import AppError, Conflict, InternalError, NotFound
from app.core.logging import configure_logging, get_logger
from app.core.middleware import (
    RequestContextMiddleware,
    SecurityHeadersMiddleware,
)
from app.core.redis import close_redis
from app.core.responses import fail

logger = get_logger("app")


@asynccontextmanager
async def lifespan(_app: FastAPI) -> AsyncIterator[None]:
    configure_logging()
    settings.validate_runtime()
    logger.info(
        "app.start",
        environment=settings.environment,
        ai_enabled=settings.ai_enabled,
        google_oauth=settings.google_oauth_enabled,
        github_oauth=settings.github_oauth_enabled,
    )
    yield
    await close_redis()
    await dispose_engine()
    logger.info("app.stop")


app = FastAPI(
    title="StackForge API",
    version="0.1.0",
    description="AI engineering workbench — plan, compare, and cost AI stacks.",
    openapi_url="/openapi.json",
    docs_url="/docs" if not settings.is_production else None,
    redoc_url=None,
    lifespan=lifespan,
    # Operation ids drive the names in the generated TypeScript. Without this
    # they include the router prefix and read terribly on the client.
    generate_unique_id_function=lambda route: route.name,
)

# Outermost first.
app.add_middleware(SecurityHeadersMiddleware)
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins,
    allow_credentials=True,  # the refresh and anon cookies depend on this
    allow_methods=["GET", "POST", "PATCH", "PUT", "DELETE", "OPTIONS"],
    allow_headers=["Authorization", "Content-Type", "X-Request-ID", "Idempotency-Key"],
    expose_headers=["X-Request-ID", "X-RateLimit-Remaining", "X-RateLimit-Reset", "Retry-After"],
    max_age=600,
)
# Added last, so it is outermost: the request id must exist before anything
# else can log, and the access line is written in this same task.
app.add_middleware(RequestContextMiddleware)


# ── Exception handlers ──────────────────────────────────────────────────────
# Four handlers. Everything that reaches the client goes through one of them,
# so no endpoint has to build an error body by hand.


@app.exception_handler(AppError)
async def handle_app_error(_request: Request, exc: AppError) -> JSONResponse:
    return JSONResponse(
        status_code=exc.http_status,
        content=fail(exc.code, exc.message, details=exc.details),
        headers=exc.headers,
    )


@app.exception_handler(RequestValidationError)
async def handle_validation_error(_request: Request, exc: RequestValidationError) -> JSONResponse:
    fields = [
        {
            # Drop the "body"/"query" prefix — the client maps these onto form
            # field names, and the location is noise there.
            "path": ".".join(str(part) for part in error["loc"][1:]) or str(error["loc"][0]),
            "message": error["msg"],
        }
        for error in exc.errors()
    ]
    return JSONResponse(
        status_code=422,
        content=fail(
            "VALIDATION_ERROR",
            "The request could not be validated.",
            details={"fields": fields},
        ),
    )


@app.exception_handler(IntegrityError)
async def handle_integrity_error(_request: Request, exc: IntegrityError) -> JSONResponse:
    logger.warning("db.integrity_error", constraint=str(getattr(exc.orig, "constraint_name", "")))
    error = Conflict()
    return JSONResponse(
        status_code=error.http_status,
        content=fail(error.code, error.message),
    )


@app.exception_handler(StarletteHTTPException)
async def handle_http_exception(_request: Request, exc: StarletteHTTPException) -> JSONResponse:
    """Starlette's own 404s and 405s, reshaped into the envelope so the client
    never sees two different error formats."""
    if exc.status_code == 404:
        error: AppError = NotFound()
    else:
        error = AppError(str(exc.detail))
        error.code = "HTTP_ERROR"
        error.http_status = exc.status_code
    return JSONResponse(
        status_code=error.http_status,
        content=fail(error.code, error.message),
        headers=getattr(exc, "headers", None),
    )


@app.exception_handler(Exception)
async def handle_unexpected(_request: Request, exc: Exception) -> JSONResponse:
    """Last resort. The traceback goes to the logs; the client gets a request id.

    A stack trace or a SQL fragment must never reach the caller.
    """
    logger.exception("request.unhandled", error=type(exc).__name__)
    error = InternalError()
    return JSONResponse(
        status_code=error.http_status,
        content=fail(error.code, error.message),
    )


# ── Routes ──────────────────────────────────────────────────────────────────

app.include_router(health_router.router)
app.include_router(auth_router.router, prefix="/api/v1/auth")
app.include_router(catalog_router.router, prefix="/api/v1/catalog")
app.include_router(cost_router.router, prefix="/api/v1/tools/cost")
app.include_router(compare_router.router, prefix="/api/v1/tools/compare")
app.include_router(agents_router.router, prefix="/api/v1/tools/agents")
app.include_router(infra_router.router, prefix="/api/v1/tools/infra")
app.include_router(rag_router.router, prefix="/api/v1/tools/rag")
app.include_router(roi_router.router, prefix="/api/v1/tools/roi")
app.include_router(architect_router.router, prefix="/api/v1/architect")
app.include_router(stacks_router.router, prefix="/api/v1/stacks")
app.include_router(runs_router.router, prefix="/api/v1/runs")
app.include_router(projects_router.router, prefix="/api/v1/projects")
app.include_router(exports_router.router, prefix="/api/v1/exports")
app.include_router(templates_router.router, prefix="/api/v1/templates")
# Mounted at the version root rather than under a prefix: this router owns both
# `/shares` (owner, authenticated) and `/s/{token}` (public, unauthenticated),
# and the public path is deliberately short — it is pasted into chat messages.
app.include_router(shares_router.router, prefix="/api/v1")
app.include_router(workspace_router.router, prefix="/api/v1")


@app.get("/", include_in_schema=False)
async def root() -> dict[str, Any]:
    return {"service": "stackforge-api", "docs": "/docs", "health": "/health"}
