from __future__ import annotations

from typing import Any

from fastapi import APIRouter, Response, status

from app.core.database import check_database
from app.core.redis import check_redis
from app.core.responses import ok

router = APIRouter(tags=["system"])


@router.get("/health", summary="Liveness")
async def health() -> dict[str, Any]:
    """Touches nothing. Always fast.

    Separate from readiness on purpose: wiring a dependency check to the
    liveness probe restarts a healthy container when the database blips.
    """
    return ok({"status": "ok"})


@router.get("/health/ready", summary="Readiness")
async def ready(response: Response) -> dict[str, Any]:
    database_ok = await check_database()
    redis_ok = await check_redis()
    healthy = database_ok and redis_ok

    if not healthy:
        response.status_code = status.HTTP_503_SERVICE_UNAVAILABLE

    return ok(
        {
            "status": "ok" if healthy else "degraded",
            "checks": {"database": database_ok, "redis": redis_ok},
        }
    )
