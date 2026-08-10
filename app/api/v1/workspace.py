"""The dashboard and the search behind ⌘K (M17)."""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter, Query

from app.api.deps import CurrentUser, Db
from app.core.responses import Envelope, ok
from app.schemas.workspace import DashboardOut, SearchHitOut
from app.services import dashboard_service

router = APIRouter(tags=["workspace"])


@router.get("/dashboard", response_model=Envelope[DashboardOut], name="get_dashboard")
async def get_dashboard(db: Db, user: CurrentUser) -> dict[str, Any]:
    """Every panel in one request.

    Seven panels, seven round trips would be seven loading states on the first
    screen after signing in. The aggregate is small and its parts are all
    owner-scoped reads of the same user's rows.
    """
    return ok(await dashboard_service.overview(db, user))


@router.get("/search", response_model=Envelope[list[SearchHitOut]], name="search_workspace")
async def search_workspace(
    db: Db,
    user: CurrentUser,
    q: str = Query(min_length=1, max_length=120),
    limit: int = Query(default=20, ge=1, le=50),
) -> dict[str, Any]:
    return ok(await dashboard_service.search(db, user, q, limit=limit))
