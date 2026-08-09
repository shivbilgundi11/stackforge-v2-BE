"""Tool-run history and quota.

Small surface, but it is what makes a run more than a fire-and-forget POST:
the dashboard's recent-activity feed, the "open a previous result" path, and
the quota figures the upgrade dialog needs.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any

from fastapi import APIRouter, Query
from pydantic import BaseModel, ConfigDict

from app.api.deps import CallerIdentity, Db
from app.core.errors import NotFound
from app.core.responses import Envelope, ok
from app.schemas.tools import QuotaOut, ToolRunOut
from app.services import tool_service

router = APIRouter(tags=["runs"])


class RunSummaryOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: str
    tool_slug: str
    workflow: str
    source: str
    duration_ms: int
    saved: bool
    created_at: datetime


class RunDetailOut(RunSummaryOut):
    """A stored run, reopened.

    `output` is typed as the same `ToolRunOut` a live run returns, so the
    frontend renders a reopened run through exactly the same path as a fresh
    one rather than growing a second, subtly different renderer.
    """

    input: dict[str, Any]
    output: ToolRunOut


@router.get("", response_model=Envelope[list[RunSummaryOut]], name="list_runs")
async def list_runs(
    db: Db,
    identity: CallerIdentity,
    workflow: str | None = None,
    limit: int = Query(default=10, ge=1, le=100),
) -> dict[str, Any]:
    runs = await tool_service.recent_runs(db, identity, workflow=workflow, limit=limit)
    return ok([RunSummaryOut.model_validate(run) for run in runs])


@router.get("/quota", response_model=Envelope[QuotaOut], name="get_quota")
async def get_quota(identity: CallerIdentity) -> dict[str, Any]:
    quota = await tool_service.check_quota(identity)
    return ok(quota.to_schema())


@router.get("/{run_id}", response_model=Envelope[RunDetailOut], name="get_run")
async def get_run(db: Db, identity: CallerIdentity, run_id: str) -> dict[str, Any]:
    run = await tool_service.get_run(db, run_id, identity)
    if run is None:
        # 404 rather than 403 for a run owned by someone else: confirming that
        # a run id exists is itself information.
        raise NotFound("No such run.")

    return ok(
        RunDetailOut(
            id=run.id,
            tool_slug=run.tool_slug,
            workflow=run.workflow,
            source=run.source,
            duration_ms=run.duration_ms,
            saved=run.saved,
            created_at=run.created_at,
            input=run.input,
            output=_envelope(run),
        )
    )


def _envelope(run: Any) -> ToolRunOut:
    """Rebuild the wire shape from a stored run.

    Runs written before the stored blob became the full envelope hold only the
    inner `ToolOutput`, so the identifying fields are taken from the row and
    the blob supplies whatever it has. Reading the row rather than trusting the
    blob also means a stored `run_id` can never disagree with the row it lives
    on.
    """
    blob = dict(run.output or {})
    for key in ("run_id", "tool_slug", "source", "duration_ms", "created_at"):
        blob.pop(key, None)

    return ToolRunOut(
        run_id=run.id,
        tool_slug=run.tool_slug,
        source=run.source,
        duration_ms=run.duration_ms,
        created_at=run.created_at,
        **blob,
    )
