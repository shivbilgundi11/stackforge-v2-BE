"""Saved runs, and the purge that keeps the table from becoming a landfill.

Every run is logged whether or not the user saves it (M08). That is what makes
recent activity useful from the first session and what feeds the North Star
metric. Saving is a second state on the same row, not a copy into another
table — a saved run and the run it came from must never be able to disagree.

Unsaved rows are purged after 30 days. Saved rows are never purged, which is
the only durability promise the product makes about a run and therefore the
one thing this module cannot get wrong.
"""

from __future__ import annotations

from datetime import timedelta
from typing import Any, Final, cast

from sqlalchemy import delete, func, select
from sqlalchemy.engine import CursorResult
from sqlalchemy.ext.asyncio import AsyncSession

from app.api.deps import Identity
from app.core.database import utcnow
from app.core.errors import Forbidden
from app.core.logging import get_logger
from app.models.tool_run import ToolRun
from app.models.user import User
from app.repositories import scoped

logger = get_logger("runs")

#: How long an unsaved run survives. Long enough that "I closed the tab" is
#: recoverable; short enough that anonymous traffic does not become the
#: largest table in the database inside a year.
RETENTION_DAYS: Final = 30


async def save(db: AsyncSession, run_id: str, identity: Identity) -> ToolRun:
    """Promote an ephemeral run to a saved one.

    Requires an account. An anonymous user can run anything and export the
    result immediately, but keeping it is what the account is for — and it is
    the highest-intent moment the product has to ask for one.
    """
    if identity.user is None:
        raise Forbidden("Sign in to keep this run.")

    run = await scoped.get_visible(db, ToolRun, run_id, identity, label="run")
    run.saved = True
    await db.flush()
    logger.info("runs.saved", run_id=run.id, user_id=identity.user.id)
    return run


async def unsave(db: AsyncSession, run_id: str, identity: Identity) -> ToolRun:
    """Drop back to ephemeral.

    The row stays — it is still real history and still counts toward usage.
    What changes is that it is no longer exempt from the purge.
    """
    run = await scoped.get_visible(db, ToolRun, run_id, identity, label="run")
    run.saved = False
    await db.flush()
    return run


async def delete_run(db: AsyncSession, run_id: str, identity: Identity) -> None:
    run = await scoped.get_visible(db, ToolRun, run_id, identity, label="run")
    await db.delete(run)
    await db.flush()


async def list_runs(
    db: AsyncSession,
    identity: Identity,
    *,
    workflow: str | None = None,
    tool_slug: str | None = None,
    saved_only: bool = False,
    limit: int = 20,
) -> list[ToolRun]:
    statement = scoped.visible_to(ToolRun, identity).order_by(ToolRun.created_at.desc())
    if workflow:
        statement = statement.where(ToolRun.workflow == workflow)
    if tool_slug:
        statement = statement.where(ToolRun.tool_slug == tool_slug)
    if saved_only:
        statement = statement.where(ToolRun.saved.is_(True))
    return list((await db.execute(statement.limit(limit))).scalars().all())


async def purge_expired(db: AsyncSession, *, now: object = None) -> int:
    """Delete unsaved runs past the retention window.

    Deliberately not filtered by owner: this is the scheduled job, and it is
    the one place in the module that operates across users. It is also the one
    place the `saved` predicate is load-bearing — a bug here silently destroys
    work a user chose to keep, so the condition is asserted directly in the
    test rather than inferred from a count.
    """
    cutoff = (now or utcnow()) - timedelta(days=RETENTION_DAYS)  # type: ignore[operator]
    result = await db.execute(
        delete(ToolRun).where(ToolRun.saved.is_(False), ToolRun.created_at < cutoff)
    )
    removed = int(cast("CursorResult[Any]", result).rowcount or 0)
    if removed:
        logger.info("runs.purged", count=removed, retention_days=RETENTION_DAYS)
    return removed


async def counts_for(db: AsyncSession, user: User) -> dict[str, int]:
    """Totals for the dashboard, in one round trip rather than three."""
    row = (
        await db.execute(
            select(
                func.count().label("total"),
                func.count().filter(ToolRun.saved.is_(True)).label("saved"),
                func.count()
                .filter(ToolRun.created_at >= func.date_trunc("day", func.now()))
                .label("today"),
            ).where(ToolRun.user_id == user.id)
        )
    ).one()
    return {"total": int(row.total), "saved": int(row.saved), "today": int(row.today)}
