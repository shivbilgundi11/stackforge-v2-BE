"""The arq queue, and the jobs on it (D-05, first used by M18).

Two jobs, and neither of them is a fetcher:

  * `build_export` — renders a queued export and writes the bytes back onto
    its row. Only bundles over the size threshold get here; a 2 KB Markdown
    export goes through the request that asked for it.
  * `purge_expired_exports` — the scheduled delete. `PRD.md` and M18's
    definition of done both require that expired export objects leave storage,
    and because the bytes live on the row this is one statement rather than a
    reconciliation between a table and a bucket.

**Enqueuing degrades rather than fails.** `enqueue` returns False when Redis
is unreachable and the caller builds inline instead. The alternative is a
`pending` row that nothing will ever pick up — a job queue being down should
make the product slower, not silently lossy, which is the same trade the quota
counter makes.
"""

from __future__ import annotations

from typing import Any, ClassVar

from arq import cron
from arq.connections import RedisSettings
from sqlalchemy.ext.asyncio import AsyncSession

from app.api.deps import Identity
from app.core.config import settings
from app.core.database import SessionLocal
from app.core.logging import configure_logging, get_logger
from app.models.export import Export, ExportStatus
from app.services import export_service

logger = get_logger("worker")

#: Serialised through arq's default pickle. Job names are strings on both
#: sides, so they live here once rather than as literals at call sites.
BUILD_EXPORT = "build_export"
PURGE_EXPORTS = "purge_expired_exports"


def redis_settings() -> RedisSettings:
    return RedisSettings.from_dsn(settings.redis_url)


async def enqueue(job: str, *args: Any) -> bool:
    """Put a job on the queue. False means it did not land.

    Never raises. Every caller has an inline fallback, and one that had to
    wrap this in a try/except would eventually be written without one.
    """
    from arq import create_pool

    try:
        pool = await create_pool(redis_settings())
    except Exception as exc:
        logger.warning("worker.enqueue_failed", job=job, error=str(exc))
        return False

    try:
        await pool.enqueue_job(job, *args)
    except Exception as exc:
        logger.warning("worker.enqueue_failed", job=job, error=str(exc))
        return False
    else:
        return True
    finally:
        await pool.aclose()


# ── jobs ─────────────────────────────────────────────────────────────────────


async def build_export(_context: dict[str, Any], export_id: str) -> str:
    """Render a queued export.

    A failure is recorded on the row rather than left to arq's retry log. The
    user is watching a progress indicator; "failed, and here is why" is a state
    the UI can show, and a job that vanished is not.
    """
    from app.services.artifacts import sources

    async with SessionLocal() as session:
        export = await session.get(Export, export_id)
        if export is None:
            logger.warning("worker.export_missing", export_id=export_id)
            return "missing"
        if export.status is not ExportStatus.PENDING:
            # Already built, by an inline fallback or a duplicate delivery.
            # arq guarantees at-least-once, so this is a normal path.
            return str(export.status.value)

        try:
            identity = await _identity_of(session, export)
            source = await sources.resolve(
                session,
                source_type=export.source_type,
                source_id=export.source_id,
                identity=identity,
            )
            rendered = export_service.render(
                source,
                artifact_type=export.artifact_type,
                export_format=export.format,
            )
            export_service.complete(export, rendered)
            logger.info(
                "worker.export_built",
                export_id=export_id,
                size_bytes=export.size_bytes,
            )
        except Exception as exc:
            logger.exception("worker.export_failed", export_id=export_id)
            export_service.fail(export, str(exc) or type(exc).__name__)

        await session.commit()
        return str(export.status.value)


async def _identity_of(session: AsyncSession, export: Export) -> Identity:
    """Rebuild the caller's identity from the row.

    The job runs outside a request, so there is no cookie and no bearer token.
    Reconstructing the identity from the owner columns means the generator
    resolves exactly the sources that caller could see — a background job must
    not be a way to read something the request could not.
    """
    from app.services import auth_service

    if export.user_id is not None:
        user = await auth_service.get_user(session, export.user_id)
        return Identity(user=user, anonymous_id=None, session_id=None)
    return Identity(user=None, anonymous_id=export.anonymous_session_id, session_id=None)


async def purge_expired_exports(_context: dict[str, Any]) -> int:
    async with SessionLocal() as session:
        removed = await export_service.purge_expired(session)
        await session.commit()
        return removed


# ── worker ───────────────────────────────────────────────────────────────────


async def startup(_context: dict[str, Any]) -> None:
    configure_logging()
    logger.info("worker.start", environment=settings.environment)


async def shutdown(_context: dict[str, Any]) -> None:
    from app.core.database import dispose_engine

    await dispose_engine()
    logger.info("worker.stop")


class WorkerSettings:
    """`uv run arq app.workers.queue.WorkerSettings`."""

    functions: ClassVar[list[Any]] = [build_export, purge_expired_exports]
    cron_jobs: ClassVar[list[Any]] = [
        # 03:17 rather than on the hour. Every scheduled job in every service
        # firing at :00 is how a database gets a load spike it did not need.
        # `cron` is typed against arq's own coroutine protocol, which a
        # plain async function satisfies structurally but not nominally.
        cron(purge_expired_exports, hour=3, minute=17, run_at_startup=False)  # type: ignore[arg-type]
    ]
    on_startup = startup
    on_shutdown = shutdown
    max_jobs = 4
    job_timeout = 120
    keep_result = 3600


__all__ = [
    "BUILD_EXPORT",
    "PURGE_EXPORTS",
    "WorkerSettings",
    "build_export",
    "enqueue",
    "purge_expired_exports",
]
