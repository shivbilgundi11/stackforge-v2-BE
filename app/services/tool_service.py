"""The shared path every tool endpoint takes.

    quota → compute → AI enrichment → persist → provenance → envelope

Every tool endpoint is three lines: parse, call `run_tool` with its compute
function, return. Quota, logging, provenance, and AI enrichment are impossible
to forget because they are not the tool author's responsibility — which is the
difference between this and writing the same shape 28 times.

`compute` takes plain values and returns a `ToolOutput`. No request object, no
session, no I/O. That constraint is load-bearing: it is what makes a tool's
test `assert result.metrics["monthly_cost"] == Decimal("126.00")` rather than a
fixture-heavy integration test, and it is why the arithmetic in `cost_service`
can be verified by hand.
"""

from __future__ import annotations

import time
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from typing import Any, Final

from pydantic import BaseModel
from redis.exceptions import RedisError
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.api.deps import Identity
from app.core.database import new_id, utcnow
from app.core.errors import QuotaExceeded
from app.core.logging import get_logger
from app.core.redis import Keys, get_redis
from app.models.catalog import DataSource, GpuPricing, ModelPricing
from app.models.tool_run import RunSource, ToolRun
from app.models.user import Plan
from app.schemas.tools import (
    AiMeta,
    Provenance,
    ProvenanceSource,
    QuotaOut,
    ToolOutput,
    ToolRunOut,
)
from app.services import provenance_service

logger = get_logger("tools")

# Daily run allowance per plan. Anonymous callers get a small taste — enough to
# see that the tool works, not enough to make signing up optional.
DAILY_RUN_LIMIT: Final[dict[str, int]] = {
    "anonymous": 5,
    Plan.FREE.value: 25,
    Plan.PRO.value: 1_000,
    Plan.TEAM.value: 5_000,
    Plan.ENTERPRISE.value: 100_000,
}

Compute = Callable[[], ToolOutput] | Callable[[], Awaitable[ToolOutput]]


@dataclass(frozen=True)
class QuotaState:
    metric: str
    limit: int
    used: int
    period: str
    resets_at: datetime
    plan: str

    @property
    def remaining(self) -> int:
        return max(0, self.limit - self.used)

    @property
    def exceeded(self) -> bool:
        return self.used >= self.limit

    def to_schema(self) -> QuotaOut:
        return QuotaOut(
            metric=self.metric,
            limit=self.limit,
            used=self.used,
            remaining=self.remaining,
            period=self.period,
            resets_at=self.resets_at,
            plan=self.plan,
        )


def _plan_key(identity: Identity) -> str:
    return identity.plan.value if identity.is_authenticated else "anonymous"


def _period() -> tuple[str, datetime]:
    """UTC day. Returns the bucket label and when it rolls over."""
    now = utcnow()
    tomorrow = (now + timedelta(days=1)).replace(hour=0, minute=0, second=0, microsecond=0)
    return now.strftime("%Y-%m-%d"), tomorrow.replace(tzinfo=UTC)


async def check_quota(identity: Identity, *, metric: str = "tool_runs") -> QuotaState:
    """Read the counter without incrementing it.

    Fail-open on a Redis outage: an unavailable cache must not stop people
    using the product. The trade is that an outage window is uncapped, which
    is the right way round — the alternative is a cache failure looking like a
    billing failure to every user at once.
    """
    period, resets_at = _period()
    plan = _plan_key(identity)
    limit = DAILY_RUN_LIMIT.get(plan, DAILY_RUN_LIMIT[Plan.FREE.value])

    used = 0
    try:
        raw = await get_redis().get(Keys.quota(metric, identity.key, period))
        used = int(raw) if raw else 0
    except (RedisError, OSError, ValueError) as exc:
        logger.warning("quota.read_failed", identity=identity.key, error=str(exc))

    return QuotaState(
        metric=metric,
        limit=limit,
        used=used,
        period=period,
        resets_at=resets_at,
        plan=plan,
    )


async def consume_quota(identity: Identity, *, metric: str = "tool_runs") -> None:
    period, resets_at = _period()
    key = Keys.quota(metric, identity.key, period)
    try:
        redis = get_redis()
        value = await redis.incr(key)
        if value == 1:
            # Expire slightly past the rollover so a run at 23:59:59 does not
            # leave a key that outlives its own period.
            await redis.expireat(key, int(resets_at.timestamp()) + 60)
    except (RedisError, OSError) as exc:
        logger.warning("quota.increment_failed", identity=identity.key, error=str(exc))


async def run_tool(
    db: AsyncSession,
    *,
    slug: str,
    workflow: str,
    payload: BaseModel,
    identity: Identity,
    compute: Compute,
    enrich: Callable[[ToolOutput], Awaitable[AiMeta | None]] | None = None,
) -> ToolRunOut:
    quota = await check_quota(identity)
    if quota.exceeded:
        raise QuotaExceeded(
            f"You have used all {quota.limit} runs for today.",
            details={"quota": quota.to_schema().model_dump(mode="json")},
        )

    started = time.perf_counter()
    result = compute()
    if isinstance(result, Awaitable):
        result = await result
    output: ToolOutput = result

    # AI enrichment is never blocking and never fatal. A tool whose deterministic
    # answer is correct must not fail because a model call timed out — the
    # arithmetic is the product, the prose is the garnish.
    ai: AiMeta | None = None
    source = RunSource.RULE_BASED
    if enrich is not None:
        try:
            ai = await enrich(output)
            if ai is not None:
                source = RunSource.HYBRID
        except Exception as exc:
            logger.warning("tools.enrichment_failed", slug=slug, error=str(exc))
            output.warnings.append(_degraded_warning("AI commentary was unavailable for this run."))

    duration_ms = int((time.perf_counter() - started) * 1000)
    provenance = await build_provenance(db, output.sourced_from)

    run = ToolRun(
        id=new_id("run"),
        tool_slug=slug,
        workflow=workflow,
        user_id=identity.user.id if identity.user else None,
        anonymous_session_id=None if identity.user else identity.anonymous_id,
        input=payload.model_dump(mode="json"),
        output=output.model_dump(mode="json"),
        source=source,
        duration_ms=duration_ms,
        created_at=utcnow(),
    )
    db.add(run)
    await db.flush()

    await consume_quota(identity)

    logger.info(
        "tools.run",
        slug=slug,
        workflow=workflow,
        run_id=run.id,
        duration_ms=duration_ms,
        source=source.value,
        authenticated=identity.is_authenticated,
    )

    return ToolRunOut(
        run_id=run.id,
        tool_slug=slug,
        source=source.value,
        duration_ms=duration_ms,
        created_at=run.created_at,
        metrics=output.metrics,
        tables=output.tables,
        series=output.series,
        artifacts=output.artifacts,
        warnings=output.warnings,
        provenance=provenance,
        ai=ai,
    )


def _degraded_warning(message: str) -> Any:
    from app.schemas.tools import ToolWarning

    return ToolWarning(level="info", message=message)


async def build_provenance(db: AsyncSession, entity_ids: list[str]) -> Provenance:
    """The verification dates of the exact rows this run read.

    A global "catalog updated 3 days ago" chip is worse than none: it is true
    and irrelevant. What the user needs to know is how old the numbers *in
    front of them* are, and that is the oldest of the rows actually touched.
    """
    if not entity_ids:
        return Provenance()

    unique = list(dict.fromkeys(entity_ids))
    now = utcnow()
    sources: list[ProvenanceSource] = []

    model_rows = (
        await db.execute(
            select(ModelPricing, DataSource)
            .join(DataSource, DataSource.id == ModelPricing.source_id)
            .where(ModelPricing.id.in_(unique))
        )
    ).all()
    gpu_rows = (
        await db.execute(
            select(GpuPricing, DataSource)
            .join(DataSource, DataSource.id == GpuPricing.source_id)
            .where(GpuPricing.id.in_(unique))
        )
    ).all()

    # Typed explicitly: the two result sets have different row models, and
    # splatting them into one list otherwise widens the element type to
    # `object` and loses every attribute below.
    priced: list[tuple[ModelPricing | GpuPricing, DataSource]] = [
        *((row, source) for row, source in model_rows),
        *((row, source) for row, source in gpu_rows),
    ]

    seen: set[str] = set()
    for row, source in priced:
        age = provenance_service.verification_age(row.last_verified_at, now=now)
        # One entry per source, keeping the oldest verification behind it —
        # eight OpenAI models should not produce eight identical chips.
        if source.slug in seen:
            for existing in sources:
                if existing.name == source.name and row.last_verified_at < (
                    existing.last_verified_at
                ):
                    existing.last_verified_at = row.last_verified_at
                    existing.age_days = age
                    existing.variant = provenance_service.variant_for(age)
            continue
        seen.add(source.slug)
        sources.append(
            ProvenanceSource(
                name=source.name,
                url=source.url,
                last_verified_at=row.last_verified_at,
                age_days=age,
                variant=provenance_service.variant_for(age),
            )
        )

    if not sources:
        return Provenance()

    oldest = min(source.last_verified_at for source in sources)
    oldest_age = provenance_service.verification_age(oldest, now=now)
    return Provenance(
        oldest_verified_at=oldest,
        variant=provenance_service.variant_for(oldest_age),
        sources=sorted(sources, key=lambda s: s.last_verified_at),
    )


async def recent_runs(
    db: AsyncSession,
    identity: Identity,
    *,
    workflow: str | None = None,
    limit: int = 10,
) -> list[ToolRun]:
    stmt = select(ToolRun).order_by(ToolRun.created_at.desc()).limit(limit)
    if identity.user:
        stmt = stmt.where(ToolRun.user_id == identity.user.id)
    elif identity.anonymous_id:
        stmt = stmt.where(ToolRun.anonymous_session_id == identity.anonymous_id)
    else:
        return []
    if workflow:
        stmt = stmt.where(ToolRun.workflow == workflow)
    return list((await db.execute(stmt)).scalars().all())


async def get_run(db: AsyncSession, run_id: str, identity: Identity) -> ToolRun | None:
    """A run is readable by whoever created it.

    Sharing a run publicly is M18's job and needs a share token; until then a
    run id is not a capability.
    """
    run = await db.get(ToolRun, run_id)
    if run is None:
        return None
    if identity.user and run.user_id == identity.user.id:
        return run
    if identity.anonymous_id and run.anonymous_session_id == identity.anonymous_id:
        return run
    return None


async def claim_anonymous_runs(db: AsyncSession, *, anonymous_id: str, user_id: str) -> int:
    """Move an anonymous session's runs onto the account it just created.

    Someone who ran four calculations before signing up should find them in
    their history afterwards; losing them is the moment they learn that
    signing up cost them something.
    """
    runs = (
        (await db.execute(select(ToolRun).where(ToolRun.anonymous_session_id == anonymous_id)))
        .scalars()
        .all()
    )
    for run in runs:
        run.user_id = user_id
        run.anonymous_session_id = None
    if runs:
        logger.info("tools.runs_claimed", count=len(runs), user_id=user_id)
    return len(runs)
