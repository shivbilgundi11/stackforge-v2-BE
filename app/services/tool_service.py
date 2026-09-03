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
from typing import Any, Final

from pydantic import BaseModel
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.api.deps import Identity
from app.core.database import new_id, utcnow
from app.core.logging import get_logger
from app.models.billing import Metric
from app.models.catalog import DataSource, GpuPricing, ModelPricing
from app.models.tool_run import RunSource, ToolRun
from app.schemas.tools import (
    AiMeta,
    Provenance,
    ProvenanceSource,
    ToolOutput,
    ToolRunOut,
)
from app.services import feature_service, provenance_service

logger = get_logger("tools")

Compute = Callable[[], ToolOutput] | Callable[[], Awaitable[ToolOutput]]

#: The run allowance used to be a dict here, alongside four others elsewhere.
#: It now lives in `plan_quotas` and is read through `FeatureService` (M20).
#: These two functions are kept as the tool engine's vocabulary — the routes
#: and the run history endpoint ask about "quota", not about metrics.
RUN_METRIC: Final = Metric.TOOL_RUNS_PER_DAY


async def check_quota(db: AsyncSession, identity: Identity) -> feature_service.QuotaState:
    """Read the counter without incrementing it."""
    return await feature_service.check(db, identity, RUN_METRIC)


async def consume_quota(db: AsyncSession, identity: Identity) -> feature_service.QuotaState:
    """Take one run, or raise `QuotaExceeded` with the real figures."""
    return await feature_service.consume(db, identity, RUN_METRIC)


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
    # Consumed up front rather than after the run persists (M20). The counter
    # increments and compares in one Redis round trip, so two requests arriving
    # together at the limit are both refused — where a read-then-run-then-write
    # would let both through. The cost is that a compute that raises has still
    # spent a run, which is the right way round: the alternative is a tool that
    # can be run without ever being counted.
    await consume_quota(db, identity)

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
    created_at = utcnow()
    run_id = new_id("run")

    wire = ToolRunOut(
        run_id=run_id,
        tool_slug=slug,
        source=source.value,
        duration_ms=duration_ms,
        created_at=created_at,
        metrics=output.metrics,
        tables=output.tables,
        series=output.series,
        artifacts=output.artifacts,
        warnings=output.warnings,
        provenance=provenance,
        ai=ai,
    )

    run = ToolRun(
        id=run_id,
        tool_slug=slug,
        workflow=workflow,
        user_id=identity.user.id if identity.user else None,
        input=payload.model_dump(mode="json"),
        # The full wire shape, not the bare `ToolOutput`. Provenance is
        # attached here rather than by `compute`, so storing the inner object
        # would drop it — and a reopened run would render its figures with no
        # verification dates at all. Numbers whose provenance disappears the
        # moment you revisit them are worse than numbers that never had any:
        # the chips are present on first view, so their absence later reads as
        # "this data has no source" rather than "we forgot to save it".
        output=wire.model_dump(mode="json"),
        source=source,
        duration_ms=duration_ms,
        created_at=created_at,
    )
    db.add(run)
    await db.flush()

    logger.info(
        "tools.run",
        slug=slug,
        workflow=workflow,
        run_id=run.id,
        duration_ms=duration_ms,
        source=source.value,
    )

    return wire


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
    stmt = (
        select(ToolRun)
        .where(ToolRun.user_id == identity.user.id)
        .order_by(ToolRun.created_at.desc())
        .limit(limit)
    )
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
    if run.user_id == identity.user.id:
        return run
    return None
