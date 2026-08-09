"""Catalog reads.

Cached in Redis for 24 hours — this data changes daily at most, and every cost
calculation reads it, so an uncached catalog turns one tool run into a dozen
round trips.

Caching is **fail-open**: a Redis outage degrades to database reads rather than
502s. A cache that can take the site down is worse than no cache.
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from datetime import datetime
from decimal import Decimal
from typing import Any, Final

from redis.exceptions import RedisError
from sqlalchemy import Select, func, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.database import utcnow
from app.core.errors import NotFound
from app.core.logging import get_logger
from app.core.redis import get_redis
from app.models.catalog import (
    Compatibility,
    DataSource,
    GpuPricing,
    LifecycleStatus,
    ModelPricing,
    Tool,
    ToolStatus,
)
from app.schemas.catalog import (
    CompatibilityOut,
    CompatibilityPairOut,
    GpuOut,
    GraveyardEntryOut,
    ModelOut,
    ProvenanceOut,
    ToolOut,
)

logger = get_logger("catalog")

CACHE_TTL: Final = 60 * 60 * 24
CACHE_PREFIX: Final = "cache:catalog"

# Provenance thresholds. One definition, server-side, so the chip in the UI and
# the staleness report in the admin view can never disagree.
FRESH_DAYS: Final = 7
AGING_DAYS: Final = 30

# Hours in an average month, for turning an hourly GPU rate into the monthly
# figure every infra estimate actually wants. 730 = 365 * 24 / 12.
HOURS_PER_MONTH: Final = Decimal(730)


def _cache_key(kind: str, *parts: object) -> str:
    rendered = ":".join("_" if p is None else str(p) for p in parts)
    return f"{CACHE_PREFIX}:{kind}:{rendered}"


async def _cached(key: str) -> Any | None:
    try:
        raw = await get_redis().get(key)
    except (RedisError, OSError) as exc:
        logger.warning("catalog.cache_read_failed", key=key, error=str(exc))
        return None
    return json.loads(raw) if raw else None


async def _store(key: str, value: Any) -> None:
    try:
        # `set(..., ex=)` rather than `setex` — the latter is deprecated in
        # redis-py and raises a DeprecationWarning the test suite treats as an
        # error.
        await get_redis().set(key, json.dumps(value, default=str), ex=CACHE_TTL)
    except (RedisError, OSError) as exc:
        logger.warning("catalog.cache_write_failed", key=key, error=str(exc))


async def invalidate() -> int:
    """Drop every catalog key.

    Called after any write. Scanning a namespace is cheap at this key count and
    saves tracking which of the dozen filter permutations a given row appears
    in — a bookkeeping job that is wrong the moment a new filter is added.
    """
    redis = get_redis()
    removed = 0
    try:
        async for key in redis.scan_iter(match=f"{CACHE_PREFIX}:*", count=500):
            removed += await redis.delete(key)
    except (RedisError, OSError) as exc:
        logger.warning("catalog.cache_invalidate_failed", error=str(exc))
        return 0
    logger.info("catalog.cache_invalidated", keys=removed)
    return removed


def provenance_for(
    last_verified_at: datetime, source: DataSource, *, now: datetime | None = None
) -> ProvenanceOut:
    age = max(0, ((now or utcnow()) - last_verified_at).days)
    if age <= FRESH_DAYS:
        variant = "fresh"
    elif age <= AGING_DAYS:
        variant = "aging"
    else:
        variant = "stale"
    return ProvenanceOut(
        last_verified_at=last_verified_at,
        age_days=age,
        variant=variant,
        source_name=source.name,
        source_url=source.url,
        source_kind=source.kind.value,
    )


def _model_out(row: ModelPricing, source: DataSource, now: datetime) -> ModelOut:
    return ModelOut(
        id=row.id,
        provider=row.provider,
        model_id=row.model_id,
        display_name=row.display_name,
        family=row.family.value,
        input_cost_per_1k=row.input_cost_per_1k,
        output_cost_per_1k=row.output_cost_per_1k,
        cached_input_cost_per_1k=row.cached_input_cost_per_1k,
        context_window=row.context_window,
        max_output_tokens=row.max_output_tokens,
        dimensions=row.dimensions,
        capabilities=row.capabilities,
        tokenizer=row.tokenizer,
        status=row.status.value,
        status_reason=row.status_reason,
        provenance=provenance_for(row.last_verified_at, source, now=now),
    )


def _gpu_out(row: GpuPricing, source: DataSource, now: datetime) -> GpuOut:
    return GpuOut(
        id=row.id,
        slug=row.slug,
        provider=row.provider,
        instance_name=row.instance_name,
        gpu_model=row.gpu_model,
        gpu_count=row.gpu_count,
        vram_gb=row.vram_gb,
        vram_total_gb=row.vram_total_gb,
        vcpu=row.vcpu,
        ram_gb=row.ram_gb,
        hourly_cost_usd=row.hourly_cost_usd,
        monthly_cost_usd=(row.hourly_cost_usd * HOURS_PER_MONTH).quantize(Decimal("0.01")),
        region=row.region,
        spot=row.spot,
        provenance=provenance_for(row.last_verified_at, source, now=now),
    )


async def list_models(
    db: AsyncSession,
    *,
    family: str | None = None,
    provider: str | None = None,
    status: str | None = "active",
    include_all_statuses: bool = False,
) -> list[ModelOut]:
    key = _cache_key("models", family, provider, status, include_all_statuses)
    if (hit := await _cached(key)) is not None:
        return [ModelOut.model_validate(item) for item in hit]

    stmt: Select[Any] = (
        select(ModelPricing, DataSource)
        .join(DataSource, DataSource.id == ModelPricing.source_id)
        .order_by(ModelPricing.provider, ModelPricing.input_cost_per_1k)
    )
    if family:
        stmt = stmt.where(ModelPricing.family == family)
    if provider:
        stmt = stmt.where(ModelPricing.provider == provider)
    if not include_all_statuses:
        stmt = stmt.where(ModelPricing.status == (status or LifecycleStatus.ACTIVE.value))

    now = utcnow()
    rows = (await db.execute(stmt)).all()
    result = [_model_out(model, source, now) for model, source in rows]
    await _store(key, [item.model_dump(mode="json") for item in result])
    return result


async def get_model(db: AsyncSession, model_id: str) -> ModelOut:
    """Lookup by canonical `model_id` or by row id.

    Accepting both means a caller holding `"gpt-4o-mini"` from a spec and a
    caller holding `"mdl_…"` from a saved run both work, which removes a class
    of "which id is this?" bug from every consumer.
    """
    stmt = (
        select(ModelPricing, DataSource)
        .join(DataSource, DataSource.id == ModelPricing.source_id)
        .where((ModelPricing.model_id == model_id) | (ModelPricing.id == model_id))
        .limit(1)
    )
    row = (await db.execute(stmt)).first()
    if row is None:
        raise NotFound(f"No pricing for model '{model_id}'.")
    return _model_out(row[0], row[1], utcnow())


async def get_models_by_ids(db: AsyncSession, model_ids: Sequence[str]) -> dict[str, ModelOut]:
    """Batch lookup, keyed by canonical `model_id`.

    Comparison and budget tools need several models at once; issuing one query
    per model turns a four-model comparison into four round trips.
    """
    if not model_ids:
        return {}
    stmt = (
        select(ModelPricing, DataSource)
        .join(DataSource, DataSource.id == ModelPricing.source_id)
        .where(ModelPricing.model_id.in_(list(model_ids)))
    )
    now = utcnow()
    rows = (await db.execute(stmt)).all()
    return {model.model_id: _model_out(model, source, now) for model, source in rows}


async def list_gpus(
    db: AsyncSession,
    *,
    provider: str | None = None,
    min_vram: int | None = None,
    region: str | None = None,
    spot: bool | None = None,
) -> list[GpuOut]:
    key = _cache_key("gpus", provider, min_vram, region, spot)
    if (hit := await _cached(key)) is not None:
        return [GpuOut.model_validate(item) for item in hit]

    stmt = (
        select(GpuPricing, DataSource)
        .join(DataSource, DataSource.id == GpuPricing.source_id)
        .order_by(GpuPricing.hourly_cost_usd)
    )
    if provider:
        stmt = stmt.where(GpuPricing.provider == provider)
    if min_vram is not None:
        # Total VRAM, not per-card: an 8x80GB node satisfies a 200GB
        # requirement even though no single card does.
        stmt = stmt.where(GpuPricing.vram_gb * GpuPricing.gpu_count >= min_vram)
    if region:
        stmt = stmt.where(GpuPricing.region == region)
    if spot is not None:
        stmt = stmt.where(GpuPricing.spot.is_(spot))

    now = utcnow()
    rows = (await db.execute(stmt)).all()
    result = [_gpu_out(gpu, source, now) for gpu, source in rows]
    await _store(key, [item.model_dump(mode="json") for item in result])
    return result


async def list_tools(
    db: AsyncSession,
    *,
    category: str | None = None,
    status: str | None = None,
    use_case: str | None = None,
    tags: Sequence[str] | None = None,
) -> list[ToolOut]:
    tag_key = ",".join(sorted(tags)) if tags else None
    key = _cache_key("tools", category, status, use_case, tag_key)
    if (hit := await _cached(key)) is not None:
        return [ToolOut.model_validate(item) for item in hit]

    stmt = select(Tool).order_by(Tool.category, Tool.maturity_score.desc(), Tool.name)
    if category:
        stmt = stmt.where(Tool.category == category)
    if status:
        stmt = stmt.where(Tool.status == status)
    if use_case:
        stmt = stmt.where(Tool.use_cases.contains([use_case]))
    if tags:
        # Every tag must match — an OR here would make adding a second tag
        # widen the result set, which is the opposite of what a filter means.
        stmt = stmt.where(Tool.tags.contains(list(tags)))

    rows = (await db.execute(stmt)).scalars().all()
    result = [ToolOut.model_validate(row) for row in rows]
    await _store(key, [item.model_dump(mode="json") for item in result])
    return result


async def get_tool(db: AsyncSession, slug: str) -> ToolOut:
    row = (await db.execute(select(Tool).where(Tool.slug == slug))).scalar_one_or_none()
    if row is None:
        raise NotFound(f"No tool with slug '{slug}'.")
    return ToolOut.model_validate(row)


async def get_graveyard(db: AsyncSession) -> list[GraveyardEntryOut]:
    """Everything buried, with its reason and its replacements."""
    key = _cache_key("graveyard")
    if (hit := await _cached(key)) is not None:
        return [GraveyardEntryOut.model_validate(item) for item in hit]

    buried = (
        (
            await db.execute(
                select(Tool)
                .where(Tool.status.in_([ToolStatus.DEPRECATED, ToolStatus.NOT_FOR_PRODUCTION]))
                .order_by(Tool.status, Tool.name)
            )
        )
        .scalars()
        .all()
    )

    wanted = {slug for tool in buried for slug in tool.alternatives}
    replacements: dict[str, ToolOut] = {}
    if wanted:
        rows = (await db.execute(select(Tool).where(Tool.slug.in_(list(wanted))))).scalars().all()
        replacements = {row.slug: ToolOut.model_validate(row) for row in rows}

    result = []
    for tool in buried:
        base = ToolOut.model_validate(tool).model_dump()
        # Guaranteed non-null by the seeder's validation; the fallback keeps a
        # hand-edited row from 500ing the whole page.
        base["status_reason"] = tool.status_reason or "Marked unsuitable for production use."
        result.append(
            GraveyardEntryOut(
                **base,
                alternative_tools=[
                    replacements[slug] for slug in tool.alternatives if slug in replacements
                ],
            )
        )
    await _store(key, [item.model_dump(mode="json") for item in result])
    return result


async def get_compatibility(db: AsyncSession, tool_slugs: Sequence[str]) -> CompatibilityOut:
    """Pairwise scores for an arbitrary tool set.

    Order-independent by construction: every pair is normalised to
    `(min, max)` before the lookup, which is the same ordering the check
    constraint enforces on the way in.
    """
    slugs = sorted({slug.strip() for slug in tool_slugs if slug.strip()})
    if len(slugs) < 2:
        raise NotFound("Compatibility needs at least two tools.")

    key = _cache_key("compat", ",".join(slugs))
    if (hit := await _cached(key)) is not None:
        return CompatibilityOut.model_validate(hit)

    rows = (
        (
            await db.execute(
                select(Compatibility).where(
                    Compatibility.tool_a_slug.in_(slugs),
                    Compatibility.tool_b_slug.in_(slugs),
                )
            )
        )
        .scalars()
        .all()
    )

    found = {
        (row.tool_a_slug, row.tool_b_slug): CompatibilityPairOut(
            tool_a=row.tool_a_slug,
            tool_b=row.tool_b_slug,
            score=row.score,
            dimensions=row.dimensions,
            notes=row.notes,
            warnings=list(row.warnings),
        )
        for row in rows
    }

    pairs: list[CompatibilityPairOut] = []
    missing: list[list[str]] = []
    for index, a in enumerate(slugs):
        for b in slugs[index + 1 :]:
            pair = found.get((a, b))
            if pair is None:
                # Reported, never assumed. An unscored pair silently treated as
                # 100 would let the Stack Architect recommend a combination
                # nobody has ever looked at.
                missing.append([a, b])
            else:
                pairs.append(pair)

    pairs.sort(key=lambda p: p.score)
    weakest = pairs[0] if pairs else None
    result = CompatibilityOut(
        tools=slugs,
        pairs=pairs,
        overall=weakest.score if weakest else 0,
        weakest_pair=weakest,
        warnings=sorted({w for pair in pairs for w in pair.warnings}),
        missing_pairs=missing,
    )
    await _store(key, result.model_dump(mode="json"))
    return result


async def count_rows(db: AsyncSession) -> dict[str, int]:
    async def _count(model: type[Any]) -> int:
        return int((await db.execute(select(func.count()).select_from(model))).scalar_one())

    return {
        "models": await _count(ModelPricing),
        "gpus": await _count(GpuPricing),
        "tools": await _count(Tool),
        "compatibility_pairs": await _count(Compatibility),
    }
