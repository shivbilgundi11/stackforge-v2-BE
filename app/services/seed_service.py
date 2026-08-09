"""Loading the catalog seed.

Idempotent, and **non-destructive by default**. Running the seeder twice does
not duplicate rows, and — critically — it does not overwrite a row an editor
has corrected. The whole point of `POST /catalog/flag` and the editorial review
loop is that humans fix numbers; a seeder that reverts those fixes on the next
deploy makes the review process pointless.

The rule is: insert what is missing, leave what exists. `--refresh` opts into
overwriting, for the case where the seed file itself is the correction.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import UTC, date, datetime
from decimal import Decimal

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.database import new_id, utcnow
from app.core.logging import get_logger
from app.data.compatibility_seed import build_pairs
from app.data.gpus_seed import GPUS
from app.data.gpus_seed import VERIFIED as GPU_VERIFIED
from app.data.models_seed import MODELS
from app.data.sources import SOURCES
from app.data.tools_seed import REVIEWED as TOOLS_REVIEWED
from app.data.tools_seed import TOOLS
from app.models.catalog import (
    Compatibility,
    DataSource,
    GpuPricing,
    ModelPricing,
    Tool,
)
from app.services import catalog_service

logger = get_logger("seed")

PER_MILLION = Decimal(1000)


@dataclass
class SeedReport:
    inserted: dict[str, int] = field(default_factory=dict)
    updated: dict[str, int] = field(default_factory=dict)
    skipped: dict[str, int] = field(default_factory=dict)

    @property
    def total_inserted(self) -> int:
        return sum(self.inserted.values())

    @property
    def total_updated(self) -> int:
        return sum(self.updated.values())

    def note(self, table: str, inserted: int, updated: int, skipped: int) -> None:
        self.inserted[table] = inserted
        self.updated[table] = updated
        self.skipped[table] = skipped


def _at_midnight(day: date) -> datetime:
    return datetime(day.year, day.month, day.day, tzinfo=UTC)


def _per_1k(per_million: str | None) -> Decimal | None:
    """Providers publish per 1M; the catalog stores per 1k.

    Done once here rather than at every read, and via `Decimal` so
    `0.05 / 1000` is exactly `0.00005` rather than a float approximation of it.
    """
    if per_million is None:
        return None
    return (Decimal(per_million) / PER_MILLION).quantize(Decimal("0.000001"))


async def seed_all(db: AsyncSession, *, refresh: bool = False) -> SeedReport:
    report = SeedReport()
    sources = await _seed_sources(db, report)
    await _seed_models(db, sources, report, refresh=refresh)
    await _seed_gpus(db, sources, report, refresh=refresh)
    await _seed_tools(db, report, refresh=refresh)
    await db.flush()
    await _seed_compatibility(db, report, refresh=refresh)

    logger.info(
        "seed.complete",
        inserted=report.total_inserted,
        updated=report.total_updated,
        refresh=refresh,
    )
    if report.total_inserted or report.total_updated:
        await catalog_service.invalidate()
    return report


async def _seed_sources(db: AsyncSession, report: SeedReport) -> dict[str, DataSource]:
    existing = {row.slug: row for row in (await db.execute(select(DataSource))).scalars().all()}
    inserted = 0
    for seed in SOURCES:
        if seed.slug in existing:
            continue
        row = DataSource(
            id=new_id("src"),
            slug=seed.slug,
            name=seed.name,
            url=seed.url,
            kind=seed.kind,
        )
        db.add(row)
        existing[seed.slug] = row
        inserted += 1
    await db.flush()
    report.note("data_sources", inserted, 0, len(SOURCES) - inserted)
    return existing


async def _seed_models(
    db: AsyncSession,
    sources: dict[str, DataSource],
    report: SeedReport,
    *,
    refresh: bool,
) -> None:
    existing = {
        (row.provider, row.model_id): row
        for row in (await db.execute(select(ModelPricing))).scalars().all()
    }
    inserted = updated = skipped = 0

    for seed in MODELS:
        source = sources[seed.source]
        current = existing.get((seed.provider, seed.model_id))
        values = {
            "display_name": seed.display_name,
            "family": seed.family,
            "input_cost_per_1k": _per_1k(seed.input_per_m),
            "output_cost_per_1k": _per_1k(seed.output_per_m),
            "cached_input_cost_per_1k": _per_1k(seed.cached_input_per_m),
            "context_window": seed.context_window,
            "max_output_tokens": seed.max_output_tokens,
            "dimensions": seed.dimensions,
            "capabilities": dict(seed.capabilities),
            "tokenizer": seed.tokenizer,
            "status": seed.status,
            "status_reason": seed.status_reason,
            "source_id": source.id,
            "last_verified_at": _at_midnight(seed.verified),
        }

        if current is None:
            db.add(
                ModelPricing(
                    id=new_id("mdl"),
                    provider=seed.provider,
                    model_id=seed.model_id,
                    **values,
                )
            )
            inserted += 1
        elif refresh:
            for key, value in values.items():
                setattr(current, key, value)
            updated += 1
        else:
            skipped += 1

    report.note("model_pricing", inserted, updated, skipped)


async def _seed_gpus(
    db: AsyncSession,
    sources: dict[str, DataSource],
    report: SeedReport,
    *,
    refresh: bool,
) -> None:
    existing = {
        (row.provider, row.instance_name, row.region, row.spot): row
        for row in (await db.execute(select(GpuPricing))).scalars().all()
    }
    inserted = updated = skipped = 0
    verified = _at_midnight(GPU_VERIFIED)

    for seed in GPUS:
        source = sources[seed.source]
        key = (seed.provider, seed.instance_name, seed.region, seed.spot)
        current = existing.get(key)
        values = {
            "gpu_model": seed.gpu_model,
            "gpu_count": seed.gpu_count,
            "vram_gb": seed.vram_gb,
            "vcpu": seed.vcpu,
            "ram_gb": seed.ram_gb,
            "hourly_cost_usd": Decimal(seed.hourly),
            "source_id": source.id,
            "last_verified_at": verified,
        }

        if current is None:
            db.add(
                GpuPricing(
                    id=new_id("gpu"),
                    provider=seed.provider,
                    instance_name=seed.instance_name,
                    region=seed.region,
                    spot=seed.spot,
                    **values,
                )
            )
            inserted += 1
        elif refresh:
            for attr, value in values.items():
                setattr(current, attr, value)
            updated += 1
        else:
            skipped += 1

    report.note("gpu_pricing", inserted, updated, skipped)


async def _seed_tools(db: AsyncSession, report: SeedReport, *, refresh: bool) -> None:
    existing = {row.slug: row for row in (await db.execute(select(Tool))).scalars().all()}
    inserted = updated = skipped = 0
    reviewed = _at_midnight(TOOLS_REVIEWED)

    for seed in TOOLS:
        # A buried tool with no reason renders an empty Graveyard row, which
        # reads as a bug. Catch it at seed time rather than in production.
        if seed.status in ("deprecated", "not_for_production") and not seed.status_reason:
            raise ValueError(f"Tool '{seed.slug}' is buried but has no status_reason.")

        current = existing.get(seed.slug)
        values = {
            "name": seed.name,
            "category": seed.category,
            "description": seed.description,
            "status": seed.status,
            "status_reason": seed.status_reason,
            "alternatives": list(seed.alternatives),
            "maturity_score": seed.maturity,
            "license": seed.license,
            "self_hostable": seed.self_hostable,
            "pricing_model": seed.pricing_model,
            "docs_url": seed.docs_url,
            "tags": list(seed.tags),
            "use_cases": list(seed.use_cases),
            "facts": dict(seed.facts),
            "last_reviewed_at": reviewed,
            "reviewed_by": "editorial",
        }

        if current is None:
            db.add(Tool(id=new_id("tool"), slug=seed.slug, **values))
            inserted += 1
        elif refresh:
            for attr, value in values.items():
                setattr(current, attr, value)
            updated += 1
        else:
            skipped += 1

    report.note("tool_catalog", inserted, updated, skipped)


async def _seed_compatibility(db: AsyncSession, report: SeedReport, *, refresh: bool) -> None:
    existing = {
        (row.tool_a_slug, row.tool_b_slug): row
        for row in (await db.execute(select(Compatibility))).scalars().all()
    }
    known = {slug for (slug,) in (await db.execute(select(Tool.slug))).all()}

    inserted = updated = skipped = 0
    reviewed = _at_midnight(TOOLS_REVIEWED)

    for pair in build_pairs():
        if pair.tool_a not in known or pair.tool_b not in known:
            continue
        current = existing.get((pair.tool_a, pair.tool_b))
        values = {
            "score": pair.score,
            "dimensions": dict(pair.dimensions),
            "notes": pair.notes,
            "warnings": list(pair.warnings),
            "last_reviewed_at": reviewed,
        }

        if current is None:
            db.add(
                Compatibility(
                    id=new_id("cmp"),
                    tool_a_slug=pair.tool_a,
                    tool_b_slug=pair.tool_b,
                    **values,
                )
            )
            inserted += 1
        elif refresh:
            for attr, value in values.items():
                setattr(current, attr, value)
            updated += 1
        else:
            skipped += 1

    report.note("compatibility_matrix", inserted, updated, skipped)


async def seed_if_empty(db: AsyncSession) -> SeedReport | None:
    """Startup hook.

    Only fires on a genuinely empty catalog, so a running deployment is never
    surprised by a seed pass it did not ask for.
    """
    count = (await db.execute(select(Tool.slug).limit(1))).first()
    if count is not None:
        return None
    logger.info("seed.empty_catalog_detected")
    return await seed_all(db)


__all__ = ["SeedReport", "seed_all", "seed_if_empty", "utcnow"]
