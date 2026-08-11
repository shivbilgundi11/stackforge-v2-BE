"""Loading the catalog seed.

Idempotent, and **non-destructive by default**. Running the seeder twice does
not duplicate rows, and — critically — it does not overwrite a row an editor
has corrected. The whole point of `POST /catalog/flag` and the editorial review
loop is that humans fix numbers; a seeder that reverts those fixes on the next
deploy makes the review process pointless.

The rule is: insert what is missing, leave what exists. `--refresh` opts into
overwriting, for the case where the seed file itself is the correction.

**These files are the only way a price ever changes.** Prices are hardcoded and
verified by hand against the provider's published page — no pricing APIs, no
scrapers (D-16). That makes `--refresh` the single write path for a price, so
it records every movement it makes into `pricing_history`: an edit here is an
editorial decision, and it leaves the same audit trail as any other.
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
from app.data.models_seed import MODELS
from app.data.sources import SOURCES
from app.data.tools_seed import REVIEWED as TOOLS_REVIEWED
from app.data.tools_seed import TOOLS
from app.models.catalog import (
    Compatibility,
    DataSource,
    GpuPricing,
    ModelPricing,
    PricedEntity,
    Tool,
)
from app.services import catalog_service, provenance_service

logger = get_logger("seed")

PER_MILLION = Decimal(1000)


@dataclass
class SeedReport:
    inserted: dict[str, int] = field(default_factory=dict)
    updated: dict[str, int] = field(default_factory=dict)
    skipped: dict[str, int] = field(default_factory=dict)
    #: Rows in the database that the seed no longer describes.
    unmanaged: list[str] = field(default_factory=list)
    #: Price movements this refresh wrote to `pricing_history`.
    price_changes: int = 0

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


async def _record_price_changes(
    db: AsyncSession,
    *,
    entity_type: PricedEntity,
    entity_id: str,
    source_id: str,
    changes: dict[str, tuple[Decimal | None, Decimal | None]],
    report: SeedReport,
) -> None:
    """Write a `pricing_history` row for every price this refresh moves.

    Prices are hardcoded and verified by hand, which makes this file the only
    path by which a price ever changes. A `setattr` that silently drops the old
    value would leave that path as the one part of the system with no audit
    trail — no way to answer "when did Mistral Large get cheaper, and by how
    much", which is exactly the question the history table was added for.

    Marked `applied=True`, unlike the drift rows: this is a change that has
    already been made, not one awaiting review.
    """
    for field_name, (old, new) in changes.items():
        if old == new:
            continue
        await provenance_service.record_change(
            db,
            entity_type=entity_type,
            entity_id=entity_id,
            field=field_name,
            old_value=old,
            new_value=new,
            source_id=source_id,
            applied=True,
        )
        report.price_changes += 1


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
    await _seed_templates(db, report, refresh=refresh)
    await seed_plan_quotas(db, report, refresh=refresh)
    await _find_unmanaged(db, report)

    logger.info(
        "seed.complete",
        inserted=report.total_inserted,
        updated=report.total_updated,
        price_changes=report.price_changes,
        unmanaged=len(report.unmanaged),
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
        # Kept as typed locals as well as dict entries: the change record needs
        # `Decimal | None`, and the values dict is heterogeneous enough that
        # reading them back out of it loses the type.
        prices: dict[str, Decimal | None] = {
            "input_cost_per_1k": _per_1k(seed.input_per_m),
            "output_cost_per_1k": _per_1k(seed.output_per_m),
            "cached_input_cost_per_1k": _per_1k(seed.cached_input_per_m),
        }
        values = {
            "display_name": seed.display_name,
            "family": seed.family,
            **prices,
            "context_window": seed.context_window,
            "max_output_tokens": seed.max_output_tokens,
            "dimensions": seed.dimensions,
            "capabilities": dict(seed.capabilities),
            "tokenizer": seed.tokenizer,
            "status": seed.status,
            "status_reason": seed.status_reason,
            "price_unit": seed.price_unit,
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
            await _record_price_changes(
                db,
                entity_type=PricedEntity.MODEL,
                entity_id=current.id,
                source_id=source.id,
                changes={name: (getattr(current, name), price) for name, price in prices.items()},
                report=report,
            )
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

    for seed in GPUS:
        source = sources[seed.source]
        key = (seed.provider, seed.instance_name, seed.region, seed.spot)
        current = existing.get(key)
        hourly = Decimal(seed.hourly)
        values = {
            "gpu_model": seed.gpu_model,
            "gpu_count": seed.gpu_count,
            "vram_gb": seed.vram_gb,
            "vcpu": seed.vcpu,
            "ram_gb": seed.ram_gb,
            "hourly_cost_usd": hourly,
            "source_id": source.id,
            # Per row, not per file: auction-cleared rows keep an older date
            # than list-price rows because nobody can verify them the same
            # way. A single date for the file would quietly promote a
            # marketplace guess to the standing of a published rate card.
            "last_verified_at": _at_midnight(seed.verified),
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
            await _record_price_changes(
                db,
                entity_type=PricedEntity.GPU,
                entity_id=current.id,
                source_id=source.id,
                changes={"hourly_cost_usd": (current.hourly_cost_usd, hourly)},
                report=report,
            )
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


async def _seed_templates(db: AsyncSession, report: SeedReport, *, refresh: bool) -> None:
    """Load `app/data/templates/` into the table (M19).

    Templates break the non-destructive default that governs the rest of this
    module, and the difference is worth stating. A price is corrected by an
    *editor* through the flag-and-review loop, so a seeder that overwrote it
    would undo human work — that is why everything above is insert-only unless
    `--refresh` is passed.

    A template has no such loop. The Markdown file **is** the source of truth;
    there is no admin UI and nothing else writes `content_markdown`. So a
    changed file always wins, and editing a typo is a commit plus a seed run
    rather than a commit plus a flag passed to a command nobody remembers.

    The two columns that are *not* overwritten are `view_count` and
    `copy_count`. Those are measurements, not content, and resetting them on
    every deploy would destroy the only reliable input to the content roadmap.
    """
    from app.data.templates_loader import load_all
    from app.models.template import Difficulty, Template, TemplateCategory

    seeds = load_all()
    existing = {row.slug: row for row in (await db.execute(select(Template))).scalars().all()}

    inserted = updated = skipped = 0
    published = utcnow()

    for seed in seeds:
        current = existing.get(seed.slug)
        values = {
            "title": seed.title,
            "category": TemplateCategory(seed.category),
            "difficulty": Difficulty(seed.difficulty),
            "summary": seed.summary.strip(),
            "content_markdown": seed.content_markdown,
            "files": [file.as_dict() for file in seed.files],
            "stack_input": seed.stack_input,
            "use_cases": seed.use_cases,
            "tags": seed.tags,
            "related_tools": seed.related_tools,
            "is_premium": seed.is_premium,
        }

        if current is None:
            db.add(Template(id=new_id("tpl"), slug=seed.slug, published_at=published, **values))
            inserted += 1
            continue

        # Compared before writing so the report distinguishes "three templates
        # changed" from "thirty templates were touched", which is the only way
        # a seed run tells an operator anything.
        changed = any(getattr(current, attr) != value for attr, value in values.items())
        if not changed:
            skipped += 1
            continue

        for attr, value in values.items():
            setattr(current, attr, value)
        updated += 1

    report.note("templates", inserted, updated, skipped)

    # Deliberately not deleted. A template file removed in a branch that is
    # later reverted would otherwise take its view and copy counts with it, and
    # `_find_unmanaged` already reports the row so the removal is visible.
    _ = refresh


async def seed_plan_quotas(db: AsyncSession, report: SeedReport, *, refresh: bool) -> None:
    """Load the default limits into `plan_quotas` (M20).

    Insert-only, and this one is not a default that `--refresh` should casually
    override — it is the point of the table. M20's rule is that every limit is
    changeable without a deploy, so an operator who raises the free tier to 10
    runs must not have that reverted by the next release's seed run. `--refresh`
    still overwrites, because the case where the seed file *is* the correction
    exists here as it does for prices, but it is a deliberate act.

    A metric added to the enum after this table was first seeded arrives as a
    new row on the next run, which is why this iterates the metric map rather
    than skipping a plan once it has any rows at all.
    """
    from app.data.plans import DEFAULT_QUOTAS
    from app.models.billing import PlanQuota

    existing = {
        (row.plan, row.anonymous, row.metric): row
        for row in (await db.execute(select(PlanQuota))).scalars().all()
    }

    inserted = updated = skipped = 0
    for (plan, anonymous), limits in DEFAULT_QUOTAS.items():
        for metric, limit in limits.items():
            current = existing.get((plan, anonymous, metric))
            if current is None:
                db.add(
                    PlanQuota(
                        id=new_id("pq"),
                        plan=plan,
                        anonymous=anonymous,
                        metric=metric,
                        limit_value=limit,
                    )
                )
                inserted += 1
            elif refresh and current.limit_value != limit:
                current.limit_value = limit
                updated += 1
            else:
                skipped += 1

    await db.flush()
    report.note("plan_quotas", inserted, updated, skipped)


async def _find_unmanaged(db: AsyncSession, report: SeedReport) -> None:
    """Rows the seed no longer describes.

    Renaming a `model_id` — which happens every time a provider version-stamps
    its API ids — inserts the new row and silently orphans the old one. The
    orphan then sits there at its last verification date, ages into looking
    stale, and nothing ever updates it because no seed entry claims it.

    These are reported, never deleted. A row could be orphaned because a
    provider retired it, or because someone fat-fingered an id in the seed
    file, and deleting priced history on that ambiguity is not a trade worth
    making automatically.
    """
    seeded_models = {(seed.provider, seed.model_id) for seed in MODELS}
    for row in (await db.execute(select(ModelPricing))).scalars().all():
        if (row.provider, row.model_id) not in seeded_models:
            report.unmanaged.append(f"model_pricing: {row.provider}/{row.model_id}")

    seeded_gpus = {(seed.provider, seed.instance_name, seed.region, seed.spot) for seed in GPUS}
    for gpu in (await db.execute(select(GpuPricing))).scalars().all():
        if (gpu.provider, gpu.instance_name, gpu.region, gpu.spot) not in seeded_gpus:
            report.unmanaged.append(f"gpu_pricing: {gpu.provider}/{gpu.instance_name}")

    seeded_tools = {seed.slug for seed in TOOLS}
    for tool in (await db.execute(select(Tool))).scalars().all():
        if tool.slug not in seeded_tools:
            report.unmanaged.append(f"tool_catalog: {tool.slug}")

    if report.unmanaged:
        logger.warning("seed.unmanaged_rows", rows=report.unmanaged)


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
