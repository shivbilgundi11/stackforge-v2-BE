"""Price staleness reporting, and the editorial accept path.

**No fetchers ship, and none are planned.** Prices are hardcoded in
`app/data/` and verified by a human reading the provider's published page
(D-16). Automated scraping was considered and rejected: a scraper misreading a
pricing page silently corrupts every downstream estimate, and this product's
entire claim is that its numbers are right. Reading a page and editing a seed
file costs minutes and cannot fail silently.

What survives here, and why it is still worth having:

  * `apply_change` — the editorial accept: takes a recorded change and writes
    it to the row. The one place a `pricing_history` entry becomes a price.
  * `verify_all` — walks the sources and reports. With no fetchers registered
    every source is *skipped*, which is the honest outcome: it tells an
    operator how many sources exist and that none is machine-readable, rather
    than reporting a clean run over zero work.

The fetcher registry is kept because the shape is right if this decision is
ever revisited, and because the tests pin the contract that a fetcher must
never mutate a price — only record what it saw.
"""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from datetime import datetime
from decimal import Decimal

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.database import utcnow
from app.core.logging import get_logger
from app.models.catalog import (
    DataSource,
    GpuPricing,
    ModelPricing,
    PricedEntity,
    PricingHistory,
)
from app.services import provenance_service

logger = get_logger("worker.pricing")

FAILURE_ALERT_THRESHOLD = 3

# A fetcher answers: "for this source, what does the provider publish today?"
# Keyed `(provider, model_id)` -> {field: price per 1k}.
Fetcher = Callable[[DataSource], Awaitable[dict[tuple[str, str], dict[str, Decimal]]]]

_FETCHERS: dict[str, Fetcher] = {}


def register_fetcher(source_slug: str, fetcher: Fetcher) -> None:
    _FETCHERS[source_slug] = fetcher


def registered_sources() -> frozenset[str]:
    return frozenset(_FETCHERS)


def clear_fetchers() -> None:
    """Test seam."""
    _FETCHERS.clear()


@dataclass
class VerificationResult:
    checked: int = 0
    changes_detected: int = 0
    sources_failed: int = 0
    sources_skipped: int = 0
    alerts: list[str] = field(default_factory=list)
    ran_at: datetime = field(default_factory=utcnow)


# Fields compared, in the order they are reported.
PRICE_FIELDS = ("input_cost_per_1k", "output_cost_per_1k", "cached_input_cost_per_1k")


async def verify_all(db: AsyncSession, *, source_slug: str | None = None) -> VerificationResult:
    result = VerificationResult()
    stmt = select(DataSource)
    if source_slug:
        stmt = stmt.where(DataSource.slug == source_slug)
    sources = (await db.execute(stmt)).scalars().all()

    for source in sources:
        fetcher = _FETCHERS.get(source.slug)
        if fetcher is None:
            result.sources_skipped += 1
            continue
        await _verify_source(db, source, fetcher, result)

    logger.info(
        "pricing.verification_complete",
        checked=result.checked,
        changes=result.changes_detected,
        failed=result.sources_failed,
        skipped=result.sources_skipped,
    )
    return result


async def _verify_source(
    db: AsyncSession,
    source: DataSource,
    fetcher: Fetcher,
    result: VerificationResult,
) -> None:
    try:
        published = await fetcher(source)
    except Exception as exc:
        await _record_failure(db, source, result, exc)
        return

    rows = (
        (await db.execute(select(ModelPricing).where(ModelPricing.source_id == source.id)))
        .scalars()
        .all()
    )
    now = utcnow()

    for row in rows:
        current = published.get((row.provider, row.model_id))
        if current is None:
            continue
        result.checked += 1

        for field_name in PRICE_FIELDS:
            if field_name not in current:
                continue
            stored: Decimal | None = getattr(row, field_name)
            observed = current[field_name]
            if stored is not None and stored == observed:
                continue

            await provenance_service.record_change(
                db,
                entity_type=PricedEntity.MODEL,
                entity_id=row.id,
                field=field_name,
                old_value=stored,
                new_value=observed,
                source_id=source.id,
                applied=False,
                detected_at=now,
            )
            result.changes_detected += 1
            result.alerts.append(
                f"{row.provider}/{row.model_id} {field_name}: "
                f"{stored} → {observed} (source: {source.name})"
            )

        # A row confirmed unchanged is a row verified today. This is the only
        # write the job makes to a priced row, and it moves the date, never
        # the price.
        if not any(
            field_name in current
            and getattr(row, field_name) is not None
            and getattr(row, field_name) != current[field_name]
            for field_name in PRICE_FIELDS
        ):
            row.last_verified_at = now

    source.last_success_at = now
    source.failure_count = 0


async def _record_failure(
    db: AsyncSession,
    source: DataSource,
    result: VerificationResult,
    exc: Exception,
) -> None:
    """A failed fetch changes nothing but the counter.

    The stored price stays exactly as it was — a source being unreachable is
    not evidence that its price changed.
    """
    source.last_failure_at = utcnow()
    source.failure_count += 1
    result.sources_failed += 1

    logger.warning(
        "pricing.source_fetch_failed",
        source=source.slug,
        failure_count=source.failure_count,
        error=str(exc),
    )
    if source.failure_count >= FAILURE_ALERT_THRESHOLD:
        message = (
            f"{source.name} has failed {source.failure_count} consecutive "
            f"verification runs — the page may have moved or changed format."
        )
        result.alerts.append(message)
        logger.error("pricing.source_alert", source=source.slug, message=message)


async def apply_change(db: AsyncSession, history_id: str) -> ModelPricing | GpuPricing | None:
    """Editorial accept: take a recorded change and write it to the row.

    The only path by which an observed price reaches the catalog, and it is
    reached by a person clicking accept.
    """
    history = await db.get(PricingHistory, history_id)
    if history is None or history.applied or history.new_value is None:
        return None

    if history.entity_type is PricedEntity.MODEL:
        row: ModelPricing | GpuPricing | None = await db.get(ModelPricing, history.entity_id)
    else:
        row = await db.get(GpuPricing, history.entity_id)
    if row is None:
        return None

    setattr(row, history.field, history.new_value)
    row.last_verified_at = utcnow()
    history.applied = True

    logger.info(
        "pricing.change_applied",
        entity_id=history.entity_id,
        field=history.field,
        new_value=str(history.new_value),
    )
    return row
