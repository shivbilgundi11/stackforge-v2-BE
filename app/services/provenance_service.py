"""Provenance, change history, and drift detection.

The rule this module exists to enforce: **the system records price changes, it
does not apply them.** A scraper that misreads a pricing page and writes the
result straight into `model_pricing` silently corrupts every downstream
estimate, and nobody finds out until a user disputes an invoice. Detecting the
change and raising it for a human costs minutes of review and makes that
failure mode impossible.

(Q-05 in the decision log. The alternative — scheduled scraping that auto-
applies — was considered and rejected on exactly this trade.)
"""

from __future__ import annotations

from datetime import datetime, timedelta
from decimal import Decimal
from typing import Literal

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.database import new_id, utcnow
from app.core.logging import get_logger
from app.models.catalog import (
    DataSource,
    GpuPricing,
    ModelPricing,
    PricedEntity,
    PricingHistory,
)
from app.schemas.catalog import DriftEntryOut

logger = get_logger("provenance")

FRESH_DAYS = 7
AGING_DAYS = 30

Variant = Literal["fresh", "aging", "stale"]


def verification_age(last_verified_at: datetime, *, now: datetime | None = None) -> int:
    return max(0, ((now or utcnow()) - last_verified_at).days)


def variant_for(age_days: int) -> Variant:
    if age_days <= FRESH_DAYS:
        return "fresh"
    if age_days <= AGING_DAYS:
        return "aging"
    return "stale"


def pct_change(old: Decimal, new: Decimal) -> Decimal:
    """Percentage change, guarding the zero case.

    A price moving off zero is an infinite percentage change. Reporting 100
    keeps it in the drift list and out of a `DivisionByZero` in a nightly job.
    """
    if old == 0:
        return Decimal(0) if new == 0 else Decimal(100)
    return ((new - old) / old * 100).quantize(Decimal("0.0001"))


async def record_change(
    db: AsyncSession,
    *,
    entity_type: PricedEntity,
    entity_id: str,
    field: str,
    old_value: Decimal | None,
    new_value: Decimal | None,
    source_id: str | None,
    applied: bool = False,
    detected_at: datetime | None = None,
) -> PricingHistory:
    change = (
        pct_change(old_value, new_value)
        if old_value is not None and new_value is not None
        else None
    )
    row = PricingHistory(
        id=new_id("ph"),
        entity_type=entity_type,
        entity_id=entity_id,
        field=field,
        old_value=old_value,
        new_value=new_value,
        pct_change=change,
        applied=applied,
        source_id=source_id,
        detected_at=detected_at or utcnow(),
    )
    db.add(row)
    logger.info(
        "pricing.change_recorded",
        entity_type=entity_type.value,
        entity_id=entity_id,
        field=field,
        pct_change=str(change) if change is not None else None,
        applied=applied,
    )
    return row


async def history_for(
    db: AsyncSession,
    *,
    entity_id: str | None = None,
    since: datetime | None = None,
    limit: int = 100,
) -> list[PricingHistory]:
    stmt = select(PricingHistory).order_by(PricingHistory.detected_at.desc()).limit(limit)
    if entity_id:
        stmt = stmt.where(PricingHistory.entity_id == entity_id)
    if since:
        stmt = stmt.where(PricingHistory.detected_at >= since)
    return list((await db.execute(stmt)).scalars().all())


async def detect_drift(
    db: AsyncSession,
    *,
    threshold_pct: Decimal = Decimal(5),
    since: datetime | None = None,
) -> list[DriftEntryOut]:
    """Unapplied changes above the threshold.

    Reads recorded history rather than re-fetching sources — the verification
    job does the fetching, this answers "what has it found that nobody has
    acted on yet".
    """
    window = since or utcnow() - timedelta(days=30)
    stmt = (
        select(PricingHistory, DataSource)
        .outerjoin(DataSource, DataSource.id == PricingHistory.source_id)
        .where(
            PricingHistory.detected_at >= window,
            PricingHistory.applied.is_(False),
            PricingHistory.pct_change.isnot(None),
        )
        .order_by(PricingHistory.detected_at.desc())
    )
    rows = (await db.execute(stmt)).all()

    labels = await _labels_for(db, [row.entity_id for row, _ in rows])
    entries: list[DriftEntryOut] = []
    for change, source in rows:
        if change.pct_change is None or abs(change.pct_change) < threshold_pct:
            continue
        if change.old_value is None or change.new_value is None:
            continue
        entries.append(
            DriftEntryOut(
                entity_type=change.entity_type.value,
                entity_id=change.entity_id,
                label=labels.get(change.entity_id, change.entity_id),
                field=change.field,
                old_value=change.old_value,
                new_value=change.new_value,
                pct_change=change.pct_change,
                source_name=source.name if source else "unknown",
                detected_at=change.detected_at,
            )
        )
    return entries


async def _labels_for(db: AsyncSession, entity_ids: list[str]) -> dict[str, str]:
    """Human-readable names, so a drift alert says "Claude Opus 5" not "mdl_01…"."""
    if not entity_ids:
        return {}
    unique = list(set(entity_ids))
    labels: dict[str, str] = {}

    models = (
        (await db.execute(select(ModelPricing).where(ModelPricing.id.in_(unique)))).scalars().all()
    )
    for model in models:
        labels[model.id] = f"{model.provider} / {model.display_name}"

    gpus = (await db.execute(select(GpuPricing).where(GpuPricing.id.in_(unique)))).scalars().all()
    for gpu in gpus:
        labels[gpu.id] = f"{gpu.provider} / {gpu.instance_name}"

    return labels


async def stale_rows(db: AsyncSession, *, older_than_days: int = AGING_DAYS) -> int:
    cutoff = utcnow() - timedelta(days=older_than_days)
    models = (
        await db.execute(select(ModelPricing.id).where(ModelPricing.last_verified_at < cutoff))
    ).all()
    gpus = (
        await db.execute(select(GpuPricing.id).where(GpuPricing.last_verified_at < cutoff))
    ).all()
    return len(models) + len(gpus)


async def oldest_verification(db: AsyncSession) -> datetime | None:
    stmt = select(ModelPricing.last_verified_at).order_by(ModelPricing.last_verified_at).limit(1)
    return (await db.execute(stmt)).scalar_one_or_none()
