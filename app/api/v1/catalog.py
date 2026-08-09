"""Catalog read endpoints.

All reads are public and unauthenticated. The pricing catalog is a genuine SEO
and trust asset — a comparison page that a search engine cannot index, or that
a prospective user cannot look at before signing up, is worth much less than
one that is open. Writes (`refresh`) are admin-only; flagging is open to
anyone, deliberately, because the people who spot a stale price first are
usually the ones who have not signed in.
"""

from __future__ import annotations

from datetime import datetime, timedelta
from typing import Annotated, Any

from fastapi import APIRouter, Query

from app.api.deps import AdminUser, CallerIdentity, Db
from app.core.database import new_id, utcnow
from app.core.errors import ValidationFailed
from app.core.logging import get_logger
from app.core.responses import Envelope, ok
from app.data.architectures_seed import ARCHITECTURES
from app.models.catalog import CatalogFlag
from app.schemas.catalog import (
    ArchitectureOut,
    CatalogStatsOut,
    CompatibilityOut,
    FlagIn,
    FlagOut,
    GpuOut,
    GraveyardEntryOut,
    ModelOut,
    PricingHistoryOut,
    RefreshResultOut,
    ToolOut,
)
from app.services import catalog_service, provenance_service
from app.workers import pricing as pricing_worker

logger = get_logger("api.catalog")

router = APIRouter(tags=["catalog"])


@router.get("/models", response_model=Envelope[list[ModelOut]], name="list_models")
async def list_models(
    db: Db,
    family: Annotated[str | None, Query(description="chat | embedding | rerank")] = None,
    provider: str | None = None,
    status: str | None = "active",
    include_all_statuses: bool = False,
) -> dict[str, Any]:
    data = await catalog_service.list_models(
        db,
        family=family,
        provider=provider,
        status=status,
        include_all_statuses=include_all_statuses,
    )
    return ok(data)


@router.get("/models/{model_id}", response_model=Envelope[ModelOut], name="get_model")
async def get_model(db: Db, model_id: str) -> dict[str, Any]:
    return ok(await catalog_service.get_model(db, model_id))


@router.get("/gpus", response_model=Envelope[list[GpuOut]], name="list_gpus")
async def list_gpus(
    db: Db,
    provider: str | None = None,
    min_vram: Annotated[int | None, Query(ge=0, description="Total VRAM across the node")] = None,
    region: str | None = None,
    spot: bool | None = None,
) -> dict[str, Any]:
    data = await catalog_service.list_gpus(
        db, provider=provider, min_vram=min_vram, region=region, spot=spot
    )
    return ok(data)


@router.get("/tools", response_model=Envelope[list[ToolOut]], name="list_tools")
async def list_tools(
    db: Db,
    category: str | None = None,
    status: str | None = None,
    use_case: str | None = None,
    tags: Annotated[str | None, Query(description="Comma-separated; all must match")] = None,
) -> dict[str, Any]:
    tag_list = [tag.strip() for tag in tags.split(",") if tag.strip()] if tags else None
    data = await catalog_service.list_tools(
        db, category=category, status=status, use_case=use_case, tags=tag_list
    )
    return ok(data)


@router.get("/tools/{slug}", response_model=Envelope[ToolOut], name="get_tool")
async def get_tool(db: Db, slug: str) -> dict[str, Any]:
    return ok(await catalog_service.get_tool(db, slug))


@router.get("/compatibility", response_model=Envelope[CompatibilityOut], name="get_compatibility")
async def get_compatibility(
    db: Db,
    tools: Annotated[str, Query(description="Comma-separated tool slugs, 2-12")],
) -> dict[str, Any]:
    slugs = [slug.strip() for slug in tools.split(",") if slug.strip()]
    if len(slugs) < 2:
        raise ValidationFailed("Provide at least two tool slugs.")
    if len(slugs) > 12:
        # 12 tools is 66 pairs. Beyond that the matrix stops being readable
        # and the request starts being a way to make the server do work.
        raise ValidationFailed("Compare at most 12 tools at once.")
    return ok(await catalog_service.get_compatibility(db, slugs))


@router.get(
    "/architectures",
    response_model=Envelope[list[ArchitectureOut]],
    name="list_architectures",
)
async def list_architectures() -> dict[str, Any]:
    """Open-weight model architectures, for VRAM estimation.

    No database and no provenance chip. These are physical properties fixed at
    publication — a layer count cannot go stale — so the freshness machinery
    that surrounds every price would be answering a question nobody has.
    """
    return ok(
        [
            ArchitectureOut(
                key=arch.key,
                name=arch.name,
                family=arch.family,
                params_b=arch.params_b,
                layers=arch.layers,
                hidden_size=arch.hidden_size,
                heads=arch.heads,
                kv_heads=arch.kv_heads,
                head_dim=arch.head_dim,
                max_context=arch.max_context,
                uses_gqa=arch.uses_gqa,
            )
            for arch in ARCHITECTURES
        ]
    )


@router.get("/graveyard", response_model=Envelope[list[GraveyardEntryOut]], name="get_graveyard")
async def get_graveyard(db: Db) -> dict[str, Any]:
    return ok(await catalog_service.get_graveyard(db))


@router.get(
    "/pricing/history",
    response_model=Envelope[list[PricingHistoryOut]],
    name="get_pricing_history",
)
async def get_pricing_history(
    db: Db,
    entity_id: str | None = None,
    since: datetime | None = None,
    limit: Annotated[int, Query(ge=1, le=500)] = 100,
) -> dict[str, Any]:
    rows = await provenance_service.history_for(db, entity_id=entity_id, since=since, limit=limit)
    return ok([PricingHistoryOut.model_validate(row) for row in rows])


@router.get("/stats", response_model=Envelope[CatalogStatsOut], name="get_catalog_stats")
async def get_catalog_stats(db: Db) -> dict[str, Any]:
    counts = await catalog_service.count_rows(db)
    oldest = await provenance_service.oldest_verification(db)
    return ok(
        CatalogStatsOut(
            **counts,
            oldest_verification=oldest.date() if oldest else None,
            stale_rows=await provenance_service.stale_rows(db),
        )
    )


@router.post(
    "/pricing/refresh",
    response_model=Envelope[RefreshResultOut],
    name="refresh_pricing",
    summary="Force a verification run (admin)",
)
async def refresh_pricing(
    db: Db,
    _admin: AdminUser,
    source: str | None = None,
) -> dict[str, Any]:
    """Report what needs an editor's attention.

    Prices are hardcoded and verified by hand (D-16), so this endpoint reads
    rather than writes: it never changes a price, and with no fetchers
    registered it never will. What it returns is the editorial queue — recent
    unapplied drift, and how many rows are past their freshness window.

    Kept as POST rather than GET because it walks every source and every
    priced row, which is not something to hand a crawler.
    """
    result = await pricing_worker.verify_all(db, source_slug=source)
    entries = await provenance_service.detect_drift(db, since=utcnow() - timedelta(days=90))
    if result.changes_detected:
        await catalog_service.invalidate()
    return ok(
        RefreshResultOut(
            checked=result.checked,
            changes_detected=result.changes_detected,
            sources_failed=result.sources_failed,
            sources_skipped=result.sources_skipped,
            stale_rows=await provenance_service.stale_rows(db),
            entries=entries,
            ran_at=result.ran_at,
        )
    )


@router.post("/flag", response_model=Envelope[FlagOut], name="flag_catalog_entry")
async def flag_catalog_entry(db: Db, identity: CallerIdentity, payload: FlagIn) -> dict[str, Any]:
    """Report a value as wrong or stale. Open to anyone, signed in or not."""
    flag = CatalogFlag(
        id=new_id("flag"),
        entity_type=payload.entity_type,
        entity_id=payload.entity_id,
        field=payload.field,
        suggested_value=payload.suggested_value,
        note=payload.note,
        source_url=payload.source_url,
        reported_by_user_id=identity.user.id if identity.user else None,
        created_at=utcnow(),
    )
    db.add(flag)
    await db.flush()

    logger.info(
        "catalog.flagged",
        entity_type=payload.entity_type,
        entity_id=payload.entity_id,
        field=payload.field,
        authenticated=identity.is_authenticated,
    )
    return ok(FlagOut.model_validate(flag))
