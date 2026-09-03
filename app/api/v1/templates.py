"""The template library (M19).

Every route here is public. No `CurrentUser`, no gate on reading — `PRD.md` §15
makes this the product's best organic acquisition surface, and an endpoint
behind a token cannot be crawled, shared, or be the thing that brings someone
in. `CallerIdentity` appears only to decide whether a premium *body* is
unlocked.
"""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter, Query, Response

from app.api.deps import CallerIdentity, Db
from app.core.responses import Envelope, ok
from app.models.template import Template
from app.schemas.templates import (
    CategoryOut,
    FacetsOut,
    LibraryOut,
    TemplateDetailOut,
    TemplateFileOut,
    TemplateSummaryOut,
)
from app.services import template_service
from app.services.template_service import Filters

router = APIRouter(tags=["templates"])

#: Featured on the hub. Hand-picked rather than derived from `view_count`,
#: which would make the hub a popularity feedback loop that buries every new
#: template the day it ships.
FEATURED = ("rag-chatbot", "fastapi-rag", "ai-production-readiness", "cursor-rules")


def _summary(row: Template) -> TemplateSummaryOut:
    return TemplateSummaryOut(
        slug=row.slug,
        title=row.title,
        category=row.category.value,
        difficulty=row.difficulty.value,
        summary=row.summary,
        use_cases=list(row.use_cases),
        tags=list(row.tags),
        is_premium=row.is_premium,
        file_count=len(row.files),
        is_stack_template=bool(row.stack_input),
        view_count=row.view_count,
        copy_count=row.copy_count,
        published_at=row.published_at,
    )


@router.get("", response_model=Envelope[LibraryOut], name="get_template_library")
async def get_template_library(db: Db, identity: CallerIdentity) -> dict[str, Any]:
    """The hub, in one request.

    Category counts, the featured set, and the newest templates together —
    three round trips to render one page is three chances for it to paint half
    populated.
    """
    counts = await template_service.counts_by_category(db, identity)
    recent = await template_service.search(db, identity, filters=Filters(), limit=6)

    featured: list[Template] = []
    for slug in FEATURED:
        row = await db.scalar(template_service._visible().where(Template.slug == slug))
        if row is not None:
            featured.append(row)

    return ok(
        LibraryOut(
            total=sum(counts.values()),
            categories=[
                CategoryOut(
                    key=key,
                    label=template_service.CATEGORY_LABELS[key],
                    description=template_service.CATEGORY_BLURBS[key],
                    count=counts.get(key, 0),
                )
                for key in template_service.CATEGORY_ORDER
            ],
            featured=[_summary(row) for row in featured],
            recent=[_summary(row) for row in recent],
        )
    )


@router.get("/facets", response_model=Envelope[FacetsOut], name="get_template_facets")
async def get_template_facets(db: Db, identity: CallerIdentity) -> dict[str, Any]:
    return ok(FacetsOut(**await template_service.facets(db, identity)))


@router.get("/list", response_model=Envelope[list[TemplateSummaryOut]], name="list_templates")
async def list_templates(
    db: Db,
    identity: CallerIdentity,
    q: str | None = Query(default=None, max_length=200),
    category: str | None = Query(default=None, max_length=40),
    use_case: str | None = Query(default=None, max_length=40),
    difficulty: str | None = Query(default=None, max_length=20),
    premium: bool | None = Query(default=None),
    tag: str | None = Query(default=None, max_length=40),
    limit: int = Query(default=60, ge=1, le=100),
) -> dict[str, Any]:
    """Search and filter.

    `/list` rather than the collection root because the root is the hub, and a
    hub that changed shape when a query string appeared would be two endpoints
    wearing one URL.
    """
    rows = await template_service.search(
        db,
        identity,
        filters=Filters(
            query=q,
            category=category,
            use_case=use_case,
            difficulty=difficulty,
            premium=premium,
            tag=tag,
        ),
        limit=limit,
    )
    return ok([_summary(row) for row in rows])


@router.get("/{slug}", response_model=Envelope[TemplateDetailOut], name="get_template")
async def get_template(
    db: Db, response: Response, identity: CallerIdentity, slug: str
) -> dict[str, Any]:
    row = await template_service.get(db, slug, identity)
    rendered = template_service.render(row, identity)
    await template_service.record_view(db, row)

    # Public and cacheable, but only for the unlocked variant. A shared cache
    # holding a Pro user's full body and serving it to a free one is the
    # obvious way to give away everything this gate protects.
    response.headers["Cache-Control"] = (
        "public, max-age=300" if not row.is_premium else "private, max-age=0, no-store"
    )

    return ok(
        TemplateDetailOut(
            **_summary(row).model_dump(),
            content_markdown=rendered.content_markdown,
            files=[TemplateFileOut(**file) for file in rendered.files],
            locked=rendered.locked,
            truncated=rendered.truncated,
            stack_input=dict(row.stack_input),
            related_tools=list(row.related_tools),
            related=[_summary(other) for other in await template_service.related(db, row)],
        )
    )


@router.post("/{slug}/copy", response_model=Envelope[dict[str, int]], name="record_template_copy")
async def record_template_copy(db: Db, identity: CallerIdentity, slug: str) -> dict[str, Any]:
    """Counted on copy or download, never on view.

    A separate endpoint rather than a flag on the GET, because the signal is
    "someone took this" and only the client knows when that happened.
    """
    row = await template_service.get(db, slug, identity)
    await template_service.record_copy(db, row)
    return ok({"copy_count": row.copy_count + 1})


__all__ = ["router"]
