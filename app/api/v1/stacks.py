"""Saved stacks: CRUD, clone, versions, and diff (M15).

Every read re-scores against today's catalog. That is the whole reason the
score is not a column: a stack saved last month with a component that has
since been buried shows the lower score and the deprecation flag the next time
it is opened, without anyone running a backfill.

Saving requires an account — it is the conversion moment the public
`/architect/recommend` sets up.
"""

from __future__ import annotations

from datetime import datetime
from decimal import Decimal
from typing import Any

from fastapi import APIRouter, Query
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.api.deps import CurrentMembership, CurrentUser, Db
from app.core.database import utcnow
from app.core.errors import Forbidden, NotFound, ValidationFailed
from app.core.responses import Envelope, ok
from app.models.organization import OrganizationMember
from app.models.stack import Stack, StackVersion
from app.models.user import User
from app.repositories import scoped
from app.schemas.architect import (
    StackDiffOut,
    StackIn,
    StackOut,
    StackPatch,
    StackVersionOut,
)
from app.services import catalog_service, stack_architect_service, stack_score_service

router = APIRouter(tags=["stacks"])


async def _score_of(db: AsyncSession, slugs: list[str], requirements: dict[str, Any]) -> Decimal:
    """Re-score against the catalog as it is now.

    Components that no longer exist are simply absent from the scoring set —
    a slug that has been removed from the catalog cannot contribute a
    dimension, and inventing a neutral score for it would hide the removal.
    """
    catalog = await catalog_service.list_tools(db)
    by_slug = {tool.slug: tool for tool in catalog}
    components = [by_slug[slug] for slug in slugs if slug in by_slug]
    if not components:
        return Decimal(0)

    # A one-component stack is legitimate and has no pairs to score. Asking
    # for compatibility anyway raises, which would turn "I saved a stack with
    # one thing in it" into a 404.
    compatibility = (
        await catalog_service.get_compatibility(db, [t.slug for t in components])
        if len(components) > 1
        else None
    )
    return stack_score_service.score(
        components,
        monthly_budget=int(requirements.get("monthly_budget", 2_000) or 0),
        scale_target=str(requirements.get("scale_target", "medium")),
        sensitivity=str(requirements.get("sensitivity", "internal")),
        compatibility=compatibility,
    ).total


async def _out(db: AsyncSession, stack: Stack, user: User) -> StackOut:
    catalog = await catalog_service.list_tools(db)
    by_slug = {tool.slug: tool for tool in catalog}
    deprecated = [
        slug
        for slug in stack.component_slugs
        if slug in by_slug and by_slug[slug].status not in stack_architect_service.RECOMMENDABLE
    ]

    owner_name: str | None = user.name
    if stack.user_id != user.id:
        owner = await db.get(User, stack.user_id)
        owner_name = owner.name if owner else None

    return StackOut(
        id=stack.id,
        name=stack.name,
        description=stack.description,
        component_slugs=list(stack.component_slugs),
        requirements=stack.requirements,
        current_version=stack.current_version,
        score=str(await _score_of(db, list(stack.component_slugs), stack.requirements)),
        deprecated_components=deprecated,
        visibility=stack.visibility.value,
        organization_id=stack.organization_id,
        is_yours=stack.user_id == user.id,
        owner_name=owner_name,
        created_at=stack.created_at,
        updated_at=stack.updated_at,
    )


async def _readable(
    db: AsyncSession, stack_id: str, user: User, member: OrganizationMember | None
) -> Stack:
    """Own rows, plus the team's shared rows when acting inside an org (M21).
    Not-visible and does-not-exist are the same 404."""
    return await scoped.get_team_readable(db, Stack, stack_id, user, member, label="stack")


async def _editable(
    db: AsyncSession, stack_id: str, user: User, member: OrganizationMember | None
) -> Stack:
    return await scoped.get_team_editable(db, Stack, stack_id, user, member, label="stack")


_apply_visibility = scoped.set_visibility


def _snapshot(stack: Stack, *, version: int, summary: str | None, at: datetime) -> StackVersion:
    return StackVersion(
        stack_id=stack.id,
        version=version,
        name=stack.name,
        requirements=stack.requirements,
        component_slugs=list(stack.component_slugs),
        change_summary=summary,
        created_at=at,
    )


@router.post("", response_model=Envelope[StackOut], name="create_stack", status_code=201)
async def create_stack(
    db: Db, user: CurrentUser, member: CurrentMembership, payload: StackIn
) -> dict[str, Any]:
    now = utcnow()
    stack = Stack(
        user_id=user.id,
        name=payload.name,
        description=payload.description,
        requirements=payload.requirements,
        component_slugs=payload.component_slugs,
        current_version=1,
        source_run_id=payload.source_run_id,
    )
    if payload.visibility is not None:
        _apply_visibility(stack, payload.visibility, member)
    db.add(stack)
    await db.flush()

    db.add(_snapshot(stack, version=1, summary="Initial version", at=now))
    await db.flush()
    # `created_at`/`updated_at` are server defaults and are not populated by
    # the flush. Reading them would lazy-load, which is IO in an attribute
    # access — and that is not allowed outside a greenlet on the async driver.
    await db.refresh(stack)

    return ok(await _out(db, stack, user))


@router.get("", response_model=Envelope[list[StackOut]], name="list_stacks")
async def list_stacks(
    db: Db,
    user: CurrentUser,
    member: CurrentMembership,
    scope: str = Query(default="mine", pattern="^(mine|team)$"),
    limit: int = Query(default=50, ge=1, le=100),
) -> dict[str, Any]:
    """`scope=team` lists the acting organization's shared stacks.

    Without an organization header the team is empty rather than an error —
    the same request every personal caller has been making since M15.
    """
    if scope == "team":
        if member is None:
            return ok([])
        statement = scoped.team_shared(Stack, member)
    else:
        statement = scoped.owned_by(Stack, user)

    rows = (
        (await db.execute(statement.order_by(Stack.updated_at.desc()).limit(limit)))
        .scalars()
        .all()
    )
    return ok([await _out(db, stack, user) for stack in rows])


@router.get("/{stack_id}", response_model=Envelope[StackOut], name="get_stack")
async def get_stack(
    db: Db, user: CurrentUser, member: CurrentMembership, stack_id: str
) -> dict[str, Any]:
    return ok(await _out(db, await _readable(db, stack_id, user, member), user))


@router.patch("/{stack_id}", response_model=Envelope[StackOut], name="update_stack")
async def update_stack(
    db: Db, user: CurrentUser, member: CurrentMembership, stack_id: str, payload: StackPatch
) -> dict[str, Any]:
    """Every save creates a version.

    Not only component changes: a rename is a version too, because the diff
    view is the record of what happened to a stack and a rename that leaves no
    trace makes the history lie by omission.
    """
    stack = await _editable(db, stack_id, user, member)

    if payload.name is not None:
        stack.name = payload.name
    if payload.description is not None:
        stack.description = payload.description
    if payload.component_slugs is not None:
        stack.component_slugs = payload.component_slugs
    if payload.requirements is not None:
        stack.requirements = payload.requirements
    if payload.visibility is not None:
        # Only the author moves work in and out of the team. A teammate
        # editing components is collaboration; a teammate flipping someone
        # else's work private-or-public is not.
        if stack.user_id != user.id:
            raise Forbidden("Only the stack's author can change its visibility.")
        _apply_visibility(stack, payload.visibility, member)

    stack.current_version += 1
    db.add(
        _snapshot(
            stack,
            version=stack.current_version,
            summary=payload.change_summary,
            at=utcnow(),
        )
    )
    await db.flush()
    await db.refresh(stack)

    return ok(await _out(db, stack, user))


@router.delete("/{stack_id}", status_code=204, name="delete_stack")
async def delete_stack(db: Db, user: CurrentUser, stack_id: str) -> None:
    # Author-only, deliberately tighter than edit: a teammate can improve
    # shared work but not destroy it.
    stack = await scoped.get_owned(db, Stack, stack_id, user, label="stack")
    await db.delete(stack)
    await db.flush()


@router.post(
    "/{stack_id}/clone",
    response_model=Envelope[StackOut],
    name="clone_stack",
    status_code=201,
)
async def clone_stack(
    db: Db, user: CurrentUser, member: CurrentMembership, stack_id: str
) -> dict[str, Any]:
    """A clone starts its own history at v1.

    Carrying the source's version numbers over would make two stacks claim the
    same v3 and mean different things by it. A teammate's shared stack can be
    cloned — the clone is private personal work, which is the supported way to
    build on shared work without editing it.
    """
    source = await _readable(db, stack_id, user, member)
    now = utcnow()

    clone = Stack(
        user_id=user.id,
        name=f"{source.name} (copy)",
        description=source.description,
        requirements=source.requirements,
        component_slugs=list(source.component_slugs),
        current_version=1,
        cloned_from_id=source.id,
    )
    db.add(clone)
    await db.flush()

    db.add(_snapshot(clone, version=1, summary=f"Cloned from {source.name}", at=now))
    await db.flush()
    await db.refresh(clone)

    return ok(await _out(db, clone, user))


@router.get(
    "/{stack_id}/versions",
    response_model=Envelope[list[StackVersionOut]],
    name="list_stack_versions",
)
async def list_stack_versions(
    db: Db, user: CurrentUser, member: CurrentMembership, stack_id: str
) -> dict[str, Any]:
    await _readable(db, stack_id, user, member)
    rows = (
        (
            await db.execute(
                select(StackVersion)
                .where(StackVersion.stack_id == stack_id)
                .order_by(StackVersion.version.desc())
            )
        )
        .scalars()
        .all()
    )
    return ok([StackVersionOut.model_validate(row, from_attributes=True) for row in rows])


@router.get(
    "/{stack_id}/versions/{version}",
    response_model=Envelope[StackVersionOut],
    name="get_stack_version",
)
async def get_stack_version(
    db: Db, user: CurrentUser, member: CurrentMembership, stack_id: str, version: int
) -> dict[str, Any]:
    await _readable(db, stack_id, user, member)
    row = await _version(db, stack_id, version)
    return ok(StackVersionOut.model_validate(row, from_attributes=True))


async def _version(db: AsyncSession, stack_id: str, version: int) -> StackVersion:
    row = (
        await db.execute(
            select(StackVersion).where(
                StackVersion.stack_id == stack_id, StackVersion.version == version
            )
        )
    ).scalar_one_or_none()
    if row is None:
        raise NotFound(f"This stack has no version {version}.")
    return row


@router.get("/{stack_id}/diff", response_model=Envelope[StackDiffOut], name="diff_stack_versions")
async def diff_stack_versions(
    db: Db,
    user: CurrentUser,
    member: CurrentMembership,
    stack_id: str,
    from_version: int = Query(alias="from", ge=1),
    to_version: int = Query(alias="to", ge=1),
) -> dict[str, Any]:
    """What changed, and what it did to the score.

    Both sides are re-scored against today's catalog. Comparing a score stored
    at v1 against a fresh v2 score would blame the user's edit for a month of
    catalog drift.
    """
    await _readable(db, stack_id, user, member)
    if from_version == to_version:
        raise ValidationFailed.on_field("to", "Pick two different versions to compare.")

    earlier = await _version(db, stack_id, from_version)
    later = await _version(db, stack_id, to_version)

    before = set(earlier.component_slugs)
    after = set(later.component_slugs)

    score_from = await _score_of(db, list(earlier.component_slugs), earlier.requirements)
    score_to = await _score_of(db, list(later.component_slugs), later.requirements)

    return ok(
        StackDiffOut(
            from_version=from_version,
            to_version=to_version,
            added=sorted(after - before),
            removed=sorted(before - after),
            unchanged=sorted(before & after),
            score_from=str(score_from),
            score_to=str(score_to),
            score_delta=str(score_to - score_from),
            change_summary=later.change_summary,
        )
    )
