"""Projects, items, and the carried session (M17).

Every handler here takes `CurrentUser` and hands it to the service, which
scopes the query. No handler compares a `user_id` itself — that check exists
once, in `repositories.scoped`, and the endpoints inherit it.
"""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter, Query
from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.api.deps import CurrentUser, Db
from app.core.responses import Envelope, ok
from app.models.project import Project, ProjectItem, ProjectItemType
from app.models.stack import Stack
from app.models.tool_run import ToolRun
from app.schemas.workspace import (
    CarriedSessionOut,
    CarriedSessionPatch,
    PinIn,
    ProjectIn,
    ProjectItemIn,
    ProjectItemOut,
    ProjectOut,
    ProjectPatch,
    ReorderIn,
)
from app.services import project_service

router = APIRouter(tags=["projects"])


async def _out(db: AsyncSession, project: Project) -> ProjectOut:
    count = int(
        await db.scalar(
            select(func.count())
            .select_from(ProjectItem)
            .where(ProjectItem.project_id == project.id)
        )
        or 0
    )
    return ProjectOut(
        id=project.id,
        name=project.name,
        description=project.description,
        use_case=project.use_case,
        session_state=project.session_state,
        item_count=count,
        archived_at=project.archived_at,
        created_at=project.created_at,
        updated_at=project.updated_at,
    )


@router.post("", response_model=Envelope[ProjectOut], name="create_project", status_code=201)
async def create_project(db: Db, user: CurrentUser, payload: ProjectIn) -> dict[str, Any]:
    project = await project_service.create(
        db,
        user,
        name=payload.name,
        description=payload.description,
        use_case=payload.use_case,
    )
    return ok(await _out(db, project))


@router.get("", response_model=Envelope[list[ProjectOut]], name="list_projects")
async def list_projects(
    db: Db, user: CurrentUser, include_archived: bool = Query(default=False)
) -> dict[str, Any]:
    projects = await project_service.list_for(db, user, include_archived=include_archived)
    return ok([await _out(db, project) for project in projects])


@router.get("/{project_id}", response_model=Envelope[ProjectOut], name="get_project")
async def get_project(db: Db, user: CurrentUser, project_id: str) -> dict[str, Any]:
    return ok(await _out(db, await project_service.get(db, project_id, user)))


@router.patch("/{project_id}", response_model=Envelope[ProjectOut], name="update_project")
async def update_project(
    db: Db, user: CurrentUser, project_id: str, payload: ProjectPatch
) -> dict[str, Any]:
    project = await project_service.update(
        db,
        project_id,
        user,
        name=payload.name,
        description=payload.description,
        use_case=payload.use_case,
        archived=payload.archived,
    )
    return ok(await _out(db, project))


@router.delete("/{project_id}", status_code=204, name="delete_project")
async def delete_project(db: Db, user: CurrentUser, project_id: str) -> None:
    await project_service.remove(db, project_id, user)


# ── items ────────────────────────────────────────────────────────────────────


async def _resolve(db: AsyncSession, item: ProjectItem) -> ProjectItemOut:
    """Render an item by looking up whatever it points at.

    A row whose target has been deleted comes back with `title: null` rather
    than being hidden. The polymorphic join cannot be a foreign key, so a
    dangling item is a real state, and silently dropping it would make a
    project quietly lose things.
    """
    title: str | None = None
    subtitle: str | None = None
    href: str | None = None

    if item.item_type is ProjectItemType.RUN:
        run = await db.get(ToolRun, item.item_id)
        if run is not None:
            title = run.tool_slug
            subtitle = run.workflow
            href = f"/{run.workflow}?run={run.id}"
    elif item.item_type is ProjectItemType.STACK:
        stack = await db.get(Stack, item.item_id)
        if stack is not None:
            title = stack.name
            subtitle = ", ".join(stack.component_slugs[:4])
            href = f"/stack-architect/my-stacks?stack={stack.id}"

    return ProjectItemOut(
        id=item.id,
        item_type=item.item_type.value,
        item_id=item.item_id,
        position=item.position,
        pinned=item.pinned,
        note=item.note,
        title=title,
        subtitle=subtitle,
        href=href,
        created_at=item.created_at,
    )


@router.get(
    "/{project_id}/items", response_model=Envelope[list[ProjectItemOut]], name="list_project_items"
)
async def list_project_items(db: Db, user: CurrentUser, project_id: str) -> dict[str, Any]:
    items = await project_service.list_items(db, project_id, user)
    return ok([await _resolve(db, item) for item in items])


@router.post(
    "/{project_id}/items",
    response_model=Envelope[ProjectItemOut],
    name="add_project_item",
    status_code=201,
)
async def add_project_item(
    db: Db, user: CurrentUser, project_id: str, payload: ProjectItemIn
) -> dict[str, Any]:
    item = await project_service.add_item(
        db,
        project_id,
        user,
        item_type=ProjectItemType(payload.item_type),
        item_id=payload.item_id,
        note=payload.note,
        pinned=payload.pinned,
    )
    return ok(await _resolve(db, item))


@router.delete("/{project_id}/items/{item_id}", status_code=204, name="remove_project_item")
async def remove_project_item(db: Db, user: CurrentUser, project_id: str, item_id: str) -> None:
    await project_service.remove_item(db, project_id, user, item_id)


@router.patch(
    "/{project_id}/items/order",
    response_model=Envelope[list[ProjectItemOut]],
    name="reorder_project_items",
)
async def reorder_project_items(
    db: Db, user: CurrentUser, project_id: str, payload: ReorderIn
) -> dict[str, Any]:
    items = await project_service.reorder(db, project_id, user, item_ids=payload.item_ids)
    return ok([await _resolve(db, item) for item in items])


@router.patch(
    "/{project_id}/items/{item_id}/pin",
    response_model=Envelope[ProjectItemOut],
    name="pin_project_item",
)
async def pin_project_item(
    db: Db, user: CurrentUser, project_id: str, item_id: str, payload: PinIn
) -> dict[str, Any]:
    item = await project_service.set_pinned(db, project_id, user, item_id, pinned=payload.pinned)
    return ok(await _resolve(db, item))


# ── the carried session ──────────────────────────────────────────────────────


@router.get(
    "/{project_id}/session",
    response_model=Envelope[CarriedSessionOut],
    name="get_session",
)
async def get_session(db: Db, user: CurrentUser, project_id: str) -> dict[str, Any]:
    project = await project_service.get(db, project_id, user)
    return ok(CarriedSessionOut(session_state=project.session_state))


@router.patch(
    "/{project_id}/session",
    response_model=Envelope[CarriedSessionOut],
    name="update_session",
)
async def update_session(
    db: Db, user: CurrentUser, project_id: str, payload: CarriedSessionPatch
) -> dict[str, Any]:
    """Merge, never replace.

    A tool handing its two figures forward must not wipe the six a previous
    tool contributed.
    """
    state = await project_service.merge_session(db, project_id, user, values=payload.values)
    return ok(CarriedSessionOut(session_state=state))


@router.delete(
    "/{project_id}/session",
    response_model=Envelope[CarriedSessionOut],
    name="clear_session",
)
async def clear_session(db: Db, user: CurrentUser, project_id: str) -> dict[str, Any]:
    state = await project_service.clear_session(db, project_id, user)
    return ok(CarriedSessionOut(session_state=state))
