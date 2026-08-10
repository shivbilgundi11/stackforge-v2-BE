"""Projects, their items, and the carried session (M17).

Everything here goes through `repositories.scoped`, so no function in this
module can read another user's row even by mistake.

The quota is per plan and read from a table rather than branched on in code
(M20 owns the table; this reads it). "Free gets no projects" is a pricing
decision, and pricing decisions that live in `if` statements get out of step
with the pricing page.
"""

from __future__ import annotations

from typing import Any, Final, cast

from sqlalchemy import delete, func, select
from sqlalchemy.engine import CursorResult
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.database import utcnow
from app.core.errors import NotFound, QuotaExceeded, ValidationFailed
from app.core.logging import get_logger
from app.models.project import Project, ProjectItem, ProjectItemType
from app.models.stack import Stack
from app.models.tool_run import ToolRun
from app.models.user import Plan, User
from app.repositories import scoped

logger = get_logger("projects")

#: Projects allowed per plan. Free gets none — projects are the thing an
#: account is *for*, so giving them away removes the reason to upgrade while
#: adding the storage cost of keeping them.
PROJECT_LIMIT: Final[dict[str, int]] = {
    Plan.FREE.value: 0,
    Plan.PRO.value: 20,
    Plan.TEAM.value: 200,
    Plan.ENTERPRISE.value: 10_000,
}

#: How an item type resolves to a row, so a project can render its contents
#: without the caller knowing the shape of each one.
ITEM_MODELS: Final = {
    ProjectItemType.RUN: ToolRun,
    ProjectItemType.STACK: Stack,
}


def limit_for(user: User) -> int:
    return PROJECT_LIMIT.get(user.plan.value, PROJECT_LIMIT[Plan.FREE.value])


async def count_for(db: AsyncSession, user: User) -> int:
    return int(
        await db.scalar(
            select(func.count())
            .select_from(Project)
            .where(Project.user_id == user.id, Project.archived_at.is_(None))
        )
        or 0
    )


async def create(
    db: AsyncSession,
    user: User,
    *,
    name: str,
    description: str | None = None,
    use_case: str | None = None,
) -> Project:
    limit = limit_for(user)
    used = await count_for(db, user)
    if used >= limit:
        raise QuotaExceeded(
            (
                "Projects are not included on the free plan."
                if limit == 0
                else f"You have used all {limit} projects on your plan."
            ),
            details={
                "quota": {
                    "metric": "projects",
                    "limit": limit,
                    "used": used,
                    "remaining": 0,
                    "period": "plan",
                    "plan": user.plan.value,
                }
            },
        )

    project = Project(user_id=user.id, name=name, description=description, use_case=use_case)
    db.add(project)
    await db.flush()
    await db.refresh(project)
    logger.info("projects.created", project_id=project.id, user_id=user.id)
    return project


async def list_for(
    db: AsyncSession, user: User, *, include_archived: bool = False
) -> list[Project]:
    statement = scoped.owned_by(Project, user).order_by(Project.updated_at.desc())
    if not include_archived:
        statement = statement.where(Project.archived_at.is_(None))
    return list((await db.execute(statement)).scalars().all())


async def get(db: AsyncSession, project_id: str, user: User) -> Project:
    return await scoped.get_owned(db, Project, project_id, user, label="project")


async def update(
    db: AsyncSession,
    project_id: str,
    user: User,
    *,
    name: str | None = None,
    description: str | None = None,
    use_case: str | None = None,
    archived: bool | None = None,
) -> Project:
    project = await get(db, project_id, user)

    if name is not None:
        project.name = name
    if description is not None:
        project.description = description
    if use_case is not None:
        project.use_case = use_case
    if archived is not None:
        project.archived_at = utcnow() if archived else None

    await db.flush()
    await db.refresh(project)
    return project


async def remove(db: AsyncSession, project_id: str, user: User) -> None:
    """Delete the container, not the contents.

    Items cascade because they are the *arrangement*; the runs and stacks they
    point at are owned separately and survive. Deleting a project must not be
    a way to lose work the user did not intend to delete.
    """
    project = await get(db, project_id, user)
    await db.delete(project)
    await db.flush()


# ── items ────────────────────────────────────────────────────────────────────


async def add_item(
    db: AsyncSession,
    project_id: str,
    user: User,
    *,
    item_type: ProjectItemType,
    item_id: str,
    note: str | None = None,
    pinned: bool = False,
) -> ProjectItem:
    """Attach something the user owns.

    Ownership of the *item* is checked as well as of the project. Without it,
    a project would be a way to read any run id by adding it and listing the
    project back.
    """
    project = await get(db, project_id, user)
    await _assert_owns_item(db, user, item_type, item_id)

    existing = await db.scalar(
        select(ProjectItem).where(
            ProjectItem.project_id == project.id,
            ProjectItem.item_type == item_type,
            ProjectItem.item_id == item_id,
        )
    )
    if existing is not None:
        return existing

    next_position = int(
        await db.scalar(
            select(func.coalesce(func.max(ProjectItem.position), -1)).where(
                ProjectItem.project_id == project.id
            )
        )
        or -1
    )

    item = ProjectItem(
        project_id=project.id,
        item_type=item_type,
        item_id=item_id,
        position=next_position + 1,
        pinned=pinned,
        note=note,
        created_at=utcnow(),
    )
    db.add(item)
    # Touch the project so "last activity" means what it says.
    project.updated_at = utcnow()
    await db.flush()
    return item


async def _assert_owns_item(
    db: AsyncSession, user: User, item_type: ProjectItemType, item_id: str
) -> None:
    model = ITEM_MODELS.get(item_type)
    if model is None:
        # Artifacts and templates have no table yet (M18, M19). Refusing is
        # right: silently accepting an id nothing can resolve would produce a
        # project full of rows that render as nothing.
        raise ValidationFailed.on_field(
            "item_type", f"{item_type.value} items cannot be attached yet."
        )

    owned = await db.scalar(
        select(model.id).where(  # type: ignore[attr-defined]
            model.id == item_id,  # type: ignore[attr-defined]
            model.user_id == user.id,  # type: ignore[attr-defined]
        )
    )
    if owned is None:
        raise NotFound(f"No {item_type.value} with that id.")


async def list_items(db: AsyncSession, project_id: str, user: User) -> list[ProjectItem]:
    await get(db, project_id, user)
    return list(
        (
            await db.execute(
                select(ProjectItem)
                .where(ProjectItem.project_id == project_id)
                # Pinned first, then the user's arrangement.
                .order_by(ProjectItem.pinned.desc(), ProjectItem.position)
            )
        )
        .scalars()
        .all()
    )


async def remove_item(db: AsyncSession, project_id: str, user: User, item_id: str) -> None:
    await get(db, project_id, user)
    result = await db.execute(
        delete(ProjectItem).where(ProjectItem.project_id == project_id, ProjectItem.id == item_id)
    )
    if cast("CursorResult[Any]", result).rowcount == 0:
        raise NotFound("No item with that id in this project.")
    await db.flush()


async def reorder(
    db: AsyncSession, project_id: str, user: User, *, item_ids: list[str]
) -> list[ProjectItem]:
    """Set the arrangement to exactly this order.

    Items the caller omitted keep their relative order *after* the ones named,
    rather than being dropped — a client that reorders a filtered view must
    not silently discard what it could not see.
    """
    items = await list_items(db, project_id, user)
    by_id = {item.id: item for item in items}

    unknown = [item_id for item_id in item_ids if item_id not in by_id]
    if unknown:
        raise NotFound(f"No item with id {unknown[0]} in this project.")

    position = 0
    for item_id in item_ids:
        by_id[item_id].position = position
        position += 1
    for item in items:
        if item.id not in item_ids:
            item.position = position
            position += 1

    await db.flush()
    return await list_items(db, project_id, user)


async def set_pinned(
    db: AsyncSession, project_id: str, user: User, item_id: str, *, pinned: bool
) -> ProjectItem:
    items = await list_items(db, project_id, user)
    item = next((row for row in items if row.id == item_id), None)
    if item is None:
        raise NotFound("No item with that id in this project.")
    item.pinned = pinned
    await db.flush()
    return item


# ── the carried session ──────────────────────────────────────────────────────


async def merge_session(
    db: AsyncSession, project_id: str, user: User, *, values: dict[str, Any]
) -> dict[str, Any]:
    """Merge values into the project's carried session.

    Merge rather than replace: a tool carrying its own two figures forward must
    not wipe the six a previous tool contributed. `carried_from` accumulates so
    the UI can say where each number came from — a prefilled figure whose
    origin is invisible stops being the user's estimate.
    """
    project = await get(db, project_id, user)

    state = dict(project.session_state or {})
    incoming = dict(values)

    provenance = list(state.get("carried_from") or [])
    for entry in incoming.pop("carried_from", []) or []:
        provenance = [row for row in provenance if row.get("tool") != entry.get("tool")]
        provenance.append(entry)

    state.update(incoming)
    state["carried_from"] = provenance[-20:]

    # Reassigned rather than mutated: SQLAlchemy does not track in-place edits
    # to a JSONB dict, and the update would be silently dropped on flush.
    project.session_state = state
    project.updated_at = utcnow()
    await db.flush()
    await db.refresh(project)
    return project.session_state


async def clear_session(db: AsyncSession, project_id: str, user: User) -> dict[str, Any]:
    project = await get(db, project_id, user)
    project.session_state = {}
    await db.flush()
    return {}
