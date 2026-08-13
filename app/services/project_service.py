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

from app.api.deps import Identity
from app.core.database import utcnow
from app.core.errors import Forbidden, NotFound, ValidationFailed
from app.core.logging import get_logger
from app.models.billing import Metric
from app.models.export import Export
from app.models.organization import OrganizationMember
from app.models.project import Project, ProjectItem, ProjectItemType
from app.models.stack import Stack
from app.models.tool_run import ToolRun
from app.models.user import User
from app.repositories import scoped

logger = get_logger("projects")

#: How an item type resolves to a row, so a project can render its contents
#: without the caller knowing the shape of each one.
#:
#: `ARTIFACT` maps onto `exports`, added by M18. An artifact is not a row of
#: its own — it is a rendered export of a run or a stack — so "save this to a
#: project" saves the thing that was rendered rather than a fresh noun with its
#: own lifecycle to keep in step.
ITEM_MODELS: Final = {
    ProjectItemType.RUN: ToolRun,
    ProjectItemType.STACK: Stack,
    ProjectItemType.ARTIFACT: Export,
}


def _identity(user: User) -> Identity:
    """`FeatureService` speaks in identities, and this module in users.

    Projects require an account, so a user is always convertible to an identity
    — the reverse is not true, which is why the quota layer takes the wider
    type.
    """
    return Identity(user=user, anonymous_id=None, session_id=None)


async def limit_for(db: AsyncSession, user: User) -> int | None:
    """The plan's project allowance. `None` is unlimited.

    Read from `plan_quotas` (M20). It used to be a dict in this module, which
    meant "Free gets no projects" — a pricing decision — was a code change.
    """
    from app.services import feature_service

    return await feature_service.limit_for(db, _identity(user), Metric.PROJECTS)


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
    visibility: str | None = None,
    member: OrganizationMember | None = None,
) -> Project:
    from app.services import feature_service

    # Raises `QuotaExceeded` with the real figures. A level metric, so this
    # only asks whether there is room — the insert below is the increment.
    await feature_service.consume(db, _identity(user), Metric.PROJECTS)

    project = Project(user_id=user.id, name=name, description=description, use_case=use_case)
    if visibility is not None:
        scoped.set_visibility(project, visibility, member)
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


async def list_team(db: AsyncSession, member: OrganizationMember) -> list[Project]:
    """The acting organization's shared projects (M21)."""
    statement = (
        scoped.team_shared(Project, member)
        .where(Project.archived_at.is_(None))
        .order_by(Project.updated_at.desc())
    )
    return list((await db.execute(statement)).scalars().all())


async def get(
    db: AsyncSession,
    project_id: str,
    user: User,
    member: OrganizationMember | None = None,
) -> Project:
    """Read access. Without a membership this is exactly the M17 owner check;
    with one, the team's shared projects are readable too."""
    if member is None:
        return await scoped.get_owned(db, Project, project_id, user, label="project")
    return await scoped.get_team_readable(db, Project, project_id, user, member, label="project")


async def update(
    db: AsyncSession,
    project_id: str,
    user: User,
    *,
    name: str | None = None,
    description: str | None = None,
    use_case: str | None = None,
    archived: bool | None = None,
    visibility: str | None = None,
    member: OrganizationMember | None = None,
) -> Project:
    project = await scoped.get_team_editable(db, Project, project_id, user, member, label="project")

    if name is not None:
        project.name = name
    if description is not None:
        project.description = description
    if use_case is not None:
        project.use_case = use_case
    if archived is not None:
        project.archived_at = utcnow() if archived else None
    if visibility is not None:
        # Only the author moves work in and out of the team.
        if project.user_id != user.id:
            raise Forbidden("Only the project's author can change its visibility.")
        scoped.set_visibility(project, visibility, member)

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
        # Templates have no table yet (M19). Refusing is right: silently
        # accepting an id nothing can resolve would produce a project full of
        # rows that render as nothing.
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


async def list_items(
    db: AsyncSession,
    project_id: str,
    user: User,
    member: OrganizationMember | None = None,
) -> list[ProjectItem]:
    await get(db, project_id, user, member)
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
