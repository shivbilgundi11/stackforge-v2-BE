"""Comments and approvals on team resources (M21).

Both anchor to a resource polymorphically — same shape and same trade as
`project_items`. The org they belong to is resolved *from the resource*, never
trusted from the caller: a comment claiming to be about another org's stack
dies at membership resolution with the same 404 as a stack that does not
exist.

**Which resources can carry a thread.** Stacks and projects when they are
shared with the team. Runs have no visibility of their own — a run is
commentable when it sits inside a team-shared project, and the thread belongs
to that project's organization. This is the smallest rule that makes "comment
on this run" work without inventing per-run sharing.

**Deliberately not real-time.** Polling on focus is sufficient for planning
artifacts that change on the scale of days; websockets would be infrastructure
for an interaction pattern this product does not have.
"""

from __future__ import annotations

from typing import Any

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.config import settings
from app.core.database import utcnow
from app.core.errors import Conflict, Forbidden, NotFound, ValidationFailed
from app.core.logging import get_logger
from app.integrations import email as email_integration
from app.models.organization import (
    Approval,
    ApprovalStatus,
    Comment,
    Organization,
    OrganizationMember,
    OrgRole,
    TeamResourceType,
    Visibility,
)
from app.models.project import Project, ProjectItem, ProjectItemType
from app.models.stack import Stack
from app.models.tool_run import ToolRun
from app.models.user import User
from app.services import email_templates, organization_service

logger = get_logger("collaboration")


async def _resource_org_id(db: AsyncSession, kind: TeamResourceType, resource_id: str) -> str:
    """The organization a resource's thread belongs to, or `NotFound`.

    Private resources have no thread — commenting is a team act, and a
    resource nobody shared has no team to talk in.
    """
    if kind is TeamResourceType.STACK:
        stack = await db.get(Stack, resource_id)
        if (
            stack is not None
            and stack.organization_id is not None
            and stack.visibility is not Visibility.PRIVATE
        ):
            return stack.organization_id
    elif kind is TeamResourceType.PROJECT:
        project = await db.get(Project, resource_id)
        if (
            project is not None
            and project.organization_id is not None
            and project.visibility is not Visibility.PRIVATE
        ):
            return project.organization_id
    else:
        run = await db.get(ToolRun, resource_id)
        if run is not None:
            # A run is commentable through the team-shared project that holds
            # it. The first qualifying project wins; a run in two shared
            # projects is one thread, not two.
            org_id = await db.scalar(
                select(Project.organization_id)
                .join(ProjectItem, ProjectItem.project_id == Project.id)
                .where(
                    ProjectItem.item_type == ProjectItemType.RUN,
                    ProjectItem.item_id == resource_id,
                    Project.organization_id.is_not(None),
                    Project.visibility != Visibility.PRIVATE,
                )
                .limit(1)
            )
            if org_id is not None:
                return str(org_id)
    raise NotFound("Nothing to discuss with that id.")


async def _context(
    db: AsyncSession,
    user: User,
    kind: TeamResourceType,
    resource_id: str,
    *,
    minimum: OrgRole,
) -> tuple[Organization, OrganizationMember]:
    """Resolve the resource's org, then the caller's membership and role.

    Order matters for what a probe learns: an outsider gets the resource 404
    (or the membership 404), never a 403 that confirms the resource exists.
    """
    org_id = await _resource_org_id(db, kind, resource_id)
    org, member = await organization_service.get_membership(db, user=user, organization_id=org_id)
    if not member.role.covers(minimum):
        raise Forbidden("Your role in this organization does not allow that.")
    return org, member


def _resource_href(kind: TeamResourceType, row: Any) -> str:
    if kind is TeamResourceType.STACK:
        return f"/stack-architect/my-stacks?stack={row.id}"
    if kind is TeamResourceType.PROJECT:
        return f"/projects/{row.id}"
    return f"/{row.workflow}?run={row.id}"


def _resource_label(kind: TeamResourceType, row: Any) -> str:
    if kind is TeamResourceType.RUN:
        return str(row.tool_slug)
    return str(row.name)


# ── Comments ─────────────────────────────────────────────────────────────────


async def list_comments(
    db: AsyncSession, user: User, *, kind: TeamResourceType, resource_id: str
) -> list[tuple[Comment, str | None]]:
    """The thread, flat and in order, with author names. Deleted comments come
    back as tombstones so replies keep their anchor; the router empties the
    body."""
    await _context(db, user, kind, resource_id, minimum=OrgRole.VIEWER)
    rows = (
        await db.execute(
            select(Comment, User.name)
            .join(User, User.id == Comment.author_id, isouter=True)
            .where(Comment.resource_type == kind, Comment.resource_id == resource_id)
            # Same-transaction rows share a Postgres now(); the UUIDv7 id
            # keeps a reply after the comment it answers.
            .order_by(Comment.created_at, Comment.id)
        )
    ).all()
    return [(comment, name) for comment, name in rows]


async def add_comment(
    db: AsyncSession,
    user: User,
    *,
    kind: TeamResourceType,
    resource_id: str,
    body: str,
    parent_id: str | None = None,
    mentions: list[str] | None = None,
) -> Comment:
    org, _ = await _context(db, user, kind, resource_id, minimum=OrgRole.MEMBER)

    if parent_id is not None:
        parent = await db.scalar(
            select(Comment).where(
                Comment.id == parent_id,
                Comment.resource_type == kind,
                Comment.resource_id == resource_id,
            )
        )
        if parent is None:
            raise NotFound("No comment with that id to reply to.")
        if parent.parent_id is not None:
            # One level of threading. A reply to a reply attaches to the root,
            # by refusal rather than by silent reparenting.
            raise ValidationFailed.on_field(
                "parent_id", "Replies cannot be nested — reply to the top-level comment."
            )

    comment = Comment(
        resource_type=kind,
        resource_id=resource_id,
        organization_id=org.id,
        author_id=user.id,
        body=body,
        parent_id=parent_id,
    )
    db.add(comment)
    await db.flush()
    await db.refresh(comment)

    await _notify_mentions(
        db, org, author=user, kind=kind, resource_id=resource_id, mentions=mentions or []
    )
    logger.info(
        "collaboration.comment_added",
        comment_id=comment.id,
        organization_id=org.id,
        resource=f"{kind.value}:{resource_id}",
    )
    return comment


async def _notify_mentions(
    db: AsyncSession,
    org: Organization,
    *,
    author: User,
    kind: TeamResourceType,
    resource_id: str,
    mentions: list[str],
) -> None:
    if not mentions:
        return

    members = {
        member_user.id: member_user
        for _, member_user in await organization_service.list_members(db, org)
    }
    unknown = [user_id for user_id in mentions if user_id not in members]
    if unknown:
        raise ValidationFailed.on_field(
            "mentions", "Mentions must name members of this organization."
        )

    model = {
        TeamResourceType.STACK: Stack,
        TeamResourceType.PROJECT: Project,
        TeamResourceType.RUN: ToolRun,
    }[kind]
    row = await db.get(model, resource_id)
    if row is None:  # pragma: no cover - the context check just resolved it
        return
    url = f"{settings.web_base_url}{_resource_href(kind, row)}"

    for user_id in dict.fromkeys(mentions):
        if user_id == author.id:
            continue
        target = members[user_id]
        await email_integration.send(
            email_templates.comment_mention(
                to=target.email,
                name=target.name,
                author_name=author.name,
                org_name=org.name,
                resource_label=_resource_label(kind, row),
                url=url,
            )
        )


async def _get_comment(db: AsyncSession, user: User, comment_id: str) -> tuple[Comment, OrgRole]:
    """The comment plus the caller's role in its org — 404 for outsiders."""
    comment = await db.get(Comment, comment_id)
    if comment is None:
        raise NotFound("No comment with that id.")
    _, member = await organization_service.get_membership(
        db, user=user, organization_id=comment.organization_id
    )
    return comment, member.role


async def edit_comment(db: AsyncSession, user: User, *, comment_id: str, body: str) -> Comment:
    comment, _ = await _get_comment(db, user, comment_id)
    if comment.deleted_at is not None:
        raise NotFound("No comment with that id.")
    if comment.author_id != user.id:
        raise Forbidden("Only the author can edit a comment.")
    comment.body = body
    await db.flush()
    await db.refresh(comment)
    return comment


async def delete_comment(db: AsyncSession, user: User, *, comment_id: str) -> Comment:
    """Soft delete, preserving thread structure. The author can delete their
    own; an admin can moderate anyone's."""
    comment, role = await _get_comment(db, user, comment_id)
    if comment.author_id != user.id and not role.covers(OrgRole.ADMIN):
        raise Forbidden("Only the author or an admin can delete a comment.")
    comment.deleted_at = utcnow()
    await db.flush()
    return comment


async def resolve_comment(
    db: AsyncSession, user: User, *, comment_id: str, resolved: bool
) -> Comment:
    comment, role = await _get_comment(db, user, comment_id)
    if not role.covers(OrgRole.MEMBER):
        raise Forbidden("Viewers cannot resolve comments.")
    if comment.parent_id is not None:
        raise ValidationFailed.on_field(
            "comment_id", "Resolve the top-level comment — replies follow it."
        )
    comment.resolved_at = utcnow() if resolved else None
    await db.flush()
    await db.refresh(comment)
    return comment


# ── Approvals ────────────────────────────────────────────────────────────────


async def list_approvals(
    db: AsyncSession, user: User, *, kind: TeamResourceType, resource_id: str
) -> list[tuple[Approval, str | None, str | None]]:
    """History, newest first, with requester and decider names."""
    await _context(db, user, kind, resource_id, minimum=OrgRole.VIEWER)
    requester = select(User.name).where(User.id == Approval.requested_by_user_id)
    decider = select(User.name).where(User.id == Approval.decided_by_user_id)
    rows = (
        await db.execute(
            select(
                Approval,
                requester.scalar_subquery(),
                decider.scalar_subquery(),
            )
            .where(Approval.resource_type == kind, Approval.resource_id == resource_id)
            # The id is a UUIDv7, so it breaks created_at ties in time order —
            # two requests in one transaction share a Postgres now().
            .order_by(Approval.created_at.desc(), Approval.id.desc())
        )
    ).all()
    return [(approval, requested_by, decided_by) for approval, requested_by, decided_by in rows]


async def request_approval(
    db: AsyncSession, user: User, *, kind: TeamResourceType, resource_id: str
) -> Approval:
    org, _ = await _context(db, user, kind, resource_id, minimum=OrgRole.MEMBER)

    pending = await db.scalar(
        select(Approval.id).where(
            Approval.resource_type == kind,
            Approval.resource_id == resource_id,
            Approval.status == ApprovalStatus.PENDING,
        )
    )
    if pending is not None:
        raise Conflict("An approval is already pending for this.")

    approval = Approval(
        resource_type=kind,
        resource_id=resource_id,
        organization_id=org.id,
        requested_by_user_id=user.id,
    )
    db.add(approval)
    await db.flush()
    await db.refresh(approval)
    logger.info(
        "collaboration.approval_requested",
        approval_id=approval.id,
        organization_id=org.id,
        resource=f"{kind.value}:{resource_id}",
    )
    return approval


async def decide_approval(
    db: AsyncSession,
    user: User,
    *,
    approval_id: str,
    approve: bool,
    note: str | None = None,
) -> Approval:
    approval = await db.get(Approval, approval_id)
    if approval is None:
        raise NotFound("No approval with that id.")
    _, member = await organization_service.get_membership(
        db, user=user, organization_id=approval.organization_id
    )
    if not member.role.covers(OrgRole.ADMIN):
        raise Forbidden("Only an admin or the owner can decide an approval.")
    if approval.status is not ApprovalStatus.PENDING:
        # The state machine has exactly one legal transition, out of pending.
        raise Conflict("This approval has already been decided.")

    approval.status = ApprovalStatus.APPROVED if approve else ApprovalStatus.REJECTED
    approval.decided_by_user_id = user.id
    approval.decision_note = note
    approval.decided_at = utcnow()
    await db.flush()
    await db.refresh(approval)
    logger.info(
        "collaboration.approval_decided",
        approval_id=approval.id,
        status=approval.status.value,
    )
    return approval
