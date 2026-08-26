"""Comments and approvals (M21).

No `organization_id` appears in these paths — the org is resolved from the
resource inside the service, and the caller's membership is checked there.
The router stays a translation layer, as everywhere else.
"""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter, Query

from app.api.deps import CurrentUser, Db
from app.core.responses import Envelope, ok
from app.models.organization import Approval, Comment, TeamResourceType
from app.models.user import User
from app.schemas.team import (
    ApprovalDecisionIn,
    ApprovalIn,
    ApprovalOut,
    CommentIn,
    CommentOut,
    CommentPatch,
    ResolveIn,
    ResourceTypeLiteral,
)
from app.services import collaboration_service

router = APIRouter(tags=["collaboration"])


def _comment_out(comment: Comment, author_name: str | None, user: User) -> CommentOut:
    deleted = comment.deleted_at is not None
    return CommentOut(
        id=comment.id,
        resource_type=comment.resource_type.value,
        resource_id=comment.resource_id,
        author_id=None if deleted else comment.author_id,
        author_name=None if deleted else author_name,
        # The tombstone: structure survives, words do not.
        body="" if deleted else comment.body,
        parent_id=comment.parent_id,
        resolved_at=comment.resolved_at,
        deleted=deleted,
        is_yours=not deleted and comment.author_id == user.id,
        created_at=comment.created_at,
        updated_at=comment.updated_at,
    )


def _approval_out(
    approval: Approval, requested_by: str | None, decided_by: str | None
) -> ApprovalOut:
    return ApprovalOut(
        id=approval.id,
        resource_type=approval.resource_type.value,
        resource_id=approval.resource_id,
        status=approval.status.value,
        requested_by=requested_by,
        decided_by=decided_by,
        decision_note=approval.decision_note,
        requested_at=approval.created_at,
        decided_at=approval.decided_at,
    )


# ── Comments ─────────────────────────────────────────────────────────────────


@router.get("/comments", response_model=Envelope[list[CommentOut]], name="list_comments")
async def list_comments(
    db: Db,
    user: CurrentUser,
    resource_type: ResourceTypeLiteral = Query(),
    resource_id: str = Query(min_length=1, max_length=64),
) -> dict[str, Any]:
    rows = await collaboration_service.list_comments(
        db, user, kind=TeamResourceType(resource_type), resource_id=resource_id
    )
    return ok([_comment_out(comment, name, user) for comment, name in rows])


@router.post(
    "/comments", response_model=Envelope[CommentOut], name="create_comment", status_code=201
)
async def create_comment(db: Db, user: CurrentUser, payload: CommentIn) -> dict[str, Any]:
    comment = await collaboration_service.add_comment(
        db,
        user,
        kind=TeamResourceType(payload.resource_type),
        resource_id=payload.resource_id,
        body=payload.body,
        parent_id=payload.parent_id,
        mentions=payload.mentions,
    )
    return ok(_comment_out(comment, user.name, user))


@router.patch("/comments/{comment_id}", response_model=Envelope[CommentOut], name="update_comment")
async def update_comment(
    db: Db, user: CurrentUser, comment_id: str, payload: CommentPatch
) -> dict[str, Any]:
    comment = await collaboration_service.edit_comment(
        db, user, comment_id=comment_id, body=payload.body
    )
    return ok(_comment_out(comment, user.name, user))


@router.delete("/comments/{comment_id}", status_code=204, name="delete_comment")
async def delete_comment(db: Db, user: CurrentUser, comment_id: str) -> None:
    await collaboration_service.delete_comment(db, user, comment_id=comment_id)


@router.post(
    "/comments/{comment_id}/resolve",
    response_model=Envelope[CommentOut],
    name="resolve_comment",
)
async def resolve_comment(
    db: Db, user: CurrentUser, comment_id: str, payload: ResolveIn
) -> dict[str, Any]:
    comment = await collaboration_service.resolve_comment(
        db, user, comment_id=comment_id, resolved=payload.resolved
    )
    author = await db.get(User, comment.author_id) if comment.author_id else None
    return ok(_comment_out(comment, author.name if author else None, user))


# ── Approvals ────────────────────────────────────────────────────────────────


@router.get("/approvals", response_model=Envelope[list[ApprovalOut]], name="list_approvals")
async def list_approvals(
    db: Db,
    user: CurrentUser,
    resource_type: ResourceTypeLiteral = Query(),
    resource_id: str = Query(min_length=1, max_length=64),
) -> dict[str, Any]:
    rows = await collaboration_service.list_approvals(
        db, user, kind=TeamResourceType(resource_type), resource_id=resource_id
    )
    return ok([_approval_out(*row) for row in rows])


@router.post(
    "/approvals", response_model=Envelope[ApprovalOut], name="request_approval", status_code=201
)
async def request_approval(db: Db, user: CurrentUser, payload: ApprovalIn) -> dict[str, Any]:
    approval = await collaboration_service.request_approval(
        db, user, kind=TeamResourceType(payload.resource_type), resource_id=payload.resource_id
    )
    return ok(_approval_out(approval, user.name, None))


@router.patch(
    "/approvals/{approval_id}", response_model=Envelope[ApprovalOut], name="decide_approval"
)
async def decide_approval(
    db: Db, user: CurrentUser, approval_id: str, payload: ApprovalDecisionIn
) -> dict[str, Any]:
    approval = await collaboration_service.decide_approval(
        db,
        user,
        approval_id=approval_id,
        approve=payload.action == "approve",
        note=payload.note,
    )
    requester = (
        await db.get(User, approval.requested_by_user_id) if approval.requested_by_user_id else None
    )
    return ok(_approval_out(approval, requester.name if requester else None, user.name))
