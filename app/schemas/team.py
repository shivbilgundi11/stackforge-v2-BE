"""Team wire shapes (M21)."""

from __future__ import annotations

from datetime import datetime
from typing import Literal

from pydantic import BaseModel, EmailStr, Field

OrgRoleLiteral = Literal["owner", "admin", "member", "viewer"]
#: Roles that can be granted. Ownership moves only by explicit transfer.
GrantableRole = Literal["admin", "member", "viewer"]
VisibilityLiteral = Literal["private", "team", "public"]


class OrganizationIn(BaseModel):
    name: str = Field(min_length=1, max_length=160)


class OrganizationPatch(BaseModel):
    name: str | None = Field(default=None, min_length=1, max_length=160)


class SeatsOut(BaseModel):
    used: int
    #: `null` is unlimited (M20's convention: never a sentinel).
    limit: int | None
    purchased: int


class OrganizationSettingsOut(BaseModel):
    approved_tools: list[str]
    require_approval: bool
    default_visibility: VisibilityLiteral


class OrganizationSettingsPatch(BaseModel):
    approved_tools: list[str] | None = Field(default=None, max_length=200)
    require_approval: bool | None = None
    default_visibility: VisibilityLiteral | None = None


class OrganizationOut(BaseModel):
    id: str
    name: str
    slug: str
    plan: str
    #: The caller's role — every org payload is read through a membership.
    role: OrgRoleLiteral
    seats: SeatsOut
    settings: OrganizationSettingsOut
    created_at: datetime


class MemberOut(BaseModel):
    id: str
    user_id: str
    name: str
    email: str
    avatar_url: str | None
    role: OrgRoleLiteral
    is_current_user: bool
    joined_at: datetime


class MemberPatch(BaseModel):
    role: GrantableRole


class TransferOwnershipIn(BaseModel):
    membership_id: str = Field(min_length=1, max_length=64)


class InvitationIn(BaseModel):
    email: EmailStr
    role: GrantableRole = "member"


class InvitationOut(BaseModel):
    id: str
    email: str
    role: OrgRoleLiteral
    invited_by: str | None
    expires_at: datetime
    created_at: datetime


class InvitePreviewOut(BaseModel):
    """What the accept page shows before anyone commits. Possession of the
    token is the credential; the invited email comes back so signup can
    prefill and lock it (acceptance path 3)."""

    organization_name: str
    email: str
    role: OrgRoleLiteral
    invited_by: str | None
    expires_at: datetime


class AcceptInvitationIn(BaseModel):
    token: str = Field(min_length=16, max_length=256)


class AcceptInvitationOut(BaseModel):
    organization: OrganizationOut


# ── Comments ─────────────────────────────────────────────────────────────────

ResourceTypeLiteral = Literal["stack", "run", "project"]


class CommentIn(BaseModel):
    resource_type: ResourceTypeLiteral
    resource_id: str = Field(min_length=1, max_length=64)
    body: str = Field(min_length=1, max_length=5000)
    #: One level of threading — a reply's parent must itself be a root.
    parent_id: str | None = Field(default=None, max_length=64)
    #: User ids to notify. Validated against the organization's membership —
    #: a mention of an outsider is refused, not silently dropped.
    mentions: list[str] = Field(default_factory=list, max_length=20)


class CommentPatch(BaseModel):
    body: str = Field(min_length=1, max_length=5000)


class CommentOut(BaseModel):
    id: str
    resource_type: ResourceTypeLiteral
    resource_id: str
    author_id: str | None
    author_name: str | None
    #: Empty string when deleted — the tombstone keeps replies anchored.
    body: str
    parent_id: str | None
    resolved_at: datetime | None
    deleted: bool
    is_yours: bool
    created_at: datetime
    updated_at: datetime


class ResolveIn(BaseModel):
    resolved: bool = True


# ── Approvals ────────────────────────────────────────────────────────────────


class ApprovalIn(BaseModel):
    resource_type: ResourceTypeLiteral
    resource_id: str = Field(min_length=1, max_length=64)


class ApprovalDecisionIn(BaseModel):
    action: Literal["approve", "reject"]
    note: str | None = Field(default=None, max_length=2000)


class ApprovalOut(BaseModel):
    id: str
    resource_type: ResourceTypeLiteral
    resource_id: str
    status: Literal["pending", "approved", "rejected"]
    requested_by: str | None
    decided_by: str | None
    decision_note: str | None
    requested_at: datetime
    decided_at: datetime | None
