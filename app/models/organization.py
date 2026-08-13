"""Organizations, membership, invitations, comments, approvals (M21).

Five tables, and the decisions worth stating up front:

**Exactly one owner, enforced by the database.** A partial unique index on
`organization_members` allows one `owner` row per organization. Ownership
transfer is an explicit two-row swap inside one transaction — an organization
whose owner left the company and cannot be replaced is a support ticket that
takes days, so the invariant lives where a bug cannot bend it.

**`organizations.owner_id` is a denormalised convenience.** The membership row
is the authority; the column exists so "who owns this" is a read, not a join,
and it changes in the same transaction as the membership swap.

**Invitation tokens are hashed at rest.** Same rule as auth tokens (D-14): the
plaintext is shown once, in the email, and a database dump must not yield
usable invites. Share links are the deliberate exception, not the rule.

**Comments and approvals point at resources polymorphically** — same shape and
same trade as `project_items`: no FK the database can enforce, in exchange for
not growing a nullable column per commentable thing. The service resolves the
target and 404s on a dangling id.

**Personal work stays personal.** `organization_id` on stacks and projects is
nullable, and an organization's deletion SET NULLs it — the rows revert to
private personal work rather than vanishing with the org.
"""

from __future__ import annotations

import enum
from datetime import datetime
from typing import Any

from sqlalchemy import (
    CheckConstraint,
    DateTime,
    Enum,
    ForeignKey,
    Index,
    Integer,
    String,
    Text,
    UniqueConstraint,
    text,
)
from sqlalchemy.dialects.postgresql import CITEXT, JSONB
from sqlalchemy.orm import Mapped, mapped_column

from app.core.database import Base, SoftDeleteMixin, TimestampMixin, new_id
from app.models.user import Plan


class OrgRole(str, enum.Enum):
    """Organization role. Independent of the platform role on `users.role`."""

    OWNER = "owner"
    ADMIN = "admin"
    MEMBER = "member"
    VIEWER = "viewer"

    @property
    def rank(self) -> int:
        return _ROLE_RANK[self]

    def covers(self, minimum: OrgRole) -> bool:
        """True when this role has at least `minimum`'s privileges."""
        return self.rank >= minimum.rank


_ROLE_RANK = {
    OrgRole.VIEWER: 0,
    OrgRole.MEMBER: 1,
    OrgRole.ADMIN: 2,
    OrgRole.OWNER: 3,
}


class Visibility(str, enum.Enum):
    """Who can see a stack or a project.

    `private` is the default everywhere — joining an organization must not
    silently publish existing work, and new work becomes team work by choice.
    """

    PRIVATE = "private"
    TEAM = "team"
    PUBLIC = "public"


class TeamResourceType(str, enum.Enum):
    """What a comment or an approval is anchored to."""

    STACK = "stack"
    RUN = "run"
    PROJECT = "project"


class ApprovalStatus(str, enum.Enum):
    PENDING = "pending"
    APPROVED = "approved"
    REJECTED = "rejected"


class Organization(Base, TimestampMixin, SoftDeleteMixin):
    __tablename__ = "organizations"

    id: Mapped[str] = mapped_column(String(64), primary_key=True, default=lambda: new_id("org"))

    name: Mapped[str] = mapped_column(String(160), nullable=False)
    slug: Mapped[str] = mapped_column(String(80), unique=True, nullable=False)

    #: Denormalised from the membership row with `role = 'owner'` — that row is
    #: the authority, and both change in one transaction on transfer. RESTRICT
    #: because deleting a user who still owns an organization would orphan the
    #: org; transfer first, then delete.
    owner_id: Mapped[str] = mapped_column(
        String(64), ForeignKey("users.id", ondelete="RESTRICT"), nullable=False
    )

    #: Denormalised from the organization's subscription, same as `users.plan`.
    plan: Mapped[Plan] = mapped_column(
        Enum(Plan, name="plan", values_callable=lambda e: [m.value for m in e], create_type=False),
        default=Plan.FREE,
        server_default=Plan.FREE.value,
        nullable=False,
    )
    #: Denormalised from `subscriptions.seats`. The effective seat limit is
    #: resolved by FeatureService (plan floor vs purchased), never read raw.
    seats_purchased: Mapped[int] = mapped_column(
        Integer, nullable=False, default=1, server_default=text("1")
    )

    #: `approved_tools` (list of catalog slugs), `require_approval` (bool),
    #: `default_visibility` (a `Visibility` value). Absent keys mean the
    #: defaults; OrganizationSettings in the schema layer is the one reader.
    settings: Mapped[dict[str, Any]] = mapped_column(JSONB, nullable=False, default=dict)

    __table_args__ = (
        Index("ix_organizations_active", "deleted_at", postgresql_where=text("deleted_at IS NULL")),
    )

    def __repr__(self) -> str:  # pragma: no cover
        return f"<Organization {self.id} {self.slug}>"


class OrganizationMember(Base, TimestampMixin):
    """One user's seat in one organization. `created_at` is the join date."""

    __tablename__ = "organization_members"

    id: Mapped[str] = mapped_column(String(64), primary_key=True, default=lambda: new_id("mem"))

    organization_id: Mapped[str] = mapped_column(
        String(64), ForeignKey("organizations.id", ondelete="CASCADE"), nullable=False
    )
    user_id: Mapped[str] = mapped_column(
        String(64), ForeignKey("users.id", ondelete="CASCADE"), nullable=False
    )

    role: Mapped[OrgRole] = mapped_column(
        Enum(OrgRole, name="org_role", values_callable=lambda e: [m.value for m in e]),
        nullable=False,
    )

    invited_by_user_id: Mapped[str | None] = mapped_column(
        String(64), ForeignKey("users.id", ondelete="SET NULL")
    )

    __table_args__ = (
        UniqueConstraint(
            "organization_id", "user_id", name="uq_organization_members_organization_id_user_id"
        ),
        # Exactly one owner. The invariant the whole role model leans on, held
        # at the database level so no code path can create a second.
        Index(
            "uq_organization_members_one_owner",
            "organization_id",
            unique=True,
            postgresql_where=text("role = 'owner'"),
        ),
        Index("ix_organization_members_user_id", "user_id"),
    )


class Invitation(Base, TimestampMixin):
    """An email invited to an organization, with a role waiting for it.

    Open means `accepted_at IS NULL AND revoked_at IS NULL` — one open invite
    per (organization, email), enforced partially so a revoked or accepted
    invite never blocks a re-invite.
    """

    __tablename__ = "invitations"

    id: Mapped[str] = mapped_column(String(64), primary_key=True, default=lambda: new_id("inv"))

    organization_id: Mapped[str] = mapped_column(
        String(64), ForeignKey("organizations.id", ondelete="CASCADE"), nullable=False
    )

    email: Mapped[str] = mapped_column(CITEXT, nullable=False)
    role: Mapped[OrgRole] = mapped_column(
        Enum(
            OrgRole,
            name="org_role",
            values_callable=lambda e: [m.value for m in e],
            create_type=False,
        ),
        nullable=False,
    )

    #: SHA-256 of the token in the email link. Plaintext exists only there.
    token_hash: Mapped[str] = mapped_column(String(64), unique=True, nullable=False)

    invited_by_user_id: Mapped[str | None] = mapped_column(
        String(64), ForeignKey("users.id", ondelete="SET NULL")
    )

    expires_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    accepted_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    accepted_by_user_id: Mapped[str | None] = mapped_column(
        String(64), ForeignKey("users.id", ondelete="SET NULL")
    )
    revoked_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))

    __table_args__ = (
        # Ownership is transferred, never granted by invite.
        CheckConstraint("role <> 'owner'", name="no_owner_invites"),
        Index(
            "uq_invitations_organization_id_email_open",
            "organization_id",
            "email",
            unique=True,
            postgresql_where=text("accepted_at IS NULL AND revoked_at IS NULL"),
        ),
        Index("ix_invitations_organization_id", "organization_id"),
    )

    @property
    def is_open(self) -> bool:
        return self.accepted_at is None and self.revoked_at is None


class Comment(Base, TimestampMixin, SoftDeleteMixin):
    """One comment on a team resource.

    One level of threading: a comment either has no parent or its parent has
    none — the service enforces the depth, the FK only enforces existence.
    Soft delete keeps thread structure: a deleted comment renders as a
    tombstone so its replies keep their context.
    """

    __tablename__ = "comments"

    id: Mapped[str] = mapped_column(String(64), primary_key=True, default=lambda: new_id("cmt"))

    resource_type: Mapped[TeamResourceType] = mapped_column(
        Enum(
            TeamResourceType,
            name="team_resource_type",
            values_callable=lambda e: [m.value for m in e],
        ),
        nullable=False,
    )
    #: Polymorphic — points at a row in whichever table `resource_type` names.
    resource_id: Mapped[str] = mapped_column(String(64), nullable=False)

    organization_id: Mapped[str] = mapped_column(
        String(64), ForeignKey("organizations.id", ondelete="CASCADE"), nullable=False
    )
    #: SET NULL, not CASCADE: a purged author must not take the thread's
    #: replies down with them.
    author_id: Mapped[str | None] = mapped_column(
        String(64), ForeignKey("users.id", ondelete="SET NULL")
    )

    body: Mapped[str] = mapped_column(Text, nullable=False)

    parent_id: Mapped[str | None] = mapped_column(
        String(64), ForeignKey("comments.id", ondelete="CASCADE")
    )

    resolved_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))

    __table_args__ = (
        Index(
            "ix_comments_resource_type_resource_id_created_at",
            "resource_type",
            "resource_id",
            "created_at",
        ),
    )


class Approval(Base, TimestampMixin):
    """A lightweight gate on a team resource. `created_at` is the request time.

    One pending approval per resource, held partially — a decided approval is
    history and must not block the next request.
    """

    __tablename__ = "approvals"

    id: Mapped[str] = mapped_column(String(64), primary_key=True, default=lambda: new_id("apr"))

    resource_type: Mapped[TeamResourceType] = mapped_column(
        Enum(
            TeamResourceType,
            name="team_resource_type",
            values_callable=lambda e: [m.value for m in e],
            create_type=False,
        ),
        nullable=False,
    )
    resource_id: Mapped[str] = mapped_column(String(64), nullable=False)

    organization_id: Mapped[str] = mapped_column(
        String(64), ForeignKey("organizations.id", ondelete="CASCADE"), nullable=False
    )
    requested_by_user_id: Mapped[str | None] = mapped_column(
        String(64), ForeignKey("users.id", ondelete="SET NULL")
    )
    decided_by_user_id: Mapped[str | None] = mapped_column(
        String(64), ForeignKey("users.id", ondelete="SET NULL")
    )

    status: Mapped[ApprovalStatus] = mapped_column(
        Enum(
            ApprovalStatus,
            name="approval_status",
            values_callable=lambda e: [m.value for m in e],
        ),
        default=ApprovalStatus.PENDING,
        server_default=ApprovalStatus.PENDING.value,
        nullable=False,
    )
    decision_note: Mapped[str | None] = mapped_column(Text)
    decided_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))

    __table_args__ = (
        Index(
            "uq_approvals_resource_pending",
            "resource_type",
            "resource_id",
            unique=True,
            postgresql_where=text("status = 'pending'"),
        ),
        Index("ix_approvals_organization_id_status", "organization_id", "status"),
    )


__all__ = [
    "Approval",
    "ApprovalStatus",
    "Comment",
    "Invitation",
    "OrgRole",
    "Organization",
    "OrganizationMember",
    "TeamResourceType",
    "Visibility",
]
