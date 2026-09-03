"""Organizations, membership, and invitations (M21).

The rules that keep this module honest:

**Membership resolution is the only door.** Every org-scoped read starts at
`get_membership`, and a user with no membership row gets the same `NotFound`
as a nonexistent organization — cross-org probing learns nothing.

**Exactly one owner, and the database holds the line.** `transfer_ownership`
is the only path that moves the role, demoting the old owner and promoting the
new one in one transaction; the partial unique index rejects any code path
that tries to mint a second.

**Seats are checked at send and again at accept.** The send check stops
inviting into a team that is already full; the accept check is the one that
matters, because invitations do not reserve seats — a team with five seats
that invited six people gets a clear `SEATS_EXCEEDED` at the fifth accept,
not a silent overage.

**Invite tokens are hashed, single-purpose, and time-boxed.** An expired or
revoked token is a 404, indistinguishable from a token that never existed.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import timedelta

from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.api.deps import Identity
from app.core.config import settings
from app.core.database import utcnow
from app.core.errors import Conflict, NotFound, SeatsExceeded, ValidationFailed
from app.core.logging import get_logger
from app.data.plans import Feature
from app.integrations import email as email_integration
from app.models.billing import Metric, PlanQuota, Subscription, SubscriptionStatus
from app.models.organization import (
    Invitation,
    Organization,
    OrganizationMember,
    OrgRole,
    Visibility,
)
from app.models.user import Plan, User
from app.services import email_templates, token_service

logger = get_logger("organizations")

_SLUG_RE = re.compile(r"[^a-z0-9]+")

#: Settings keys, with their defaults. `organizations.settings` is JSONB and
#: absent keys mean these — one reader, one spelling.
DEFAULT_SETTINGS = {
    "approved_tools": [],
    "require_approval": False,
    "default_visibility": Visibility.PRIVATE.value,
}


@dataclass(frozen=True)
class OrgSettings:
    approved_tools: list[str]
    require_approval: bool
    default_visibility: Visibility


def settings_of(org: Organization) -> OrgSettings:
    raw = org.settings or {}
    return OrgSettings(
        approved_tools=list(raw.get("approved_tools") or []),
        require_approval=bool(raw.get("require_approval", False)),
        default_visibility=Visibility(raw.get("default_visibility") or Visibility.PRIVATE.value),
    )


def _identity(user: User) -> Identity:
    return Identity(user=user, session_id=None)


# ── Membership resolution ────────────────────────────────────────────────────


async def get_membership(
    db: AsyncSession, *, user: User, organization_id: str
) -> tuple[Organization, OrganizationMember]:
    """The door. No membership and no such org are the same `NotFound`."""
    row = (
        await db.execute(
            select(Organization, OrganizationMember)
            .join(
                OrganizationMember,
                OrganizationMember.organization_id == Organization.id,
            )
            .where(
                Organization.id == organization_id,
                Organization.deleted_at.is_(None),
                OrganizationMember.user_id == user.id,
            )
        )
    ).first()
    if row is None:
        raise NotFound("No organization with that id.")
    org, member = row
    return org, member


async def list_for(db: AsyncSession, user: User) -> list[tuple[Organization, OrganizationMember]]:
    rows = (
        await db.execute(
            select(Organization, OrganizationMember)
            .join(
                OrganizationMember,
                OrganizationMember.organization_id == Organization.id,
            )
            .where(
                OrganizationMember.user_id == user.id,
                Organization.deleted_at.is_(None),
            )
            .order_by(Organization.created_at)
        )
    ).all()
    return [(org, member) for org, member in rows]


# ── Organizations ────────────────────────────────────────────────────────────


async def _unique_slug(db: AsyncSession, name: str) -> str:
    base = _SLUG_RE.sub("-", name.lower()).strip("-")[:60] or "team"
    slug = base
    suffix = 2
    while await db.scalar(select(Organization.id).where(Organization.slug == slug)):
        slug = f"{base}-{suffix}"
        suffix += 1
    return slug


async def create(db: AsyncSession, user: User, *, name: str) -> Organization:
    """Create an organization with the caller as its owner.

    Requires the Team feature on the caller's plan. If the caller has a live
    per-seat subscription, it becomes the *organization's* subscription — the
    org is the thing being paid for from here on, and seat changes adjust this
    subscription's quantity.
    """
    from app.services import billing_service, feature_service

    feature_service.require(_identity(user), Feature.TEAM_WORKSPACE)

    org = Organization(
        name=name,
        slug=await _unique_slug(db, name),
        owner_id=user.id,
        plan=user.plan,
        settings=dict(DEFAULT_SETTINGS),
    )
    db.add(org)
    await db.flush()

    db.add(
        OrganizationMember(
            organization_id=org.id,
            user_id=user.id,
            role=OrgRole.OWNER,
        )
    )

    subscription = await billing_service.get_subscription(db, user)
    if subscription is not None and subscription.is_paid and subscription.plan is Plan.TEAM:
        subscription.user_id = None
        subscription.organization_id = org.id
        org.plan = subscription.plan
        org.seats_purchased = subscription.seats
        await db.flush()
        # The user's plan now flows from membership rather than from a
        # personal subscription.
        await billing_service.sync_user_plan(db, user)

    await db.flush()
    await db.refresh(org)
    logger.info("organizations.created", organization_id=org.id, owner_id=user.id)
    return org


async def update(db: AsyncSession, org: Organization, *, name: str | None = None) -> Organization:
    if name is not None:
        org.name = name
    await db.flush()
    await db.refresh(org)
    return org


async def update_settings(
    db: AsyncSession,
    org: Organization,
    *,
    approved_tools: list[str] | None = None,
    require_approval: bool | None = None,
    default_visibility: Visibility | None = None,
) -> Organization:
    current = dict(org.settings or {})

    if approved_tools is not None:
        await _validate_tool_slugs(db, approved_tools)
        current["approved_tools"] = sorted(set(approved_tools))
    if require_approval is not None:
        current["require_approval"] = require_approval
    if default_visibility is not None:
        current["default_visibility"] = default_visibility.value

    # Reassigned, not mutated — SQLAlchemy does not track in-place JSONB edits.
    org.settings = current
    await db.flush()
    await db.refresh(org)
    return org


async def _validate_tool_slugs(db: AsyncSession, slugs: list[str]) -> None:
    """An allowlist with a typo in it silently approves nothing — refuse it."""
    from app.models.catalog import Tool

    if not slugs:
        return
    known = set((await db.execute(select(Tool.slug).where(Tool.slug.in_(slugs)))).scalars().all())
    unknown = [slug for slug in slugs if slug not in known]
    if unknown:
        raise ValidationFailed.on_field("approved_tools", f"Unknown tool slug: {unknown[0]}")


async def delete(db: AsyncSession, org: Organization) -> None:
    """Soft delete. Team-shared work reverts to its authors as personal work
    (the FKs SET NULL on hard delete; membership resolution stops at
    `deleted_at` immediately). A live subscription is set to cancel at period
    end rather than cut off — the time is paid for."""
    from app.services import billing_service

    org.deleted_at = utcnow()

    subscription = await db.scalar(
        select(Subscription).where(
            Subscription.organization_id == org.id,
            Subscription.status != SubscriptionStatus.CANCELED,
        )
    )
    if subscription is not None:
        await billing_service.cancel_org_subscription(db, subscription)

    members = (
        (
            await db.execute(
                select(User)
                .join(OrganizationMember, OrganizationMember.user_id == User.id)
                .where(OrganizationMember.organization_id == org.id)
            )
        )
        .scalars()
        .all()
    )
    await db.flush()
    for member_user in members:
        await billing_service.sync_user_plan(db, member_user)

    logger.info("organizations.deleted", organization_id=org.id)


# ── Members ──────────────────────────────────────────────────────────────────


async def list_members(
    db: AsyncSession, org: Organization
) -> list[tuple[OrganizationMember, User]]:
    rows = (
        await db.execute(
            select(OrganizationMember, User)
            .join(User, User.id == OrganizationMember.user_id)
            .where(OrganizationMember.organization_id == org.id)
            .order_by(OrganizationMember.created_at)
        )
    ).all()
    return [(member, user) for member, user in rows]


async def _get_member(
    db: AsyncSession, org: Organization, membership_id: str
) -> OrganizationMember:
    member = await db.scalar(
        select(OrganizationMember).where(
            OrganizationMember.id == membership_id,
            OrganizationMember.organization_id == org.id,
        )
    )
    if member is None:
        raise NotFound("No member with that id.")
    return member


async def change_role(
    db: AsyncSession,
    org: Organization,
    *,
    membership_id: str,
    role: OrgRole,
) -> OrganizationMember:
    if role is OrgRole.OWNER:
        raise ValidationFailed.on_field(
            "role", "Ownership is transferred explicitly, not assigned."
        )
    member = await _get_member(db, org, membership_id)
    if member.role is OrgRole.OWNER:
        raise ValidationFailed.on_field(
            "role", "The owner's role changes only by ownership transfer."
        )
    member.role = role
    await db.flush()
    logger.info(
        "organizations.role_changed",
        organization_id=org.id,
        membership_id=member.id,
        role=role.value,
    )
    return member


async def remove_member(db: AsyncSession, org: Organization, *, membership_id: str) -> None:
    from app.services import billing_service

    member = await _get_member(db, org, membership_id)
    if member.role is OrgRole.OWNER:
        raise ValidationFailed.on_field(
            "membership_id", "Transfer ownership before removing the owner."
        )
    user = await db.get(User, member.user_id)
    await db.delete(member)
    await db.flush()
    if user is not None:
        await billing_service.sync_user_plan(db, user)
    logger.info("organizations.member_removed", organization_id=org.id, membership_id=membership_id)


async def transfer_ownership(
    db: AsyncSession,
    org: Organization,
    *,
    owner_member: OrganizationMember,
    to_membership_id: str,
) -> OrganizationMember:
    """Move the owner role in one transaction.

    The old owner becomes an admin — transfer is a handover, not an exit. The
    demotion flushes before the promotion so the one-owner index never sees
    two.
    """
    successor = await _get_member(db, org, to_membership_id)
    if successor.id == owner_member.id:
        raise ValidationFailed.on_field("membership_id", "You already own this organization.")

    owner_member.role = OrgRole.ADMIN
    await db.flush()
    successor.role = OrgRole.OWNER
    org.owner_id = successor.user_id
    await db.flush()

    logger.info(
        "organizations.ownership_transferred",
        organization_id=org.id,
        from_user_id=owner_member.user_id,
        to_user_id=successor.user_id,
    )
    return successor


# ── Seats ────────────────────────────────────────────────────────────────────


async def seat_limit(db: AsyncSession, org: Organization) -> int | None:
    """The plan's floor or the purchased count, whichever is higher.

    `None` is unlimited (Enterprise). Reading the floor from `plan_quotas`
    keeps this the same number the pricing page shows.
    """
    floor = await db.scalar(
        select(PlanQuota.limit_value).where(
            PlanQuota.plan == org.plan,
            PlanQuota.metric == Metric.SEATS,
        )
    )
    if floor is None:
        return None
    return max(int(floor), org.seats_purchased)


async def seats_used(db: AsyncSession, org: Organization) -> int:
    return int(
        await db.scalar(
            select(func.count())
            .select_from(OrganizationMember)
            .where(OrganizationMember.organization_id == org.id)
        )
        or 0
    )


async def _require_seat(db: AsyncSession, org: Organization) -> None:
    limit = await seat_limit(db, org)
    if limit is None:
        return
    used = await seats_used(db, org)
    if used >= limit:
        raise SeatsExceeded(details={"limit": limit, "used": used})


# ── Invitations ──────────────────────────────────────────────────────────────


async def invite(
    db: AsyncSession,
    org: Organization,
    inviter: User,
    *,
    email: str,
    role: OrgRole,
) -> Invitation:
    if role is OrgRole.OWNER:
        raise ValidationFailed.on_field(
            "role", "Ownership is transferred explicitly, not granted by invite."
        )

    # Seat check at send: a full team cannot invite anyone. Invitations do not
    # reserve seats — the authoritative check is at accept.
    await _require_seat(db, org)

    already = await db.scalar(
        select(OrganizationMember.id)
        .join(User, User.id == OrganizationMember.user_id)
        .where(
            OrganizationMember.organization_id == org.id,
            func.lower(User.email) == email.lower(),
        )
    )
    if already is not None:
        raise Conflict("That person is already a member.")

    open_invite = await db.scalar(
        select(Invitation.id).where(
            Invitation.organization_id == org.id,
            func.lower(Invitation.email) == email.lower(),
            Invitation.accepted_at.is_(None),
            Invitation.revoked_at.is_(None),
        )
    )
    if open_invite is not None:
        raise Conflict("There is already an open invitation for that address.")

    token = token_service.generate_secret()
    invitation = Invitation(
        organization_id=org.id,
        email=email,
        role=role,
        token_hash=token_service.hash_secret(token),
        invited_by_user_id=inviter.id,
        expires_at=utcnow() + timedelta(days=settings.invite_ttl_days),
    )
    db.add(invitation)
    await db.flush()
    await db.refresh(invitation)

    await email_integration.send(
        email_templates.organization_invite(
            to=email,
            org_name=org.name,
            inviter_name=inviter.name,
            role=role.value,
            token=token,
        )
    )
    logger.info(
        "organizations.invited",
        organization_id=org.id,
        invitation_id=invitation.id,
        role=role.value,
    )
    return invitation


async def list_invitations(db: AsyncSession, org: Organization) -> list[Invitation]:
    """Open invitations only. Accepted and revoked rows are history, and the
    pending-invites page is a work list, not a log."""
    return list(
        (
            await db.execute(
                select(Invitation)
                .where(
                    Invitation.organization_id == org.id,
                    Invitation.accepted_at.is_(None),
                    Invitation.revoked_at.is_(None),
                )
                .order_by(Invitation.created_at.desc())
            )
        )
        .scalars()
        .all()
    )


async def _get_invitation(db: AsyncSession, org: Organization, invitation_id: str) -> Invitation:
    invitation = await db.scalar(
        select(Invitation).where(
            Invitation.id == invitation_id,
            Invitation.organization_id == org.id,
        )
    )
    if invitation is None or not invitation.is_open:
        raise NotFound("No open invitation with that id.")
    return invitation


async def resend(
    db: AsyncSession, org: Organization, inviter: User, *, invitation_id: str
) -> Invitation:
    """Re-send with a fresh token and a fresh clock. The old link dies — two
    live links to the same seat is one more than anyone needs."""
    invitation = await _get_invitation(db, org, invitation_id)

    token = token_service.generate_secret()
    invitation.token_hash = token_service.hash_secret(token)
    invitation.expires_at = utcnow() + timedelta(days=settings.invite_ttl_days)
    await db.flush()

    await email_integration.send(
        email_templates.organization_invite(
            to=invitation.email,
            org_name=org.name,
            inviter_name=inviter.name,
            role=invitation.role.value,
            token=token,
        )
    )
    return invitation


async def revoke(db: AsyncSession, org: Organization, *, invitation_id: str) -> None:
    invitation = await _get_invitation(db, org, invitation_id)
    invitation.revoked_at = utcnow()
    await db.flush()
    logger.info("organizations.invite_revoked", organization_id=org.id, invitation_id=invitation_id)


async def _open_invitation_by_token(db: AsyncSession, token: str) -> Invitation:
    """Missing, revoked, accepted, and expired are all the same 404."""
    invitation = await db.scalar(
        select(Invitation).where(Invitation.token_hash == token_service.hash_secret(token))
    )
    if invitation is None or not invitation.is_open or invitation.expires_at <= utcnow():
        raise NotFound("This invitation is invalid or has expired.")
    return invitation


async def preview(db: AsyncSession, *, token: str) -> tuple[Invitation, Organization]:
    """What the accept page shows before anyone commits: the org's name, the
    invited address, and the role. Possession of the token is the credential."""
    invitation = await _open_invitation_by_token(db, token)
    org = await db.get(Organization, invitation.organization_id)
    if org is None or org.deleted_at is not None:
        raise NotFound("This invitation is invalid or has expired.")
    return invitation, org


async def accept(
    db: AsyncSession, *, token: str, user: User
) -> tuple[Organization, OrganizationMember]:
    """All three acceptance paths end here with a signed-in user and a token.

    The membership goes to the invited address only. A signed-in account with
    a different email gets a clear conflict rather than a silent join as the
    wrong person — the recipient can sign out and use the right account.
    """
    from app.services import billing_service

    invitation = await _open_invitation_by_token(db, token)
    org = await db.get(Organization, invitation.organization_id)
    if org is None or org.deleted_at is not None:
        raise NotFound("This invitation is invalid or has expired.")

    if user.email.lower() != invitation.email.lower():
        raise Conflict(
            "This invitation was sent to a different email address. "
            "Sign in with the invited address to accept it."
        )

    already = await db.scalar(
        select(OrganizationMember.id).where(
            OrganizationMember.organization_id == org.id,
            OrganizationMember.user_id == user.id,
        )
    )
    if already is not None:
        raise Conflict("You are already a member of this organization.")

    # Seat check at accept — the authoritative one.
    await _require_seat(db, org)

    member = OrganizationMember(
        organization_id=org.id,
        user_id=user.id,
        role=invitation.role,
        invited_by_user_id=invitation.invited_by_user_id,
    )
    db.add(member)
    invitation.accepted_at = utcnow()
    invitation.accepted_by_user_id = user.id
    await db.flush()
    await db.refresh(member)

    await billing_service.sync_user_plan(db, user)

    logger.info(
        "organizations.invite_accepted",
        organization_id=org.id,
        invitation_id=invitation.id,
        user_id=user.id,
    )
    return org, member


async def invited_email_for(db: AsyncSession, *, token: str) -> str:
    """The locked email for signup-from-invite (path 3)."""
    invitation = await _open_invitation_by_token(db, token)
    return invitation.email
