"""Organizations, members, and invitations (M21).

Every org-scoped handler takes one of the four `Org*` gates from `deps` — the
role matrix lives there and in the service, never in a route body. The
invitation accept endpoints live here too, outside the `/organizations/{id}`
tree, because their credential is the token, not a membership.
"""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter, Query
from sqlalchemy.ext.asyncio import AsyncSession

from app.api.deps import CurrentUser, Db, OrgAdmin, OrgContext, OrgOwner, OrgViewer
from app.core.responses import Envelope, ok
from app.models.organization import (
    Invitation,
    Organization,
    OrganizationMember,
    OrgRole,
    Visibility,
)
from app.models.user import User
from app.schemas.team import (
    AcceptInvitationIn,
    AcceptInvitationOut,
    InvitationIn,
    InvitationOut,
    InvitePreviewOut,
    MemberOut,
    MemberPatch,
    OrganizationIn,
    OrganizationOut,
    OrganizationPatch,
    OrganizationSettingsOut,
    OrganizationSettingsPatch,
    SeatsOut,
    TransferOwnershipIn,
)
from app.services import organization_service

router = APIRouter(tags=["organizations"])


async def _out(db: AsyncSession, org: Organization, member: OrganizationMember) -> OrganizationOut:
    org_settings = organization_service.settings_of(org)
    return OrganizationOut(
        id=org.id,
        name=org.name,
        slug=org.slug,
        plan=org.plan.value,
        role=member.role.value,
        seats=SeatsOut(
            used=await organization_service.seats_used(db, org),
            limit=await organization_service.seat_limit(db, org),
            purchased=org.seats_purchased,
        ),
        settings=OrganizationSettingsOut(
            approved_tools=org_settings.approved_tools,
            require_approval=org_settings.require_approval,
            default_visibility=org_settings.default_visibility.value,
        ),
        created_at=org.created_at,
    )


# ── Organizations ────────────────────────────────────────────────────────────


@router.post(
    "/organizations",
    response_model=Envelope[OrganizationOut],
    name="create_organization",
    status_code=201,
)
async def create_organization(db: Db, user: CurrentUser, payload: OrganizationIn) -> dict[str, Any]:
    org = await organization_service.create(db, user, name=payload.name)
    _, member = await organization_service.get_membership(db, user=user, organization_id=org.id)
    return ok(await _out(db, org, member))


@router.get(
    "/organizations",
    response_model=Envelope[list[OrganizationOut]],
    name="list_organizations",
)
async def list_organizations(db: Db, user: CurrentUser) -> dict[str, Any]:
    memberships = await organization_service.list_for(db, user)
    return ok([await _out(db, org, member) for org, member in memberships])


@router.get(
    "/organizations/{organization_id}",
    response_model=Envelope[OrganizationOut],
    name="get_organization",
)
async def get_organization(db: Db, ctx: OrgViewer) -> dict[str, Any]:
    return ok(await _out(db, ctx.org, ctx.member))


@router.patch(
    "/organizations/{organization_id}",
    response_model=Envelope[OrganizationOut],
    name="update_organization",
)
async def update_organization(db: Db, ctx: OrgAdmin, payload: OrganizationPatch) -> dict[str, Any]:
    org = await organization_service.update(db, ctx.org, name=payload.name)
    return ok(await _out(db, org, ctx.member))


@router.delete("/organizations/{organization_id}", status_code=204, name="delete_organization")
async def delete_organization(db: Db, ctx: OrgOwner) -> None:
    await organization_service.delete(db, ctx.org)


# ── Settings ─────────────────────────────────────────────────────────────────


@router.get(
    "/organizations/{organization_id}/settings",
    response_model=Envelope[OrganizationSettingsOut],
    name="get_organization_settings",
)
async def get_organization_settings(ctx: OrgViewer) -> dict[str, Any]:
    org_settings = organization_service.settings_of(ctx.org)
    return ok(
        OrganizationSettingsOut(
            approved_tools=org_settings.approved_tools,
            require_approval=org_settings.require_approval,
            default_visibility=org_settings.default_visibility.value,
        )
    )


@router.patch(
    "/organizations/{organization_id}/settings",
    response_model=Envelope[OrganizationSettingsOut],
    name="update_organization_settings",
)
async def update_organization_settings(
    db: Db, ctx: OrgAdmin, payload: OrganizationSettingsPatch
) -> dict[str, Any]:
    org = await organization_service.update_settings(
        db,
        ctx.org,
        approved_tools=payload.approved_tools,
        require_approval=payload.require_approval,
        default_visibility=(
            Visibility(payload.default_visibility)
            if payload.default_visibility is not None
            else None
        ),
    )
    org_settings = organization_service.settings_of(org)
    return ok(
        OrganizationSettingsOut(
            approved_tools=org_settings.approved_tools,
            require_approval=org_settings.require_approval,
            default_visibility=org_settings.default_visibility.value,
        )
    )


# ── Members ──────────────────────────────────────────────────────────────────


def _member_out(member: OrganizationMember, user_row: User, ctx: OrgContext) -> MemberOut:
    return MemberOut(
        id=member.id,
        user_id=member.user_id,
        name=user_row.name,
        email=user_row.email,
        avatar_url=user_row.avatar_url,
        role=member.role.value,
        is_current_user=member.user_id == ctx.user.id,
        joined_at=member.created_at,
    )


@router.get(
    "/organizations/{organization_id}/members",
    response_model=Envelope[list[MemberOut]],
    name="list_organization_members",
)
async def list_organization_members(db: Db, ctx: OrgViewer) -> dict[str, Any]:
    members = await organization_service.list_members(db, ctx.org)
    return ok([_member_out(member, user_row, ctx) for member, user_row in members])


@router.patch(
    "/organizations/{organization_id}/members/{membership_id}",
    response_model=Envelope[MemberOut],
    name="update_organization_member",
)
async def update_organization_member(
    db: Db, ctx: OrgAdmin, membership_id: str, payload: MemberPatch
) -> dict[str, Any]:
    member = await organization_service.change_role(
        db, ctx.org, membership_id=membership_id, role=OrgRole(payload.role)
    )
    user_row = await db.get_one(User, member.user_id)
    return ok(_member_out(member, user_row, ctx))


@router.delete(
    "/organizations/{organization_id}/members/{membership_id}",
    status_code=204,
    name="remove_organization_member",
)
async def remove_organization_member(db: Db, ctx: OrgAdmin, membership_id: str) -> None:
    await organization_service.remove_member(db, ctx.org, membership_id=membership_id)


@router.post(
    "/organizations/{organization_id}/ownership-transfer",
    response_model=Envelope[MemberOut],
    name="transfer_organization_ownership",
)
async def transfer_organization_ownership(
    db: Db, ctx: OrgOwner, payload: TransferOwnershipIn
) -> dict[str, Any]:
    successor = await organization_service.transfer_ownership(
        db, ctx.org, owner_member=ctx.member, to_membership_id=payload.membership_id
    )
    user_row = await db.get_one(User, successor.user_id)
    return ok(_member_out(successor, user_row, ctx))


# ── Invitations (org side) ───────────────────────────────────────────────────


def _invitation_out(invitation: Invitation, inviter_name: str | None) -> InvitationOut:
    return InvitationOut(
        id=invitation.id,
        email=invitation.email,
        role=invitation.role.value,
        invited_by=inviter_name,
        expires_at=invitation.expires_at,
        created_at=invitation.created_at,
    )


async def _inviter_name(db: AsyncSession, user_id: str | None) -> str | None:
    if user_id is None:
        return None
    inviter = await db.get(User, user_id)
    return inviter.name if inviter else None


@router.post(
    "/organizations/{organization_id}/invitations",
    response_model=Envelope[InvitationOut],
    name="create_invitation",
    status_code=201,
)
async def create_invitation(db: Db, ctx: OrgAdmin, payload: InvitationIn) -> dict[str, Any]:
    invitation = await organization_service.invite(
        db, ctx.org, ctx.user, email=payload.email, role=OrgRole(payload.role)
    )
    return ok(_invitation_out(invitation, ctx.user.name))


@router.get(
    "/organizations/{organization_id}/invitations",
    response_model=Envelope[list[InvitationOut]],
    name="list_invitations",
)
async def list_invitations(db: Db, ctx: OrgAdmin) -> dict[str, Any]:
    invitations = await organization_service.list_invitations(db, ctx.org)
    return ok(
        [
            _invitation_out(inv, await _inviter_name(db, inv.invited_by_user_id))
            for inv in invitations
        ]
    )


@router.post(
    "/organizations/{organization_id}/invitations/{invitation_id}/resend",
    response_model=Envelope[InvitationOut],
    name="resend_invitation",
)
async def resend_invitation(db: Db, ctx: OrgAdmin, invitation_id: str) -> dict[str, Any]:
    invitation = await organization_service.resend(
        db, ctx.org, ctx.user, invitation_id=invitation_id
    )
    return ok(_invitation_out(invitation, ctx.user.name))


@router.delete(
    "/organizations/{organization_id}/invitations/{invitation_id}",
    status_code=204,
    name="revoke_invitation",
)
async def revoke_invitation(db: Db, ctx: OrgAdmin, invitation_id: str) -> None:
    await organization_service.revoke(db, ctx.org, invitation_id=invitation_id)


# ── Invitations (recipient side) ─────────────────────────────────────────────


@router.get(
    "/invitations/preview",
    response_model=Envelope[InvitePreviewOut],
    name="preview_invitation",
)
async def preview_invitation(
    db: Db, token: str = Query(min_length=16, max_length=256)
) -> dict[str, Any]:
    """Public — the token is the credential. Dead, revoked, and expired tokens
    are the same 404 as tokens that never existed."""
    invitation, org = await organization_service.preview(db, token=token)
    return ok(
        InvitePreviewOut(
            organization_name=org.name,
            email=invitation.email,
            role=invitation.role.value,
            invited_by=await _inviter_name(db, invitation.invited_by_user_id),
            expires_at=invitation.expires_at,
        )
    )


@router.post(
    "/invitations/accept",
    response_model=Envelope[AcceptInvitationOut],
    name="accept_invitation",
)
async def accept_invitation(
    db: Db, user: CurrentUser, payload: AcceptInvitationIn
) -> dict[str, Any]:
    org, member = await organization_service.accept(db, token=payload.token, user=user)
    return ok(AcceptInvitationOut(organization=await _out(db, org, member)))
