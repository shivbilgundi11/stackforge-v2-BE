"""Organizations, roles, invitations, visibility, comments, approvals (M21).

The two tests that matter most here are the role matrix and cross-org
isolation. The matrix is the definitive statement of who may do what, endpoint
by endpoint; isolation asserts that another organization's id behaves exactly
like an id that does not exist. Everything else — invitation paths, seats,
threads, approvals — is the module's behaviour on top of those two guarantees.

No test talks to Stripe; the seat-change test installs a fake that records the
quantity it was asked for. Invitation tokens are read out of the email outbox
the way a recipient would read them.
"""

from __future__ import annotations

import re
from collections.abc import Iterator
from datetime import timedelta
from typing import Any

import pytest
from httpx import AsyncClient
from sqlalchemy import select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.database import utcnow
from app.integrations import email as email_integration
from app.integrations import stripe as stripe_integration
from app.models.billing import Metric, Subscription, SubscriptionStatus
from app.models.organization import (
    Invitation,
    Organization,
    OrganizationMember,
    OrgRole,
)
from app.models.user import Plan, PlanSource, User
from tests.conftest import GOOD_PASSWORD, register_and_verify, set_limit

ORGS = "/api/v1/organizations"

# ── Helpers ─────────────────────────────────────────────────────────────────


async def _actor(
    client: AsyncClient,
    db: AsyncSession,
    email: str,
    *,
    plan: Plan | None = None,
) -> tuple[User, str]:
    """Register, verify, optionally set a plan, sign in. Returns (user, token)
    and leaves the client authorized as them."""
    user_id = await register_and_verify(client, db, email=email)
    user = await db.get(User, user_id)
    assert user is not None
    if plan is not None:
        user.plan = plan
        await db.flush()
    response = await client.post(
        "/api/v1/auth/login", json={"email": email, "password": GOOD_PASSWORD}
    )
    token = str(response.json()["data"]["tokens"]["access_token"])
    _use(client, token)
    return user, token


def _use(client: AsyncClient, token: str) -> None:
    client.headers["Authorization"] = f"Bearer {token}"


async def _create_org(client: AsyncClient, name: str = "Acme") -> dict[str, Any]:
    response = await client.post(ORGS, json={"name": name})
    assert response.status_code == 201, response.text
    return dict(response.json()["data"])


def _invite_token(outbox: email_integration.ConsoleSender) -> str:
    """Read the token the way the recipient would: out of the email."""
    match = re.search(r"token=([\w\-]+)", outbox.outbox[-1].text)
    assert match, outbox.outbox[-1].text
    return match.group(1)


async def _join(
    client: AsyncClient,
    db: AsyncSession,
    outbox: email_integration.ConsoleSender,
    org_id: str,
    email: str,
    role: str,
    *,
    owner_token: str,
) -> tuple[User, str]:
    """Invite (as whoever owner_token is), then sign the invitee up through
    acceptance path 1: account, sign in, accept."""
    _use(client, owner_token)
    response = await client.post(
        f"{ORGS}/{org_id}/invitations", json={"email": email, "role": role}
    )
    assert response.status_code == 201, response.text
    token = _invite_token(outbox)

    user, user_token = await _actor(client, db, email)
    response = await client.post("/api/v1/invitations/accept", json={"token": token})
    assert response.status_code == 200, response.text
    return user, user_token


async def _membership_id(db: AsyncSession, org_id: str, user_id: str) -> str:
    member = await db.scalar(
        select(OrganizationMember).where(
            OrganizationMember.organization_id == org_id,
            OrganizationMember.user_id == user_id,
        )
    )
    assert member is not None
    return member.id


# ── The role matrix ─────────────────────────────────────────────────────────

#: (method, path suffix, payload, minimum role). `{inv}` and `{target}` are
#: filled per run. Owner-only rows are asserted as 403 for everyone else here
#: and as working in their own dedicated tests below.
MATRIX: list[tuple[str, str, dict[str, Any] | None, OrgRole]] = [
    ("GET", "", None, OrgRole.VIEWER),
    ("GET", "/settings", None, OrgRole.VIEWER),
    ("GET", "/members", None, OrgRole.VIEWER),
    ("GET", "/invitations", None, OrgRole.ADMIN),
    ("PATCH", "", {"name": "Renamed"}, OrgRole.ADMIN),
    ("PATCH", "/settings", {"require_approval": True}, OrgRole.ADMIN),
    ("POST", "/invitations", {"email": "fresh@example.com", "role": "member"}, OrgRole.ADMIN),
    ("POST", "/invitations/{inv}/resend", {}, OrgRole.ADMIN),
    ("DELETE", "/invitations/{inv}", None, OrgRole.ADMIN),
    ("PATCH", "/members/{target}", {"role": "viewer"}, OrgRole.ADMIN),
    ("DELETE", "/members/{target}", None, OrgRole.ADMIN),
    ("POST", "/ownership-transfer", {"membership_id": "{target}"}, OrgRole.OWNER),
    ("DELETE", "", None, OrgRole.OWNER),
]


@pytest.mark.parametrize("role", list(OrgRole), ids=lambda role: role.value)
async def test_role_matrix(
    client: AsyncClient,
    db: AsyncSession,
    outbox: email_integration.ConsoleSender,
    role: OrgRole,
) -> None:
    """Every team endpoint x this role, allow or deny — the definitive test."""
    _, owner_token = await _actor(client, db, f"owner-{role.value}@example.com", plan=Plan.TEAM)
    org = await _create_org(client, f"Matrix {role.value}")

    # A member-role target for member-mutating endpoints.
    target_user, _ = await _join(
        client, db, outbox, org["id"], f"target-{role.value}@example.com", "member",
        owner_token=owner_token,
    )
    target = await _membership_id(db, org["id"], target_user.id)

    # An open invitation for the invitation-mutating endpoints.
    _use(client, owner_token)
    response = await client.post(
        f"{ORGS}/{org['id']}/invitations",
        json={"email": f"pending-{role.value}@example.com", "role": "viewer"},
    )
    assert response.status_code == 201
    invitation_id = response.json()["data"]["id"]

    if role is OrgRole.OWNER:
        actor_token = owner_token
    else:
        _, actor_token = await _join(
            client, db, outbox, org["id"], f"actor-{role.value}@example.com", role.value,
            owner_token=owner_token,
        )

    _use(client, actor_token)
    for method, suffix, payload, minimum in MATRIX:
        if role is OrgRole.OWNER and minimum is OrgRole.OWNER:
            continue  # exercised in their own tests; running both here would
            # have the transfer strip the owner before the delete.
        path = f"{ORGS}/{org['id']}" + suffix.format(inv=invitation_id, target=target)
        body = (
            {
                key: (value.format(target=target) if isinstance(value, str) else value)
                for key, value in payload.items()
            }
            if payload is not None
            else None
        )
        response = await client.request(method, path, json=body)

        label = f"{role.value} {method} {suffix or '/'}"
        if role.covers(minimum):
            assert response.status_code in (200, 201, 204), f"{label}: {response.text}"
        else:
            assert response.status_code == 403, f"{label}: {response.text}"
            assert response.json()["error"]["code"] == "FORBIDDEN"


async def test_cross_org_isolation(
    client: AsyncClient,
    db: AsyncSession,
    outbox: email_integration.ConsoleSender,
) -> None:
    """A user in org A asking about org B gets 404 — the same answer as an id
    that does not exist, on every org-scoped endpoint."""
    await _actor(client, db, "owner-b@example.com", plan=Plan.TEAM)
    org_b = await _create_org(client, "Org B")

    # B shares a stack with their team; A must not be able to see it.
    response = await client.post(
        "/api/v1/stacks",
        json={"name": "B's stack", "component_slugs": ["pgvector"], "visibility": "team"},
        headers={"X-Organization-Id": org_b["id"]},
    )
    assert response.status_code == 201
    stack_b = response.json()["data"]["id"]

    _, token_a = await _actor(client, db, "owner-a@example.com", plan=Plan.TEAM)
    org_a = await _create_org(client, "Org A")

    _use(client, token_a)
    for method, suffix, payload, _minimum in MATRIX:
        path = f"{ORGS}/{org_b['id']}" + suffix.format(inv="inv_x", target="mem_x")
        response = await client.request(method, path, json=payload)
        assert response.status_code == 404, f"{method} {suffix}: {response.status_code}"
        assert response.json()["error"]["code"] == "NOT_FOUND"

    # The org header is scoped the same way: naming an org you are not in is
    # indistinguishable from naming one that does not exist.
    response = await client.get(
        "/api/v1/stacks", params={"scope": "team"}, headers={"X-Organization-Id": org_b["id"]}
    )
    assert response.status_code == 404

    # B's team stack, through A's own org context: not visible, not confirmed.
    response = await client.get(
        f"/api/v1/stacks/{stack_b}", headers={"X-Organization-Id": org_a["id"]}
    )
    assert response.status_code == 404

    response = await client.get(
        "/api/v1/comments",
        params={"resource_type": "stack", "resource_id": stack_b},
        headers={"X-Organization-Id": org_a["id"]},
    )
    assert response.status_code == 404


# ── Ownership ───────────────────────────────────────────────────────────────


async def test_one_owner_enforced_by_the_database(
    client: AsyncClient, db: AsyncSession
) -> None:
    await _actor(client, db, "solo-owner@example.com", plan=Plan.TEAM)
    org = await _create_org(client, "One Owner")

    intruder_id = await register_and_verify(client, db, email="second-owner@example.com")
    db.add(
        OrganizationMember(
            organization_id=org["id"], user_id=intruder_id, role=OrgRole.OWNER
        )
    )
    with pytest.raises(IntegrityError):
        await db.flush()


async def test_ownership_transfer(
    client: AsyncClient,
    db: AsyncSession,
    outbox: email_integration.ConsoleSender,
) -> None:
    owner, owner_token = await _actor(client, db, "old-owner@example.com", plan=Plan.TEAM)
    org = await _create_org(client, "Handover")
    successor, _ = await _join(
        client, db, outbox, org["id"], "new-owner@example.com", "admin",
        owner_token=owner_token,
    )
    successor_membership = await _membership_id(db, org["id"], successor.id)

    _use(client, owner_token)
    response = await client.post(
        f"{ORGS}/{org['id']}/ownership-transfer",
        json={"membership_id": successor_membership},
    )
    assert response.status_code == 200
    assert response.json()["data"]["role"] == "owner"

    row = await db.get(Organization, org["id"])
    assert row is not None and row.owner_id == successor.id

    old = await db.scalar(
        select(OrganizationMember).where(
            OrganizationMember.organization_id == org["id"],
            OrganizationMember.user_id == owner.id,
        )
    )
    assert old is not None and old.role is OrgRole.ADMIN

    # The old owner is an admin now: owner-only actions deny.
    response = await client.delete(f"{ORGS}/{org['id']}")
    assert response.status_code == 403


# ── Invitations ─────────────────────────────────────────────────────────────


async def test_signup_from_invite_locks_and_verifies_the_email(
    client: AsyncClient,
    db: AsyncSession,
    outbox: email_integration.ConsoleSender,
) -> None:
    """Acceptance path 3, the one most often broken and the highest-value one."""
    await _actor(client, db, "grower@example.com", plan=Plan.TEAM)
    org = await _create_org(client, "Growing Team")

    response = await client.post(
        f"{ORGS}/{org['id']}/invitations",
        json={"email": "brand-new@example.com", "role": "member"},
    )
    assert response.status_code == 201
    token = _invite_token(outbox)

    # The accept page previews the invite from the token alone.
    client.headers.pop("Authorization", None)
    response = await client.get("/api/v1/invitations/preview", params={"token": token})
    assert response.status_code == 200
    preview = response.json()["data"]
    assert preview["email"] == "brand-new@example.com"
    assert preview["organization_name"] == "Growing Team"

    # A signup that edits the locked email away from the invited address dies.
    response = await client.post(
        "/api/v1/auth/register",
        json={
            "email": "someone-else@example.com",
            "password": GOOD_PASSWORD,
            "name": "Impostor",
            "invite_token": token,
        },
    )
    assert response.status_code == 422

    # The real signup: verified implicitly by possession of the invite.
    response = await client.post(
        "/api/v1/auth/register",
        json={
            "email": "brand-new@example.com",
            "password": GOOD_PASSWORD,
            "name": "Brand New",
            "invite_token": token,
        },
    )
    assert response.status_code == 202

    user = (
        await db.execute(select(User).where(User.email == "brand-new@example.com"))
    ).scalar_one()
    assert user.email_verified_at is not None
    assert not any(
        mail.to == "brand-new@example.com" and "Verify" in mail.subject
        for mail in outbox.outbox
    ), "no verification round-trip — the invite already proved the inbox"

    # Sign in and accept with the same token.
    response = await client.post(
        "/api/v1/auth/login",
        json={"email": "brand-new@example.com", "password": GOOD_PASSWORD},
    )
    _use(client, response.json()["data"]["tokens"]["access_token"])
    response = await client.post("/api/v1/invitations/accept", json={"token": token})
    assert response.status_code == 200
    assert response.json()["data"]["organization"]["id"] == org["id"]


async def test_accept_requires_the_invited_address(
    client: AsyncClient,
    db: AsyncSession,
    outbox: email_integration.ConsoleSender,
) -> None:
    await _actor(client, db, "strict@example.com", plan=Plan.TEAM)
    org = await _create_org(client)
    response = await client.post(
        f"{ORGS}/{org['id']}/invitations",
        json={"email": "intended@example.com", "role": "member"},
    )
    assert response.status_code == 201
    token = _invite_token(outbox)

    await _actor(client, db, "wrong-account@example.com")
    response = await client.post("/api/v1/invitations/accept", json={"token": token})
    assert response.status_code == 409


async def test_expired_and_revoked_invites_are_404(
    client: AsyncClient,
    db: AsyncSession,
    outbox: email_integration.ConsoleSender,
) -> None:
    _, owner_token = await _actor(client, db, "expiry@example.com", plan=Plan.TEAM)
    org = await _create_org(client)

    response = await client.post(
        f"{ORGS}/{org['id']}/invitations", json={"email": "late@example.com", "role": "member"}
    )
    invitation_id = response.json()["data"]["id"]
    token = _invite_token(outbox)

    invitation = await db.get(Invitation, invitation_id)
    assert invitation is not None
    invitation.expires_at = utcnow() - timedelta(minutes=1)
    await db.flush()

    await _actor(client, db, "late@example.com")
    response = await client.post("/api/v1/invitations/accept", json={"token": token})
    assert response.status_code == 404
    response = await client.get("/api/v1/invitations/preview", params={"token": token})
    assert response.status_code == 404

    # Revoked reads the same as expired.
    _use(client, owner_token)
    response = await client.post(
        f"{ORGS}/{org['id']}/invitations", json={"email": "gone@example.com", "role": "member"}
    )
    revoke_id = response.json()["data"]["id"]
    token = _invite_token(outbox)
    await client.delete(f"{ORGS}/{org['id']}/invitations/{revoke_id}")
    response = await client.get("/api/v1/invitations/preview", params={"token": token})
    assert response.status_code == 404


async def test_duplicate_invites_and_existing_members_conflict(
    client: AsyncClient,
    db: AsyncSession,
    outbox: email_integration.ConsoleSender,
) -> None:
    await _actor(client, db, "dupes@example.com", plan=Plan.TEAM)
    org = await _create_org(client)

    payload = {"email": "twice@example.com", "role": "member"}
    assert (await client.post(f"{ORGS}/{org['id']}/invitations", json=payload)).status_code == 201
    response = await client.post(f"{ORGS}/{org['id']}/invitations", json=payload)
    assert response.status_code == 409

    response = await client.post(
        f"{ORGS}/{org['id']}/invitations", json={"email": "dupes@example.com", "role": "member"}
    )
    assert response.status_code == 409, "the owner is already a member"


# ── Seats ───────────────────────────────────────────────────────────────────


async def test_seat_limits_at_send_and_accept(
    client: AsyncClient,
    db: AsyncSession,
    outbox: email_integration.ConsoleSender,
) -> None:
    """Invitations do not reserve seats; the accept check is authoritative."""
    await set_limit(db, plan=Plan.TEAM, metric=Metric.SEATS, value=2)

    _, owner_token = await _actor(client, db, "seats@example.com", plan=Plan.TEAM)
    org = await _create_org(client, "Tight Ship")

    # One seat free (owner holds the other). Three invites all send.
    tokens: list[str] = []
    for index in range(3):
        response = await client.post(
            f"{ORGS}/{org['id']}/invitations",
            json={"email": f"crew-{index}@example.com", "role": "member"},
        )
        assert response.status_code == 201, response.text
        tokens.append(_invite_token(outbox))

    # First accept fills the team.
    await _actor(client, db, "crew-0@example.com")
    response = await client.post("/api/v1/invitations/accept", json={"token": tokens[0]})
    assert response.status_code == 200

    # Second accept gets the clear seat error, not a silent overage.
    await _actor(client, db, "crew-1@example.com")
    response = await client.post("/api/v1/invitations/accept", json={"token": tokens[1]})
    assert response.status_code == 402
    body = response.json()["error"]
    assert body["code"] == "SEATS_EXCEEDED"
    assert body["details"] == {"limit": 2, "used": 2}

    # And a full team cannot send at all.
    _use(client, owner_token)
    response = await client.post(
        f"{ORGS}/{org['id']}/invitations", json={"email": "one-more@example.com", "role": "member"}
    )
    assert response.status_code == 402


class _SeatStripe:
    """Only what the seat test needs to observe."""

    def __init__(self) -> None:
        self.quantities: list[dict[str, Any]] = []

    async def create_customer(self, **kwargs: Any) -> str:
        return "cus_seat"

    async def create_checkout_session(self, **kwargs: Any) -> tuple[str, str]:
        return "cs_seat", "https://checkout.stripe.test/cs_seat"

    async def create_portal_session(self, **kwargs: Any) -> str:
        return "https://portal.stripe.test/seat"

    async def list_invoices(self, **kwargs: Any) -> list[dict[str, Any]]:
        return []

    async def cancel_at_period_end(self, **kwargs: Any) -> None:
        return None

    async def update_subscription_quantity(self, **kwargs: Any) -> None:
        self.quantities.append(kwargs)


@pytest.fixture
def seat_stripe() -> Iterator[_SeatStripe]:
    fake = _SeatStripe()
    stripe_integration.set_client(fake)
    yield fake
    stripe_integration.set_client(None)


async def test_seat_change_adjusts_stripe_quantity(
    client: AsyncClient,
    db: AsyncSession,
    outbox: email_integration.ConsoleSender,
    seat_stripe: _SeatStripe,
) -> None:
    _, owner_token = await _actor(client, db, "buyer@example.com", plan=Plan.TEAM)
    org = await _create_org(client, "Paying Team")

    db.add(
        Subscription(
            organization_id=org["id"],
            stripe_customer_id="cus_seat",
            stripe_subscription_id="sub_seat_1",
            plan=Plan.TEAM,
            status=SubscriptionStatus.ACTIVE,
            seats=5,
        )
    )
    await db.flush()

    response = await client.post("/api/v1/billing/seats", json={"seats": 7})
    assert response.status_code == 200, response.text
    assert response.json()["data"] == {"seats": 7, "used": 1}
    assert seat_stripe.quantities == [{"subscription_id": "sub_seat_1", "quantity": 7}]

    subscription = await db.scalar(
        select(Subscription).where(Subscription.organization_id == org["id"])
    )
    assert subscription is not None and subscription.seats == 7
    row = await db.get(Organization, org["id"])
    assert row is not None and row.seats_purchased == 7

    # Below the current membership is refused — remove members, then seats.
    _, _ = await _join(
        client, db, outbox, org["id"], "occupant@example.com", "member",
        owner_token=owner_token,
    )
    _use(client, owner_token)
    response = await client.post("/api/v1/billing/seats", json={"seats": 1})
    assert response.status_code == 422

    # And only the owner may buy. The admin gets a 403, not a bill.
    _use(client, owner_token)
    _, admin_token = await _join(
        client, db, outbox, org["id"], "spender@example.com", "admin",
        owner_token=owner_token,
    )
    _use(client, admin_token)
    response = await client.post(
        "/api/v1/billing/seats", json={"seats": 9, "organization_id": org["id"]}
    )
    assert response.status_code == 403


# ── Plan fan-out ────────────────────────────────────────────────────────────


async def test_membership_grants_and_removal_revokes_the_team_plan(
    client: AsyncClient,
    db: AsyncSession,
    outbox: email_integration.ConsoleSender,
) -> None:
    _, owner_token = await _actor(client, db, "granter@example.com", plan=Plan.TEAM)
    org = await _create_org(client, "Plan Grants")

    member, _ = await _join(
        client, db, outbox, org["id"], "beneficiary@example.com", "member",
        owner_token=owner_token,
    )
    await db.refresh(member)
    assert member.plan is Plan.TEAM
    assert member.plan_source is PlanSource.ORGANIZATION

    membership = await _membership_id(db, org["id"], member.id)
    _use(client, owner_token)
    response = await client.delete(f"{ORGS}/{org['id']}/members/{membership}")
    assert response.status_code == 204

    await db.refresh(member)
    assert member.plan is Plan.FREE
    assert member.plan_source is PlanSource.PERSONAL


# ── Team visibility ─────────────────────────────────────────────────────────


async def test_team_visibility_on_stacks(
    client: AsyncClient,
    db: AsyncSession,
    outbox: email_integration.ConsoleSender,
) -> None:
    _, owner_token = await _actor(client, db, "sharer@example.com", plan=Plan.TEAM)
    org = await _create_org(client, "Sharers")
    header = {"X-Organization-Id": org["id"]}

    # One shared, one private.
    shared = (
        await client.post(
            "/api/v1/stacks",
            json={"name": "Shared", "component_slugs": ["pgvector"], "visibility": "team"},
            headers=header,
        )
    ).json()["data"]
    private = (
        await client.post(
            "/api/v1/stacks", json={"name": "Private", "component_slugs": ["redis"]}
        )
    ).json()["data"]

    _, teammate_token = await _join(
        client, db, outbox, org["id"], "teammate@example.com", "member",
        owner_token=owner_token,
    )

    _use(client, teammate_token)
    response = await client.get("/api/v1/stacks", params={"scope": "team"}, headers=header)
    names = [row["name"] for row in response.json()["data"]]
    assert names == ["Shared"], "private work never becomes team work by default"
    row = response.json()["data"][0]
    assert row["is_yours"] is False
    assert row["owner_name"] == "Ada Lovelace"

    # Readable and editable by a member; the private one is 404.
    assert (
        await client.get(f"/api/v1/stacks/{shared['id']}", headers=header)
    ).status_code == 200
    assert (
        await client.get(f"/api/v1/stacks/{private['id']}", headers=header)
    ).status_code == 404

    response = await client.patch(
        f"/api/v1/stacks/{shared['id']}",
        json={"description": "teammate's note", "change_summary": "annotated"},
        headers=header,
    )
    assert response.status_code == 200

    # Visibility stays the author's call, and so does deletion.
    response = await client.patch(
        f"/api/v1/stacks/{shared['id']}", json={"visibility": "private"}, headers=header
    )
    assert response.status_code == 403
    assert (
        await client.delete(f"/api/v1/stacks/{shared['id']}", headers=header)
    ).status_code == 404

    # A viewer reads and never writes.
    _, viewer_token = await _join(
        client, db, outbox, org["id"], "watcher@example.com", "viewer",
        owner_token=owner_token,
    )
    _use(client, viewer_token)
    assert (
        await client.get(f"/api/v1/stacks/{shared['id']}", headers=header)
    ).status_code == 200
    response = await client.patch(
        f"/api/v1/stacks/{shared['id']}", json={"name": "Vandalised"}, headers=header
    )
    assert response.status_code == 404, "read-only looks like not-found, not like forbidden"


# ── Comments ────────────────────────────────────────────────────────────────


async def _shared_stack(client: AsyncClient, org_id: str) -> str:
    response = await client.post(
        "/api/v1/stacks",
        json={"name": "Discussed", "component_slugs": ["pgvector"], "visibility": "team"},
        headers={"X-Organization-Id": org_id},
    )
    return str(response.json()["data"]["id"])


async def test_comment_thread_mentions_resolve_and_tombstone(
    client: AsyncClient,
    db: AsyncSession,
    outbox: email_integration.ConsoleSender,
) -> None:
    _, owner_token = await _actor(client, db, "author@example.com", plan=Plan.TEAM)
    org = await _create_org(client, "Discussers")
    stack_id = await _shared_stack(client, org["id"])
    anchor = {"resource_type": "stack", "resource_id": stack_id}

    teammate, teammate_token = await _join(
        client, db, outbox, org["id"], "replier@example.com", "member",
        owner_token=owner_token,
    )

    # Root comment, with a mention that lands in the teammate's inbox.
    _use(client, owner_token)
    response = await client.post(
        "/api/v1/comments",
        json={**anchor, "body": "Swap the cache?", "mentions": [teammate.id]},
    )
    assert response.status_code == 201, response.text
    root = response.json()["data"]
    mention_mail = outbox.outbox[-1]
    assert mention_mail.to == "replier@example.com"
    assert "mentioned" in mention_mail.subject

    # Mentioning an outsider is refused, not silently dropped.
    outsider_id = await register_and_verify(client, db, email="outsider@example.com")
    response = await client.post(
        "/api/v1/comments", json={**anchor, "body": "psst", "mentions": [outsider_id]}
    )
    assert response.status_code == 422

    # One level of threading, enforced.
    _use(client, teammate_token)
    response = await client.post(
        "/api/v1/comments", json={**anchor, "body": "Yes — Valkey.", "parent_id": root["id"]}
    )
    assert response.status_code == 201
    reply = response.json()["data"]
    response = await client.post(
        "/api/v1/comments", json={**anchor, "body": "nested", "parent_id": reply["id"]}
    )
    assert response.status_code == 422

    # Resolve the root; a viewer can read the thread but not join it.
    response = await client.post(f"/api/v1/comments/{root['id']}/resolve", json={"resolved": True})
    assert response.status_code == 200
    assert response.json()["data"]["resolved_at"] is not None

    _, viewer_token = await _join(
        client, db, outbox, org["id"], "lurker@example.com", "viewer",
        owner_token=owner_token,
    )
    _use(client, viewer_token)
    response = await client.get("/api/v1/comments", params=anchor)
    assert response.status_code == 200
    assert len(response.json()["data"]) == 2
    response = await client.post("/api/v1/comments", json={**anchor, "body": "me too"})
    assert response.status_code == 403

    # Deleting the root leaves a tombstone so the reply keeps its context.
    _use(client, owner_token)
    assert (await client.delete(f"/api/v1/comments/{root['id']}")).status_code == 204
    thread = (await client.get("/api/v1/comments", params=anchor)).json()["data"]
    tombstone = next(row for row in thread if row["id"] == root["id"])
    assert tombstone["deleted"] is True
    assert tombstone["body"] == ""
    assert tombstone["author_name"] is None
    assert any(row["id"] == reply["id"] for row in thread)

    # Nothing to discuss on a private stack: same 404 as no stack at all.
    private = (
        await client.post(
            "/api/v1/stacks", json={"name": "Quiet", "component_slugs": ["redis"]}
        )
    ).json()["data"]
    response = await client.get(
        "/api/v1/comments",
        params={"resource_type": "stack", "resource_id": private["id"]},
    )
    assert response.status_code == 404


# ── Approvals ───────────────────────────────────────────────────────────────


async def test_approval_state_machine(
    client: AsyncClient,
    db: AsyncSession,
    outbox: email_integration.ConsoleSender,
) -> None:
    _, owner_token = await _actor(client, db, "approver@example.com", plan=Plan.TEAM)
    org = await _create_org(client, "Approvers")
    stack_id = await _shared_stack(client, org["id"])
    anchor = {"resource_type": "stack", "resource_id": stack_id}

    _, member_token = await _join(
        client, db, outbox, org["id"], "requester@example.com", "member",
        owner_token=owner_token,
    )

    _use(client, member_token)
    response = await client.post("/api/v1/approvals", json=anchor)
    assert response.status_code == 201
    approval = response.json()["data"]
    assert approval["status"] == "pending"

    # One pending approval per resource.
    assert (await client.post("/api/v1/approvals", json=anchor)).status_code == 409

    # A member cannot decide, and the requester cannot approve themselves.
    response = await client.patch(
        f"/api/v1/approvals/{approval['id']}", json={"action": "approve"}
    )
    assert response.status_code == 403

    _use(client, owner_token)
    response = await client.patch(
        f"/api/v1/approvals/{approval['id']}",
        json={"action": "approve", "note": "Costs check out."},
    )
    assert response.status_code == 200
    decided = response.json()["data"]
    assert decided["status"] == "approved"
    assert decided["decision_note"] == "Costs check out."
    assert decided["decided_by"] == "Ada Lovelace"

    # Deciding twice is an invalid transition.
    response = await client.patch(
        f"/api/v1/approvals/{approval['id']}", json={"action": "reject"}
    )
    assert response.status_code == 409

    # A new request opens after a decision; rejection closes it the same way.
    _use(client, member_token)
    second = (await client.post("/api/v1/approvals", json=anchor)).json()["data"]
    _use(client, owner_token)
    response = await client.patch(
        f"/api/v1/approvals/{second['id']}", json={"action": "reject", "note": "Too pricey."}
    )
    assert response.json()["data"]["status"] == "rejected"

    history = (await client.get("/api/v1/approvals", params=anchor)).json()["data"]
    assert [row["status"] for row in history] == ["rejected", "approved"]


# ── Approved tools ──────────────────────────────────────────────────────────


@pytest.mark.usefixtures("seeded_catalog")
async def test_approved_tools_flag_and_never_exclude(
    client: AsyncClient,
    db: AsyncSession,
    outbox: email_integration.ConsoleSender,
) -> None:
    await _actor(client, db, "policy@example.com", plan=Plan.TEAM)
    org = await _create_org(client, "Policy Shop")
    header = {"X-Organization-Id": org["id"]}

    # A typo'd slug is refused — an allowlist that approves nothing is worse
    # than no allowlist.
    response = await client.patch(
        f"{ORGS}/{org['id']}/settings", json={"approved_tools": ["postgress"]}
    )
    assert response.status_code == 422

    response = await client.patch(
        f"{ORGS}/{org['id']}/settings", json={"approved_tools": ["pgvector", "anthropic-api"]}
    )
    assert response.status_code == 200

    response = await client.post("/api/v1/architect/recommend", json={}, headers=header)
    assert response.status_code == 200, response.text
    output = response.json()["data"]

    rows = output["tables"]["components"]
    assert rows, "the recommendation still recommends"
    assert all(row["approved"] in ("yes", "no") for row in rows)
    unapproved = [row for row in rows if row["approved"] == "no"]
    assert unapproved, "a two-tool allowlist cannot cover a whole stack"

    # Flagged with a note — and present, which is the point.
    for row in unapproved:
        assert any(
            row["name"] in warning["message"] and "approved tool list" in warning["message"]
            for warning in output["warnings"]
        ), f"{row['name']} is unapproved and must carry a note"

    # Never excluded: the exclusions table only ever names hard constraints.
    assert all(
        "approved" not in exclusion["constraint"] and "policy" not in exclusion["constraint"]
        for exclusion in output["tables"]["exclusions"]
    )

    # Without the org header the same request carries no policy at all.
    response = await client.post("/api/v1/architect/recommend", json={})
    rows = response.json()["data"]["tables"]["components"]
    assert all("approved" not in row for row in rows)
