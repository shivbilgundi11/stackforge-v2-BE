"""Choosing a plan at signup, and the wall that follows it (M20/M21 follow-on).

The distinction every test here is protecting: **choosing a plan is not having
one.** `users.pending_plan` records the choice, `users.plan` records what was
paid for, and nothing but a webhook moves the second. A signup form that wrote
straight to `plan` would hand out Pro to anyone who picked it and closed the
tab, and no later event would ever take it back — Stripe does not send webhooks
for checkouts that did not happen.

The wall itself is a redirect, not a permission. Every quota and feature
decision still reads `user.plan`, so an account that skips the wall entirely
gets the free tier rather than the plan it selected. That is asserted here too,
because "the paywall is also the authorization" is exactly the shortcut this
design exists to avoid.
"""

from __future__ import annotations

from collections.abc import Iterator
from datetime import timedelta
from typing import Any

import pytest
from httpx import AsyncClient
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.config import settings
from app.core.database import utcnow
from app.integrations import stripe as stripe_integration
from app.models.billing import Subscription, SubscriptionStatus
from app.models.user import Plan, User
from app.services import auth_service
from tests.conftest import GOOD_PASSWORD, register_and_verify

REGISTER = "/api/v1/auth/register"
LOGIN = "/api/v1/auth/login"
SUBSCRIPTION = "/api/v1/billing/subscription"
SELECTION = "/api/v1/billing/plan-selection"
PLANS = "/api/v1/billing/plans"

PRO_MONTHLY = "price_pro_monthly_test"
TEAM_MONTHLY = "price_team_monthly_test"


class _FakeStripe:
    def __init__(self) -> None:
        self.checkouts: list[dict[str, Any]] = []

    async def create_customer(self, *, email: str, name: str, user_id: str) -> str:
        return "cus_selection"

    async def create_checkout_session(self, **kwargs: Any) -> tuple[str, str]:
        self.checkouts.append(kwargs)
        return "cs_1", "https://checkout.stripe.test/cs_1"


@pytest.fixture
def stripe(monkeypatch: pytest.MonkeyPatch) -> Iterator[_FakeStripe]:
    monkeypatch.setattr(settings, "stripe_secret_key", "sk_test_fake")
    monkeypatch.setattr(settings, "stripe_webhook_secret", "whsec_fake")
    monkeypatch.setattr(settings, "stripe_price_pro_monthly", PRO_MONTHLY)
    monkeypatch.setattr(settings, "stripe_price_team_monthly", TEAM_MONTHLY)

    fake = _FakeStripe()
    stripe_integration.set_client(fake)
    yield fake
    stripe_integration.set_client(None)


async def _register(client: AsyncClient, email: str, **extra: Any) -> None:
    response = await client.post(
        REGISTER,
        json={"email": email, "password": GOOD_PASSWORD, "name": "Plan Picker", **extra},
    )
    assert response.status_code == 202, response.text


async def _sign_in(client: AsyncClient, email: str) -> None:
    response = await client.post(LOGIN, json={"email": email, "password": GOOD_PASSWORD})
    assert response.status_code == 200, response.text
    token = response.json()["data"]["tokens"]["access_token"]
    client.headers["Authorization"] = f"Bearer {token}"


async def _user(db: AsyncSession, email: str) -> User:
    user = await auth_service.get_user_by_email(db, email)
    assert user is not None
    return user


# ── Registration records the choice ─────────────────────────────────────────


async def test_choosing_a_paid_plan_records_a_debt_not_a_plan(
    client: AsyncClient, db: AsyncSession
) -> None:
    """The whole design in one assertion."""
    await _register(client, "picks.pro@example.com", plan="pro", interval="annual")

    user = await _user(db, "picks.pro@example.com")
    assert user.pending_plan is Plan.PRO
    assert user.pending_interval == "annual"
    # Not upgraded. Nobody has paid anything yet.
    assert user.plan is Plan.FREE


async def test_choosing_free_owes_nothing(client: AsyncClient, db: AsyncSession) -> None:
    await _register(client, "picks.free@example.com", plan="free")

    user = await _user(db, "picks.free@example.com")
    assert user.pending_plan is None
    assert user.pending_interval is None


async def test_omitting_the_plan_is_the_same_as_choosing_free(
    client: AsyncClient, db: AsyncSession
) -> None:
    """Every caller written before this field existed still works."""
    await _register(client, "omits.plan@example.com")

    user = await _user(db, "omits.plan@example.com")
    assert user.pending_plan is None


async def test_enterprise_cannot_be_chosen_at_signup(client: AsyncClient) -> None:
    """It has no self-serve price, so a wall demanding payment for it would be
    a dead end. Rejected at the schema rather than silently downgraded."""
    response = await client.post(
        REGISTER,
        json={
            "email": "picks.enterprise@example.com",
            "password": GOOD_PASSWORD,
            "name": "Big Co",
            "plan": "enterprise",
        },
    )
    assert response.status_code == 422


# ── The wall ────────────────────────────────────────────────────────────────


async def test_the_subscription_endpoint_reports_the_wall(
    client: AsyncClient, db: AsyncSession
) -> None:
    await _register(client, "walled@example.com", plan="team", interval="monthly")
    user = await _user(db, "walled@example.com")
    user.email_verified_at = utcnow()
    await db.flush()
    await _sign_in(client, "walled@example.com")

    body = (await client.get(SUBSCRIPTION)).json()["data"]
    assert body["payment_required"] is True
    assert body["pending_plan"] == "team"
    assert body["pending_interval"] == "monthly"
    # And they are still on Free while they stand there.
    assert body["plan"] == "free"


async def test_a_walled_account_still_only_has_the_free_tier(
    client: AsyncClient, db: AsyncSession
) -> None:
    """The wall is a redirect, not an authorization.

    An account that never completes checkout must get Free — not the plan it
    picked. If this ever inverts, the paywall becomes the only thing standing
    between a signup form and a free Pro account.
    """
    await _register(client, "sneaky@example.com", plan="pro")
    user = await _user(db, "sneaky@example.com")
    user.email_verified_at = utcnow()
    await db.flush()
    await _sign_in(client, "sneaky@example.com")

    usage = (await client.get("/api/v1/billing/usage")).json()["data"]
    assert usage["plan"] == "free"


async def test_declining_the_wall_clears_the_debt(
    client: AsyncClient, db: AsyncSession
) -> None:
    """"Continue on Free" is always available. A wall with no way past it
    converts worse than one that can be declined, and support ends up clearing
    the column by hand."""
    await _register(client, "declines@example.com", plan="pro")
    user = await _user(db, "declines@example.com")
    user.email_verified_at = utcnow()
    await db.flush()
    await _sign_in(client, "declines@example.com")

    response = await client.post(SELECTION, json={"plan": None})
    assert response.status_code == 200, response.text
    assert response.json()["data"]["payment_required"] is False

    await db.refresh(user)
    assert user.pending_plan is None


async def test_the_wall_can_change_its_mind_about_the_plan(
    client: AsyncClient, db: AsyncSession
) -> None:
    await _register(client, "switches@example.com", plan="pro", interval="monthly")
    user = await _user(db, "switches@example.com")
    user.email_verified_at = utcnow()
    await db.flush()
    await _sign_in(client, "switches@example.com")

    response = await client.post(SELECTION, json={"plan": "team", "interval": "annual"})
    assert response.status_code == 200, response.text

    await db.refresh(user)
    assert user.pending_plan is Plan.TEAM
    assert user.pending_interval == "annual"


# ── Settling the debt ───────────────────────────────────────────────────────


async def test_paying_clears_the_wall(client: AsyncClient, db: AsyncSession) -> None:
    """The subscription arriving is what settles it — not the redirect."""
    from app.services import billing_service

    await _register(client, "pays@example.com", plan="pro")
    user = await _user(db, "pays@example.com")
    user.email_verified_at = utcnow()
    db.add(
        Subscription(
            user_id=user.id,
            stripe_customer_id="cus_pays",
            stripe_subscription_id="sub_pays",
            plan=Plan.PRO,
            status=SubscriptionStatus.ACTIVE,
            seats=1,
        )
    )
    await db.flush()

    await billing_service.sync_user_plan(db, user)

    assert user.plan is Plan.PRO
    assert user.pending_plan is None


async def test_a_better_plan_than_the_one_owed_also_settles_it(
    client: AsyncClient, db: AsyncSession
) -> None:
    """Someone who chose Pro and was then granted Team by an organization must
    not be held at a wall asking them to buy the lesser plan. That is a loop
    with no exit but support."""
    from app.services import billing_service

    await _register(client, "granted@example.com", plan="pro")
    user = await _user(db, "granted@example.com")
    db.add(
        Subscription(
            user_id=user.id,
            stripe_customer_id="cus_granted",
            stripe_subscription_id="sub_granted",
            plan=Plan.TEAM,
            status=SubscriptionStatus.ACTIVE,
            seats=5,
        )
    )
    await db.flush()

    await billing_service.sync_user_plan(db, user)

    assert user.plan is Plan.TEAM
    assert user.pending_plan is None


async def test_a_lesser_plan_does_not_settle_the_debt(
    client: AsyncClient, db: AsyncSession
) -> None:
    """The mirror. Owing Team is not discharged by holding Pro."""
    from app.services import billing_service

    await _register(client, "partly@example.com", plan="team")
    user = await _user(db, "partly@example.com")
    db.add(
        Subscription(
            user_id=user.id,
            stripe_customer_id="cus_partly",
            stripe_subscription_id="sub_partly",
            plan=Plan.PRO,
            status=SubscriptionStatus.ACTIVE,
            seats=1,
        )
    )
    await db.flush()

    await billing_service.sync_user_plan(db, user)

    assert user.plan is Plan.PRO
    assert user.pending_plan is Plan.TEAM


async def test_an_abandoned_upgrade_does_not_grant_the_plan(
    client: AsyncClient, db: AsyncSession, stripe: _FakeStripe
) -> None:
    """The one that actually reached production behaviour in manual testing.

    A Pro subscriber starts a Team checkout and closes the tab. Nothing has
    been charged and Stripe still says Pro. If starting the checkout wrote
    `plan=team` onto the live row, `sync_user_plan` — which reads that field —
    hands out Team on the next webhook for that account, free.
    """
    from app.services import billing_service

    user_id = await register_and_verify(client, db, email="abandons@example.com")
    user = await db.get(User, user_id)
    assert user is not None

    db.add(
        Subscription(
            user_id=user.id,
            stripe_customer_id="cus_abandons",
            stripe_subscription_id="sub_abandons",
            stripe_price_id=settings.stripe_price_pro_monthly,
            plan=Plan.PRO,
            status=SubscriptionStatus.ACTIVE,
            seats=1,
        )
    )
    await db.flush()
    await billing_service.sync_user_plan(db, user)
    assert user.plan is Plan.PRO

    # Off to Stripe for Team — and never comes back.
    await billing_service.start_checkout(db, user, plan=Plan.TEAM, interval="monthly")

    subscription = await billing_service.get_subscription(db, user)
    assert subscription is not None
    assert subscription.plan is Plan.PRO, "a live row must keep the plan Stripe confirmed"

    await billing_service.sync_user_plan(db, user)
    assert user.plan is Plan.PRO, "an abandoned checkout must not grant the plan"


async def test_an_abandoned_upgrade_can_be_retried(
    client: AsyncClient, db: AsyncSession, stripe: _FakeStripe
) -> None:
    """The other half of the same bug. With the intended plan written onto the
    row, the "already on this plan" guard compared against a plan the user was
    not on, and the retry was refused with no way forward."""
    from app.services import billing_service

    user_id = await register_and_verify(client, db, email="retries@example.com")
    user = await db.get(User, user_id)
    assert user is not None

    db.add(
        Subscription(
            user_id=user.id,
            stripe_customer_id="cus_retries",
            stripe_subscription_id="sub_retries",
            stripe_price_id=settings.stripe_price_pro_monthly,
            plan=Plan.PRO,
            status=SubscriptionStatus.ACTIVE,
            seats=1,
        )
    )
    await db.flush()

    await billing_service.start_checkout(db, user, plan=Plan.TEAM, interval="monthly")
    # Second attempt, same plan. Must be allowed.
    await billing_service.start_checkout(db, user, plan=Plan.TEAM, interval="monthly")

    assert len(stripe.checkouts) == 2
    assert stripe.checkouts[-1]["price_id"] == settings.stripe_price_team_monthly


async def test_an_unpaid_row_is_still_repointed(
    client: AsyncClient, db: AsyncSession, stripe: _FakeStripe
) -> None:
    """The behaviour the reuse exists for, kept intact: an abandoned checkout
    that never became a subscription is a scratch row, and the next checkout
    takes it over rather than tripping the one-subscription index."""
    from app.services import billing_service

    user_id = await register_and_verify(client, db, email="scratch@example.com")
    user = await db.get(User, user_id)
    assert user is not None

    await billing_service.start_checkout(db, user, plan=Plan.PRO, interval="monthly")
    first = await billing_service.get_subscription(db, user)
    assert first is not None
    assert first.plan is Plan.PRO
    assert not first.is_paid

    await billing_service.start_checkout(db, user, plan=Plan.TEAM, interval="monthly")
    second = await billing_service.get_subscription(db, user)
    assert second is not None
    assert second.id == first.id, "one row, reused"
    assert second.plan is Plan.TEAM, "an unpaid row follows the new intent"


async def test_the_wall_opens_checkout_on_the_plan_that_was_chosen(
    client: AsyncClient, db: AsyncSession, stripe: _FakeStripe
) -> None:
    """End to end, minus Stripe: choose at signup, pay at the wall."""
    await _register(client, "buys@example.com", plan="pro", interval="monthly")
    user = await _user(db, "buys@example.com")
    user.email_verified_at = utcnow()
    await db.flush()
    await _sign_in(client, "buys@example.com")

    summary = (await client.get(SUBSCRIPTION)).json()["data"]
    response = await client.post(
        "/api/v1/billing/checkout-session",
        json={"plan": summary["pending_plan"], "interval": summary["pending_interval"]},
    )
    assert response.status_code == 200, response.text
    assert response.json()["data"]["url"].startswith("https://checkout.stripe.test/")

    assert stripe.checkouts[0]["price_id"] == PRO_MONTHLY
    # The success redirect must not land on the dashboard: it races the
    # webhook, and a browser that wins is bounced straight back to the wall.
    assert "/checkout/done" in stripe.checkouts[0]["success_url"]


# ── Dunning reaches the same wall ───────────────────────────────────────────


async def test_a_past_due_subscription_walls_only_after_the_grace_period(
    client: AsyncClient, db: AsyncSession
) -> None:
    """Inside the grace period the account keeps working and is merely warned.
    Stripe's own dunning is still retrying the card, and locking someone out on
    day one of a failed payment is how a temporary card decline becomes a
    cancellation."""
    from app.services import billing_service

    user_id = await register_and_verify(client, db, email="dunned@example.com")
    user = await db.get(User, user_id)
    assert user is not None

    subscription = Subscription(
        user_id=user.id,
        stripe_customer_id="cus_dunned",
        stripe_subscription_id="sub_dunned",
        plan=Plan.PRO,
        status=SubscriptionStatus.PAST_DUE,
        seats=1,
        past_due_since=utcnow() - timedelta(days=1),
    )
    db.add(subscription)
    await db.flush()

    assert billing_service.payment_required(user, subscription) is False

    subscription.past_due_since = utcnow() - timedelta(
        days=settings.dunning_grace_days + 1
    )
    assert billing_service.payment_required(user, subscription) is True


# ── The pricing page ────────────────────────────────────────────────────────


async def test_self_serve_does_not_depend_on_this_environment_having_prices(
    client: AsyncClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Two different questions with two different answers on screen.

    Pro with no price configured is still a plan a signed-out visitor should be
    invited to sign up for; Enterprise never is. Collapsing both into one flag
    sent every unpriced plan to the "talk to us" page.

    The prices are blanked explicitly rather than assumed absent — whether the
    developer's `.env` happens to hold a Stripe key must not decide which
    branch this asserts.
    """
    for field in (
        "stripe_price_pro_monthly",
        "stripe_price_pro_annual",
        "stripe_price_team_monthly",
        "stripe_price_team_annual",
    ):
        monkeypatch.setattr(settings, field, "")

    rows = {row["key"]: row for row in (await client.get(PLANS)).json()["data"]}

    assert rows["pro"]["self_serve"] is True
    assert rows["team"]["self_serve"] is True
    assert rows["free"]["self_serve"] is False
    assert rows["enterprise"]["self_serve"] is False
    # Self-serve in principle, unbuyable here. The pricing page needs both
    # facts to send a signed-out visitor to signup and a signed-in one to an
    # explanation rather than to a 402.
    assert rows["pro"]["checkout"] is False


async def test_a_priced_plan_is_buyable(
    client: AsyncClient, stripe: _FakeStripe
) -> None:
    """The mirror, so the pair pins both directions rather than one."""
    rows = {row["key"]: row for row in (await client.get(PLANS)).json()["data"]}

    assert rows["pro"]["checkout"] is True
    # Still never Enterprise, however much is configured.
    assert rows["enterprise"]["checkout"] is False
