"""Checkout, webhooks, and the lifecycle they drive (M20).

The most important test in this file is `test_a_duplicate_delivery_applies_once`.
Stripe retries, retries are normal, and a duplicate that re-applies a plan
change bills or downgrades someone twice. Everything else here is a consequence
of getting that one right.

No test talks to Stripe. The client is a fake that records its arguments, and
webhook payloads are built by hand — which is also how the signature
verification is bypassed, since a real signature would need a real secret and
the thing under test is the handler, not the SDK's HMAC.
"""

from __future__ import annotations

from collections.abc import Iterator
from datetime import timedelta
from typing import Any

import pytest
from httpx import AsyncClient
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.config import settings
from app.core.database import utcnow
from app.integrations import stripe as stripe_integration
from app.models.billing import StripeEvent, Subscription, SubscriptionStatus
from app.models.user import Plan, User
from app.services import billing_service
from tests.conftest import GOOD_PASSWORD, register_and_verify

WEBHOOK = "/api/v1/billing/webhook"
CHECKOUT = "/api/v1/billing/checkout-session"

PRO_MONTHLY = "price_pro_monthly_test"
PRO_ANNUAL = "price_pro_annual_test"
TEAM_MONTHLY = "price_team_monthly_test"


# ── Fakes ───────────────────────────────────────────────────────────────────


class FakeStripe:
    """Records what it was asked to do. Asserted against, never called out."""

    def __init__(self) -> None:
        self.customers: list[dict[str, Any]] = []
        self.checkouts: list[dict[str, Any]] = []
        self.portals: list[dict[str, Any]] = []
        self.cancellations: list[dict[str, Any]] = []
        self.invoices: list[dict[str, Any]] = []

    async def create_customer(self, *, email: str, name: str, user_id: str) -> str:
        self.customers.append({"email": email, "name": name, "user_id": user_id})
        return f"cus_{len(self.customers)}"

    async def create_checkout_session(self, **kwargs: Any) -> tuple[str, str]:
        self.checkouts.append(kwargs)
        return "cs_test_1", "https://checkout.stripe.test/cs_test_1"

    async def create_portal_session(self, **kwargs: Any) -> str:
        self.portals.append(kwargs)
        return "https://portal.stripe.test/session"

    async def list_invoices(self, **kwargs: Any) -> list[dict[str, Any]]:
        return self.invoices

    async def cancel_at_period_end(self, **kwargs: Any) -> None:
        self.cancellations.append(kwargs)


@pytest.fixture
def stripe(monkeypatch: pytest.MonkeyPatch) -> Iterator[FakeStripe]:
    """A configured Stripe, without Stripe.

    The price ids are patched onto settings too: an environment with no prices
    configured refuses checkout by design, which is correct behaviour and
    useless for testing everything after checkout.
    """
    monkeypatch.setattr(settings, "stripe_secret_key", "sk_test_fake")
    monkeypatch.setattr(settings, "stripe_webhook_secret", "whsec_fake")
    monkeypatch.setattr(settings, "stripe_price_pro_monthly", PRO_MONTHLY)
    monkeypatch.setattr(settings, "stripe_price_pro_annual", PRO_ANNUAL)
    monkeypatch.setattr(settings, "stripe_price_team_monthly", TEAM_MONTHLY)

    fake = FakeStripe()
    stripe_integration.set_client(fake)
    yield fake
    stripe_integration.set_client(None)


@pytest.fixture
def unsigned(monkeypatch: pytest.MonkeyPatch) -> None:
    """Skip HMAC verification.

    Under test the payload is built in Python, so signing it would mean
    re-implementing Stripe's HMAC to hand it straight back to Stripe's own
    verifier — a test of the SDK, not of this application. What the endpoint
    does *with* a verified event is the thing worth asserting.
    """

    def _accept(payload: bytes, signature: str | None) -> dict[str, Any]:
        import json

        return dict(json.loads(payload))

    monkeypatch.setattr(stripe_integration, "verify_signature", _accept)


# ── Helpers ─────────────────────────────────────────────────────────────────


async def _sign_in(client: AsyncClient, db: AsyncSession, email: str) -> User:
    user_id = await register_and_verify(client, db, email=email)
    response = await client.post(
        "/api/v1/auth/login", json={"email": email, "password": GOOD_PASSWORD}
    )
    token = response.json()["data"]["tokens"]["access_token"]
    client.headers["Authorization"] = f"Bearer {token}"
    user = await db.get(User, user_id)
    assert user is not None
    return user


def _subscription_event(
    *,
    event_id: str,
    event_type: str = "customer.subscription.updated",
    subscription_id: str = "sub_stripe_1",
    customer_id: str = "cus_1",
    status: str = "active",
    price_id: str = PRO_MONTHLY,
    user_id: str | None = None,
    trial_end: int | None = None,
    cancel_at_period_end: bool = False,
    quantity: int = 1,
) -> dict[str, Any]:
    now = int(utcnow().timestamp())
    return {
        "id": event_id,
        "type": event_type,
        "data": {
            "object": {
                "id": subscription_id,
                "customer": customer_id,
                "status": status,
                "cancel_at_period_end": cancel_at_period_end,
                "trial_end": trial_end,
                "metadata": {"client_reference_id": user_id} if user_id else {},
                "items": {
                    "data": [
                        {
                            "price": {"id": price_id},
                            "quantity": quantity,
                            "current_period_start": now,
                            "current_period_end": now + 30 * 86_400,
                        }
                    ]
                },
            }
        },
    }


async def _post_event(client: AsyncClient, event: dict[str, Any]) -> int:
    response = await client.post(WEBHOOK, json=event, headers={"Stripe-Signature": "t=1,v1=x"})
    return response.status_code


# ── Plans ───────────────────────────────────────────────────────────────────


async def test_the_pricing_page_reads_its_limits_from_the_table(client: AsyncClient) -> None:
    """A marketing page with its own copy of the numbers drifts from the
    enforcement, and the drift is invisible until a user hits a wall the page
    said was not there."""
    response = await client.get("/api/v1/billing/plans")
    assert response.status_code == 200

    plans = {plan["key"]: plan for plan in response.json()["data"]}
    assert set(plans) == {"free", "pro", "team", "enterprise"}

    runs = next(
        limit for limit in plans["free"]["limits"] if limit["metric"] == "tool_runs_per_day"
    )
    assert runs["limit"] == 25


async def test_a_changed_limit_changes_the_pricing_page(
    client: AsyncClient, db: AsyncSession
) -> None:
    from app.models.billing import Metric
    from tests.conftest import set_limit

    await set_limit(db, plan=Plan.FREE, metric=Metric.TOOL_RUNS_PER_DAY, value=10)

    body = (await client.get("/api/v1/billing/plans")).json()["data"]
    plans = {plan["key"]: plan for plan in body}
    runs = next(
        limit for limit in plans["free"]["limits"] if limit["metric"] == "tool_runs_per_day"
    )
    assert runs["limit"] == 10, "the page must not have its own copy of the number"


async def test_unlimited_is_null_not_a_large_number(client: AsyncClient) -> None:
    body = (await client.get("/api/v1/billing/plans")).json()["data"]
    plans = {plan["key"]: plan for plan in body}
    runs = next(limit for limit in plans["pro"]["limits"] if limit["metric"] == "tool_runs_per_day")
    assert runs["limit"] is None


async def test_a_plan_with_no_configured_price_is_not_offered(client: AsyncClient) -> None:
    """No key, no checkout. Offering a button that 402s is worse than not
    offering one."""
    body = (await client.get("/api/v1/billing/plans")).json()["data"]
    plans = {plan["key"]: plan for plan in body}
    assert plans["pro"]["checkout"] is False
    assert plans["enterprise"]["checkout"] is False


async def test_plans_are_public(client: AsyncClient) -> None:
    """The pricing page cannot be crawled from behind a token."""
    assert (await client.get("/api/v1/billing/plans")).status_code == 200


# ── Checkout ────────────────────────────────────────────────────────────────


async def test_checkout_carries_the_user_id_as_client_reference_id(
    client: AsyncClient, db: AsyncSession, stripe: FakeStripe
) -> None:
    """Attribution cannot depend on the browser coming back."""
    user = await _sign_in(client, db, "buyer@example.com")

    response = await client.post(CHECKOUT, json={"plan": "pro", "interval": "monthly"})
    assert response.status_code == 200
    assert response.json()["data"]["url"].startswith("https://checkout.stripe.test/")

    assert len(stripe.checkouts) == 1
    assert stripe.checkouts[0]["client_reference_id"] == user.id
    assert stripe.checkouts[0]["price_id"] == PRO_MONTHLY


async def test_checkout_writes_an_incomplete_subscription_before_redirecting(
    client: AsyncClient, db: AsyncSession, stripe: FakeStripe
) -> None:
    """An abandoned checkout should leave an explicable state, not nothing."""
    user = await _sign_in(client, db, "abandoner@example.com")
    await client.post(CHECKOUT, json={"plan": "pro", "interval": "monthly"})

    subscription = await billing_service.get_subscription(db, user)
    assert subscription is not None
    assert subscription.status is SubscriptionStatus.INCOMPLETE
    assert subscription.plan is Plan.PRO
    assert user.plan is Plan.FREE, "an unpaid checkout must not grant anything"


async def test_a_second_checkout_reuses_the_row_and_the_customer(
    client: AsyncClient, db: AsyncSession, stripe: FakeStripe
) -> None:
    """One live subscription per user is a database constraint; a user who
    abandons Pro and then buys Team must not trip it."""
    user = await _sign_in(client, db, "switcher@example.com")

    await client.post(CHECKOUT, json={"plan": "pro", "interval": "monthly"})
    await client.post(CHECKOUT, json={"plan": "team", "interval": "monthly", "seats": 3})

    rows = (
        (await db.execute(select(Subscription).where(Subscription.user_id == user.id)))
        .scalars()
        .all()
    )
    assert len(rows) == 1
    assert rows[0].plan is Plan.TEAM
    assert rows[0].seats == 3
    assert len(stripe.customers) == 1, "a second customer would split the invoice history"


async def test_a_trial_is_offered_once(
    client: AsyncClient, db: AsyncSession, stripe: FakeStripe
) -> None:
    """Cancel-and-resubscribe is not a supported way to get another free week."""
    user = await _sign_in(client, db, "trialist@example.com")
    await client.post(CHECKOUT, json={"plan": "pro", "interval": "monthly"})
    assert stripe.checkouts[0]["trial_days"] == 7

    subscription = await billing_service.get_subscription(db, user)
    assert subscription is not None
    subscription.trial_ends_at = utcnow() + timedelta(days=7)
    await db.flush()

    await client.post(CHECKOUT, json={"plan": "pro", "interval": "annual"})
    assert stripe.checkouts[1]["trial_days"] == 0


async def test_checkout_without_stripe_configured_is_refused_clearly(
    client: AsyncClient, db: AsyncSession
) -> None:
    await _sign_in(client, db, "nostripe@example.com")
    response = await client.post(CHECKOUT, json={"plan": "pro", "interval": "monthly"})

    assert response.status_code == 402
    assert response.json()["error"]["details"]["reason"] == "billing_not_configured"


async def test_checkout_requires_an_account(client: AsyncClient, stripe: FakeStripe) -> None:
    assert (await client.post(CHECKOUT, json={"plan": "pro"})).status_code == 401


async def test_seats_are_refused_on_a_plan_that_is_not_per_seat(
    client: AsyncClient, db: AsyncSession, stripe: FakeStripe
) -> None:
    await _sign_in(client, db, "seats@example.com")
    response = await client.post(CHECKOUT, json={"plan": "pro", "interval": "monthly", "seats": 4})
    assert response.status_code == 422


# ── Webhooks: idempotency ───────────────────────────────────────────────────


async def test_a_duplicate_delivery_applies_once(
    client: AsyncClient, db: AsyncSession, stripe: FakeStripe, unsigned: None
) -> None:
    """The single most important test in this module.

    Stripe retries. A duplicate that re-applies a plan change upgrades or
    downgrades someone twice, and the second application is silent.
    """
    user = await _sign_in(client, db, "dupe@example.com")
    await client.post(CHECKOUT, json={"plan": "pro", "interval": "monthly"})
    subscription = await billing_service.get_subscription(db, user)
    assert subscription is not None

    event = _subscription_event(
        event_id="evt_duplicate",
        subscription_id="sub_dupe",
        customer_id=subscription.stripe_customer_id,
        user_id=user.id,
    )

    assert await _post_event(client, event) == 200
    assert await _post_event(client, event) == 200
    assert await _post_event(client, event) == 200

    rows = (
        (await db.execute(select(StripeEvent).where(StripeEvent.id == "evt_duplicate")))
        .scalars()
        .all()
    )
    assert len(rows) == 1, "one row per event id, whatever the delivery count"
    assert rows[0].processed_at is not None

    await db.refresh(user)
    assert user.plan is Plan.PRO

    subscriptions = (
        (await db.execute(select(Subscription).where(Subscription.user_id == user.id)))
        .scalars()
        .all()
    )
    assert len(subscriptions) == 1


async def test_an_unprocessed_event_is_retried_rather_than_skipped(
    client: AsyncClient, db: AsyncSession, stripe: FakeStripe, unsigned: None
) -> None:
    """A row that exists but never applied is a *failed* attempt, not a
    duplicate. Treating existence alone as "done" is how a customer who paid
    is never upgraded."""
    user = await _sign_in(client, db, "retryable@example.com")
    await client.post(CHECKOUT, json={"plan": "pro", "interval": "monthly"})
    subscription = await billing_service.get_subscription(db, user)
    assert subscription is not None

    event = _subscription_event(
        event_id="evt_retryable",
        subscription_id="sub_retry",
        customer_id=subscription.stripe_customer_id,
        user_id=user.id,
    )
    db.add(
        StripeEvent(
            id="evt_retryable",
            type=event["type"],
            payload=event,
            attempts=1,
            error="boom",
        )
    )
    await db.flush()

    assert await _post_event(client, event) == 200

    record = await db.get(StripeEvent, "evt_retryable")
    assert record is not None
    assert record.processed_at is not None
    assert record.error is None

    await db.refresh(user)
    assert user.plan is Plan.PRO


async def test_a_failing_handler_records_the_error_and_still_answers_200(
    client: AsyncClient,
    db: AsyncSession,
    stripe: FakeStripe,
    unsigned: None,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A 500 asks Stripe to retry work that is already recorded, and enough of
    them disables the endpoint. The failure is written down instead."""

    async def _explode(*_: object, **__: object) -> str:
        raise RuntimeError("handler is broken")

    monkeypatch.setattr(billing_service, "process_event", _explode)

    status = await _post_event(
        client, _subscription_event(event_id="evt_broken", user_id="usr_missing")
    )
    assert status == 200

    record = await db.get(StripeEvent, "evt_broken")
    assert record is not None
    assert record.processed_at is None
    assert record.attempts == 1
    assert record.error is not None
    assert "handler is broken" in record.error


async def test_an_unhandled_event_type_is_recorded_and_marked_done(
    client: AsyncClient, db: AsyncSession, unsigned: None
) -> None:
    """Stripe sends dozens of types this product has no opinion on. Retrying
    them forever would fill the retry job with work that can never succeed."""
    noise = {"id": "evt_noise", "type": "customer.discount.created", "data": {}}
    assert await _post_event(client, noise) == 200

    record = await db.get(StripeEvent, "evt_noise")
    assert record is not None
    assert record.processed_at is not None


async def test_an_unsigned_webhook_is_rejected(client: AsyncClient, db: AsyncSession) -> None:
    """Without the signature this endpoint is an unauthenticated
    'make me Pro' API."""
    response = await client.post(WEBHOOK, json={"id": "evt_forged", "type": "invoice.paid"})
    assert response.status_code == 502
    assert await db.get(StripeEvent, "evt_forged") is None, "nothing is written before verifying"


# ── Webhooks: lifecycle ─────────────────────────────────────────────────────


async def test_checkout_completed_attaches_the_stripe_subscription(
    client: AsyncClient, db: AsyncSession, stripe: FakeStripe, unsigned: None
) -> None:
    user = await _sign_in(client, db, "attach@example.com")
    await client.post(CHECKOUT, json={"plan": "pro", "interval": "monthly"})
    subscription = await billing_service.get_subscription(db, user)
    assert subscription is not None

    await _post_event(
        client,
        {
            "id": "evt_checkout",
            "type": "checkout.session.completed",
            "data": {
                "object": {
                    "client_reference_id": user.id,
                    "subscription": "sub_attached",
                    "customer": subscription.stripe_customer_id,
                }
            },
        },
    )

    await db.refresh(subscription)
    assert subscription.stripe_subscription_id == "sub_attached"


async def test_an_active_subscription_grants_the_plan(
    client: AsyncClient, db: AsyncSession, stripe: FakeStripe, unsigned: None
) -> None:
    user = await _sign_in(client, db, "granted@example.com")
    await client.post(CHECKOUT, json={"plan": "pro", "interval": "monthly"})
    subscription = await billing_service.get_subscription(db, user)
    assert subscription is not None

    await _post_event(
        client,
        _subscription_event(
            event_id="evt_active",
            customer_id=subscription.stripe_customer_id,
            user_id=user.id,
        ),
    )

    await db.refresh(user)
    await db.refresh(subscription)
    assert user.plan is Plan.PRO
    assert subscription.status is SubscriptionStatus.ACTIVE
    assert subscription.current_period_end is not None


async def test_a_trialing_subscription_grants_the_plan_too(
    client: AsyncClient, db: AsyncSession, stripe: FakeStripe, unsigned: None
) -> None:
    """A trial without the features is not a trial."""
    user = await _sign_in(client, db, "trialing@example.com")
    await client.post(CHECKOUT, json={"plan": "pro", "interval": "monthly"})
    subscription = await billing_service.get_subscription(db, user)
    assert subscription is not None

    await _post_event(
        client,
        _subscription_event(
            event_id="evt_trialing",
            customer_id=subscription.stripe_customer_id,
            user_id=user.id,
            status="trialing",
            trial_end=int((utcnow() + timedelta(days=7)).timestamp()),
        ),
    )

    await db.refresh(user)
    assert user.plan is Plan.PRO


async def test_an_upgrade_and_a_downgrade_are_the_same_code_path(
    client: AsyncClient, db: AsyncSession, stripe: FakeStripe, unsigned: None
) -> None:
    """The plan is read from the price on the subscription as it is now, so
    delivery order cannot invert a change."""
    user = await _sign_in(client, db, "mover@example.com")
    await client.post(CHECKOUT, json={"plan": "pro", "interval": "monthly"})
    subscription = await billing_service.get_subscription(db, user)
    assert subscription is not None
    customer = subscription.stripe_customer_id

    await _post_event(
        client,
        _subscription_event(event_id="evt_up_1", customer_id=customer, user_id=user.id),
    )
    await db.refresh(user)
    assert user.plan is Plan.PRO

    await _post_event(
        client,
        _subscription_event(
            event_id="evt_up_2", customer_id=customer, user_id=user.id, price_id=TEAM_MONTHLY
        ),
    )
    await db.refresh(user)
    assert user.plan is Plan.TEAM

    await _post_event(
        client,
        _subscription_event(
            event_id="evt_down", customer_id=customer, user_id=user.id, price_id=PRO_MONTHLY
        ),
    )
    await db.refresh(user)
    assert user.plan is Plan.PRO


async def test_a_deleted_subscription_drops_the_user_to_free(
    client: AsyncClient, db: AsyncSession, stripe: FakeStripe, unsigned: None
) -> None:
    user = await _sign_in(client, db, "canceller@example.com")
    await client.post(CHECKOUT, json={"plan": "pro", "interval": "monthly"})
    subscription = await billing_service.get_subscription(db, user)
    assert subscription is not None

    await _post_event(
        client,
        _subscription_event(
            event_id="evt_live", customer_id=subscription.stripe_customer_id, user_id=user.id
        ),
    )
    await _post_event(
        client,
        _subscription_event(
            event_id="evt_gone",
            event_type="customer.subscription.deleted",
            customer_id=subscription.stripe_customer_id,
            user_id=user.id,
            status="canceled",
        ),
    )

    await db.refresh(user)
    await db.refresh(subscription)
    assert user.plan is Plan.FREE
    assert subscription.status is SubscriptionStatus.CANCELED
    assert subscription.canceled_at is not None


async def test_a_failed_payment_starts_dunning_without_downgrading(
    client: AsyncClient,
    db: AsyncSession,
    stripe: FakeStripe,
    unsigned: None,
    outbox: Any,
) -> None:
    """Stripe is still retrying the card. Cutting access to a customer one
    retry from success costs more than the week of usage it saves."""
    user = await _sign_in(client, db, "dunned@example.com")
    await client.post(CHECKOUT, json={"plan": "pro", "interval": "monthly"})
    subscription = await billing_service.get_subscription(db, user)
    assert subscription is not None

    await _post_event(
        client,
        _subscription_event(
            event_id="evt_paid_up",
            customer_id=subscription.stripe_customer_id,
            user_id=user.id,
        ),
    )
    await _post_event(
        client,
        {
            "id": "evt_failed",
            "type": "invoice.payment_failed",
            "data": {"object": {"customer": subscription.stripe_customer_id}},
        },
    )

    await db.refresh(user)
    await db.refresh(subscription)
    assert subscription.status is SubscriptionStatus.PAST_DUE
    assert subscription.past_due_since is not None
    assert user.plan is Plan.PRO, "the grace window keeps the features"
    assert any("did not go through" in mail.subject for mail in outbox.outbox)


async def test_a_recovered_payment_clears_dunning(
    client: AsyncClient, db: AsyncSession, stripe: FakeStripe, unsigned: None
) -> None:
    user = await _sign_in(client, db, "recovered@example.com")
    await client.post(CHECKOUT, json={"plan": "pro", "interval": "monthly"})
    subscription = await billing_service.get_subscription(db, user)
    assert subscription is not None
    customer = subscription.stripe_customer_id

    await _post_event(
        client, _subscription_event(event_id="evt_r1", customer_id=customer, user_id=user.id)
    )
    await _post_event(
        client,
        {
            "id": "evt_r2",
            "type": "invoice.payment_failed",
            "data": {"object": {"customer": customer}},
        },
    )
    await _post_event(
        client,
        {"id": "evt_r3", "type": "invoice.paid", "data": {"object": {"customer": customer}}},
    )

    await db.refresh(subscription)
    assert subscription.status is SubscriptionStatus.ACTIVE
    assert subscription.past_due_since is None


async def test_a_trial_ending_soon_sends_one_email(
    client: AsyncClient, db: AsyncSession, stripe: FakeStripe, unsigned: None, outbox: Any
) -> None:
    user = await _sign_in(client, db, "reminded@example.com")
    await client.post(CHECKOUT, json={"plan": "pro", "interval": "monthly"})
    subscription = await billing_service.get_subscription(db, user)
    assert subscription is not None

    await _post_event(
        client,
        _subscription_event(
            event_id="evt_trial_end",
            event_type="customer.subscription.trial_will_end",
            customer_id=subscription.stripe_customer_id,
            user_id=user.id,
            status="trialing",
            trial_end=int((utcnow() + timedelta(days=3)).timestamp()),
        ),
    )

    assert any("trial ends" in mail.subject for mail in outbox.outbox)


# ── Subscription surface ────────────────────────────────────────────────────


async def test_the_billing_page_reports_the_real_state(
    client: AsyncClient, db: AsyncSession, stripe: FakeStripe, unsigned: None
) -> None:
    user = await _sign_in(client, db, "surface@example.com")
    await client.post(CHECKOUT, json={"plan": "pro", "interval": "monthly"})
    subscription = await billing_service.get_subscription(db, user)
    assert subscription is not None

    await _post_event(
        client,
        _subscription_event(
            event_id="evt_surface",
            customer_id=subscription.stripe_customer_id,
            user_id=user.id,
        ),
    )

    data = (await client.get("/api/v1/billing/subscription")).json()["data"]
    assert data["plan"] == "pro"
    assert data["status"] == "active"
    assert data["checkout_available"] is True
    assert data["current_period_end"]


async def test_cancellation_is_at_period_end_and_reversible(
    client: AsyncClient, db: AsyncSession, stripe: FakeStripe, unsigned: None
) -> None:
    """The period is paid for and stays. 'I changed my mind' happens more often
    than the cancellation itself."""
    user = await _sign_in(client, db, "reversible@example.com")
    await client.post(CHECKOUT, json={"plan": "pro", "interval": "monthly"})
    subscription = await billing_service.get_subscription(db, user)
    assert subscription is not None
    await _post_event(
        client,
        _subscription_event(
            event_id="evt_rev", customer_id=subscription.stripe_customer_id, user_id=user.id
        ),
    )

    cancelled = await client.post("/api/v1/billing/cancellation", json={"cancel": True})
    assert cancelled.json()["data"]["cancel_at_period_end"] is True
    await db.refresh(user)
    assert user.plan is Plan.PRO, "cancelling must not take the plan away immediately"

    resumed = await client.post("/api/v1/billing/cancellation", json={"cancel": False})
    assert resumed.json()["data"]["cancel_at_period_end"] is False
    assert [call["cancel"] for call in stripe.cancellations] == [True, False]


async def test_the_portal_needs_a_billing_account(
    client: AsyncClient, db: AsyncSession, stripe: FakeStripe
) -> None:
    await _sign_in(client, db, "noportal@example.com")
    assert (await client.post("/api/v1/billing/portal-session")).status_code == 404


async def test_the_portal_returns_a_url_once_there_is_one(
    client: AsyncClient, db: AsyncSession, stripe: FakeStripe
) -> None:
    await _sign_in(client, db, "portal@example.com")
    await client.post(CHECKOUT, json={"plan": "pro", "interval": "monthly"})

    response = await client.post("/api/v1/billing/portal-session")
    assert response.status_code == 200
    assert response.json()["data"]["url"].startswith("https://portal.stripe.test/")
    assert stripe.portals[0]["return_url"].endswith("/settings/billing")


async def test_invoices_are_empty_rather_than_an_error_without_a_customer(
    client: AsyncClient, db: AsyncSession
) -> None:
    await _sign_in(client, db, "noinvoices@example.com")
    response = await client.get("/api/v1/billing/invoices")
    assert response.status_code == 200
    assert response.json()["data"] == []
