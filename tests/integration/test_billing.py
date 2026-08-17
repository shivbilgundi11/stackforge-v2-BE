"""Checkout, webhooks, and the lifecycle they drive (M20, Razorpay since D-50).

The most important test in this file is `test_a_duplicate_delivery_applies_once`.
Providers retry, retries are normal, and a duplicate that re-applies a plan
change bills or downgrades someone twice. Everything else here is a consequence
of getting that one right.

No test talks to Razorpay. The client is a fake that records its arguments, and
webhook payloads are built by hand — which is also how signature verification is
bypassed, since a real signature would need a real secret and the thing under
test is the handler, not the HMAC. `tests/unit/test_razorpay_signature.py` is
where the HMAC itself is exercised, for real.

## What changed with the provider

Razorpay has no checkout session, so there is no "checkout completed" event to
attach a subscription: the subscription exists from the moment `start_checkout`
runs, and the customer authorizes it afterwards. Every lifecycle event carries
the same subscription entity with its current status, so one handler covers all
nine event types and these tests drive them by varying `status` rather than by
varying the event name.
"""

from __future__ import annotations

import json
from collections.abc import Iterator
from datetime import timedelta
from typing import Any

import pytest
from httpx import AsyncClient
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.config import settings
from app.core.database import utcnow
from app.integrations import razorpay as razorpay_integration
from app.models.billing import BillingEvent, Subscription, SubscriptionStatus
from app.models.user import Plan, User
from app.services import billing_service
from tests.conftest import GOOD_PASSWORD, register_and_verify

WEBHOOK = "/api/v1/billing/webhook"
CHECKOUT = "/api/v1/billing/checkout-session"

PRO_MONTHLY = "plan_pro_monthly_test"
PRO_ANNUAL = "plan_pro_annual_test"
TEAM_MONTHLY = "plan_team_monthly_test"


# ── Fakes ───────────────────────────────────────────────────────────────────


class FakeRazorpay:
    """Records what it was asked to do. Asserted against, never called out."""

    def __init__(self) -> None:
        self.customers: list[dict[str, Any]] = []
        self.subscriptions: list[dict[str, Any]] = []
        self.cancellations: list[dict[str, Any]] = []
        self.scheduled_cancellations: list[dict[str, Any]] = []
        self.quantities: list[dict[str, Any]] = []
        self.invoices: list[dict[str, Any]] = []
        self.entities: dict[str, dict[str, Any]] = {}

    async def create_customer(self, *, email: str, name: str, user_id: str) -> str:
        self.customers.append({"email": email, "name": name, "user_id": user_id})
        return f"cust_{len(self.customers)}"

    async def create_subscription(self, **kwargs: Any) -> str:
        self.subscriptions.append(kwargs)
        return f"sub_test_{len(self.subscriptions)}"

    async def fetch_subscription(self, **kwargs: Any) -> dict[str, Any]:
        """What the provider would say if asked directly.

        Seeded per subscription id by the reconciliation tests, which is the
        whole point of that path: it is the only route to the truth when no
        webhook ever arrives.
        """
        return self.entities.get(str(kwargs.get("subscription_id")), {})

    async def list_invoices(self, **kwargs: Any) -> list[dict[str, Any]]:
        return self.invoices

    async def cancel_subscription(self, **kwargs: Any) -> dict[str, Any]:
        self.cancellations.append(kwargs)
        return {}

    async def cancel_scheduled_changes(self, **kwargs: Any) -> dict[str, Any]:
        self.scheduled_cancellations.append(kwargs)
        return {}

    async def update_subscription_quantity(self, **kwargs: Any) -> dict[str, Any]:
        self.quantities.append(kwargs)
        return {}


@pytest.fixture
def razorpay(monkeypatch: pytest.MonkeyPatch) -> Iterator[FakeRazorpay]:
    """A configured Razorpay, without Razorpay.

    The plan ids are patched onto settings too: an environment with no plans
    configured refuses checkout by design, which is correct behaviour and
    useless for testing everything after checkout.
    """
    monkeypatch.setattr(settings, "razorpay_enabled", True)
    monkeypatch.setattr(settings, "razorpay_key_id", "rzp_test_fake")
    monkeypatch.setattr(settings, "razorpay_key_secret", "secret_fake")
    monkeypatch.setattr(settings, "razorpay_webhook_secret", "whsec_fake")
    monkeypatch.setattr(settings, "razorpay_plan_pro_monthly", PRO_MONTHLY)
    monkeypatch.setattr(settings, "razorpay_plan_pro_annual", PRO_ANNUAL)
    monkeypatch.setattr(settings, "razorpay_plan_team_monthly", TEAM_MONTHLY)

    fake = FakeRazorpay()
    razorpay_integration.set_client(fake)
    yield fake
    razorpay_integration.set_client(None)


@pytest.fixture
def unsigned(monkeypatch: pytest.MonkeyPatch) -> None:
    """Skip HMAC verification.

    Under test the payload is built in Python, so signing it would mean
    re-implementing the HMAC to hand it straight back to our own verifier — a
    test of `hmac`, not of this application. What the endpoint does *with* a
    verified event is the thing worth asserting, and the verifier itself has
    its own unit tests.
    """

    def _accept(payload: bytes, signature: str | None) -> dict[str, Any]:
        return dict(json.loads(payload))

    monkeypatch.setattr(razorpay_integration, "verify_signature", _accept)


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
    event: str = "subscription.activated",
    subscription_id: str = "sub_test_1",
    customer_id: str = "cust_1",
    status: str = "active",
    plan_id: str = PRO_MONTHLY,
    user_id: str | None = None,
    start_at: int | None = None,
    end_at: int | None = None,
    quantity: int = 1,
) -> dict[str, Any]:
    """One Razorpay webhook body.

    Nested two deep and keyed by entity name, which is the shape the real thing
    sends — `payload.subscription.entity`. Getting that wrong is exactly the
    class of bug these tests exist to catch.
    """
    now = int(utcnow().timestamp())
    return {
        "entity": "event",
        "event": event,
        "contains": ["subscription"],
        "payload": {
            "subscription": {
                "entity": {
                    "id": subscription_id,
                    "entity": "subscription",
                    "plan_id": plan_id,
                    "customer_id": customer_id,
                    "status": status,
                    "quantity": quantity,
                    "current_start": now,
                    "current_end": now + 30 * 86_400,
                    "start_at": start_at,
                    "end_at": end_at,
                    "total_count": 120,
                    "paid_count": 1,
                    "notes": {"user_id": user_id} if user_id else {},
                }
            }
        },
        "created_at": now,
    }


async def _post_event(
    client: AsyncClient, event: dict[str, Any], *, event_id: str = "evt_test_1"
) -> int:
    response = await client.post(
        WEBHOOK,
        json=event,
        headers={"X-Razorpay-Signature": "deadbeef", "X-Razorpay-Event-Id": event_id},
    )
    return response.status_code


# ── Plans ───────────────────────────────────────────────────────────────────


async def test_the_pricing_page_reads_its_limits_from_the_table(client: AsyncClient) -> None:
    """A marketing page with its own copy of the numbers drifts from the
    enforcement, and the drift is invisible until a user hits a wall the page
    said was not there."""
    body = (await client.get("/api/v1/billing/plans")).json()["data"]
    plans = {plan["key"]: plan for plan in body}

    assert set(plans) == {"free", "pro", "team", "enterprise"}
    assert plans["pro"]["monthly_minor"] == 159_900
    assert plans["pro"]["currency"] == "inr"
    runs = next(row for row in plans["free"]["limits"] if row["metric"] == "tool_runs_per_day")
    assert runs["limit"] == 25


async def test_a_changed_limit_changes_the_pricing_page(
    client: AsyncClient, db: AsyncSession
) -> None:
    """The whole reason the limits live in a table."""
    from app.models.billing import Metric, PlanQuota

    row = await db.scalar(
        select(PlanQuota).where(
            PlanQuota.plan == Plan.FREE,
            PlanQuota.metric == Metric.TOOL_RUNS_PER_DAY,
            PlanQuota.anonymous.is_(False),
        )
    )
    assert row is not None
    row.limit_value = 40
    await db.flush()

    body = (await client.get("/api/v1/billing/plans")).json()["data"]
    free = next(plan for plan in body if plan["key"] == "free")
    runs = next(item for item in free["limits"] if item["metric"] == "tool_runs_per_day")
    assert runs["limit"] == 40


async def test_unlimited_is_null_not_a_large_number(client: AsyncClient) -> None:
    """A meter reading '3 of 999999' is a meter nobody believes."""
    body = (await client.get("/api/v1/billing/plans")).json()["data"]
    pro = next(plan for plan in body if plan["key"] == "pro")
    runs = next(item for item in pro["limits"] if item["metric"] == "tool_runs_per_day")
    assert runs["limit"] is None


async def test_a_plan_with_no_configured_price_is_not_offered(
    client: AsyncClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    """No plan id, no checkout. Offering a button that 402s is worse than not
    offering one.

    The plan ids are blanked explicitly. This used to rely on the developer's
    `.env` having no keys, which meant the assertion inverted the moment one
    was added — the test was reading the machine, not the code.
    """
    for field in (
        "razorpay_plan_pro_monthly",
        "razorpay_plan_pro_annual",
        "razorpay_plan_team_monthly",
        "razorpay_plan_team_annual",
    ):
        monkeypatch.setattr(settings, field, "")

    body = (await client.get("/api/v1/billing/plans")).json()["data"]
    plans = {plan["key"]: plan for plan in body}
    assert plans["pro"]["checkout"] is False
    assert plans["enterprise"]["checkout"] is False


async def test_plans_are_public(client: AsyncClient) -> None:
    """The pricing page cannot be crawled from behind a token."""
    assert (await client.get("/api/v1/billing/plans")).status_code == 200


# ── Checkout ────────────────────────────────────────────────────────────────


async def test_checkout_carries_the_user_id_in_the_notes(
    client: AsyncClient, db: AsyncSession, razorpay: FakeRazorpay
) -> None:
    """Attribution must not depend on the browser coming back. Razorpay echoes
    `notes` on every subscription event, which is how a webhook finds the
    account without racing the redirect."""
    user = await _sign_in(client, db, "reference@example.com")

    response = await client.post(CHECKOUT, json={"plan": "pro", "interval": "monthly"})
    assert response.status_code == 200, response.text
    assert response.json()["data"]["subscription_id"] == "sub_test_1"

    assert razorpay.subscriptions[0]["notes"] == {"user_id": user.id}
    assert razorpay.subscriptions[0]["plan_id"] == PRO_MONTHLY


async def test_checkout_writes_an_incomplete_subscription_before_redirecting(
    client: AsyncClient, db: AsyncSession, razorpay: FakeRazorpay
) -> None:
    """The row is what a webhook arriving during the redirect attaches to, and
    it means an abandoned authorization leaves a visible, explicable state
    rather than nothing at all."""
    user = await _sign_in(client, db, "incomplete@example.com")
    await client.post(CHECKOUT, json={"plan": "pro", "interval": "monthly"})

    subscription = await billing_service.get_subscription(db, user)
    assert subscription is not None
    assert subscription.status is SubscriptionStatus.INCOMPLETE
    # Razorpay creates the subscription up front, so unlike Stripe the id is
    # already on the row before anybody has paid.
    assert subscription.provider_subscription_id == "sub_test_1"
    assert user.plan is Plan.FREE, "creating a subscription is not being on it"


async def test_a_second_checkout_reuses_the_row_and_the_customer(
    client: AsyncClient, db: AsyncSession, razorpay: FakeRazorpay
) -> None:
    """One live subscription per user is a database invariant, so an abandoned
    Pro attempt followed by a Team attempt has to update rather than insert."""
    user = await _sign_in(client, db, "reuse@example.com")
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
    assert len(razorpay.customers) == 1, "one customer per user, forever"


async def test_checkout_returns_what_the_browser_opens_razorpay_with(
    client: AsyncClient, db: AsyncSession, razorpay: FakeRazorpay
) -> None:
    """A subscription id and a key, never a URL.

    The hosted page this used to return takes no callback, so anyone who
    authorized on it stayed there (D-52). Checkout is opened in the browser
    instead, and it needs both of these.
    """
    await _sign_in(client, db, "opens-checkout@example.com")

    response = await client.post(CHECKOUT, json={"plan": "pro", "interval": "monthly"})

    assert response.status_code == 200
    body = response.json()["data"]
    assert body["subscription_id"] == "sub_test_1"
    assert body["key_id"]
    assert "url" not in body


async def test_a_later_event_without_a_customer_does_not_erase_it(
    client: AsyncClient, db: AsyncSession, razorpay: FakeRazorpay, unsigned: None
) -> None:
    """The link is written once and kept.

    Not every entity carries every field, and treating a missing one as an
    instruction to clear would drop the only handle we have on who paid.
    """
    user = await _sign_in(client, db, "keeps-customer@example.com")
    await client.post(CHECKOUT, json={"plan": "pro", "interval": "monthly"})
    await _post_event(
        client,
        _subscription_event(customer_id="cust_kept", user_id=user.id),
        event_id="evt_first",
    )

    event = _subscription_event(event="subscription.charged", user_id=user.id)
    del event["payload"]["subscription"]["entity"]["customer_id"]
    assert await _post_event(client, event, event_id="evt_second") == 200

    subscription = await db.scalar(select(Subscription).where(Subscription.user_id == user.id))
    assert subscription is not None
    assert subscription.provider_customer_id == "cust_kept"


async def test_checkout_never_delays_the_first_charge(
    client: AsyncClient, db: AsyncSession, razorpay: FakeRazorpay
) -> None:
    """The product sells no trial, and `start_at` is the only thing that could
    create one.

    This guards a failure that is invisible from the code: a future `start_at`
    makes Razorpay treat the subscription as mandate-first, so Checkout shows
    its token authorization amount (₹5) rather than the ₹1,599 on the pricing
    page. The customer sees a price nobody wrote down.
    """
    user = await _sign_in(client, db, "buyer@example.com")

    await client.post(CHECKOUT, json={"plan": "pro", "interval": "monthly"})
    assert razorpay.subscriptions[0]["start_at"] is None

    subscription = await billing_service.get_subscription(db, user)
    assert subscription is not None
    assert subscription.trial_ends_at is None

    # Still none for an account carrying a trial from before trials were
    # withdrawn — that history must not change what a new checkout charges.
    subscription.trial_ends_at = utcnow() + timedelta(days=3)
    await db.flush()

    await client.post(CHECKOUT, json={"plan": "team", "interval": "monthly"})
    assert razorpay.subscriptions[1]["start_at"] is None


async def test_checkout_without_razorpay_configured_is_refused_clearly(
    client: AsyncClient, db: AsyncSession
) -> None:
    """A 402 that says why, not a 500."""
    await _sign_in(client, db, "unconfigured@example.com")
    response = await client.post(CHECKOUT, json={"plan": "pro", "interval": "monthly"})
    assert response.status_code == 402
    assert response.json()["error"]["details"]["reason"] == "billing_not_configured"


async def test_checkout_requires_an_account(client: AsyncClient, razorpay: FakeRazorpay) -> None:
    assert (await client.post(CHECKOUT, json={"plan": "pro"})).status_code == 401


async def test_seats_are_refused_on_a_plan_that_is_not_per_seat(
    client: AsyncClient, db: AsyncSession, razorpay: FakeRazorpay
) -> None:
    await _sign_in(client, db, "seats@example.com")
    response = await client.post(CHECKOUT, json={"plan": "pro", "seats": 4})
    assert response.status_code == 422


# ── Idempotency ─────────────────────────────────────────────────────────────


async def test_a_duplicate_delivery_applies_once(
    client: AsyncClient, db: AsyncSession, razorpay: FakeRazorpay, unsigned: None
) -> None:
    """The test this whole module is arranged around.

    Providers retry. A duplicate that re-applies a plan change upgrades someone
    twice or downgrades them mid-period, and the event id is the only thing
    standing between a retry and that.
    """
    user = await _sign_in(client, db, "dupe@example.com")
    await client.post(CHECKOUT, json={"plan": "pro", "interval": "monthly"})
    subscription = await billing_service.get_subscription(db, user)
    assert subscription is not None

    event = _subscription_event(
        customer_id=subscription.provider_customer_id, user_id=user.id, quantity=1
    )

    assert await _post_event(client, event, event_id="evt_dupe") == 200
    await db.refresh(user)
    assert user.plan is Plan.PRO

    # The same delivery again, byte for byte.
    assert await _post_event(client, event, event_id="evt_dupe") == 200

    rows = (
        (await db.execute(select(BillingEvent).where(BillingEvent.id == "evt_dupe")))
        .scalars()
        .all()
    )
    assert len(rows) == 1, "one row per event id"
    assert rows[0].processed_at is not None


async def test_an_unprocessed_event_is_retried_rather_than_skipped(
    client: AsyncClient, db: AsyncSession, razorpay: FakeRazorpay, unsigned: None
) -> None:
    """The case a naive 'insert or skip' gets wrong.

    A handler that raised leaves a row behind. If existence alone meant 'done',
    the provider's next retry would be discarded and the customer who paid
    would never be upgraded.
    """
    user = await _sign_in(client, db, "retry@example.com")
    await client.post(CHECKOUT, json={"plan": "pro", "interval": "monthly"})
    subscription = await billing_service.get_subscription(db, user)
    assert subscription is not None

    event = _subscription_event(
        customer_id=subscription.provider_customer_id, user_id=user.id
    )
    # The row a previous, failed attempt left behind — same id, same body, and
    # no `processed_at`.
    db.add(
        BillingEvent(
            id="evt_unprocessed",
            type=event["event"],
            payload=event,
            attempts=1,
            error="boom",
        )
    )
    await db.flush()

    assert await _post_event(client, event, event_id="evt_unprocessed") == 200

    await db.refresh(user)
    assert user.plan is Plan.PRO, "the retry was discarded"
    record = await db.get(BillingEvent, "evt_unprocessed")
    assert record is not None
    assert record.processed_at is not None
    assert record.error is None


async def test_a_failing_handler_records_the_error_and_still_answers_200(
    client: AsyncClient,
    db: AsyncSession,
    razorpay: FakeRazorpay,
    unsigned: None,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A 500 asks the provider to retry work already recorded, and after enough
    failures a provider disables the endpoint — a far worse state than a row
    with an error on it that a job will pick up in an hour."""

    async def _boom(*args: Any, **kwargs: Any) -> str:
        raise RuntimeError("handler exploded")

    monkeypatch.setattr(billing_service, "_on_subscription_changed", _boom)

    await _sign_in(client, db, "failing@example.com")
    assert (
        await _post_event(client, _subscription_event(), event_id="evt_fail") == 200
    )

    record = await db.get(BillingEvent, "evt_fail")
    assert record is not None
    assert record.processed_at is None
    assert record.attempts == 1
    assert record.error is not None
    assert "handler exploded" in record.error


async def test_an_unhandled_event_type_is_recorded_and_marked_done(
    client: AsyncClient, db: AsyncSession, razorpay: FakeRazorpay, unsigned: None
) -> None:
    """Razorpay sends dozens of types this product has no opinion on. Retrying
    them forever would fill the retry job with work that can never succeed."""
    assert (
        await _post_event(
            client,
            {"entity": "event", "event": "payment.authorized", "payload": {}},
            event_id="evt_noise",
        )
        == 200
    )

    record = await db.get(BillingEvent, "evt_noise")
    assert record is not None
    assert record.processed_at is not None


async def test_an_unsigned_webhook_is_rejected(client: AsyncClient, db: AsyncSession) -> None:
    """The endpoint's entire authentication. Without it this is an
    unauthenticated 'make me Pro' API."""
    response = await client.post(WEBHOOK, json=_subscription_event())
    assert response.status_code != 200
    assert await db.get(BillingEvent, "evt_test_1") is None


async def test_a_webhook_without_an_event_id_is_rejected(
    client: AsyncClient, db: AsyncSession, razorpay: FakeRazorpay, unsigned: None
) -> None:
    """Razorpay puts the id in a header, not the body. Without one there is no
    idempotency key, and every redelivery would look new."""
    response = await client.post(
        WEBHOOK, json=_subscription_event(), headers={"X-Razorpay-Signature": "deadbeef"}
    )
    assert response.status_code == 422


# ── Lifecycle ───────────────────────────────────────────────────────────────


async def test_an_active_subscription_grants_the_plan(
    client: AsyncClient, db: AsyncSession, razorpay: FakeRazorpay, unsigned: None
) -> None:
    user = await _sign_in(client, db, "active@example.com")
    await client.post(CHECKOUT, json={"plan": "pro", "interval": "monthly"})
    subscription = await billing_service.get_subscription(db, user)
    assert subscription is not None

    await _post_event(
        client,
        _subscription_event(
            customer_id=subscription.provider_customer_id, user_id=user.id
        ),
        event_id="evt_active",
    )

    await db.refresh(user)
    await db.refresh(subscription)
    assert user.plan is Plan.PRO
    assert subscription.status is SubscriptionStatus.ACTIVE
    assert subscription.current_period_end is not None


async def test_an_authenticated_subscription_is_the_trial_and_grants_the_plan(
    client: AsyncClient, db: AsyncSession, razorpay: FakeRazorpay, unsigned: None
) -> None:
    """`authenticated` means the mandate is confirmed and the first charge is
    waiting for `start_at`. A trial without the features is not a trial."""
    user = await _sign_in(client, db, "trialing@example.com")
    await client.post(CHECKOUT, json={"plan": "pro", "interval": "monthly"})
    subscription = await billing_service.get_subscription(db, user)
    assert subscription is not None

    start_at = int((utcnow() + timedelta(days=7)).timestamp())
    await _post_event(
        client,
        _subscription_event(
            event="subscription.authenticated",
            status="authenticated",
            customer_id=subscription.provider_customer_id,
            user_id=user.id,
            start_at=start_at,
        ),
        event_id="evt_auth",
    )

    await db.refresh(user)
    await db.refresh(subscription)
    assert user.plan is Plan.PRO
    assert subscription.status is SubscriptionStatus.TRIALING
    assert subscription.trial_ends_at is not None


async def test_an_upgrade_and_a_downgrade_are_the_same_code_path(
    client: AsyncClient, db: AsyncSession, razorpay: FakeRazorpay, unsigned: None
) -> None:
    """Reading the tier from the plan id rather than from event ordering is
    what makes this one path: the subscription says what it is now, and whether
    that is up or down from before does not matter."""
    user = await _sign_in(client, db, "moves@example.com")
    await client.post(CHECKOUT, json={"plan": "pro", "interval": "monthly"})
    subscription = await billing_service.get_subscription(db, user)
    assert subscription is not None
    customer_id = subscription.provider_customer_id

    await _post_event(
        client,
        _subscription_event(customer_id=customer_id, user_id=user.id, plan_id=TEAM_MONTHLY),
        event_id="evt_up",
    )
    await db.refresh(user)
    assert user.plan is Plan.TEAM

    await _post_event(
        client,
        _subscription_event(customer_id=customer_id, user_id=user.id, plan_id=PRO_MONTHLY),
        event_id="evt_down",
    )
    await db.refresh(user)
    assert user.plan is Plan.PRO


async def test_a_cancelled_subscription_drops_the_user_to_free(
    client: AsyncClient, db: AsyncSession, razorpay: FakeRazorpay, unsigned: None
) -> None:
    """And touches nothing else. A downgrade is reversible by paying again."""
    user = await _sign_in(client, db, "cancelled@example.com")
    await client.post(CHECKOUT, json={"plan": "pro", "interval": "monthly"})
    subscription = await billing_service.get_subscription(db, user)
    assert subscription is not None
    customer_id = subscription.provider_customer_id

    await _post_event(
        client,
        _subscription_event(customer_id=customer_id, user_id=user.id),
        event_id="evt_on",
    )
    await db.refresh(user)
    assert user.plan is Plan.PRO

    await _post_event(
        client,
        _subscription_event(
            event="subscription.cancelled",
            status="cancelled",
            customer_id=customer_id,
            user_id=user.id,
        ),
        event_id="evt_off",
    )
    await db.refresh(user)
    assert user.plan is Plan.FREE
    assert subscription.canceled_at is not None


async def test_a_failed_payment_starts_dunning_without_downgrading(
    client: AsyncClient, db: AsyncSession, razorpay: FakeRazorpay, unsigned: None
) -> None:
    """The features stay on through the grace window. Cutting access to a
    customer whose payment is one retry from succeeding costs more than the
    week of usage it saves."""
    user = await _sign_in(client, db, "dunned@example.com")
    await client.post(CHECKOUT, json={"plan": "pro", "interval": "monthly"})
    subscription = await billing_service.get_subscription(db, user)
    assert subscription is not None
    customer_id = subscription.provider_customer_id

    await _post_event(
        client,
        _subscription_event(customer_id=customer_id, user_id=user.id),
        event_id="evt_paid",
    )
    await _post_event(
        client,
        _subscription_event(
            event="subscription.pending",
            status="pending",
            customer_id=customer_id,
            user_id=user.id,
        ),
        event_id="evt_pending",
    )

    await db.refresh(user)
    await db.refresh(subscription)
    assert subscription.status is SubscriptionStatus.PAST_DUE
    assert subscription.past_due_since is not None
    assert user.plan is Plan.PRO, "dunning must not downgrade on the first failure"


async def test_halted_is_past_due_not_cancelled(
    client: AsyncClient, db: AsyncSession, razorpay: FakeRazorpay, unsigned: None
) -> None:
    """Razorpay stops retrying at `halted`, but the grace period is ours to
    run. Treating it as cancelled would downgrade on the day the card
    bounced."""
    user = await _sign_in(client, db, "halted@example.com")
    await client.post(CHECKOUT, json={"plan": "pro", "interval": "monthly"})
    subscription = await billing_service.get_subscription(db, user)
    assert subscription is not None
    customer_id = subscription.provider_customer_id

    # Paying first, so there is a plan for halting to threaten.
    await _post_event(
        client,
        _subscription_event(customer_id=customer_id, user_id=user.id),
        event_id="evt_halted_paid",
    )

    await _post_event(
        client,
        _subscription_event(
            event="subscription.halted",
            status="halted",
            customer_id=customer_id,
            user_id=user.id,
        ),
        event_id="evt_halted",
    )

    await db.refresh(user)
    await db.refresh(subscription)
    assert subscription.status is SubscriptionStatus.PAST_DUE
    assert user.plan is Plan.PRO


async def test_a_recovered_payment_clears_dunning(
    client: AsyncClient, db: AsyncSession, razorpay: FakeRazorpay, unsigned: None
) -> None:
    """A user whose card recovered should not keep a 'payment failed' banner."""
    user = await _sign_in(client, db, "recovered@example.com")
    await client.post(CHECKOUT, json={"plan": "pro", "interval": "monthly"})
    subscription = await billing_service.get_subscription(db, user)
    assert subscription is not None
    customer_id = subscription.provider_customer_id

    await _post_event(
        client,
        _subscription_event(
            event="subscription.pending",
            status="pending",
            customer_id=customer_id,
            user_id=user.id,
        ),
        event_id="evt_fail2",
    )
    await db.refresh(subscription)
    assert subscription.past_due_since is not None

    await _post_event(
        client,
        _subscription_event(
            event="subscription.charged", customer_id=customer_id, user_id=user.id
        ),
        event_id="evt_charged",
    )

    await db.refresh(subscription)
    assert subscription.status is SubscriptionStatus.ACTIVE
    assert subscription.past_due_since is None


async def test_a_failed_payment_emails_once_not_once_per_retry(
    client: AsyncClient,
    db: AsyncSession,
    razorpay: FakeRazorpay,
    unsigned: None,
    outbox: Any,
) -> None:
    """Razorpay re-sends `pending` on every retry. The mail is driven off the
    transition into past due, not off the delivery."""
    user = await _sign_in(client, db, "onceonly@example.com")
    await client.post(CHECKOUT, json={"plan": "pro", "interval": "monthly"})
    subscription = await billing_service.get_subscription(db, user)
    assert subscription is not None
    customer_id = subscription.provider_customer_id

    before = len(outbox.outbox)
    for index in range(3):
        await _post_event(
            client,
            _subscription_event(
                event="subscription.pending",
                status="pending",
                customer_id=customer_id,
                user_id=user.id,
            ),
            event_id=f"evt_pending_{index}",
        )

    dunning = [
        message
        for message in outbox.outbox[before:]
        if "payment" in message.subject.lower()
    ]
    assert len(dunning) == 1


# ── Surfaces ────────────────────────────────────────────────────────────────


async def test_the_billing_page_reports_the_real_state(
    client: AsyncClient, db: AsyncSession, razorpay: FakeRazorpay, unsigned: None
) -> None:
    user = await _sign_in(client, db, "surface@example.com")
    await client.post(CHECKOUT, json={"plan": "pro", "interval": "monthly"})
    subscription = await billing_service.get_subscription(db, user)
    assert subscription is not None

    await _post_event(
        client,
        _subscription_event(
            customer_id=subscription.provider_customer_id, user_id=user.id
        ),
        event_id="evt_surface",
    )

    data = (await client.get("/api/v1/billing/subscription")).json()["data"]
    assert data["plan"] == "pro"
    assert data["status"] == "active"
    assert data["checkout_available"] is True
    assert data["current_period_end"]


async def test_cancellation_is_at_period_end_and_reversible(
    client: AsyncClient, db: AsyncSession, razorpay: FakeRazorpay, unsigned: None
) -> None:
    """The period is paid for and stays. 'I changed my mind' happens more often
    than the cancellation itself — and on Razorpay the undo is a different call
    rather than the same one with a flag, because a scheduled cancellation is a
    pending change rather than a boolean."""
    user = await _sign_in(client, db, "reversible@example.com")
    await client.post(CHECKOUT, json={"plan": "pro", "interval": "monthly"})
    subscription = await billing_service.get_subscription(db, user)
    assert subscription is not None
    await _post_event(
        client,
        _subscription_event(
            customer_id=subscription.provider_customer_id, user_id=user.id
        ),
        event_id="evt_rev",
    )

    cancelled = await client.post("/api/v1/billing/cancellation", json={"cancel": True})
    assert cancelled.json()["data"]["cancel_at_period_end"] is True
    await db.refresh(user)
    assert user.plan is Plan.PRO, "cancelling must not take the plan away immediately"

    resumed = await client.post("/api/v1/billing/cancellation", json={"cancel": False})
    assert resumed.json()["data"]["cancel_at_period_end"] is False

    assert [call["at_cycle_end"] for call in razorpay.cancellations] == [True]
    assert len(razorpay.scheduled_cancellations) == 1


async def test_there_is_no_billing_portal(client: AsyncClient, db: AsyncSession) -> None:
    """It went with Stripe (D-50). Razorpay has no hosted portal, and the two
    things the portal did — invoices and cancellation — are served in-app."""
    await _sign_in(client, db, "noportal@example.com")
    assert (await client.post("/api/v1/billing/portal-session")).status_code == 404


async def test_invoices_are_empty_rather_than_an_error_without_a_subscription(
    client: AsyncClient, db: AsyncSession
) -> None:
    await _sign_in(client, db, "noinvoices@example.com")
    response = await client.get("/api/v1/billing/invoices")
    assert response.status_code == 200
    assert response.json()["data"] == []


# ── Upgrading between paid plans ────────────────────────────────────────────
#
# Razorpay has no call that changes the plan on a subscription. An upgrade is a
# second subscription, and until its mandate is authorized both exist and both
# bill. These tests are the contract for that window, and they exist because it
# went wrong on a real account: it ended up subscribed to Pro *and* Team with
# the Pro id recorded nowhere, so nothing could cancel it and nothing could see
# it.


async def _upgrade_to_team(
    client: AsyncClient, db: AsyncSession, email: str
) -> tuple[User, Subscription, str, str]:
    """Pro, paid for, then a Team checkout started but not yet authorized."""
    user = await _sign_in(client, db, email)
    await client.post(CHECKOUT, json={"plan": "pro", "interval": "monthly"})
    subscription = await billing_service.get_subscription(db, user)
    assert subscription is not None
    pro_id = subscription.provider_subscription_id
    assert pro_id is not None

    assert (
        await _post_event(
            client,
            _subscription_event(
                subscription_id=pro_id,
                customer_id=subscription.provider_customer_id,
                user_id=user.id,
            ),
            event_id=f"evt_pro_{email}",
        )
        == 200
    )
    await db.refresh(user)
    assert user.plan is Plan.PRO, "the Pro subscription has to be live before it can be replaced"

    await client.post(CHECKOUT, json={"plan": "team", "interval": "monthly", "seats": 5})
    await db.refresh(subscription)
    team_id = subscription.pending_subscription_id
    assert team_id is not None
    return user, subscription, pro_id, team_id


async def test_an_upgrade_does_not_overwrite_the_subscription_still_being_paid_for(
    client: AsyncClient, db: AsyncSession, razorpay: FakeRazorpay, unsigned: None
) -> None:
    """Until the new mandate is authorized the old subscription is the one
    charging, so it stays in `provider_subscription_id` and goes on being
    tracked. Overwriting it lost the reference to a subscription that was still
    billing, and nothing could then cancel it."""
    user, subscription, pro_id, team_id = await _upgrade_to_team(
        client, db, "upgrade-window@example.com"
    )

    assert subscription.provider_subscription_id == pro_id
    assert subscription.pending_subscription_id == team_id
    assert pro_id != team_id
    assert subscription.plan is Plan.PRO
    assert user.plan is Plan.PRO, "nothing is granted before the money moves"
    assert razorpay.cancellations == [], "an unauthorized upgrade cancels nothing"


async def test_activating_an_upgrade_cancels_the_subscription_it_replaced(
    client: AsyncClient, db: AsyncSession, razorpay: FakeRazorpay, unsigned: None
) -> None:
    """The one that stops the double charge.

    Immediately, not at cycle end: Razorpay does not prorate (D-51), so leaving
    the old subscription to run out its cycle bills twice to deliver one plan.
    """
    user, subscription, pro_id, team_id = await _upgrade_to_team(
        client, db, "upgrade-activates@example.com"
    )

    assert (
        await _post_event(
            client,
            _subscription_event(
                subscription_id=team_id,
                customer_id=subscription.provider_customer_id,
                plan_id=TEAM_MONTHLY,
                user_id=user.id,
                quantity=5,
            ),
            event_id="evt_team_activated",
        )
        == 200
    )

    await db.refresh(user)
    await db.refresh(subscription)

    assert user.plan is Plan.TEAM
    assert subscription.provider_subscription_id == team_id, "the new one is now the live one"
    assert subscription.pending_subscription_id is None
    assert subscription.seats == 5
    assert razorpay.cancellations == [{"subscription_id": pro_id, "at_cycle_end": False}]


async def test_the_replaced_subscription_cannot_downgrade_the_account_on_its_way_out(
    client: AsyncClient, db: AsyncSession, razorpay: FakeRazorpay, unsigned: None
) -> None:
    """Cancelling the old subscription makes Razorpay send an event for it, and
    that event resolves onto this same row through the `notes.user_id`
    fallback. Applied, it puts the account back on the plan it just left — the
    upgrade undoing itself a second after it succeeded."""
    user, subscription, pro_id, team_id = await _upgrade_to_team(
        client, db, "upgrade-late-event@example.com"
    )
    await _post_event(
        client,
        _subscription_event(
            subscription_id=team_id,
            customer_id=subscription.provider_customer_id,
            plan_id=TEAM_MONTHLY,
            user_id=user.id,
            quantity=5,
        ),
        event_id="evt_team_live",
    )

    # The cancellation we asked for, coming back at us.
    assert (
        await _post_event(
            client,
            _subscription_event(
                event="subscription.cancelled",
                subscription_id=pro_id,
                customer_id=subscription.provider_customer_id,
                plan_id=PRO_MONTHLY,
                status="cancelled",
                user_id=user.id,
            ),
            event_id="evt_pro_cancelled",
        )
        == 200
    )

    await db.refresh(user)
    await db.refresh(subscription)
    assert user.plan is Plan.TEAM, "the account keeps the plan it paid for"
    assert subscription.provider_subscription_id == team_id
    assert subscription.status is SubscriptionStatus.ACTIVE


async def test_a_live_subscription_does_not_report_itself_as_cancelling(
    client: AsyncClient, db: AsyncSession, razorpay: FakeRazorpay, unsigned: None
) -> None:
    """`end_at` is set on every subscription this product creates: Razorpay
    requires a finite `total_count` and we send ten years of cycles. Reading it
    as a scheduled cancellation told every paying customer their plan was about
    to end.

    Razorpay reports a real `cancel_at_cycle_end` nowhere on the entity — the
    cancel call returns 200 and changes no field — so the flag is owned locally
    and this handler must not derive it.
    """
    user = await _sign_in(client, db, "not-cancelling@example.com")
    await client.post(CHECKOUT, json={"plan": "pro", "interval": "monthly"})
    subscription = await billing_service.get_subscription(db, user)
    assert subscription is not None

    far_future = int((utcnow() + timedelta(days=3650)).timestamp())
    assert (
        await _post_event(
            client,
            _subscription_event(
                subscription_id=subscription.provider_subscription_id or "sub_test_1",
                customer_id=subscription.provider_customer_id,
                user_id=user.id,
                end_at=far_future,
            ),
            event_id="evt_end_at",
        )
        == 200
    )

    await db.refresh(subscription)
    assert subscription.status is SubscriptionStatus.ACTIVE
    assert subscription.cancel_at_period_end is False

    body = (await client.get("/api/v1/billing/subscription")).json()["data"]
    assert body["cancel_at_period_end"] is False


# ── Reconciliation ──────────────────────────────────────────────────────────


async def test_a_payment_whose_webhook_never_arrived_can_still_be_applied(
    client: AsyncClient, db: AsyncSession, razorpay: FakeRazorpay, unsigned: None
) -> None:
    """The repair path.

    Razorpay creates a subscription before it is paid for, so `activated` is
    the only thing that says the money moved — and it ships no CLI that
    forwards webhooks, so locally that delivery is absent rather than merely
    unreliable. Without this, one lost event is a customer who has paid and an
    account that never moves.
    """
    user = await _sign_in(client, db, "reconcile@example.com")
    await client.post(CHECKOUT, json={"plan": "pro", "interval": "monthly"})
    subscription = await billing_service.get_subscription(db, user)
    assert subscription is not None
    sub_id = subscription.provider_subscription_id
    assert sub_id is not None

    await db.refresh(user)
    assert user.plan is Plan.FREE, "no delivery, no plan — that is the bug being repaired"

    # What Razorpay would say if asked: the mandate was authorized and it is
    # charging. Nobody told us.
    razorpay.entities[sub_id] = _subscription_event(
        subscription_id=sub_id,
        customer_id=subscription.provider_customer_id,
        user_id=user.id,
    )["payload"]["subscription"]["entity"]

    response = await client.post("/api/v1/billing/reconcile")
    assert response.status_code == 200
    assert response.json()["data"]["plan"] == "pro"

    await db.refresh(user)
    assert user.plan is Plan.PRO


async def test_reconciling_twice_over_unchanged_state_changes_nothing(
    client: AsyncClient, db: AsyncSession, razorpay: FakeRazorpay, unsigned: None
) -> None:
    """It is a button a worried customer can press repeatedly, so it has to be
    idempotent — and it has to stay callable, which a bare subscription id as
    the event key would have broken after the first press."""
    user = await _sign_in(client, db, "reconcile-twice@example.com")
    await client.post(CHECKOUT, json={"plan": "pro", "interval": "monthly"})
    subscription = await billing_service.get_subscription(db, user)
    assert subscription is not None
    sub_id = subscription.provider_subscription_id
    assert sub_id is not None
    razorpay.entities[sub_id] = _subscription_event(
        subscription_id=sub_id,
        customer_id=subscription.provider_customer_id,
        user_id=user.id,
    )["payload"]["subscription"]["entity"]

    for _ in range(3):
        assert (await client.post("/api/v1/billing/reconcile")).status_code == 200

    await db.refresh(user)
    assert user.plan is Plan.PRO
    assert razorpay.cancellations == [], "nothing was replaced, so nothing is cancelled"
