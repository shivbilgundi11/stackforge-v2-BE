"""Subscriptions, checkout, and the webhook that makes them real (M20).

The old build's billing endpoint returned "checkout unavailable". This is the
module that makes money possible, and almost all of its difficulty is in one
place: **a webhook is delivered at least once, and sometimes more than once.**

## Idempotency

Every delivery is written to `stripe_events` keyed on the Stripe event id
*before* any handler runs, and the row's `processed_at` — not its existence —
is what marks it done:

  * no row            → insert, process, stamp `processed_at`
  * row, processed    → skip; this is a redelivery of work already applied
  * row, not processed → process it now

That third case is the one a naive "insert or skip" gets wrong. A handler that
raises leaves a row behind; if existence alone meant "done", Stripe's next
retry would be discarded and the customer who paid would never be upgraded.

## What is authoritative

Stripe is, for subscription state. This module never infers a plan from the
success redirect — the browser can be closed before it lands, and the redirect
is not authenticated. `users.plan` is a denormalised cache of
`subscriptions.plan`, written by these handlers and read by `FeatureService`.

## Nothing is deleted on a downgrade

A user who drops to Free keeps their 25 projects; they can read and export
them, and creating the 26th is what fails. Deleting customer data on a plan
change is the fastest way to lose them permanently, so no code path here
removes a row that belongs to a user.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from typing import Any, Final

from sqlalchemy import select
from sqlalchemy.dialects.postgresql import insert as pg_insert
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.config import settings
from app.core.database import new_id, utcnow
from app.core.errors import Conflict, NotFound, PlanRequired, ValidationFailed
from app.core.logging import get_logger
from app.data import plans as plan_data
from app.integrations import stripe as stripe_integration
from app.models.billing import StripeEvent, Subscription, SubscriptionStatus
from app.models.user import Plan, PlanSource, User

logger = get_logger("billing")

#: Which webhook types this module acts on. Anything else is recorded and
#: marked processed — Stripe sends far more than these, and an unrecognised
#: type is not an error, it is noise the endpoint has to absorb without
#: retrying forever.
HANDLED_EVENTS: Final[frozenset[str]] = frozenset(
    {
        "checkout.session.completed",
        "customer.subscription.created",
        "customer.subscription.updated",
        "customer.subscription.deleted",
        "invoice.paid",
        "invoice.payment_failed",
        "customer.subscription.trial_will_end",
    }
)

#: Stripe's status vocabulary is wider than ours. The three extras all mean one
#: of our five, and mapping them here rather than widening the enum keeps the
#: states the product actually reasons about down to five.
_STATUS_MAP: Final[dict[str, SubscriptionStatus]] = {
    "trialing": SubscriptionStatus.TRIALING,
    "active": SubscriptionStatus.ACTIVE,
    "past_due": SubscriptionStatus.PAST_DUE,
    "unpaid": SubscriptionStatus.PAST_DUE,
    "canceled": SubscriptionStatus.CANCELED,
    "incomplete": SubscriptionStatus.INCOMPLETE,
    "incomplete_expired": SubscriptionStatus.CANCELED,
    "paused": SubscriptionStatus.CANCELED,
}

Interval = str  # "monthly" | "annual"


# ── Price configuration ─────────────────────────────────────────────────────


def price_id_for(plan: Plan, interval: Interval) -> str:
    match (plan, interval):
        case (Plan.PRO, "monthly"):
            return settings.stripe_price_pro_monthly
        case (Plan.PRO, "annual"):
            return settings.stripe_price_pro_annual
        case (Plan.TEAM, "monthly"):
            return settings.stripe_price_team_monthly
        case (Plan.TEAM, "annual"):
            return settings.stripe_price_team_annual
        case _:
            return ""


def plan_for_price(price_id: str | None) -> Plan | None:
    """Reverse the map, so a subscription's price identifies its plan.

    Reading the plan from the price rather than from event ordering is what
    makes an upgrade and a downgrade the same code path: the subscription says
    what it is now, and whether that is up or down from before does not matter.
    """
    if not price_id:
        return None
    for plan in (Plan.PRO, Plan.TEAM):
        for interval in ("monthly", "annual"):
            if price_id_for(plan, interval) == price_id:
                return plan
    return None


# ── Reading ─────────────────────────────────────────────────────────────────


async def get_subscription(db: AsyncSession, user: User) -> Subscription | None:
    """The user's live subscription, or None.

    Canceled rows stay for history; they are never the answer to "what is this
    user on".
    """
    stmt = (
        select(Subscription)
        .where(
            Subscription.user_id == user.id,
            Subscription.status != SubscriptionStatus.CANCELED,
        )
        .limit(1)
    )
    return (await db.execute(stmt)).scalar_one_or_none()


async def _by_customer(db: AsyncSession, customer_id: str) -> Subscription | None:
    stmt = (
        select(Subscription)
        .where(Subscription.stripe_customer_id == customer_id)
        .order_by(Subscription.created_at.desc())
        .limit(1)
    )
    return (await db.execute(stmt)).scalar_one_or_none()


async def _by_stripe_subscription(db: AsyncSession, subscription_id: str) -> Subscription | None:
    stmt = select(Subscription).where(Subscription.stripe_subscription_id == subscription_id)
    return (await db.execute(stmt)).scalar_one_or_none()


@dataclass(frozen=True)
class BillingSummary:
    """What `settings/billing` renders. Assembled here so the route is three
    lines and the shape has one owner."""

    plan: Plan
    status: SubscriptionStatus | None
    #: True while a payment provider is configured. The UI hides checkout
    #: rather than offering a button that 402s.
    checkout_available: bool
    seats: int
    cancel_at_period_end: bool
    current_period_end: datetime | None
    trial_ends_at: datetime | None
    past_due_since: datetime | None
    #: Days left before a past-due subscription is downgraded, or None.
    grace_days_left: int | None


async def summary(db: AsyncSession, user: User) -> BillingSummary:
    subscription = await get_subscription(db, user)

    grace_left: int | None = None
    if subscription is not None and subscription.past_due_since is not None:
        deadline = subscription.past_due_since + timedelta(days=settings.dunning_grace_days)
        grace_left = max(0, (deadline - utcnow()).days)

    return BillingSummary(
        plan=user.plan,
        status=subscription.status if subscription else None,
        checkout_available=settings.billing_enabled,
        seats=subscription.seats if subscription else 0,
        cancel_at_period_end=subscription.cancel_at_period_end if subscription else False,
        current_period_end=subscription.current_period_end if subscription else None,
        trial_ends_at=subscription.trial_ends_at if subscription else None,
        past_due_since=subscription.past_due_since if subscription else None,
        grace_days_left=grace_left,
    )


async def list_invoices(db: AsyncSession, user: User, *, limit: int = 12) -> list[dict[str, Any]]:
    """Live from Stripe. A user with no customer record has no invoices, which
    is an empty list rather than an error — most accounts are in that state."""
    client = stripe_integration.get_client()
    subscription = await get_subscription(db, user)
    if client is None or subscription is None:
        return []
    return await client.list_invoices(customer_id=subscription.stripe_customer_id, limit=limit)


# ── Checkout ────────────────────────────────────────────────────────────────


async def start_checkout(
    db: AsyncSession,
    user: User,
    *,
    plan: Plan,
    interval: Interval,
    seats: int = 1,
) -> str:
    """Create a Checkout Session and return its URL.

    The subscription row is written *before* the redirect, with status
    `incomplete`. That row is what a webhook arriving during the redirect
    attaches to, and it means an abandoned checkout leaves a visible,
    explicable state rather than nothing at all.
    """
    client = stripe_integration.get_client()
    if client is None:
        raise PlanRequired(
            "Checkout is not available in this environment.",
            details={"reason": "billing_not_configured"},
        )

    spec = plan_data.spec_for(plan)
    if not spec.checkout:
        raise ValidationFailed.on_field("plan", f"The {spec.label} plan is not self-serve.")

    if interval not in ("monthly", "annual"):
        raise ValidationFailed.on_field("interval", "Choose monthly or annual billing.")

    price_id = price_id_for(plan, interval)
    if not price_id:
        raise PlanRequired(
            f"The {spec.label} plan has no price configured in this environment.",
            details={"reason": "price_not_configured"},
        )

    if not spec.per_seat and seats != 1:
        raise ValidationFailed.on_field("seats", "This plan is not billed per seat.")
    if seats < 1:
        raise ValidationFailed.on_field("seats", "At least one seat is required.")

    subscription = await get_subscription(db, user)
    if subscription is not None and subscription.is_paid and subscription.plan == plan:
        raise Conflict("You are already on this plan.")

    customer_id = await _ensure_customer(client, subscription, user)
    subscription = await _ensure_row(db, user, subscription, plan=plan, customer_id=customer_id)

    # A trial is offered once. Stripe would happily grant a second one on a
    # resubscription, and "cancel and re-subscribe" is not a supported way to
    # get another free week.
    trial_days = spec.trial_days if not await _has_trialed(db, user) else 0

    session_id, url = await client.create_checkout_session(
        customer_id=customer_id,
        price_id=price_id,
        client_reference_id=user.id,
        success_url=f"{settings.web_base_url}/dashboard?session_id={{CHECKOUT_SESSION_ID}}",
        cancel_url=f"{settings.web_base_url}/pricing",
        quantity=seats,
        trial_days=trial_days,
    )

    subscription.seats = seats
    subscription.stripe_price_id = price_id
    await db.flush()

    logger.info(
        "billing.checkout_started",
        user_id=user.id,
        plan=plan.value,
        interval=interval,
        seats=seats,
        trial_days=trial_days,
        session_id=session_id,
    )
    return url


async def open_portal(db: AsyncSession, user: User) -> str:
    """The Stripe-hosted portal: payment method, invoices, cancellation.

    Not rebuilt in-app. Card collection is the one thing worth handing to a
    provider wholesale — PCI scope, 3DS, and every card-update edge case come
    with it, and none of them are this product.
    """
    client = stripe_integration.get_client()
    subscription = await get_subscription(db, user)
    if client is None or subscription is None:
        raise NotFound("There is no billing account to manage yet.")

    return await client.create_portal_session(
        customer_id=subscription.stripe_customer_id,
        return_url=f"{settings.web_base_url}/settings/billing",
    )


async def set_cancellation(db: AsyncSession, user: User, *, cancel: bool) -> Subscription:
    """Cancel at period end, or undo that.

    Never an immediate cancellation. The user paid for the period and keeps it;
    a mid-period cutoff would be taking back something already bought.
    """
    client = stripe_integration.get_client()
    subscription = await get_subscription(db, user)
    if subscription is None or subscription.stripe_subscription_id is None:
        raise NotFound("There is no active subscription to cancel.")

    if client is not None:
        await client.cancel_at_period_end(
            subscription_id=subscription.stripe_subscription_id, cancel=cancel
        )

    # Written locally too. The webhook will confirm it, but the user just
    # clicked the button and should not have to reload until Stripe calls back.
    subscription.cancel_at_period_end = cancel
    await db.flush()
    logger.info("billing.cancellation_set", user_id=user.id, cancel=cancel)
    return subscription


async def _ensure_customer(
    client: stripe_integration.StripeClient,
    subscription: Subscription | None,
    user: User,
) -> str:
    """One Stripe customer per user, forever.

    Reused across subscriptions on purpose: a user who cancels and comes back
    six months later must land on the same customer record, or their invoice
    history splits in two and support cannot answer "what have I paid you".
    """
    if subscription is not None and subscription.stripe_customer_id:
        return subscription.stripe_customer_id
    return await client.create_customer(email=user.email, name=user.name, user_id=user.id)


async def _ensure_row(
    db: AsyncSession,
    user: User,
    subscription: Subscription | None,
    *,
    plan: Plan,
    customer_id: str,
) -> Subscription:
    """Reuse the live row rather than inserting a second one.

    The partial unique index allows exactly one non-canceled subscription per
    user, so an abandoned Pro checkout followed by a Team checkout has to
    update rather than insert — otherwise the second checkout fails on a
    constraint the user has no way to understand.
    """
    if subscription is not None:
        subscription.plan = plan
        subscription.stripe_customer_id = customer_id
        await db.flush()
        return subscription

    row = Subscription(
        id=new_id("sub"),
        user_id=user.id,
        stripe_customer_id=customer_id,
        plan=plan,
        status=SubscriptionStatus.INCOMPLETE,
    )
    db.add(row)
    await db.flush()
    return row


async def _has_trialed(db: AsyncSession, user: User) -> bool:
    stmt = select(Subscription.id).where(
        Subscription.user_id == user.id, Subscription.trial_ends_at.is_not(None)
    )
    return (await db.execute(stmt.limit(1))).scalar_one_or_none() is not None


# ── Webhooks ────────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class EventOutcome:
    event_id: str
    type: str
    #: False when this delivery was a duplicate of one already applied.
    processed: bool
    detail: str


async def record_event(db: AsyncSession, event: dict[str, Any]) -> StripeEvent | None:
    """Write the delivery down. `None` means it has already been applied.

    `ON CONFLICT DO NOTHING` rather than a SELECT-then-INSERT: two deliveries
    of the same event can arrive concurrently, and the check-then-act version
    lets both through the gap between the two statements.
    """
    event_id = str(event.get("id") or "")
    if not event_id:
        raise ValidationFailed("The webhook payload carried no event id.")

    stmt = (
        pg_insert(StripeEvent)
        .values(
            id=event_id,
            type=str(event.get("type") or "unknown"),
            payload=event,
        )
        .on_conflict_do_nothing(index_elements=["id"])
        .returning(StripeEvent.id)
    )
    inserted = (await db.execute(stmt)).scalar_one_or_none()

    if inserted is None:
        existing = await db.get(StripeEvent, event_id)
        # Already applied — a genuine redelivery, and the whole reason this
        # table exists. An *unprocessed* row is a previous attempt that failed,
        # and Stripe's retry is exactly the second chance it needs.
        if existing is not None and existing.processed_at is not None:
            return None
        return existing

    return await db.get(StripeEvent, event_id)


async def process_event(db: AsyncSession, record: StripeEvent) -> str:
    """Apply one event. Raises on failure; the caller records that."""
    payload = record.payload if isinstance(record.payload, dict) else {}
    obj = _event_object(payload)

    match record.type:
        case "checkout.session.completed":
            detail = await _on_checkout_completed(db, obj)
        case "customer.subscription.created" | "customer.subscription.updated":
            detail = await _on_subscription_changed(db, obj)
        case "customer.subscription.deleted":
            detail = await _on_subscription_deleted(db, obj)
        case "invoice.paid":
            detail = await _on_invoice_paid(db, obj)
        case "invoice.payment_failed":
            detail = await _on_payment_failed(db, obj)
        case "customer.subscription.trial_will_end":
            detail = await _on_trial_will_end(db, obj)
        case _:
            # Recorded and marked done. Stripe sends dozens of types this
            # product has no opinion on, and retrying them forever would fill
            # the retry job with work that can never succeed.
            detail = "ignored"

    record.processed_at = utcnow()
    record.error = None
    await db.flush()
    return detail


def _event_object(payload: dict[str, Any]) -> dict[str, Any]:
    data = payload.get("data")
    if isinstance(data, dict):
        obj = data.get("object")
        if isinstance(obj, dict):
            return obj
    return {}


# ── Handlers ────────────────────────────────────────────────────────────────


async def _on_checkout_completed(db: AsyncSession, obj: dict[str, Any]) -> str:
    """Attach the new Stripe subscription to the row the checkout created.

    `client_reference_id` is read rather than the customer id: it was set from
    the authenticated user at checkout time, so attribution does not depend on
    a customer lookup that could match a row mid-update.
    """
    user_id = obj.get("client_reference_id")
    subscription_id = obj.get("subscription")
    customer_id = obj.get("customer")

    if not isinstance(user_id, str) or not user_id:
        return "no client_reference_id"

    user = await db.get(User, user_id)
    if user is None:
        logger.warning("billing.checkout_unknown_user", user_id=user_id)
        return "unknown user"

    subscription = await get_subscription(db, user)
    if subscription is None:
        if not isinstance(customer_id, str):
            return "no customer"
        subscription = Subscription(
            id=new_id("sub"),
            user_id=user.id,
            stripe_customer_id=customer_id,
            plan=Plan.PRO,
            status=SubscriptionStatus.INCOMPLETE,
        )
        db.add(subscription)

    if isinstance(subscription_id, str):
        subscription.stripe_subscription_id = subscription_id
    if isinstance(customer_id, str):
        subscription.stripe_customer_id = customer_id

    await db.flush()
    logger.info("billing.checkout_completed", user_id=user.id, subscription_id=subscription_id)
    return "checkout attached"


async def _on_subscription_changed(db: AsyncSession, obj: dict[str, Any]) -> str:
    """The workhorse. Upgrade, downgrade, trial start, and renewal all land here.

    Everything is read from the object as it is *now* rather than diffed
    against what we held. Stripe does not guarantee delivery order, and a
    handler that computed "this is an upgrade" from the previous local state
    would apply an out-of-order pair backwards.
    """
    subscription = await _resolve_subscription(db, obj)
    if subscription is None:
        return "no matching subscription"

    status = _STATUS_MAP.get(str(obj.get("status") or ""), SubscriptionStatus.INCOMPLETE)
    price_id = _price_id(obj)
    plan = plan_for_price(price_id)

    subscription.status = status
    if price_id:
        subscription.stripe_price_id = price_id
    if plan is not None:
        subscription.plan = plan
    if isinstance(sub_id := obj.get("id"), str):
        subscription.stripe_subscription_id = sub_id

    subscription.cancel_at_period_end = bool(obj.get("cancel_at_period_end"))
    subscription.current_period_start = _timestamp(_period_field(obj, "current_period_start"))
    subscription.current_period_end = _timestamp(_period_field(obj, "current_period_end"))
    subscription.trial_ends_at = _timestamp(obj.get("trial_end"))
    subscription.seats = _quantity(obj) or subscription.seats

    if status is SubscriptionStatus.PAST_DUE:
        subscription.past_due_since = subscription.past_due_since or utcnow()
    else:
        subscription.past_due_since = None

    if status is SubscriptionStatus.CANCELED:
        subscription.canceled_at = subscription.canceled_at or utcnow()

    await _apply_plan(db, subscription)
    await db.flush()
    return f"subscription {status.value}"


async def _on_subscription_deleted(db: AsyncSession, obj: dict[str, Any]) -> str:
    subscription = await _resolve_subscription(db, obj)
    if subscription is None:
        return "no matching subscription"

    subscription.status = SubscriptionStatus.CANCELED
    subscription.canceled_at = utcnow()
    subscription.past_due_since = None
    await _apply_plan(db, subscription)
    await db.flush()
    return "subscription canceled"


async def _on_invoice_paid(db: AsyncSession, obj: dict[str, Any]) -> str:
    """A successful payment clears dunning.

    Written even though `customer.subscription.updated` usually follows: the
    two events are independent deliveries, and a user whose card recovered
    should not keep a "payment failed" banner because one of them was delayed.
    """
    customer_id = obj.get("customer")
    if not isinstance(customer_id, str):
        return "no customer"

    subscription = await _by_customer(db, customer_id)
    if subscription is None:
        return "no matching subscription"

    if subscription.status is SubscriptionStatus.PAST_DUE:
        subscription.status = SubscriptionStatus.ACTIVE
    subscription.past_due_since = None
    await _apply_plan(db, subscription)
    await db.flush()
    return "payment recorded"


async def _on_payment_failed(db: AsyncSession, obj: dict[str, Any]) -> str:
    """Start the dunning clock. Do not downgrade.

    The features stay on through the grace window: Stripe is still retrying
    the card, and cutting access to a customer whose payment is one retry from
    succeeding costs more than the week of usage it saves.
    """
    customer_id = obj.get("customer")
    if not isinstance(customer_id, str):
        return "no customer"

    subscription = await _by_customer(db, customer_id)
    if subscription is None:
        return "no matching subscription"

    subscription.status = SubscriptionStatus.PAST_DUE
    subscription.past_due_since = subscription.past_due_since or utcnow()
    await db.flush()

    if subscription.user_id:
        user = await db.get(User, subscription.user_id)
        if user is not None:
            from app.services import email_templates

            await _send(
                email_templates.payment_failed(
                    to=user.email,
                    name=user.name,
                    grace_days=settings.dunning_grace_days,
                )
            )

    logger.info("billing.payment_failed", subscription_id=subscription.id)
    return "dunning started"


async def _on_trial_will_end(db: AsyncSession, obj: dict[str, Any]) -> str:
    """Three days out, per Stripe's own schedule.

    The email exists because a trial that ends silently is a downgrade the user
    experiences as a bug.
    """
    subscription = await _resolve_subscription(db, obj)
    if subscription is None or subscription.user_id is None:
        return "no matching subscription"

    user = await db.get(User, subscription.user_id)
    if user is None:
        return "unknown user"

    from app.services import email_templates

    ends_at = _timestamp(obj.get("trial_end")) or subscription.trial_ends_at
    await _send(
        email_templates.trial_ending(
            to=user.email,
            name=user.name,
            plan=plan_data.spec_for(subscription.plan).label,
            ends_at=ends_at,
        )
    )
    return "trial reminder sent"


async def _send(message: object) -> None:
    """Email never fails a webhook.

    A mail provider outage must not leave an event unprocessed — the retry
    would re-apply the plan change to send one message, and the plan change is
    the part that matters.
    """
    from app.integrations.email import Email, send

    if isinstance(message, Email):
        await send(message)


# ── Plan application ────────────────────────────────────────────────────────


async def _apply_plan(db: AsyncSession, subscription: Subscription) -> None:
    """Push the subscription's plan onto the user row.

    This is the denormalisation `FeatureService` reads on every request. It is
    written here and nowhere else, so "what changes a user's plan" has exactly
    one answer.

    A subscription that is not paid resolves to Free — and resolves *only* the
    plan. No project, stack, export, or run is touched, which is what makes a
    downgrade reversible by paying again.
    """
    if subscription.user_id is None:
        return

    user = await db.get(User, subscription.user_id)
    if user is None:
        return

    target = subscription.plan if subscription.is_paid else Plan.FREE
    if user.plan is not target:
        logger.info(
            "billing.plan_changed",
            user_id=user.id,
            was=user.plan.value,
            now=target.value,
            status=subscription.status.value,
        )
    user.plan = target
    user.plan_source = PlanSource.PERSONAL
    await db.flush()


async def _resolve_subscription(db: AsyncSession, obj: dict[str, Any]) -> Subscription | None:
    """Find our row from a Stripe subscription object.

    Three routes, in order of reliability: the subscription id we stored, the
    metadata we set at checkout, then the customer. The last is a fallback
    because a customer can in principle hold more than one subscription, and
    matching on it alone would attach the wrong one.
    """
    if isinstance(sub_id := obj.get("id"), str):
        found = await _by_stripe_subscription(db, sub_id)
        if found is not None:
            return found

    metadata = obj.get("metadata")
    if isinstance(metadata, dict):
        user_id = metadata.get("client_reference_id")
        if isinstance(user_id, str):
            user = await db.get(User, user_id)
            if user is not None:
                found = await get_subscription(db, user)
                if found is not None:
                    return found

    if isinstance(customer_id := obj.get("customer"), str):
        return await _by_customer(db, customer_id)

    return None


def _price_id(obj: dict[str, Any]) -> str | None:
    items = obj.get("items")
    if isinstance(items, dict):
        data = items.get("data")
        if isinstance(data, list) and data:
            first = data[0]
            if isinstance(first, dict):
                price = first.get("price")
                if isinstance(price, dict) and isinstance(price.get("id"), str):
                    return str(price["id"])
    plan_obj = obj.get("plan")
    if isinstance(plan_obj, dict) and isinstance(plan_obj.get("id"), str):
        return str(plan_obj["id"])
    return None


def _quantity(obj: dict[str, Any]) -> int | None:
    items = obj.get("items")
    if isinstance(items, dict):
        data = items.get("data")
        if isinstance(data, list) and data and isinstance(data[0], dict):
            quantity = data[0].get("quantity")
            if isinstance(quantity, int):
                return quantity
    quantity = obj.get("quantity")
    return quantity if isinstance(quantity, int) else None


def _period_field(obj: dict[str, Any], field: str) -> Any:
    """Read a period boundary from wherever this API version put it.

    Stripe moved `current_period_start`/`_end` from the subscription onto its
    items. Reading both means the same handler works either side of an API
    version bump, which is not a migration anyone should have to schedule.
    """
    value = obj.get(field)
    if value is not None:
        return value

    items = obj.get("items")
    if isinstance(items, dict):
        data = items.get("data")
        if isinstance(data, list) and data and isinstance(data[0], dict):
            return data[0].get(field)
    return None


def _timestamp(value: Any) -> datetime | None:
    if isinstance(value, int | float) and value > 0:
        return datetime.fromtimestamp(int(value), tz=UTC)
    return None


__all__ = [
    "HANDLED_EVENTS",
    "BillingSummary",
    "EventOutcome",
    "get_subscription",
    "list_invoices",
    "open_portal",
    "plan_for_price",
    "price_id_for",
    "process_event",
    "record_event",
    "set_cancellation",
    "start_checkout",
    "summary",
]
