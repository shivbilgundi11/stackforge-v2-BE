"""Subscriptions, checkout, and the webhook that makes them real (M20).

The old build's billing endpoint returned "checkout unavailable". This is the
module that makes money possible, and almost all of its difficulty is in one
place: **a webhook is delivered at least once, and sometimes more than once.**

## Idempotency

Every delivery is written to `billing_events` keyed on the provider's event id
*before* any handler runs, and the row's `processed_at` — not its existence —
is what marks it done:

  * no row            → insert, process, stamp `processed_at`
  * row, processed    → skip; this is a redelivery of work already applied
  * row, not processed → process it now

That third case is the one a naive "insert or skip" gets wrong. A handler that
raises leaves a row behind; if existence alone meant "done", the next retry
would be discarded and the customer who paid would never be upgraded.

Razorpay does not put its event id in the body — it arrives in the
`X-Razorpay-Event-Id` header, and the route passes it in. An unsigned or
header-less delivery is refused before it reaches here.

## What is authoritative

Razorpay is, for subscription state. This module never infers a plan from the
return redirect — the browser can be closed before it lands, and the redirect
is not authenticated. `users.plan` is a denormalised cache of
`subscriptions.plan`, written by these handlers and read by `FeatureService`.

## Why a subscription row exists before anyone pays

Razorpay has no checkout session. A subscription is created server side and
carries a `short_url` where the customer authorizes a mandate, so the row is
written in `created`/`incomplete` state and the browser is sent to that URL.
`subscription.activated` — never the redirect — is what grants a plan.

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

from sqlalchemy import func, select
from sqlalchemy.dialects.postgresql import insert as pg_insert
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.config import settings
from app.core.database import new_id, utcnow
from app.core.errors import Conflict, Forbidden, NotFound, PlanRequired, ValidationFailed
from app.core.logging import get_logger
from app.data import plans as plan_data
from app.integrations import razorpay as razorpay_integration
from app.models.billing import BillingEvent, Subscription, SubscriptionStatus
from app.models.organization import OrgRole
from app.models.user import Plan, PlanSource, User

logger = get_logger("billing")

#: Which webhook types this module acts on. Anything else is recorded and
#: marked processed — Razorpay sends far more than these, and an unrecognised
#: type is not an error, it is noise the endpoint has to absorb without
#: retrying forever.
HANDLED_EVENTS: Final[frozenset[str]] = frozenset(
    {
        "subscription.authenticated",
        "subscription.activated",
        "subscription.charged",
        "subscription.updated",
        "subscription.pending",
        "subscription.halted",
        "subscription.cancelled",
        "subscription.completed",
        "subscription.expired",
    }
)

#: Razorpay's status vocabulary is nine words wide; ours is five. Every extra
#: collapses into one of these for every decision the product makes, and
#: mapping them here rather than widening the enum keeps the states the product
#: reasons about down to five.
#:
#: `authenticated` is the trial: the mandate is confirmed and the first charge
#: is scheduled for `start_at`. `halted` is past due rather than cancelled —
#: Razorpay has stopped retrying, but the dunning grace period is ours to run,
#: and treating it as cancelled would downgrade on the same day the card
#: bounced.
_STATUS_MAP: Final[dict[str, SubscriptionStatus]] = {
    "created": SubscriptionStatus.INCOMPLETE,
    "authenticated": SubscriptionStatus.TRIALING,
    "active": SubscriptionStatus.ACTIVE,
    "pending": SubscriptionStatus.PAST_DUE,
    "halted": SubscriptionStatus.PAST_DUE,
    "cancelled": SubscriptionStatus.CANCELED,
    "completed": SubscriptionStatus.CANCELED,
    "expired": SubscriptionStatus.CANCELED,
    "paused": SubscriptionStatus.CANCELED,
}

Interval = str  # "monthly" | "annual"

#: Razorpay requires a finite `total_count`. Ten years of billing cycles is the
#: ceiling, not a commitment: cancelling ends the subscription earlier, and
#: nobody reaches a hundred and twenty months.
_TOTAL_COUNT: Final[dict[Interval, int]] = {"monthly": 120, "annual": 10}


# ── Plan configuration ──────────────────────────────────────────────────────


def plan_id_for(plan: Plan, interval: Interval) -> str:
    match (plan, interval):
        case (Plan.PRO, "monthly"):
            return settings.razorpay_plan_pro_monthly
        case (Plan.PRO, "annual"):
            return settings.razorpay_plan_pro_annual
        case (Plan.TEAM, "monthly"):
            return settings.razorpay_plan_team_monthly
        case (Plan.TEAM, "annual"):
            return settings.razorpay_plan_team_annual
        case _:
            return ""


def pending_plan_for(chosen: str) -> Plan | None:
    """The signup form's plan choice, as an owed plan.

    Free is not owed — it is what an account already is — so it maps to None
    rather than to `Plan.FREE`. Anything that is not a self-serve plan maps to
    None too: a wall demanding payment for a plan with no checkout is a dead
    end, and Enterprise is a conversation, not a button.
    """
    try:
        plan = Plan(chosen)
    except ValueError:
        return None
    if plan is Plan.FREE:
        return None
    return plan if plan_data.spec_for(plan).checkout else None


def plan_for_provider_plan(provider_plan_id: str | None) -> Plan | None:
    """Reverse the map, so a subscription's plan id identifies its tier.

    Reading the tier from the plan id rather than from event ordering is what
    makes an upgrade and a downgrade the same code path: the subscription says
    what it is now, and whether that is up or down from before does not matter.
    """
    if not provider_plan_id:
        return None
    for plan in (Plan.PRO, Plan.TEAM):
        for interval in ("monthly", "annual"):
            if plan_id_for(plan, interval) == provider_plan_id:
                return plan
    return None


def interval_for_provider_plan(provider_plan_id: str | None) -> Interval | None:
    """Which cadence a plan id is on. Needed because Razorpay reports the seat
    count and the period on the subscription, not the price."""
    if not provider_plan_id:
        return None
    for plan in (Plan.PRO, Plan.TEAM):
        for interval in ("monthly", "annual"):
            if plan_id_for(plan, interval) == provider_plan_id:
                return interval
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
        .where(Subscription.provider_customer_id == customer_id)
        .order_by(Subscription.created_at.desc())
        .limit(1)
    )
    return (await db.execute(stmt)).scalar_one_or_none()


async def _by_provider_subscription(db: AsyncSession, subscription_id: str) -> Subscription | None:
    stmt = select(Subscription).where(Subscription.provider_subscription_id == subscription_id)
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
    #: A plan chosen at signup and not yet paid for, and the interval it was
    #: chosen on. What the payment wall renders.
    pending_plan: Plan | None
    pending_interval: str | None
    #: Whether the wall should stand in front of the account-only surfaces.
    payment_required: bool


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
        pending_plan=user.pending_plan,
        pending_interval=user.pending_interval,
        payment_required=payment_required(user, subscription),
    )


def payment_required(user: User, subscription: Subscription | None) -> bool:
    """Whether this account should be held at the payment wall.

    Two causes, and only two. A plan was chosen and never paid for, or a
    subscription went past due and its grace period ran out — at which point
    the plan has already been downgraded and the account is being *told* why
    rather than being denied anything it still has.

    Deliberately not a permission check. Every quota and feature decision reads
    `user.plan`, which is Free until a webhook says otherwise; this only
    decides which screen a browser lands on.
    """
    if user.pending_plan is not None:
        return True
    if subscription is None or subscription.past_due_since is None:
        return False
    deadline = subscription.past_due_since + timedelta(days=settings.dunning_grace_days)
    return utcnow() >= deadline


async def select_plan(
    db: AsyncSession, user: User, *, plan: Plan | None, interval: Interval
) -> None:
    """Record — or clear — the plan this account intends to buy.

    `None` is the escape from the wall: "continue on Free". It is deliberately
    always available. A signup form that can trap someone on a payment screen
    with no way past it converts worse than one they can decline, and support
    ends up clearing the column by hand.
    """
    if plan is None:
        user.pending_plan = None
        user.pending_interval = None
        await db.flush()
        logger.info("billing.plan_selection_cleared", user_id=user.id)
        return

    spec = plan_data.spec_for(plan)
    if not spec.checkout:
        raise ValidationFailed.on_field("plan", f"The {spec.label} plan is not self-serve.")
    if interval not in ("monthly", "annual"):
        raise ValidationFailed.on_field("interval", "Choose monthly or annual billing.")

    user.pending_plan = plan
    user.pending_interval = interval
    await db.flush()
    logger.info(
        "billing.plan_selected", user_id=user.id, plan=plan.value, interval=interval
    )


async def list_invoices(db: AsyncSession, user: User, *, limit: int = 12) -> list[dict[str, Any]]:
    """Live from Razorpay. A user who has never subscribed has no invoices,
    which is an empty list rather than an error — most accounts are in that
    state.

    Keyed on the subscription rather than the customer: Razorpay filters the
    invoice list by subscription, and a customer who has resubscribed would
    otherwise get an old subscription's invoices mixed into theirs.
    """
    client = razorpay_integration.get_client()
    subscription = await get_subscription(db, user)
    if client is None or subscription is None or subscription.provider_subscription_id is None:
        return []
    return await client.list_invoices(
        subscription_id=subscription.provider_subscription_id, limit=limit
    )


# ── Checkout ────────────────────────────────────────────────────────────────


async def start_checkout(
    db: AsyncSession,
    user: User,
    *,
    plan: Plan,
    interval: Interval,
    seats: int = 1,
) -> str:
    """Create a Razorpay subscription and return its hosted authorization URL.

    There is no checkout session on this provider. The subscription is created
    server side in `created` state and carries a `short_url` — the hosted page
    where the customer authorizes a mandate — so the row here is written before
    anyone has paid. `subscription.activated` is what grants the plan; the
    redirect never does.

    An abandoned authorization therefore leaves a real Razorpay subscription
    sitting in `created`. That is harmless — nothing can be charged against an
    unauthorized mandate — and the next attempt reuses the row.
    """
    client = razorpay_integration.get_client()
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

    provider_plan_id = plan_id_for(plan, interval)
    if not provider_plan_id:
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

    subscription = await _ensure_row(db, user, subscription, plan=plan)

    # A trial is offered once. Razorpay would happily start another delayed
    # subscription, and "cancel and re-subscribe" is not a supported way to get
    # another free week.
    trial_days = spec.trial_days if not await _has_trialed(db, user) else 0
    # The mandate is authorized now either way; `start_at` only moves the first
    # charge. Razorpay has no card-free trial, which is the one property of
    # D-42 that D-50 gives up.
    start_at = int((utcnow() + timedelta(days=trial_days)).timestamp()) if trial_days else None

    provider_subscription_id, url = await client.create_subscription(
        plan_id=provider_plan_id,
        notify_email=user.email,
        quantity=seats,
        total_count=_TOTAL_COUNT.get(interval, 120),
        start_at=start_at,
        # Carries the account through the redirect, so a webhook attributes the
        # subscription without racing the browser back to the return page.
        notes={"user_id": user.id},
    )

    subscription.seats = seats
    subscription.provider_plan_id = provider_plan_id
    subscription.provider_subscription_id = provider_subscription_id
    if trial_days:
        subscription.trial_ends_at = utcnow() + timedelta(days=trial_days)
    await db.flush()

    logger.info(
        "billing.checkout_started",
        user_id=user.id,
        plan=plan.value,
        interval=interval,
        seats=seats,
        trial_days=trial_days,
        subscription_id=provider_subscription_id,
    )
    return url


# `open_portal` went with Stripe (D-50). Razorpay has no hosted billing portal,
# so the two things the portal was for are served in-app instead: invoices are
# read live by `list_invoices`, and cancellation is a button on the billing
# page. Changing a saved card means authorizing a new mandate, which is what
# re-subscribing already does.


async def set_cancellation(db: AsyncSession, user: User, *, cancel: bool) -> Subscription:
    """Cancel at period end, or undo that.

    Never an immediate cancellation. The user paid for the period and keeps it;
    a mid-period cutoff would be taking back something already bought.
    """
    client = razorpay_integration.get_client()
    subscription = await get_subscription(db, user)
    if subscription is None or subscription.provider_subscription_id is None:
        raise NotFound("There is no active subscription to cancel.")

    if client is not None:
        if cancel:
            await client.cancel_subscription(
                subscription_id=subscription.provider_subscription_id, at_cycle_end=True
            )
        else:
            # Razorpay has no "un-cancel" flag. A scheduled cancellation is a
            # pending change on the subscription, and this is the call that
            # drops it — which is why undo is a different method rather than
            # the same one with the flag inverted.
            await client.cancel_scheduled_changes(
                subscription_id=subscription.provider_subscription_id
            )

    # Written locally too. The webhook will confirm it, but the user just
    # clicked the button and should not have to reload until Razorpay calls
    # back.
    subscription.cancel_at_period_end = cancel
    await db.flush()
    logger.info("billing.cancellation_set", user_id=user.id, cancel=cancel)
    return subscription


async def _ensure_row(
    db: AsyncSession,
    user: User,
    subscription: Subscription | None,
    *,
    plan: Plan,
) -> Subscription:
    """Reuse the live row rather than inserting a second one.

    The partial unique index allows exactly one non-canceled subscription per
    user, so an abandoned Pro checkout followed by a Team checkout has to
    update rather than insert — otherwise the second checkout fails on a
    constraint the user has no way to understand.

    **A paid row keeps its plan.** `plan` here is what the caller *intends to
    buy*, and on a live subscription that is not a fact yet — Razorpay has not
    charged anything and may never. Writing it down anyway meant an upgrade
    that got as far as the authorization page and was then abandoned left the row
    claiming the better plan, and `sync_user_plan` reads exactly that field:
    the next webhook for that account granted Team to somebody paying for Pro.
    It also made the "already on this plan" guard fire against a plan they were
    not on, so retrying the upgrade was refused with no way forward.

    An unpaid row is different. It exists only to be attached to an in-flight
    checkout, nothing has been granted from it, and repointing it is the whole
    reason this function reuses rows at all.
    """
    if subscription is not None:
        if not subscription.is_paid:
            subscription.plan = plan
        await db.flush()
        return subscription

    row = Subscription(
        id=new_id("sub"),
        user_id=user.id,
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


async def record_event(
    db: AsyncSession, event: dict[str, Any], *, event_id: str
) -> BillingEvent | None:
    """Write the delivery down. `None` means it has already been applied.

    `event_id` is passed in rather than read from the body: Razorpay puts it in
    the `X-Razorpay-Event-Id` header, not the payload. That is the whole
    idempotency key, so a delivery without one is refused rather than given a
    generated id that would make every redelivery look new.

    `ON CONFLICT DO NOTHING` rather than a SELECT-then-INSERT: two deliveries
    of the same event can arrive concurrently, and the check-then-act version
    lets both through the gap between the two statements.
    """
    if not event_id:
        raise ValidationFailed("The webhook carried no event id.")

    stmt = (
        pg_insert(BillingEvent)
        .values(
            id=event_id,
            type=str(event.get("event") or "unknown"),
            payload=event,
        )
        .on_conflict_do_nothing(index_elements=["id"])
        .returning(BillingEvent.id)
    )
    inserted = (await db.execute(stmt)).scalar_one_or_none()

    if inserted is None:
        existing = await db.get(BillingEvent, event_id)
        # Already applied — a genuine redelivery, and the whole reason this
        # table exists. An *unprocessed* row is a previous attempt that failed,
        # and the provider's retry is exactly the second chance it needs.
        if existing is not None and existing.processed_at is not None:
            return None
        return existing

    return await db.get(BillingEvent, event_id)


async def process_event(db: AsyncSession, record: BillingEvent) -> str:
    """Apply one event. Raises on failure; the caller records that."""
    payload = record.payload if isinstance(record.payload, dict) else {}
    obj = _event_object(payload)

    match record.type:
        case (
            "subscription.authenticated"
            | "subscription.activated"
            | "subscription.charged"
            | "subscription.updated"
            | "subscription.pending"
            | "subscription.halted"
            | "subscription.cancelled"
            | "subscription.completed"
            | "subscription.expired"
        ):
            # One handler for all of them. Every Razorpay subscription event
            # carries the whole subscription entity with its current status, so
            # the event type adds nothing the object does not already say — and
            # branching on the type would mean nine handlers that each have to
            # agree about what the object means.
            detail = await _on_subscription_changed(db, obj)
        case _:
            # Recorded and marked done. Razorpay sends dozens of types this
            # product has no opinion on, and retrying them forever would fill
            # the retry job with work that can never succeed.
            detail = "ignored"

    record.processed_at = utcnow()
    record.error = None
    await db.flush()
    return detail


def _event_object(payload: dict[str, Any]) -> dict[str, Any]:
    """The subscription entity out of a Razorpay webhook body.

    Razorpay nests it two deep and keys the outer level by entity name:
    `payload.subscription.entity`. The `contains` array says which entities a
    delivery carries — a `subscription.charged` also carries `payment` — but
    only the subscription is needed here.
    """
    container = payload.get("payload")
    if isinstance(container, dict):
        subscription = container.get("subscription")
        if isinstance(subscription, dict):
            entity = subscription.get("entity")
            if isinstance(entity, dict):
                return entity
    return {}


# ── Handlers ────────────────────────────────────────────────────────────────


# There is no `_on_checkout_completed` here. Stripe's checkout session
# completing was a separate event that attached a brand-new subscription id to
# the row; Razorpay creates the subscription *before* the customer authorizes
# it, so the id is already on the row from `start_checkout` and every later
# event carries the same one. Attachment is not a step on this provider.


async def _on_subscription_changed(db: AsyncSession, obj: dict[str, Any]) -> str:
    """The workhorse. Every subscription event lands here.

    Everything is read from the object as it is *now* rather than diffed
    against what we held. Razorpay does not guarantee delivery order, and a
    handler that computed "this is an upgrade" from the previous local state
    would apply an out-of-order pair backwards.
    """
    subscription = await _resolve_subscription(db, obj)
    if subscription is None:
        return "no matching subscription"

    raw_status = str(obj.get("status") or "")
    status = _STATUS_MAP.get(raw_status, SubscriptionStatus.INCOMPLETE)
    provider_plan_id = _provider_plan_id(obj)
    plan = plan_for_provider_plan(provider_plan_id)

    was_past_due = subscription.status is SubscriptionStatus.PAST_DUE

    subscription.status = status
    if provider_plan_id:
        subscription.provider_plan_id = provider_plan_id
    if plan is not None:
        subscription.plan = plan
    if isinstance(sub_id := obj.get("id"), str):
        subscription.provider_subscription_id = sub_id
    # The customer does not exist until the mandate is authorized (D-50), so
    # this is the first and only place we learn its id. Never overwritten with
    # nothing: a later event without the field must not erase the link.
    if isinstance(customer_id := obj.get("customer_id"), str) and customer_id:
        subscription.provider_customer_id = customer_id

    # Razorpay reports a scheduled cancellation as an end date rather than a
    # boolean: `end_at` is set while the subscription is still running.
    subscription.cancel_at_period_end = bool(obj.get("end_at")) and status not in (
        SubscriptionStatus.CANCELED,
    )
    subscription.current_period_start = _timestamp(obj.get("current_start"))
    subscription.current_period_end = _timestamp(obj.get("current_end"))
    # `start_at` in the future *is* the trial: the mandate is authorized and
    # the first charge is waiting. Once it has passed, Razorpay keeps sending
    # the same value, so it is only a trial end while it is still ahead of us.
    start_at = _timestamp(obj.get("start_at"))
    if start_at is not None and start_at > utcnow():
        subscription.trial_ends_at = start_at
    subscription.seats = _quantity(obj) or subscription.seats

    if status is SubscriptionStatus.PAST_DUE:
        subscription.past_due_since = subscription.past_due_since or utcnow()
    else:
        subscription.past_due_since = None

    if status is SubscriptionStatus.CANCELED:
        subscription.canceled_at = subscription.canceled_at or utcnow()

    # Past due does **not** touch the plan. `is_paid` excludes it, so applying
    # here would downgrade the account on the first failed charge — and the
    # grace window exists precisely so that does not happen. `close_dunning` is
    # the one place in this program that downgrades an account for money, and
    # it runs when the window closes.
    #
    # Stripe kept this property by having a separate `invoice.payment_failed`
    # handler that never called `_apply_plan`. With one handler for every
    # status, the same property has to be an explicit branch.
    if status is not SubscriptionStatus.PAST_DUE:
        await _apply_plan(db, subscription)
    await db.flush()

    # Dunning mail, driven off the transition rather than a separate event.
    # Razorpay has no `invoice.payment_failed` equivalent that names our
    # customer, and sending on every `pending` delivery would mail the user
    # once per retry.
    if status is SubscriptionStatus.PAST_DUE and not was_past_due:
        await _notify_payment_failed(db, subscription)

    logger.info(
        "billing.subscription_changed",
        subscription_id=subscription.id,
        provider_status=raw_status,
        status=status.value,
    )
    return f"subscription {status.value}"


async def _notify_payment_failed(db: AsyncSession, subscription: Subscription) -> None:
    """Start of dunning. The features stay on through the grace window:
    Razorpay is still retrying the mandate, and cutting access to a customer
    whose payment is one retry from succeeding costs more than the week of
    usage it saves."""
    if subscription.user_id is None:
        return
    user = await db.get(User, subscription.user_id)
    if user is None:
        return

    from app.services import email_templates

    await _send(
        email_templates.payment_failed(
            to=user.email,
            name=user.name,
            grace_days=settings.dunning_grace_days,
        )
    )
    logger.info("billing.payment_failed", subscription_id=subscription.id)


# Four Stripe handlers ended here, and all four are answered by the single
# subscription handler above:
#
#   _on_subscription_deleted   -> `subscription.cancelled` carries the entity
#                                 with status `cancelled`, so the workhorse
#                                 already applies it.
#   _on_invoice_paid           -> `subscription.charged` carries the same
#                                 entity back in `active`, which clears dunning
#                                 through the same path.
#   _on_payment_failed         -> `subscription.pending` does the same in
#                                 reverse; the mail moved to
#                                 `_notify_payment_failed`, fired on the
#                                 transition rather than on every retry.
#   _on_trial_will_end         -> Razorpay has no such event. The reminder is
#                                 sent by `workers.billing.remind_expiring_trials`
#                                 instead, which is a cron job rather than a
#                                 delivery we can be denied.


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


_PLAN_RANK: Final[dict[Plan, int]] = {
    Plan.FREE: 0,
    Plan.PRO: 1,
    Plan.TEAM: 2,
    Plan.ENTERPRISE: 3,
}


async def sync_user_plan(db: AsyncSession, user: User) -> None:
    """Recompute a user's plan from everything that can grant one (M21).

    Two sources: a live personal subscription, and membership of organizations
    whose plan is paid. The higher tier wins, and `plan_source` records which
    source granted it — so a personal downgrade cannot erase a team grant, and
    leaving a team cannot erase a personal subscription.
    """
    from app.models.organization import Organization, OrganizationMember

    personal = await get_subscription(db, user)
    personal_plan = personal.plan if personal is not None and personal.is_paid else Plan.FREE

    org_plan = Plan.FREE
    granted = (
        (
            await db.execute(
                select(Organization.plan)
                .join(
                    OrganizationMember,
                    OrganizationMember.organization_id == Organization.id,
                )
                .where(
                    OrganizationMember.user_id == user.id,
                    Organization.deleted_at.is_(None),
                )
            )
        )
        .scalars()
        .all()
    )
    for candidate in granted:
        if _PLAN_RANK[candidate] > _PLAN_RANK[org_plan]:
            org_plan = candidate

    if _PLAN_RANK[org_plan] > _PLAN_RANK[personal_plan]:
        target, source = org_plan, PlanSource.ORGANIZATION
    else:
        target, source = personal_plan, PlanSource.PERSONAL

    if user.plan is not target:
        logger.info(
            "billing.plan_changed",
            user_id=user.id,
            was=user.plan.value,
            now=target.value,
            source=source.value,
        )
    user.plan = target
    user.plan_source = source

    # The debt is settled the moment the plan arrives, from whichever source.
    # `>=` rather than `==` because being granted Team by an organization
    # satisfies a personal intent to buy Pro — holding that user at a wall
    # asking them to pay for a lesser plan than the one they already have is
    # the kind of loop nobody escapes without support.
    if user.pending_plan is not None and _PLAN_RANK[target] >= _PLAN_RANK[user.pending_plan]:
        logger.info(
            "billing.plan_selection_settled",
            user_id=user.id,
            wanted=user.pending_plan.value,
            got=target.value,
        )
        user.pending_plan = None
        user.pending_interval = None

    await db.flush()


async def _apply_plan(db: AsyncSession, subscription: Subscription) -> None:
    """Push the subscription's plan onto whoever it belongs to.

    This is the denormalisation `FeatureService` reads on every request. It is
    written here and nowhere else, so "what changes a user's plan" has exactly
    one answer.

    A subscription that is not paid resolves to Free — and resolves *only* the
    plan. No project, stack, export, or run is touched, which is what makes a
    downgrade reversible by paying again.
    """
    if subscription.organization_id is not None:
        await _apply_org_plan(db, subscription)
        return
    if subscription.user_id is None:
        return

    user = await db.get(User, subscription.user_id)
    if user is None:
        return
    await sync_user_plan(db, user)


async def _apply_org_plan(db: AsyncSession, subscription: Subscription) -> None:
    """The organization variant: update the org row, then fan the change out
    to every member — a team downgrade must reach the people it covers."""
    from app.models.organization import Organization, OrganizationMember

    org = await db.get(Organization, subscription.organization_id)
    if org is None:
        return

    target = subscription.plan if subscription.is_paid else Plan.FREE
    if org.plan is not target:
        logger.info(
            "billing.org_plan_changed",
            organization_id=org.id,
            was=org.plan.value,
            now=target.value,
            status=subscription.status.value,
        )
    org.plan = target
    org.seats_purchased = subscription.seats
    await db.flush()

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
    for member_user in members:
        await sync_user_plan(db, member_user)


async def change_seats(
    db: AsyncSession,
    user: User,
    *,
    seats: int,
    organization_id: str | None = None,
) -> tuple[int, int]:
    """Adjust the team's purchased seats, prorated (M21). Owner only.

    Returns `(purchased, used)`. The local rows are updated optimistically and
    the webhook confirms — same trust model as every other provider write.
    Shrinking below the current membership is refused: removing a member frees
    a seat, a seat change does not remove members.
    """
    from app.models.organization import Organization, OrganizationMember
    from app.services import organization_service

    if organization_id is not None:
        org, member = await organization_service.get_membership(
            db, user=user, organization_id=organization_id
        )
        if member.role is not OrgRole.OWNER:
            raise Forbidden("Only the owner can change seats.")
    else:
        found = await db.scalar(
            select(Organization).where(
                Organization.owner_id == user.id, Organization.deleted_at.is_(None)
            )
        )
        if found is None:
            raise NotFound("You do not own an organization.")
        org = found

    used = int(
        await db.scalar(
            select(func.count())
            .select_from(OrganizationMember)
            .where(OrganizationMember.organization_id == org.id)
        )
        or 0
    )
    if seats < used:
        raise ValidationFailed.on_field(
            "seats",
            f"The team has {used} members — remove members before reducing seats.",
        )

    subscription = await db.scalar(
        select(Subscription).where(
            Subscription.organization_id == org.id,
            Subscription.status != SubscriptionStatus.CANCELED,
        )
    )
    if subscription is None:
        raise PlanRequired(
            "Seat billing requires a Team subscription.",
            details={"required_plan": Plan.TEAM.value},
        )

    client = razorpay_integration.get_client()
    if client is not None and subscription.provider_subscription_id:
        await client.update_subscription_quantity(
            subscription_id=subscription.provider_subscription_id, quantity=seats
        )

    subscription.seats = seats
    org.seats_purchased = seats
    await db.flush()
    logger.info(
        "billing.seats_changed", organization_id=org.id, seats=seats, used=used
    )
    return seats, used


async def cancel_org_subscription(db: AsyncSession, subscription: Subscription) -> None:
    """Cancel at period end — the time is paid for. When Razorpay is not
    configured the local mark is all there is, and that is fine: unconfigured
    billing means no card was ever charged."""
    client = razorpay_integration.get_client()
    if client is not None and subscription.provider_subscription_id:
        await client.cancel_subscription(
            subscription_id=subscription.provider_subscription_id, at_cycle_end=True
        )
    subscription.cancel_at_period_end = True
    await db.flush()


async def _resolve_subscription(db: AsyncSession, obj: dict[str, Any]) -> Subscription | None:
    """Find our row from a Razorpay subscription entity.

    Three routes, in order of reliability: the subscription id we stored, the
    notes we set at checkout, then the customer. The last is a fallback because
    a customer can in principle hold more than one subscription, and matching
    on it alone would attach the wrong one.

    The middle and last routes read `notes` and `customer_id` — Razorpay's
    field names. They were `metadata.client_reference_id` and `customer` when
    this was Stripe, which meant both fallbacks were dead: every delivery
    resolved on the first route or not at all.
    """
    if isinstance(sub_id := obj.get("id"), str):
        found = await _by_provider_subscription(db, sub_id)
        if found is not None:
            return found

    notes = obj.get("notes")
    if isinstance(notes, dict):
        user_id = notes.get("user_id")
        if isinstance(user_id, str):
            user = await db.get(User, user_id)
            if user is not None:
                found = await get_subscription(db, user)
                if found is not None:
                    return found

    if isinstance(customer_id := obj.get("customer_id"), str):
        return await _by_customer(db, customer_id)

    return None


def _provider_plan_id(obj: dict[str, Any]) -> str | None:
    """The plan id off a Razorpay subscription entity.

    Flat, unlike Stripe's `items.data[0].price.id` — Razorpay subscriptions
    carry exactly one plan, which is also why there is no item id to thread
    through a seat change.
    """
    value = obj.get("plan_id")
    return value if isinstance(value, str) and value else None


def _quantity(obj: dict[str, Any]) -> int | None:
    """Seat count. Absent on some deliveries, in which case the row keeps what
    it had rather than silently dropping to one."""
    value = obj.get("quantity")
    if isinstance(value, int) and value > 0:
        return value
    return None


def _timestamp(value: Any) -> datetime | None:
    """Razorpay sends Unix seconds, and `null` for dates that do not apply
    yet — an unstarted subscription has no `current_end`."""
    if not isinstance(value, int) or value <= 0:
        return None
    return datetime.fromtimestamp(value, tz=UTC)
