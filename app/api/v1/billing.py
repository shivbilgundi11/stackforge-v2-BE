"""Plans, checkout, the portal, and the webhook (M20).

`GET /plans` is public and unauthenticated: it is what the pricing page renders
from, and a pricing page behind a token cannot be crawled or linked. Everything
else needs an account, except the webhook — which needs a *signature*, and is
the one endpoint in this application that is authenticated by something other
than a session.
"""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any

from fastapi import APIRouter, Header, Request
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.api.deps import CallerIdentity, CurrentUser, Db
from app.core.errors import ValidationFailed
from app.core.logging import get_logger
from app.core.responses import Envelope, ok
from app.data import plans as plan_data
from app.integrations import razorpay as razorpay_integration
from app.models.billing import Metric, PlanQuota
from app.models.user import Plan
from app.schemas.billing import (
    CancellationIn,
    CheckoutIn,
    CheckoutOut,
    InvoiceOut,
    PlanFeatureOut,
    PlanLimitOut,
    PlanSelectionIn,
    PricingPlanOut,
    SeatChangeIn,
    SeatChangeOut,
    SubscriptionOut,
    UsageSummaryOut,
    WebhookOut,
)
from app.services import billing_service, feature_service

logger = get_logger("billing.api")

router = APIRouter(tags=["billing"])

#: Human labels for the meters. The metric keys are stable machine names; these
#: are what a usage row is called on screen.
METRIC_LABELS: dict[Metric, str] = {
    Metric.TOOL_RUNS_PER_DAY: "Tool runs per day",
    Metric.AI_CALLS_PER_DAY: "AI-assisted runs per day",
    Metric.PROJECTS: "Projects",
    Metric.SAVED_STACKS: "Saved stacks",
    Metric.EXPORTS_PER_MONTH: "Exports per month",
    Metric.SEATS: "Seats",
}

#: Which meters a person is shown. Seats joined the list with M21 — the
#: membership count now consumes them.
VISIBLE_METRICS = (
    Metric.TOOL_RUNS_PER_DAY,
    Metric.AI_CALLS_PER_DAY,
    Metric.PROJECTS,
    Metric.SAVED_STACKS,
    Metric.EXPORTS_PER_MONTH,
    Metric.SEATS,
)


@router.get("/plans", response_model=Envelope[list[PricingPlanOut]], name="list_plans")
async def list_plans(db: Db, identity: CallerIdentity) -> dict[str, Any]:
    """The pricing table, limits included.

    Read from `plan_quotas` rather than from the copy in `data/plans.py`, so
    raising the free tier with an `UPDATE` changes the page too. That is the
    whole reason the limits live in a table: a marketing number and an enforced
    number that can disagree eventually will.
    """
    limits = await _limits_by_plan(db)

    payload: list[PricingPlanOut] = []
    for spec in plan_data.PLANS:
        payload.append(
            PricingPlanOut(
                key=spec.plan.value,
                label=spec.label,
                tagline=spec.tagline,
                monthly_minor=spec.monthly_minor,
                annual_minor=spec.annual_minor,
                annual_saving_minor=plan_data.annual_saving_minor(spec),
                currency=plan_data.CURRENCY,
                per_seat=spec.per_seat,
                trial_days=spec.trial_days,
                highlights=list(spec.highlights),
                cta=spec.cta,
                self_serve=spec.checkout,
                # A plan with no price configured cannot be bought here, however
                # much the copy would like to sell it.
                checkout=spec.checkout and _has_price(spec.plan),
                current=identity.is_authenticated and identity.plan is spec.plan,
                features=[
                    PlanFeatureOut(
                        key=feature.key.value,
                        label=feature.label,
                        included=plan_data.outranks(spec.plan, feature.minimum_plan),
                        pitch=feature.pitch,
                    )
                    for feature in plan_data.FEATURES
                ],
                limits=[
                    PlanLimitOut(
                        metric=metric.value,
                        label=METRIC_LABELS[metric],
                        limit=limits.get(spec.plan, {}).get(metric),
                    )
                    for metric in VISIBLE_METRICS
                ],
            )
        )

    return ok(payload)


def _has_price(plan: Plan) -> bool:
    return any(billing_service.plan_id_for(plan, interval) for interval in ("monthly", "annual"))


async def _limits_by_plan(db: AsyncSession) -> dict[Plan, dict[Metric, int | None]]:
    """The registered-account limits, per plan.

    The anonymous rows are deliberately excluded: they are a rate-limiting
    decision, not a plan being sold, and listing them on a pricing page would
    advertise how much can be had without signing up.
    """
    rows = (await db.execute(select(PlanQuota).where(PlanQuota.anonymous.is_(False)))).scalars()
    out: dict[Plan, dict[Metric, int | None]] = {}
    for row in rows:
        out.setdefault(row.plan, {})[row.metric] = row.limit_value
    return out


@router.get("/subscription", response_model=Envelope[SubscriptionOut], name="get_subscription")
async def get_subscription(db: Db, user: CurrentUser) -> dict[str, Any]:
    summary = await billing_service.summary(db, user)
    return ok(
        SubscriptionOut(
            plan=summary.plan.value,
            status=summary.status.value if summary.status else None,
            checkout_available=summary.checkout_available,
            seats=summary.seats,
            cancel_at_period_end=summary.cancel_at_period_end,
            current_period_end=summary.current_period_end,
            trial_ends_at=summary.trial_ends_at,
            past_due_since=summary.past_due_since,
            grace_days_left=summary.grace_days_left,
            pending_plan=summary.pending_plan.value if summary.pending_plan else None,
            pending_interval=summary.pending_interval,
            payment_required=summary.payment_required,
        )
    )


@router.post("/plan-selection", response_model=Envelope[SubscriptionOut], name="select_plan")
async def select_plan(db: Db, user: CurrentUser, payload: PlanSelectionIn) -> dict[str, Any]:
    """Choose the plan this account intends to buy, or decline and stay free.

    Separate from `checkout-session` because choosing and paying are separated
    in time: the choice is made on the signup form, and the card arrives at the
    wall — possibly on a different device, days later. Storing the choice is
    what lets the second half of that happen at all.
    """
    chosen = billing_service.pending_plan_for(payload.plan) if payload.plan else None
    if payload.plan not in (None, "free") and chosen is None:
        raise ValidationFailed.on_field("plan", "That plan cannot be bought here.")

    await billing_service.select_plan(db, user, plan=chosen, interval=payload.interval)
    return await get_subscription(db, user)


@router.get("/usage", response_model=Envelope[UsageSummaryOut], name="get_usage")
async def get_usage(db: Db, identity: CallerIdentity) -> dict[str, Any]:
    """Every meter, for an anonymous caller as much as a paying one.

    Anonymous is not an error case here. They are metered too, and the sidebar
    meter is the thing that makes the limit visible before it is hit — which is
    the only moment the gate can convert rather than annoy.
    """
    states = [await feature_service.check(db, identity, metric) for metric in VISIBLE_METRICS]
    return ok(
        UsageSummaryOut(
            plan=identity.plan.value if identity.is_authenticated else "anonymous",
            quotas=[state.to_schema() for state in states],
        )
    )


@router.post("/checkout-session", response_model=Envelope[CheckoutOut], name="create_checkout")
async def create_checkout(db: Db, user: CurrentUser, payload: CheckoutIn) -> dict[str, Any]:
    try:
        plan = Plan(payload.plan)
    except ValueError:
        raise ValidationFailed.on_field("plan", "No such plan.") from None

    handle = await billing_service.start_checkout(
        db, user, plan=plan, interval=payload.interval, seats=payload.seats
    )
    return ok(CheckoutOut(subscription_id=handle.subscription_id, key_id=handle.key_id))


@router.post("/reconcile", response_model=Envelope[SubscriptionOut], name="reconcile_subscription")
async def reconcile_subscription(db: Db, user: CurrentUser) -> dict[str, Any]:
    """Ask Razorpay what this account is actually subscribed to, and apply it.

    The webhook stays authoritative; this is the repair path for when one does
    not arrive. Because the provider creates a subscription before it is paid
    for, `subscription.activated` is the only thing that says a payment
    succeeded — and a single lost delivery used to mean a customer who had paid
    and an account that never moved, with no way back through the product.

    Callable by the account itself rather than operators only. The person who
    just paid is the one who knows something is wrong, is already looking at
    the screen, and should not have to open a support ticket to have a plan
    they have been charged for.
    """
    await billing_service.reconcile(db, user)
    return await get_subscription(db, user)


# `POST /portal-session` went with Stripe (D-50). Razorpay has no hosted
# billing portal, and the two things the portal did are already here:
# `GET /invoices` reads them live, and `POST /cancellation` is the button.


@router.post("/cancellation", response_model=Envelope[SubscriptionOut], name="set_cancellation")
async def set_cancellation(db: Db, user: CurrentUser, payload: CancellationIn) -> dict[str, Any]:
    """Cancel at period end, or undo it.

    Never an immediate cutoff: the period is paid for and stays. Undo is the
    same endpoint because "I changed my mind" happens far more often than the
    cancellation itself, and it should not require finding a different button.
    """
    await billing_service.set_cancellation(db, user, cancel=payload.cancel)
    return await get_subscription(db, user)


@router.post("/seats", response_model=Envelope[SeatChangeOut], name="update_seats")
async def update_seats(db: Db, user: CurrentUser, payload: SeatChangeIn) -> dict[str, Any]:
    """Adjust the team's purchased seats (M21). Owner only — the one capability
    the role matrix reserves for the owner alone.

    Takes effect at the end of the billing cycle, not immediately: Razorpay
    does not prorate (D-51)."""
    seats, used = await billing_service.change_seats(
        db, user, seats=payload.seats, organization_id=payload.organization_id
    )
    return ok(SeatChangeOut(seats=seats, used=used))


@router.get("/invoices", response_model=Envelope[list[InvoiceOut]], name="list_invoices")
async def list_invoices(db: Db, user: CurrentUser) -> dict[str, Any]:
    invoices = await billing_service.list_invoices(db, user)
    return ok(
        [
            InvoiceOut(
                id=str(invoice["id"]),
                number=invoice.get("number"),
                status=invoice.get("status"),
                amount_due=int(invoice.get("amount_due") or 0),
                amount_paid=int(invoice.get("amount_paid") or 0),
                currency=str(invoice.get("currency") or plan_data.CURRENCY),
                created=datetime.fromtimestamp(int(invoice.get("created") or 0), tz=UTC),
                hosted_invoice_url=invoice.get("hosted_invoice_url"),
                invoice_pdf=invoice.get("invoice_pdf"),
            )
            for invoice in invoices
        ]
    )


# ── Webhook ─────────────────────────────────────────────────────────────────


@router.post("/webhook", response_model=Envelope[WebhookOut], name="razorpay_webhook")
async def razorpay_webhook(
    request: Request,
    db: Db,
    razorpay_signature: str | None = Header(default=None, alias="X-Razorpay-Signature"),
    razorpay_event_id: str | None = Header(default=None, alias="X-Razorpay-Event-Id"),
) -> dict[str, Any]:
    """The one endpoint authenticated by a signature rather than a session.

    Reads the raw body, because the signature covers the exact bytes Razorpay
    sent — re-serialising parsed JSON changes them and every delivery fails
    verification.

    The event id comes from `X-Razorpay-Event-Id`, not the body. It is the
    idempotency key, so a delivery without one is rejected rather than given a
    generated id that would make every redelivery look new.

    Always answers 200 once the signature checks out, including when a handler
    raised. The failure is written onto the event row and retried by the
    scheduled job; returning a 500 would ask Razorpay to retry work that has
    already been recorded, and after enough failures a provider disables the
    endpoint entirely — which is a far worse state than a row with an error on
    it that a job will pick up in an hour.
    """
    body = await request.body()
    event = razorpay_integration.verify_signature(body, razorpay_signature)

    if not razorpay_event_id:
        raise ValidationFailed("The webhook carried no event id.")

    record = await billing_service.record_event(db, event, event_id=razorpay_event_id)
    if record is None:
        # A redelivery of something already applied. This is the case the whole
        # table exists for, and it is a normal, frequent, uninteresting one.
        logger.info(
            "billing.webhook_duplicate",
            event_id=razorpay_event_id,
            type=event.get("event"),
        )
        return ok(WebhookOut(received=True))

    try:
        # Savepoint, so a handler that fails part way leaves nothing behind but
        # still leaves the event row and its error message committed.
        async with db.begin_nested():
            detail = await billing_service.process_event(db, record)
    except Exception as exc:
        record.attempts += 1
        record.error = f"{type(exc).__name__}: {exc}"[:2000]
        logger.exception(
            "billing.webhook_failed",
            event_id=record.id,
            type=record.type,
            attempts=record.attempts,
        )
        return ok(WebhookOut(received=True))

    logger.info("billing.webhook_processed", event_id=record.id, type=record.type, detail=detail)
    return ok(WebhookOut(received=True))


__all__ = ["router"]
