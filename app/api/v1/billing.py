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
from app.integrations import stripe as stripe_integration
from app.models.billing import Metric, PlanQuota
from app.models.user import Plan
from app.schemas.billing import (
    CancellationIn,
    CheckoutIn,
    CheckoutOut,
    InvoiceOut,
    PlanFeatureOut,
    PlanLimitOut,
    PortalOut,
    PricingPlanOut,
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

#: Which meters a person is shown. Seats are meaningless until M21 gives them
#: members to fill, and a meter permanently reading "0 of 5" reads as broken.
VISIBLE_METRICS = (
    Metric.TOOL_RUNS_PER_DAY,
    Metric.AI_CALLS_PER_DAY,
    Metric.PROJECTS,
    Metric.SAVED_STACKS,
    Metric.EXPORTS_PER_MONTH,
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
                monthly_cents=spec.monthly_cents,
                annual_cents=spec.annual_cents,
                annual_saving_cents=plan_data.annual_saving_cents(spec),
                currency=plan_data.CURRENCY,
                per_seat=spec.per_seat,
                trial_days=spec.trial_days,
                highlights=list(spec.highlights),
                cta=spec.cta,
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
    return any(billing_service.price_id_for(plan, interval) for interval in ("monthly", "annual"))


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
        )
    )


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

    url = await billing_service.start_checkout(
        db, user, plan=plan, interval=payload.interval, seats=payload.seats
    )
    return ok(CheckoutOut(url=url))


@router.post("/portal-session", response_model=Envelope[PortalOut], name="create_portal_session")
async def create_portal_session(db: Db, user: CurrentUser) -> dict[str, Any]:
    return ok(PortalOut(url=await billing_service.open_portal(db, user)))


@router.post("/cancellation", response_model=Envelope[SubscriptionOut], name="set_cancellation")
async def set_cancellation(db: Db, user: CurrentUser, payload: CancellationIn) -> dict[str, Any]:
    """Cancel at period end, or undo it.

    Never an immediate cutoff: the period is paid for and stays. Undo is the
    same endpoint because "I changed my mind" happens far more often than the
    cancellation itself, and it should not require finding a different button.
    """
    await billing_service.set_cancellation(db, user, cancel=payload.cancel)
    return await get_subscription(db, user)


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


@router.post("/webhook", response_model=Envelope[WebhookOut], name="stripe_webhook")
async def stripe_webhook(
    request: Request,
    db: Db,
    stripe_signature: str | None = Header(default=None, alias="Stripe-Signature"),
) -> dict[str, Any]:
    """The one endpoint authenticated by a signature rather than a session.

    Reads the raw body, because the signature covers the exact bytes Stripe
    sent — re-serialising parsed JSON changes them and every delivery fails
    verification.

    Always answers 200 once the signature checks out, including when a handler
    raised. The failure is written onto the event row and retried by the
    scheduled job; returning a 500 would ask Stripe to retry work that has
    already been recorded, and after enough failures Stripe disables the
    endpoint entirely — which is a far worse state than a row with an error on
    it that a job will pick up in an hour.
    """
    body = await request.body()
    event = stripe_integration.verify_signature(body, stripe_signature)

    record = await billing_service.record_event(db, event)
    if record is None:
        # A redelivery of something already applied. This is the case the whole
        # table exists for, and it is a normal, frequent, uninteresting one.
        logger.info("billing.webhook_duplicate", event_id=event.get("id"), type=event.get("type"))
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
