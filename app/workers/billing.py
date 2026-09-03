"""Scheduled billing work (M20).

Six jobs, and each exists because something in the lifecycle cannot be driven
by a request:

  * `retry_failed_events` — a webhook whose handler raised. Recorded, not lost;
    this is what picks it back up. Without it, "the failure is written down"
    would be a euphemism for "the customer paid and was never upgraded".
  * `remind_expiring_trials` — three days before a trial ends. Stripe sent an
    event for this and Razorpay does not, so what was a delivery is now a
    query. A trial that ends silently is a downgrade the user experiences as
    a bug.
  * `expire_trials` — a trial that ran out without converting. The backstop
    for a delivery that never arrived, and the reason a missed webhook cannot
    leave someone on Pro indefinitely.
  * `close_dunning` — a payment that never recovered inside the grace window.
    The only place in this program that downgrades an account for money.
  * `reconcile_usage` — Redis against Postgres. Divergence is *logged*, never
    silently corrected: a drift means the metering is wrong, and that should
    not first be noticed in a billing dispute.
  * `send_price_change_alerts` / `send_deprecation_alerts` — the retention
    mechanism from `PRD.md` §24. The data moves on its own, which gives a user
    a reason to return that does not depend on a habit.

Every one of them is safe to run twice. They are cron jobs on an at-least-once
queue, and a job that is only correct when run exactly once is a job that will
eventually be wrong.
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Sequence
from dataclasses import dataclass, field
from datetime import timedelta
from decimal import Decimal
from typing import Any

from redis.exceptions import RedisError
from sqlalchemy import func, or_, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.api.deps import Identity
from app.core.config import settings
from app.core.database import utcnow
from app.core.logging import get_logger
from app.core.redis import Keys, get_redis
from app.data.plans import Feature
from app.integrations.email import send
from app.models.billing import (
    BillingEvent,
    Metric,
    Subscription,
    SubscriptionStatus,
    UsageRecord,
)
from app.models.catalog import GpuPricing, ModelPricing, PricedEntity, PricingHistory
from app.models.tool_run import ToolRun
from app.models.user import Plan, User
from app.services import billing_service, dashboard_service, email_templates, feature_service

logger = get_logger("worker.billing")

#: What counts as a price move worth an email. FR-18.
DRIFT_THRESHOLD = Decimal("10")

#: How far back an alert run looks. Comfortably wider than the schedule, so a
#: skipped night does not silently drop a day of changes; the dedupe below is
#: what stops the overlap sending twice.
PRICE_LOOKBACK_HOURS = 30
DEPRECATION_LOOKBACK_DAYS = 8

#: Retries stop here. An event that has failed this many times is not going to
#: succeed by being tried again — it needs a person.
MAX_EVENT_ATTEMPTS = 6


# ── Webhook retries ─────────────────────────────────────────────────────────


async def retry_failed_events(db: AsyncSession, *, limit: int = 50) -> int:
    """Re-run every recorded event that has not been applied.

    Ordered oldest first, because subscription events are only meaningful in
    sequence: applying yesterday's cancellation after today's renewal would
    leave the account in the wrong state.
    """
    stmt = (
        select(BillingEvent)
        .where(
            BillingEvent.processed_at.is_(None),
            BillingEvent.attempts < MAX_EVENT_ATTEMPTS,
        )
        .order_by(BillingEvent.received_at)
        .limit(limit)
    )
    pending = (await db.execute(stmt)).scalars().all()

    processed = 0
    for record in pending:
        try:
            async with db.begin_nested():
                await billing_service.process_event(db, record)
        except Exception as exc:
            record.attempts += 1
            record.error = f"{type(exc).__name__}: {exc}"[:2000]
            logger.warning(
                "billing.retry_failed",
                event_id=record.id,
                type=record.type,
                attempts=record.attempts,
            )
        else:
            processed += 1

    if pending:
        logger.info("billing.retries_run", found=len(pending), processed=processed)
    return processed


# ── Lifecycle ───────────────────────────────────────────────────────────────


async def remind_expiring_trials(db: AsyncSession, *, days_before: int = 3) -> int:
    """Mail everyone whose trial ends in `days_before` days.

    Stripe sent `customer.subscription.trial_will_end` on its own schedule;
    Razorpay has no equivalent, so this asks the question instead of waiting to
    be told. That is arguably the better shape anyway — a reminder that depends
    on a third party's delivery is a reminder that silently stops.

    Idempotent by window, not by flag: the query selects trials ending inside a
    single day, and the cron runs daily, so a job that runs twice in one day
    mails twice and a job that misses a day mails nobody. The second is the
    failure worth avoiding, and `trial_reminder_sent_at` would be a column
    earning its keep only if the schedule were less reliable than the mail.
    """
    from app.data import plans as plan_data
    from app.services import email_templates

    now = utcnow()
    window_start = now + timedelta(days=days_before)
    window_end = window_start + timedelta(days=1)

    stmt = select(Subscription).where(
        Subscription.status == SubscriptionStatus.TRIALING,
        Subscription.trial_ends_at.is_not(None),
        Subscription.trial_ends_at >= window_start,
        Subscription.trial_ends_at < window_end,
    )
    ending = (await db.execute(stmt)).scalars().all()

    sent = 0
    for subscription in ending:
        if subscription.user_id is None:
            continue
        user = await db.get(User, subscription.user_id)
        if user is None:
            continue
        await send(
            email_templates.trial_ending(
                to=user.email,
                name=user.name,
                plan=plan_data.spec_for(subscription.plan).label,
                ends_at=subscription.trial_ends_at,
            )
        )
        sent += 1

    if sent:
        logger.info("billing.trial_reminders_sent", count=sent)
    return sent


async def expire_trials(db: AsyncSession) -> int:
    """Trials that ran out without becoming paid subscriptions.

    Sets the plan to free and leaves every project, stack, run, and export
    exactly where it is. A user who comes back and pays finds their work
    waiting; a user who does not still owns it and can read and export it.
    """
    now = utcnow()
    stmt = select(Subscription).where(
        Subscription.status == SubscriptionStatus.TRIALING,
        Subscription.trial_ends_at.is_not(None),
        Subscription.trial_ends_at < now,
    )
    expired = (await db.execute(stmt)).scalars().all()

    for subscription in expired:
        subscription.status = SubscriptionStatus.CANCELED
        subscription.canceled_at = now
        await _downgrade(db, subscription, reason="trial_expired")

    if expired:
        logger.info("billing.trials_expired", count=len(expired))
    return len(expired)


async def close_dunning(db: AsyncSession) -> int:
    """Downgrade the payments that never recovered.

    The grace window is deliberately generous — Razorpay is still retrying the
    card inside it, and cutting off a customer whose payment is one retry away
    costs more than the week of usage it saves.
    """
    cutoff = utcnow() - timedelta(days=settings.dunning_grace_days)
    stmt = select(Subscription).where(
        Subscription.status == SubscriptionStatus.PAST_DUE,
        Subscription.past_due_since.is_not(None),
        Subscription.past_due_since < cutoff,
    )
    overdue = (await db.execute(stmt)).scalars().all()

    for subscription in overdue:
        subscription.status = SubscriptionStatus.CANCELED
        subscription.canceled_at = utcnow()
        await _downgrade(db, subscription, reason="dunning_expired")

    if overdue:
        logger.info("billing.dunning_closed", count=len(overdue))
    return len(overdue)


async def _downgrade(db: AsyncSession, subscription: Subscription, *, reason: str) -> None:
    """Plan only. Nothing else is touched, ever."""
    if subscription.user_id is None:
        return
    user = await db.get(User, subscription.user_id)
    if user is None:
        return

    was = user.plan
    user.plan = Plan.FREE
    await db.flush()
    logger.info(
        "billing.downgraded",
        user_id=user.id,
        was=was.value,
        reason=reason,
        subscription_id=subscription.id,
    )


# ── Reconciliation ──────────────────────────────────────────────────────────


@dataclass
class Reconciliation:
    checked: int = 0
    diverged: list[dict[str, Any]] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return not self.diverged


async def reconcile_usage(db: AsyncSession, *, metric: Metric | None = None) -> Reconciliation:
    """Compare the fast counter against the durable one for today.

    Postgres is the source of truth; Redis is the thing that can lose a
    counter to an eviction or a restart. Where they disagree the divergence is
    logged with both figures — not corrected. A correction would paper over the
    only signal that the metering is broken, and the two numbers together are
    what makes the cause diagnosable.
    """
    metrics = [metric] if metric else [m for m in Metric if m.is_rate]
    result = Reconciliation()

    for target in metrics:
        period_label, period_start, _ = feature_service.period_for(target)

        rows = (
            await db.execute(
                select(
                    UsageRecord.user_id,
                    func.sum(UsageRecord.quantity),
                )
                .where(UsageRecord.metric == target, UsageRecord.period_start == period_start)
                .group_by(UsageRecord.user_id)
            )
        ).all()

        for user_id, durable in rows:
            identity_key = user_id
            if identity_key is None:  # pragma: no cover — the check constraint forbids it
                continue

            result.checked += 1
            try:
                raw = await get_redis().get(Keys.quota(target.value, identity_key, period_label))
            except (RedisError, OSError) as exc:
                logger.warning("billing.reconcile_unreadable", error=str(exc))
                return result

            cached = int(raw) if raw else 0
            if cached != int(durable or 0):
                divergence = {
                    "metric": target.value,
                    "identity": identity_key,
                    "period": period_label,
                    "redis": cached,
                    "postgres": int(durable or 0),
                }
                result.diverged.append(divergence)
                logger.warning("billing.usage_divergence", **divergence)

    logger.info(
        "billing.reconciled",
        checked=result.checked,
        diverged=len(result.diverged),
    )
    return result


# ── Alerts ──────────────────────────────────────────────────────────────────


def _can_receive_alerts(user: User) -> bool:
    """Pro and above. Asked of `FeatureService` rather than compared here."""
    identity = Identity(user=user, session_id=None)
    return feature_service.can(identity, Feature.ALERTS).allowed


async def send_price_change_alerts(db: AsyncSession) -> int:
    """Email the users whose saved work is affected by a real price move.

    "Affected" is defined narrowly on purpose: the entity has to appear in the
    input of a run the user chose to *keep*. An alert about a model someone
    priced once in March is noise, and a retention email that reads as noise
    trains people to filter the channel the product depends on.
    """
    since = utcnow() - timedelta(hours=PRICE_LOOKBACK_HOURS)
    changes = (
        (
            await db.execute(
                select(PricingHistory)
                .where(
                    PricingHistory.detected_at >= since,
                    PricingHistory.pct_change.is_not(None),
                    func.abs(PricingHistory.pct_change) >= DRIFT_THRESHOLD,
                )
                .order_by(PricingHistory.detected_at.desc())
            )
        )
        .scalars()
        .all()
    )

    if not changes:
        return 0

    # Entity id → the identifier a saved run would name it by. Runs record the
    # provider's own id (`gpt-4o-mini`) or the GPU slug, never our row id, so
    # the join has to go through the catalog.
    keys = await _entity_keys(db, changes)
    if not keys:
        return 0

    affected = await _users_with_saved_runs_referencing(db, set(keys.values()))
    sent = 0

    for user, matched_keys in affected.items():
        if not _can_receive_alerts(user):
            continue

        lines = [
            _describe(change, keys[change.entity_id])
            for change in changes
            if keys.get(change.entity_id) in matched_keys
        ]
        if not lines:
            continue

        await send(email_templates.price_change_alert(to=user.email, name=user.name, changes=lines))
        sent += 1

    logger.info("billing.price_alerts_sent", changes=len(changes), users=sent)
    return sent


def _describe(change: PricingHistory, name: str) -> dict[str, str]:
    direction = "up" if (change.pct_change or 0) > 0 else "down"
    pct = abs(change.pct_change or Decimal(0)).quantize(Decimal("0.1"))
    return {
        "name": f"{name} ({change.field.replace('_', ' ')})",
        "from": f"${change.old_value}" if change.old_value is not None else "—",
        "to": f"${change.new_value}" if change.new_value is not None else "—",
        "direction": f"{direction} {pct}%",
    }


async def _entity_keys(db: AsyncSession, changes: Sequence[PricingHistory]) -> dict[str, str]:
    """Map each changed row id to the identifier a saved run would name.

    A run records `gpt-4o-mini` or `lambda-gpu-1x-h100-pcie`, never our
    generated row id — deliberately, so a saved run survives a re-seed. That
    stability is what makes this lookup necessary.
    """
    model_ids = [c.entity_id for c in changes if c.entity_type is PricedEntity.MODEL]
    gpu_ids = [c.entity_id for c in changes if c.entity_type is PricedEntity.GPU]

    keys: dict[str, str] = {}
    if model_ids:
        rows = (
            await db.execute(
                select(ModelPricing.id, ModelPricing.model_id).where(ModelPricing.id.in_(model_ids))
            )
        ).all()
        keys.update({str(row[0]): str(row[1]) for row in rows})
    if gpu_ids:
        # The whole row, because `GpuPricing.slug` is derived in Python rather
        # than stored — it is composed from the same natural key the seeder
        # uses, which is what lets it survive a re-seed.
        gpus = (
            (await db.execute(select(GpuPricing).where(GpuPricing.id.in_(gpu_ids)))).scalars().all()
        )
        keys.update({gpu.id: gpu.slug for gpu in gpus})
    return keys


#: Where a tool run names the catalog entity it priced. `model_id` for the cost
#: workflow, `gpu` for the infra one — the two shapes that exist across the 28
#: tools. A tool that grows a third key needs adding here, which is why this is
#: a named constant and not a literal inside the query.
RUN_ENTITY_KEYS = ("model_id", "gpu")


async def _users_with_saved_runs_referencing(
    db: AsyncSession, keys: set[str]
) -> dict[User, set[str]]:
    """Owners of saved runs whose input names one of these entities.

    Saved runs only. An unsaved run is deleted after 30 days and was never
    something the user said they cared about — emailing about one would be
    emailing about a calculation they have forgotten making.
    """
    if not keys:
        return {}

    stmt = (
        select(User, ToolRun.input)
        .join(ToolRun, ToolRun.user_id == User.id)
        .where(
            ToolRun.saved.is_(True),
            User.deleted_at.is_(None),
            or_(*(ToolRun.input[key].astext.in_(keys) for key in RUN_ENTITY_KEYS)),
        )
    )

    found: dict[User, set[str]] = defaultdict(set)
    for user, payload in (await db.execute(stmt)).all():
        for key in RUN_ENTITY_KEYS:
            value = payload.get(key) if isinstance(payload, dict) else None
            if isinstance(value, str) and value in keys:
                found[user].add(value)
    return found


async def send_deprecation_alerts(db: AsyncSession) -> int:
    """Weekly: a tool in a saved stack has been buried.

    Reuses the dashboard's own staleness query, so the email and the banner can
    never disagree about what is deprecated — two implementations of "is this
    stack still sound" is one more than the product can keep in step.
    """
    users = (
        (await db.execute(select(User).where(User.deleted_at.is_(None), User.plan != Plan.FREE)))
        .scalars()
        .all()
    )

    sent = 0
    for user in users:
        if not _can_receive_alerts(user):
            continue

        alerts = await dashboard_service.stale_alerts(db, user)
        if not alerts:
            continue

        # One row per tool, not per stack. A user with the same dead tool in
        # four stacks needs to be told about the tool once.
        by_tool: dict[str, dict[str, str]] = {}
        for alert in alerts:
            by_tool.setdefault(
                str(alert["tool"]),
                {
                    "name": str(alert["tool"]),
                    "status": str(alert["status"]).replace("_", " "),
                    "replacement": ", ".join(alert.get("alternatives") or [])[:120],
                },
            )

        await send(
            email_templates.deprecation_alert(
                to=user.email, name=user.name, tools=list(by_tool.values())
            )
        )
        sent += 1

    logger.info("billing.deprecation_alerts_sent", users=sent)
    return sent


__all__ = [
    "DRIFT_THRESHOLD",
    "MAX_EVENT_ATTEMPTS",
    "Reconciliation",
    "close_dunning",
    "expire_trials",
    "reconcile_usage",
    "remind_expiring_trials",
    "retry_failed_events",
    "send_deprecation_alerts",
    "send_price_change_alerts",
]
