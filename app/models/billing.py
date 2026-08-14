"""Subscriptions, quotas, usage, and the billing event log (M20).

Four tables, and the interesting decisions are in the last three.

**`plan_quotas` is configuration in a table, not constants in a module.** Every
limit the product enforces lives here. Changing the free tier from 5 runs to 10
is an `UPDATE`, not a deploy — which matters because `PRD.md` §17 names "Pro
conversion below 2 % after month 4" as a failure signal, and the first response
to that signal is to move exactly these numbers. A limit that requires a
release to change will not be tuned.

**`usage_records` is the durable half of a two-tier counter.** Redis serves the
hot path; this table is what survives a flush and what a billing dispute is
answered from. They are reconciled nightly, and a divergence is logged rather
than silently corrected, because a drift means the metering is wrong and that
should not be discovered from a customer's invoice.

**`billing_events.id` is the provider's event id, and it is the primary key.**
That is the whole idempotency design: insert first, process second. Payment
providers retry deliveries, and a duplicate that re-applies a plan change would
upgrade someone twice or downgrade them mid-period. A duplicate here hits the
primary key and is skipped before any handler runs.
"""

from __future__ import annotations

import enum
from datetime import date, datetime

from sqlalchemy import (
    CheckConstraint,
    Date,
    DateTime,
    Enum,
    ForeignKey,
    Index,
    Integer,
    String,
    Text,
    UniqueConstraint,
    text,
)
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import Mapped, mapped_column

from app.core.database import Base, TimestampMixin, new_id
from app.models.user import Plan


class SubscriptionStatus(str, enum.Enum):
    """The five states this product reasons about.

    Deliberately fewer than any provider's own vocabulary — Razorpay alone has
    nine — because the extra ones all collapse into one of these for every
    decision the product makes. The mapping lives in `billing_service`, in one
    dictionary, rather than spreading a provider's words through the app."""

    TRIALING = "trialing"
    ACTIVE = "active"
    PAST_DUE = "past_due"
    CANCELED = "canceled"
    INCOMPLETE = "incomplete"


class Metric(str, enum.Enum):
    """What is metered.

    Two shapes, deliberately in one enum. `TOOL_RUNS_PER_DAY` and
    `AI_CALLS_PER_DAY` are *rates* — counters that reset on a period boundary.
    `PROJECTS`, `SAVED_STACKS`, and `SEATS` are *levels* — a count of rows that
    exist right now, with no reset. `EXPORTS_PER_MONTH` is a rate on a longer
    period.

    They share an enum because they share a wire shape and a table, and the
    dialog that reports "used 5 of 5" does not care which kind it is holding.
    `FeatureService` is where the difference is handled, once.
    """

    TOOL_RUNS_PER_DAY = "tool_runs_per_day"
    AI_CALLS_PER_DAY = "ai_calls_per_day"
    PROJECTS = "projects"
    SAVED_STACKS = "saved_stacks"
    EXPORTS_PER_MONTH = "exports_per_month"
    SEATS = "seats"

    @property
    def is_rate(self) -> bool:
        return self in _RATE_METRICS

    @property
    def period(self) -> str:
        """`day`, `month`, or `plan` — the window the counter resets on."""
        if self in _DAILY_METRICS:
            return "day"
        if self is Metric.EXPORTS_PER_MONTH:
            return "month"
        return "plan"


_DAILY_METRICS = frozenset({Metric.TOOL_RUNS_PER_DAY, Metric.AI_CALLS_PER_DAY})
_RATE_METRICS = _DAILY_METRICS | {Metric.EXPORTS_PER_MONTH}


class Subscription(Base, TimestampMixin):
    """One paid relationship, owned by a user or an organization.

    `organization_id` is nullable and unused until M21, but the column exists
    now: retrofitting a second owner type onto a table that assumed one is the
    kind of migration that has to touch every query, and seat billing is
    already a named dependency of this module.
    """

    __tablename__ = "subscriptions"

    id: Mapped[str] = mapped_column(String(64), primary_key=True, default=lambda: new_id("sub"))

    user_id: Mapped[str | None] = mapped_column(
        String(64), ForeignKey("users.id", ondelete="CASCADE")
    )
    organization_id: Mapped[str | None] = mapped_column(
        String(64), ForeignKey("organizations.id", ondelete="CASCADE")
    )

    #: The provider's customer record outlives any single subscription — a user
    #: who cancels and resubscribes six months later must land on the same
    #: record, or their invoice history splits in two.
    #:
    #: Nullable only because of the rows in between: checkout briefly created
    #: subscriptions without a customer, to keep Razorpay's hosted page
    #: working, and those rows can never be backfilled because Razorpay never
    #: made a customer for them (D-50 → D-52). Every row written since carries
    #: one from the moment it is created.
    provider_customer_id: Mapped[str | None] = mapped_column(String(80))
    provider_subscription_id: Mapped[str | None] = mapped_column(String(80), unique=True)
    #: Which plan this subscription is on, so a plan change can be detected by
    #: comparing against configuration rather than by trusting the event order.
    provider_plan_id: Mapped[str | None] = mapped_column(String(80))

    plan: Mapped[Plan] = mapped_column(
        Enum(Plan, name="plan", values_callable=lambda e: [m.value for m in e], create_type=False),
        nullable=False,
    )
    status: Mapped[SubscriptionStatus] = mapped_column(
        Enum(
            SubscriptionStatus,
            name="subscription_status",
            values_callable=lambda e: [m.value for m in e],
        ),
        nullable=False,
    )

    seats: Mapped[int] = mapped_column(Integer, nullable=False, default=1, server_default=text("1"))

    current_period_start: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    current_period_end: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    cancel_at_period_end: Mapped[bool] = mapped_column(
        nullable=False, default=False, server_default=text("false")
    )
    trial_ends_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    canceled_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))

    #: When the dunning window opened. A payment failure does not downgrade
    #: immediately — it starts a clock, and this is the clock.
    past_due_since: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))

    __table_args__ = (
        CheckConstraint(
            "num_nonnulls(user_id, organization_id) = 1",
            name="exactly_one_owner",
        ),
        # One live subscription per owner. Two would make "what plan is this
        # user on" a question with two answers, and the denormalised
        # `users.plan` column can only hold one of them.
        Index(
            "uq_subscriptions_user_live",
            "user_id",
            unique=True,
            postgresql_where=text("user_id IS NOT NULL AND status <> 'canceled'"),
        ),
        # The same invariant for the other owner type (M21).
        Index(
            "uq_subscriptions_organization_live",
            "organization_id",
            unique=True,
            postgresql_where=text("organization_id IS NOT NULL AND status <> 'canceled'"),
        ),
        Index("ix_subscriptions_provider_customer_id", "provider_customer_id"),
        Index(
            "ix_subscriptions_trial_ends_at",
            "trial_ends_at",
            postgresql_where=text("status = 'trialing'"),
        ),
        Index(
            "ix_subscriptions_past_due_since",
            "past_due_since",
            postgresql_where=text("status = 'past_due'"),
        ),
    )

    @property
    def is_paid(self) -> bool:
        """Trialing counts. A trial without the features is not a trial."""
        return self.status in (SubscriptionStatus.TRIALING, SubscriptionStatus.ACTIVE)


class PlanQuota(Base, TimestampMixin):
    """One limit, for one plan.

    `limit_value` NULL means unlimited — not a very large number. A sentinel
    like 999_999 reads as a real limit in the UI ("0 of 999999 used") and
    invites arithmetic that treats it as one.

    The `anonymous` pseudo-plan is stored here too, as a row with
    `plan = free` and `anonymous = true`. Anonymous callers are metered like
    everyone else (they are the top of the funnel, and an unmetered anonymous
    tier is a free API), but their limits are lower than a registered free
    account's — otherwise signing up buys nothing.
    """

    __tablename__ = "plan_quotas"

    id: Mapped[str] = mapped_column(String(64), primary_key=True, default=lambda: new_id("pq"))

    plan: Mapped[Plan] = mapped_column(
        Enum(Plan, name="plan", values_callable=lambda e: [m.value for m in e], create_type=False),
        nullable=False,
    )
    anonymous: Mapped[bool] = mapped_column(
        nullable=False, default=False, server_default=text("false")
    )
    metric: Mapped[Metric] = mapped_column(
        Enum(Metric, name="quota_metric", values_callable=lambda e: [m.value for m in e]),
        nullable=False,
    )
    #: NULL is unlimited.
    limit_value: Mapped[int | None] = mapped_column(Integer)

    __table_args__ = (
        UniqueConstraint(
            "plan", "anonymous", "metric", name="uq_plan_quotas_plan_anonymous_metric"
        ),
        CheckConstraint(
            "limit_value IS NULL OR limit_value >= 0",
            name="limit_not_negative",
        ),
        # Anonymous is a variant of free, never of a paid plan.
        CheckConstraint(
            "anonymous IS FALSE OR plan = 'free'",
            name="anonymous_is_free",
        ),
    )


class UsageRecord(Base):
    """The durable counter.

    One row per consumption event rather than one row per period with a
    running total: an increment is an insert, which never contends, and the
    period total is a `SUM` over an index. A single mutable counter row per
    (user, metric, period) would be the hottest write in the system and would
    serialise every tool run by the same user.
    """

    __tablename__ = "usage_records"

    id: Mapped[str] = mapped_column(String(64), primary_key=True, default=lambda: new_id("use"))

    user_id: Mapped[str | None] = mapped_column(
        String(64), ForeignKey("users.id", ondelete="CASCADE")
    )
    anonymous_session_id: Mapped[str | None] = mapped_column(
        String(64), ForeignKey("anonymous_sessions.id", ondelete="CASCADE")
    )
    organization_id: Mapped[str | None] = mapped_column(String(64))

    metric: Mapped[Metric] = mapped_column(
        Enum(
            Metric,
            name="quota_metric",
            values_callable=lambda e: [m.value for m in e],
            create_type=False,
        ),
        nullable=False,
    )
    quantity: Mapped[int] = mapped_column(Integer, nullable=False, default=1)

    #: The bucket this consumption belongs to, as a date. A daily metric uses
    #: the day; a monthly metric uses the first of the month. Storing the
    #: bucket rather than deriving it from `created_at` means a reconciliation
    #: query is an index scan and never has to agree with Redis about which
    #: timezone a day starts in.
    period_start: Mapped[date] = mapped_column(Date, nullable=False)

    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, server_default=text("now()")
    )

    __table_args__ = (
        CheckConstraint(
            "num_nonnulls(user_id, anonymous_session_id, organization_id) = 1",
            name="exactly_one_owner",
        ),
        Index("ix_usage_records_user_id_metric_period_start", "user_id", "metric", "period_start"),
        Index(
            "ix_usage_records_anonymous_session_id_metric_period_start",
            "anonymous_session_id",
            "metric",
            "period_start",
        ),
    )


class BillingEvent(Base):
    """The idempotency ledger.

    Rows are never deleted by the app. A processed event is the evidence that a
    plan change was applied once, and pruning that evidence would make a
    redelivered month-old event look new.
    """

    __tablename__ = "billing_events"

    #: The provider's own event id. This *is* the idempotency key — hence a
    #: natural primary key rather than a generated one with a unique index
    #: beside it.
    id: Mapped[str] = mapped_column(String(80), primary_key=True)

    type: Mapped[str] = mapped_column(String(80), nullable=False)
    payload: Mapped[dict[str, object]] = mapped_column(JSONB, nullable=False)

    received_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, server_default=text("now()")
    )
    #: NULL until a handler has succeeded. The retry job selects on exactly
    #: this: received, not processed.
    processed_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    attempts: Mapped[int] = mapped_column(
        Integer, nullable=False, default=0, server_default=text("0")
    )
    error: Mapped[str | None] = mapped_column(Text)

    __table_args__ = (
        Index(
            "ix_billing_events_unprocessed",
            "received_at",
            postgresql_where=text("processed_at IS NULL"),
        ),
    )


__all__ = [
    "BillingEvent",
    "Metric",
    "PlanQuota",
    "Subscription",
    "SubscriptionStatus",
    "UsageRecord",
]
