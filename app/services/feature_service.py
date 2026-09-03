"""The only place a plan question is answered (M20).

Before this module, five services each held their own opinion: `tool_service`
had a daily-run dict, `ai_service` had another, `project_service` had a project
cap, `export_service` had a format→plan map, and `template_service` had a
premium check. Five copies of "what may a Free user do" is five places to
forget, and the old build's gating could not be reasoned about in one sitting.
Everything now routes through `can`, `check`, and `consume`.

Three questions, three answers:

    can(identity, Feature.EXPORT_PDF)          -> Allow | Deny
    check(db, identity, Metric.TOOL_RUNS_PER_DAY)   -> QuotaState
    consume(db, identity, Metric.TOOL_RUNS_PER_DAY) -> QuotaState, or raises

**`can` is pure and `check` is not.** A feature verdict is a comparison against
a map that lives in code (`data/plans.py`), so it needs no I/O and can be
called from anywhere, including a template renderer. A quota needs the limits
table and a counter, so it needs a session.

## Two kinds of metric

*Rates* — tool runs and AI calls per day, exports per month — are counters
against a period. They live in Redis on the hot path and in `usage_records`
durably, and `consume` is what moves them.

*Levels* — projects, saved stacks, seats — are a count of rows that exist right
now. There is nothing to increment: creating the row **is** the increment, and
deleting it is the decrement. For these, `consume` only asks "is there room",
and the caller's insert does the rest.

Conflating the two is how a project cap ends up unable to notice a deletion.

## Failing open

A Redis outage does not stop people using the product. `check` returns zero
used, and `consume` allows. The trade is that an outage window is uncapped,
which is the right way round: the alternative is a cache failure presenting as
a billing failure to every user at once. The durable counter in `usage_records`
is still written, so the reconciliation job can see what the window cost.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, date, datetime, timedelta
from typing import Final

from redis.exceptions import RedisError
from sqlalchemy import Integer, func, select
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.ext.asyncio import AsyncSession

from app.api.deps import Identity
from app.core.database import new_id, utcnow
from app.core.errors import PlanRequired, QuotaExceeded
from app.core.logging import get_logger
from app.core.redis import Keys, get_redis
from app.data.plans import Feature, feature_spec, outranks
from app.models.billing import Metric, PlanQuota, UsageRecord
from app.models.project import Project
from app.models.stack import Stack
from app.models.user import Plan
from app.schemas.tools import QuotaOut

logger = get_logger("features")


# ── Verdicts ────────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class Allow:
    allowed: bool = True


@dataclass(frozen=True)
class Deny:
    """Carries everything the upgrade dialog needs to be specific.

    There was a second axis here — `requires_account`, for a caller with no
    account at all, which led to a signup button rather than a checkout. Every
    caller has an account now, so a denial is always about the plan.
    """

    reason: str
    required_plan: Plan | None = None
    allowed: bool = False

    def as_details(self) -> dict[str, object]:
        return {"required_plan": self.required_plan.value if self.required_plan else None}


Verdict = Allow | Deny


def can(identity: Identity, feature: Feature) -> Verdict:
    """May this caller use this feature?

    Reads `users.plan` through the identity — never a JWT claim. The claim
    mirrors the column but can be a full access-token lifetime stale, which is
    exactly long enough for a downgraded account to keep a paid feature for
    fifteen minutes.
    """
    spec = feature_spec(feature)

    if not outranks(identity.plan, spec.minimum_plan):
        return Deny(
            reason=f"{spec.label} is on the {spec.minimum_plan.value.title()} plan.",
            required_plan=spec.minimum_plan,
        )

    return Allow()


def require(identity: Identity, feature: Feature) -> None:
    """`can`, as a guard. Raises the 402 the client already knows how to render."""
    verdict = can(identity, feature)
    if isinstance(verdict, Deny):
        raise PlanRequired(verdict.reason, details=verdict.as_details())


# ── Limits ──────────────────────────────────────────────────────────────────
#
# `plan_quotas` is read constantly and written about twice a year. An
# in-process cache with a short TTL keeps the hot path off the database without
# making a limit change wait for a deploy — the worst case is that a raised
# limit takes a minute to reach every worker, which is a change nobody is
# watching a clock over.

_CACHE_TTL_SECONDS: Final = 60

_limits_cache: dict[Plan, dict[Metric, int | None]] = {}
_limits_cached_at: datetime | None = None


def invalidate_limits() -> None:
    """Called after a write to `plan_quotas`. Also the test seam."""
    global _limits_cached_at
    _limits_cache.clear()
    _limits_cached_at = None


async def _load_limits(db: AsyncSession) -> dict[Plan, dict[Metric, int | None]]:
    global _limits_cached_at

    fresh = (
        _limits_cached_at is not None
        and (utcnow() - _limits_cached_at).total_seconds() < _CACHE_TTL_SECONDS
    )
    if fresh and _limits_cache:
        return _limits_cache

    rows = (await db.execute(select(PlanQuota))).scalars().all()
    loaded: dict[Plan, dict[Metric, int | None]] = {}
    for row in rows:
        loaded.setdefault(row.plan, {})[row.metric] = row.limit_value

    _limits_cache.clear()
    _limits_cache.update(loaded)
    _limits_cached_at = utcnow()
    return _limits_cache


async def limit_for(db: AsyncSession, identity: Identity, metric: Metric) -> int | None:
    """The limit for this caller and metric. `None` is unlimited.

    An unseeded database has no rows at all. Rather than treat a missing row as
    unlimited — which would silently disable every quota the first time someone
    forgot to seed — a missing row falls back to the free tier's value, and a
    missing free row denies. Failing closed here is safe because the seeder
    runs in the same command as the migrations.
    """
    limits = await _load_limits(db)

    bucket = limits.get(identity.plan)
    if bucket is None or metric not in bucket:
        bucket = limits.get(Plan.FREE, {})
    if metric not in bucket:
        logger.warning("features.missing_quota_row", metric=metric.value, plan=identity.plan.value)
        return 0
    return bucket[metric]


# ── Periods ─────────────────────────────────────────────────────────────────


def period_for(metric: Metric) -> tuple[str, date, datetime | None]:
    """The bucket label, its start date, and when it rolls over.

    The label is what Redis keys on; the date is what `usage_records` stores.
    They are derived from the same `now` so the two counters can never disagree
    about which day a request belonged to — a reconciliation that compares
    different buckets reports drift that is not there.
    """
    now = utcnow()

    if metric.period == "day":
        start = now.date()
        resets = (now + timedelta(days=1)).replace(hour=0, minute=0, second=0, microsecond=0)
        return start.isoformat(), start, resets.replace(tzinfo=UTC)

    if metric.period == "month":
        start = now.date().replace(day=1)
        # The 28th plus four days is always in the next month, for every month.
        next_month = (start + timedelta(days=32)).replace(day=1)
        resets = datetime(next_month.year, next_month.month, 1, tzinfo=UTC)
        return start.strftime("%Y-%m"), start, resets

    # A level metric has no period. The label still has to be stable, because
    # it is part of a Redis key that is never written for these.
    return "plan", now.date(), None


# ── Quota state ─────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class QuotaState:
    metric: Metric
    #: `None` is unlimited. Deliberately not a large sentinel — "3 of 999999
    #: used" renders as a real limit and invites arithmetic that treats it so.
    limit: int | None
    used: int
    period: str
    resets_at: datetime | None
    plan: str

    @property
    def unlimited(self) -> bool:
        return self.limit is None

    @property
    def remaining(self) -> int | None:
        return None if self.limit is None else max(0, self.limit - self.used)

    @property
    def exceeded(self) -> bool:
        return self.limit is not None and self.used >= self.limit

    def to_schema(self) -> QuotaOut:
        return QuotaOut(
            metric=self.metric.value,
            limit=self.limit,
            used=self.used,
            remaining=self.remaining,
            period=self.period,
            resets_at=self.resets_at,
            plan=self.plan,
        )

    def message(self) -> str:
        if self.limit == 0:
            return f"{_METRIC_LABELS[self.metric]} are not included on the {self.plan} plan."
        return (
            f"You have used all {self.limit} {_METRIC_LABELS[self.metric]} on the {self.plan} plan."
        )


_METRIC_LABELS: Final[dict[Metric, str]] = {
    Metric.TOOL_RUNS_PER_DAY: "runs for today",
    Metric.AI_CALLS_PER_DAY: "AI-assisted runs for today",
    Metric.PROJECTS: "projects",
    Metric.SAVED_STACKS: "saved stacks",
    Metric.EXPORTS_PER_MONTH: "exports this month",
    Metric.SEATS: "seats",
}


def _plan_label(identity: Identity) -> str:
    return identity.plan.value


# ── Counting ────────────────────────────────────────────────────────────────


async def _rate_used(identity: Identity, metric: Metric, period: str) -> int:
    try:
        raw = await get_redis().get(Keys.quota(metric.value, identity.key, period))
        return int(raw) if raw else 0
    except (RedisError, OSError, ValueError) as exc:
        logger.warning(
            "features.quota_read_failed",
            metric=metric.value,
            identity=identity.key,
            error=str(exc),
        )
        return 0


async def _level_used(db: AsyncSession, identity: Identity, metric: Metric) -> int:
    """Count the rows a level metric measures."""
    match metric:
        case Metric.PROJECTS:
            stmt = (
                select(func.count())
                .select_from(Project)
                .where(Project.user_id == identity.user.id, Project.archived_at.is_(None))
            )
        case Metric.SAVED_STACKS:
            stmt = select(func.count()).select_from(Stack).where(Stack.user_id == identity.user.id)
        case Metric.SEATS:
            # The membership count of the organization the user acts for:
            # the one they own, else the first they belong to. A user in no
            # organization consumes no seats (M21).
            from app.models.organization import Organization, OrganizationMember

            org_id = await db.scalar(
                select(Organization.id)
                .join(
                    OrganizationMember,
                    OrganizationMember.organization_id == Organization.id,
                )
                .where(
                    OrganizationMember.user_id == identity.user.id,
                    Organization.deleted_at.is_(None),
                )
                .order_by(
                    # Owned orgs first, then oldest membership.
                    (Organization.owner_id != identity.user.id).cast(Integer),
                    OrganizationMember.created_at,
                )
                .limit(1)
            )
            if org_id is None:
                return 0
            stmt = (
                select(func.count())
                .select_from(OrganizationMember)
                .where(OrganizationMember.organization_id == org_id)
            )
        case _:  # pragma: no cover — a rate metric reaching here is a bug
            return 0

    return int(await db.scalar(stmt) or 0)


# ── The three entry points ──────────────────────────────────────────────────


async def check(db: AsyncSession, identity: Identity, metric: Metric) -> QuotaState:
    """Read a counter without moving it. Never raises."""
    period, _, resets_at = period_for(metric)
    limit = await limit_for(db, identity, metric)

    used = (
        await _rate_used(identity, metric, period)
        if metric.is_rate
        else await _level_used(db, identity, metric)
    )

    return QuotaState(
        metric=metric,
        limit=limit,
        used=used,
        period=period,
        resets_at=resets_at,
        plan=_plan_label(identity),
    )


async def consume(
    db: AsyncSession,
    identity: Identity,
    metric: Metric,
    amount: int = 1,
) -> QuotaState:
    """Take `amount` from the allowance, or raise `QuotaExceeded`.

    For a rate metric this increments first and compares second, which is what
    makes it safe under concurrency: two simultaneous requests at the limit
    both see a post-increment value above it, and both are refused rather than
    both reading "one left". The increment is given back when it is refused, so
    a burst of rejections does not permanently consume the day.

    For a level metric there is nothing to increment — the caller's own insert
    is the increment — so this only asks whether there is room. That leaves a
    narrow race where two concurrent creates both find room; the cost is one
    project over the cap, which is not worth a lock. A rate metric, where the
    same race is worth money, is the one that is atomic.
    """
    period, period_start, resets_at = period_for(metric)
    limit = await limit_for(db, identity, metric)
    plan = _plan_label(identity)

    if limit is None:
        # Still recorded. An unlimited plan's usage is exactly what the next
        # pricing conversation needs, and a metric that is only written when it
        # is capped has no history the day someone proposes capping it.
        if metric.is_rate:
            await _increment(identity, metric, period, resets_at, amount)
            await _record(db, identity, metric, period_start, amount)
        return QuotaState(
            metric=metric,
            limit=None,
            used=0,
            period=period,
            resets_at=resets_at,
            plan=plan,
        )

    if not metric.is_rate:
        used = await _level_used(db, identity, metric)
        state = QuotaState(
            metric=metric,
            limit=limit,
            used=used,
            period=period,
            resets_at=resets_at,
            plan=plan,
        )
        if used + amount > limit:
            raise QuotaExceeded(
                state.message(),
                details={"quota": state.to_schema().model_dump(mode="json")},
            )
        return state

    used = await _increment(identity, metric, period, resets_at, amount)
    state = QuotaState(
        metric=metric,
        limit=limit,
        used=used,
        period=period,
        resets_at=resets_at,
        plan=plan,
    )

    if used > limit:
        await _increment(identity, metric, period, resets_at, -amount)
        raise QuotaExceeded(
            state.message(),
            details={
                "quota": QuotaState(
                    metric=metric,
                    limit=limit,
                    used=limit,
                    period=period,
                    resets_at=resets_at,
                    plan=plan,
                )
                .to_schema()
                .model_dump(mode="json")
            },
        )

    await _record(db, identity, metric, period_start, amount)
    return state


async def record(
    db: AsyncSession,
    identity: Identity,
    metric: Metric,
    amount: int = 1,
) -> None:
    """Move both counters without enforcing anything. Never raises.

    For the caller who has already decided to allow the work and only needs it
    counted — the AI layer, whose exhausted allowance returns the rule-based
    result rather than a 402, so the decision is made before the model call and
    the counting happens after it succeeds.

    Splitting check from record gives up `consume`'s atomicity, and that is the
    right trade here: the failure mode is one extra AI call at the boundary of
    a hundred-a-day allowance, against the alternative of a paid model call
    that happened and was never counted.
    """
    if not metric.is_rate:
        return
    period, period_start, resets_at = period_for(metric)
    await _increment(identity, metric, period, resets_at, amount)
    await _record(db, identity, metric, period_start, amount)


async def usage(db: AsyncSession, identity: Identity) -> list[QuotaState]:
    """Every metric at once, for the usage meters on the dashboard and the
    billing page. One call rather than six round trips from the client."""
    return [await check(db, identity, metric) for metric in Metric]


# ── Counter mechanics ───────────────────────────────────────────────────────


async def _increment(
    identity: Identity,
    metric: Metric,
    period: str,
    resets_at: datetime | None,
    amount: int,
) -> int:
    """Move the Redis counter and return the new value.

    Returns 0 on a Redis failure, which reads as "nothing used" and therefore
    allows. Failing open is the documented trade at the top of this module.
    """
    key = Keys.quota(metric.value, identity.key, period)
    try:
        redis = get_redis()
        value = int(await redis.incrby(key, amount))
        if value == amount and resets_at is not None:
            # Expire just past the rollover, so a run at 23:59:59 does not
            # leave a key that outlives its own period.
            await redis.expireat(key, int(resets_at.timestamp()) + 60)
        return value
    except (RedisError, OSError, ValueError) as exc:
        logger.warning(
            "features.quota_increment_failed",
            metric=metric.value,
            identity=identity.key,
            error=str(exc),
        )
        return 0


async def _record(
    db: AsyncSession,
    identity: Identity,
    metric: Metric,
    period_start: date,
    amount: int,
) -> None:
    """The durable half of the counter.

    Wrapped in a savepoint because the owner column carries a foreign key and
    the account can be deleted between the read and this write. Without the
    savepoint that insert would poison the request's transaction and turn a
    routine tool run into a 500 — the usage row is a record, and a record that
    cannot be written must not take the request with it.
    """
    row = UsageRecord(
        id=new_id("use"),
        user_id=identity.user.id,
        metric=metric,
        quantity=amount,
        period_start=period_start,
    )

    try:
        async with db.begin_nested():
            db.add(row)
            await db.flush()
    except SQLAlchemyError as exc:
        logger.warning(
            "features.usage_record_failed",
            metric=metric.value,
            identity=identity.key,
            error=str(exc),
        )


__all__ = [
    "Allow",
    "Deny",
    "QuotaState",
    "Verdict",
    "can",
    "check",
    "consume",
    "invalidate_limits",
    "limit_for",
    "period_for",
    "record",
    "require",
    "usage",
]
