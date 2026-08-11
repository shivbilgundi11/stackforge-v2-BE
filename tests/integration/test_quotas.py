"""FeatureService: the gate, the meters, and what a downgrade does not do (M20).

Two properties are worth more than the rest of this file:

  * the feature matrix is asserted for every feature-and-plan pair, because a
    gate that is right for four of five plans is a gate nobody can reason
    about — and reasoning about it in one place is the whole point of the
    module;
  * a downgrade deletes nothing. A user who drops to Free keeps their
    projects, can read and export them, and only fails when creating another.
"""

from __future__ import annotations

from datetime import timedelta

import pytest
import time_machine
from httpx import AsyncClient
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.api.deps import Identity
from app.core.database import utcnow
from app.core.errors import QuotaExceeded
from app.data.plans import FEATURES, Feature, outranks
from app.models.billing import Metric, UsageRecord
from app.models.project import Project
from app.models.user import Plan, User
from app.services import feature_service
from tests.conftest import GOOD_PASSWORD, register_and_verify, set_limit


def _identity(plan: Plan | None = None, *, anonymous: bool = False) -> Identity:
    if anonymous:
        return Identity(user=None, anonymous_id="anon_test", session_id=None)
    user = User(id="usr_test", email="q@example.com", name="Q", plan=plan or Plan.FREE)
    return Identity(user=user, anonymous_id=None, session_id=None)


async def _sign_in(
    client: AsyncClient, db: AsyncSession, email: str, *, plan: Plan = Plan.PRO
) -> User:
    user_id = await register_and_verify(client, db, email=email)
    user = await db.get(User, user_id)
    assert user is not None
    user.plan = plan
    await db.flush()

    response = await client.post(
        "/api/v1/auth/login", json={"email": email, "password": GOOD_PASSWORD}
    )
    client.headers["Authorization"] = f"Bearer {response.json()['data']['tokens']['access_token']}"
    return user


# ── The feature matrix ──────────────────────────────────────────────────────


@pytest.mark.parametrize("feature", [spec.key for spec in FEATURES], ids=lambda f: f.value)
@pytest.mark.parametrize("plan", list(Plan), ids=lambda p: p.value)
def test_every_feature_and_plan_pair_has_the_right_verdict(feature: Feature, plan: Plan) -> None:
    """The parametrised matrix M20 asks for. Twelve features by four plans, so
    a feature moved between tiers is checked everywhere at once."""
    from app.data.plans import feature_spec

    spec = feature_spec(feature)
    verdict = feature_service.can(_identity(plan), feature)

    assert verdict.allowed is outranks(plan, spec.minimum_plan)


@pytest.mark.parametrize("feature", [spec.key for spec in FEATURES], ids=lambda f: f.value)
def test_an_anonymous_caller_is_refused_everything_that_needs_an_account(
    feature: Feature,
) -> None:
    from app.data.plans import feature_spec

    spec = feature_spec(feature)
    verdict = feature_service.can(_identity(anonymous=True), feature)

    assert verdict.allowed is not spec.requires_account


def test_an_anonymous_denial_asks_for_an_account_not_a_card() -> None:
    """Two different walls, two different buttons. Sending someone without an
    account to a billing page sends them somewhere they cannot act."""
    verdict = feature_service.can(_identity(anonymous=True), Feature.SAVE_WORK)

    assert isinstance(verdict, feature_service.Deny)
    assert verdict.requires_account is True
    assert verdict.required_plan is None


def test_a_free_users_denial_names_the_plan_to_buy() -> None:
    verdict = feature_service.can(_identity(Plan.FREE), Feature.EXPORT_PDF)

    assert isinstance(verdict, feature_service.Deny)
    assert verdict.requires_account is False
    assert verdict.required_plan is Plan.PRO


# ── Rate metrics ────────────────────────────────────────────────────────────


async def test_the_last_run_succeeds_and_the_next_one_is_refused(db: AsyncSession) -> None:
    identity = _identity(anonymous=True)
    limit = await feature_service.limit_for(db, identity, Metric.TOOL_RUNS_PER_DAY)
    assert limit is not None

    for _ in range(limit):
        await feature_service.consume(db, identity, Metric.TOOL_RUNS_PER_DAY)

    with pytest.raises(QuotaExceeded) as raised:
        await feature_service.consume(db, identity, Metric.TOOL_RUNS_PER_DAY)

    assert raised.value.http_status == 402
    assert raised.value.details is not None
    quota = raised.value.details["quota"]
    assert quota["used"] == limit
    assert quota["limit"] == limit
    assert quota["remaining"] == 0
    assert quota["resets_at"]


async def test_a_refused_run_does_not_permanently_consume_the_day(db: AsyncSession) -> None:
    """The increment is given back, so a burst of rejections at the boundary
    does not push `used` past the limit and make the meter nonsense."""
    identity = _identity(anonymous=True)
    limit = await feature_service.limit_for(db, identity, Metric.TOOL_RUNS_PER_DAY)
    assert limit is not None

    for _ in range(limit):
        await feature_service.consume(db, identity, Metric.TOOL_RUNS_PER_DAY)
    for _ in range(5):
        with pytest.raises(QuotaExceeded):
            await feature_service.consume(db, identity, Metric.TOOL_RUNS_PER_DAY)

    state = await feature_service.check(db, identity, Metric.TOOL_RUNS_PER_DAY)
    assert state.used == limit


async def test_an_unlimited_plan_never_refuses_and_is_still_counted(db: AsyncSession) -> None:
    """A metric only recorded when it is capped has no history the day someone
    proposes capping it."""
    user = User(id="usr_unlimited", email="pro@example.com", name="Pro", plan=Plan.PRO)
    db.add(user)
    await db.flush()
    identity = Identity(user=user, anonymous_id=None, session_id=None)

    for _ in range(50):
        state = await feature_service.consume(db, identity, Metric.TOOL_RUNS_PER_DAY)
        assert state.unlimited

    rows = (
        (
            await db.execute(
                select(UsageRecord).where(
                    UsageRecord.user_id == user.id,
                    UsageRecord.metric == Metric.TOOL_RUNS_PER_DAY,
                )
            )
        )
        .scalars()
        .all()
    )
    assert len(rows) == 50


async def test_the_counter_resets_on_the_period_boundary(db: AsyncSession) -> None:
    identity = _identity(anonymous=True)

    with time_machine.travel("2026-08-11 23:30:00+00:00", tick=False):
        await feature_service.consume(db, identity, Metric.TOOL_RUNS_PER_DAY)
        state = await feature_service.check(db, identity, Metric.TOOL_RUNS_PER_DAY)
        assert state.used == 1
        assert state.period == "2026-08-11"

    with time_machine.travel("2026-08-12 00:30:00+00:00", tick=False):
        state = await feature_service.check(db, identity, Metric.TOOL_RUNS_PER_DAY)
        assert state.used == 0, "a new day is a new bucket"
        assert state.period == "2026-08-12"


async def test_a_monthly_metric_buckets_by_month(db: AsyncSession) -> None:
    identity = _identity(anonymous=True)

    with time_machine.travel("2026-08-31 12:00:00+00:00", tick=False):
        state = await feature_service.consume(db, identity, Metric.EXPORTS_PER_MONTH)
        assert state.period == "2026-08"
        assert state.resets_at is not None
        assert state.resets_at.month == 9

    with time_machine.travel("2026-09-01 00:05:00+00:00", tick=False):
        assert (await feature_service.check(db, identity, Metric.EXPORTS_PER_MONTH)).used == 0


async def test_anonymous_usage_keys_on_the_session_not_the_ip(db: AsyncSession) -> None:
    """Two people behind one office NAT are two users. Keying on the IP would
    make the first of them spend the second's allowance."""
    from app.core.database import new_id
    from app.core.database import utcnow as now
    from app.models.auth import AnonymousSession

    first_id, second_id = new_id("anon"), new_id("anon")
    db.add(AnonymousSession(id=first_id, last_seen_at=now()))
    db.add(AnonymousSession(id=second_id, last_seen_at=now()))
    await db.flush()

    first = Identity(user=None, anonymous_id=first_id, session_id=None)
    second = Identity(user=None, anonymous_id=second_id, session_id=None)

    await feature_service.consume(db, first, Metric.TOOL_RUNS_PER_DAY)
    await feature_service.consume(db, first, Metric.TOOL_RUNS_PER_DAY)

    assert (await feature_service.check(db, first, Metric.TOOL_RUNS_PER_DAY)).used == 2
    assert (await feature_service.check(db, second, Metric.TOOL_RUNS_PER_DAY)).used == 0

    rows = (
        (await db.execute(select(UsageRecord).where(UsageRecord.anonymous_session_id == first_id)))
        .scalars()
        .all()
    )
    assert len(rows) == 2


async def test_a_usage_row_whose_owner_has_vanished_does_not_fail_the_request(
    db: AsyncSession,
) -> None:
    """A stale `anon_` cookie from a purged session must not turn a routine
    tool run into a 500 — the durable record is a record, not a gate."""
    identity = Identity(user=None, anonymous_id="anon_purged_long_ago", session_id=None)

    state = await feature_service.consume(db, identity, Metric.TOOL_RUNS_PER_DAY)

    assert state.used == 1
    assert (await db.execute(select(UsageRecord))).scalars().all() == []


# ── Level metrics, and what a downgrade keeps ───────────────────────────────


async def test_projects_are_counted_from_rows_so_a_deletion_frees_one(
    client: AsyncClient, db: AsyncSession
) -> None:
    """A level metric has no counter to decrement. Conflating it with a rate is
    how a project cap ends up unable to notice a deletion."""
    await _sign_in(client, db, "leveller@example.com")
    await set_limit(db, plan=Plan.PRO, metric=Metric.PROJECTS, value=2)

    first = (await client.post("/api/v1/projects", json={"name": "One"})).json()["data"]
    await client.post("/api/v1/projects", json={"name": "Two"})
    assert (await client.post("/api/v1/projects", json={"name": "Three"})).status_code == 402

    await client.delete(f"/api/v1/projects/{first['id']}")
    assert (await client.post("/api/v1/projects", json={"name": "Three"})).status_code == 201


async def test_a_downgrade_keeps_every_row_and_only_refuses_the_next_one(
    client: AsyncClient, db: AsyncSession
) -> None:
    """The rule the whole module is built around: nothing is deleted on a
    downgrade. Reads succeed, creation returns 402, and paying again restores
    the account exactly as it was."""
    user = await _sign_in(client, db, "downgraded@example.com")
    for index in range(3):
        assert (
            await client.post("/api/v1/projects", json={"name": f"Kept {index}"})
        ).status_code == 201

    user.plan = Plan.FREE
    await db.flush()

    listed = await client.get("/api/v1/projects")
    assert listed.status_code == 200
    assert len(listed.json()["data"]) == 3, "a downgrade must not delete anything"

    rows = (await db.execute(select(Project).where(Project.user_id == user.id))).scalars().all()
    assert len(rows) == 3

    refused = await client.post("/api/v1/projects", json={"name": "One too many"})
    assert refused.status_code == 402
    assert refused.json()["error"]["details"]["quota"]["metric"] == "projects"

    # And paying again restores it, with the work still there.
    user.plan = Plan.PRO
    await db.flush()
    assert (await client.post("/api/v1/projects", json={"name": "Back"})).status_code == 201


# ── The meters ──────────────────────────────────────────────────────────────


async def test_usage_reports_every_visible_meter_for_an_anonymous_caller(
    client: AsyncClient,
) -> None:
    """Anonymous is not an error case. The meter is what makes the limit
    visible before it is hit, which is the only moment a gate converts."""
    response = await client.get("/api/v1/billing/usage")
    assert response.status_code == 200

    data = response.json()["data"]
    assert data["plan"] == "anonymous"
    metrics = {quota["metric"] for quota in data["quotas"]}
    assert metrics == {
        "tool_runs_per_day",
        "ai_calls_per_day",
        "projects",
        "saved_stacks",
        "exports_per_month",
    }


async def test_usage_shows_unlimited_as_null(client: AsyncClient, db: AsyncSession) -> None:
    await _sign_in(client, db, "meters@example.com")

    data = (await client.get("/api/v1/billing/usage")).json()["data"]
    runs = next(q for q in data["quotas"] if q["metric"] == "tool_runs_per_day")

    assert runs["limit"] is None
    assert runs["remaining"] is None


async def test_a_plan_change_flips_a_gate_within_one_request(
    client: AsyncClient, db: AsyncSession
) -> None:
    """The plan is read from the column on every request, never from the JWT
    claim — a 15-minute-stale claim would let a downgraded user keep a paid
    feature for a quarter of an hour."""
    user = await _sign_in(client, db, "flipper@example.com", plan=Plan.FREE)
    run = await client.post(
        "/api/v1/tools/cost/llm-pricing",
        json={
            "model_id": "gpt-4o-mini",
            "input_tokens": 100,
            "output_tokens": 50,
            "requests_per_day": 10,
        },
    )
    run_id = run.json()["data"]["run_id"]

    locked = await client.post(
        "/api/v1/exports", json={"source_type": "run", "source_id": run_id, "format": "pdf"}
    )
    assert locked.status_code == 402

    user.plan = Plan.PRO
    await db.flush()

    # Same token, same session — only the column changed.
    unlocked = await client.post(
        "/api/v1/exports", json={"source_type": "run", "source_id": run_id, "format": "pdf"}
    )
    assert unlocked.status_code in (200, 201, 202)


test_a_plan_change_flips_a_gate_within_one_request = pytest.mark.usefixtures("seeded_catalog")(
    test_a_plan_change_flips_a_gate_within_one_request
)


# ── Limits are data ─────────────────────────────────────────────────────────


async def test_a_limit_change_takes_effect_without_a_deploy(db: AsyncSession) -> None:
    """M20's headline promise, asserted rather than assumed."""
    identity = _identity(anonymous=True)
    await set_limit(db, plan=Plan.FREE, metric=Metric.TOOL_RUNS_PER_DAY, value=1, anonymous=True)

    await feature_service.consume(db, identity, Metric.TOOL_RUNS_PER_DAY)
    with pytest.raises(QuotaExceeded):
        await feature_service.consume(db, identity, Metric.TOOL_RUNS_PER_DAY)

    await set_limit(db, plan=Plan.FREE, metric=Metric.TOOL_RUNS_PER_DAY, value=3, anonymous=True)
    await feature_service.consume(db, identity, Metric.TOOL_RUNS_PER_DAY)


async def test_a_missing_quota_row_denies_rather_than_allowing(db: AsyncSession) -> None:
    """An unseeded database should not silently disable every quota. Failing
    closed is safe here because the seeder runs with the migrations."""
    from app.models.billing import PlanQuota

    for row in (
        (await db.execute(select(PlanQuota).where(PlanQuota.metric == Metric.SAVED_STACKS)))
        .scalars()
        .all()
    ):
        await db.delete(row)
    await db.flush()
    feature_service.invalidate_limits()

    assert await feature_service.limit_for(db, _identity(Plan.PRO), Metric.SAVED_STACKS) == 0


# ── Reconciliation ──────────────────────────────────────────────────────────


async def test_reconciliation_reports_an_injected_divergence(db: AsyncSession) -> None:
    """Divergence is logged, never corrected: a drift means the metering is
    wrong, and papering over it removes the only signal that says so."""
    from app.core.database import new_id
    from app.core.database import utcnow as now
    from app.core.redis import Keys, get_redis
    from app.models.auth import AnonymousSession
    from app.workers import billing as billing_jobs

    anon_id = new_id("anon")
    db.add(AnonymousSession(id=anon_id, last_seen_at=now()))
    await db.flush()
    identity = Identity(user=None, anonymous_id=anon_id, session_id=None)

    await feature_service.consume(db, identity, Metric.TOOL_RUNS_PER_DAY)
    await feature_service.consume(db, identity, Metric.TOOL_RUNS_PER_DAY)

    clean = await billing_jobs.reconcile_usage(db, metric=Metric.TOOL_RUNS_PER_DAY)
    assert clean.ok, clean.diverged

    # Lose a counter, the way an eviction or a restart would.
    period, _, _ = feature_service.period_for(Metric.TOOL_RUNS_PER_DAY)
    await get_redis().delete(Keys.quota(Metric.TOOL_RUNS_PER_DAY.value, anon_id, period))

    drifted = await billing_jobs.reconcile_usage(db, metric=Metric.TOOL_RUNS_PER_DAY)
    assert not drifted.ok
    assert drifted.diverged[0]["redis"] == 0
    assert drifted.diverged[0]["postgres"] == 2


# ── Lifecycle jobs ──────────────────────────────────────────────────────────


async def test_an_expired_trial_drops_to_free_and_keeps_the_work(
    client: AsyncClient, db: AsyncSession
) -> None:
    from app.core.database import new_id
    from app.models.billing import Subscription, SubscriptionStatus
    from app.workers import billing as billing_jobs

    user = await _sign_in(client, db, "expiring@example.com")
    assert (await client.post("/api/v1/projects", json={"name": "Trial work"})).status_code == 201

    db.add(
        Subscription(
            id=new_id("sub"),
            user_id=user.id,
            stripe_customer_id="cus_trial",
            plan=Plan.PRO,
            status=SubscriptionStatus.TRIALING,
            trial_ends_at=utcnow() - timedelta(hours=1),
        )
    )
    await db.flush()

    assert await billing_jobs.expire_trials(db) == 1

    await db.refresh(user)
    assert user.plan is Plan.FREE
    kept = (await db.execute(select(Project).where(Project.user_id == user.id))).scalars().all()
    assert len(kept) == 1, "a trial that ends must not take the work with it"


async def test_dunning_downgrades_only_after_the_grace_window(db: AsyncSession) -> None:
    from app.core.config import settings
    from app.core.database import new_id
    from app.models.billing import Subscription, SubscriptionStatus
    from app.workers import billing as billing_jobs

    user = User(id=new_id("usr"), email="lapsed@example.com", name="Lapsed", plan=Plan.PRO)
    db.add(user)
    # Flushed before the subscription: no `relationship()` joins the two, so
    # the unit of work has no dependency to order the inserts by.
    await db.flush()

    subscription = Subscription(
        id=new_id("sub"),
        user_id=user.id,
        stripe_customer_id="cus_lapsed",
        plan=Plan.PRO,
        status=SubscriptionStatus.PAST_DUE,
        past_due_since=utcnow() - timedelta(days=settings.dunning_grace_days - 1),
    )
    db.add(subscription)
    await db.flush()

    assert await billing_jobs.close_dunning(db) == 0
    assert user.plan is Plan.PRO, "inside the window the features stay"

    subscription.past_due_since = utcnow() - timedelta(days=settings.dunning_grace_days + 1)
    await db.flush()

    assert await billing_jobs.close_dunning(db) == 1
    await db.refresh(user)
    assert user.plan is Plan.FREE
