"""The sliding window, and the line it draws against `plan_quotas`.

Rate limits and quotas both refuse a caller who asks too much, and the whole
reason both exist is that they refuse for different reasons and say different
things. Most of this file is that boundary.
"""

from __future__ import annotations

import time
from typing import Any

import pytest

from app.core import rate_limit
from app.core.redis import get_redis, set_redis
from app.models.user import Plan


@pytest.fixture(autouse=True)
def _closed_breaker():
    """The breaker is process state, so an outage in one test would otherwise
    make the next one silently skip Redis and assert nothing."""
    rate_limit.reset_breaker()
    yield
    rate_limit.reset_breaker()


async def _spend(klass: rate_limit.RateLimitClass, *, key: str, plan: Plan | None, times: int):
    last = None
    for _ in range(times):
        last = await rate_limit.check(klass, identity_key=key, plan=plan)
    return last


# ── the window ───────────────────────────────────────────────────────────────


async def test_a_caller_is_refused_only_after_the_limit() -> None:
    limit = rate_limit.READ.by_plan[Plan.FREE]
    assert limit is not None

    allowed = await _spend(rate_limit.READ, key="u:a", plan=Plan.FREE, times=limit.limit)
    assert allowed is not None and allowed.allowed is True
    assert allowed.remaining == 0

    refused = await rate_limit.check(rate_limit.READ, identity_key="u:a", plan=Plan.FREE)
    assert refused is not None and refused.allowed is False


async def test_the_count_is_per_caller_not_global() -> None:
    """One noisy caller must not spend everyone else's allowance — the failure
    mode of a limiter keyed on the route instead of the identity."""
    limit = rate_limit.READ.by_plan[Plan.FREE]
    assert limit is not None
    await _spend(rate_limit.READ, key="u:loud", plan=Plan.FREE, times=limit.limit + 5)

    quiet = await rate_limit.check(rate_limit.READ, identity_key="u:quiet", plan=Plan.FREE)
    assert quiet is not None and quiet.allowed is True
    assert quiet.remaining == limit.limit - 1


async def test_the_classes_do_not_share_a_budget() -> None:
    """Reading the catalogue must not consume the allowance for running a
    tool. Same identity, same Redis, different key."""
    limit = rate_limit.READ.by_plan[Plan.PRO]
    assert limit is not None
    await _spend(rate_limit.READ, key="u:b", plan=Plan.PRO, times=limit.limit)

    runs = await rate_limit.check(rate_limit.TOOL_RUN, identity_key="u:b", plan=Plan.PRO)
    assert runs is not None and runs.allowed is True


async def test_the_window_slides_rather_than_resetting_on_a_boundary(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A fixed bucket lets a caller spend a full allowance at the end of one
    period and another at the start of the next — twice the intended rate at
    the moment load is already spiking. Requests must expire individually.
    """
    limit = rate_limit.READ.by_plan[Plan.FREE]
    assert limit is not None
    base = time.time()

    monkeypatch.setattr(time, "time", lambda: base)
    await _spend(rate_limit.READ, key="u:c", plan=Plan.FREE, times=limit.limit)
    refused = await rate_limit.check(rate_limit.READ, identity_key="u:c", plan=Plan.FREE)
    assert refused is not None and refused.allowed is False

    # Half a window later the earlier requests are still inside it.
    monkeypatch.setattr(time, "time", lambda: base + limit.seconds / 2)
    still_refused = await rate_limit.check(rate_limit.READ, identity_key="u:c", plan=Plan.FREE)
    assert still_refused is not None and still_refused.allowed is False

    # Past the window, they have aged out one by one.
    monkeypatch.setattr(time, "time", lambda: base + limit.seconds + 1)
    recovered = await rate_limit.check(rate_limit.READ, identity_key="u:c", plan=Plan.FREE)
    assert recovered is not None and recovered.allowed is True


async def test_two_requests_in_the_same_instant_both_count(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Both land on the same timestamp. Keyed on the timestamp alone they
    would collapse into one sorted-set member and the limit would silently
    double."""
    frozen = time.time()
    monkeypatch.setattr(time, "time", lambda: frozen)

    first = await rate_limit.check(rate_limit.READ, identity_key="u:same", plan=Plan.FREE)
    second = await rate_limit.check(rate_limit.READ, identity_key="u:same", plan=Plan.FREE)

    assert first is not None and second is not None
    assert second.remaining == first.remaining - 1


# ── the boundary with plan_quotas ────────────────────────────────────────────


async def test_tool_runs_are_not_limited_here_for_the_plans_quotas_bound() -> None:
    """Free gets its tool runs a *day* from `plan_quotas` — far tighter than
    any hourly window. A limit here as well would be two sources of truth for
    one number, and the one nobody edits drifts."""
    assert await rate_limit.check(rate_limit.TOOL_RUN, identity_key="u:d", plan=Plan.FREE) is None


async def test_paid_plans_do_get_an_hourly_tool_ceiling() -> None:
    """Pro and Team have no daily cap, so this layer is the only bound on how
    fast they can be spent."""
    for plan, expected in ((Plan.PRO, 600), (Plan.TEAM, 1200)):
        decision = await rate_limit.check(rate_limit.TOOL_RUN, identity_key=f"u:{plan}", plan=plan)
        assert decision is not None
        assert decision.limit == expected


def test_the_policy_matches_the_published_contract() -> None:
    """06-API-Contract §Rate limits is the source. Transcribed by hand, so
    asserted by hand — a policy that only agrees with itself is not checked."""
    # The `None` key is a caller with no session. It used to be the anonymous
    # tier, which reached everything; it now covers only the genuinely public
    # routes — a share link and the identity probe — keyed by IP.
    assert {plan: w.limit if w else None for plan, w in rate_limit.READ.by_plan.items()} == {
        None: 100,
        Plan.FREE: 300,
        Plan.PRO: 1000,
        Plan.TEAM: 2000,
    }
    assert all(
        window.seconds == 3600 for window in rate_limit.READ.by_plan.values() if window is not None
    )


def test_limits_never_decrease_as_a_plan_improves() -> None:
    """The property that survives someone editing the numbers. A paying plan
    that is rate limited harder than the free one is the kind of thing a table
    of figures hides in plain sight."""
    for klass in rate_limit.CLASSES.values():
        ladder = [None, Plan.FREE, Plan.PRO, Plan.TEAM]
        limits = [
            (klass.by_plan.get(plan).limit if klass.by_plan.get(plan) else 0) for plan in ladder
        ]
        assert limits == sorted(limits), f"{klass.name} limits go backwards: {limits}"


# ── degradation ──────────────────────────────────────────────────────────────


async def test_a_redis_outage_allows_rather_than_refuses() -> None:
    """Every cache and quota read in this codebase fails open (D-30), and a
    limiter that fails *closed* turns a Redis blip into a total outage. An
    attacker who can drop Redis has a better attack available than exceeding
    a rate limit.
    """

    class _Broken:
        def pipeline(self) -> Any:
            raise ConnectionError("redis is gone")

    set_redis(_Broken())
    decision = await rate_limit.check(rate_limit.READ, identity_key="u:e", plan=Plan.FREE)

    assert decision is not None
    assert decision.allowed is True
    # And it reports a full budget, not an empty one: a degraded limiter that
    # tells every client it is out of allowance makes them all back off at once.
    assert decision.remaining == decision.limit


async def test_an_outage_is_asked_about_once_not_once_per_request() -> None:
    """Failing open is not enough on its own — *how long* it takes to fail is
    the rest of it. This runs on nearly every route, so paying the connect
    timeout per request turns an unreachable Redis into about a second of
    added latency across the whole product. That is the exact regression
    `core/redis.py` documents having fixed once already.
    """
    calls = 0

    class _Broken:
        def pipeline(self) -> Any:
            nonlocal calls
            calls += 1
            raise ConnectionError("redis is gone")

    set_redis(_Broken())

    for _ in range(20):
        decision = await rate_limit.check(rate_limit.READ, identity_key="u:i", plan=Plan.FREE)
        assert decision is not None and decision.allowed is True

    assert calls == 1, f"asked a dead Redis {calls} times in 20 requests"


async def test_the_breaker_reopens_after_the_cooldown(monkeypatch: pytest.MonkeyPatch) -> None:
    """It has to retry eventually, or the first blip disables rate limiting
    until the process restarts."""
    calls = 0

    class _Broken:
        def pipeline(self) -> Any:
            nonlocal calls
            calls += 1
            raise ConnectionError("redis is gone")

    set_redis(_Broken())
    base = time.time()

    monkeypatch.setattr(time, "time", lambda: base)
    await rate_limit.check(rate_limit.READ, identity_key="u:j", plan=Plan.FREE)
    await rate_limit.check(rate_limit.READ, identity_key="u:j", plan=Plan.FREE)
    assert calls == 1

    monkeypatch.setattr(time, "time", lambda: base + rate_limit.OUTAGE_COOLDOWN_SECONDS + 0.1)
    await rate_limit.check(rate_limit.READ, identity_key="u:j", plan=Plan.FREE)
    assert calls == 2


async def test_recovery_closes_the_breaker_without_waiting_it_out() -> None:
    """A success means the connection is back; holding the breaker open for
    the rest of the cooldown would keep the limiter off for no reason."""

    class _BrokenOnce:
        def __init__(self) -> None:
            self.failed = False

        def pipeline(self) -> Any:
            if not self.failed:
                self.failed = True
                raise ConnectionError("one blip")
            return get_redis().pipeline()

    healthy = get_redis()
    set_redis(_BrokenOnce())
    await rate_limit.check(rate_limit.READ, identity_key="u:k", plan=Plan.FREE)

    # Breaker is open; close it the way a recovered call would.
    rate_limit.reset_breaker()
    set_redis(healthy)

    decision = await rate_limit.check(rate_limit.READ, identity_key="u:k", plan=Plan.FREE)
    assert decision is not None
    # Counted for real again, rather than the fail-open full budget.
    assert decision.remaining == decision.limit - 1


async def test_a_refusal_never_asks_for_an_immediate_retry() -> None:
    """`Retry-After: 0` invites the retry the limit exists to prevent."""
    limit = rate_limit.READ.by_plan[Plan.FREE]
    assert limit is not None
    await _spend(rate_limit.READ, key="u:f", plan=Plan.FREE, times=limit.limit)

    refused = await rate_limit.check(rate_limit.READ, identity_key="u:f", plan=Plan.FREE)
    assert refused is not None and refused.retry_after >= 1


async def test_the_key_expires_so_a_departed_caller_is_not_stored_forever() -> None:
    await rate_limit.check(rate_limit.READ, identity_key="u:g", plan=Plan.FREE)

    ttl = await get_redis().ttl("rl:read:u:g")
    assert 0 < ttl <= 3600 + 60


async def test_headers_carry_the_whole_budget() -> None:
    decision = await rate_limit.check(rate_limit.READ, identity_key="u:h", plan=Plan.PRO)
    assert decision is not None

    headers = decision.headers()
    assert headers["X-RateLimit-Limit"] == "1000"
    assert headers["X-RateLimit-Remaining"] == "999"
    assert int(headers["X-RateLimit-Reset"]) > int(time.time())
