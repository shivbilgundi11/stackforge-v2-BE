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
    """Anonymous and Free get 5 tool runs a *day* from `plan_quotas` — far
    tighter than any hourly window. A limit here as well would be two sources
    of truth for one number, and the one nobody edits drifts."""
    for plan in (None, Plan.FREE):
        assert await rate_limit.check(rate_limit.TOOL_RUN, identity_key="u:d", plan=plan) is None


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
