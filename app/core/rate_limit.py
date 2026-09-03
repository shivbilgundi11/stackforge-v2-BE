"""Sliding-window rate limiting (M23).

**This is the abuse layer, not the business layer.** The distinction is the
whole design and it is easy to lose:

* `plan_quotas` (M20, `feature_service`) owns what a plan *entitles* you to —
  tool runs per day, AI calls per day, exports per month. Those are product
  decisions, editable per plan, surfaced as "used 5 of 5", and refused with a
  402 that carries an upgrade prompt.
* This module owns how fast anyone may ask, whoever they are. Its limits are
  operational, not commercial; they exist so one caller cannot exhaust the
  database, the Redis instance, or the API budget for everyone else. Refusal
  is a 429 and it says come back later, not pay us.

Enforcing the same daily figure in both places would be two sources of truth
for one number, and the one nobody edits would drift silently. So the hourly
windows live here, the daily and monthly allowances live in `plan_quotas`, and
neither reimplements the other. See D-53 in the decision log.

**It fails open, and it stops asking.** A Redis outage must not take the API
down with it — the same trade every cache and quota read in this codebase
already makes (D-30). An attacker who can also take Redis down does not need
to be rate limited; they have a better attack available.

Failing open is not enough on its own, because *how long it takes to fail*
matters as much as the verdict. This runs on nearly every route, so paying the
connect timeout per request turns an unreachable Redis into roughly a second
of added latency on every call in the product — which is the failure
`core/redis.py` already documents having fixed once, by cutting the retry
policy down to a single attempt. One attempt is still one second. So a failure
opens a breaker: for `OUTAGE_COOLDOWN_SECONDS` afterwards the limiter does not
touch Redis at all and returns the fail-open verdict immediately.
"""

from __future__ import annotations

import time
from typing import Final, NamedTuple

from app.core.logging import get_logger
from app.core.redis import Keys, get_redis
from app.models.user import Plan

logger = get_logger("ratelimit")


class Window(NamedTuple):
    """A limit and the period it applies over."""

    limit: int
    seconds: int

    @property
    def label(self) -> str:
        return f"{self.limit}/{'h' if self.seconds == 3600 else f'{self.seconds}s'}"


HOUR: Final = 3600

#: How long to stop calling Redis after a failure.
#:
#: Long enough that a sustained outage costs one timeout rather than one per
#: request, short enough that recovery is not noticeable. Deliberately not
#: tied to the window length: this is about the health of the connection, not
#: about the policy being enforced.
OUTAGE_COOLDOWN_SECONDS: Final = 5.0

#: When the breaker closes again. Module state, because the connection it
#: describes is process-wide — a per-caller breaker would learn the same fact
#: separately for every identity and save nothing.
_unavailable_until: float = 0.0


class RateLimitClass(NamedTuple):
    """One row of the policy table in 06-API-Contract §Rate limits."""

    name: str
    #: By plan. `None` as a *value* means the class does not limit that plan
    #: here — which is not the same as unlimited: the daily allowance in
    #: `plan_quotas` may still apply, and for tool runs on Free it is the only
    #: thing that does.
    by_plan: dict[Plan | None, Window | None]


#: The `None` key is a caller with no session. That used to be the anonymous
#: tier, which reached everything; it is now only the genuinely public routes —
#: a share link and the identity probe — which are keyed by IP and get the
#: tightest window, because an unauthenticated caller is the one that cannot be
#: held to an account.
READ: Final = RateLimitClass(
    "read",
    {
        None: Window(100, HOUR),
        Plan.FREE: Window(300, HOUR),
        Plan.PRO: Window(1000, HOUR),
        Plan.TEAM: Window(2000, HOUR),
    },
)

#: Only the paid tiers appear. Free is capped by the daily allowance in
#: `plan_quotas`, which is a far tighter bound than any hourly window would be
#: — adding one here would be dead code that looks load-bearing. No tool route
#: is reachable without a session, so the `None` key never resolves here.
TOOL_RUN: Final = RateLimitClass(
    "tool_run",
    {
        None: None,
        Plan.FREE: None,
        Plan.PRO: Window(600, HOUR),
        Plan.TEAM: Window(1200, HOUR),
    },
)

CLASSES: Final = {klass.name: klass for klass in (READ, TOOL_RUN)}


class Decision(NamedTuple):
    """The outcome, and everything the response headers need.

    Returned rather than raised so the caller can attach headers to a *success*
    too. `X-RateLimit-Remaining` only on the refusal would mean a client
    learns its budget exclusively by being refused, which is precisely when it
    is too late to slow down.
    """

    allowed: bool
    limit: int
    remaining: int
    #: Unix seconds when the window frees up enough to retry.
    reset_at: int

    @property
    def retry_after(self) -> int:
        """Never zero: a `Retry-After: 0` invites an immediate retry, which is
        the behaviour the limit exists to prevent."""
        return max(1, self.reset_at - int(time.time()))

    def headers(self) -> dict[str, str]:
        return {
            "X-RateLimit-Limit": str(self.limit),
            "X-RateLimit-Remaining": str(self.remaining),
            "X-RateLimit-Reset": str(self.reset_at),
        }


#: What a fail-open looks like. `remaining` is the full limit rather than zero
#: so a degraded Redis does not make every client believe it is out of budget
#: and back off in unison.
def _unlimited(window: Window) -> Decision:
    return Decision(True, window.limit, window.limit, int(time.time()) + window.seconds)


async def check(klass: RateLimitClass, *, identity_key: str, plan: Plan | None) -> Decision | None:
    """Count this request against the window. `None` when the class does not
    limit this plan, which is not a refusal and carries no headers.

    The window is a sorted set of request timestamps, trimmed to the period on
    every call — a true sliding window rather than a fixed bucket. A fixed
    bucket lets a caller spend the whole allowance in the last second of one
    period and the whole of the next in the first second of the next, which is
    twice the intended rate at exactly the moment load is already spiking.
    """
    global _unavailable_until

    window = klass.by_plan.get(plan)
    if window is None:
        return None

    key = Keys.rate_limit(klass.name, identity_key)
    now = time.time()
    cutoff = now - window.seconds

    # The breaker. Checked before the call rather than after a failure,
    # because the point is to not make the call.
    if now < _unavailable_until:
        return _unlimited(window)

    try:
        redis = get_redis()
        pipe = redis.pipeline()
        # Trim first, so `zcard` counts only what is still inside the window.
        pipe.zremrangebyscore(key, 0, cutoff)
        pipe.zcard(key)
        # The member has to be unique per request or two calls in the same
        # millisecond collapse into one and the limit silently doubles.
        pipe.zadd(key, {f"{now:.6f}:{_nonce()}": now})
        # Without this a key for a caller who never returns lives forever.
        pipe.expire(key, window.seconds + 60)
        pipe.zrange(key, 0, 0, withscores=True)
        results = await pipe.execute()
    except Exception as exc:
        _unavailable_until = time.time() + OUTAGE_COOLDOWN_SECONDS
        # Logged at the moment the breaker opens, so an outage produces a
        # readable trickle rather than one line per request.
        logger.warning(
            "ratelimit.unavailable",
            klass=klass.name,
            error=str(exc),
            cooldown_seconds=OUTAGE_COOLDOWN_SECONDS,
        )
        return _unlimited(window)

    # A success closes the breaker early, so recovery does not wait out a
    # cooldown that is already over.
    _unavailable_until = 0.0

    # `zcard` ran *before* this request was added, so it is the count of
    # prior requests: `used` is that plus this one.
    used = int(results[1] or 0) + 1
    oldest = results[4]
    oldest_at = float(oldest[0][1]) if oldest else now
    reset_at = int(oldest_at + window.seconds) + 1

    allowed = used <= window.limit
    if not allowed:
        logger.info(
            "ratelimit.exceeded",
            klass=klass.name,
            limit=window.limit,
            plan=plan.value if plan else "no-session",
        )

    return Decision(
        allowed=allowed,
        limit=window.limit,
        remaining=max(0, window.limit - used),
        reset_at=reset_at,
    )


_counter = 0


def _nonce() -> str:
    """Distinguishes two requests landing in the same microsecond.

    A process-local counter rather than a uuid: this runs on every limited
    request, and the only requirement is uniqueness within one key inside one
    window, which the timestamp already nearly provides.
    """
    global _counter
    _counter = (_counter + 1) % 1_000_000
    return str(_counter)


def reset_breaker() -> None:
    """Close the breaker. Test seam, and the runbook's answer to "Redis is back
    but the limiter has not noticed yet"."""
    global _unavailable_until
    _unavailable_until = 0.0


async def reset(klass: RateLimitClass, identity_key: str) -> None:
    """Drop a caller's window. Test seam, and the runbook's answer to
    "someone is locked out and should not be"."""
    try:
        await get_redis().delete(Keys.rate_limit(klass.name, identity_key))
    except Exception as exc:  # pragma: no cover — the fail-open path again
        logger.warning("ratelimit.reset_failed", error=str(exc))
