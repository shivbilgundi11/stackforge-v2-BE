"""Redis client and the key namespace.

Every key template lives here. A typo'd key string is a silent cache miss that
nobody notices for months, so they are not written at call sites.
"""

from __future__ import annotations

from typing import Final

from redis.asyncio import Redis, from_url
from redis.backoff import NoBackoff
from redis.retry import Retry

from app.core.config import settings


class Keys:
    """Four namespaces on one instance."""

    RATE_LIMIT: Final = "rl:{scope}:{identity}"
    LOGIN_FAILURES: Final = "rl:login-fail:{identity}"
    PRICING: Final = "cache:pricing:{family}:{provider}"
    AI_RESPONSE: Final = "cache:ai:{purpose}:{digest}"
    QUOTA: Final = "quota:{metric}:{identity}:{period}"

    @staticmethod
    def rate_limit(scope: str, identity: str) -> str:
        return Keys.RATE_LIMIT.format(scope=scope, identity=identity)

    @staticmethod
    def login_failures(identity: str) -> str:
        return Keys.LOGIN_FAILURES.format(identity=identity)

    @staticmethod
    def quota(metric: str, identity: str, period: str) -> str:
        return Keys.QUOTA.format(metric=metric, identity=identity, period=period)


_client: Redis | None = None


def get_redis() -> Redis:
    """The shared client, configured to fail *fast* when Redis is down.

    Every caller of this module fails open: a cache miss or a quota read that
    errors is treated as "allow and carry on". That is the right trade, but it
    is only a good one if the failure is cheap. With redis-py's default retry
    policy a single unreachable call costs the connect timeout three times
    over, and an endpoint that touches the cache in three places pays it three
    times in series — which turned a 200ms recommendation into a 17-second one
    on a machine whose Redis was simply not running.

    So: one attempt, no backoff, a one-second ceiling. A degraded cache should
    cost milliseconds, not seconds.
    """
    global _client
    if _client is None:
        _client = from_url(
            settings.redis_url,
            encoding="utf-8",
            decode_responses=True,
            socket_connect_timeout=1,
            socket_timeout=1,
            retry=Retry(NoBackoff(), 0),
            health_check_interval=30,
        )
    return _client


def set_redis(client: Redis | None) -> None:
    """Test seam. `conftest` swaps in fakeredis so unit tests need no container."""
    global _client
    _client = client


async def check_redis() -> bool:
    """Readiness probe. Never raises — reports."""
    try:
        return bool(await get_redis().ping())
    except Exception:
        return False


async def close_redis() -> None:
    global _client
    if _client is not None:
        await _client.aclose()
        _client = None
