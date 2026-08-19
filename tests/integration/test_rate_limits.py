"""Rate limiting as a caller experiences it (M23).

The unit tests cover the window arithmetic. What is asserted here is the part
a client actually depends on: the headers arrive on ordinary responses, the
refusal is a 429 in the standard envelope, and the endpoints that must never
be limited are not.
"""

from __future__ import annotations

import pytest
from httpx import AsyncClient

from app.core import rate_limit
from app.models.user import Plan

pytestmark = pytest.mark.usefixtures("seeded_catalog")

READS = "/api/v1/catalog/models"


async def test_a_read_carries_its_budget_on_success(client: AsyncClient) -> None:
    """Not only on the refusal. A client that discovers its budget by being
    refused finds out at the point it is already too late to slow down."""
    response = await client.get(READS)

    assert response.status_code == 200
    assert response.headers["X-RateLimit-Limit"] == "100"  # anonymous
    assert response.headers["X-RateLimit-Remaining"] == "99"
    assert int(response.headers["X-RateLimit-Reset"]) > 0


async def test_the_budget_counts_down_across_requests(client: AsyncClient) -> None:
    first = await client.get(READS)
    second = await client.get(READS)

    assert int(second.headers["X-RateLimit-Remaining"]) == (
        int(first.headers["X-RateLimit-Remaining"]) - 1
    )


async def test_exhausting_the_window_returns_429_in_the_standard_envelope(
    client: AsyncClient,
) -> None:
    window = rate_limit.READ.by_plan[None]
    assert window is not None
    for _ in range(window.limit):
        await client.get(READS)

    response = await client.get(READS)

    assert response.status_code == 429
    body = response.json()
    assert body["success"] is False
    assert body["error"]["code"] == "RATE_LIMITED"
    # A client that has to parse prose to know how long to wait will not.
    assert int(response.headers["Retry-After"]) >= 1
    assert response.headers["X-RateLimit-Remaining"] == "0"


async def test_health_is_never_rate_limited(client: AsyncClient) -> None:
    """A throttled liveness probe is an outage that restarts healthy
    containers — the failure mode that makes a limiter worse than none."""
    window = rate_limit.READ.by_plan[None]
    assert window is not None
    for _ in range(window.limit + 5):
        await client.get(READS)

    health = await client.get("/health")

    assert health.status_code == 200
    assert "X-RateLimit-Limit" not in health.headers


async def test_a_tool_run_does_not_spend_the_read_budget(client: AsyncClient) -> None:
    """Different classes, different keys. Browsing the catalogue must not use
    up the allowance for running a tool, or the two limits are really one.

    The first tool run is what *mints* the anonymous session, which moves this
    caller's limiter key from `ip:` to `a:`. So the run happens first and the
    two reads are measured either side of a stable key — otherwise this
    measures the key change rather than the thing under test.
    """
    await client.post(
        "/api/v1/tools/cost/token-calculator",
        json={"text": "hello world", "model_id": "gpt-4o"},
    )

    before = await client.get(READS)
    await client.post(
        "/api/v1/tools/cost/token-calculator",
        json={"text": "hello again", "model_id": "gpt-4o"},
    )
    after = await client.get(READS)

    spent = int(before.headers["X-RateLimit-Remaining"]) - int(
        after.headers["X-RateLimit-Remaining"]
    )
    assert spent == 1, "the tool run consumed part of the read budget"


async def test_an_anonymous_read_does_not_mint_a_session(client: AsyncClient) -> None:
    """The limiter keys on identity, and reaching for the *run* identity to
    get one would set an anonymous session cookie on every crawler that
    touches a public read."""
    response = await client.get(READS)

    assert response.status_code == 200
    assert "set-cookie" not in {key.lower() for key in response.headers}


def test_every_router_that_should_be_limited_is() -> None:
    """Grepped rather than trusted, because the failure is silent: a new
    router added without a limit looks exactly like one that has one, and
    nothing fails until it is abused."""
    from pathlib import Path

    source = Path("app/main.py").read_text(encoding="utf-8")
    # The three exempt routers, each for a stated reason in main.py.
    exempt = {"health_router", "auth_router", "billing_router"}

    unlimited: list[str] = []
    for line in source.splitlines():
        if not line.strip().startswith("app.include_router("):
            continue
        name = line.split("app.include_router(")[1].split(".router")[0]
        if name in exempt:
            continue
        if "dependencies=" not in line and "dependencies=" not in source:
            unlimited.append(name)

    assert unlimited == [], f"routers with no rate limit class: {unlimited}"


def test_the_two_classes_cover_every_plan() -> None:
    """A plan missing from a class silently means "no limit", which is the
    wrong default for a security control."""
    for klass in rate_limit.CLASSES.values():
        assert set(klass.by_plan) == {None, Plan.FREE, Plan.PRO, Plan.TEAM}
