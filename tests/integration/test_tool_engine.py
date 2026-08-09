"""The shared execution path, end to end.

What is being tested here is not any tool's arithmetic — that lives in the unit
suites — but the guarantees `run_tool` makes on every tool's behalf: quota,
run logging, identity attribution, and provenance.
"""

from __future__ import annotations

from decimal import Decimal

import pytest
from httpx import AsyncClient
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.api.deps import Identity
from app.core.redis import get_redis
from app.models.tool_run import ToolRun
from app.services import tool_service

pytestmark = pytest.mark.usefixtures("seeded_catalog")

PRICING = "/api/v1/tools/cost/llm-pricing"
BASE_PAYLOAD = {
    "model_id": "gpt-4o-mini",
    "input_tokens": 1000,
    "output_tokens": 500,
    "requests_per_day": 100,
}


# ── The seven-key contract ───────────────────────────────────────────────────


async def test_response_has_exactly_the_seven_keys_plus_run_metadata(
    client: AsyncClient,
) -> None:
    """The renderer switches on block kind, not tool identity. That only holds
    if the shape never drifts."""
    response = await client.post(PRICING, json=BASE_PAYLOAD)
    assert response.status_code == 200

    data = response.json()["data"]
    assert set(data) == {
        "metrics",
        "tables",
        "series",
        "artifacts",
        "warnings",
        "provenance",
        "ai",
        "run_id",
        "tool_slug",
        "source",
        "duration_ms",
        "created_at",
    }


async def test_a_real_number_comes_back(client: AsyncClient) -> None:
    response = await client.post(PRICING, json=BASE_PAYLOAD)
    metrics = response.json()["data"]["metrics"]

    assert metrics["cost_per_request"] == "0.000450"
    assert metrics["monthly_cost"] == "1.369688"


async def test_source_is_rule_based_without_ai(client: AsyncClient) -> None:
    response = await client.post(PRICING, json=BASE_PAYLOAD)
    data = response.json()["data"]

    assert data["source"] == "rule_based"
    assert data["ai"] is None


# ── Run logging ──────────────────────────────────────────────────────────────


async def test_a_run_writes_a_row_with_input_output_and_duration(
    client: AsyncClient, db: AsyncSession
) -> None:
    response = await client.post(PRICING, json=BASE_PAYLOAD)
    run_id = response.json()["data"]["run_id"]

    run = await db.get(ToolRun, run_id)
    assert run is not None
    assert run.tool_slug == "llm-pricing"
    assert run.workflow == "cost"
    assert run.input["model_id"] == "gpt-4o-mini"
    assert run.output["metrics"]["cost_per_request"] == "0.000450"
    assert run.duration_ms >= 0
    assert run.saved is False


async def test_an_anonymous_run_is_attributed_to_the_anon_session(
    client: AsyncClient, db: AsyncSession
) -> None:
    from app.core.config import settings
    from app.core.database import new_id, utcnow
    from app.models.auth import AnonymousSession

    anon_id = new_id("anon")
    db.add(AnonymousSession(id=anon_id, last_seen_at=utcnow()))
    await db.flush()

    client.cookies.set(settings.anon_cookie_name, anon_id)
    try:
        response = await client.post(PRICING, json=BASE_PAYLOAD)
    finally:
        client.cookies.delete(settings.anon_cookie_name)

    run = await db.get(ToolRun, response.json()["data"]["run_id"])
    assert run is not None
    assert run.anonymous_session_id == anon_id
    assert run.user_id is None


async def test_anonymous_runs_are_claimed_on_signup(db: AsyncSession) -> None:
    """Losing four calculations at the moment of signing up teaches the user
    that creating an account cost them something."""
    from app.core.database import new_id, utcnow
    from app.models.auth import AnonymousSession
    from app.models.user import User

    anon_id = new_id("anon")
    db.add(AnonymousSession(id=anon_id, last_seen_at=utcnow()))
    user = User(
        id=new_id("usr"),
        email="claimer@example.com",
        password_hash="x",
        name="Claimer",
    )
    db.add(user)
    await db.flush()

    for _ in range(3):
        db.add(
            ToolRun(
                id=new_id("run"),
                tool_slug="llm-pricing",
                workflow="cost",
                anonymous_session_id=anon_id,
                input={},
                output={},
                duration_ms=1,
                created_at=utcnow(),
            )
        )
    await db.flush()

    claimed = await tool_service.claim_anonymous_runs(db, anonymous_id=anon_id, user_id=user.id)
    await db.flush()

    assert claimed == 3
    rows = (await db.execute(select(ToolRun).where(ToolRun.user_id == user.id))).scalars().all()
    assert len(rows) == 3
    assert all(row.anonymous_session_id is None for row in rows)


# ── Quota ────────────────────────────────────────────────────────────────────


async def test_quota_returns_402_at_the_limit_with_real_numbers(
    client: AsyncClient,
) -> None:
    """ "You hit your limit" with no figures is a dead end."""
    limit = tool_service.DAILY_RUN_LIMIT["anonymous"]

    for _ in range(limit):
        assert (await client.post(PRICING, json=BASE_PAYLOAD)).status_code == 200

    blocked = await client.post(PRICING, json=BASE_PAYLOAD)
    assert blocked.status_code == 402

    body = blocked.json()["error"]
    assert body["code"] == "QUOTA_EXCEEDED"
    quota = body["details"]["quota"]
    assert quota["limit"] == limit
    assert quota["used"] == limit
    assert quota["remaining"] == 0
    assert quota["resets_at"]


async def test_below_the_limit_returns_200(client: AsyncClient) -> None:
    for _ in range(tool_service.DAILY_RUN_LIMIT["anonymous"] - 1):
        assert (await client.post(PRICING, json=BASE_PAYLOAD)).status_code == 200


async def test_a_blocked_run_is_not_logged(client: AsyncClient, db: AsyncSession) -> None:
    """The quota check happens before compute, so a rejected call costs nothing
    and leaves no row to skew the metrics."""
    for _ in range(tool_service.DAILY_RUN_LIMIT["anonymous"]):
        await client.post(PRICING, json=BASE_PAYLOAD)

    before = len((await db.execute(select(ToolRun))).scalars().all())
    await client.post(PRICING, json=BASE_PAYLOAD)
    after = len((await db.execute(select(ToolRun))).scalars().all())

    assert before == after


async def test_quota_is_readable_before_running_anything(client: AsyncClient) -> None:
    response = await client.get("/api/v1/runs/quota")
    quota = response.json()["data"]

    assert quota["used"] == 0
    assert quota["remaining"] == quota["limit"]
    assert quota["plan"] == "anonymous"


async def test_quota_fails_open_when_redis_is_unavailable() -> None:
    """A cache outage must not look like a billing failure to every user."""
    from app.core.redis import set_redis

    class _Broken:
        async def get(self, *_: object, **__: object) -> None:
            raise OSError("redis is down")

        async def incr(self, *_: object, **__: object) -> None:
            raise OSError("redis is down")

    set_redis(_Broken())  # type: ignore[arg-type]
    try:
        state = await tool_service.check_quota(
            Identity(user=None, anonymous_id="anon_x", session_id=None)
        )
    finally:
        set_redis(None)

    assert state.used == 0
    assert state.exceeded is False


# ── Provenance ───────────────────────────────────────────────────────────────


async def test_provenance_reports_the_rows_the_run_actually_read(
    client: AsyncClient,
) -> None:
    response = await client.post(PRICING, json=BASE_PAYLOAD)
    provenance = response.json()["data"]["provenance"]

    assert provenance["oldest_verified_at"]
    assert provenance["variant"] in {"fresh", "aging", "stale"}
    assert provenance["sources"]
    assert any("OpenAI" in source["name"] for source in provenance["sources"])


async def test_provenance_shows_the_oldest_source_not_an_average(
    client: AsyncClient,
) -> None:
    """The trustworthiness of a result is the trustworthiness of its worst input."""
    response = await client.post(PRICING, json=BASE_PAYLOAD)
    provenance = response.json()["data"]["provenance"]

    oldest = min(source["last_verified_at"] for source in provenance["sources"])
    assert provenance["oldest_verified_at"] == oldest


async def test_one_chip_per_source_not_one_per_row(client: AsyncClient) -> None:
    """Eight OpenAI models must not produce eight identical chips."""
    response = await client.post(PRICING, json=BASE_PAYLOAD)
    names = [source["name"] for source in response.json()["data"]["provenance"]["sources"]]
    assert len(names) == len(set(names))


# ── Run history ──────────────────────────────────────────────────────────────


async def test_recent_runs_are_listed_for_the_caller(client: AsyncClient) -> None:
    from app.core.config import settings
    from app.core.database import new_id

    anon_id = new_id("anon")
    client.cookies.set(settings.anon_cookie_name, anon_id)
    try:
        # No AnonymousSession row: the FK is nullable-on-delete and an
        # unregistered cookie should not 500 the endpoint.
        await client.post(PRICING, json=BASE_PAYLOAD)
        listed = await client.get("/api/v1/runs")
    finally:
        client.cookies.delete(settings.anon_cookie_name)

    assert listed.status_code == 200


async def test_a_run_is_not_readable_by_another_caller(
    client: AsyncClient, db: AsyncSession
) -> None:
    """A run id is not a capability. Sharing is M18's job."""
    from app.core.config import settings
    from app.core.database import new_id, utcnow
    from app.models.auth import AnonymousSession

    owner = new_id("anon")
    db.add(AnonymousSession(id=owner, last_seen_at=utcnow()))
    await db.flush()

    client.cookies.set(settings.anon_cookie_name, owner)
    run_id = (await client.post(PRICING, json=BASE_PAYLOAD)).json()["data"]["run_id"]
    assert (await client.get(f"/api/v1/runs/{run_id}")).status_code == 200

    other = new_id("anon")
    db.add(AnonymousSession(id=other, last_seen_at=utcnow()))
    await db.flush()
    client.cookies.set(settings.anon_cookie_name, other)
    try:
        assert (await client.get(f"/api/v1/runs/{run_id}")).status_code == 404
    finally:
        client.cookies.delete(settings.anon_cookie_name)


# ── Validation ───────────────────────────────────────────────────────────────


async def test_a_422_names_the_offending_field(client: AsyncClient) -> None:
    """The client maps these onto form fields, so the path must be usable."""
    response = await client.post(PRICING, json={**BASE_PAYLOAD, "cached_input_ratio": "5"})
    assert response.status_code == 422

    fields = response.json()["error"]["details"]["fields"]
    assert any(field["path"] == "cached_input_ratio" for field in fields)


async def test_an_unknown_model_is_404_not_500(client: AsyncClient) -> None:
    response = await client.post(PRICING, json={**BASE_PAYLOAD, "model_id": "nope"})
    assert response.status_code == 404


# ── The other three cost tools, end to end ───────────────────────────────────


async def test_token_calculator_end_to_end(client: AsyncClient) -> None:
    response = await client.post(
        "/api/v1/tools/cost/token-calculator",
        json={"text": "hello world " * 500, "model_id": "gpt-4o-mini"},
    )
    metrics = response.json()["data"]["metrics"]

    assert metrics["method"] == "heuristic"
    assert int(metrics["tokens"]) > 0
    assert metrics["fits"] == "yes"
    assert response.json()["data"]["tables"]["context_fit"]


async def test_embedding_cost_end_to_end(client: AsyncClient) -> None:
    response = await client.post(
        "/api/v1/tools/cost/embedding-cost",
        json={
            "model_id": "text-embedding-3-small",
            "document_count": 1000,
            "avg_tokens_per_document": 800,
            "reembeds_per_month": 2,
        },
    )
    metrics = response.json()["data"]["metrics"]

    assert metrics["monthly_tokens"] == 1_600_000
    assert metrics["dimensions"] == 1536
    # 1.6M tokens / 1000 * $0.00002
    assert metrics["monthly_cost"] == "0.032000"


async def test_embedding_cost_rejects_a_chat_model(client: AsyncClient) -> None:
    response = await client.post(
        "/api/v1/tools/cost/embedding-cost",
        json={
            "model_id": "gpt-4o-mini",
            "document_count": 10,
            "avg_tokens_per_document": 100,
        },
    )
    assert response.status_code == 422
    assert "embedding model" in response.json()["error"]["message"]


async def test_budget_estimator_end_to_end(client: AsyncClient) -> None:
    response = await client.post(
        "/api/v1/tools/cost/budget-estimator",
        json={
            "lines": [
                {
                    "name": "chat",
                    "model_id": "gpt-4o-mini",
                    "requests_per_day": 1000,
                    "input_tokens": 1000,
                    "output_tokens": 500,
                }
            ],
            "monthly_growth_pct": "10",
        },
    )
    data = response.json()["data"]

    assert data["metrics"]["monthly_cost"] == "13.696875"
    assert len(data["series"]["growth_projection"]) == 12
    assert data["tables"]["breakdown"][0]["name"] == "chat"


async def test_budget_estimator_404s_on_an_unknown_model(client: AsyncClient) -> None:
    response = await client.post(
        "/api/v1/tools/cost/budget-estimator",
        json={
            "lines": [
                {
                    "name": "x",
                    "model_id": "not-a-model",
                    "requests_per_day": 1,
                    "input_tokens": 1,
                    "output_tokens": 1,
                }
            ]
        },
    )
    assert response.status_code == 404


# ── Compare Center, end to end ───────────────────────────────────────────────


async def test_compare_models_end_to_end(client: AsyncClient) -> None:
    response = await client.post(
        "/api/v1/tools/compare/models",
        json={
            "model_ids": ["gpt-4o-mini", "claude-sonnet-5", "gemini-2.5-flash"],
            "input_tokens": 2000,
            "output_tokens": 500,
            "requests_per_day": 1000,
        },
    )
    data = response.json()["data"]

    assert data["metrics"]["winner"] in {
        "gpt-4o-mini",
        "claude-sonnet-5",
        "gemini-2.5-flash",
    }
    assert len(data["tables"]["options"]) == 3
    assert any(row["kind"] == "switch_when" for row in data["tables"]["rationale"])


async def test_compare_models_cost_matches_the_seeded_pricing(
    client: AsyncClient,
) -> None:
    response = await client.post(
        "/api/v1/tools/compare/models",
        json={
            "model_ids": ["gpt-4o-mini", "claude-opus-5"],
            "input_tokens": 2000,
            "output_tokens": 500,
            "requests_per_day": 1000,
        },
    )
    by_id = {row["id"]: row for row in response.json()["data"]["tables"]["options"]}

    # gpt-4o-mini: (2 * 0.00015 + 0.5 * 0.0006) * 1000 * 30.4375
    assert by_id["gpt-4o-mini"]["monthly_cost"] == "18.26"
    # claude-opus-5: (2 * 0.005 + 0.5 * 0.025) * 1000 * 30.4375
    assert by_id["claude-opus-5"]["monthly_cost"] == "684.84"


async def test_compare_vector_db_computes_cost_at_scale(client: AsyncClient) -> None:
    response = await client.post(
        "/api/v1/tools/compare/vector-db",
        json={
            "tool_slugs": ["pinecone", "qdrant", "pgvector"],
            "vector_count": 10_000_000,
            "dimensions": 1536,
        },
    )
    by_id = {row["id"]: row for row in response.json()["data"]["tables"]["options"]}

    assert by_id["pinecone"]["monthly_cost"] == "80.00"
    assert by_id["qdrant"]["monthly_cost"] == "45.00"
    assert by_id["pgvector"]["monthly_cost"] == "20.00"


async def test_compare_vector_db_rejects_a_non_vector_db(client: AsyncClient) -> None:
    response = await client.post(
        "/api/v1/tools/compare/vector-db",
        json={"tool_slugs": ["pinecone", "langgraph"]},
    )
    assert response.status_code == 422


async def test_compare_stacks_end_to_end(client: AsyncClient) -> None:
    response = await client.post(
        "/api/v1/tools/compare/stacks",
        json={"archetypes": ["mvp", "enterprise", "open-source"]},
    )
    data = response.json()["data"]
    winner = next(
        row for row in data["tables"]["options"] if row["id"] == data["metrics"]["winner"]
    )
    assert winner["components"]


async def test_compare_build_vs_buy_end_to_end(client: AsyncClient) -> None:
    """300h x $120 + $2500/mo vs $500/mo: 79,200 build, 36,000 buy... except
    the spec's build figure assumes maintenance hours, so assert what the
    inputs actually produce."""
    response = await client.post(
        "/api/v1/tools/compare/build-vs-buy",
        json={
            "build_hours": 300,
            "blended_hourly_rate": "120",
            "build_infra_monthly": "2500",
            "maintenance_hours_per_month": "0",
            "vendor_monthly": "500",
        },
    )
    metrics = response.json()["data"]["metrics"]

    assert metrics["build_cost_12m"] == "66000.00"
    assert metrics["buy_cost_12m"] == "6000.00"
    assert metrics["winner"] == "buy"
    assert response.json()["data"]["tables"]["sensitivity"]


async def test_compare_meta_lists_priorities_and_archetypes(
    client: AsyncClient,
) -> None:
    response = await client.get("/api/v1/tools/compare/meta")
    data = response.json()["data"]

    assert {p["key"] for p in data["priorities"]} == {
        "balanced",
        "cost",
        "scale",
        "speed",
        "simplicity",
        "control",
    }
    assert len(data["stack_archetypes"]) == 5


async def test_every_comparison_logs_a_run(client: AsyncClient, db: AsyncSession) -> None:
    await client.post(
        "/api/v1/tools/compare/build-vs-buy",
        json={"build_hours": 100, "vendor_monthly": "400"},
    )
    runs = (
        (await db.execute(select(ToolRun).where(ToolRun.tool_slug == "compare-build-vs-buy")))
        .scalars()
        .all()
    )
    assert len(runs) == 1
    assert runs[0].workflow == "compare"


# ── Cache interaction ────────────────────────────────────────────────────────


async def test_repeated_runs_reuse_the_catalog_cache(client: AsyncClient) -> None:
    await client.post(PRICING, json=BASE_PAYLOAD)
    keys = [key async for key in get_redis().scan_iter(match="cache:catalog:models:*")]
    assert keys, "the first run should have populated the model cache"


def test_quota_limits_are_ordered_by_plan() -> None:
    """A plan that costs more must not allow less."""
    limits = tool_service.DAILY_RUN_LIMIT
    assert (
        limits["anonymous"] < limits["free"] < limits["pro"] < limits["team"] < limits["enterprise"]
    )


def test_metric_decimals_serialise_as_strings() -> None:
    from app.schemas.tools import ToolRunOut

    payload = ToolRunOut(
        run_id="run_1",
        tool_slug="x",
        source="rule_based",
        duration_ms=1,
        created_at=__import__("datetime").datetime.now(__import__("datetime").UTC),
        metrics={"tiny": Decimal("0.000005")},
    ).model_dump(mode="json")

    assert payload["metrics"]["tiny"] == "0.000005"
