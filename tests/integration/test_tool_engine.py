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
from app.services import feature_service, tool_service
from app.services.tool_service import RUN_METRIC
from tests.conftest import sign_in

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


async def test_a_run_is_attributed_to_the_account_that_made_it(
    client: AsyncClient, db: AsyncSession
) -> None:
    """One owner column, NOT NULL.

    This used to be two nullable columns held to exactly one by a check
    constraint, because a run could belong to an anonymous session instead. The
    tier is gone and so is the union.
    """
    from app.models.user import User

    response = await client.post(PRICING, json=BASE_PAYLOAD)

    run = await db.get(ToolRun, response.json()["data"]["run_id"])
    assert run is not None
    owner = await db.get(User, run.user_id)
    assert owner is not None
    assert owner.email == "ada@example.com"


async def test_a_run_needs_a_session(anon_client: AsyncClient) -> None:
    """The front door. Every tool route is behind it."""
    response = await anon_client.post(PRICING, json=BASE_PAYLOAD)

    assert response.status_code == 401
    assert response.json()["error"]["code"] == "UNAUTHENTICATED"


# ── Quota ────────────────────────────────────────────────────────────────────


async def _free_run_limit(db: AsyncSession) -> int:
    """The free allowance, read from `plan_quotas` rather than a constant.

    The limit moved into the table in M20 precisely so it could be changed
    without a deploy; a test that hardcoded 25 would have to be edited every
    time the number is tuned, which is the coupling the table removed.
    """
    from app.models.user import Plan, User

    probe = Identity(
        user=User(id="usr_probe", email="probe@example.com", plan=Plan.FREE), session_id=None
    )
    limit = await feature_service.limit_for(db, probe, RUN_METRIC)
    assert limit is not None, "the free tier must be capped"
    return limit


async def test_quota_returns_402_at_the_limit_with_real_numbers(
    client: AsyncClient,
    db: AsyncSession,
) -> None:
    """ "You hit your limit" with no figures is a dead end."""
    limit = await _free_run_limit(db)

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


async def test_below_the_limit_returns_200(client: AsyncClient, db: AsyncSession) -> None:
    for _ in range(await _free_run_limit(db) - 1):
        assert (await client.post(PRICING, json=BASE_PAYLOAD)).status_code == 200


async def test_a_blocked_run_is_not_logged(client: AsyncClient, db: AsyncSession) -> None:
    """The quota check happens before compute, so a rejected call costs nothing
    and leaves no row to skew the metrics."""
    for _ in range(await _free_run_limit(db)):
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
    assert quota["plan"] == "free"


async def test_quota_fails_open_when_redis_is_unavailable(db: AsyncSession) -> None:
    """A cache outage must not look like a billing failure to every user."""
    from app.core.redis import set_redis

    class _Broken:
        async def get(self, *_: object, **__: object) -> None:
            raise OSError("redis is down")

        async def incrby(self, *_: object, **__: object) -> None:
            raise OSError("redis is down")

    from app.models.user import Plan, User

    caller = Identity(
        user=User(id="usr_broken", email="broken@example.com", plan=Plan.FREE), session_id=None
    )

    set_redis(_Broken())  # type: ignore[arg-type]
    try:
        state = await tool_service.check_quota(db, caller)
        # And the enforcing path allows rather than refusing: an unreadable
        # counter reads as zero used, which is the whole point of failing open.
        await tool_service.consume_quota(db, caller)
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
    await client.post(PRICING, json=BASE_PAYLOAD)
    listed = await client.get("/api/v1/runs")

    assert listed.status_code == 200
    assert [run["tool_slug"] for run in listed.json()["data"]] == ["llm-pricing"]


async def test_a_reopened_run_keeps_its_provenance(client: AsyncClient) -> None:
    """Reopening a run must not quietly strip its verification dates.

    Provenance is attached by the engine, not by `compute`, so storing the
    inner output object dropped it — the chips were on screen for the live
    result and gone the moment the run was opened from history. Absent chips
    read as "this number has no source", which is a stronger and falser claim
    than anything the tool makes.
    """
    live = (await client.post(PRICING, json=BASE_PAYLOAD)).json()["data"]
    assert live["provenance"]["sources"], "the live run should have sources to lose"

    reopened = await client.get(f"/api/v1/runs/{live['run_id']}")
    assert reopened.status_code == 200

    output = reopened.json()["data"]["output"]
    assert output["provenance"] == live["provenance"]
    assert output["metrics"] == live["metrics"]
    assert output["run_id"] == live["run_id"]


async def test_a_run_stored_in_the_old_shape_still_reopens(
    client: AsyncClient, db: AsyncSession
) -> None:
    """Rows written before the stored blob became the full envelope.

    They hold a bare `ToolOutput` with no `run_id`, `source`, or provenance.
    Those runs are real history; the endpoint rebuilds the envelope from the
    row rather than 500ing on a missing key.
    """
    run_id = (await client.post(PRICING, json=BASE_PAYLOAD)).json()["data"]["run_id"]

    row = await db.get(ToolRun, run_id)
    assert row is not None
    row.output = {
        key: value
        for key, value in row.output.items()
        if key in {"metrics", "tables", "series", "artifacts", "warnings"}
    }
    await db.flush()

    reopened = await client.get(f"/api/v1/runs/{run_id}")
    assert reopened.status_code == 200

    output = reopened.json()["data"]["output"]
    assert output["run_id"] == run_id
    assert output["metrics"]["monthly_cost"]
    # Nothing was stored, so nothing is claimed.
    assert output["provenance"]["sources"] == []


async def test_a_run_is_not_readable_by_another_caller(
    client: AsyncClient, db: AsyncSession
) -> None:
    """A run id is not a capability. Sharing is M18's job."""
    run_id = (await client.post(PRICING, json=BASE_PAYLOAD)).json()["data"]["run_id"]
    assert (await client.get(f"/api/v1/runs/{run_id}")).status_code == 200

    # A second account on the same client — the header is overwritten, so this
    # is a different caller asking for the first one's work.
    await sign_in(client, db, email="stranger@example.com")
    assert (await client.get(f"/api/v1/runs/{run_id}")).status_code == 404


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


async def test_token_calculator_counts_an_openai_model_with_its_real_tokenizer(
    client: AsyncClient,
) -> None:
    """M16 replaced `ceil(chars / 4)` here. An OpenAI model now goes through
    tiktoken, and `method` on the response is what says so — the figure and
    the label have to move together, or the tool is lying about its own
    accuracy."""
    import tiktoken

    text = "hello world " * 500
    response = await client.post(
        "/api/v1/tools/cost/token-calculator",
        json={"text": text, "model_id": "gpt-4o-mini"},
    )
    metrics = response.json()["data"]["metrics"]

    assert metrics["method"] == "tokenizer"
    assert int(metrics["tokens"]) == len(tiktoken.get_encoding("o200k_base").encode(text))
    assert metrics["fits"] == "yes"
    assert response.json()["data"]["tables"]["context_fit"]


async def test_token_calculator_says_when_it_is_estimating(client: AsyncClient) -> None:
    """A model with no reachable tokenizer still returns a number — labelled."""
    response = await client.post(
        "/api/v1/tools/cost/token-calculator",
        json={"text": "hello world " * 500, "model_id": "gemini-3-flash"},
    )
    metrics = response.json()["data"]["metrics"]

    assert metrics["method"] == "heuristic"
    assert int(metrics["tokens"]) > 0
    assert any("heuristic" in w["message"] for w in response.json()["data"]["warnings"])


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


async def test_quota_limits_are_ordered_by_plan(db: AsyncSession) -> None:
    """A plan that costs more must not allow less.

    Asserted against the seeded rows rather than the constants: the table is
    what is enforced, and an operator raising the free tier past Pro's with an
    `UPDATE` is exactly the mistake this catches.
    """
    from app.models.billing import Metric, PlanQuota
    from app.models.user import Plan

    rows = (
        await db.execute(select(PlanQuota).where(PlanQuota.metric == Metric.TOOL_RUNS_PER_DAY))
    ).scalars()
    # `None` is unlimited, so it sorts above every real number.
    limits = {row.plan: row.limit_value for row in rows}
    ordered = [
        limits[Plan.FREE],
        limits[Plan.PRO],
        limits[Plan.TEAM],
        limits[Plan.ENTERPRISE],
    ]

    ranked = [float("inf") if value is None else value for value in ordered]
    assert ranked == sorted(ranked), f"a cheaper plan allows more: {ordered}"


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
