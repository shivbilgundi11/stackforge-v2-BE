"""Catalog reads, compatibility, provenance, and flagging.

Every assertion here is a real value from the seed. A test that asserts `200`
and a non-empty list passes when the seeder silently loads nothing, which is
the failure this module can least afford.
"""

from __future__ import annotations

from datetime import timedelta
from decimal import Decimal

import pytest
from httpx import AsyncClient
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.database import utcnow
from app.models.catalog import ModelPricing, PricedEntity, Tool, ToolStatus
from app.services import catalog_service, provenance_service

pytestmark = pytest.mark.usefixtures("seeded_catalog")


# ── Model pricing ────────────────────────────────────────────────────────────


async def test_model_list_returns_real_prices(client: AsyncClient) -> None:
    response = await client.get("/api/v1/catalog/models", params={"family": "chat"})
    assert response.status_code == 200

    models = {item["model_id"]: item for item in response.json()["data"]}

    # Per 1M published → per 1k stored. $0.15/1M input becomes $0.00015/1k.
    assert models["gpt-4o-mini"]["input_cost_per_1k"] == "0.000150"
    assert models["gpt-4o-mini"]["output_cost_per_1k"] == "0.000600"
    assert models["claude-opus-5"]["input_cost_per_1k"] == "0.005000"
    assert models["claude-opus-5"]["output_cost_per_1k"] == "0.025000"
    assert models["claude-opus-5"]["cached_input_cost_per_1k"] == "0.000500"


async def test_sub_cent_prices_survive_the_round_trip(client: AsyncClient) -> None:
    """GPT-5 nano input is $0.00005 per 1k. A float loses this; a string does not."""
    response = await client.get("/api/v1/catalog/models/gpt-5-nano")
    model = response.json()["data"]

    assert model["input_cost_per_1k"] == "0.000050"
    assert Decimal(model["input_cost_per_1k"]) > 0
    assert Decimal(model["cached_input_cost_per_1k"]) == Decimal("0.000005")


async def test_deprecated_models_are_excluded_by_default(client: AsyncClient) -> None:
    active = await client.get("/api/v1/catalog/models")
    ids = {item["model_id"] for item in active.json()["data"]}
    assert "text-embedding-ada-002" not in ids

    everything = await client.get("/api/v1/catalog/models", params={"include_all_statuses": True})
    all_ids = {item["model_id"] for item in everything.json()["data"]}
    assert "text-embedding-ada-002" in all_ids


async def test_embedding_models_carry_dimensions(client: AsyncClient) -> None:
    """`vectordb-estimate` downstream needs this — storage cost scales with it."""
    response = await client.get("/api/v1/catalog/models", params={"family": "embedding"})
    by_id = {item["model_id"]: item for item in response.json()["data"]}

    assert by_id["text-embedding-3-small"]["dimensions"] == 1536
    assert by_id["text-embedding-3-large"]["dimensions"] == 3072
    assert by_id["text-embedding-3-small"]["output_cost_per_1k"] is None


async def test_provider_filter(client: AsyncClient) -> None:
    response = await client.get("/api/v1/catalog/models", params={"provider": "anthropic"})
    providers = {item["provider"] for item in response.json()["data"]}
    assert providers == {"anthropic"}


async def test_unknown_model_is_404(client: AsyncClient) -> None:
    response = await client.get("/api/v1/catalog/models/gpt-99-imaginary")
    assert response.status_code == 404
    assert response.json()["error"]["code"] == "NOT_FOUND"


# ── Provenance ───────────────────────────────────────────────────────────────


async def test_every_model_has_a_source_and_a_verification_date(
    client: AsyncClient,
) -> None:
    response = await client.get("/api/v1/catalog/models", params={"include_all_statuses": True})
    for item in response.json()["data"]:
        provenance = item["provenance"]
        assert provenance["last_verified_at"]
        assert provenance["source_url"].startswith("http")
        assert provenance["variant"] in {"fresh", "aging", "stale"}


@pytest.mark.parametrize(
    ("age_days", "expected"),
    [(0, "fresh"), (7, "fresh"), (8, "aging"), (30, "aging"), (31, "stale"), (400, "stale")],
)
def test_provenance_variant_boundaries(age_days: int, expected: str) -> None:
    """7 and 30 are the boundaries, and they are inclusive on the younger side."""
    assert provenance_service.variant_for(age_days) == expected


async def test_provenance_age_is_computed_from_the_row(db: AsyncSession) -> None:
    model = await catalog_service.get_model(db, "claude-opus-5")
    expected = (utcnow() - model.provenance.last_verified_at).days
    assert model.provenance.age_days == expected


# ── GPUs ─────────────────────────────────────────────────────────────────────


async def test_gpu_list_returns_real_hourly_and_monthly(client: AsyncClient) -> None:
    response = await client.get("/api/v1/catalog/gpus", params={"provider": "lambda"})
    by_name = {item["instance_name"]: item for item in response.json()["data"]}

    single = by_name["gpu_1x_h100_pcie"]
    assert single["hourly_cost_usd"] == "3.290000"
    # 3.29 * 730 hours
    assert single["monthly_cost_usd"] == "2401.70"
    assert single["vram_total_gb"] == 80


async def test_min_vram_filters_on_node_total_not_per_card(client: AsyncClient) -> None:
    """An 8x80GB node satisfies 200GB even though no single card does."""
    response = await client.get("/api/v1/catalog/gpus", params={"min_vram": 200})
    names = {item["instance_name"] for item in response.json()["data"]}

    assert "gpu_8x_h100_sxm5" in names  # 8 * 80 = 640
    assert "gpu_1x_h100_pcie" not in names  # 1 * 80 = 80


async def test_spot_filter(client: AsyncClient) -> None:
    response = await client.get("/api/v1/catalog/gpus", params={"spot": True})
    items = response.json()["data"]
    assert items and all(item["spot"] for item in items)


# ── Tools and the Graveyard ──────────────────────────────────────────────────


async def test_tool_lookup_by_slug(client: AsyncClient) -> None:
    response = await client.get("/api/v1/catalog/tools/qdrant")
    tool = response.json()["data"]

    assert tool["name"] == "Qdrant"
    assert tool["category"] == "vector-db"
    assert tool["self_hostable"] is True
    assert "rag" in tool["use_cases"]


async def test_tag_filter_requires_every_tag(client: AsyncClient) -> None:
    """Adding a tag must narrow the result, not widen it."""
    one = await client.get("/api/v1/catalog/tools", params={"tags": "open-source"})
    two = await client.get("/api/v1/catalog/tools", params={"tags": "open-source,self-hostable"})

    single = {item["slug"] for item in one.json()["data"]}
    both = {item["slug"] for item in two.json()["data"]}

    assert both < single
    assert "qdrant" in both
    assert "pinecone" not in both  # open-source? no


async def test_graveyard_returns_every_buried_tool_with_a_reason(
    client: AsyncClient,
) -> None:
    response = await client.get("/api/v1/catalog/graveyard")
    entries = {item["slug"]: item for item in response.json()["data"]}

    assert "autogpt" in entries
    assert "babyagi" in entries
    assert "gptcache" in entries

    for entry in entries.values():
        assert entry["status_reason"], f"{entry['slug']} is buried with no reason"
        assert entry["status"] in {"deprecated", "not_for_production"}

    # Alternatives resolve to real catalog rows, not bare slugs.
    autogpt_alternatives = {item["slug"] for item in entries["autogpt"]["alternative_tools"]}
    assert "langgraph" in autogpt_alternatives


async def test_changing_a_status_changes_the_graveyard_with_no_deploy(
    client: AsyncClient, db: AsyncSession
) -> None:
    """`PRD.md` §22: catalog updates must need no deployment."""
    before = await client.get("/api/v1/catalog/graveyard")
    assert "chroma" not in {item["slug"] for item in before.json()["data"]}

    tool = (await db.execute(select(Tool).where(Tool.slug == "chroma"))).scalar_one()
    tool.status = ToolStatus.DEPRECATED
    tool.status_reason = "Marked deprecated during this test."
    await db.flush()
    await catalog_service.invalidate()

    after = await client.get("/api/v1/catalog/graveyard")
    entries = {item["slug"]: item for item in after.json()["data"]}
    assert "chroma" in entries
    assert entries["chroma"]["status_reason"] == "Marked deprecated during this test."


# ── Compatibility ────────────────────────────────────────────────────────────


async def test_compatibility_is_order_independent(client: AsyncClient) -> None:
    forward = await client.get(
        "/api/v1/catalog/compatibility", params={"tools": "pinecone,langgraph"}
    )
    backward = await client.get(
        "/api/v1/catalog/compatibility", params={"tools": "langgraph,pinecone"}
    )

    assert forward.json()["data"] == backward.json()["data"]
    assert forward.json()["data"]["pairs"][0]["score"] > 0


async def test_compatibility_overall_is_the_weakest_pair(client: AsyncClient) -> None:
    """A stack is only as compatible as its worst pairing — never the mean."""
    response = await client.get(
        "/api/v1/catalog/compatibility",
        params={"tools": "langgraph,pinecone,anthropic-api"},
    )
    data = response.json()["data"]

    scores = [pair["score"] for pair in data["pairs"]]
    assert data["overall"] == min(scores)
    assert data["weakest_pair"]["score"] == min(scores)


async def test_editorial_override_reaches_the_response(client: AsyncClient) -> None:
    response = await client.get(
        "/api/v1/catalog/compatibility", params={"tools": "langchain,llamaindex"}
    )
    pair = response.json()["data"]["pairs"][0]

    assert pair["score"] == 52
    assert "two chunking implementations" in pair["notes"]
    assert len(pair["warnings"]) == 1
    assert "Overlapping responsibilities" in pair["warnings"][0]


async def test_incompatible_pair_scores_low_and_warns(client: AsyncClient) -> None:
    response = await client.get(
        "/api/v1/catalog/compatibility", params={"tools": "cloudflare-workers,vllm"}
    )
    data = response.json()["data"]

    assert data["overall"] == 15
    assert any("Workers cannot host GPU inference" in w for w in data["warnings"])


async def test_pairs_carry_all_nine_dimensions(client: AsyncClient) -> None:
    from app.data.compatibility_seed import DIMENSIONS

    response = await client.get(
        "/api/v1/catalog/compatibility", params={"tools": "qdrant,langgraph"}
    )
    dimensions = response.json()["data"]["pairs"][0]["dimensions"]

    assert set(dimensions) == set(DIMENSIONS)
    assert all(0 <= value <= 100 for value in dimensions.values())


async def test_unscored_pair_is_reported_not_assumed(client: AsyncClient) -> None:
    """Silently treating an unknown pair as compatible would let the Stack
    Architect recommend a combination nobody has looked at."""
    response = await client.get(
        "/api/v1/catalog/compatibility", params={"tools": "redis,memcached"}
    )
    data = response.json()["data"]

    assert data["missing_pairs"] == [["memcached", "redis"]]
    assert data["pairs"] == []


async def test_compatibility_needs_at_least_two_tools(client: AsyncClient) -> None:
    response = await client.get("/api/v1/catalog/compatibility", params={"tools": "qdrant"})
    assert response.status_code == 422


async def test_compatibility_caps_the_tool_count(client: AsyncClient) -> None:
    too_many = ",".join(f"tool-{i}" for i in range(13))
    response = await client.get("/api/v1/catalog/compatibility", params={"tools": too_many})
    assert response.status_code == 422


# ── Caching ──────────────────────────────────────────────────────────────────


async def test_second_identical_read_is_served_from_cache(db: AsyncSession) -> None:
    from app.core.redis import get_redis

    await catalog_service.invalidate()
    await catalog_service.list_models(db, family="chat")

    keys = [key async for key in get_redis().scan_iter(match="cache:catalog:models:*")]
    assert keys, "the first read should have populated the cache"

    # Second read must not touch the database. Closing the session makes any
    # query raise, so a cache miss here fails loudly rather than passing.
    await db.close()
    again = await catalog_service.list_models(db, family="chat")
    assert any(item.model_id == "gpt-4o-mini" for item in again)


async def test_a_write_invalidates_the_cache(db: AsyncSession) -> None:
    from app.core.redis import get_redis

    await catalog_service.list_models(db, family="chat")
    assert [key async for key in get_redis().scan_iter(match="cache:catalog:*")]

    await catalog_service.invalidate()
    assert not [key async for key in get_redis().scan_iter(match="cache:catalog:*")]


# ── Pricing history and drift ────────────────────────────────────────────────


async def test_recording_a_change_computes_the_percentage(db: AsyncSession) -> None:
    model = (
        await db.execute(select(ModelPricing).where(ModelPricing.model_id == "gpt-4o-mini"))
    ).scalar_one()

    await provenance_service.record_change(
        db,
        entity_type=PricedEntity.MODEL,
        entity_id=model.id,
        field="input_cost_per_1k",
        old_value=Decimal("0.000150"),
        new_value=Decimal("0.0001725"),  # +15%
        source_id=model.source_id,
    )
    await db.flush()

    history = await provenance_service.history_for(db, entity_id=model.id)
    assert len(history) == 1
    assert history[0].pct_change == Decimal("15.0000")
    assert history[0].applied is False


async def test_recording_a_change_does_not_move_the_price(db: AsyncSession) -> None:
    """The whole design: detect, alert, never auto-apply."""
    model = (
        await db.execute(select(ModelPricing).where(ModelPricing.model_id == "gpt-4o-mini"))
    ).scalar_one()
    before = model.input_cost_per_1k

    await provenance_service.record_change(
        db,
        entity_type=PricedEntity.MODEL,
        entity_id=model.id,
        field="input_cost_per_1k",
        old_value=before,
        new_value=Decimal("0.999"),
        source_id=model.source_id,
    )
    await db.flush()
    await db.refresh(model)

    assert model.input_cost_per_1k == before


async def test_drift_detection_respects_the_threshold(db: AsyncSession) -> None:
    model = (
        await db.execute(select(ModelPricing).where(ModelPricing.model_id == "gpt-5-mini"))
    ).scalar_one()

    await provenance_service.record_change(
        db,
        entity_type=PricedEntity.MODEL,
        entity_id=model.id,
        field="input_cost_per_1k",
        old_value=Decimal("0.000250"),
        new_value=Decimal("0.000255"),  # +2%
        source_id=model.source_id,
    )
    await provenance_service.record_change(
        db,
        entity_type=PricedEntity.MODEL,
        entity_id=model.id,
        field="output_cost_per_1k",
        old_value=Decimal("0.002000"),
        new_value=Decimal("0.002300"),  # +15%
        source_id=model.source_id,
    )
    await db.flush()

    drift = await provenance_service.detect_drift(db, threshold_pct=Decimal(5))
    fields = {entry.field for entry in drift if entry.entity_id == model.id}

    assert fields == {"output_cost_per_1k"}
    entry = next(e for e in drift if e.entity_id == model.id)
    assert entry.pct_change == Decimal("15.0000")
    assert entry.label == "openai / GPT-5 Mini"


def test_pct_change_handles_a_price_moving_off_zero() -> None:
    assert provenance_service.pct_change(Decimal(0), Decimal(0)) == 0
    assert provenance_service.pct_change(Decimal(0), Decimal("0.001")) == 100


# ── Flagging ─────────────────────────────────────────────────────────────────


async def test_anyone_can_flag_a_stale_price(client: AsyncClient) -> None:
    response = await client.post(
        "/api/v1/catalog/flag",
        json={
            "entity_type": "model",
            "entity_id": "mdl_whatever",
            "field": "input_cost_per_1k",
            "suggested_value": "0.000100",
            "note": "OpenAI dropped this last week.",
            "source_url": "https://developers.openai.com/api/docs/pricing",
        },
    )
    assert response.status_code == 200
    flag = response.json()["data"]
    assert flag["status"] == "open"
    assert flag["id"].startswith("flag_")


async def test_flag_rejects_an_unknown_entity_type(client: AsyncClient) -> None:
    response = await client.post(
        "/api/v1/catalog/flag",
        json={"entity_type": "spaceship", "entity_id": "x"},
    )
    assert response.status_code == 422


# ── Stats ────────────────────────────────────────────────────────────────────


async def test_stats_reports_real_counts(client: AsyncClient) -> None:
    response = await client.get("/api/v1/catalog/stats")
    stats = response.json()["data"]

    assert stats["models"] >= 50
    assert stats["tools"] >= 70
    assert stats["gpus"] >= 30
    assert stats["compatibility_pairs"] >= 200
    assert stats["oldest_verification"] is not None


# ── Seeding ──────────────────────────────────────────────────────────────────


async def test_seeding_twice_does_not_duplicate(db: AsyncSession) -> None:
    from app.services.seed_service import seed_all

    before = await catalog_service.count_rows(db)
    report = await seed_all(db)
    after = await catalog_service.count_rows(db)

    assert report.total_inserted == 0
    assert before == after


async def test_seeding_does_not_overwrite_an_editorial_correction(
    db: AsyncSession,
) -> None:
    """An editor's fix must survive the next deploy, or the review loop is theatre."""
    from app.services.seed_service import seed_all

    model = (
        await db.execute(select(ModelPricing).where(ModelPricing.model_id == "gpt-4o-mini"))
    ).scalar_one()
    model.input_cost_per_1k = Decimal("0.000123")
    model.last_verified_at = utcnow()
    await db.flush()

    await seed_all(db)
    await db.refresh(model)

    assert model.input_cost_per_1k == Decimal("0.000123")


async def test_refresh_overwrites_when_asked(db: AsyncSession) -> None:
    from app.services.seed_service import seed_all

    model = (
        await db.execute(select(ModelPricing).where(ModelPricing.model_id == "gpt-4o-mini"))
    ).scalar_one()
    model.input_cost_per_1k = Decimal("0.000123")
    await db.flush()

    await seed_all(db, refresh=True)
    await db.refresh(model)

    assert model.input_cost_per_1k == Decimal("0.000150")


async def test_compatibility_pairs_respect_the_ordering_constraint(
    db: AsyncSession,
) -> None:
    from app.models.catalog import Compatibility

    rows = (await db.execute(select(Compatibility).limit(500))).scalars().all()
    assert rows
    for row in rows:
        assert row.tool_a_slug < row.tool_b_slug


# ── Verification job ─────────────────────────────────────────────────────────


async def test_verification_records_drift_without_mutating_the_price(
    db: AsyncSession,
) -> None:
    from app.models.catalog import DataSource
    from app.workers import pricing as worker

    source = (
        await db.execute(select(DataSource).where(DataSource.slug == "openai-pricing"))
    ).scalar_one()
    model = (
        await db.execute(select(ModelPricing).where(ModelPricing.model_id == "gpt-4o-mini"))
    ).scalar_one()
    original = model.input_cost_per_1k

    async def fake_fetch(_source: DataSource) -> dict[tuple[str, str], dict[str, Decimal]]:
        return {("openai", "gpt-4o-mini"): {"input_cost_per_1k": Decimal("0.0001725")}}

    worker.clear_fetchers()
    worker.register_fetcher("openai-pricing", fake_fetch)
    try:
        result = await worker.verify_all(db, source_slug="openai-pricing")
    finally:
        worker.clear_fetchers()

    await db.flush()
    await db.refresh(model)

    assert result.changes_detected == 1
    assert model.input_cost_per_1k == original, "the job must never move a price"

    history = await provenance_service.history_for(db, entity_id=model.id)
    assert history[0].pct_change == Decimal("15.0000")
    assert history[0].applied is False
    assert source.failure_count == 0


async def test_a_failing_source_keeps_the_last_good_value_and_counts_up(
    db: AsyncSession,
) -> None:
    from app.models.catalog import DataSource
    from app.workers import pricing as worker

    source = (
        await db.execute(select(DataSource).where(DataSource.slug == "openai-pricing"))
    ).scalar_one()
    model = (
        await db.execute(select(ModelPricing).where(ModelPricing.model_id == "gpt-4o-mini"))
    ).scalar_one()
    original = model.input_cost_per_1k

    async def broken(_source: DataSource) -> dict[tuple[str, str], dict[str, Decimal]]:
        raise ConnectionError("pricing page moved")

    worker.clear_fetchers()
    worker.register_fetcher("openai-pricing", broken)
    try:
        for _ in range(3):
            result = await worker.verify_all(db, source_slug="openai-pricing")
    finally:
        worker.clear_fetchers()

    await db.refresh(model)
    assert model.input_cost_per_1k == original
    assert source.failure_count == 3
    assert result.sources_failed == 1
    assert any("3 consecutive" in alert for alert in result.alerts)


async def test_an_unchanged_price_refreshes_the_verification_date(
    db: AsyncSession,
) -> None:
    from app.models.catalog import DataSource
    from app.workers import pricing as worker

    model = (
        await db.execute(select(ModelPricing).where(ModelPricing.model_id == "gpt-4o-mini"))
    ).scalar_one()
    model.last_verified_at = utcnow() - timedelta(days=90)
    await db.flush()

    async def unchanged(_source: DataSource) -> dict[tuple[str, str], dict[str, Decimal]]:
        return {("openai", "gpt-4o-mini"): {"input_cost_per_1k": model.input_cost_per_1k}}

    worker.clear_fetchers()
    worker.register_fetcher("openai-pricing", unchanged)
    try:
        result = await worker.verify_all(db, source_slug="openai-pricing")
    finally:
        worker.clear_fetchers()

    await db.flush()
    await db.refresh(model)
    assert result.changes_detected == 0
    assert (utcnow() - model.last_verified_at).days == 0


async def test_a_source_with_no_fetcher_is_skipped_not_failed(db: AsyncSession) -> None:
    """ "Nobody wrote the parser" and "the page is down" are different facts."""
    from app.workers import pricing as worker

    worker.clear_fetchers()
    result = await worker.verify_all(db)

    assert result.sources_failed == 0
    assert result.sources_skipped >= 18


async def test_accepting_a_change_is_what_moves_the_price(db: AsyncSession) -> None:
    from app.workers import pricing as worker

    model = (
        await db.execute(select(ModelPricing).where(ModelPricing.model_id == "gpt-4o-mini"))
    ).scalar_one()

    history = await provenance_service.record_change(
        db,
        entity_type=PricedEntity.MODEL,
        entity_id=model.id,
        field="input_cost_per_1k",
        old_value=model.input_cost_per_1k,
        new_value=Decimal("0.000200"),
        source_id=model.source_id,
    )
    await db.flush()

    await worker.apply_change(db, history.id)
    await db.flush()
    await db.refresh(model)

    assert model.input_cost_per_1k == Decimal("0.000200")
    assert history.applied is True


async def test_a_renamed_model_id_is_reported_not_silently_orphaned(
    db: AsyncSession,
) -> None:
    """Renaming an id inserts the new row and orphans the old one.

    The orphan then ages into looking stale while nothing updates it, because
    no seed entry claims it. Reported, never deleted — a row can be orphaned
    by a real retirement or by a typo in the seed, and deleting priced history
    on that ambiguity is not a trade worth making automatically.
    """
    from app.core.database import new_id
    from app.models.catalog import DataSource
    from app.services.seed_service import seed_all

    source = (
        await db.execute(select(DataSource).where(DataSource.slug == "openai-pricing"))
    ).scalar_one()
    db.add(
        ModelPricing(
            id=new_id("mdl"),
            provider="openai",
            model_id="gpt-4o-mini-old-id",
            display_name="Renamed away",
            family="chat",
            input_cost_per_1k=Decimal("0.000150"),
            capabilities={},
            source_id=source.id,
            last_verified_at=utcnow(),
        )
    )
    await db.flush()

    report = await seed_all(db)

    assert "model_pricing: openai/gpt-4o-mini-old-id" in report.unmanaged
    # And it is still there afterwards.
    survivor = (
        await db.execute(select(ModelPricing).where(ModelPricing.model_id == "gpt-4o-mini-old-id"))
    ).scalar_one_or_none()
    assert survivor is not None


async def test_a_clean_catalog_reports_no_unmanaged_rows(db: AsyncSession) -> None:
    from app.services.seed_service import seed_all

    report = await seed_all(db)
    assert report.unmanaged == []


async def test_a_refresh_records_every_price_it_changes(db: AsyncSession) -> None:
    """The seed file is the only path by which a price changes, so it has to
    leave the same audit trail as any other change to one."""
    from app.services.seed_service import seed_all

    model = (
        await db.execute(select(ModelPricing).where(ModelPricing.model_id == "gpt-4o-mini"))
    ).scalar_one()
    model.input_cost_per_1k = Decimal("0.000300")  # pretend the seed moved
    await db.flush()

    report = await seed_all(db, refresh=True)
    await db.flush()
    await db.refresh(model)

    assert report.price_changes >= 1
    assert model.input_cost_per_1k == Decimal("0.000150")  # back to the seed

    history = await provenance_service.history_for(db, entity_id=model.id)
    entry = next(h for h in history if h.field == "input_cost_per_1k")
    assert entry.old_value == Decimal("0.000300")
    assert entry.new_value == Decimal("0.000150")
    assert entry.pct_change == Decimal("-50.0000")
    # Already applied, unlike a drift row awaiting review.
    assert entry.applied is True


async def test_a_refresh_that_changes_nothing_records_nothing(db: AsyncSession) -> None:
    """Re-running a deploy must not fill the history with no-op rows."""
    from app.services.seed_service import seed_all

    report = await seed_all(db, refresh=True)
    assert report.price_changes == 0
