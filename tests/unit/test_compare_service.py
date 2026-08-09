"""Comparison scoring.

The two properties worth guarding: reweighting genuinely changes the answer
(otherwise `priority` is decoration), and `switch_when` is never empty
(otherwise the tool is a leaderboard).
"""

from __future__ import annotations

from datetime import UTC, datetime
from decimal import Decimal

import pytest

from app.data.compare_criteria import PRIORITIES, STACK_ARCHETYPES_BY_KEY
from app.schemas.catalog import ModelOut, ProvenanceOut, ToolOut
from app.services import compare_service

PROVENANCE = ProvenanceOut(
    last_verified_at=datetime(2026, 8, 9, tzinfo=UTC),
    age_days=0,
    variant="fresh",
    source_name="test source",
    source_url="https://example.com/pricing",
    source_kind="manual",
)


def model(
    model_id: str,
    *,
    input_per_1k: str,
    output_per_1k: str,
    context_window: int = 200_000,
    cached: str | None = None,
    thinking: bool = True,
    vision: bool = True,
    status: str = "active",
) -> ModelOut:
    return ModelOut(
        id=f"mdl_{model_id}",
        provider="test",
        model_id=model_id,
        display_name=model_id,
        family="chat",
        input_cost_per_1k=Decimal(input_per_1k),
        output_cost_per_1k=Decimal(output_per_1k),
        cached_input_cost_per_1k=Decimal(cached) if cached else None,
        context_window=context_window,
        max_output_tokens=32_000,
        capabilities={"thinking": thinking, "tools": True, "vision": vision},
        status=status,
        provenance=PROVENANCE,
    )


def tool(
    slug: str,
    *,
    cost_per_m: float,
    ops_burden: int,
    scale: int,
    lock_in: int,
    status: str = "recommended",
    self_hostable: bool = True,
    min_monthly: float = 0,
) -> ToolOut:
    return ToolOut(
        id=f"tool_{slug}",
        slug=slug,
        name=slug,
        category="vector-db",
        description="",
        status=status,
        status_reason="buried for testing" if status == "deprecated" else None,
        maturity_score=80,
        self_hostable=self_hostable,
        facts={
            "cost_per_m_vectors_month": cost_per_m,
            "min_monthly": min_monthly,
            "ops_burden": ops_burden,
            "filtering": 4,
            "hybrid_search": True,
            "scale_ceiling": scale,
            "ecosystem": 4,
            "lock_in": lock_in,
        },
        last_reviewed_at=datetime(2026, 6, 29, tzinfo=UTC),
    )


# ── compare-models ───────────────────────────────────────────────────────────

# Capability-identical on purpose, differing only in price and context window,
# so a reweighting test isolates the cost-versus-scale tradeoff rather than
# accidentally measuring which model has vision.
CHEAP = model(
    "cheap",
    input_per_1k="0.000050",
    output_per_1k="0.000400",
    context_window=64_000,
    cached="0.000005",
)
MID = model(
    "mid",
    input_per_1k="0.001000",
    output_per_1k="0.008000",
    context_window=272_000,
    cached="0.000100",
)
PREMIUM = model(
    "premium",
    input_per_1k="0.005000",
    output_per_1k="0.025000",
    context_window=1_000_000,
    cached="0.000500",
)


def test_cost_is_computed_from_the_usage_profile() -> None:
    result = compare_service.compare_models(
        models=[CHEAP, MID, PREMIUM],
        input_tokens=2000,
        output_tokens=500,
        requests_per_day=1000,
    )
    by_id = {row["id"]: row for row in result.tables["options"]}

    # cheap: (2*0.00005 + 0.5*0.0004) * 1000 * 30.4375 = 0.0003 * 30437.5
    assert by_id["cheap"]["monthly_cost"] == "9.13"
    # premium: (2*0.005 + 0.5*0.025) * 1000 * 30.4375 = 0.0225 * 30437.5
    assert by_id["premium"]["monthly_cost"] == "684.84"


def test_reweighting_changes_the_winner() -> None:
    """A comparison with one fixed weighting is an opinion, not a tool."""
    by_cost = compare_service.compare_models(
        models=[CHEAP, MID, PREMIUM],
        input_tokens=2000,
        output_tokens=500,
        requests_per_day=1000,
        priority="cost",
    )
    by_scale = compare_service.compare_models(
        models=[CHEAP, MID, PREMIUM],
        input_tokens=2000,
        output_tokens=500,
        requests_per_day=1000,
        priority="scale",
    )

    # The documented case the definition of done asks for: same three models,
    # same usage profile, different winner.
    assert by_cost.metrics["winner"] == "cheap"
    assert by_scale.metrics["winner"] == "premium"


@pytest.mark.parametrize("priority", PRIORITIES)
def test_every_priority_produces_a_ranking_and_a_switch_when(priority: str) -> None:
    result = compare_service.compare_models(
        models=[CHEAP, MID, PREMIUM],
        input_tokens=2000,
        output_tokens=500,
        requests_per_day=1000,
        priority=priority,  # type: ignore[arg-type]
    )
    switch = [r for r in result.tables["rationale"] if r["kind"] == "switch_when"]

    assert result.metrics["winner"] in {"cheap", "mid", "premium"}
    assert switch, "a comparison without switch_when is a leaderboard"
    assert len(result.tables["options"]) == 3
    assert [row["rank"] for row in result.tables["options"]] == [1, 2, 3]


def test_matrix_has_a_row_per_criterion_and_a_cell_per_option() -> None:
    result = compare_service.compare_models(
        models=[CHEAP, MID], input_tokens=1000, output_tokens=200, requests_per_day=100
    )
    for row in result.tables["matrix"]:
        assert 0 <= row["cheap"]["score"] <= 100
        assert 0 <= row["mid"]["score"] <= 100
        assert row["cheap"]["value"]  # a raw value, not just a score


def test_weights_sum_to_one() -> None:
    result = compare_service.compare_models(
        models=[CHEAP, MID], input_tokens=1000, output_tokens=200, requests_per_day=100
    )
    total = sum(row["weight"] for row in result.tables["matrix"])
    assert abs(total - 1.0) < 0.001


def test_a_near_tie_is_reported_as_low_confidence() -> None:
    """Two points of separation is a coin flip; saying so is more useful."""
    twin_a = model("twin-a", input_per_1k="0.001", output_per_1k="0.004")
    twin_b = model("twin-b", input_per_1k="0.001", output_per_1k="0.004")

    result = compare_service.compare_models(
        models=[twin_a, twin_b], input_tokens=1000, output_tokens=500, requests_per_day=100
    )
    assert result.metrics["confidence"] == "low"
    assert any("tie" in w.message for w in result.warnings)


def test_a_deprecated_model_is_penalised() -> None:
    retired = model(
        "retired", input_per_1k="0.000001", output_per_1k="0.000001", status="deprecated"
    )
    result = compare_service.compare_models(
        models=[retired, MID], input_tokens=1000, output_tokens=500, requests_per_day=1000
    )
    # It is by far the cheapest, and it still must not win.
    assert result.metrics["winner"] != "retired"


# ── compare-vector-db ────────────────────────────────────────────────────────

PINECONE = tool("pinecone", cost_per_m=8.0, ops_burden=1, scale=5, lock_in=4, self_hostable=False)
QDRANT = tool("qdrant", cost_per_m=4.5, ops_burden=3, scale=5, lock_in=1)
PGVECTOR = tool("pgvector", cost_per_m=2.0, ops_burden=2, scale=3, lock_in=1)


def test_vector_db_cost_scales_with_count_and_dimensions() -> None:
    """10M vectors at 1536 dims: 8.0 * 10 * 1.0 = $80/month for Pinecone."""
    result = compare_service.compare_vector_db(
        tools=[PINECONE, QDRANT, PGVECTOR], vector_count=10_000_000, dimensions=1536
    )
    by_id = {row["id"]: row for row in result.tables["options"]}

    assert by_id["pinecone"]["monthly_cost"] == "80.00"
    assert by_id["qdrant"]["monthly_cost"] == "45.00"
    assert by_id["pgvector"]["monthly_cost"] == "20.00"


def test_doubling_dimensions_doubles_the_cost() -> None:
    small = compare_service.compare_vector_db(
        tools=[PINECONE, QDRANT], vector_count=10_000_000, dimensions=1536
    )
    large = compare_service.compare_vector_db(
        tools=[PINECONE, QDRANT], vector_count=10_000_000, dimensions=3072
    )
    cheap_small = {r["id"]: r for r in small.tables["options"]}["pinecone"]
    cheap_large = {r["id"]: r for r in large.tables["options"]}["pinecone"]

    assert Decimal(cheap_large["monthly_cost"]) == Decimal(cheap_small["monthly_cost"]) * 2


def test_a_minimum_monthly_floor_is_respected() -> None:
    """A managed service with a $95 floor does not cost $2 at tiny scale."""
    floored = tool("elastic", cost_per_m=9.0, ops_burden=4, scale=5, lock_in=3, min_monthly=95)
    result = compare_service.compare_vector_db(
        tools=[floored, PGVECTOR], vector_count=100_000, dimensions=1536
    )
    by_id = {row["id"]: row for row in result.tables["options"]}
    assert by_id["elastic"]["monthly_cost"] == "95.00"


def test_simplicity_priority_favours_the_managed_option() -> None:
    result = compare_service.compare_vector_db(
        tools=[PINECONE, QDRANT, PGVECTOR],
        vector_count=10_000_000,
        dimensions=1536,
        priority="simplicity",
    )
    assert result.metrics["winner"] == "pinecone"


def test_control_priority_favours_a_self_hostable_option() -> None:
    result = compare_service.compare_vector_db(
        tools=[PINECONE, QDRANT, PGVECTOR],
        vector_count=10_000_000,
        dimensions=1536,
        priority="control",
    )
    winner = next(row for row in result.tables["options"] if row["id"] == result.metrics["winner"])
    assert winner["self_hostable"] is True


def test_a_buried_tool_warns_with_its_reason() -> None:
    buried = tool("chroma", cost_per_m=0.0, ops_burden=2, scale=2, lock_in=1, status="deprecated")
    result = compare_service.compare_vector_db(
        tools=[buried, QDRANT], vector_count=1_000_000, dimensions=1536
    )
    assert any("buried for testing" in w.message for w in result.warnings)
    assert result.metrics["winner"] == "qdrant"


def test_small_corpus_suggests_pgvector_in_switch_when() -> None:
    result = compare_service.compare_vector_db(
        tools=[PINECONE, QDRANT], vector_count=1_000_000, dimensions=1536
    )
    switch = " ".join(
        row["text"] for row in result.tables["rationale"] if row["kind"] == "switch_when"
    )
    assert "pgvector" in switch


# ── compare-stacks ───────────────────────────────────────────────────────────


def test_stack_tco_includes_engineering_time() -> None:
    """A stack that saves $600/month and costs three engineer-weeks is not cheaper."""
    mvp = STACK_ARCHETYPES_BY_KEY["mvp"]
    oss = STACK_ARCHETYPES_BY_KEY["open-source"]

    result = compare_service.compare_stacks(
        archetypes=[mvp, oss],
        monthly_model_spend=Decimal(500),
        blended_hourly_rate=Decimal(120),
    )
    by_id = {row["id"]: row for row in result.tables["options"]}

    # mvp: setup 3d*8h*120 = 2,880; maintenance 1*2*8*120*12 = 23,040;
    #      infra 45*12 = 540; models 500*12 = 6,000  => 32,460
    assert by_id["mvp"]["tco_12_month"] == "32460.00"
    # open-source: 21*8*120 = 20,160; 5*2*8*120*12 = 115,200;
    #      620*12 = 7,440; 6,000 => 148,800
    assert by_id["open-source"]["tco_12_month"] == "148800.00"


def test_stack_options_carry_their_components_for_the_handoff() -> None:
    """The winner has to convert into a Stack Architect project."""
    result = compare_service.compare_stacks(
        archetypes=[
            STACK_ARCHETYPES_BY_KEY["mvp"],
            STACK_ARCHETYPES_BY_KEY["enterprise"],
        ]
    )
    winner_id = result.metrics["winner"]
    winner = next(row for row in result.tables["options"] if row["id"] == winner_id)

    assert winner["components"]
    assert all(isinstance(component, str) for component in winner["components"])


def test_speed_priority_favours_the_fastest_to_deploy() -> None:
    result = compare_service.compare_stacks(
        archetypes=[
            STACK_ARCHETYPES_BY_KEY["mvp"],
            STACK_ARCHETYPES_BY_KEY["self-hosted"],
        ],
        priority="speed",
    )
    assert result.metrics["winner"] == "mvp"


# ── compare-build-vs-buy ─────────────────────────────────────────────────────


def test_build_vs_buy_headline_figures() -> None:
    """300h x $120 upfront + $2500/mo running vs $500/mo vendor."""
    result = compare_service.compare_build_vs_buy(
        build_hours=300,
        blended_hourly_rate=Decimal(120),
        build_infra_monthly=Decimal(2500),
        maintenance_hours_per_month=Decimal(0),
        vendor_monthly=Decimal(500),
    )

    # build: 36,000 upfront + 2,500 * 12 = 66,000
    assert result.metrics["build_cost_12m"] == Decimal("66000.00")
    # buy: 500 * 12
    assert result.metrics["buy_cost_12m"] == Decimal("6000.00")
    assert result.metrics["winner"] == "buy"


def test_build_wins_over_a_long_horizon_against_an_expensive_vendor() -> None:
    result = compare_service.compare_build_vs_buy(
        build_hours=300,
        blended_hourly_rate=Decimal(120),
        build_infra_monthly=Decimal(200),
        maintenance_hours_per_month=Decimal(4),
        vendor_monthly=Decimal(6000),
    )
    # build monthly = 200 + 4*120 = 680; buy = 6000. Gap 5,320/month against
    # 36,000 upfront => break-even in month 7.
    assert result.metrics["break_even_month"] == 7
    assert result.metrics["winner"] == "build"


def test_break_even_is_never_when_the_vendor_is_always_cheaper() -> None:
    result = compare_service.compare_build_vs_buy(
        build_hours=2000,
        blended_hourly_rate=Decimal(150),
        build_infra_monthly=Decimal(3000),
        maintenance_hours_per_month=Decimal(20),
        vendor_monthly=Decimal(200),
    )
    assert result.metrics["break_even_month"] == "never"


def test_sensitivity_table_covers_the_argued_range() -> None:
    """A single break-even number invites distrust; the table is what survives
    a board meeting."""
    result = compare_service.compare_build_vs_buy(
        build_hours=300,
        blended_hourly_rate=Decimal(120),
        build_infra_monthly=Decimal(200),
        maintenance_hours_per_month=Decimal(4),
        vendor_monthly=Decimal(1500),
    )
    sensitivity = result.tables["sensitivity"]

    assert len(sensitivity) == 15  # 5 hour factors x 3 rates
    assert {row["winner"] for row in sensitivity} <= {"build", "buy"}
    for row in sensitivity:
        assert Decimal(row["build_36m"]) > 0
        assert Decimal(row["buy_36m"]) > 0


def test_cumulative_projection_spans_36_months() -> None:
    result = compare_service.compare_build_vs_buy(
        build_hours=100,
        blended_hourly_rate=Decimal(100),
        build_infra_monthly=Decimal(100),
        maintenance_hours_per_month=Decimal(1),
        vendor_monthly=Decimal(400),
    )
    projection = result.series["cumulative_cost"]

    assert len(projection) == 36
    assert projection[0]["month"] == 1
    assert projection[-1]["month"] == 36


def test_every_comparison_returns_rationale_and_tradeoffs() -> None:
    result = compare_service.compare_build_vs_buy(
        build_hours=300,
        blended_hourly_rate=Decimal(120),
        build_infra_monthly=Decimal(2500),
        maintenance_hours_per_month=Decimal(0),
        vendor_monthly=Decimal(500),
    )
    kinds = {row["kind"] for row in result.tables["rationale"]}
    assert kinds == {"why", "tradeoff", "switch_when"}


# ── criteria as data ─────────────────────────────────────────────────────────


def test_adding_a_criterion_changes_the_output_with_no_code_change() -> None:
    """`PRD.md` §22: new comparison behaviour is a data change."""
    from app.data import compare_criteria
    from app.data.compare_criteria import Criterion

    original = compare_criteria.CRITERIA_BY_TOOL["compare-models"]
    before = compare_service.compare_models(
        models=[CHEAP, MID], input_tokens=1000, output_tokens=200, requests_per_day=100
    )

    extra = Criterion(
        "invented",
        "Invented criterion",
        "Added by a test.",
        "computed",
        5.0,
        dict.fromkeys(compare_criteria.PRIORITIES, 1.0),
    )
    compare_criteria.CRITERIA_BY_TOOL["compare-models"] = (*original, extra)
    try:
        after = compare_service.compare_models(
            models=[CHEAP, MID], input_tokens=1000, output_tokens=200, requests_per_day=100
        )
    finally:
        compare_criteria.CRITERIA_BY_TOOL["compare-models"] = original

    assert len(after.tables["matrix"]) == len(before.tables["matrix"]) + 1
    assert any(row["criterion"] == "invented" for row in after.tables["matrix"])
