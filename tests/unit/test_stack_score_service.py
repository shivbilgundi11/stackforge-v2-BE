"""The Stack Score, asserted by hand.

The weighted total is worked out independently of the implementation below —
a test that recomputes the function's own formula proves it is stable, not
that it is right, and this is the number the flagship screen is built around.
"""

from __future__ import annotations

from datetime import UTC, datetime
from decimal import Decimal
from typing import Any

from app.schemas.catalog import CompatibilityOut, CompatibilityPairOut, ToolOut
from app.services import stack_score_service
from app.services.stack_score_service import DIMENSIONS, score


def _tool(
    slug: str = "thing",
    *,
    category: str = "vector-db",
    maturity: int = 80,
    self_hostable: bool = True,
    license: str | None = "Apache-2.0",
    docs: bool = True,
    **facts: Any,
) -> ToolOut:
    base = {
        "managed": True,
        "ops_burden": 3,
        "filtering": 3,
        "hybrid_search": True,
        "scale_ceiling": 3,
        "ecosystem": 3,
        "lock_in": 3,
        "free_tier": True,
    }
    base.update(facts)
    return ToolOut(
        id=f"tool_{slug}",
        slug=slug,
        name=slug.title(),
        category=category,
        description="A thing.",
        status="recommended",
        maturity_score=maturity,
        license=license,
        self_hostable=self_hostable,
        docs_url="https://example.com/docs" if docs else None,
        facts=base,
        last_reviewed_at=datetime(2026, 6, 29, tzinfo=UTC),
    )


def _compatibility(overall: int = 80) -> CompatibilityOut:
    return CompatibilityOut(
        tools=["a", "b"],
        pairs=[CompatibilityPairOut(tool_a="a", tool_b="b", score=overall, dimensions={})],
        overall=overall,
        weakest_pair=None,
    )


# ── the weights ──────────────────────────────────────────────────────────────


def test_the_weights_sum_to_one() -> None:
    """A table that silently sums to 0.97 produces scores wrong by 3% that
    look entirely plausible."""
    assert sum((dimension.weight for dimension in DIMENSIONS), Decimal(0)) == Decimal(1)


def test_there_are_exactly_ten_dimensions_with_the_documented_weights() -> None:
    weights = {dimension.key: dimension.weight for dimension in DIMENSIONS}

    assert len(DIMENSIONS) == 10
    assert weights["cost_efficiency"] == Decimal("0.15")
    assert weights["scalability"] == Decimal("0.12")
    assert weights["developer_experience"] == Decimal("0.12")
    assert weights["production_readiness"] == Decimal("0.12")
    assert weights["security_readiness"] == Decimal("0.10")
    assert weights["vendor_lock_in"] == Decimal("0.10")
    assert weights["integration_compatibility"] == Decimal("0.10")
    assert weights["deployment_complexity"] == Decimal("0.08")
    assert weights["community_maturity"] == Decimal("0.06")
    assert weights["documentation_quality"] == Decimal("0.05")


# ── the total ────────────────────────────────────────────────────────────────


def test_every_dimension_is_computed_and_the_total_is_the_weighted_sum() -> None:
    """One component with every fact pinned, worked through by hand.

    ops_burden 1 → deployment complexity 10; lock_in 1 → vendor lock-in 10;
    ecosystem 5 → 10; scale_ceiling 5 against a medium target (needs 2) →
    headroom 3 → 10; maturity 90 → production readiness 9; compatibility 80 →
    8. Documentation: ecosystem 10 + docs present, capped at 10.
    """
    tool = _tool(
        maturity=90,
        ops_burden=1,
        lock_in=1,
        ecosystem=5,
        scale_ceiling=5,
        managed=True,
        free_tier=True,
    )

    result = score(
        [tool],
        monthly_budget=2_000,
        scale_target="medium",
        sensitivity="internal",
        compatibility=_compatibility(80),
    )

    assert result.dimensions["scalability"] == Decimal(10)
    assert result.dimensions["deployment_complexity"] == Decimal(10)
    assert result.dimensions["vendor_lock_in"] == Decimal(10)
    assert result.dimensions["developer_experience"] == Decimal(10)
    assert result.dimensions["production_readiness"] == Decimal(9)
    assert result.dimensions["integration_compatibility"] == Decimal(8)
    assert result.dimensions["documentation_quality"] == Decimal(10)
    assert result.dimensions["community_maturity"] == Decimal("9.5")

    # Budget is "moderate", so licence cost dominates: 4 base + 3 open source
    # + 2 self-hostable + 1 free tier = 10.
    assert result.dimensions["cost_efficiency"] == Decimal(10)
    # Not restricted, so managed is worth 4 on top of maturity/20 = 4.5.
    assert result.dimensions["security_readiness"] == Decimal("8.5")

    expected = sum(
        (result.dimensions[dimension.key] * dimension.weight * 10 for dimension in DIMENSIONS),
        Decimal(0),
    )
    assert result.total == expected.quantize(Decimal("0.1"))
    # 15.0 + 12.0 + 12.0 + 10.8 + 8.5 + 10.0 + 8.0 + 8.0 + 5.7 + 5.0
    assert result.total == Decimal("95.0")


def test_the_breakdown_contributions_sum_to_the_headline() -> None:
    """What makes the score checkable on screen: the ten rows add up."""
    result = score(
        [_tool(), _tool("other", category="database")],
        monthly_budget=2_000,
        scale_target="medium",
        sensitivity="internal",
        compatibility=_compatibility(75),
    )

    contributions = sum((Decimal(row["contribution"]) for row in result.breakdown()), Decimal(0))
    assert abs(contributions - result.total) <= Decimal("0.5")


# ── the dimensions that depend on the user ───────────────────────────────────


def test_cost_efficiency_scores_the_same_stack_differently_under_two_budgets() -> None:
    """An absolute cost score tells the user nothing they did not know."""
    managed_only = _tool(
        "pinecone", self_hostable=False, license=None, managed=True, free_tier=False, ops_burden=1
    )

    tight = score([managed_only], monthly_budget=300, scale_target="medium", sensitivity="internal")
    generous = score(
        [managed_only], monthly_budget=40_000, scale_target="medium", sensitivity="internal"
    )

    assert tight.dimensions["cost_efficiency"] < generous.dimensions["cost_efficiency"]


def test_scalability_is_scored_against_the_target_not_in_the_abstract() -> None:
    mid = _tool(scale_ceiling=3)

    small = score([mid], monthly_budget=2_000, scale_target="small", sensitivity="internal")
    huge = score([mid], monthly_budget=2_000, scale_target="xlarge", sensitivity="internal")

    assert small.dimensions["scalability"] == Decimal(10)
    assert huge.dimensions["scalability"] == Decimal(2)


def test_security_readiness_inverts_with_sensitivity() -> None:
    """On restricted data, self-hostability is the security property that
    matters. On public data, a vendor patching for you is worth more."""
    self_hosted = _tool("pgvector", self_hostable=True, managed=False, maturity=80)

    restricted = score(
        [self_hosted], monthly_budget=2_000, scale_target="medium", sensitivity="restricted"
    )
    public = score([self_hosted], monthly_budget=2_000, scale_target="medium", sensitivity="public")

    assert restricted.dimensions["security_readiness"] > public.dimensions["security_readiness"]


def test_integration_compatibility_uses_the_worst_pair_not_the_average() -> None:
    """Averaging lets four good pairings hide the one that does not work."""
    compatibility = CompatibilityOut(
        tools=["a", "b", "c"],
        pairs=[
            CompatibilityPairOut(tool_a="a", tool_b="b", score=95, dimensions={}),
            CompatibilityPairOut(tool_a="a", tool_b="c", score=95, dimensions={}),
            CompatibilityPairOut(tool_a="b", tool_b="c", score=30, dimensions={}),
        ],
        overall=30,
        weakest_pair=CompatibilityPairOut(tool_a="b", tool_b="c", score=30, dimensions={}),
    )

    result = score(
        [_tool()],
        monthly_budget=2_000,
        scale_target="medium",
        sensitivity="internal",
        compatibility=compatibility,
    )
    assert result.dimensions["integration_compatibility"] == Decimal(3)


def test_an_unscored_stack_is_neutral_rather_than_zero() -> None:
    """No compatibility data is not evidence of incompatibility."""
    result = score([_tool()], monthly_budget=2_000, scale_target="medium", sensitivity="internal")

    assert result.dimensions["integration_compatibility"] == Decimal(5)


def test_a_facts_free_tool_still_scores_rather_than_crashing() -> None:
    bare = ToolOut(
        id="tool_bare",
        slug="bare",
        name="Bare",
        category="database",
        description="No facts recorded.",
        status="stable",
        maturity_score=50,
        self_hostable=False,
        facts={},
        last_reviewed_at=datetime(2026, 6, 29, tzinfo=UTC),
    )

    result = score([bare], monthly_budget=2_000, scale_target="medium", sensitivity="internal")
    assert Decimal(0) <= result.total <= Decimal(100)


def test_budget_bands_are_ordered() -> None:
    assert stack_score_service.budget_band(100) == "tight"
    assert stack_score_service.budget_band(1_500) == "moderate"
    assert stack_score_service.budget_band(5_000) == "comfortable"
    assert stack_score_service.budget_band(50_000) == "generous"
