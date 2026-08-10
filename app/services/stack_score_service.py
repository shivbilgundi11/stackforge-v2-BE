"""The Stack Score. Ten weighted dimensions, computed and never stored.

Computed on read is the load-bearing choice. A score stored with a stack is a
snapshot of a catalog that has since moved: the tool got deprecated, the
maturity dropped, the compatibility pair got re-reviewed — and the saved stack
goes on showing 87 forever. Computing it means a catalog change updates every
existing stack's score, which is what makes the deprecation alert meaningful
rather than cosmetic.

Every dimension reads facts the catalog actually holds. Nothing here invents a
number to fill a dimension out: where the data does not support a judgement,
the dimension says so through `confidence` rather than guessing and presenting
the guess at the same weight as a measurement.

Weights are `PRD.md` §8.3 and sum to 1.0, which is asserted in the tests —
a weight table that silently sums to 0.97 produces scores that are wrong by
3% and look entirely plausible.
"""

from __future__ import annotations

from decimal import ROUND_HALF_UP, Decimal
from typing import Any, Final, NamedTuple

from app.schemas.catalog import CompatibilityOut, ToolOut

TENTH: Final = Decimal("0.1")


class Dimension(NamedTuple):
    key: str
    label: str
    weight: Decimal
    description: str


#: Order is display order. Weights are from the PRD and must sum to 1.
DIMENSIONS: Final[tuple[Dimension, ...]] = (
    Dimension(
        "cost_efficiency",
        "Cost efficiency",
        Decimal("0.15"),
        "Scored against your stated budget, not in absolute terms.",
    ),
    Dimension(
        "scalability",
        "Scalability",
        Decimal("0.12"),
        "Largest deployment each component is credible at, against your scale target.",
    ),
    Dimension(
        "developer_experience",
        "Developer experience",
        Decimal("0.12"),
        "Ecosystem, integrations, and how much of the work is already done for you.",
    ),
    Dimension(
        "production_readiness",
        "Production readiness",
        Decimal("0.12"),
        "Catalog maturity — how much of this has been run in anger by other people.",
    ),
    Dimension(
        "security_readiness",
        "Security readiness",
        Decimal("0.10"),
        "Whether the stack can satisfy your data sensitivity without heroics.",
    ),
    Dimension(
        "vendor_lock_in",
        "Vendor lock-in risk",
        Decimal("0.10"),
        "How hard each component would be to leave. Higher is more portable.",
    ),
    Dimension(
        "integration_compatibility",
        "Integration compatibility",
        Decimal("0.10"),
        "Pairwise compatibility across the whole stack, scored on its worst pairing.",
    ),
    Dimension(
        "deployment_complexity",
        "Deployment complexity",
        Decimal("0.08"),
        "Operational burden. Higher is less to run.",
    ),
    Dimension(
        "community_maturity",
        "Community maturity",
        Decimal("0.06"),
        "Docs, hiring pool, and how likely your problem is already answered.",
    ),
    Dimension(
        "documentation_quality",
        "Documentation quality",
        Decimal("0.05"),
        "Whether the docs exist and are worth reading.",
    ),
)

BY_KEY: Final[dict[str, Dimension]] = {dimension.key: dimension for dimension in DIMENSIONS}

#: Budget bands, in dollars per month. Cost efficiency is scored against the
#: user's stated budget rather than absolutely: a $4,000/month stack is
#: excellent on a $10,000 budget and terrible on a $1,000 one, and an absolute
#: score tells the user nothing they did not already know.
BUDGET_BANDS: Final[tuple[tuple[int, str], ...]] = (
    (500, "tight"),
    (2_000, "moderate"),
    (10_000, "comfortable"),
)

SCALE_TARGETS: Final[dict[str, int]] = {"small": 1, "medium": 2, "large": 4, "xlarge": 5}


def _fact(tool: ToolOut, key: str, default: float = 3.0) -> float:
    value = (tool.facts or {}).get(key, default)
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _flag(tool: ToolOut, key: str, *, default: bool = False) -> bool:
    return bool((tool.facts or {}).get(key, default))


def _scale(value: float, *, maximum: float = 5.0, invert: bool = False) -> Decimal:
    """A 1-`maximum` catalog fact onto 0-10.

    `invert` for the facts where a high raw value is bad — operational burden
    and lock-in both read "5 = worst" in the catalog and "0 = worst" here.
    """
    clamped = max(1.0, min(maximum, value))
    fraction = (clamped - 1) / (maximum - 1)
    if invert:
        fraction = 1 - fraction
    return (Decimal(str(fraction)) * 10).quantize(TENTH, rounding=ROUND_HALF_UP)


def _mean(values: list[Decimal]) -> Decimal:
    if not values:
        return Decimal(0)
    return (sum(values, Decimal(0)) / len(values)).quantize(TENTH, rounding=ROUND_HALF_UP)


def budget_band(monthly_budget: int) -> str:
    for ceiling, name in BUDGET_BANDS:
        if monthly_budget <= ceiling:
            return name
    return "generous"


# ── the ten dimensions ───────────────────────────────────────────────────────


def _cost_efficiency(components: list[ToolOut], *, monthly_budget: int) -> Decimal:
    """How well the stack's cost *shape* fits the stated budget.

    A fit score, not a bill. The catalog holds a real floor for some
    components (`min_monthly`) and a pricing model for the rest; inventing a
    dollar figure for the ones it does not would be exactly the fabricated
    precision D-16 exists to prevent. The bill is WF1's job, and the result
    page links into it.

    On a tight budget an open-source, self-hostable component scores best. On
    a generous one that inverts: paying a vendor to run it is the better trade
    when the money is there, because the cost that actually bites is the
    engineer-hours.
    """
    band = budget_band(monthly_budget)
    scores: list[Decimal] = []

    for tool in components:
        free = _flag(tool, "free_tier")
        managed = _flag(tool, "managed")
        open_source = bool(tool.license) and "proprietary" not in (tool.license or "").lower()
        ops = _fact(tool, "ops_burden")

        if band in {"tight", "moderate"}:
            # Licence cost dominates when there is little to spend.
            score = Decimal(4)
            if open_source:
                score += Decimal(3)
            if tool.self_hostable:
                score += Decimal(2)
            if free:
                score += Decimal(1)
            if managed and not free and band == "tight":
                score -= Decimal(3)
        else:
            # With budget available, operational burden is the real cost.
            score = Decimal(5)
            if managed:
                score += Decimal(3)
            if ops <= 2:
                score += Decimal(2)
            elif ops >= 4:
                score -= Decimal(2)

        floor = (tool.facts or {}).get("min_monthly")
        if floor is not None and monthly_budget > 0:
            share = float(floor) / monthly_budget
            if share > 0.5:
                score -= Decimal(4)
            elif share > 0.25:
                score -= Decimal(2)

        scores.append(max(Decimal(0), min(Decimal(10), score)))

    return _mean(scores)


def _scalability(components: list[ToolOut], *, scale_target: str) -> Decimal:
    """Ceiling against the target, not in the abstract.

    A component credible to 3 is fine for a medium deployment and a liability
    at xlarge — the same tool, two different answers.
    """
    needed = SCALE_TARGETS.get(scale_target, 2)
    scores: list[Decimal] = []

    for tool in components:
        ceiling = _fact(tool, "scale_ceiling")
        headroom = ceiling - needed
        if headroom >= 1:
            scores.append(Decimal(10))
        elif headroom >= 0:
            scores.append(Decimal(8))
        elif headroom >= -1:
            scores.append(Decimal(5))
        else:
            scores.append(Decimal(2))
    return _mean(scores)


def _developer_experience(components: list[ToolOut]) -> Decimal:
    scores: list[Decimal] = []
    for tool in components:
        ecosystem = _scale(_fact(tool, "ecosystem"))
        friction = _scale(_fact(tool, "ops_burden"), invert=True)
        # Ecosystem dominates: what makes a component pleasant to build on is
        # mostly that someone has already written the integration you need.
        scores.append((ecosystem * 7 + friction * 3) / 10)
    return _mean(scores)


def _production_readiness(components: list[ToolOut]) -> Decimal:
    return _mean([Decimal(tool.maturity_score) / 10 for tool in components])


def _security_readiness(components: list[ToolOut], *, sensitivity: str) -> Decimal:
    """Whether the stack can meet the stated sensitivity without heroics.

    Restricted data makes self-hostability the security property that matters;
    on public data a managed vendor patching for you is worth more than the
    ability to run it yourself.
    """
    restricted = sensitivity in {"restricted", "regulated"}
    scores: list[Decimal] = []

    for tool in components:
        base = Decimal(tool.maturity_score) / 20  # 0-5 from a 0-100 maturity
        if restricted:
            base += Decimal(5) if tool.self_hostable else Decimal(0)
        else:
            base += Decimal(4) if _flag(tool, "managed") else Decimal(2)
        scores.append(max(Decimal(0), min(Decimal(10), base)))
    return _mean(scores)


def _vendor_lock_in(components: list[ToolOut]) -> Decimal:
    return _mean([_scale(_fact(tool, "lock_in"), invert=True) for tool in components])


def _integration_compatibility(compatibility: CompatibilityOut | None) -> Decimal:
    """The stack's *worst* pairing, not its average.

    A stack is only as compatible as its weakest pair. Averaging lets four
    good pairings hide the one combination that does not work, which is
    precisely what this dimension exists to surface.
    """
    if compatibility is None:
        return Decimal(5)
    return (Decimal(compatibility.overall) / 10).quantize(TENTH, rounding=ROUND_HALF_UP)


def _deployment_complexity(components: list[ToolOut]) -> Decimal:
    return _mean([_scale(_fact(tool, "ops_burden"), invert=True) for tool in components])


def _community_maturity(components: list[ToolOut]) -> Decimal:
    return _mean(
        [
            (_scale(_fact(tool, "ecosystem")) + Decimal(tool.maturity_score) / 10) / 2
            for tool in components
        ]
    )


def _documentation_quality(components: list[ToolOut]) -> Decimal:
    """Docs are not scored in the catalog, so this is composed from proxies —
    whether documentation exists at all, and how large the ecosystem writing
    about it is. Stated plainly rather than dressed up as a measurement."""
    scores: list[Decimal] = []
    for tool in components:
        score = _scale(_fact(tool, "ecosystem"))
        if tool.docs_url:
            score = min(Decimal(10), score + Decimal(2))
        else:
            score = max(Decimal(0), score - Decimal(3))
        scores.append(score)
    return _mean(scores)


# ── the score ────────────────────────────────────────────────────────────────


class StackScore(NamedTuple):
    total: Decimal
    dimensions: dict[str, Decimal]

    def breakdown(self) -> list[dict[str, Any]]:
        """Display rows, with the weighted contribution spelled out.

        The contribution column is what makes the total checkable: the ten of
        them sum to the headline, and a reader can see which dimension is
        carrying it.
        """
        return [
            {
                "dimension": dimension.label,
                "key": dimension.key,
                "score": str(self.dimensions[dimension.key]),
                "weight_pct": str((dimension.weight * 100).quantize(TENTH)),
                "contribution": str(
                    (self.dimensions[dimension.key] * dimension.weight * 10).quantize(
                        TENTH, rounding=ROUND_HALF_UP
                    )
                ),
                "description": dimension.description,
            }
            for dimension in DIMENSIONS
        ]


def score(
    components: list[ToolOut],
    *,
    monthly_budget: int,
    scale_target: str,
    sensitivity: str,
    compatibility: CompatibilityOut | None = None,
) -> StackScore:
    """The ten dimensions and the weighted total, out of 100."""
    dimensions: dict[str, Decimal] = {
        "cost_efficiency": _cost_efficiency(components, monthly_budget=monthly_budget),
        "scalability": _scalability(components, scale_target=scale_target),
        "developer_experience": _developer_experience(components),
        "production_readiness": _production_readiness(components),
        "security_readiness": _security_readiness(components, sensitivity=sensitivity),
        "vendor_lock_in": _vendor_lock_in(components),
        "integration_compatibility": _integration_compatibility(compatibility),
        "deployment_complexity": _deployment_complexity(components),
        "community_maturity": _community_maturity(components),
        "documentation_quality": _documentation_quality(components),
    }

    total = sum(
        (dimensions[dimension.key] * dimension.weight * 10 for dimension in DIMENSIONS),
        Decimal(0),
    )
    return StackScore(
        total=total.quantize(TENTH, rounding=ROUND_HALF_UP),
        dimensions=dimensions,
    )
