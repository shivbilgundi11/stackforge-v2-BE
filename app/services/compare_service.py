"""The comparison engine.

All four P1 comparisons share one output shape:

    criteria · options · matrix · winner · rationale · tradeoffs · switch_when

One contract, one renderer, four tools. `switch_when` is the field that makes a
comparison useful rather than a leaderboard — "Pinecone wins on operational
simplicity; choose pgvector if you already run Postgres and your corpus is
under 5M vectors" is the sentence a senior engineer came for, and no amount of
scoring produces it.

Cost is always computed from the user's stated scale, never scored. A stored
"cost: 7/10" is wrong the day a provider changes price, and the day after that
it is wrong and nobody knows.
"""

from __future__ import annotations

from decimal import ROUND_HALF_UP, Decimal
from typing import Any

from app.data.compare_criteria import (
    CRITERIA_BY_TOOL,
    STACK_ARCHETYPES_BY_KEY,
    Criterion,
    Priority,
    StackArchetype,
)
from app.schemas.catalog import ModelOut, ToolOut
from app.schemas.tools import ToolOutput, ToolWarning

CENTS = Decimal("0.01")


def _money_from(display: str) -> Decimal:
    """Read a `$1,234.56` display string back to a Decimal."""
    return Decimal(display.replace("$", "").replace(",", ""))


def _usd(value: Decimal) -> str:
    """A display string with thousands separators.

    These land straight in the matrix cell, so `$49920.00` versus `$49,920.00`
    is the difference between a figure the reader parses and one they squint
    at. Grouping happens here because the cell is a pre-formatted string by
    the time the frontend sees it.
    """
    return f"${value.quantize(CENTS):,.2f}"


DAYS_PER_MONTH = Decimal("30.4375")
THOUSAND = Decimal(1000)

# A deprecated option cannot win, whatever its numbers say. Recommending a
# buried tool because it scores well on cost is exactly the failure the Tool
# Graveyard exists to prevent.
STATUS_MULTIPLIER = {
    "recommended": 1.0,
    "stable": 1.0,
    "caution": 0.85,
    "deprecated": 0.5,
    "not_for_production": 0.35,
}


def _score(value: float) -> int:
    return max(0, min(100, round(value)))


def _normalise_lower_is_better(values: list[float]) -> list[float]:
    """Map a set of costs (or times, or burdens) onto 0-100, cheapest = 100."""
    if not values:
        return []
    low, high = min(values), max(values)
    if high == low:
        return [100.0] * len(values)
    return [100.0 * (high - value) / (high - low) for value in values]


def _normalise_higher_is_better(values: list[float]) -> list[float]:
    if not values:
        return []
    low, high = min(values), max(values)
    if high == low:
        return [100.0] * len(values)
    return [100.0 * (value - low) / (high - low) for value in values]


def _weights(criteria: tuple[Criterion, ...], priority: Priority) -> dict[str, float]:
    raw = {c.key: c.weight * c.weights.get(priority, 1.0) for c in criteria}
    total = sum(raw.values()) or 1.0
    return {key: value / total for key, value in raw.items()}


def _assemble(
    *,
    tool_slug: str,
    priority: Priority,
    options: list[dict[str, Any]],
    scores: dict[str, dict[str, float]],
    raw_values: dict[str, dict[str, str]],
    rationale_for: Any,
    warnings: list[ToolWarning] | None = None,
    sourced_from: list[str] | None = None,
) -> ToolOutput:
    """Weight, rank, and package. Shared by all four comparisons."""
    criteria = CRITERIA_BY_TOOL[tool_slug]
    weights = _weights(criteria, priority)

    totals: dict[str, float] = {}
    for option in options:
        option_id = option["id"]
        total = sum(
            scores[option_id].get(criterion.key, 0.0) * weights[criterion.key]
            for criterion in criteria
        )
        totals[option_id] = total * option.get("status_multiplier", 1.0)

    ranked = sorted(options, key=lambda o: totals[o["id"]], reverse=True)
    winner = ranked[0]
    runner_up = ranked[1] if len(ranked) > 1 else None

    # Confidence is the gap to the runner-up. A two-point win is a coin flip
    # dressed up as a recommendation, and saying so is more useful than
    # pretending otherwise.
    gap = totals[winner["id"]] - (totals[runner_up["id"]] if runner_up else 0.0)
    if runner_up is None or gap >= 12:
        confidence = "high"
    elif gap >= 5:
        confidence = "medium"
    else:
        confidence = "low"

    matrix = [
        {
            "criterion": criterion.key,
            "label": criterion.label,
            "description": criterion.description,
            "weight": round(weights[criterion.key], 4),
            "unit": criterion.unit,
            **{
                option["id"]: {
                    "score": _score(scores[option["id"]].get(criterion.key, 0.0)),
                    "value": raw_values[option["id"]].get(criterion.key, "-"),
                }
                for option in options
            },
        }
        for criterion in criteria
    ]

    rationale, tradeoffs, switch_when = rationale_for(winner, ranked, priority)

    all_warnings = list(warnings or [])
    if confidence == "low" and runner_up:
        all_warnings.append(
            ToolWarning(
                level="info",
                message=(
                    f"{winner['name']} and {runner_up['name']} score within "
                    f"{gap:.1f} points. Treat this as a tie and decide on the "
                    f"tradeoffs below rather than the ranking."
                ),
            )
        )

    return ToolOutput(
        metrics={
            "winner": winner["id"],
            "winner_name": winner["name"],
            "confidence": confidence,
            "score": _score(totals[winner["id"]]),
            "priority": priority,
            "options_compared": len(options),
        },
        tables={
            "matrix": matrix,
            "options": [
                {
                    **{k: v for k, v in option.items() if not k.startswith("_")},
                    "total_score": _score(totals[option["id"]]),
                    "rank": index + 1,
                    "is_winner": index == 0,
                }
                for index, option in enumerate(ranked)
            ],
            "rationale": [
                {"kind": "why", "text": rationale},
                *[{"kind": "tradeoff", "text": item} for item in tradeoffs],
                *[{"kind": "switch_when", "text": item} for item in switch_when],
            ],
        },
        series={
            "scores": [
                {"option": option["name"], "score": _score(totals[option["id"]])}
                for option in ranked
            ]
        },
        warnings=all_warnings,
        sourced_from=sourced_from or [],
    )


# ── compare-models ───────────────────────────────────────────────────────────


def compare_models(
    *,
    models: list[ModelOut],
    input_tokens: int,
    output_tokens: int,
    requests_per_day: int,
    cached_input_ratio: Decimal = Decimal(0),
    priority: Priority = "balanced",
) -> ToolOutput:
    from app.services.cost_service import cost_per_request

    monthly_costs: list[float] = []
    for model in models:
        per_request = cost_per_request(
            model,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            cached_input_ratio=cached_input_ratio,
        )
        monthly_costs.append(float(per_request * requests_per_day * DAYS_PER_MONTH))

    cost_scores = _normalise_lower_is_better(monthly_costs)
    window_scores = _normalise_higher_is_better(
        [float(model.context_window or 0) for model in models]
    )
    output_scores = _normalise_higher_is_better(
        [float(model.max_output_tokens or 0) for model in models]
    )

    options: list[dict[str, Any]] = []
    scores: dict[str, dict[str, float]] = {}
    raw: dict[str, dict[str, str]] = {}

    for index, model in enumerate(models):
        capabilities = model.capabilities or {}
        monthly = Decimal(str(monthly_costs[index])).quantize(CENTS)
        options.append(
            {
                "id": model.model_id,
                "name": model.display_name,
                "provider": model.provider,
                "status": model.status,
                "monthly_cost": str(monthly),
                "context_window": model.context_window,
                "last_verified_at": model.provenance.last_verified_at.date().isoformat(),
                "provenance_variant": model.provenance.variant,
                "status_multiplier": 1.0 if model.status == "active" else 0.5,
            }
        )
        scores[model.model_id] = {
            "blended_cost": cost_scores[index],
            "context_window": window_scores[index],
            "cache_support": 100.0 if model.cached_input_cost_per_1k else 0.0,
            "reasoning": 100.0 if capabilities.get("thinking") else 40.0,
            "tool_use": 100.0 if capabilities.get("tools") else 0.0,
            "multimodal": 100.0 if capabilities.get("vision") else 0.0,
            "output_ceiling": output_scores[index],
            "freshness": 100.0 if model.status == "active" else 25.0,
        }
        raw[model.model_id] = {
            "blended_cost": f"{_usd(monthly)}/mo",
            "context_window": f"{model.context_window:,}" if model.context_window else "-",
            "cache_support": (
                f"${model.cached_input_cost_per_1k}/1k" if model.cached_input_cost_per_1k else "no"
            ),
            "reasoning": "yes" if capabilities.get("thinking") else "no",
            "tool_use": "yes" if capabilities.get("tools") else "no",
            "multimodal": "yes" if capabilities.get("vision") else "no",
            "output_ceiling": (f"{model.max_output_tokens:,}" if model.max_output_tokens else "-"),
            "freshness": model.status,
        }

    def rationale_for(
        winner: dict[str, Any], ranked: list[dict[str, Any]], priority: Priority
    ) -> tuple[str, list[str], list[str]]:
        cheapest = min(ranked, key=lambda o: Decimal(o["monthly_cost"]))
        widest = max(ranked, key=lambda o: o["context_window"] or 0)
        why = (
            f"{winner['name']} wins on a {priority} weighting at "
            f"${winner['monthly_cost']}/month for {requests_per_day:,} requests a day."
        )
        tradeoffs: list[str] = []
        switch: list[str] = []

        if cheapest["id"] != winner["id"]:
            delta = Decimal(winner["monthly_cost"]) - Decimal(cheapest["monthly_cost"])
            tradeoffs.append(
                f"It costs {_usd(delta)}/month more than "
                f"{cheapest['name']}, the cheapest option here."
            )
            switch.append(
                f"Choose {cheapest['name']} if this workload is cost-bound and its "
                f"quality is good enough on your evals — it is "
                f"{_usd(delta)}/month cheaper at this volume."
            )
        if widest["id"] != winner["id"] and widest["context_window"]:
            switch.append(
                f"Choose {widest['name']} if prompts may exceed "
                f"{winner['context_window']:,} tokens — it accepts "
                f"{widest['context_window']:,}."
            )
        if cached := [o for o in ranked if o["id"] != winner["id"]]:
            tradeoffs.append(
                f"Ranked above {', '.join(o['name'] for o in cached[:2])} on this "
                f"weighting; a different priority may reorder them."
            )
        switch.append(
            "Re-run with priority=cost if budget is the binding constraint, or "
            "priority=scale if context length is."
        )
        return why, tradeoffs, switch

    return _assemble(
        tool_slug="compare-models",
        priority=priority,
        options=options,
        scores=scores,
        raw_values=raw,
        rationale_for=rationale_for,
        sourced_from=[model.id for model in models],
    )


# ── compare-vector-db ────────────────────────────────────────────────────────


def compare_vector_db(
    *,
    tools: list[ToolOut],
    vector_count: int,
    dimensions: int,
    priority: Priority = "balanced",
) -> ToolOutput:
    """Cost at the stated scale is computed, not asserted.

    Storage scales with both vector count and dimensionality, so a 3072-dim
    corpus costs roughly twice a 1536-dim one at the same vector count — a
    fact a static score cannot express and the single biggest driver of the
    answer.
    """
    millions = Decimal(vector_count) / Decimal(1_000_000)
    dimension_factor = Decimal(dimensions) / Decimal(1536)

    monthly_costs: list[float] = []
    for tool in tools:
        facts = tool.facts or {}
        per_million = Decimal(str(facts.get("cost_per_m_vectors_month", 5.0)))
        minimum = Decimal(str(facts.get("min_monthly", 0)))
        cost = max(minimum, (per_million * millions * dimension_factor))
        monthly_costs.append(float(cost.quantize(CENTS)))

    cost_scores = _normalise_lower_is_better(monthly_costs)

    options: list[dict[str, Any]] = []
    scores: dict[str, dict[str, float]] = {}
    raw: dict[str, dict[str, str]] = {}

    for index, tool in enumerate(tools):
        facts = tool.facts or {}
        monthly = Decimal(str(monthly_costs[index])).quantize(CENTS)
        options.append(
            {
                "id": tool.slug,
                "name": tool.name,
                "status": tool.status,
                "status_reason": tool.status_reason,
                "monthly_cost": str(monthly),
                "self_hostable": tool.self_hostable,
                "license": tool.license,
                "last_reviewed_at": tool.last_reviewed_at.date().isoformat(),
                "status_multiplier": STATUS_MULTIPLIER.get(tool.status, 1.0),
            }
        )
        scores[tool.slug] = {
            "monthly_cost": cost_scores[index],
            "ops_burden": (5 - float(facts.get("ops_burden", 3))) / 4 * 100,
            "filtering": float(facts.get("filtering", 3)) / 5 * 100,
            "hybrid_search": 100.0 if facts.get("hybrid_search") else 0.0,
            "scale_ceiling": float(facts.get("scale_ceiling", 3)) / 5 * 100,
            "ecosystem": float(facts.get("ecosystem", 3)) / 5 * 100,
            "vendor_lock_in": (5 - float(facts.get("lock_in", 3))) / 4 * 100,
            "lifecycle": STATUS_MULTIPLIER.get(tool.status, 1.0) * 100,
        }
        raw[tool.slug] = {
            "monthly_cost": f"{_usd(monthly)}/mo",
            "ops_burden": f"{facts.get('ops_burden', 3)}/5",
            "filtering": f"{facts.get('filtering', 3)}/5",
            "hybrid_search": "yes" if facts.get("hybrid_search") else "no",
            "scale_ceiling": f"{facts.get('scale_ceiling', 3)}/5",
            "ecosystem": f"{facts.get('ecosystem', 3)}/5",
            "vendor_lock_in": "self-hostable" if tool.self_hostable else "managed only",
            "lifecycle": tool.status,
        }

    def rationale_for(
        winner: dict[str, Any], ranked: list[dict[str, Any]], priority: Priority
    ) -> tuple[str, list[str], list[str]]:
        cheapest = min(ranked, key=lambda o: Decimal(o["monthly_cost"]))
        self_hosted = [o for o in ranked if o["self_hostable"] and o["id"] != winner["id"]]
        why = (
            f"{winner['name']} wins on a {priority} weighting at "
            f"{vector_count:,} vectors x {dimensions} dimensions, costing about "
            f"${winner['monthly_cost']}/month."
        )
        tradeoffs: list[str] = []
        switch: list[str] = []

        if not winner["self_hostable"]:
            tradeoffs.append(
                "Managed only — no self-hosting path if data residency or "
                "cost control later demands one."
            )
        if cheapest["id"] != winner["id"]:
            delta = Decimal(winner["monthly_cost"]) - Decimal(cheapest["monthly_cost"])
            tradeoffs.append(f"{_usd(delta)}/month more than {cheapest['name']} at this scale.")
        if vector_count <= 5_000_000 and winner["id"] != "pgvector":
            switch.append(
                "Choose pgvector if you already run Postgres: under about 5M vectors "
                "it wins on total operational cost because it is not a second "
                "datastore to run, back up, and monitor."
            )
        if self_hosted:
            switch.append(
                f"Choose {self_hosted[0]['name']} if data cannot leave your network "
                f"or you need to cap cost by owning the hardware."
            )
        if vector_count >= 50_000_000:
            switch.append(
                "At this corpus size, benchmark on your own data before committing "
                "— published figures diverge sharply above 50M vectors."
            )
        switch.append(
            "Re-run with priority=simplicity to weight operational burden, or "
            "priority=control to weight portability."
        )
        return why, tradeoffs, switch

    buried = [tool for tool in tools if tool.status in ("deprecated", "not_for_production")]
    warnings = [
        ToolWarning(
            level="warning",
            message=f"{tool.name}: {tool.status_reason}",
        )
        for tool in buried
        if tool.status_reason
    ]

    return _assemble(
        tool_slug="compare-vector-db",
        priority=priority,
        options=options,
        scores=scores,
        raw_values=raw,
        rationale_for=rationale_for,
        warnings=warnings,
    )


# ── compare-stacks ───────────────────────────────────────────────────────────


def compare_stacks(
    *,
    archetypes: list[StackArchetype],
    monthly_model_spend: Decimal = Decimal(500),
    blended_hourly_rate: Decimal = Decimal(120),
    priority: Priority = "balanced",
) -> ToolOutput:
    """12-month TCO includes engineering time, because it dominates.

    A stack that is $600/month cheaper in infrastructure and takes three extra
    engineer-weeks to stand up is not cheaper. Costing only the invoice is how
    "open source is free" survives as a belief.
    """
    hours_per_day = Decimal(8)

    tco: list[float] = []
    for archetype in archetypes:
        setup = Decimal(archetype.setup_days) * hours_per_day * blended_hourly_rate
        # Ongoing maintenance scaled off operational burden: one point of
        # burden is about two engineer-days a month.
        maintenance = Decimal(archetype.ops_burden) * 2 * hours_per_day * blended_hourly_rate * 12
        infra = Decimal(str(archetype.infra_monthly)) * 12
        models = monthly_model_spend * 12
        tco.append(float(setup + maintenance + infra + models))

    tco_scores = _normalise_lower_is_better(tco)
    deploy_scores = _normalise_lower_is_better(
        [float(archetype.setup_days) for archetype in archetypes]
    )
    scale_scores = _normalise_higher_is_better(
        [float(archetype.scaling_ceiling) for archetype in archetypes]
    )

    options: list[dict[str, Any]] = []
    scores: dict[str, dict[str, float]] = {}
    raw: dict[str, dict[str, str]] = {}

    for index, archetype in enumerate(archetypes):
        total = Decimal(str(tco[index])).quantize(CENTS)
        options.append(
            {
                "id": archetype.key,
                "name": archetype.name,
                "description": archetype.description,
                # The component list is on the option so the winner converts
                # straight into a Stack Architect project.
                "components": list(archetype.components),
                "tco_12_month": str(total),
                "setup_days": archetype.setup_days,
                "infra_monthly": str(Decimal(str(archetype.infra_monthly)).quantize(CENTS)),
            }
        )
        scores[archetype.key] = {
            "tco_12_month": tco_scores[index],
            "time_to_deploy": deploy_scores[index],
            "scaling_ceiling": scale_scores[index],
            "vendor_lock_in": (5 - archetype.lock_in) / 4 * 100,
            "team_skill": (5 - archetype.team_skill) / 4 * 100,
            "operational_burden": (5 - archetype.ops_burden) / 4 * 100,
        }
        raw[archetype.key] = {
            "tco_12_month": _usd(total),
            "time_to_deploy": f"{archetype.setup_days} days",
            "scaling_ceiling": f"{archetype.scaling_ceiling}/5",
            "vendor_lock_in": f"{archetype.lock_in}/5",
            "team_skill": f"{archetype.team_skill}/5",
            "operational_burden": f"{archetype.ops_burden}/5",
        }

    def rationale_for(
        winner: dict[str, Any], ranked: list[dict[str, Any]], priority: Priority
    ) -> tuple[str, list[str], list[str]]:
        fastest = min(ranked, key=lambda o: o["setup_days"])
        cheapest = min(ranked, key=lambda o: _money_from(o["tco_12_month"]))
        why = (
            f"{winner['name']} wins on a {priority} weighting: "
            f"${winner['tco_12_month']} over twelve months including engineering "
            f"time, live in {winner['setup_days']} days."
        )
        tradeoffs = [
            f"Infrastructure alone is ${winner['infra_monthly']}/month; the rest of "
            f"the TCO is engineering time, which is the part budgets usually miss."
        ]
        switch = []
        if fastest["id"] != winner["id"]:
            switch.append(
                f"Choose {fastest['name']} if you need something live in "
                f"{fastest['setup_days']} days rather than {winner['setup_days']}."
            )
        if cheapest["id"] != winner["id"]:
            saving = _money_from(winner["tco_12_month"]) - _money_from(cheapest["tco_12_month"])
            switch.append(
                f"Choose {cheapest['name']} if the twelve-month budget is fixed — "
                f"it is {_usd(saving)} less."
            )
        switch.append(
            "Choose the self-hosted archetype if data residency is a hard "
            "requirement, regardless of how it scores here."
        )
        return why, tradeoffs, switch

    return _assemble(
        tool_slug="compare-stacks",
        priority=priority,
        options=options,
        scores=scores,
        raw_values=raw,
        rationale_for=rationale_for,
    )


# ── compare-build-vs-buy ─────────────────────────────────────────────────────


def compare_build_vs_buy(
    *,
    build_hours: int,
    blended_hourly_rate: Decimal,
    build_infra_monthly: Decimal,
    maintenance_hours_per_month: Decimal,
    vendor_monthly: Decimal,
    vendor_integration_hours: int = 0,
    priority: Priority = "balanced",
) -> ToolOutput:
    """Build versus buy over 12, 24, and 36 months, with a sensitivity table.

    A single break-even number invites the reader to distrust it — every
    assumption behind it is arguable. The sensitivity table is what makes the
    conclusion survive a board meeting: it shows the answer holding (or not)
    across the range of rates and hours the room will actually propose.
    """
    build_upfront = Decimal(build_hours) * blended_hourly_rate
    build_monthly = build_infra_monthly + maintenance_hours_per_month * blended_hourly_rate
    buy_upfront = Decimal(vendor_integration_hours) * blended_hourly_rate

    def build_at(months: int) -> Decimal:
        return (build_upfront + build_monthly * months).quantize(CENTS)

    def buy_at(months: int) -> Decimal:
        return (buy_upfront + vendor_monthly * months).quantize(CENTS)

    horizons = {months: (build_at(months), buy_at(months)) for months in (12, 24, 36)}

    # First month at which build's cumulative cost drops below buy's. None
    # means it never does inside a sensible planning horizon.
    break_even: int | None = None
    for month in range(1, 121):
        if build_at(month) <= buy_at(month):
            break_even = month
            break

    options: list[dict[str, Any]] = [
        {
            "id": "build",
            "name": "Build",
            "upfront_cost": str(build_upfront),
            "monthly_cost": str(build_monthly.quantize(CENTS)),
            "cost_12m": str(horizons[12][0]),
            "cost_24m": str(horizons[24][0]),
            "cost_36m": str(horizons[36][0]),
        },
        {
            "id": "buy",
            "name": "Buy",
            "upfront_cost": str(buy_upfront),
            "monthly_cost": str(vendor_monthly.quantize(CENTS)),
            "cost_12m": str(horizons[12][1]),
            "cost_24m": str(horizons[24][1]),
            "cost_36m": str(horizons[36][1]),
        },
    ]

    cost_12 = _normalise_lower_is_better([float(horizons[12][0]), float(horizons[12][1])])
    cost_36 = _normalise_lower_is_better([float(horizons[36][0]), float(horizons[36][1])])
    build_months_to_value = max(Decimal(1), Decimal(build_hours) / Decimal(160))
    buy_months_to_value = max(Decimal("0.25"), Decimal(vendor_integration_hours) / Decimal(160))
    time_scores = _normalise_lower_is_better(
        [float(build_months_to_value), float(buy_months_to_value)]
    )

    scores = {
        "build": {
            "total_cost_12m": cost_12[0],
            "total_cost_36m": cost_36[0],
            "time_to_value": time_scores[0],
            "control": 100.0,
            "risk": 35.0,
            "maintenance": 25.0,
        },
        "buy": {
            "total_cost_12m": cost_12[1],
            "total_cost_36m": cost_36[1],
            "time_to_value": time_scores[1],
            "control": 30.0,
            "risk": 85.0,
            "maintenance": 90.0,
        },
    }
    raw = {
        "build": {
            "total_cost_12m": _usd(horizons[12][0]),
            "total_cost_36m": _usd(horizons[36][0]),
            "time_to_value": f"{build_months_to_value.quantize(CENTS)} months",
            "control": "full",
            "risk": "delivery risk on your team",
            "maintenance": f"{maintenance_hours_per_month}h/month",
        },
        "buy": {
            "total_cost_12m": _usd(horizons[12][1]),
            "total_cost_36m": _usd(horizons[36][1]),
            "time_to_value": f"{buy_months_to_value.quantize(CENTS)} months",
            "control": "vendor roadmap",
            "risk": "vendor viability",
            "maintenance": "included",
        },
    }

    # Sensitivity: how the answer moves with the two most-argued inputs.
    sensitivity = []
    for hours_factor in (Decimal("0.5"), Decimal("0.75"), Decimal(1), Decimal("1.5"), Decimal(2)):
        for rate in (
            blended_hourly_rate * Decimal("0.75"),
            blended_hourly_rate,
            blended_hourly_rate * Decimal("1.5"),
        ):
            hours = Decimal(build_hours) * hours_factor
            upfront = hours * rate
            monthly = build_infra_monthly + maintenance_hours_per_month * rate
            build_36 = (upfront + monthly * 36).quantize(CENTS)
            buy_36 = (buy_upfront + vendor_monthly * 36).quantize(CENTS)
            sensitivity.append(
                {
                    "build_hours": int(hours),
                    "hourly_rate": _usd(rate),
                    "build_36m": _usd(build_36),
                    "buy_36m": _usd(buy_36),
                    "winner": "build" if build_36 < buy_36 else "buy",
                }
            )

    projection = [
        {
            "month": month,
            "build": str(build_at(month)),
            "buy": str(buy_at(month)),
        }
        for month in range(1, 37)
    ]

    def rationale_for(
        winner: dict[str, Any], ranked: list[dict[str, Any]], priority: Priority
    ) -> tuple[str, list[str], list[str]]:
        flips = {row["winner"] for row in sensitivity}
        if winner["id"] == "buy":
            why = (
                f"Buy wins on a {priority} weighting: {_usd(horizons[12][1])} against "
                f"{_usd(horizons[12][0])} over twelve months, and it is in production "
                f"months earlier."
            )
            tradeoffs = [
                "You inherit the vendor's roadmap, pricing changes, and outages.",
                f"At {_usd(vendor_monthly)}/month the cost never stops, "
                f"whereas a build's largest cost is upfront.",
            ]
        else:
            why = (
                f"Build wins on a {priority} weighting: {_usd(horizons[36][0])} against "
                f"{_usd(horizons[36][1])} over three years."
            )
            tradeoffs = [
                f"{_usd(build_upfront)} of engineering time before anything ships.",
                f"{maintenance_hours_per_month}h/month of maintenance forever, "
                f"which is the cost most build cases forget.",
            ]

        switch = []
        if break_even:
            switch.append(
                f"Build overtakes buy at month {break_even}. If your planning "
                f"horizon is shorter than that, buy is correct regardless of the "
                f"three-year figure."
            )
        else:
            switch.append(
                "Build never overtakes buy within ten years at these inputs — "
                "the vendor price would have to roughly double to change that."
            )
        if len(flips) > 1:
            switch.append(
                "The sensitivity table flips between build and buy inside the "
                "plausible range of hours and rates. Treat the recommendation as "
                "conditional on your estimate holding."
            )
        else:
            switch.append(
                "The answer holds across every rate and hour combination in the "
                "sensitivity table, so it does not hinge on the estimate."
            )
        return why, tradeoffs, switch

    output = _assemble(
        tool_slug="compare-build-vs-buy",
        priority=priority,
        options=options,
        scores=scores,
        raw_values=raw,
        rationale_for=rationale_for,
    )
    output.tables["sensitivity"] = sensitivity
    output.series["cumulative_cost"] = projection
    output.metrics["break_even_month"] = break_even if break_even else "never"
    output.metrics["build_cost_12m"] = horizons[12][0]
    output.metrics["buy_cost_12m"] = horizons[12][1]
    output.metrics["build_cost_36m"] = horizons[36][0]
    output.metrics["buy_cost_36m"] = horizons[36][1]
    return output


def resolve_archetypes(keys: list[str]) -> list[StackArchetype]:
    return [STACK_ARCHETYPES_BY_KEY[key] for key in keys if key in STACK_ARCHETYPES_BY_KEY]


def _round(value: Decimal) -> Decimal:
    return value.quantize(CENTS, rounding=ROUND_HALF_UP)
