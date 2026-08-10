"""Agent economics, rate-limit headroom, and tool-schema generation (WF3).

Three pure functions. Each takes plain values plus already-fetched catalog
rows, so every figure is checkable by hand.

The theme running through all three is that the naive version of each is wrong
in the same direction — it flatters the plan:

**Agent cost** ignores schema overhead and retries. Tool definitions are
re-sent on every single turn of the loop, so with twenty tools the definitions
alone are often the largest input line; and agent loops retry more than chat
products do. Omitting both understates real spend by a multiple, which is the
exact failure `PRD.md` §3 says the product exists to prevent.

**Rate limits** get checked against RPM, which is almost never the binding
constraint. Token budgets bind first on long-context work, output budgets bind
first on agent loops at providers that meter them separately, and the client's
own concurrency binds first more often than any of them.

**Function schemas** get eyeballed rather than validated. An invalid schema
fails at the provider with an opaque error, days later, in production.
"""

from __future__ import annotations

import json
import re
from decimal import ROUND_HALF_UP, Decimal
from typing import Any, Final

from jsonschema import Draft202012Validator
from jsonschema.exceptions import SchemaError, ValidationError

from app.data.rate_limits import ProviderLimits, TierLimits
from app.schemas.catalog import ModelOut
from app.schemas.tools import Artifact, ToolOutput, ToolWarning
from app.services.cost_service import DAYS_PER_MONTH, MONTHS_PER_YEAR, THOUSAND

CENTS: Final = Decimal("0.01")
MICRO: Final = Decimal("0.000001")
PERCENT: Final = Decimal("0.1")


def _money(value: Decimal) -> Decimal:
    """Six decimals. A single agent step is routinely sub-cent."""
    return value.quantize(MICRO, rounding=ROUND_HALF_UP)


def _display(value: Decimal) -> Decimal:
    return value.quantize(CENTS, rounding=ROUND_HALF_UP)


def _pct(value: Decimal) -> Decimal:
    return value.quantize(PERCENT, rounding=ROUND_HALF_UP)


def _usd(value: Decimal) -> str:
    return f"${value:,.2f}" if value >= 1 else f"${value:,.6f}".rstrip("0").rstrip(".")


# ── agent-cost ───────────────────────────────────────────────────────────────

#: Tokens a single tool definition costs once serialised into the request.
#: A name, a sentence of description, and three or four typed parameters lands
#: here; it is offered as a default, not a constant, because a tool with a
#: nested object schema is several times this.
DEFAULT_TOKENS_PER_TOOL_SCHEMA: Final = 120


class AgentRole:
    """One row of the agent roster.

    A class rather than a tuple because the route builds these from a payload
    and the compute function reads them by name in six places; positional
    unpacking there is how a `count` ends up multiplied by `steps_per_task`.
    """

    __slots__ = ("count", "model", "role", "steps_per_task")

    def __init__(self, *, role: str, model: ModelOut, count: int, steps_per_task: int) -> None:
        self.role = role
        self.model = model
        self.count = count
        self.steps_per_task = steps_per_task


#: The lines an agent step is decomposed into. Order is display order, and the
#: `side` decides which price applies — a memory write is tokens the model
#: emits, so it is billed at the output rate, not the input rate.
_COMPONENTS: Final[tuple[tuple[str, str, str], ...]] = (
    ("base_prompt", "Base prompt", "input"),
    ("tool_schemas", "Tool definitions", "input"),
    ("memory_reads", "Memory reads", "input"),
    ("tool_results", "Tool results", "input"),
    ("model_output", "Model output", "output"),
    ("memory_writes", "Memory writes", "output"),
)

_REDUCTIONS: Final[dict[str, str]] = {
    "tool_schemas": (
        "Tool definitions are re-sent every turn. Cache the system block where the "
        "provider supports it, or split the roster so each agent only carries the "
        "tools its role uses — a router agent with 4 tools and a worker with 6 costs "
        "far less than two agents carrying all 10."
    ),
    "base_prompt": (
        "The system prompt is re-sent every turn and barely changes. Prompt caching "
        "is the single highest-leverage change available here."
    ),
    "memory_reads": (
        "Memory is being read back in full each step. Summarise older turns, or "
        "retrieve the two or three relevant entries instead of the whole history."
    ),
    "tool_results": (
        "Tool output is dominating input. Truncate or summarise results before they "
        "re-enter context — a 4,000-token API response usually has 200 useful tokens."
    ),
    "model_output": (
        "Output is the dominant line, which is the healthy case: the model is doing "
        "work rather than re-reading its own instructions. A smaller model for the "
        "mechanical steps is the remaining lever."
    ),
    "memory_writes": (
        "The agent is writing more state than it reads. Write summaries rather than transcripts."
    ),
}


def agent_cost(
    *,
    agents: list[AgentRole],
    tasks_per_day: int,
    input_tokens_per_step: int,
    output_tokens_per_step: int,
    tool_count: int,
    tokens_per_tool_schema: int = DEFAULT_TOKENS_PER_TOOL_SCHEMA,
    tool_calls_per_step: int = 1,
    tokens_per_tool_result: int = 400,
    memory_read_tokens: int = 0,
    memory_write_tokens: int = 0,
    retry_rate_pct: Decimal = Decimal(0),
    cached_input_ratio: Decimal = Decimal(0),
) -> ToolOutput:
    """What the loop actually costs, decomposed into the lines that drive it.

    The retry rate multiplies everything uniformly: a retried step re-sends the
    same context and re-generates a comparable response, so a 15% retry rate is
    15% more spend. Modelling it as anything more clever would be a precision
    the input does not support.

    Caching applies only to the two lines that repeat verbatim — the system
    prompt and the tool definitions. Applying it to tool results or memory,
    which differ every turn, would produce a discount that does not exist.
    """
    retry_multiplier = Decimal(1) + (retry_rate_pct / Decimal(100))
    cache_ratio = max(Decimal(0), min(Decimal(1), cached_input_ratio))

    per_step_tokens: dict[str, int] = {
        "base_prompt": input_tokens_per_step,
        "tool_schemas": tool_count * tokens_per_tool_schema,
        "memory_reads": memory_read_tokens,
        "tool_results": tool_calls_per_step * tokens_per_tool_result,
        "model_output": output_tokens_per_step,
        "memory_writes": memory_write_tokens,
    }
    cacheable = {"base_prompt", "tool_schemas"}

    component_cost: dict[str, Decimal] = dict.fromkeys(per_step_tokens, Decimal(0))
    component_tokens: dict[str, Decimal] = dict.fromkeys(per_step_tokens, Decimal(0))
    agent_rows: list[dict[str, Any]] = []
    warnings: list[ToolWarning] = []
    sourced_from: list[str] = []

    total_per_task = Decimal(0)

    for agent in agents:
        model = agent.model
        sourced_from.append(model.id)
        steps = Decimal(agent.steps_per_task) * Decimal(agent.count)

        agent_per_task = Decimal(0)
        for key, _label, side in _COMPONENTS:
            tokens = Decimal(per_step_tokens[key])
            if side == "input":
                rate = _effective_input_rate(model, cache_ratio if key in cacheable else Decimal(0))
            else:
                rate = model.output_cost_per_1k or Decimal(0)

            cost = tokens / THOUSAND * rate * steps
            component_cost[key] += cost
            component_tokens[key] += tokens * steps
            agent_per_task += cost

        total_per_task += agent_per_task
        agent_rows.append(
            {
                "role": agent.role,
                "model": model.display_name,
                "instances": agent.count,
                "steps_per_task": agent.steps_per_task,
                "cost_per_task": _usd(_money(agent_per_task * retry_multiplier)),
                "cost_per_month": _usd(
                    _display(
                        agent_per_task * retry_multiplier * Decimal(tasks_per_day) * DAYS_PER_MONTH
                    )
                ),
            }
        )

        if model.cached_input_cost_per_1k is None and cache_ratio > 0:
            warnings.append(
                ToolWarning(
                    level="warning",
                    field="cached_input_ratio",
                    message=(
                        f"{model.display_name} has no published cached-input rate, so no "
                        f"caching discount was applied to the {agent.role} agent. Its "
                        f"figures are an upper bound."
                    ),
                )
            )

    per_task = total_per_task * retry_multiplier
    per_day = per_task * Decimal(tasks_per_day)
    per_month = per_day * DAYS_PER_MONTH
    per_year = per_month * MONTHS_PER_YEAR

    retry_premium = (per_month - (total_per_task * Decimal(tasks_per_day) * DAYS_PER_MONTH)).max(
        Decimal(0)
    )

    input_cost = sum(
        (component_cost[key] for key, _, side in _COMPONENTS if side == "input"), Decimal(0)
    )
    schema_share = (
        component_cost["tool_schemas"] / input_cost * Decimal(100) if input_cost > 0 else Decimal(0)
    )

    biggest = max(component_cost.items(), key=lambda item: item[1])[0] if component_cost else ""
    labels = {key: label for key, label, _ in _COMPONENTS}

    breakdown = [
        {
            "line": label,
            "tokens_per_task": int(component_tokens[key]),
            "cost_per_task": _usd(_money(component_cost[key] * retry_multiplier)),
            "share_of_cost": (
                f"{_pct(component_cost[key] / total_per_task * Decimal(100))}%"
                if total_per_task > 0
                else "0%"
            ),
        }
        for key, label, _side in _COMPONENTS
        if component_tokens[key] > 0
    ]

    tokens_per_task = int(sum(component_tokens.values()))

    if tool_count >= 15:
        warnings.append(
            ToolWarning(
                level="warning",
                field="tool_count",
                message=(
                    f"{tool_count} tool definitions are re-sent on every one of the "
                    f"{sum(a.steps_per_task * a.count for a in agents)} steps in a task. "
                    f"That is {int(component_tokens['tool_schemas']):,} tokens per task "
                    f"before the agent has read anything."
                ),
            )
        )
    if retry_rate_pct == 0:
        warnings.append(
            ToolWarning(
                level="warning",
                field="retry_rate_pct",
                message=(
                    "A zero retry rate assumes no step ever fails validation, times out, "
                    "or produces an unparseable tool call. 10-20% is the usual range on a "
                    "loop with real tools, and it is spend, not an edge case."
                ),
            )
        )
    if biggest in _REDUCTIONS:
        warnings.append(
            ToolWarning(
                level="info",
                message=f"Largest line is {labels[biggest].lower()}. {_REDUCTIONS[biggest]}",
            )
        )

    return ToolOutput(
        metrics={
            "cost_per_task": _money(per_task),
            "cost_per_day": _display(per_day),
            "cost_per_month": _display(per_month),
            "cost_per_year": _display(per_year),
            "tokens_per_task": tokens_per_task,
            "schema_overhead_pct": _pct(schema_share),
            "retry_premium_monthly": _display(retry_premium),
            "biggest_contributor": labels.get(biggest, "—"),
        },
        tables={"breakdown": breakdown, "agents": agent_rows},
        warnings=warnings,
        sourced_from=sourced_from,
    )


def _effective_input_rate(model: ModelOut, cache_ratio: Decimal) -> Decimal:
    """Blended input rate for a line that is `cache_ratio` cache hits.

    A model with no published cached rate is charged in full. No published rate
    means caching cannot be priced, not that it is free.
    """
    cached = model.cached_input_cost_per_1k
    if cached is None or cache_ratio <= 0:
        return model.input_cost_per_1k
    return model.input_cost_per_1k * (Decimal(1) - cache_ratio) + cached * cache_ratio


# ── rate-limits ──────────────────────────────────────────────────────────────

SECONDS_PER_MINUTE: Final = Decimal(60)
MINUTES_PER_DAY: Final = Decimal(1440)


class _Constraint:
    __slots__ = ("headroom", "label", "limit", "note", "required", "unit")

    def __init__(self, *, label: str, required: Decimal, limit: Decimal, unit: str, note: str):
        self.label = label
        self.required = required
        self.limit = limit
        self.unit = unit
        self.note = note
        # An unlimited or unpublished ceiling is infinite headroom, not zero.
        self.headroom = limit / required if required > 0 else Decimal("Infinity")

    @property
    def utilisation_pct(self) -> Decimal:
        return self.required / self.limit * Decimal(100) if self.limit > 0 else Decimal(0)

    @property
    def status(self) -> str:
        if self.headroom < 1:
            return "over"
        if self.headroom < Decimal("1.25"):
            return "tight"
        return "ok"


def rate_limits(
    *,
    provider: ProviderLimits,
    tier: TierLimits,
    requests_per_min: int,
    input_tokens_per_request: int,
    output_tokens_per_request: int,
    concurrency: int,
    avg_request_seconds: Decimal,
    burst_multiplier: Decimal = Decimal(1),
    burst_duration_seconds: int = 60,
) -> ToolOutput:
    """Headroom against every published ceiling, and which one binds first.

    The client's own concurrency is scored as a constraint alongside the
    provider's, via Little's law: `concurrency / avg_request_seconds` is the
    ceiling your connection pool imposes regardless of what the provider
    allows. It binds before the provider's limits more often than not, and a
    tool that only checks the provider's numbers sends people to buy a tier
    upgrade that changes nothing.
    """
    rpm = Decimal(requests_per_min)
    input_tpm = rpm * Decimal(input_tokens_per_request)
    output_tpm = rpm * Decimal(output_tokens_per_request)

    constraints: list[_Constraint] = [
        _Constraint(
            label="Requests per minute",
            required=rpm,
            limit=Decimal(tier.requests_per_min),
            unit="req/min",
            note="The ceiling people plan against, and rarely the one that binds.",
        )
    ]

    if provider.combined_tokens:
        constraints.append(
            _Constraint(
                label="Tokens per minute",
                required=input_tpm + output_tpm,
                limit=Decimal(tier.input_tokens_per_min),
                unit="tok/min",
                note=f"{provider.label} meters input and output against one budget.",
            )
        )
    else:
        constraints.append(
            _Constraint(
                label="Input tokens per minute",
                required=input_tpm,
                limit=Decimal(tier.input_tokens_per_min),
                unit="tok/min",
                note="Long context and large tool rosters push this first.",
            )
        )
        constraints.append(
            _Constraint(
                label="Output tokens per minute",
                required=output_tpm,
                limit=Decimal(tier.output_tokens_per_min or 0),
                unit="tok/min",
                note=(
                    "Metered separately here, and the usual binding constraint on an "
                    "agent loop — loops emit far more output per minute than chat does."
                ),
            )
        )

    if tier.requests_per_day is not None:
        constraints.append(
            _Constraint(
                label="Requests per day",
                required=rpm * MINUTES_PER_DAY,
                limit=Decimal(tier.requests_per_day),
                unit="req/day",
                note="Assumes the rate is sustained. A daily cap kills long jobs overnight.",
            )
        )

    # Little's law. `concurrency` in flight, each held for `avg_request_seconds`,
    # completes `concurrency / seconds` per second.
    client_ceiling = (
        Decimal(concurrency) / avg_request_seconds * SECONDS_PER_MINUTE
        if avg_request_seconds > 0
        else Decimal("Infinity")
    )
    constraints.append(
        _Constraint(
            label="Client concurrency",
            required=rpm,
            limit=client_ceiling,
            unit="req/min",
            note=(
                f"{concurrency} in flight at {avg_request_seconds}s each. Self-imposed — "
                "no tier upgrade moves it."
            ),
        )
    )

    binding = min(constraints, key=lambda c: c.headroom)

    # Sustainable throughput is the tightest ceiling expressed back in requests.
    sustainable_rpm = min(
        (
            c.limit / (c.required / rpm) if c.required > 0 and rpm > 0 else Decimal("Infinity")
            for c in constraints
        ),
        default=Decimal(0),
    )

    peak_rpm = rpm * burst_multiplier
    drain_per_second = sustainable_rpm / SECONDS_PER_MINUTE
    arrival_per_second = peak_rpm / SECONDS_PER_MINUTE
    queue_depth = max(
        Decimal(0), (arrival_per_second - drain_per_second) * Decimal(burst_duration_seconds)
    )

    recommended, upgrade_note = _recommend_tier(
        provider=provider,
        current=tier,
        rpm=rpm,
        input_tpm=input_tpm,
        output_tpm=output_tpm,
    )

    warnings = _rate_limit_warnings(
        provider=provider,
        tier=tier,
        binding=binding,
        queue_depth=queue_depth,
        burst_multiplier=burst_multiplier,
    )

    constraint_rows = [
        {
            "constraint": c.label,
            "required": f"{_thousands(c.required)} {c.unit}",
            "limit": f"{_thousands(c.limit)} {c.unit}",
            "utilisation": f"{_pct(c.utilisation_pct)}%" if c.limit.is_finite() else "—",
            "status": c.status,
            "note": c.note,
        }
        for c in constraints
    ]

    return ToolOutput(
        metrics={
            "binding_constraint": binding.label,
            "headroom_pct": (
                _pct((binding.headroom - 1) * Decimal(100))
                if binding.headroom.is_finite()
                else "unbounded"
            ),
            "max_sustainable_rpm": (
                int(sustainable_rpm) if sustainable_rpm.is_finite() else "unbounded"
            ),
            "queue_depth": int(queue_depth.to_integral_value(rounding=ROUND_HALF_UP)),
            "recommended_tier": recommended.label if recommended else tier.label,
            "tier": tier.label,
        },
        tables={
            "constraints": constraint_rows,
            "backoff": _backoff_plan(binding),
            "tiers": _tier_rows(provider, rpm, input_tpm, output_tpm),
        },
        warnings=warnings + ([_upgrade_warning(upgrade_note)] if upgrade_note else []),
    )


def _thousands(value: Decimal) -> str:
    if not value.is_finite():
        return "unbounded"
    return f"{value.to_integral_value(rounding=ROUND_HALF_UP):,}"


def _recommend_tier(
    *,
    provider: ProviderLimits,
    current: TierLimits,
    rpm: Decimal,
    input_tpm: Decimal,
    output_tpm: Decimal,
) -> tuple[TierLimits | None, str]:
    """The cheapest tier that clears every provider ceiling.

    Client concurrency is deliberately excluded: no tier fixes it, and
    recommending an upgrade for a constraint the upgrade cannot move is how a
    tool loses the user's trust in one screen.
    """

    def clears(tier: TierLimits) -> bool:
        if rpm > tier.requests_per_min:
            return False
        if provider.combined_tokens:
            if input_tpm + output_tpm > tier.input_tokens_per_min:
                return False
        else:
            if input_tpm > tier.input_tokens_per_min:
                return False
            if output_tpm > Decimal(tier.output_tokens_per_min or 0):
                return False
        return not (
            tier.requests_per_day is not None and rpm * MINUTES_PER_DAY > tier.requests_per_day
        )

    if clears(current):
        return current, ""

    for tier in provider.tiers:
        if clears(tier):
            return tier, (
                f"This workload needs {tier.label} on {provider.label}. Qualifier: "
                f"{tier.qualifier}."
            )

    top = provider.tiers[-1]
    return None, (
        f"No published {provider.label} tier covers this workload — {top.label} is the "
        f"ceiling. This is a conversation with sales about a negotiated limit, or a "
        f"second account or provider to spread the load across."
    )


def _upgrade_warning(note: str) -> ToolWarning:
    return ToolWarning(level="warning", message=note)


def _backoff_plan(binding: _Constraint) -> list[dict[str, Any]]:
    """Concrete retry parameters, chosen for what is actually binding.

    A token-bound workload is not fixed by retrying: the request that failed
    will fail again until the budget window rolls. Pacing at the source is the
    answer there, and saying so is more useful than a generic backoff table
    that reads the same for every result.
    """
    token_bound = "tokens" in binding.label.lower()
    strategy = (
        "Token-bucket pacing, then exponential backoff"
        if token_bound
        else "Exponential backoff with full jitter"
    )

    rows = [
        {"parameter": "Strategy", "value": strategy},
        {"parameter": "Base delay", "value": "1s"},
        {"parameter": "Multiplier", "value": "2x"},
        {"parameter": "Max delay", "value": "60s"},
        {"parameter": "Jitter", "value": "Full — delay = random(0, computed)"},
        {"parameter": "Max attempts", "value": "5"},
        {
            "parameter": "Honour Retry-After",
            "value": "Yes — the header overrides the computed delay",
        },
    ]
    if token_bound:
        rows.append(
            {
                "parameter": "Pacing",
                "value": (
                    "Meter outbound tokens against the per-minute budget before sending. "
                    "Retrying a token-bound 429 just re-spends the budget you do not have."
                ),
            }
        )
    else:
        rows.append(
            {
                "parameter": "Note",
                "value": (
                    "Full jitter rather than equal jitter: retries from a burst must be "
                    "spread, and equal jitter keeps half the delay synchronised."
                ),
            }
        )
    return rows


def _tier_rows(
    provider: ProviderLimits, rpm: Decimal, input_tpm: Decimal, output_tpm: Decimal
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for tier in provider.tiers:
        if provider.combined_tokens:
            tokens = f"{tier.input_tokens_per_min:,} combined"
            fits = (
                rpm <= tier.requests_per_min and input_tpm + output_tpm <= tier.input_tokens_per_min
            )
        else:
            tokens = f"{tier.input_tokens_per_min:,} in / {tier.output_tokens_per_min or 0:,} out"
            fits = (
                rpm <= tier.requests_per_min
                and input_tpm <= tier.input_tokens_per_min
                and output_tpm <= Decimal(tier.output_tokens_per_min or 0)
            )
        if tier.requests_per_day is not None and rpm * MINUTES_PER_DAY > tier.requests_per_day:
            fits = False

        rows.append(
            {
                "tier": tier.label,
                "requests_per_min": f"{tier.requests_per_min:,}",
                "tokens_per_min": tokens,
                "qualifier": tier.qualifier,
                "fits": "yes" if fits else "no",
            }
        )
    return rows


def _rate_limit_warnings(
    *,
    provider: ProviderLimits,
    tier: TierLimits,
    binding: _Constraint,
    queue_depth: Decimal,
    burst_multiplier: Decimal,
) -> list[ToolWarning]:
    warnings: list[ToolWarning] = [
        ToolWarning(
            level="info",
            message=(
                f"Limits are {provider.label}'s published defaults for the "
                f"{provider.model_family}, verified {provider.verified_on.isoformat()} "
                f"against {provider.source_url}. Per-model and negotiated limits differ, "
                f"and tiers can change without notice — your dashboard is authoritative."
            ),
        )
    ]

    if binding.headroom < 1:
        warnings.append(
            ToolWarning(
                level="critical",
                message=(
                    f"{binding.label} is exceeded, not merely tight. At this volume "
                    f"requests are rejected from the first minute, so the backoff plan "
                    f"below is damage control rather than a fix."
                ),
            )
        )
    elif binding.headroom < Decimal("1.25"):
        warnings.append(
            ToolWarning(
                level="warning",
                message=(
                    f"{binding.label} has under 25% headroom. Normal variance in request "
                    f"size will cross it before your average volume does."
                ),
            )
        )

    if binding.label == "Client concurrency":
        warnings.append(
            ToolWarning(
                level="warning",
                message=(
                    "The binding constraint is your own concurrency, not the provider's "
                    "limits. Raising the tier changes nothing here — raise in-flight "
                    "requests, or cut per-request latency."
                ),
            )
        )

    if queue_depth > 0:
        warnings.append(
            ToolWarning(
                level="warning",
                message=(
                    f"A {burst_multiplier}x burst arrives faster than the account can "
                    f"drain it. You need somewhere to hold roughly "
                    f"{int(queue_depth.to_integral_value(rounding=ROUND_HALF_UP)):,} "
                    f"requests, or they become errors at the edge."
                ),
            )
        )

    if tier.requests_per_day is not None:
        warnings.append(
            ToolWarning(
                level="info",
                message=(
                    f"{tier.label} carries a daily cap of {tier.requests_per_day:,} "
                    f"requests. Per-minute headroom says nothing about surviving a "
                    f"full day at this rate."
                ),
            )
        )

    return warnings


# ── function-schema ──────────────────────────────────────────────────────────

TARGETS: Final = ("openai", "anthropic", "json-schema", "mcp")

JSON_TYPES: Final = ("string", "number", "integer", "boolean", "array", "object")

#: Provider-side name rules. Both OpenAI and Anthropic accept the same shape,
#: and a name that violates it fails at the API rather than at generation time
#: unless something checks here.
NAME_PATTERN: Final = re.compile(r"^[a-zA-Z0-9_-]{1,64}$")

#: Parameter names that describe the container rather than the content. A model
#: choosing what to put in `data` is guessing, and guessing is where tool
#: calling goes wrong.
VAGUE_NAMES: Final = frozenset(
    {"data", "info", "input", "value", "item", "object", "params", "args", "thing", "payload"}
)

#: The envelope each target expects, expressed as a schema so the generated
#: output is validated rather than inspected. `parameters` / `input_schema` are
#: themselves JSON Schema and are checked separately with `check_schema`.
_ENVELOPES: Final[dict[str, dict[str, Any]]] = {
    "openai": {
        "type": "object",
        "properties": {
            "type": {"const": "function"},
            "function": {
                "type": "object",
                "properties": {
                    "name": {"type": "string", "pattern": NAME_PATTERN.pattern},
                    "description": {"type": "string", "minLength": 1},
                    "parameters": {"type": "object"},
                    "strict": {"type": "boolean"},
                },
                "required": ["name", "description", "parameters"],
            },
        },
        "required": ["type", "function"],
    },
    "anthropic": {
        "type": "object",
        "properties": {
            "name": {"type": "string", "pattern": NAME_PATTERN.pattern},
            "description": {"type": "string", "minLength": 1},
            "input_schema": {
                "type": "object",
                "properties": {"type": {"const": "object"}},
                "required": ["type", "properties"],
            },
        },
        "required": ["name", "description", "input_schema"],
    },
    "mcp": {
        "type": "object",
        "properties": {
            "name": {"type": "string", "pattern": NAME_PATTERN.pattern},
            "description": {"type": "string", "minLength": 1},
            "inputSchema": {
                "type": "object",
                "properties": {"type": {"const": "object"}},
                "required": ["type", "properties"],
            },
        },
        "required": ["name", "description", "inputSchema"],
    },
    "json-schema": {
        "type": "object",
        "properties": {
            "title": {"type": "string"},
            "description": {"type": "string"},
            "type": {"const": "object"},
            "properties": {"type": "object"},
        },
        "required": ["title", "type", "properties"],
    },
}


def json_schema_for(tool: dict[str, Any]) -> dict[str, Any]:
    """The parameter object, in plain JSON Schema.

    Shared by every target, because all four of them wrap the same thing.
    """
    properties: dict[str, Any] = {}
    required: list[str] = []

    for param in tool.get("parameters", []):
        name = str(param["name"]).strip()
        schema: dict[str, Any] = {"type": param.get("type", "string")}
        description = str(param.get("description") or "").strip()
        if description:
            schema["description"] = description

        enum = [value for value in (param.get("enum") or []) if str(value).strip()]
        if enum:
            schema["enum"] = enum
        if schema["type"] == "array":
            # An array with no item type tells the model nothing about what to
            # put in it, and providers reject the schema outright.
            schema["items"] = {"type": param.get("item_type") or "string"}

        properties[name] = schema
        if param.get("required", True):
            required.append(name)

    return {
        "type": "object",
        "properties": properties,
        "required": required,
        # Rejecting unknown keys is what turns a hallucinated argument into a
        # validation error instead of a silently ignored one.
        "additionalProperties": False,
    }


def function_schema(*, tools: list[dict[str, Any]], target: str) -> ToolOutput:
    """Generate schemas in the requested format, and validate them there.

    Validation is against the real thing: the parameter object goes through
    `Draft202012Validator.check_schema`, and the wrapper goes through a schema
    describing the provider's envelope. "Looks like JSON Schema" is the check
    that lets an invalid tool definition reach production, where it fails with
    an opaque provider error days later.
    """
    warnings: list[ToolWarning] = []
    generated: list[dict[str, Any]] = []
    rows: list[dict[str, Any]] = []
    seen: set[str] = set()

    for tool in tools:
        raw_name = str(tool.get("name") or "").strip()
        name = raw_name if NAME_PATTERN.match(raw_name) else _safe_name(raw_name)
        if name != raw_name:
            warnings.append(
                ToolWarning(
                    level="warning",
                    message=(
                        f"Tool name {raw_name!r} is not accepted by the provider "
                        f"(letters, digits, underscore, and hyphen only, up to 64 "
                        f"characters). Emitted as {name!r}."
                    ),
                )
            )
        if name in seen:
            warnings.append(
                ToolWarning(
                    level="critical",
                    message=(
                        f"Two tools are both named {name!r}. The provider takes the last "
                        f"one and the model cannot reach the other."
                    ),
                )
            )
        seen.add(name)

        description = str(tool.get("description") or "").strip()
        parameters = json_schema_for(tool)
        warnings.extend(_parameter_warnings(name, tool, description))

        envelope = _wrap(target, name=name, description=description, parameters=parameters)
        error = _validate(target, envelope, parameters)
        if error:
            warnings.append(
                ToolWarning(
                    level="critical",
                    message=f"{name}: generated schema failed validation — {error}",
                )
            )

        generated.append(envelope)
        rows.append(
            {
                "tool": name,
                "parameters": len(parameters["properties"]),
                "required": len(parameters["required"]),
                "valid": "no" if error else "yes",
            }
        )

    if target == "openai" and any(
        len(item["function"]["parameters"]["required"])
        < len(item["function"]["parameters"]["properties"])
        for item in generated
    ):
        warnings.append(
            ToolWarning(
                level="warning",
                message=(
                    "OpenAI structured outputs (`strict: true`) require every property to "
                    "be listed in `required`. Optional parameters must be expressed as a "
                    "nullable type instead — `strict` is emitted as false where that is "
                    "not the case."
                ),
            )
        )

    if len(tools) > 20:
        warnings.append(
            ToolWarning(
                level="warning",
                message=(
                    f"{len(tools)} tool definitions are re-sent on every turn. Past about "
                    f"20, selection accuracy drops and the definitions start to dominate "
                    f"input cost — check the agent cost calculator."
                ),
            )
        )

    body = json.dumps(generated, indent=2, ensure_ascii=False)
    return ToolOutput(
        metrics={
            "tools": len(generated),
            "target": target,
            "parameters": sum(row["parameters"] for row in rows),
            "valid": "yes" if all(row["valid"] == "yes" for row in rows) else "no",
        },
        tables={"tools": rows},
        artifacts=[
            Artifact(
                type="schema",
                format="json",
                filename=f"tools.{target}.json",
                content=body,
                language="json",
            )
        ],
        warnings=warnings,
    )


def _safe_name(raw: str) -> str:
    """A provider-acceptable name, derived rather than rejected.

    Rejecting is defensible, but the user typed a title with a space in it and
    meant a tool name; deriving one and saying so gets them a working schema
    with the correction visible.
    """
    cleaned = re.sub(r"[^a-zA-Z0-9_-]+", "_", raw).strip("_")[:64]
    return cleaned or "unnamed_tool"


def _wrap(
    target: str, *, name: str, description: str, parameters: dict[str, Any]
) -> dict[str, Any]:
    match target:
        case "openai":
            strict = len(parameters["required"]) == len(parameters["properties"])
            return {
                "type": "function",
                "function": {
                    "name": name,
                    "description": description,
                    "parameters": parameters,
                    "strict": strict,
                },
            }
        case "anthropic":
            return {"name": name, "description": description, "input_schema": parameters}
        case "mcp":
            return {"name": name, "description": description, "inputSchema": parameters}
        case _:
            return {
                "$schema": "https://json-schema.org/draft/2020-12/schema",
                "title": name,
                "description": description,
                **parameters,
            }


def _validate(target: str, envelope: dict[str, Any], parameters: dict[str, Any]) -> str | None:
    """`None` when valid, else the reason."""
    try:
        Draft202012Validator.check_schema(parameters)
    except SchemaError as exc:
        return f"parameter schema is not valid JSON Schema: {exc.message}"

    meta = _ENVELOPES.get(target)
    if meta is None:
        return f"unknown target {target!r}"
    try:
        Draft202012Validator(meta).validate(envelope)
    except ValidationError as exc:
        return f"does not match the {target} tool format: {exc.message}"
    return None


def _parameter_warnings(name: str, tool: dict[str, Any], description: str) -> list[ToolWarning]:
    """Everything a model would have to guess at.

    These are the failures that do not raise: the schema is valid, the provider
    accepts it, and the model picks the wrong argument because nothing told it
    what the parameter was for.
    """
    warnings: list[ToolWarning] = []

    if len(description) < 15:
        warnings.append(
            ToolWarning(
                level="warning",
                message=(
                    f"{name} has little or no description. The description is what the "
                    f"model selects on — an unclear one produces a tool that is called "
                    f"at the wrong moments, which reads as a model problem."
                ),
            )
        )

    parameters = tool.get("parameters", [])
    if not parameters:
        warnings.append(
            ToolWarning(
                level="info",
                message=f"{name} takes no parameters. Correct for some tools, worth a check.",
            )
        )

    for param in parameters:
        param_name = str(param.get("name") or "").strip()
        param_description = str(param.get("description") or "").strip()

        if param_name.lower() in VAGUE_NAMES:
            warnings.append(
                ToolWarning(
                    level="warning",
                    message=(
                        f"{name}.{param_name} names the container, not the content. The "
                        f"model has to infer what belongs in it."
                    ),
                )
            )
        if not param_description:
            warnings.append(
                ToolWarning(
                    level="warning",
                    message=(
                        f"{name}.{param_name} has no description. Type alone does not say "
                        f"what a string should contain, what units a number is in, or what "
                        f"format a date takes."
                    ),
                )
            )
        if (
            param.get("type") == "string"
            and not param.get("enum")
            and re.search(
                r"\b(one of|either|options?|choices?)\b", param_description, re.IGNORECASE
            )
        ):
            warnings.append(
                ToolWarning(
                    level="info",
                    message=(
                        f"{name}.{param_name} describes a fixed set of values but has no "
                        f"enum. An enum is enforced; a description is a suggestion."
                    ),
                )
            )

    return warnings
