"""Agent economics, rate-limit headroom, and schema generation (WF3).

Every expected number below was worked out by hand before the implementation
was consulted. The two that matter most are schema overhead and retries: they
are the lines a naive agent calculator omits, and omitting them understates
real spend by a multiple rather than a percent.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime
from decimal import Decimal

import pytest
from jsonschema import Draft202012Validator

from app.data import rate_limits as limits
from app.schemas.catalog import ModelOut, ProvenanceOut
from app.services.agent_service import (
    AgentRole,
    agent_cost,
    function_schema,
    json_schema_for,
    rate_limits,
)


def _model(
    model_id: str = "test-model",
    *,
    input_per_1k: str = "0.001",
    output_per_1k: str = "0.005",
    cached_per_1k: str | None = None,
) -> ModelOut:
    return ModelOut(
        id=f"mdl_{model_id}",
        provider="test",
        model_id=model_id,
        display_name=model_id,
        family="chat",
        input_cost_per_1k=Decimal(input_per_1k),
        output_cost_per_1k=Decimal(output_per_1k),
        cached_input_cost_per_1k=Decimal(cached_per_1k) if cached_per_1k else None,
        context_window=200_000,
        status="active",
        provenance=ProvenanceOut(
            last_verified_at=datetime(2026, 8, 1, tzinfo=UTC),
            age_days=9,
            variant="fresh",
            source_name="Test",
            source_url="https://example.com",
            source_kind="manual",
        ),
    )


def _roster(count: int = 2, steps: int = 4) -> list[AgentRole]:
    return [
        AgentRole(role=f"agent-{index}", model=_model(), count=1, steps_per_task=steps)
        for index in range(count)
    ]


# ── agent-cost ───────────────────────────────────────────────────────────────


def test_agent_cost_matches_a_hand_computed_profile() -> None:
    # Per step, per agent:
    #   input  = 1,000 base + (10 tools x 100) schema + (1 call x 500) result
    #          = 2,500 tokens @ $0.001/1k = $0.0025
    #   output =   500 tokens            @ $0.005/1k = $0.0025
    #   step   = $0.005
    # 4 steps x 2 agents          = $0.04 per task
    # x 100 tasks/day             = $4.00 per day
    # x 30.4375 days              = $121.75 per month
    result = agent_cost(
        agents=_roster(),
        tasks_per_day=100,
        input_tokens_per_step=1_000,
        output_tokens_per_step=500,
        tool_count=10,
        tokens_per_tool_schema=100,
        tool_calls_per_step=1,
        tokens_per_tool_result=500,
        retry_rate_pct=Decimal(0),
    )

    assert result.metrics["cost_per_task"] == Decimal("0.040000")
    assert result.metrics["cost_per_day"] == Decimal("4.00")
    assert result.metrics["cost_per_month"] == Decimal("121.75")
    # 3,000 tokens/step x 4 steps x 2 agents.
    assert result.metrics["tokens_per_task"] == 24_000


def test_schema_overhead_is_a_visible_line_not_a_footnote() -> None:
    result = agent_cost(
        agents=_roster(),
        tasks_per_day=100,
        input_tokens_per_step=1_000,
        output_tokens_per_step=500,
        tool_count=10,
        tokens_per_tool_schema=100,
        tool_calls_per_step=1,
        tokens_per_tool_result=500,
        retry_rate_pct=Decimal(0),
    )

    schema_line = next(
        row for row in result.tables["breakdown"] if row["line"] == "Tool definitions"
    )
    # 10 tools x 100 tokens, re-sent on all 8 steps in the task.
    assert schema_line["tokens_per_task"] == 8_000
    assert schema_line["cost_per_task"] == "$0.008"
    # 1,000 of the 2,500 input tokens per step are tool definitions.
    assert result.metrics["schema_overhead_pct"] == Decimal("40.0")


def test_a_twenty_percent_retry_rate_costs_exactly_twenty_percent_more() -> None:
    kwargs = {
        "agents": _roster(),
        "tasks_per_day": 100,
        "input_tokens_per_step": 1_000,
        "output_tokens_per_step": 500,
        "tool_count": 10,
        "tokens_per_tool_schema": 100,
        "tool_calls_per_step": 1,
        "tokens_per_tool_result": 500,
    }
    baseline = agent_cost(**kwargs, retry_rate_pct=Decimal(0))  # type: ignore[arg-type]
    retried = agent_cost(**kwargs, retry_rate_pct=Decimal(20))  # type: ignore[arg-type]

    assert Decimal(str(retried.metrics["cost_per_month"])) == Decimal(
        str(baseline.metrics["cost_per_month"])
    ) * Decimal("1.2")
    assert retried.metrics["cost_per_month"] == Decimal("146.10")
    # The premium is reported separately, because "your retries cost $24/month"
    # is the sentence that gets someone to go and measure their retry rate.
    assert retried.metrics["retry_premium_monthly"] == Decimal("24.35")


def test_caching_applies_only_to_the_lines_that_repeat_verbatim() -> None:
    """A discount on tool results or memory would be a discount that does not
    exist — those differ every turn and are never a cache hit."""
    cached_model = _model(cached_per_1k="0.0001")
    agents = [AgentRole(role="solo", model=cached_model, count=1, steps_per_task=1)]

    kwargs = {
        "agents": agents,
        "tasks_per_day": 1,
        "input_tokens_per_step": 1_000,
        "output_tokens_per_step": 0,
        "tool_count": 10,
        "tokens_per_tool_schema": 100,
        "tool_calls_per_step": 1,
        "tokens_per_tool_result": 1_000,
    }
    full = agent_cost(**kwargs, cached_input_ratio=Decimal(0))  # type: ignore[arg-type]
    cached = agent_cost(**kwargs, cached_input_ratio=Decimal(1))  # type: ignore[arg-type]

    # 3,000 input tokens: 1,000 base + 1,000 schemas (both cacheable) and
    # 1,000 of tool results (not). At full rate $0.003; fully cached the first
    # two drop to a tenth, leaving 0.0001+0.0001+0.001 = $0.0012.
    assert full.metrics["cost_per_task"] == Decimal("0.003000")
    assert cached.metrics["cost_per_task"] == Decimal("0.001200")


def test_an_uncacheable_model_is_charged_in_full_and_says_so() -> None:
    result = agent_cost(
        agents=[AgentRole(role="solo", model=_model(), count=1, steps_per_task=1)],
        tasks_per_day=1,
        input_tokens_per_step=1_000,
        output_tokens_per_step=0,
        tool_count=0,
        tool_calls_per_step=0,
        cached_input_ratio=Decimal("0.9"),
    )

    assert result.metrics["cost_per_task"] == Decimal("0.001000")
    assert any("no published cached-input rate" in w.message for w in result.warnings)


def test_a_large_tool_roster_is_flagged_against_the_step_count() -> None:
    result = agent_cost(
        agents=_roster(),
        tasks_per_day=100,
        input_tokens_per_step=1_000,
        output_tokens_per_step=500,
        tool_count=25,
        retry_rate_pct=Decimal(10),
    )

    assert any("re-sent on every one of the" in w.message for w in result.warnings)


def test_zero_retries_is_challenged_rather_than_accepted() -> None:
    result = agent_cost(
        agents=_roster(),
        tasks_per_day=10,
        input_tokens_per_step=100,
        output_tokens_per_step=100,
        tool_count=1,
        retry_rate_pct=Decimal(0),
    )

    assert any(w.field == "retry_rate_pct" for w in result.warnings)


# ── rate-limits ──────────────────────────────────────────────────────────────


def _tier(provider: limits.ProviderLimits, key: str) -> limits.TierLimits:
    tier = provider.tier(key)
    assert tier is not None
    return tier


def test_a_token_bound_profile_reports_tokens_not_requests() -> None:
    # OpenAI tier 2: 5,000 RPM and 450,000 TPM combined.
    # 100 req/min x 4,800 tokens = 480,000 TPM — over, while RPM is at 2%.
    result = rate_limits(
        provider=limits.OPENAI,
        tier=_tier(limits.OPENAI, "tier-2"),
        requests_per_min=100,
        input_tokens_per_request=4_000,
        output_tokens_per_request=800,
        concurrency=16,
        avg_request_seconds=Decimal(4),
    )

    assert result.metrics["binding_constraint"] == "Tokens per minute"
    assert result.metrics["recommended_tier"] == "Tier 3"


def test_output_tokens_bind_first_where_they_are_metered_separately() -> None:
    """The Anthropic case, and the reason input and output are not folded into
    one number: an agent loop emits far more output per minute than chat does."""
    # Tier 1: 50 RPM, 30k input, 8k output.
    # 10 req/min x 2,000 in = 20,000 (fits) and x 1,000 out = 10,000 (does not).
    result = rate_limits(
        provider=limits.ANTHROPIC,
        tier=_tier(limits.ANTHROPIC, "tier-1"),
        requests_per_min=10,
        input_tokens_per_request=2_000,
        output_tokens_per_request=1_000,
        concurrency=16,
        avg_request_seconds=Decimal(4),
    )

    assert result.metrics["binding_constraint"] == "Output tokens per minute"
    assert result.metrics["recommended_tier"] == "Tier 2"

    rows = {row["constraint"]: row for row in result.tables["constraints"]}
    assert rows["Input tokens per minute"]["status"] == "ok"
    assert rows["Output tokens per minute"]["status"] == "over"


def test_the_clients_own_concurrency_is_scored_as_a_constraint() -> None:
    # 2 in flight at 10s each is 12 req/min, whatever the provider allows.
    result = rate_limits(
        provider=limits.ANTHROPIC,
        tier=_tier(limits.ANTHROPIC, "tier-4"),
        requests_per_min=60,
        input_tokens_per_request=1_000,
        output_tokens_per_request=200,
        concurrency=2,
        avg_request_seconds=Decimal(10),
    )

    assert result.metrics["binding_constraint"] == "Client concurrency"
    # No tier upgrade moves a self-imposed ceiling, so none is recommended.
    assert result.metrics["recommended_tier"] == "Tier 4"
    assert any("Raising the tier changes nothing" in w.message for w in result.warnings)


def test_a_token_bound_result_recommends_pacing_over_retrying() -> None:
    result = rate_limits(
        provider=limits.OPENAI,
        tier=_tier(limits.OPENAI, "tier-1"),
        requests_per_min=100,
        input_tokens_per_request=4_000,
        output_tokens_per_request=800,
        concurrency=32,
        avg_request_seconds=Decimal(4),
    )

    strategy = next(row for row in result.tables["backoff"] if row["parameter"] == "Strategy")
    assert "pacing" in strategy["value"].lower()


def test_a_burst_that_cannot_drain_reports_the_queue_depth() -> None:
    # Sustainable is 50 req/min (1 tok/req, so RPM binds); a 4x burst on 50 is
    # 200/min for 60s, and 150 of those minutes-worth have nowhere to go.
    result = rate_limits(
        provider=limits.ANTHROPIC,
        tier=_tier(limits.ANTHROPIC, "tier-1"),
        requests_per_min=50,
        input_tokens_per_request=1,
        output_tokens_per_request=1,
        concurrency=1_000,
        avg_request_seconds=Decimal(1),
        burst_multiplier=Decimal(4),
        burst_duration_seconds=60,
    )

    assert result.metrics["queue_depth"] == 150


def test_provenance_of_the_limits_is_stated_on_every_run() -> None:
    result = rate_limits(
        provider=limits.GOOGLE,
        tier=_tier(limits.GOOGLE, "tier-1"),
        requests_per_min=10,
        input_tokens_per_request=1_000,
        output_tokens_per_request=200,
        concurrency=8,
        avg_request_seconds=Decimal(2),
    )

    assert any(limits.GOOGLE.source_url in w.message for w in result.warnings)


def test_every_published_tier_is_reachable_by_key() -> None:
    """Guards the frontend's tier dropdown against a key that does not resolve."""
    for provider in limits.PROVIDERS:
        for tier in provider.tiers:
            assert provider.tier(tier.key) is tier


# ── function-schema ──────────────────────────────────────────────────────────

SAMPLE_TOOL = {
    "name": "search_orders",
    "description": "Find orders for a customer within a date range.",
    "parameters": [
        {
            "name": "customer_id",
            "type": "string",
            "description": "The customer's id, as returned by list_customers.",
            "required": True,
        },
        {
            "name": "status",
            "type": "string",
            "description": "One of: open, shipped, cancelled.",
            "required": True,
            "enum": ["open", "shipped", "cancelled"],
        },
    ],
}


@pytest.mark.parametrize("target", ["openai", "anthropic", "json-schema", "mcp"])
def test_generated_output_validates_against_its_target_format(target: str) -> None:
    result = function_schema(tools=[SAMPLE_TOOL], target=target)

    assert result.metrics["valid"] == "yes"
    assert not [w for w in result.warnings if w.level == "critical"]

    emitted = json.loads(result.artifacts[0].content)
    assert len(emitted) == 1

    # And the parameter object is real JSON Schema, not something shaped like it.
    parameters = json_schema_for(SAMPLE_TOOL)
    Draft202012Validator.check_schema(parameters)
    Draft202012Validator(parameters).validate({"customer_id": "c_1", "status": "open"})


def test_the_wrapper_key_differs_per_provider() -> None:
    openai = json.loads(function_schema(tools=[SAMPLE_TOOL], target="openai").artifacts[0].content)
    anthropic = json.loads(
        function_schema(tools=[SAMPLE_TOOL], target="anthropic").artifacts[0].content
    )
    mcp = json.loads(function_schema(tools=[SAMPLE_TOOL], target="mcp").artifacts[0].content)

    assert openai[0]["type"] == "function"
    assert openai[0]["function"]["name"] == "search_orders"
    assert anthropic[0]["input_schema"]["type"] == "object"
    assert mcp[0]["inputSchema"]["type"] == "object"


def test_an_enum_is_enforced_by_the_schema() -> None:
    parameters = json_schema_for(SAMPLE_TOOL)
    validator = Draft202012Validator(parameters)

    assert list(validator.iter_errors({"customer_id": "c_1", "status": "open"})) == []
    assert list(validator.iter_errors({"customer_id": "c_1", "status": "invented"}))


def test_unknown_arguments_are_rejected_rather_than_ignored() -> None:
    validator = Draft202012Validator(json_schema_for(SAMPLE_TOOL))
    errors = list(
        validator.iter_errors({"customer_id": "c_1", "status": "open", "hallucinated": 1})
    )
    assert errors


def test_an_optional_parameter_turns_off_openai_strict_mode() -> None:
    tool = {
        "name": "fetch",
        "description": "Fetch a record by id, optionally including its history.",
        "parameters": [
            {"name": "record_id", "type": "string", "description": "The id.", "required": True},
            {
                "name": "include_history",
                "type": "boolean",
                "description": "Include past revisions.",
                "required": False,
            },
        ],
    }
    result = function_schema(tools=[tool], target="openai")
    emitted = json.loads(result.artifacts[0].content)

    assert emitted[0]["function"]["strict"] is False
    assert any("strict" in w.message for w in result.warnings)


def test_a_name_the_provider_would_reject_is_derived_and_reported() -> None:
    result = function_schema(
        tools=[{"name": "Search Orders!", "description": "Find orders.", "parameters": []}],
        target="anthropic",
    )
    emitted = json.loads(result.artifacts[0].content)

    assert emitted[0]["name"] == "Search_Orders"
    assert any("not accepted by the provider" in w.message for w in result.warnings)
    assert result.metrics["valid"] == "yes"


def test_ambiguity_a_valid_schema_cannot_express_is_warned_about() -> None:
    result = function_schema(
        tools=[
            {
                "name": "process",
                "description": "Do it.",
                "parameters": [
                    {"name": "data", "type": "string", "description": "", "required": True},
                    {
                        "name": "mode",
                        "type": "string",
                        "description": "Either fast or careful.",
                        "required": True,
                    },
                ],
            }
        ],
        target="anthropic",
    )
    messages = " ".join(w.message for w in result.warnings)

    assert result.metrics["valid"] == "yes"  # valid, and still a bad tool definition
    assert "names the container" in messages
    assert "has no description" in messages
    assert "no enum" in messages
    assert "little or no description" in messages


def test_duplicate_tool_names_are_a_critical_finding() -> None:
    tool = {"name": "search", "description": "Search things properly.", "parameters": []}
    result = function_schema(tools=[tool, dict(tool)], target="anthropic")

    assert any(w.level == "critical" for w in result.warnings)


def test_an_array_parameter_always_declares_its_item_type() -> None:
    """A provider rejects an array schema with no `items`, and a model given
    one has no idea what to put in the list."""
    parameters = json_schema_for(
        {
            "name": "tag",
            "parameters": [
                {"name": "labels", "type": "array", "description": "Labels.", "required": True}
            ],
        }
    )

    assert parameters["properties"]["labels"]["items"] == {"type": "string"}
    Draft202012Validator.check_schema(parameters)
