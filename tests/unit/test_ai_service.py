"""The AI layer's guarantees.

The module's whole claim is that a model failure is never a request failure,
so most of this file is failure modes. Each one asserts the same two things:
`generate_json` returned `None`, and no exception escaped.

The pricing tests assert exact dollars against the internal rate table, by
hand. A cost calculation that only agrees with itself is not checked.
"""

from __future__ import annotations

from decimal import Decimal
from types import SimpleNamespace
from typing import Any

import anthropic
import httpx
import pytest
from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.api.deps import Identity
from app.core.config import settings
from app.models.ai import AiCall, AiOutcome
from app.models.billing import Metric
from app.models.user import Plan
from app.services import ai_pricing, ai_prompts, ai_service
from tests.conftest import set_limit


@pytest.fixture(autouse=True)
def _reset_client():
    ai_service.set_client(None)
    yield
    ai_service.set_client(None)


def _identity() -> Identity:
    return Identity(user=None, anonymous_id="anon_test", session_id=None)


class _FakeMessages:
    """Stands in for `client.messages`, with a scripted outcome."""

    def __init__(self, *, response: Any = None, error: Exception | None = None) -> None:
        self._response = response
        self._error = error
        self.calls: list[dict[str, Any]] = []

    async def create(self, **kwargs: Any) -> Any:
        self.calls.append(kwargs)
        if self._error is not None:
            raise self._error
        return self._response


def _client(*, response: Any = None, error: Exception | None = None) -> Any:
    return SimpleNamespace(messages=_FakeMessages(response=response, error=error))


def _response(
    text: str = '{"summary": "ok", "why": "because", "weakest_link": "a", "watch_out_for": []}',
    *,
    stop_reason: str = "end_turn",
    input_tokens: int = 1000,
    output_tokens: int = 200,
    cache_read: int = 0,
    cache_write: int = 0,
) -> Any:
    return SimpleNamespace(
        content=[SimpleNamespace(type="text", text=text)],
        stop_reason=stop_reason,
        stop_details=SimpleNamespace(category="cyber"),
        usage=SimpleNamespace(
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            cache_read_input_tokens=cache_read,
            cache_creation_input_tokens=cache_write,
        ),
    )


async def _generate(db: AsyncSession, purpose: str = "agent_plan") -> Any:
    return await ai_service.generate_json(
        db,
        purpose=purpose,
        grounding={"metrics": {}},
        variables={"goal": "test"},
        identity=_identity(),
        tool_slug="workflow-plan",
    )


async def _outcomes(db: AsyncSession) -> list[AiOutcome]:
    rows = (await db.execute(select(AiCall.outcome).order_by(AiCall.created_at))).scalars().all()
    return list(rows)


# ── the None contract ────────────────────────────────────────────────────────


async def test_no_key_returns_none_without_a_network_call(
    db: AsyncSession, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The property that keeps local development a one-hour setup.

    Forced rather than assumed: a developer with a key in their `.env` must
    see the same result as CI, or this degradation path is only ever exercised
    on one of the two machines.
    """
    monkeypatch.setattr(settings, "anthropic_api_key", "")
    assert settings.ai_enabled is False

    assert await _generate(db) is None
    assert await _outcomes(db) == [AiOutcome.DISABLED]


_REQUEST = httpx.Request("POST", "https://api.anthropic.com/v1/messages")


@pytest.mark.parametrize(
    ("error", "expected"),
    [
        (anthropic.APITimeoutError(request=_REQUEST), AiOutcome.TIMEOUT),
        (
            anthropic.RateLimitError(
                "429", response=httpx.Response(429, request=_REQUEST), body=None
            ),
            AiOutcome.RATE_LIMITED,
        ),
        (anthropic.APIConnectionError(request=_REQUEST), AiOutcome.API_ERROR),
        (
            anthropic.InternalServerError(
                "500", response=httpx.Response(500, request=_REQUEST), body=None
            ),
            AiOutcome.API_ERROR,
        ),
        (RuntimeError("something nobody predicted"), AiOutcome.API_ERROR),
    ],
)
async def test_every_failure_returns_none_and_is_recorded(
    db: AsyncSession, monkeypatch: pytest.MonkeyPatch, error: Exception, expected: AiOutcome
) -> None:
    monkeypatch.setattr(settings, "anthropic_api_key", "sk-test")
    ai_service.set_client(_client(error=error))

    assert await _generate(db) is None
    assert await _outcomes(db) == [expected]


async def test_a_refusal_returns_none_rather_than_raising(
    db: AsyncSession, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A declined request is a successful HTTP 200 with an empty content list.
    Reading `content[0]` unconditionally is how that becomes an exception."""
    monkeypatch.setattr(settings, "anthropic_api_key", "sk-test")
    refused = _response(stop_reason="refusal")
    refused.content = []
    ai_service.set_client(_client(response=refused))

    assert await _generate(db) is None
    assert await _outcomes(db) == [AiOutcome.REFUSAL]


async def test_malformed_output_degrades_instead_of_throwing(
    db: AsyncSession, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(settings, "anthropic_api_key", "sk-test")
    ai_service.set_client(_client(response=_response("this is not json at all")))

    assert await _generate(db) is None
    assert await _outcomes(db) == [AiOutcome.INVALID_OUTPUT]


async def test_truncated_json_degrades(db: AsyncSession, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(settings, "anthropic_api_key", "sk-test")
    ai_service.set_client(_client(response=_response('{"summary": "cut off mid')))

    assert await _generate(db) is None


async def test_exhausted_quota_returns_none_without_calling_the_model(
    db: AsyncSession, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(settings, "anthropic_api_key", "sk-test")
    client = _client(response=_response())
    ai_service.set_client(client)
    await set_limit(db, plan=Plan.FREE, metric=Metric.AI_CALLS_PER_DAY, value=0, anonymous=True)

    assert await _generate(db) is None
    assert client.messages.calls == []
    assert await _outcomes(db) == [AiOutcome.QUOTA_EXCEEDED]


# ── the request shape ────────────────────────────────────────────────────────


async def test_the_request_obeys_this_model_familys_rules(
    db: AsyncSession, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Each of these is a 400 on the current models, and each is easy to
    reintroduce from memory — `budget_tokens` and `temperature` in particular
    read as normal API usage."""
    monkeypatch.setattr(settings, "anthropic_api_key", "sk-test")
    client = _client(response=_response())
    ai_service.set_client(client)

    await _generate(db)
    sent = client.messages.calls[0]

    assert sent["thinking"] == {"type": "adaptive"}
    assert "budget_tokens" not in sent
    assert not {"temperature", "top_p", "top_k"} & set(sent)
    # No assistant prefill: the last turn must be the user's.
    assert [message["role"] for message in sent["messages"]] == ["user"]
    # Structured output, not "please reply with JSON".
    assert sent["output_config"]["format"]["type"] == "json_schema"
    assert sent["output_config"]["effort"] in {"low", "medium", "high", "xhigh", "max"}


async def test_the_system_prompt_is_marked_cacheable_and_never_varies(
    db: AsyncSession, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The stable half carries the breakpoint; the varying half sits after it.
    A variable interpolated into the system text would put it ahead of the
    breakpoint and silently make every request a cache miss."""
    monkeypatch.setattr(settings, "anthropic_api_key", "sk-test")
    await set_limit(db, plan=Plan.FREE, metric=Metric.AI_CALLS_PER_DAY, value=5, anonymous=True)
    client = _client(response=_response())
    ai_service.set_client(client)

    await _generate(db)
    await ai_service.generate_json(
        db,
        purpose="agent_plan",
        grounding={"metrics": {"different": "data"}},
        variables={"goal": "a completely different goal"},
        identity=_identity(),
    )

    first, second = client.messages.calls
    assert first["system"][0]["cache_control"] == {"type": "ephemeral"}
    assert first["system"] == second["system"]
    assert first["messages"] != second["messages"]


async def test_the_client_never_sees_a_prompt_from_the_caller() -> None:
    """`generate_json` takes a purpose, not a prompt. There is no parameter a
    request body could reach that ends up in the system turn."""
    import inspect

    parameters = set(inspect.signature(ai_service.generate_json).parameters)
    assert "prompt" not in parameters
    assert "system" not in parameters


def test_the_grounding_rule_is_on_every_prompt() -> None:
    for prompt in ai_prompts.REGISTRY.values():
        assert ai_prompts.GROUNDING in prompt.system
        assert prompt.model in ai_pricing.RATES


def test_every_schema_satisfies_the_structured_output_constraints() -> None:
    """`additionalProperties: false` is required on every object, and the
    numeric/length keywords are silently dropped rather than enforced — a
    schema relying on them would validate nothing."""
    unsupported = {"minimum", "maximum", "minLength", "maxLength", "multipleOf", "$ref"}

    def walk(node: Any) -> None:
        if isinstance(node, dict):
            if node.get("type") == "object":
                assert node.get("additionalProperties") is False
                assert "required" in node
            assert not unsupported & set(node)
            for value in node.values():
                walk(value)
        elif isinstance(node, list):
            for item in node:
                walk(item)

    for prompt in ai_prompts.REGISTRY.values():
        walk(prompt.schema)


# ── accounting ───────────────────────────────────────────────────────────────


def test_cost_matches_the_internal_rate_table() -> None:
    # Opus 5 is $5/1M in, $25/1M out. 10,000 in + 2,000 out
    # = 0.05 + 0.05 = $0.10.
    assert ai_pricing.cost_of(
        model="claude-opus-5", input_tokens=10_000, output_tokens=2_000
    ) == Decimal("0.100000")

    # Sonnet 5 is $3/$15. 10,000 in + 1,000 out = 0.03 + 0.015 = $0.045.
    assert ai_pricing.cost_of(
        model="claude-sonnet-5", input_tokens=10_000, output_tokens=1_000
    ) == Decimal("0.045000")

    # Haiku 4.5 is $1/$5. 1,000 in + 500 out = 0.001 + 0.0025 = $0.0035.
    assert ai_pricing.cost_of(
        model="claude-haiku-4-5", input_tokens=1_000, output_tokens=500
    ) == Decimal("0.003500")


def test_cached_reads_bill_at_a_tenth_and_writes_at_1_25x() -> None:
    # 100,000 cached reads on Opus 5: 100,000 x $5/1M x 0.1 = $0.05.
    assert ai_pricing.cost_of(
        model="claude-opus-5", input_tokens=0, output_tokens=0, cached_read_tokens=100_000
    ) == Decimal("0.050000")

    # The same 100,000 written: x 1.25 = $0.625.
    assert ai_pricing.cost_of(
        model="claude-opus-5", input_tokens=0, output_tokens=0, cached_write_tokens=100_000
    ) == Decimal("0.625000")

    # And caching is only worth doing if the read is cheaper than the write.
    assert ai_pricing.CACHE_READ_MULTIPLIER < 1 < ai_pricing.CACHE_WRITE_MULTIPLIER


def test_an_unknown_model_prices_high_rather_than_free() -> None:
    """A model with no rate should be conspicuous in the ledger, not invisible."""
    unknown = ai_pricing.cost_of(model="claude-unreleased", input_tokens=1_000, output_tokens=0)
    assert unknown > 0


async def test_a_successful_call_records_tokens_cost_and_prompt_version(
    db: AsyncSession, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(settings, "anthropic_api_key", "sk-test")
    ai_service.set_client(
        _client(response=_response(input_tokens=4_000, output_tokens=800, cache_read=2_000))
    )

    result = await _generate(db)
    assert result is not None

    row = (await db.execute(select(AiCall))).scalars().one()
    assert row.outcome is AiOutcome.SUCCESS
    assert row.input_tokens == 4_000
    assert row.output_tokens == 800
    assert row.cached_read_tokens == 2_000
    assert row.prompt_version == ai_prompts.PROMPT_VERSION
    assert row.tool_slug == "workflow-plan"
    # Sonnet 5: 4,000 x $3/1M + 800 x $15/1M + 2,000 x $3/1M x 0.1
    #         = 0.012 + 0.012 + 0.0006 = $0.0246
    assert row.cost_usd == Decimal("0.024600")
    assert result.meta.cost_usd == Decimal("0.024600")
    assert result.meta.prompt_version == ai_prompts.PROMPT_VERSION


async def test_a_repeat_call_reports_cached_reads(
    db: AsyncSession, monkeypatch: pytest.MonkeyPatch
) -> None:
    """What proves caching is on: the second call's prompt is served from the
    cache, and the figure is recorded separately so it can be seen at all."""
    monkeypatch.setattr(settings, "anthropic_api_key", "sk-test")
    await set_limit(db, plan=Plan.FREE, metric=Metric.AI_CALLS_PER_DAY, value=5, anonymous=True)
    ai_service.set_client(_client(response=_response(cache_write=1_500)))
    await _generate(db)

    ai_service.set_client(_client(response=_response(input_tokens=40, cache_read=1_500)))
    await _generate(db)

    rows = (await db.execute(select(AiCall).order_by(AiCall.created_at))).scalars().all()
    assert rows[0].cached_write_tokens == 1_500
    assert rows[0].cached_read_tokens == 0
    assert rows[1].cached_read_tokens == 1_500


async def test_failures_are_recorded_too(db: AsyncSession, monkeypatch: pytest.MonkeyPatch) -> None:
    """A ledger that only holds successes cannot answer how often this works."""
    monkeypatch.setattr(settings, "anthropic_api_key", "sk-test")
    ai_service.set_client(_client(error=anthropic.APITimeoutError(request=_REQUEST)))
    await _generate(db)

    total = (await db.execute(select(func.count()).select_from(AiCall))).scalar_one()
    assert total == 1

    row = (await db.execute(select(AiCall))).scalars().one()
    assert row.outcome is AiOutcome.TIMEOUT
    assert row.cost_usd == Decimal(0)
    assert row.error_detail and "Timeout" in row.error_detail


# ── one client ───────────────────────────────────────────────────────────────


def test_ai_service_is_the_only_anthropic_client_in_the_process() -> None:
    """Grepped rather than trusted. A second client somewhere else would have
    its own timeout, its own failure handling, and no ledger row."""
    from pathlib import Path

    allowed = {"ai_service.py"}
    offenders: list[str] = []

    for path in Path("app").rglob("*.py"):
        if path.name in allowed:
            continue
        source = path.read_text(encoding="utf-8")
        if "AsyncAnthropic(" in source or "anthropic.Anthropic(" in source:
            offenders.append(str(path))

    assert offenders == [], f"Anthropic client constructed outside ai_service: {offenders}"
