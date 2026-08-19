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

import groq
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


class _FakeCompletions:
    """Stands in for `client.chat.completions`, with a scripted outcome."""

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
    completions = _FakeCompletions(response=response, error=error)
    client = SimpleNamespace(chat=SimpleNamespace(completions=completions))
    # The tests reach for `client.completions.calls`; the nesting the SDK
    # requires would otherwise make every assertion read `chat.completions`.
    client.completions = completions
    return client


def _response(
    text: str = '{"summary": "ok", "why": "because", "weakest_link": "a", "watch_out_for": []}',
    *,
    finish_reason: str = "stop",
    prompt_tokens: int = 1000,
    completion_tokens: int = 200,
    cached: int = 0,
) -> Any:
    """One `chat.completions` response.

    `prompt_tokens` is the **whole** prompt on this provider, cached part
    included — the same convention the API uses, so a test that sets both
    figures is describing what the service actually receives rather than what
    it stores.
    """
    return SimpleNamespace(
        choices=[
            SimpleNamespace(
                finish_reason=finish_reason,
                message=SimpleNamespace(role="assistant", content=text),
            )
        ],
        usage=SimpleNamespace(
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            prompt_tokens_details=SimpleNamespace(cached_tokens=cached) if cached else None,
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
    monkeypatch.setattr(settings, "groq_api_key", "")
    assert settings.ai_enabled is False

    assert await _generate(db) is None
    assert await _outcomes(db) == [AiOutcome.DISABLED]


_REQUEST = httpx.Request("POST", "https://api.groq.com/openai/v1/chat/completions")


@pytest.mark.parametrize(
    ("error", "expected"),
    [
        (groq.APITimeoutError(request=_REQUEST), AiOutcome.TIMEOUT),
        (
            groq.RateLimitError("429", response=httpx.Response(429, request=_REQUEST), body=None),
            AiOutcome.RATE_LIMITED,
        ),
        (groq.APIConnectionError(request=_REQUEST), AiOutcome.API_ERROR),
        (
            groq.InternalServerError(
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
    monkeypatch.setattr(settings, "groq_api_key", "gsk-test")
    ai_service.set_client(_client(error=error))

    assert await _generate(db) is None
    assert await _outcomes(db) == [expected]


async def test_a_token_rate_limit_is_recorded_as_one_despite_the_413(
    db: AsyncSession, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Exceeding the per-minute *token* allowance answers `413 Request too
    large`, not `429` — a payload-size status for a throughput problem, and
    the single most likely failure on a small tier because one big grounding
    payload trips it. Filed as `api_error` it reads as "the provider is
    down", which is the wrong thing to go and check.

    The body is the one this exact call returned in a live run.
    """
    monkeypatch.setattr(settings, "groq_api_key", "gsk-test")
    too_large = groq.APIStatusError(
        "413",
        response=httpx.Response(413, request=_REQUEST),
        body={
            "error": {
                "message": (
                    "Request too large for model `openai/gpt-oss-120b` ... on tokens "
                    "per minute (TPM): Limit 8000, Requested 12206"
                ),
                "type": "tokens",
                "code": "rate_limit_exceeded",
            }
        },
    )
    ai_service.set_client(_client(error=too_large))

    assert await _generate(db) is None
    assert await _outcomes(db) == [AiOutcome.RATE_LIMITED]


async def test_an_ordinary_status_error_is_still_an_api_error(
    db: AsyncSession, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The counterpart. Widening 413 into "always a rate limit" would hide a
    genuinely oversized request behind a retry that can never succeed."""
    monkeypatch.setattr(settings, "groq_api_key", "gsk-test")
    ai_service.set_client(
        _client(
            error=groq.APIStatusError(
                "413",
                response=httpx.Response(413, request=_REQUEST),
                body={"error": {"message": "payload too large", "code": "request_too_large"}},
            )
        )
    )

    assert await _generate(db) is None
    assert await _outcomes(db) == [AiOutcome.API_ERROR]


@pytest.mark.parametrize("reason", ["content_filter", "refusal"])
async def test_a_refusal_returns_none_rather_than_raising(
    db: AsyncSession, monkeypatch: pytest.MonkeyPatch, reason: str
) -> None:
    """A declined request is a successful HTTP 200 with no usable content.
    Reading the message unconditionally is how that becomes an exception —
    and classifying it as a refusal rather than malformed output is what
    stops it being debugged as a broken schema."""
    monkeypatch.setattr(settings, "groq_api_key", "gsk-test")
    refused = _response(finish_reason=reason)
    refused.choices[0].message.content = None
    ai_service.set_client(_client(response=refused))

    assert await _generate(db) is None
    assert await _outcomes(db) == [AiOutcome.REFUSAL]


async def test_an_empty_choices_list_degrades_rather_than_raising(
    db: AsyncSession, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The one shape that would escape the function if `choices[0]` were read
    directly."""
    monkeypatch.setattr(settings, "groq_api_key", "gsk-test")
    empty = _response()
    empty.choices = []
    ai_service.set_client(_client(response=empty))

    assert await _generate(db) is None
    assert await _outcomes(db) == [AiOutcome.INVALID_OUTPUT]


async def test_malformed_output_degrades_instead_of_throwing(
    db: AsyncSession, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(settings, "groq_api_key", "gsk-test")
    ai_service.set_client(_client(response=_response("this is not json at all")))

    assert await _generate(db) is None
    assert await _outcomes(db) == [AiOutcome.INVALID_OUTPUT]


async def test_truncated_json_degrades(db: AsyncSession, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(settings, "groq_api_key", "gsk-test")
    ai_service.set_client(_client(response=_response('{"summary": "cut off mid')))

    assert await _generate(db) is None


async def test_exhausted_quota_returns_none_without_calling_the_model(
    db: AsyncSession, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(settings, "groq_api_key", "gsk-test")
    client = _client(response=_response())
    ai_service.set_client(client)
    await set_limit(db, plan=Plan.FREE, metric=Metric.AI_CALLS_PER_DAY, value=0, anonymous=True)

    assert await _generate(db) is None
    assert client.completions.calls == []
    assert await _outcomes(db) == [AiOutcome.QUOTA_EXCEEDED]


# ── the request shape ────────────────────────────────────────────────────────


async def test_the_request_obeys_this_model_familys_rules(
    db: AsyncSession, monkeypatch: pytest.MonkeyPatch
) -> None:
    """`max_tokens` is the easy one to reintroduce from memory: it reads as
    normal API usage, is accepted, and silently fails to bound the reasoning
    half of the output — which is the half that runs away."""
    monkeypatch.setattr(settings, "groq_api_key", "gsk-test")
    client = _client(response=_response())
    ai_service.set_client(client)

    await _generate(db)
    sent = client.completions.calls[0]

    assert "max_tokens" not in sent
    assert sent["max_completion_tokens"] > 0
    # Only `low` | `medium` | `high` exist on this provider; anything else is
    # a 400, and the registry is the only place that picks one.
    assert sent["reasoning_effort"] in {"low", "medium", "high"}
    # System first, user last. No assistant prefill.
    assert [message["role"] for message in sent["messages"]] == ["system", "user"]
    # Structured output, not "please reply with JSON" — and enforced, not
    # suggested, which is what `strict` buys.
    assert sent["response_format"]["type"] == "json_schema"
    assert sent["response_format"]["json_schema"]["strict"] is True
    assert (
        sent["response_format"]["json_schema"]["schema"] == ai_prompts.REGISTRY["agent_plan"].schema
    )


async def test_the_system_prompt_comes_first_and_never_varies(
    db: AsyncSession, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The stable half first, the varying half after it. The cache on this
    provider is automatic and prefix-based, so message order is the only
    lever there is — a variable interpolated into the system text would make
    every request a cache miss with nothing to show for it."""
    monkeypatch.setattr(settings, "groq_api_key", "gsk-test")
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

    first, second = client.completions.calls
    assert first["messages"][0]["role"] == "system"
    assert first["messages"][0] == second["messages"][0]
    assert first["messages"][1] != second["messages"][1]


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


def test_no_prompt_reserves_more_output_than_a_small_tier_can_afford() -> None:
    """The reservation is charged against the per-minute allowance whether or
    not it is used, so `prompt + max_tokens` — not `max_tokens` alone — is
    what has to fit. This is not a style rule: an 8,000-token reservation on
    a ~4,200-token prompt made the flagship tool fail 100% of the time on a
    tier whose limit is 8,000, and it failed as a 413 that looked nothing
    like "the number in the registry is too big".
    """
    for prompt in ai_prompts.REGISTRY.values():
        requested = ai_prompts.GROUNDING_ALLOWANCE + prompt.max_tokens
        assert prompt.max_tokens <= ai_prompts.MAX_OUTPUT_RESERVATION, (
            f"{prompt.purpose} reserves {prompt.max_tokens}, so a full-sized "
            f"request asks for ~{requested} against a "
            f"{ai_prompts.TIER_TOKENS_PER_MINUTE} limit"
        )


def test_every_effort_is_one_the_provider_accepts() -> None:
    """Anything outside this set is a 400, and the failure arrives as a
    generic api_error long after the edit that caused it."""
    for prompt in ai_prompts.REGISTRY.values():
        assert prompt.effort in {"low", "medium", "high"}


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
    # gpt-oss-120b is $0.15/1M in, $0.60/1M out. 100,000 in + 20,000 out
    # = 0.015 + 0.012 = $0.027.
    assert ai_pricing.cost_of(
        model="openai/gpt-oss-120b", input_tokens=100_000, output_tokens=20_000
    ) == Decimal("0.027000")

    # gpt-oss-20b is half that on both sides: $0.075/$0.30.
    # 100,000 in + 20,000 out = 0.0075 + 0.006 = $0.0135.
    assert ai_pricing.cost_of(
        model="openai/gpt-oss-20b", input_tokens=100_000, output_tokens=20_000
    ) == Decimal("0.013500")


def test_cached_reads_bill_at_half_and_writes_are_free() -> None:
    # 1,000,000 cached reads on 120b: 1,000,000 x $0.15/1M x 0.5 = $0.075.
    assert ai_pricing.cost_of(
        model="openai/gpt-oss-120b",
        input_tokens=0,
        output_tokens=0,
        cached_read_tokens=1_000_000,
    ) == Decimal("0.075000")

    # Caching is only worth having if the read is cheaper than a fresh token.
    assert ai_pricing.CACHE_READ_MULTIPLIER < 1

    # This provider charges nothing extra to populate the cache, so a write
    # costs exactly what the same tokens would have cost uncached. Asserted
    # rather than assumed: the multiplier existing at all invites the reading
    # that it must be a surcharge.
    assert ai_pricing.CACHE_WRITE_MULTIPLIER == 1
    assert ai_pricing.cost_of(
        model="openai/gpt-oss-120b",
        input_tokens=0,
        output_tokens=0,
        cached_write_tokens=1_000_000,
    ) == ai_pricing.cost_of(model="openai/gpt-oss-120b", input_tokens=1_000_000, output_tokens=0)


def test_an_unknown_model_prices_high_rather_than_free() -> None:
    """A model with no rate should be conspicuous in the ledger, not invisible."""
    unknown = ai_pricing.cost_of(
        model="openai/gpt-oss-unreleased", input_tokens=1_000_000, output_tokens=0
    )
    assert unknown > 0


async def test_a_successful_call_records_tokens_cost_and_prompt_version(
    db: AsyncSession, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(settings, "groq_api_key", "gsk-test")
    # 6,000 reported, 2,000 of them cached - so 4,000 were billed in full.
    ai_service.set_client(
        _client(response=_response(prompt_tokens=6_000, completion_tokens=800, cached=2_000))
    )

    result = await _generate(db)
    assert result is not None

    row = (await db.execute(select(AiCall))).scalars().one()
    assert row.outcome is AiOutcome.SUCCESS
    # The uncached remainder, not the figure the API reported. Storing 6,000
    # here would bill the cached prompt twice, once at the full rate.
    assert row.input_tokens == 4_000
    assert row.output_tokens == 800
    assert row.cached_read_tokens == 2_000
    assert row.prompt_version == ai_prompts.PROMPT_VERSION
    assert row.tool_slug == "workflow-plan"
    # gpt-oss-120b: 4,000 x $0.15/1M + 800 x $0.60/1M + 2,000 x $0.15/1M x 0.5
    #             = 0.0006 + 0.00048 + 0.00015 = $0.00123
    assert row.cost_usd == Decimal("0.001230")
    assert result.meta.cost_usd == Decimal("0.001230")
    assert result.meta.prompt_version == ai_prompts.PROMPT_VERSION


async def test_a_repeat_call_reports_cached_reads(
    db: AsyncSession, monkeypatch: pytest.MonkeyPatch
) -> None:
    """What proves caching is on: the second call's prompt prefix is served
    from the cache, and the figure is recorded separately so it can be seen at
    all. A miss and a hit have to be distinguishable in the ledger, or "is the
    cache working" is unanswerable after the fact."""
    monkeypatch.setattr(settings, "groq_api_key", "gsk-test")
    await set_limit(db, plan=Plan.FREE, metric=Metric.AI_CALLS_PER_DAY, value=5, anonymous=True)

    # A miss: the provider omits the detail block entirely rather than sending
    # a zero, which is the shape the reader has to survive.
    ai_service.set_client(_client(response=_response(prompt_tokens=1_540)))
    await _generate(db)

    ai_service.set_client(_client(response=_response(prompt_tokens=1_540, cached=1_500)))
    await _generate(db)

    rows = (await db.execute(select(AiCall).order_by(AiCall.created_at))).scalars().all()
    assert rows[0].cached_read_tokens == 0
    assert rows[0].input_tokens == 1_540
    assert rows[1].cached_read_tokens == 1_500
    assert rows[1].input_tokens == 40
    # Populating the cache is free here, so this column stays zero on both.
    assert [row.cached_write_tokens for row in rows] == [0, 0]


async def test_failures_are_recorded_too(db: AsyncSession, monkeypatch: pytest.MonkeyPatch) -> None:
    """A ledger that only holds successes cannot answer how often this works."""
    monkeypatch.setattr(settings, "groq_api_key", "gsk-test")
    ai_service.set_client(_client(error=groq.APITimeoutError(request=_REQUEST)))
    await _generate(db)

    total = (await db.execute(select(func.count()).select_from(AiCall))).scalar_one()
    assert total == 1

    row = (await db.execute(select(AiCall))).scalars().one()
    assert row.outcome is AiOutcome.TIMEOUT
    assert row.cost_usd == Decimal(0)
    assert row.error_detail and "Timeout" in row.error_detail


# ── one client ───────────────────────────────────────────────────────────────


def test_ai_service_is_the_only_generation_client_in_the_process() -> None:
    """Grepped rather than trusted. A second client somewhere else would have
    its own timeout, its own failure handling, and no ledger row."""
    from pathlib import Path

    offenders: list[str] = []
    for path in Path("app").rglob("*.py"):
        if path.name == "ai_service.py":
            continue
        source = path.read_text(encoding="utf-8")
        if "AsyncGroq(" in source or "groq.Groq(" in source:
            offenders.append(str(path))

    assert offenders == [], f"Groq client constructed outside ai_service: {offenders}"


def test_the_anthropic_sdk_is_only_ever_used_for_counting() -> None:
    """The provider swap is only real if nothing still *generates* through the
    old SDK. `tokenizer_service` keeps a client on purpose - `count_tokens` is
    the only accurate count for a Claude row in the catalogue and Groq has no
    equivalent endpoint - so it is allowed the client, and only for that."""
    from pathlib import Path

    offenders: list[str] = []
    for path in Path("app").rglob("*.py"):
        source = path.read_text(encoding="utf-8")
        if path.name != "tokenizer_service.py" and (
            "AsyncAnthropic(" in source or "anthropic.Anthropic(" in source
        ):
            offenders.append(f"{path}: constructs an Anthropic client")
        if "messages.create(" in source:
            offenders.append(f"{path}: generates through the Anthropic SDK")

    assert offenders == [], f"Anthropic SDK used for more than counting: {offenders}"
