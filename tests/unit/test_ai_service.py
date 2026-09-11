"""The AI layer's guarantees.

The module's whole claim is that a model failure is never a request failure,
so most of this file is failure modes. Each one asserts the same two things:
`generate_json` returned `None`, and no exception escaped.

The pricing tests assert exact dollars against the internal rate table, by
hand. A cost calculation that only agrees with itself is not checked.

The provider is Anthropic and there is only one, so the stub replaces the
`AsyncAnthropic` client the service opens per call. What it returns are the
SDK's own `BetaMessage` objects and what it raises are the SDK's own
exceptions, built around real status codes — so a test that says "a 429 is
recorded as a rate limit" is exercising the same branch a live 429 would.
"""

from __future__ import annotations

from decimal import Decimal
from types import SimpleNamespace
from typing import Any

import anthropic
import httpx
import pytest
from anthropic.types.beta import BetaMessage
from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.api.deps import Identity
from app.core.config import settings
from app.models.ai import AiCall, AiOutcome
from app.models.billing import Metric
from app.models.user import Plan, User
from app.services import ai_pricing, ai_prompts, ai_service
from tests.conftest import set_limit


async def _identity(db: AsyncSession) -> Identity:
    """A real row, not a detached object.

    `ai_calls.user_id` carries a foreign key, so the ledger insert this file
    asserts on fails unless the owner exists. It used to be an anonymous id on
    a column with no foreign key, which is why a made-up value worked.
    """
    user = await db.get(User, "usr_test")
    if user is None:
        user = User(
            id="usr_test",
            email="ada@example.com",
            name="Ada",
            password_hash="x",
            plan=Plan.FREE,
        )
        db.add(user)
        await db.flush()
    return Identity(user=user, session_id=None)


class _FakeAnthropic:
    """Stands in for the `AsyncAnthropic` client the service opens per call.

    Records what was sent, because half of what this file checks is the
    request shape — and the request shape is the part that turns into a 400
    at three in the morning rather than a test failure.
    """

    def __init__(self, *, responses: list[BetaMessage], error: Exception | None) -> None:
        self._responses = responses
        self._error = error
        self.calls: list[dict[str, Any]] = []
        self.options: dict[str, Any] = {}
        self.beta = SimpleNamespace(messages=SimpleNamespace(create=self._create))

    async def __aenter__(self) -> _FakeAnthropic:
        return self

    async def __aexit__(self, *args: Any) -> None:
        return None

    async def _create(self, **kwargs: Any) -> BetaMessage:
        self.calls.append(kwargs)
        if self._error is not None:
            raise self._error
        # The last response repeats, so a test that makes two calls and cares
        # about only one of them does not have to script both.
        index = min(len(self.calls) - 1, len(self._responses) - 1)
        return self._responses[index]


def _client(
    monkeypatch: pytest.MonkeyPatch,
    *,
    response: BetaMessage | None = None,
    responses: list[BetaMessage] | None = None,
    error: Exception | None = None,
) -> _FakeAnthropic:
    """Install the stub, with a key the service believes in, and hand it back
    for its `calls`."""
    monkeypatch.setattr(settings, "anthropic_api_key", "sk-ant-test")
    scripted = responses if responses is not None else [response] if response else []
    client = _FakeAnthropic(responses=scripted, error=error)

    def build(**options: Any) -> _FakeAnthropic:
        client.options = options
        return client

    monkeypatch.setattr(ai_service, "AsyncAnthropic", build)
    return client


_REQUEST = httpx.Request("POST", "https://api.anthropic.com/v1/messages?beta=true")


def _status_error(
    cls: type[anthropic.APIStatusError], status: int, kind: str
) -> anthropic.APIStatusError:
    """What the SDK raises for a non-2xx response, built the way it builds one."""
    body = {"type": "error", "error": {"type": kind, "message": f"{kind} in a test"}}
    return cls(
        f"Error code: {status} - {body}",
        response=httpx.Response(status, request=_REQUEST, json=body),
        body=body,
    )


_ANSWER = '{"summary": "ok", "why": "because", "weakest_link": "a", "watch_out_for": []}'


def _response(
    text: str | None = _ANSWER,
    *,
    stop_reason: str = "end_turn",
    refusal_category: str | None = None,
    model: str = "claude-opus-5",
    input_tokens: int = 1000,
    output_tokens: int = 200,
    cache_read: int | None = None,
    cache_write: int | None = None,
    content: list[dict[str, Any]] | None = None,
) -> BetaMessage:
    """One Messages API response, as the SDK hands it back.

    `input_tokens` is the **uncached remainder**, which is the convention the
    API itself uses — cache reads and writes are reported beside it, not
    inside it. The cache fields are omitted rather than zero when unset,
    because that is the shape a request with nothing cached arrives in.
    """
    usage: dict[str, Any] = {"input_tokens": input_tokens, "output_tokens": output_tokens}
    if cache_read is not None:
        usage["cache_read_input_tokens"] = cache_read
    if cache_write is not None:
        usage["cache_creation_input_tokens"] = cache_write
    if content is None:
        content = [{"type": "text", "text": text}] if text is not None else []
    return BetaMessage.model_validate(
        {
            "id": "msg_test",
            "type": "message",
            "role": "assistant",
            "model": model,
            "stop_reason": stop_reason,
            "stop_sequence": None,
            "stop_details": (
                {"type": "refusal", "category": refusal_category, "explanation": "declined"}
                if stop_reason == "refusal"
                else None
            ),
            "content": content,
            "usage": usage,
        }
    )


def _thinking(text: str = "") -> dict[str, Any]:
    """A thinking block. Empty by default, because that is what Opus 5 returns
    unless a summary is asked for."""
    return {"type": "thinking", "thinking": text, "signature": "sig"}


async def _generate(db: AsyncSession, purpose: str = "agent_plan") -> Any:
    return await ai_service.generate_json(
        db,
        purpose=purpose,
        grounding={"metrics": {}},
        variables={"goal": "test"},
        identity=await _identity(db),
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
    client = _client(monkeypatch, response=_response())
    monkeypatch.setattr(settings, "anthropic_api_key", "")
    assert settings.ai_enabled is False

    assert await _generate(db) is None
    assert client.calls == []
    assert await _outcomes(db) == [AiOutcome.DISABLED]


@pytest.mark.parametrize(
    ("error", "expected"),
    [
        (anthropic.APITimeoutError(request=_REQUEST), AiOutcome.TIMEOUT),
        (anthropic.APIConnectionError(request=_REQUEST), AiOutcome.API_ERROR),
        (
            _status_error(anthropic.BadRequestError, 400, "invalid_request_error"),
            AiOutcome.API_ERROR,
        ),
        (_status_error(anthropic.InternalServerError, 500, "api_error"), AiOutcome.API_ERROR),
        (_status_error(anthropic.APIStatusError, 529, "overloaded_error"), AiOutcome.API_ERROR),
        (RuntimeError("something nobody predicted"), AiOutcome.API_ERROR),
    ],
)
async def test_every_failure_returns_none_and_is_recorded(
    db: AsyncSession, monkeypatch: pytest.MonkeyPatch, error: Exception, expected: AiOutcome
) -> None:
    _client(monkeypatch, error=error)

    assert await _generate(db) is None
    assert await _outcomes(db) == [expected]


async def test_a_rate_limit_is_recorded_as_a_rate_limit(
    db: AsyncSession, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Running into the organisation's rate or spend limit is the one failure
    an operator can actually act on. Filed under `api_error` it would send that
    investigation to the wrong place — to the prompt, or to the network, rather
    than to the limits page in the Console."""
    _client(monkeypatch, error=_status_error(anthropic.RateLimitError, 429, "rate_limit_error"))

    assert await _generate(db) is None
    assert await _outcomes(db) == [AiOutcome.RATE_LIMITED]

    row = (await db.execute(select(AiCall))).scalars().one()
    assert row.error_detail and "429" in row.error_detail


@pytest.mark.parametrize(
    ("category", "detail"),
    [("cyber", "refusal:cyber"), (None, "refusal:unspecified")],
)
async def test_a_refusal_returns_none_rather_than_raising(
    db: AsyncSession, monkeypatch: pytest.MonkeyPatch, category: str | None, detail: str
) -> None:
    """A declined request is a 200 with no usable content — with the fallback
    on, it means every model in the chain declined. Unnamed, it arrives as
    malformed output and gets debugged as a bad schema."""
    _client(
        monkeypatch,
        response=_response(None, stop_reason="refusal", refusal_category=category, output_tokens=0),
    )

    assert await _generate(db) is None
    assert await _outcomes(db) == [AiOutcome.REFUSAL]

    row = (await db.execute(select(AiCall))).scalars().one()
    assert row.error_detail == detail


async def test_an_exhausted_reservation_is_recorded_with_its_reason(
    db: AsyncSession, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Thinking is drawn from `max_tokens` before the answer is, so a
    reservation that thinking exhausts stops with `max_tokens` and the JSON
    cut off mid-object. Recording the reason is the difference between raising
    a number and rewriting a schema."""
    _client(
        monkeypatch,
        response=_response(
            content=[_thinking(), {"type": "text", "text": '{"summary": "cut off mid'}],
            stop_reason="max_tokens",
            output_tokens=2_400,
        ),
    )

    assert await _generate(db) is None
    assert await _outcomes(db) == [AiOutcome.INVALID_OUTPUT]

    row = (await db.execute(select(AiCall))).scalars().one()
    assert row.error_detail == "stop_reason=max_tokens"
    # And the thinking it did spend is still counted.
    assert row.output_tokens == 2_400


async def test_malformed_output_degrades_instead_of_throwing(
    db: AsyncSession, monkeypatch: pytest.MonkeyPatch
) -> None:
    _client(monkeypatch, response=_response("this is not json at all"))

    assert await _generate(db) is None
    assert await _outcomes(db) == [AiOutcome.INVALID_OUTPUT]


async def test_truncated_json_degrades(db: AsyncSession, monkeypatch: pytest.MonkeyPatch) -> None:
    _client(monkeypatch, response=_response('{"summary": "cut off mid'))

    assert await _generate(db) is None


async def test_a_json_array_is_not_an_answer(
    db: AsyncSession, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Valid JSON of the wrong shape. Every caller indexes the result by key,
    so a list would raise at the `apply` rather than here."""
    _client(monkeypatch, response=_response('[{"summary": "ok"}]'))

    assert await _generate(db) is None
    assert await _outcomes(db) == [AiOutcome.INVALID_OUTPUT]


async def test_exhausted_quota_returns_none_without_calling_the_model(
    db: AsyncSession, monkeypatch: pytest.MonkeyPatch
) -> None:
    client = _client(monkeypatch, response=_response())
    await set_limit(db, plan=Plan.FREE, metric=Metric.AI_CALLS_PER_DAY, value=0)

    assert await _generate(db) is None
    assert client.calls == []
    assert await _outcomes(db) == [AiOutcome.QUOTA_EXCEEDED]


# ── the answer, and the model's reasoning about it ───────────────────────────


async def test_the_models_own_thinking_is_left_out_of_the_answer(
    db: AsyncSession, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Thinking arrives as its own content blocks ahead of the answer.
    Concatenating every block and parsing the result fails the moment anything
    but the answer is in the list — the JSON is valid and the string it is
    glued to is not."""
    _client(
        monkeypatch,
        response=_response(
            content=[
                _thinking("Let me work through the grounding first."),
                {"type": "text", "text": '{"summary": "the answer"}'},
            ]
        ),
    )

    result = await _generate(db)

    assert result is not None
    assert result.data == {"summary": "the answer"}


# ── the request shape ────────────────────────────────────────────────────────


async def test_the_request_carries_the_schema_and_the_depth_knob(
    db: AsyncSession, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Structured output, not "please reply with JSON" — asking in prose and
    parsing fails a few percent of the time, and every failure would degrade
    to `rule_based` with nothing separating a bad prompt from a dead
    provider."""
    client = _client(monkeypatch, response=_response())

    await _generate(db)
    sent = client.calls[0]
    prompt = ai_prompts.REGISTRY["agent_plan"]

    assert client.options["api_key"] == "sk-ant-test"
    assert sent["model"] == prompt.model
    assert sent["max_tokens"] == prompt.max_tokens
    assert sent["output_config"]["format"] == {"type": "json_schema", "schema": prompt.schema}
    # The registry is the only place that picks an effort.
    assert sent["output_config"]["effort"] == prompt.effort
    assert sent["thinking"] == {"type": "adaptive"}


async def test_a_declined_request_is_rerun_on_the_recommended_fallback(
    db: AsyncSession, monkeypatch: pytest.MonkeyPatch
) -> None:
    """`fallbacks: "default"` has the API re-run a declined request on the
    model Anthropic recommends for that refusal category, inside the same
    call. Without it, a classifier decision is a `rule_based` page. The beta
    header is exact — the array form's header with this form is a 400."""
    client = _client(monkeypatch, response=_response())

    await _generate(db)
    sent = client.calls[0]

    assert sent["fallbacks"] == "default"
    assert sent["betas"] == ["server-side-fallback-2026-07-01"]


async def test_the_model_that_answered_is_the_one_priced_and_recorded(
    db: AsyncSession, monkeypatch: pytest.MonkeyPatch
) -> None:
    """With a fallback, the model asked and the model that answered can
    differ. The ledger names and prices the one that did the work, or a row
    would reconcile against the wrong line on the invoice."""
    _client(
        monkeypatch,
        response=_response(model="claude-opus-4-8", input_tokens=1_000, output_tokens=1_000),
    )

    result = await _generate(db)
    assert result is not None
    assert result.meta.model == "claude-opus-4-8"

    row = (await db.execute(select(AiCall))).scalars().one()
    assert row.model == "claude-opus-4-8"
    # 1,000 x $5/1M + 1,000 x $25/1M = 0.005 + 0.025 = $0.03.
    assert row.cost_usd == Decimal("0.030000")


async def test_the_system_prompt_comes_first_never_varies_and_carries_the_cache_marker(
    db: AsyncSession, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The stable half in `system`, the varying half in the user turn after it.
    Caching is an explicit prefix marker here, so it belongs on the part that
    repeats — a variable interpolated into the system text would make every
    request a cache write that is never read back."""
    await set_limit(db, plan=Plan.FREE, metric=Metric.AI_CALLS_PER_DAY, value=5)
    client = _client(monkeypatch, response=_response())

    await _generate(db)
    await ai_service.generate_json(
        db,
        purpose="agent_plan",
        grounding={"metrics": {"different": "data"}},
        variables={"goal": "a completely different goal"},
        identity=await _identity(db),
    )

    first, second = client.calls
    assert first["system"] == second["system"]
    assert first["system"][-1]["cache_control"] == {"type": "ephemeral"}
    assert first["messages"] != second["messages"]
    assert [turn["role"] for turn in first["messages"]] == ["user"]


async def test_the_client_never_sees_a_prompt_from_the_caller() -> None:
    """Prompts are looked up by purpose, never passed in. A prompt that can
    arrive from a route is a prompt that can arrive from a request body."""
    import inspect

    signature = inspect.signature(ai_service.generate_json)

    assert "prompt" not in signature.parameters
    assert "system" not in signature.parameters
    assert "schema" not in signature.parameters


def test_the_grounding_rule_is_on_every_prompt() -> None:
    for prompt in ai_prompts.REGISTRY.values():
        assert prompt.system.startswith(ai_prompts.GROUNDING)


def test_every_reservation_leaves_room_to_think_and_stays_under_the_ceiling() -> None:
    """Thinking and the answer share one reservation, which makes the floor
    the number that bites.

    Reserve too little and the model spends the whole budget thinking, and the
    call returns a 200 with the answer cut off — `rule_based` on the page, and
    nothing in the ledger that looks like a token problem unless the stop
    reason is read. The Architect's assessment did exactly this at 3,000:
    2,700 tokens of reasoning, 269 left for a ten-row score breakdown and five
    prose fields.
    """
    for prompt in ai_prompts.REGISTRY.values():
        assert prompt.max_tokens >= ai_prompts.MIN_OUTPUT_RESERVATION, (
            f"{prompt.purpose} reserves {prompt.max_tokens} output tokens, under the "
            f"{ai_prompts.MIN_OUTPUT_RESERVATION} floor — thinking alone can spend that"
        )
        assert prompt.max_tokens <= ai_prompts.MAX_OUTPUT_RESERVATION, (
            f"{prompt.purpose} reserves {prompt.max_tokens} output tokens, over the "
            f"{ai_prompts.MAX_OUTPUT_RESERVATION} ceiling"
        )


def test_every_effort_is_one_the_provider_accepts() -> None:
    for prompt in ai_prompts.REGISTRY.values():
        assert prompt.effort in {"low", "medium", "high"}


def test_every_model_named_by_a_prompt_has_a_rate() -> None:
    """An unpriced model still runs and still costs money; it just reports the
    fallback, which is a number nobody can reconcile against an invoice."""
    for prompt in ai_prompts.REGISTRY.values():
        assert prompt.model in ai_pricing.RATES, f"{prompt.model} has no rate"


def test_every_model_is_a_claude_model() -> None:
    """The registry is the only place a model is chosen, and there is one
    provider. An id from anywhere else would be a 404 on every call — a
    `rule_based` page with nothing in the ledger but `api_error`."""
    for prompt in ai_prompts.REGISTRY.values():
        assert prompt.model.startswith("claude-"), f"{prompt.purpose} names {prompt.model}"


def test_every_schema_satisfies_the_structured_output_constraints() -> None:
    """`additionalProperties: false` is required on every object, and the
    numeric/length keywords are not supported — a schema relying on them
    would validate nothing, or be refused outright."""
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
    # claude-opus-5 is $5/1M in, $25/1M out. 100,000 in + 20,000 out
    # = 0.50 + 0.50 = $1.00.
    assert ai_pricing.cost_of(
        model="claude-opus-5", input_tokens=100_000, output_tokens=20_000
    ) == Decimal("1.000000")

    # claude-opus-4-8, the cyber-category fallback, bills at the same rates.
    assert ai_pricing.cost_of(
        model="claude-opus-4-8", input_tokens=100_000, output_tokens=20_000
    ) == Decimal("1.000000")


def test_cached_reads_bill_at_a_tenth_and_writes_at_a_premium() -> None:
    # 1,000,000 cached reads on Opus 5: 1,000,000 x $5/1M x 0.1 = $0.50.
    assert ai_pricing.cost_of(
        model="claude-opus-5",
        input_tokens=0,
        output_tokens=0,
        cached_read_tokens=1_000_000,
    ) == Decimal("0.500000")

    # Caching is only worth having if the read is cheaper than a fresh token.
    assert ai_pricing.CACHE_READ_MULTIPLIER < 1

    # Populating the cache costs more than sending the same tokens uncached.
    # Asserted rather than assumed: a write priced at 1x would make every
    # cache entry nobody reads look free in the ledger.
    # 1,000,000 writes x $5/1M x 1.25 = $6.25.
    assert Decimal("1.25") == ai_pricing.CACHE_WRITE_MULTIPLIER
    assert ai_pricing.cost_of(
        model="claude-opus-5",
        input_tokens=0,
        output_tokens=0,
        cached_write_tokens=1_000_000,
    ) == Decimal("6.250000")


def test_an_unknown_model_prices_high_rather_than_free() -> None:
    """A model with no rate should be conspicuous in the ledger, not invisible."""
    unknown = ai_pricing.cost_of(
        model="claude-9-unreleased", input_tokens=1_000_000, output_tokens=0
    )
    assert unknown > 0


async def test_a_successful_call_records_tokens_cost_and_prompt_version(
    db: AsyncSession, monkeypatch: pytest.MonkeyPatch
) -> None:
    # 4,000 billed in full and 2,000 more served from the cache. The API
    # reports them apart, and they are stored apart.
    _client(
        monkeypatch,
        response=_response(input_tokens=4_000, output_tokens=800, cache_read=2_000),
    )

    result = await _generate(db)
    assert result is not None

    row = (await db.execute(select(AiCall))).scalars().one()
    assert row.outcome is AiOutcome.SUCCESS
    assert row.model == "claude-opus-5"
    # The uncached remainder, exactly as reported. Adding the cached 2,000
    # back in here would bill the cached prompt twice, once at the full rate.
    assert row.input_tokens == 4_000
    assert row.output_tokens == 800
    assert row.cached_read_tokens == 2_000
    assert row.cached_write_tokens == 0
    assert row.prompt_version == ai_prompts.PROMPT_VERSION
    assert row.tool_slug == "workflow-plan"
    # agent_plan runs on claude-opus-5:
    #   4,000 x $5/1M + 800 x $25/1M + 2,000 x $5/1M x 0.1
    # = 0.02 + 0.02 + 0.001 = $0.041
    assert row.cost_usd == Decimal("0.041000")
    assert result.meta.cost_usd == Decimal("0.041000")
    assert result.meta.prompt_version == ai_prompts.PROMPT_VERSION


async def test_thinking_tokens_are_billed_as_output(
    db: AsyncSession, monkeypatch: pytest.MonkeyPatch
) -> None:
    """`output_tokens` already includes thinking, and all of it is charged at
    the output rate. Nothing may subtract it back out: a short structured
    answer routinely costs several times the JSON it produces, and a ledger
    that counted only the visible half would understate it by more than it
    counted."""
    _client(
        monkeypatch,
        response=_response(
            content=[_thinking(), {"type": "text", "text": _ANSWER}],
            input_tokens=1_000,
            output_tokens=1_000,
        ),
    )

    result = await _generate(db)
    assert result is not None
    assert result.meta.output_tokens == 1_000

    row = (await db.execute(select(AiCall))).scalars().one()
    assert row.output_tokens == 1_000
    # 1,000 x $5/1M + 1,000 x $25/1M = 0.005 + 0.025 = $0.03.
    assert row.cost_usd == Decimal("0.030000")


async def test_a_repeat_call_reports_cached_reads(
    db: AsyncSession, monkeypatch: pytest.MonkeyPatch
) -> None:
    """What proves caching is on: the first call writes the prefix, the second
    reads it back, and each figure is recorded separately so it can be seen at
    all. A write and a hit have to be distinguishable in the ledger, or "is
    the cache working" is unanswerable after the fact."""
    await set_limit(db, plan=Plan.FREE, metric=Metric.AI_CALLS_PER_DAY, value=5)
    _client(
        monkeypatch,
        responses=[
            _response(input_tokens=40, cache_write=1_500),
            _response(input_tokens=40, cache_read=1_500),
        ],
    )
    await _generate(db)
    await _generate(db)

    rows = (await db.execute(select(AiCall).order_by(AiCall.created_at))).scalars().all()
    assert [row.input_tokens for row in rows] == [40, 40]
    assert [row.cached_write_tokens for row in rows] == [1_500, 0]
    assert [row.cached_read_tokens for row in rows] == [0, 1_500]


async def test_failures_are_recorded_too(db: AsyncSession, monkeypatch: pytest.MonkeyPatch) -> None:
    """A ledger that only holds successes cannot answer how often this works."""
    _client(monkeypatch, error=anthropic.APITimeoutError(request=_REQUEST))
    await _generate(db)

    total = (await db.execute(select(func.count()).select_from(AiCall))).scalar_one()
    assert total == 1

    row = (await db.execute(select(AiCall))).scalars().one()
    assert row.outcome is AiOutcome.TIMEOUT
    assert row.model == ai_prompts.REGISTRY["agent_plan"].model
    assert row.cost_usd == Decimal(0)
    assert row.error_detail and "Timeout" in row.error_detail


# ── one provider ─────────────────────────────────────────────────────────────


def test_no_other_provider_is_called_from_anywhere_in_the_app() -> None:
    """One provider is the point.

    It was two — Groq for most tools, Gemini for the Architect — then Gemini
    alone, and now Claude. Two at once meant two request shapes, two failure
    taxonomies, and two sets of quota arithmetic to reason about before
    answering "why did this come back rule_based". This is the check that keeps
    a second one from creeping back into a single endpoint, which is exactly
    how the first one arrived.

    `app/data` is exempt: those files are the tool catalogue, where rival
    providers are *content*. A row describing Groq is the product working.
    """
    from pathlib import Path

    forbidden = (
        "AsyncGroq(",
        "groq.Groq(",
        "AsyncOpenAI(",
        "openai.OpenAI(",
        "generativelanguage.googleapis.com",
        "genai.Client(",
    )
    offenders: list[str] = []
    for path in Path("app").rglob("*.py"):
        if path.parts[1] == "data":
            continue
        source = path.read_text(encoding="utf-8")
        offenders += [f"{path}: {marker}" for marker in forbidden if marker in source]

    assert offenders == [], f"a second provider is being called: {offenders}"


def test_only_the_ai_layer_generates_and_only_it_and_the_counter_hold_a_client() -> None:
    """`ai_service` generates and `tokenizer_service` counts. Anywhere else, a
    client is a model call that skips the `None` contract, the quota, and the
    ledger row — which is everything `ai_service` exists to guarantee."""
    from pathlib import Path

    may_hold_a_client = {"ai_service.py", "tokenizer_service.py"}
    offenders: list[str] = []
    for path in Path("app").rglob("*.py"):
        source = path.read_text(encoding="utf-8")
        if path.name not in may_hold_a_client and (
            "AsyncAnthropic(" in source or "anthropic.Anthropic(" in source
        ):
            offenders.append(f"{path}: constructs an Anthropic client")
        if path.name != "ai_service.py" and "messages.create(" in source:
            offenders.append(f"{path}: generates outside ai_service")

    assert offenders == [], f"Anthropic SDK used outside the AI layer: {offenders}"


# ── several passes, one usage line ───────────────────────────────────────────


def _meta(model: str, *, cost: str, latency: int) -> Any:
    from app.schemas.tools import AiMeta

    return AiMeta(
        model=model,
        prompt_version=ai_prompts.PROMPT_VERSION,
        input_tokens=100,
        output_tokens=50,
        cost_usd=Decimal(cost),
        latency_ms=latency,
    )


async def test_chained_passes_report_as_one_run() -> None:
    """A run has one source and one cost line, so two prompts answered by two
    models still have to arrive as a single `AiMeta`. Tokens and cost add up;
    latency adds up too, because the passes ran one after another and the
    figure answers "how long did the AI part take"."""
    from app.schemas.tools import ToolOutput

    async def first(output: ToolOutput) -> Any:
        output.metrics["one"] = "written"
        return _meta("claude-opus-5", cost="0.0100", latency=800)

    async def second(output: ToolOutput) -> Any:
        output.metrics["two"] = "written"
        return _meta("claude-opus-4-8", cost="0.0025", latency=400)

    output = ToolOutput()
    meta = await ai_service.chain(first, second)(output)

    assert meta is not None
    assert meta.model == "claude-opus-5+claude-opus-4-8"
    assert meta.input_tokens == 200
    assert meta.output_tokens == 100
    assert meta.cost_usd == Decimal("0.0125")
    assert meta.latency_ms == 1200
    assert output.metrics == {"one": "written", "two": "written"}


async def test_one_failed_pass_does_not_discard_the_other() -> None:
    """Partial enrichment is the normal outcome when the allowance runs out
    mid-run. One written section is worth more than none, and the run is
    still honestly `hybrid` — a model did contribute."""
    from app.schemas.tools import ToolOutput

    async def failed(_output: ToolOutput) -> Any:
        return None

    async def worked(output: ToolOutput) -> Any:
        output.metrics["written"] = "yes"
        return _meta("claude-opus-5", cost="0.0005", latency=200)

    output = ToolOutput()
    meta = await ai_service.chain(failed, worked)(output)

    assert meta is not None
    assert meta.model == "claude-opus-5"
    assert meta.latency_ms == 200
    assert output.metrics == {"written": "yes"}


async def test_every_pass_failing_keeps_the_run_rule_based() -> None:
    """`hybrid` has to mean a model actually contributed, or the provenance
    chip is naming a model that wrote nothing."""
    from app.schemas.tools import ToolOutput

    async def failed(_output: ToolOutput) -> Any:
        return None

    assert await ai_service.chain(failed, failed)(ToolOutput()) is None


def test_every_registered_prompt_is_reachable_from_an_endpoint() -> None:
    """A prompt nothing calls is a claim nothing keeps.

    This is the check that would have caught five prompts sitting unused in
    the registry while the marketing site described what they produced. It
    greps rather than introspects on purpose: the wiring is a `purpose=`
    string at a call site, and that string is exactly what goes missing.
    """
    from pathlib import Path

    wired = set()
    for path in list(Path("app/api").rglob("*.py")) + list(Path("app/services").rglob("*.py")):
        source = path.read_text(encoding="utf-8")
        for purpose in ai_prompts.REGISTRY:
            if f'purpose="{purpose}"' in source:
                wired.add(purpose)

    assert set(ai_prompts.REGISTRY) - wired == set()


# ── restatement ──────────────────────────────────────────────────────────────


def test_a_paraphrase_of_a_grounded_line_counts_as_an_echo() -> None:
    """Grounding the prompt in the engine's own words is what keeps the model
    honest, and it is also what makes paraphrasing them back the likeliest
    output. This is the check that stops the page showing the same advice
    twice in slightly different English."""
    engine = (
        "Choose the self-hosted archetype if data residency is a hard requirement, "
        "regardless of how it scores here."
    )
    model = (
        "Choose the self-hosted option if data residency is a hard requirement, "
        "regardless of how it scores here."
    )

    assert ai_service.echoes(model, engine)


def test_a_short_restatement_of_a_long_costed_row_counts_as_an_echo() -> None:
    """The measure has to be overlap over the *smaller* side. The engine's
    rows carry their own arithmetic and run long; the restatement is one
    sentence, and a sentence wholly contained in a paragraph scores under 0.2
    on the union — which is how the first version of this let duplicates
    through."""
    engine = (
        "chat is 100.00% of LLM spend and sends 4,000 input tokens per request. "
        "Caching a stable prompt prefix at 80% would cost $124.80/month."
    )
    model = "Cache the stable prompt prefix on the chat line at 80%."

    assert ai_service.echoes(model, engine)


def test_different_advice_about_the_same_component_is_not_an_echo() -> None:
    """The guard has to leave room for the model to be useful. Two suggestions
    about one workload line share the line's name and nothing else."""
    engine = (
        "chat is 100.00% of LLM spend and sends 4,000 input tokens per request. "
        "Caching a stable prompt prefix at 80% would cost $124.80/month."
    )
    model = "Move the chat classification step to a smaller model and keep the rest."

    assert not ai_service.echoes(model, engine)


def test_an_empty_side_is_never_an_echo() -> None:
    """Nothing to compare against is not a match, and it must not be a divide
    by zero either."""
    assert not ai_service.echoes("", "anything at all")
    assert not ai_service.echoes("anything at all", "")
