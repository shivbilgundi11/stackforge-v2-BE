"""The AI layer's guarantees.

The module's whole claim is that a model failure is never a request failure,
so most of this file is failure modes. Each one asserts the same two things:
`generate_json` returned `None`, and no exception escaped.

The pricing tests assert exact dollars against the internal rate table, by
hand. A cost calculation that only agrees with itself is not checked.

The provider is Gemini and there is only one, so the stub is an `httpx`
client rather than a vendor SDK. That is deliberate beyond convenience: the
failure taxonomy is derived from real status codes and real response bodies
here, so a test that says "a 429 is recorded as a rate limit" is exercising
the same branch a live 429 would.
"""

from __future__ import annotations

from decimal import Decimal
from typing import Any

import httpx
import pytest
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


class _FakeClient:
    """Stands in for the `httpx.AsyncClient` the service opens per call.

    Records what was sent, because half of what this file checks is the
    request shape — and the request shape is the part that turns into a 400
    at three in the morning rather than a test failure.
    """

    def __init__(self, *, responses: list[httpx.Response], error: Exception | None) -> None:
        self._responses = responses
        self._error = error
        self.calls: list[dict[str, Any]] = []

    async def __aenter__(self) -> _FakeClient:
        return self

    async def __aexit__(self, *args: Any) -> None:
        return None

    async def post(self, url: str, **kwargs: Any) -> httpx.Response:
        self.calls.append({"url": url, **kwargs})
        if self._error is not None:
            raise self._error
        # The last response repeats, so a test that makes two calls and cares
        # about only one of them does not have to script both.
        index = min(len(self.calls) - 1, len(self._responses) - 1)
        return self._responses[index]


def _client(
    monkeypatch: pytest.MonkeyPatch,
    *,
    response: httpx.Response | None = None,
    responses: list[httpx.Response] | None = None,
    error: Exception | None = None,
) -> _FakeClient:
    """Install the stub and hand it back for its `calls`."""
    scripted = responses if responses is not None else [response] if response else []
    client = _FakeClient(responses=scripted, error=error)
    monkeypatch.setattr(ai_service.httpx, "AsyncClient", lambda **_: client)
    return client


_URL = "https://generativelanguage.googleapis.com/v1beta/models/gemini-3.6-flash:generateContent"


def _response(
    text: str = '{"summary": "ok", "why": "because", "weakest_link": "a", "watch_out_for": []}',
    *,
    status: int = 200,
    finish_reason: str = "STOP",
    prompt_tokens: int = 1000,
    answer_tokens: int = 200,
    thought_tokens: int = 0,
    cached: int = 0,
    parts: list[dict[str, Any]] | None = None,
    body: dict[str, Any] | None = None,
) -> httpx.Response:
    """One `generateContent` response.

    `prompt_tokens` is the **whole** prompt, cached part included, which is
    the convention the API itself uses — so a test that sets both figures is
    describing what the service actually receives rather than what it stores.
    """
    if body is None:
        usage: dict[str, Any] = {
            "promptTokenCount": prompt_tokens,
            "candidatesTokenCount": answer_tokens,
        }
        if thought_tokens:
            usage["thoughtsTokenCount"] = thought_tokens
        if cached:
            usage["cachedContentTokenCount"] = cached
        body = {
            "candidates": [
                {
                    "finishReason": finish_reason,
                    "content": {"parts": parts if parts is not None else [{"text": text}]},
                }
            ],
            "usageMetadata": usage,
        }
    return httpx.Response(status, request=httpx.Request("POST", _URL), json=body)


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
    monkeypatch.setattr(settings, "gemini_api_key", "")
    assert settings.ai_enabled is False
    client = _client(monkeypatch, response=_response())

    assert await _generate(db) is None
    assert client.calls == []
    assert await _outcomes(db) == [AiOutcome.DISABLED]


@pytest.mark.parametrize(
    ("error", "expected"),
    [
        (httpx.ConnectTimeout("timed out"), AiOutcome.TIMEOUT),
        (httpx.ReadTimeout("timed out"), AiOutcome.TIMEOUT),
        (httpx.ConnectError("no route"), AiOutcome.API_ERROR),
        (RuntimeError("something nobody predicted"), AiOutcome.API_ERROR),
    ],
)
async def test_every_failure_returns_none_and_is_recorded(
    db: AsyncSession, monkeypatch: pytest.MonkeyPatch, error: Exception, expected: AiOutcome
) -> None:
    monkeypatch.setattr(settings, "gemini_api_key", "gemini-test")
    _client(monkeypatch, error=error)

    assert await _generate(db) is None
    assert await _outcomes(db) == [expected]


async def test_the_daily_request_allowance_is_recorded_as_a_rate_limit(
    db: AsyncSession, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The free tier is 20 requests per day per model, and running out is the
    one failure an operator can actually act on. Filed under `api_error` it
    would send that investigation to the wrong place — to the prompt, or to
    the network, rather than to the billing page."""
    monkeypatch.setattr(settings, "gemini_api_key", "gemini-test")
    _client(
        monkeypatch,
        response=_response(
            status=429,
            body={
                "error": {
                    "code": 429,
                    "status": "RESOURCE_EXHAUSTED",
                    "message": "Quota exceeded for generate_content_free_tier_requests, limit: 20",
                }
            },
        ),
    )

    assert await _generate(db) is None
    assert await _outcomes(db) == [AiOutcome.RATE_LIMITED]

    row = (await db.execute(select(AiCall))).scalars().one()
    assert row.error_detail and "429" in row.error_detail


async def test_an_ordinary_status_error_is_still_an_api_error(
    db: AsyncSession, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(settings, "gemini_api_key", "gemini-test")
    _client(monkeypatch, response=_response(status=500, body={"error": {"code": 500}}))

    assert await _generate(db) is None
    assert await _outcomes(db) == [AiOutcome.API_ERROR]


@pytest.mark.parametrize("reason", ["SAFETY", "PROHIBITED_CONTENT", "RECITATION"])
async def test_a_refusal_returns_none_rather_than_raising(
    db: AsyncSession, monkeypatch: pytest.MonkeyPatch, reason: str
) -> None:
    """A declined request is a 200 with no usable content. Unnamed, it arrives
    as malformed output and gets debugged as a bad schema."""
    monkeypatch.setattr(settings, "gemini_api_key", "gemini-test")
    _client(monkeypatch, response=_response(finish_reason=reason, parts=[]))

    assert await _generate(db) is None
    assert await _outcomes(db) == [AiOutcome.REFUSAL]

    row = (await db.execute(select(AiCall))).scalars().one()
    assert row.error_detail == f"refusal:{reason}"


async def test_an_exhausted_reservation_is_recorded_with_its_reason(
    db: AsyncSession, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Thinking is drawn from `maxOutputTokens` before the answer is, so a
    reservation that thinking exhausts comes back as a 200 with `MAX_TOKENS`
    and *no parts at all*. Recording the reason is the difference between
    raising a number and rewriting a schema."""
    monkeypatch.setattr(settings, "gemini_api_key", "gemini-test")
    _client(
        monkeypatch,
        # No candidate tokens at all: the reservation went entirely on
        # thinking, which is what the empty `parts` list means.
        response=_response(
            finish_reason="MAX_TOKENS", parts=[], answer_tokens=0, thought_tokens=2_400
        ),
    )

    assert await _generate(db) is None
    assert await _outcomes(db) == [AiOutcome.INVALID_OUTPUT]

    row = (await db.execute(select(AiCall))).scalars().one()
    assert row.error_detail == "finish_reason=MAX_TOKENS"
    # And the thinking it did spend is still billed.
    assert row.output_tokens == 2_400


async def test_a_prompt_blocked_before_generation_degrades(
    db: AsyncSession, monkeypatch: pytest.MonkeyPatch
) -> None:
    """No candidate at all, only `promptFeedback`. Reading `candidates[0]`
    without checking is the one failure mode that would escape this
    function."""
    monkeypatch.setattr(settings, "gemini_api_key", "gemini-test")
    _client(
        monkeypatch,
        response=_response(body={"promptFeedback": {"blockReason": "SAFETY"}, "usageMetadata": {}}),
    )

    assert await _generate(db) is None
    assert await _outcomes(db) == [AiOutcome.REFUSAL]


async def test_malformed_output_degrades_instead_of_throwing(
    db: AsyncSession, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(settings, "gemini_api_key", "gemini-test")
    _client(monkeypatch, response=_response("this is not json at all"))

    assert await _generate(db) is None
    assert await _outcomes(db) == [AiOutcome.INVALID_OUTPUT]


async def test_truncated_json_degrades(db: AsyncSession, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(settings, "gemini_api_key", "gemini-test")
    _client(monkeypatch, response=_response('{"summary": "cut off mid'))

    assert await _generate(db) is None


async def test_a_json_array_is_not_an_answer(
    db: AsyncSession, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Valid JSON of the wrong shape. Every caller indexes the result by key,
    so a list would raise at the `apply` rather than here."""
    monkeypatch.setattr(settings, "gemini_api_key", "gemini-test")
    _client(monkeypatch, response=_response('[{"summary": "ok"}]'))

    assert await _generate(db) is None
    assert await _outcomes(db) == [AiOutcome.INVALID_OUTPUT]


async def test_exhausted_quota_returns_none_without_calling_the_model(
    db: AsyncSession, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(settings, "gemini_api_key", "gemini-test")
    client = _client(monkeypatch, response=_response())
    await set_limit(db, plan=Plan.FREE, metric=Metric.AI_CALLS_PER_DAY, value=0)

    assert await _generate(db) is None
    assert client.calls == []
    assert await _outcomes(db) == [AiOutcome.QUOTA_EXCEEDED]


# ── the answer, and the model's reasoning about it ───────────────────────────


async def test_the_models_own_thinking_is_left_out_of_the_answer(
    db: AsyncSession, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Thinking arrives as extra `parts` on the same candidate, marked
    `thought`. Concatenating every part and parsing the result is what the
    first version did, and it fails the moment the model narrates before
    answering — the JSON is valid and the string it is glued to is not."""
    monkeypatch.setattr(settings, "gemini_api_key", "gemini-test")
    _client(
        monkeypatch,
        response=_response(
            parts=[
                {"text": "Let me work through the grounding first.", "thought": True},
                {"text": '{"summary": "the answer"}'},
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
    monkeypatch.setattr(settings, "gemini_api_key", "gemini-test")
    client = _client(monkeypatch, response=_response())

    await _generate(db)
    sent = client.calls[0]
    config = sent["json"]["generationConfig"]

    assert sent["headers"] == {"x-goog-api-key": "gemini-test"}
    assert ai_prompts.REGISTRY["agent_plan"].model in sent["url"]
    assert config["responseMimeType"] == "application/json"
    assert config["responseJsonSchema"] == ai_prompts.REGISTRY["agent_plan"].schema
    assert config["maxOutputTokens"] > 0
    # Only `low` | `medium` | `high` exist, and the registry is the only place
    # that picks one.
    assert config["thinkingConfig"]["thinkingLevel"] in {"low", "medium", "high"}


async def test_the_system_prompt_comes_first_and_never_varies(
    db: AsyncSession, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The stable half in `systemInstruction`, the varying half in the user
    turn after it. Context caching here is implicit and prefix-based, so
    message order is the only lever there is — a variable interpolated into
    the system text would make every request a miss with nothing to show for
    it."""
    monkeypatch.setattr(settings, "gemini_api_key", "gemini-test")
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

    first, second = (call["json"] for call in client.calls)
    assert first["systemInstruction"] == second["systemInstruction"]
    assert first["contents"] != second["contents"]
    assert [turn["role"] for turn in first["contents"]] == ["user"]


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
    call returns a 200 with no content at all — `rule_based` on the page, and
    nothing in the ledger that looks like a token problem. The Architect's
    assessment did exactly this at 3,000: 2,700 tokens of reasoning, 269 left
    for a ten-row score breakdown and five prose fields.
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


def test_the_registry_spreads_across_more_than_one_model() -> None:
    """The free allowance is 20 requests per day *per model*, so tiering is
    not a cost optimisation here — it is the difference between a product that
    works all day and one that stops after the twentieth request."""
    assert len({prompt.model for prompt in ai_prompts.REGISTRY.values()}) > 1


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
    # gemini-3.6-flash is $0.75/1M in, $3.75/1M out. 100,000 in + 20,000 out
    # = 0.075 + 0.075 = $0.15.
    assert ai_pricing.cost_of(
        model="gemini-3.6-flash", input_tokens=100_000, output_tokens=20_000
    ) == Decimal("0.150000")

    # gemini-3.5-flash-lite is $0.10/$0.40.
    # 100,000 in + 20,000 out = 0.01 + 0.008 = $0.018.
    assert ai_pricing.cost_of(
        model="gemini-3.5-flash-lite", input_tokens=100_000, output_tokens=20_000
    ) == Decimal("0.018000")


def test_cached_reads_bill_at_a_tenth_and_writes_are_free() -> None:
    # 1,000,000 cached reads on 3.6-flash: 1,000,000 x $0.75/1M x 0.1 = $0.075.
    assert ai_pricing.cost_of(
        model="gemini-3.6-flash",
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
        model="gemini-3.6-flash",
        input_tokens=0,
        output_tokens=0,
        cached_write_tokens=1_000_000,
    ) == ai_pricing.cost_of(model="gemini-3.6-flash", input_tokens=1_000_000, output_tokens=0)


def test_an_unknown_model_prices_high_rather_than_free() -> None:
    """A model with no rate should be conspicuous in the ledger, not invisible."""
    unknown = ai_pricing.cost_of(
        model="gemini-9-unreleased", input_tokens=1_000_000, output_tokens=0
    )
    assert unknown > 0


async def test_a_successful_call_records_tokens_cost_and_prompt_version(
    db: AsyncSession, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(settings, "gemini_api_key", "gemini-test")
    # 6,000 reported, 2,000 of them cached - so 4,000 were billed in full.
    _client(
        monkeypatch,
        response=_response(prompt_tokens=6_000, answer_tokens=800, cached=2_000),
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
    # agent_plan runs on gemini-3.6-flash:
    #   4,000 x $0.75/1M + 800 x $3.75/1M + 2,000 x $0.75/1M x 0.1
    # = 0.003 + 0.003 + 0.00015 = $0.00615
    assert row.cost_usd == Decimal("0.006150")
    assert result.meta.cost_usd == Decimal("0.006150")
    assert result.meta.prompt_version == ai_prompts.PROMPT_VERSION


async def test_thinking_tokens_are_billed_as_output(
    db: AsyncSession, monkeypatch: pytest.MonkeyPatch
) -> None:
    """`candidatesTokenCount` is the visible answer only; thinking is reported
    apart and charged at the output rate. Counting only the visible half
    understates a short structured answer by more than it counts, because
    reasoning routinely runs several times the length of the JSON."""
    monkeypatch.setattr(settings, "gemini_api_key", "gemini-test")
    _client(
        monkeypatch,
        response=_response(prompt_tokens=1_000, answer_tokens=200, thought_tokens=800),
    )

    result = await _generate(db)
    assert result is not None
    assert result.meta.output_tokens == 1_000

    row = (await db.execute(select(AiCall))).scalars().one()
    assert row.output_tokens == 1_000
    # 1,000 x $0.75/1M + 1,000 x $3.75/1M = 0.00075 + 0.00375 = $0.0045.
    assert row.cost_usd == Decimal("0.004500")


async def test_a_repeat_call_reports_cached_reads(
    db: AsyncSession, monkeypatch: pytest.MonkeyPatch
) -> None:
    """What proves caching is on: the second call's prompt prefix is served
    from the cache, and the figure is recorded separately so it can be seen at
    all. A miss and a hit have to be distinguishable in the ledger, or "is the
    cache working" is unanswerable after the fact."""
    monkeypatch.setattr(settings, "gemini_api_key", "gemini-test")
    await set_limit(db, plan=Plan.FREE, metric=Metric.AI_CALLS_PER_DAY, value=5)

    # A miss: the provider omits the cached field entirely rather than sending
    # a zero, which is the shape the reader has to survive.
    _client(
        monkeypatch,
        responses=[
            _response(prompt_tokens=1_540),
            _response(prompt_tokens=1_540, cached=1_500),
        ],
    )
    await _generate(db)
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
    monkeypatch.setattr(settings, "gemini_api_key", "gemini-test")
    _client(monkeypatch, error=httpx.ReadTimeout("timed out"))
    await _generate(db)

    total = (await db.execute(select(func.count()).select_from(AiCall))).scalar_one()
    assert total == 1

    row = (await db.execute(select(AiCall))).scalars().one()
    assert row.outcome is AiOutcome.TIMEOUT
    assert row.cost_usd == Decimal(0)
    assert row.error_detail and "Timeout" in row.error_detail


# ── one provider ─────────────────────────────────────────────────────────────


def test_no_other_provider_is_called_from_anywhere_in_the_app() -> None:
    """One provider is the point.

    It was two — Groq for most tools, Gemini for the Architect — and that
    meant two request shapes, two failure taxonomies, and two sets of quota
    arithmetic to reason about before answering "why did this come back
    rule_based". This is the check that keeps the second one from creeping
    back into a single endpoint, which is exactly how the first one arrived.

    `app/data` is exempt: those files are the tool catalogue, where rival
    providers are *content*. A row describing Groq is the product working.
    """
    from pathlib import Path

    forbidden = ("AsyncGroq(", "groq.Groq(", "AsyncOpenAI(", "openai.OpenAI(")
    offenders: list[str] = []
    for path in Path("app").rglob("*.py"):
        if path.parts[1] == "data":
            continue
        source = path.read_text(encoding="utf-8")
        offenders += [f"{path}: {marker}" for marker in forbidden if marker in source]

    assert offenders == [], f"a second provider is being called: {offenders}"


def test_the_anthropic_sdk_is_only_ever_used_for_counting() -> None:
    """`tokenizer_service` keeps a client on purpose — `count_tokens` is the
    only accurate count for a Claude row in the catalogue, and no provider
    offers one for someone else's tokeniser — so it is allowed the client,
    and only for that."""
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
    """A run has one source and one cost line, so two prompts on two
    providers still have to arrive as a single `AiMeta`. Tokens and cost add
    up; latency adds up too, because the passes ran one after another and the
    figure answers "how long did the AI part take"."""
    from app.schemas.tools import ToolOutput

    async def first(output: ToolOutput) -> Any:
        output.metrics["one"] = "written"
        return _meta("gemini-3.6-flash", cost="0.0100", latency=800)

    async def second(output: ToolOutput) -> Any:
        output.metrics["two"] = "written"
        return _meta("openai/gpt-oss-120b", cost="0.0025", latency=400)

    output = ToolOutput()
    meta = await ai_service.chain(first, second)(output)

    assert meta is not None
    assert meta.model == "gemini-3.6-flash+openai/gpt-oss-120b"
    assert meta.input_tokens == 200
    assert meta.output_tokens == 100
    assert meta.cost_usd == Decimal("0.0125")
    assert meta.latency_ms == 1200
    assert output.metrics == {"one": "written", "two": "written"}


async def test_one_failed_pass_does_not_discard_the_other() -> None:
    """Partial enrichment is the normal outcome when one provider is out of
    allowance. One written section is worth more than none, and the run is
    still honestly `hybrid` — a model did contribute."""
    from app.schemas.tools import ToolOutput

    async def failed(_output: ToolOutput) -> Any:
        return None

    async def worked(output: ToolOutput) -> Any:
        output.metrics["written"] = "yes"
        return _meta("openai/gpt-oss-20b", cost="0.0005", latency=200)

    output = ToolOutput()
    meta = await ai_service.chain(failed, worked)(output)

    assert meta is not None
    assert meta.model == "openai/gpt-oss-20b"
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
